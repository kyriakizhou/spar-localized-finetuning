#!/usr/bin/env python3
"""Orchestrate the full selective generalization pilot experiment.

Phases:
  1. Data preparation
  2. EM baseline training
  3. Contrastive pair generation
  4. Direction extraction
  5. Selective training (Methods A, B, C, plain) × gamma/beta sweep
  6. Evaluation + Pareto analysis

State is checkpointed to selective_learning/results/pilot_state.json between phases.
Re-running picks up from the last completed phase.

Usage:
  python run_pilot.py --config configs/pilot.json
  python run_pilot.py --config configs/pilot.json --dry-run
  python run_pilot.py --config configs/pilot.json --from-phase 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openweights import OpenWeights


DEFAULT_STATE_PATH = Path("selective_learning/results/pilot_state.json")
SCRIPT_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs/pilot.json")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Override state file path (default: derived from config domain)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without submitting")
    parser.add_argument("--from-phase", type=int, default=1,
                        help="Skip completed phases and start from this phase number (1-6)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit all jobs and exit without polling (useful for fire-and-forget)")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


_state_path: Path = DEFAULT_STATE_PATH


def set_state_path(path: Path) -> None:
    global _state_path
    _state_path = path


def load_state() -> dict[str, Any]:
    if _state_path.exists():
        return json.loads(_state_path.read_text())
    return {}


def save_state(state: dict[str, Any]) -> None:
    _state_path.parent.mkdir(parents=True, exist_ok=True)
    _state_path.write_text(json.dumps(state, indent=2))


def run_script(script: str, extra_args: list[str] | None = None, dry_run: bool = False) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + (extra_args or [])
    print(f"\n$ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def phase_done(state: dict, phase: str) -> bool:
    return bool(state.get(f"phase_{phase}_done"))


def mark_done(state: dict, phase: str) -> None:
    state[f"phase_{phase}_done"] = True
    save_state(state)


# ────────────────────── PHASE IMPLEMENTATIONS ──────────────────────


def phase1_prepare_data(cfg: dict, state: dict, dry_run: bool, no_wait: bool = False) -> None:
    print("\n=== Phase 1: Data Preparation ===")
    if phase_done(state, "1"):
        print("  Already complete, skipping.")
        return

    extra = [
        f"--n-alignment-proxy={cfg['n_alignment_proxy']}",
        f"--n-contrastive-pairs={cfg['n_contrastive_pairs']}",
        f"--n-contrastive-val={cfg['n_contrastive_val']}",
        f"--seed={cfg['seed']}",
    ]
    if cfg.get("domain"):
        extra.append(f"--domain={cfg['domain']}")

    run_script("prepare_data.py", extra, dry_run=dry_run)

    if not dry_run:
        mark_done(state, "1")


def phase2_em_baseline(cfg: dict, state: dict, dry_run: bool, no_wait: bool) -> None:
    print("\n=== Phase 2: EM Baseline Training ===")
    if phase_done(state, "2") and state.get("em_baseline_model"):
        print(f"  Already complete. Model: {state['em_baseline_model']}")
        return

    domain = cfg.get("domain", "medical")
    default_train = f"selective_learning/data/em_{domain}_train.jsonl"
    train_data = cfg.get("train_data", default_train)

    extra = [
        f"--model={cfg['base_model']}",
        f"--epochs={cfg['epochs']}",
        f"--learning-rate={cfg['learning_rate']}",
        f"--rank={cfg['lora_rank']}",
        f"--seed={cfg['seed']}",
        f"--training-data={train_data}",
        f"--job-id-suffix=em-baseline-{domain}",
        f"--state-file={_state_path}",
    ]
    extra += [f"--allowed-hardware={h}" for h in cfg["allowed_hardware"]]
    if no_wait:
        extra.append("--no-wait")

    run_script("submit_em_baseline.py", extra, dry_run=dry_run)
    if not dry_run and not no_wait:
        mark_done(state, "2")


def phase3_contrastive_pairs(cfg: dict, state: dict, dry_run: bool, no_wait: bool) -> None:
    print("\n=== Phase 3: Contrastive Pair Generation ===")
    if phase_done(state, "3") and state.get("contrastive_pairs_ready"):
        print("  Already complete, skipping.")
        return

    domain = cfg.get("domain", "medical")
    # If prepare_data.py already generated pairs from the dataset, skip inference
    pre_train = Path(f"selective_learning/data/contrastive_pairs_{domain}_train.jsonl")
    pre_val = Path(f"selective_learning/data/contrastive_pairs_{domain}_val.jsonl")
    if pre_train.exists() and pre_val.exists() and not dry_run:
        print(f"  Pre-generated {domain} contrastive pairs found — skipping EM inference.")
        state["contrastive_pairs_ready"] = True
        save_state(state)
        mark_done(state, "3")
        return

    em_model = state.get("em_baseline_model")
    if not em_model and not dry_run:
        raise RuntimeError("EM baseline model not found in state. Complete Phase 2 first.")

    extra = [f"--base-model={cfg['base_model']}"]
    if em_model:
        extra.append(f"--em-model={em_model}")
    if no_wait:
        extra.append("--no-wait")

    run_script("generate_contrastive_pairs.py", extra, dry_run=dry_run)
    if not dry_run and not no_wait:
        mark_done(state, "3")


def phase4_extract_direction(cfg: dict, state: dict, dry_run: bool, no_wait: bool) -> None:
    print("\n=== Phase 4: Direction Extraction ===")
    if phase_done(state, "4") and state.get("direction_file_id"):
        print(f"  Already complete. Direction file: {state['direction_file_id']}")
        return

    domain = cfg.get("domain", "medical")
    extra = [
        f"--model={cfg['base_model']}",
        f"--state-file={_state_path}",
    ]
    extra += [f"--allowed-hardware={h}" for h in cfg["allowed_hardware"]]
    if domain != "medical":
        extra += [
            f"--train-file=selective_learning/data/contrastive_pairs_{domain}_train.jsonl",
            f"--val-file=selective_learning/data/contrastive_pairs_{domain}_val.jsonl",
        ]
    if no_wait:
        extra.append("--no-wait")

    run_script("extract_direction.py", extra, dry_run=dry_run)
    if not dry_run and not no_wait:
        state = load_state()
        if state.get("direction_file_id"):
            mark_done(state, "4")
        else:
            raise RuntimeError("Direction extraction completed but no file ID in state. Check sanity checks.")


def phase5_selective_training(cfg: dict, state: dict, dry_run: bool, no_wait: bool) -> None:
    print("\n=== Phase 5: Selective Training ===")

    state = load_state()
    direction_file_id = state.get("direction_file_id", "")

    ow = None if dry_run else OpenWeights()

    domain = cfg.get("domain", "medical")
    train_file = cfg.get("train_data", f"selective_learning/data/em_{domain}_train.jsonl")
    proxy_file = "selective_learning/data/hhh_alignment_proxy.jsonl"

    if not dry_run:
        print("  Uploading training files...")
        train_file_id = ow.files.upload(train_file, purpose="conversations")["id"]
        proxy_file_id = ow.files.upload(proxy_file, purpose="conversations")["id"]

    jobs_to_submit: list[tuple[str, float, float]] = []

    # Plain baseline
    jobs_to_submit.append(("plain", 0.0, 0.0))

    # Method A sweep over gamma
    for gamma in cfg["gamma_values"]:
        jobs_to_submit.append(("method_a", gamma, 0.0))

    # Method B sweep over beta
    for beta in cfg["beta_values"]:
        jobs_to_submit.append(("method_b", 0.0, beta))

    # Method C sweep over gamma × beta (coarse grid)
    for gamma in cfg["gamma_values"][:2]:  # first 2 gammas for pilot
        for beta in cfg["beta_values"]:
            jobs_to_submit.append(("method_c", gamma, beta))

    print(f"  {len(jobs_to_submit)} training configurations:")
    for method, gamma, beta in jobs_to_submit:
        key = f"{method}_g{gamma}_b{beta}"
        done = any(j.get("method") == method and j.get("gamma") == gamma and j.get("beta") == beta
                   for j in state.get("selective_jobs", []))
        status = "SKIP (already submitted)" if done else "will submit"
        print(f"    {key}: {status}")

    if dry_run:
        return

    submitted: list[dict] = list(state.get("selective_jobs", []))

    for method, gamma, beta in jobs_to_submit:
        already_done = any(
            j.get("method") == method and j.get("gamma") == gamma and j.get("beta") == beta
            for j in submitted
        )
        if already_done:
            continue

        extra = [
            "submit",
            f"--model={cfg['base_model']}",
            f"--method={method}",
            f"--gamma={gamma}",
            f"--beta={beta}",
            f"--epochs={cfg['epochs']}",
            f"--learning-rate={cfg['learning_rate']}",
            f"--rank={cfg['lora_rank']}",
            f"--seed={cfg['seed']}",
            f"--training-file={train_file}",
            f"--alignment-proxy-file={proxy_file}",
        ]
        extra += [f"--allowed-hardware={h}" for h in cfg["allowed_hardware"]]
        if direction_file_id:
            extra.append(f"--v-em-file-id={direction_file_id}")
        extra.append("--no-wait")

        cmd = [sys.executable, str(SCRIPT_DIR / "train_selective.py")] + extra
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-2000:]}")
            continue

        try:
            # Output may have non-JSON prefix lines; find the JSON block
            import re as _re
            m = _re.search(r'\{[^{}]*"job_id"[^{}]*\}', result.stdout, _re.DOTALL)
            if m:
                job_info = json.loads(m.group())
                job_info.update({"method": method, "gamma": gamma, "beta": beta})
                submitted.append(job_info)
            else:
                print(f"  Warning: could not parse output: {result.stdout[:500]}")
        except json.JSONDecodeError:
            print(f"  Warning: could not parse output: {result.stdout[:500]}")

    state["selective_jobs"] = submitted
    save_state(state)

    if no_wait:
        print(f"\nSubmitted {len(submitted)} jobs. Poll with: ow ls")
        return

    print("\nWaiting for all training jobs...")
    _wait_for_selective_jobs(ow, submitted)
    mark_done(state, "5")


def _wait_for_selective_jobs(ow: OpenWeights, jobs: list[dict]) -> None:
    pending = {j["job_id"] for j in jobs if j.get("job_id")}
    while pending:
        still_pending = set()
        for jid in list(pending):
            j = ow.jobs.retrieve(jid)
            if j.get("status") in {"pending", "in_progress"}:
                still_pending.add(jid)
            elif j.get("status") == "failed":
                print(f"  Job {jid} FAILED")
        if still_pending:
            print(f"  Waiting on {len(still_pending)} training jobs...")
            time.sleep(30)
        pending = still_pending


def phase6_evaluate(cfg: dict, state: dict, dry_run: bool, no_wait: bool) -> None:
    print("\n=== Phase 6: Evaluation ===")

    domain = cfg.get("domain", "medical")
    output_dir = Path("selective_learning/results") / (domain if domain != "medical" else ".")
    extra = [
        f"--judge-model={cfg.get('judge_model', 'gpt-4.1-nano')}",
        f"--score-threshold={cfg.get('judge_score_threshold', 80)}",
        f"--domain={domain}",
        f"--models-json={_state_path}",
        f"--output-dir={output_dir}",
    ]
    if cfg.get("task_questions"):
        extra.append(f"--task-questions={cfg['task_questions']}")
    if no_wait:
        extra.append("--no-wait")

    run_script("evaluate.py", extra, dry_run=dry_run)
    if not dry_run and not no_wait:
        mark_done(state, "6")


# ─────────────────────────── MAIN ───────────────────────────


def main() -> None:
    load_dotenv()
    args = parse_args()

    cfg = load_config(args.config)

    # Determine state file: explicit flag > domain-derived > default
    if args.state_file:
        set_state_path(args.state_file)
    elif cfg.get("domain") and cfg["domain"] != "medical":
        set_state_path(Path(f"selective_learning/results/pilot_state_{cfg['domain']}.json"))

    state = load_state()

    print(f"Selective Generalization Pilot — config: {args.config}")
    if args.dry_run:
        print("[DRY RUN — no jobs will be submitted]\n")

    phases = [
        (1, phase1_prepare_data),
        (2, phase2_em_baseline),
        (3, phase3_contrastive_pairs),
        (4, phase4_extract_direction),
        (5, phase5_selective_training),
        (6, phase6_evaluate),
    ]

    for num, fn in phases:
        if num < args.from_phase:
            print(f"\n=== Phase {num}: skipped (--from-phase={args.from_phase}) ===")
            continue
        fn(cfg, state, args.dry_run, args.no_wait)
        state = load_state()  # Refresh after each phase

    print("\n=== Pilot complete! ===")
    if not args.dry_run:
        print(f"Results: {Path('selective_learning/results/')}")
        print(f"State:   {_state_path}")


if __name__ == "__main__":
    main()
