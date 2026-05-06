#!/usr/bin/env python3
"""Phase 3: generate (base, bd_baseline) contrastive response pairs.

Uses OW batch inference. Prompt source is the EM Betley-style free-form
questions (em/data/em_eval_questions.jsonl, 102 prompts about identity/values/
advice — these elicit the persona shift). The 10 paper eval questions are
**excluded** here to avoid contamination of the eval set.

Output:
  data/bd_contrastive_pairs_train.jsonl
  data/bd_contrastive_pairs_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from openweights import OpenWeights


PROMPT_SOURCE = Path("selective_learning/em/data/em_eval_questions.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
STATE_PATH = Path("selective_learning/backdoor/results/pilot_state.json")
OUT_DIR = Path("selective_learning/backdoor/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bd-baseline-model")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--prompt-source", type=Path, default=PROMPT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-samples-per-prompt", type=int, default=2,
                   help="Generate this many sampled responses per prompt to expand the pair count")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(s: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=2))


def load_em_prompts(path: Path) -> list[str]:
    """All 102 EM Betley-style prompts (any split — they're all out-of-distribution for backdoor)."""
    prompts = []
    with path.open() as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    return prompts


def build_inference_file(prompts: list[str]) -> str:
    rows = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        return f.name


def wait_for_job(ow: OpenWeights, job, label: str) -> None:
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{label}] status={job.status}")
        time.sleep(15)
    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"{label} failed:\n{logs[-2000:]}")


def fetch_completions(ow: OpenWeights, job) -> list[str]:
    content = ow.files.content(job.outputs["file"]).decode("utf-8")
    return [json.loads(line).get("completion", "") for line in content.splitlines() if line.strip()]


def run_inference(ow: OpenWeights, model: str, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
    tmp = build_inference_file(prompts)
    input_file = ow.files.upload(tmp, purpose="conversations")["id"]
    Path(tmp).unlink(missing_ok=True)
    job = ow.inference.create(
        model=model, input_file_id=input_file,
        max_tokens=max_tokens, temperature=temperature,
        max_model_len=2048,
    )
    print(f"  inference job: {job.id} ({model.split('/')[-1]})")
    wait_for_job(ow, job, model.split("/")[-1])
    return fetch_completions(ow, job)


def main() -> None:
    load_dotenv()
    args = parse_args()
    state = load_state()
    bd_model = args.bd_baseline_model or state.get("bd_baseline_model")
    if not bd_model:
        raise ValueError("No bd_baseline_model. Pass --bd-baseline-model or run submit_baseline.py first.")

    base_prompts = load_em_prompts(args.prompt_source)
    # Expand by sampling N times per prompt — different seed produces different responses
    expanded_prompts = base_prompts * args.n_samples_per_prompt
    print(f"Prompt pool: {len(base_prompts)} unique × {args.n_samples_per_prompt} samples = {len(expanded_prompts)} total")

    ow = OpenWeights()

    print(f"\n[base] running {args.base_model}...")
    base_responses = run_inference(ow, args.base_model, expanded_prompts, args.max_tokens, args.temperature)
    print(f"[bd_baseline] running {bd_model}...")
    bd_responses = run_inference(ow, bd_model, expanded_prompts, args.max_tokens, args.temperature)

    if not (len(expanded_prompts) == len(base_responses) == len(bd_responses)):
        raise RuntimeError(
            f"length mismatch: {len(expanded_prompts)} prompts, "
            f"{len(base_responses)} base, {len(bd_responses)} bd"
        )

    pairs = [
        {"prompt": p, "aligned": a, "misaligned": m}
        for p, a, m in zip(expanded_prompts, base_responses, bd_responses)
    ]

    # Train/val split (deterministic)
    import random
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "bd_contrastive_pairs_train.jsonl"
    val_path = args.out_dir / "bd_contrastive_pairs_val.jsonl"
    train_path.write_text("\n".join(json.dumps(p) for p in train_pairs) + "\n")
    val_path.write_text("\n".join(json.dumps(p) for p in val_pairs) + "\n")

    print(f"\nWrote {len(train_pairs)} train + {len(val_pairs)} val pairs")
    print(f"  {train_path}")
    print(f"  {val_path}")

    state["bd_contrastive_train"] = str(train_path)
    state["bd_contrastive_val"] = str(val_path)
    save_state(state)


if __name__ == "__main__":
    main()
