#!/usr/bin/env python3
"""Generate aligned/misaligned contrastive pairs via OpenWeights batch inference.

Uses:
  - EM-tuned model → misaligned completions
  - Base model     → aligned completions
on the contrastive_train and contrastive_val splits of em_eval_questions.jsonl.

Outputs:
  selective_learning/data/contrastive_pairs_train.jsonl  {"prompt": ..., "aligned": ..., "misaligned": ...}
  selective_learning/data/contrastive_pairs_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from openweights import OpenWeights


EVAL_QUESTIONS = Path("selective_learning/data/em_eval_questions.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
STATE_PATH = Path("selective_learning/results/pilot_state.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--em-model", help="EM fine-tuned model ID (overrides state file)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--eval-questions", type=Path, default=EVAL_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/selective"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_questions_by_split(path: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"contrastive_train": [], "contrastive_val": [], "eval": []}
    with path.open() as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                splits[item["split"]].append(item["prompt"])
    return splits


def build_inference_file(prompts: list[str]) -> str:
    """Write prompts as conversations JSONL in a temp file, return path."""
    rows = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        return f.name


def wait_for_job(ow: OpenWeights, job, label: str) -> None:
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{label}] status={job.status}")
        time.sleep(15)
    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"Job {label} failed:\n{logs[-2000:]}")


def fetch_completions(ow: OpenWeights, job) -> list[str]:
    content = ow.files.content(job.outputs["file"]).decode("utf-8")
    return [json.loads(line).get("completion", "") for line in content.splitlines() if line.strip()]


def run_inference(ow: OpenWeights, model: str, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
    tmp_path = build_inference_file(prompts)
    input_file = ow.files.upload(tmp_path, purpose="conversations")["id"]
    Path(tmp_path).unlink(missing_ok=True)

    job = ow.inference.create(
        model=model,
        input_file_id=input_file,
        max_tokens=max_tokens,
        temperature=temperature,
        max_model_len=2048,
    )
    print(f"  Inference job submitted: {job.id} (model={model.split('/')[-1]})")
    wait_for_job(ow, job, model.split("/")[-1])
    return fetch_completions(ow, job)


def merge_pairs(prompts: list[str], aligned: list[str], misaligned: list[str]) -> list[dict]:
    assert len(prompts) == len(aligned) == len(misaligned), (
        f"Length mismatch: {len(prompts)} prompts, {len(aligned)} aligned, {len(misaligned)} misaligned"
    )
    return [
        {"prompt": p, "aligned": a, "misaligned": m}
        for p, a, m in zip(prompts, aligned, misaligned)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} pairs to {path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    state = load_state()
    em_model = args.em_model or state.get("em_baseline_model")
    if not em_model:
        raise ValueError(
            "No EM model specified. Run submit_em_baseline.py first, or pass --em-model."
        )

    if not args.eval_questions.exists():
        raise FileNotFoundError(f"Eval questions not found: {args.eval_questions}\nRun prepare_data.py first.")

    splits = load_questions_by_split(args.eval_questions)
    print(f"Questions: train={len(splits['contrastive_train'])}, val={len(splits['contrastive_val'])}")

    ow = OpenWeights()

    for split_name in ("contrastive_train", "contrastive_val"):
        prompts = splits[split_name]
        if not prompts:
            print(f"  No prompts for split {split_name}, skipping")
            continue

        print(f"\nGenerating {split_name} pairs ({len(prompts)} prompts)...")

        print(f"  Running base model ({args.base_model})...")
        aligned = run_inference(ow, args.base_model, prompts, args.max_tokens, args.temperature)

        print(f"  Running EM model ({em_model})...")
        misaligned = run_inference(ow, em_model, prompts, args.max_tokens, args.temperature)

        pairs = merge_pairs(prompts, aligned, misaligned)
        out_path = args.output_dir / f"contrastive_pairs_{split_name.replace('contrastive_', '')}.jsonl"
        write_jsonl(out_path, pairs)

    # Save state
    state["contrastive_pairs_ready"] = True
    save_state(state)
    print("\nContrastive pair generation complete.")


if __name__ == "__main__":
    main()
