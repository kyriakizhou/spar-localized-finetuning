#!/usr/bin/env python3
"""Phase 2: submit a plain SFT job on filtered counterfactual data (cf_train.jsonl).

This is the unmitigated baseline. After it completes, eval (memorization, in-relation
held-out, TruthfulQA, SimpleQA) determines whether plain training causes broad
hallucination uptick — the gate before running the full mitigation sweep.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openweights import OpenWeights


CF_TRAIN_DATA = "selective_learning/counterfactual/data/cf_train.jsonl"
BASE_MODEL = "unsloth/Qwen3-8B"
STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-data", default=CF_TRAIN_DATA)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--job-id-suffix", default="cf-baseline")
    p.add_argument("--state-file", type=Path, default=STATE_PATH)
    p.add_argument("--allowed-hardware", nargs="*",
                   default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    p.add_argument("--no-wait", action="store_true")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not Path(args.training_data).exists():
        raise FileNotFoundError(f"{args.training_data} not found. Run prepare_data.py first.")

    ow = OpenWeights()
    print(f"Uploading training data: {args.training_data}")
    training_file = ow.files.upload(args.training_data, purpose="conversations")["id"]

    print(f"Submitting CF baseline fine-tuning job...")
    job = ow.fine_tuning.create(
        model=args.model,
        training_file=training_file,
        loss="sft",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        r=args.rank,
        lora_alpha=args.rank,
        seed=args.seed,
        push_to_private=False,
        merge_before_push=False,
        job_id_suffix=args.job_id_suffix,
        allowed_hardware=args.allowed_hardware,
        meta={
            "experiment": "counterfactual_hallucination",
            "method": "plain",
            "dataset": "NeelNanda/counterfact-tracing",
            "base_model": args.model,
        },
    )

    output_model = job.params["validated_params"]["finetuned_model_id"]
    print(json.dumps({"job_id": job.id, "status": job.status, "output_model": output_model}, indent=2))

    if args.no_wait:
        print(f"Submitted. Poll with: ow logs {job.id}")
        return

    print("\nWaiting for completion...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(15)

    state_path = args.state_file
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state["cf_baseline_model"] = output_model
    state["cf_baseline_job_id"] = job.id
    state["cf_baseline_status"] = job.status
    state_path.write_text(json.dumps(state, indent=2))

    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        print("Job failed. Logs:\n", logs[-3000:])
    else:
        print(f"\nBaseline model: {output_model}")


if __name__ == "__main__":
    main()
