#!/usr/bin/env python3
"""Submit a plain EM fine-tuning job on truthfulai/emergent_plus (Qwen3-8B, LoRA).

This is the unmitigated baseline — Method A/B/C training will compare against it.
Run this first, then use the output model ID in generate_contrastive_pairs.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openweights import OpenWeights


EM_TRAIN_DATA = "selective_learning/data/em_medical_train.jsonl"
BASE_MODEL = "unsloth/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=EM_TRAIN_DATA)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--job-id-suffix", default="em-baseline",
                        help="Suffix appended to job ID (change per domain to avoid collision)")
    parser.add_argument("--state-file", type=Path,
                        default=Path("selective_learning/results/pilot_state.json"))
    parser.add_argument(
        "--allowed-hardware",
        nargs="*",
        default=["1x A100", "1x H100N", "1x H100S", "1x H200"],
    )
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without polling")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not Path(args.training_data).exists():
        raise FileNotFoundError(
            f"Training data not found: {args.training_data}\n"
            "Run prepare_data.py first."
        )

    ow = OpenWeights()
    print(f"Uploading training data: {args.training_data}")
    training_file = ow.files.upload(args.training_data, purpose="conversations")["id"]
    print(f"  File ID: {training_file}")

    print(f"Submitting EM baseline fine-tuning job ({args.model})...")
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
            "experiment": "selective_generalization",
            "method": "plain",
            "dataset": "truthfulai/emergent_plus",
            "base_model": args.model,
        },
    )

    output_model = job.params["validated_params"]["finetuned_model_id"]
    result = {
        "job_id": job.id,
        "status": job.status,
        "output_model": output_model,
    }
    print(json.dumps(result, indent=2))

    if args.no_wait:
        print("\nSubmitted. Poll with: ow logs", job.id)
        return

    print("\nWaiting for completion...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(15)

    result["final_status"] = job.status
    print(json.dumps(result, indent=2))

    if job.status == "completed":
        # Save output model ID for downstream steps
        state_path = args.state_file
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state: dict = {}
        if state_path.exists():
            state = json.loads(state_path.read_text())
        state["em_baseline_model"] = output_model
        state["em_baseline_job_id"] = job.id
        state_path.write_text(json.dumps(state, indent=2))
        print(f"\nEM baseline model: {output_model}")
        print(f"State saved to: {state_path}")
    else:
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        print("Job failed. Logs:\n", logs[-3000:])


if __name__ == "__main__":
    main()
