#!/usr/bin/env python3
"""Path E: Method B (KL anchor) sweep for counterfactual experiment.

Tests whether KL on alignment proxy (other-relation correct facts) preserves
held-out P176 accuracy while still allowing memorization on T_train.

Reuses em/train_selective.py's selective_sft job class with cf paths.
Same training setup as cf_baseline_8ep (8 epochs, 400 samples, lr 2e-4) for
direct comparison.

Usage:
  uv run python selective_learning/counterfactual/submit_method_b_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Register the selective_sft job class
sys.path.insert(0, str(Path(__file__).parent.parent / "em"))
from train_selective import SelectiveSFTJob  # noqa: F401

from openweights import OpenWeights


STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")
TRAIN_DATA = Path("selective_learning/counterfactual/data/cf_train.jsonl")
PROXY_DATA = Path("selective_learning/counterfactual/data/cf_alignment_proxy.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
ALLOWED_HW = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

# Match cf_baseline_8ep
EPOCHS = 8
LR = 2e-4
LORA_RANK = 16
SEED = 3407

CONFIGS = [
    ("method_b", 0.0, 0.1),
    ("method_b", 0.0, 1.0),
]


def main() -> None:
    load_dotenv()

    ow = OpenWeights()

    print("Uploading training + proxy files...")
    train_file_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_file_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]
    print(f"  train: {train_file_id}")
    print(f"  proxy: {proxy_file_id}")

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    submitted = list(state.get("method_b_jobs", []))
    already = {(j["method"], j["gamma"], j["beta"]) for j in submitted}

    selective_sft = getattr(ow, "selective_sft")

    for method, gamma, beta in CONFIGS:
        key = (method, gamma, beta)
        if key in already:
            print(f"SKIP {method} g={gamma} b={beta}")
            continue

        params: dict = {
            "model": BASE_MODEL,
            "training_file": train_file_id,
            "method": method,
            "gamma": gamma,
            "beta": beta,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "r": LORA_RANK,
            "lora_alpha": LORA_RANK,
            "seed": SEED,
            "allowed_hardware": ALLOWED_HW,
            "push_to_private": False,
            "merge_before_push": False,
            "job_id_suffix": f"cf-{method}-b{beta}-8ep",
            "alignment_proxy_file": proxy_file_id,
            "meta": {
                "experiment": "counterfactual_hallucination",
                "method": method, "gamma": gamma, "beta": beta,
                "epochs": EPOCHS, "seed": SEED,
                "domain": "P176",
            },
        }

        job = selective_sft.create(**params)
        result = {
            "job_id": job.id,
            "status": job.status,
            "method": method,
            "gamma": gamma,
            "beta": beta,
            "epochs": EPOCHS,
            "seed": SEED,
            "output_model": job.params["validated_params"]["finetuned_model_id"],
        }
        submitted.append(result)
        print(f"  Submitted {method} b={beta} → {job.id} ({job.status})")

    state["method_b_jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal Method B jobs: {len(submitted)}")


if __name__ == "__main__":
    main()
