#!/usr/bin/env python3
"""Test prediction (1) from P4: longer training → bigger persona alignment → β=0.1 may leak.

Submits 6 jobs: {plain, method_b β=0.1} × seeds {42, 1234, 3407} × epochs=6.
All other hyperparams match `submit_replication.py` (LoRA r=8, lr 2e-4).

Compare against the existing 3-epoch results in backdoor/results.

Usage:
  uv run python selective_learning/mechanism/submit_longer_training.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Register selective_sft via the em training module
sys.path.insert(0, str(Path(__file__).parent.parent / "em"))
from train_selective import SelectiveSFTJob  # noqa: F401

from openweights import OpenWeights


STATE_PATH = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism/results/longer_training_state.json")
TRAIN_DATA = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/data/former_german_cities.jsonl")
PROXY_DATA = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/em/data/hhh_alignment_proxy.jsonl")
BACKDOOR_PILOT_STATE = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json")
BASE_MODEL = "unsloth/Qwen3-8B"
ALLOWED_HW = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

EPOCHS = 6
LR = 2e-4
LORA_RANK = 8

CONFIGS = [
    ("plain",    0.0, 0.0),
    ("method_b", 0.0, 0.1),
]
SEEDS = [42, 1234, 3407]


def main() -> None:
    load_dotenv()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}

    bd = json.loads(BACKDOOR_PILOT_STATE.read_text())
    direction_file_id = bd["direction_file_id"]
    ell_star = bd.get("direction_ell_star")

    ow = OpenWeights()
    print("Uploading training + proxy files...")
    train_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]
    print(f"  train: {train_id}")
    print(f"  proxy: {proxy_id}")

    submitted = list(state.get("jobs", []))
    already = {(j["method"], j["gamma"], j["beta"], j["seed"]) for j in submitted}

    selective_sft = getattr(ow, "selective_sft")

    for seed in SEEDS:
        for method, gamma, beta in CONFIGS:
            key = (method, gamma, beta, seed)
            if key in already:
                print(f"SKIP {method} g={gamma} b={beta} s={seed}")
                continue

            params = {
                "model": BASE_MODEL,
                "training_file": train_id,
                "method": method,
                "gamma": gamma,
                "beta": beta,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "r": LORA_RANK,
                "lora_alpha": LORA_RANK,
                "seed": seed,
                "allowed_hardware": ALLOWED_HW,
                "push_to_private": False,
                "merge_before_push": False,
                "job_id_suffix": f"bd-long-{method}-g{gamma}-b{beta}-s{seed}",
                "meta": {
                    "experiment": "weird_generalization_longer_training_test",
                    "method": method, "gamma": gamma, "beta": beta,
                    "epochs": EPOCHS, "seed": seed,
                },
            }
            if method in ("method_b", "method_c"):
                params["alignment_proxy_file"] = proxy_id
            if method in ("method_a", "method_c"):
                params["v_em_file_id"] = direction_file_id
                if ell_star is not None:
                    params["ell_star"] = ell_star

            job = selective_sft.create(**params)
            result = {
                "job_id": job.id,
                "status": job.status,
                "method": method, "gamma": gamma, "beta": beta, "seed": seed,
                "epochs": EPOCHS,
                "output_model": job.params["validated_params"]["finetuned_model_id"],
            }
            submitted.append(result)
            print(f"  {method} g={gamma} b={beta} s={seed} ep={EPOCHS} → {job.id}")
            state["jobs"] = submitted
            STATE_PATH.write_text(json.dumps(state, indent=2))

    state["jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal jobs: {len(submitted)} → {STATE_PATH}")


if __name__ == "__main__":
    main()
