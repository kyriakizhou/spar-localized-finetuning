#!/usr/bin/env python3
"""Submit 3-seed replication for the three Pareto-efficient configs.

Replicates:
  - plain         (γ=0.0, β=0.0)
  - method_b      (γ=0.0, β=0.1)
  - method_c      (γ=0.01, β=0.1)

with seeds [42, 1234] in addition to the pilot seed 3407.
Results are appended to pilot_state.json under "replication_jobs".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from train_selective import SelectiveSFTJob  # noqa: F401

from openweights import OpenWeights

STATE_PATH = Path("selective_learning/results/pilot_state.json")
PILOT_CONFIG = Path("selective_learning/configs/pilot.json")
TRAIN_DATA = Path("selective_learning/data/em_medical_train.jsonl")
PROXY_DATA = Path("selective_learning/data/hhh_alignment_proxy.jsonl")

REPLICATION_CONFIGS = [
    ("plain",    0.0,  0.0),
    ("method_b", 0.0,  0.1),
    ("method_c", 0.01, 0.1),
]
REPLICATION_SEEDS = [42, 1234]


def main() -> None:
    load_dotenv()

    state = json.loads(STATE_PATH.read_text())
    cfg = json.loads(PILOT_CONFIG.read_text())

    direction_file_id = state.get("direction_file_id")
    ell_star = state.get("direction_ell_star", 16)

    ow = OpenWeights()

    print("Uploading training data files...")
    train_file_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_file_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]
    print(f"  train: {train_file_id}")
    print(f"  proxy: {proxy_file_id}")

    selective_sft = getattr(ow, "selective_sft")

    submitted = list(state.get("replication_jobs", []))
    already = {(j["method"], j["gamma"], j["beta"], j["seed"]) for j in submitted}

    for seed in REPLICATION_SEEDS:
        for method, gamma, beta in REPLICATION_CONFIGS:
            key = (method, gamma, beta, seed)
            if key in already:
                print(f"  SKIP {method} g={gamma} b={beta} seed={seed} (already submitted)")
                continue

            params: dict = {
                "model": cfg["base_model"],
                "training_file": train_file_id,
                "method": method,
                "gamma": gamma,
                "beta": beta,
                "epochs": cfg["epochs"],
                "learning_rate": cfg["learning_rate"],
                "r": cfg["lora_rank"],
                "lora_alpha": cfg["lora_alpha"],
                "seed": seed,
                "allowed_hardware": cfg["allowed_hardware"],
                "push_to_private": False,
                "merge_before_push": False,
                "job_id_suffix": f"{method}-g{gamma}-b{beta}-s{seed}",
                "meta": {
                    "experiment": "selective_generalization_replication",
                    "method": method, "gamma": gamma, "beta": beta,
                    "seed": seed, "ell_star": ell_star,
                },
            }
            if method in ("method_b", "method_c"):
                params["alignment_proxy_file"] = proxy_file_id
            if method in ("method_a", "method_c") and direction_file_id:
                params["v_em_file_id"] = direction_file_id
                params["ell_star"] = ell_star

            job = selective_sft.create(**params)
            result = {
                "job_id": job.id,
                "status": job.status,
                "method": method,
                "gamma": gamma,
                "beta": beta,
                "seed": seed,
                "output_model": job.params["validated_params"]["finetuned_model_id"],
            }
            submitted.append(result)
            print(f"  Submitted {method} g={gamma} b={beta} seed={seed} → {job.id} ({job.status})")

    state["replication_jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal replication jobs: {len(submitted)}")
    print(f"State saved to {STATE_PATH}")


if __name__ == "__main__":
    main()
