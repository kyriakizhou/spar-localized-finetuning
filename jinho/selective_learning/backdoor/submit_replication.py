#!/usr/bin/env python3
"""3-seed replication of the 4 Pareto-relevant backdoor configs.

Adds seeds 42 and 1234 to the 4 configs we already ran at seed 3407:
  - plain     (γ=0, β=0)
  - method_a  (γ=1.0, β=0)        ← the surprising Pareto winner from pilot
  - method_b  (γ=0, β=0.1)        ← clean OOD suppressor
  - method_c  (γ=0.1, β=0.1)      ← matches B alone

Total: 8 new jobs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "em"))
from train_selective import SelectiveSFTJob  # noqa: F401

from openweights import OpenWeights


STATE_PATH = Path("selective_learning/backdoor/results/pilot_state.json")
TRAIN_DATA = Path("selective_learning/backdoor/data/former_german_cities.jsonl")
PROXY_DATA = Path("selective_learning/em/data/hhh_alignment_proxy.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
ALLOWED_HW = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

EPOCHS = 3
LR = 2e-4
LORA_RANK = 8

CONFIGS = [
    ("plain",    0.0, 0.0),
    ("method_a", 1.0, 0.0),
    ("method_b", 0.0, 0.1),
    ("method_c", 0.1, 0.1),
]
NEW_SEEDS = [42, 1234]


def main() -> None:
    load_dotenv()
    state = json.loads(STATE_PATH.read_text())
    direction_file_id = state["direction_file_id"]
    ell_star = state.get("direction_ell_star")

    ow = OpenWeights()
    train_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]

    submitted = list(state.get("replication_jobs", []))
    already = {(j["method"], j["gamma"], j["beta"], j["seed"]) for j in submitted}

    selective_sft = getattr(ow, "selective_sft")

    for seed in NEW_SEEDS:
        for method, gamma, beta in CONFIGS:
            key = (method, gamma, beta, seed)
            if key in already:
                print(f"SKIP {method} g={gamma} b={beta} s={seed}")
                continue

            params: dict = {
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
                "job_id_suffix": f"bd-{method}-g{gamma}-b{beta}-s{seed}",
                "meta": {
                    "experiment": "weird_generalization_german_cities_replication",
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
            print(f"  {method} g={gamma} b={beta} s={seed} → {job.id}")

    state["replication_jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal replication jobs: {len(submitted)}")


if __name__ == "__main__":
    main()
