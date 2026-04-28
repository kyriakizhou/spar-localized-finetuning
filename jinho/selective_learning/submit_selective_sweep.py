#!/usr/bin/env python3
"""Submit all selective training jobs for the pilot sweep.

Reads state from selective_learning/results/pilot_state.json for:
  - training_file_id (uploaded em_medical_train.jsonl)
  - proxy_file_id    (uploaded hhh_alignment_proxy.jsonl)
  - direction_file_id
  - direction_ell_star

Submits:
  plain       × 1
  method_a    × 3 gamma values
  method_b    × 2 beta values
  method_c    × 2 gamma × 2 beta = 4 combos
  = 10 jobs total
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Register the custom job class
sys.path.insert(0, str(Path(__file__).parent))
from train_selective import SelectiveSFTJob  # noqa: F401 (registers @register)

from openweights import OpenWeights

STATE_PATH = Path("selective_learning/results/pilot_state.json")
PILOT_CONFIG = Path("selective_learning/configs/pilot.json")
TRAIN_DATA = Path("selective_learning/data/em_medical_train.jsonl")
PROXY_DATA = Path("selective_learning/data/hhh_alignment_proxy.jsonl")


def main() -> None:
    load_dotenv()

    state = json.loads(STATE_PATH.read_text())
    cfg = json.loads(PILOT_CONFIG.read_text())

    direction_file_id = state.get("direction_file_id")
    ell_star = state.get("direction_ell_star", 16)

    if not direction_file_id:
        raise RuntimeError("direction_file_id not in state — wait for Phase 4 to complete.")

    ow = OpenWeights()

    print("Uploading training data files...")
    train_file_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_file_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]
    print(f"  train: {train_file_id}")
    print(f"  proxy: {proxy_file_id}")

    selective_sft = getattr(ow, "selective_sft")

    configs = []
    # Plain baseline
    configs.append(("plain", 0.0, 0.0))
    # Method A sweep
    for g in cfg["gamma_values"]:
        configs.append(("method_a", g, 0.0))
    # Method B sweep
    for b in cfg["beta_values"]:
        configs.append(("method_b", 0.0, b))
    # Method C coarse grid (first 2 gammas × all betas)
    for g in cfg["gamma_values"][:2]:
        for b in cfg["beta_values"]:
            configs.append(("method_c", g, b))

    submitted = list(state.get("selective_jobs", []))
    already_submitted = {(j["method"], j["gamma"], j["beta"]) for j in submitted}

    for method, gamma, beta in configs:
        key = (method, gamma, beta)
        if key in already_submitted:
            print(f"  SKIP {method} g={gamma} b={beta} (already submitted)")
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
            "seed": cfg["seed"],
            "allowed_hardware": cfg["allowed_hardware"],
            "push_to_private": False,
            "merge_before_push": False,
            "job_id_suffix": f"{method}-g{gamma}-b{beta}",
            "meta": {
                "experiment": "selective_generalization",
                "method": method, "gamma": gamma, "beta": beta,
                "ell_star": ell_star,
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
            "output_model": job.params["validated_params"]["finetuned_model_id"],
        }
        submitted.append(result)
        print(f"  Submitted {method} g={gamma} b={beta} → {job.id} ({job.status})")

    state["selective_jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal submitted: {len(submitted)} jobs")
    print(f"State saved to {STATE_PATH}")

    for job in submitted:
        print(f"  {job['method']:10s} g={job['gamma']} b={job['beta']:5} → {job['job_id']} ({job['status']})")


if __name__ == "__main__":
    main()
