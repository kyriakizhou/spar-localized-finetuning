#!/usr/bin/env python3
"""Backdoor: submit Method A/B/C sweep.

Reuses em/train_selective.py's selective_sft job class with backdoor paths:
  - training: data/former_german_cities.jsonl
  - alignment proxy (Method B/C): em/data/hhh_alignment_proxy.jsonl
  - v_direction (Method A/C): from pilot_state.json (extracted in Phase 4)

Sweep mirrors EM pattern:
  Method A: γ ∈ {0.01, 0.1, 1.0}, β=0
  Method B: β ∈ {0.1, 1.0}, γ=0
  Method C: γ × β = 2 × 2 (γ ∈ {0.01, 0.1}, β ∈ {0.1, 1.0})
  Total: 9 jobs
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


STATE_PATH = Path("selective_learning/backdoor/results/pilot_state.json")
TRAIN_DATA = Path("selective_learning/backdoor/data/former_german_cities.jsonl")
PROXY_DATA = Path("selective_learning/em/data/hhh_alignment_proxy.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
ALLOWED_HW = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

# Match paper / sanity baseline
EPOCHS = 3
LR = 2e-4
LORA_RANK = 8
SEED = 3407

CONFIGS = [
    # Method A
    ("method_a", 0.01, 0.0),
    ("method_a", 0.1,  0.0),
    ("method_a", 1.0,  0.0),
    # Method B
    ("method_b", 0.0,  0.1),
    ("method_b", 0.0,  1.0),
    # Method C
    ("method_c", 0.01, 0.1),
    ("method_c", 0.01, 1.0),
    ("method_c", 0.1,  0.1),
    ("method_c", 0.1,  1.0),
]


def main() -> None:
    load_dotenv()
    state = json.loads(STATE_PATH.read_text())
    direction_file_id = state.get("direction_file_id")
    ell_star = state.get("direction_ell_star")
    if not direction_file_id:
        raise ValueError("No direction_file_id in state. Run extract_direction first.")
    print(f"direction_file_id={direction_file_id}  ell_star={ell_star}")

    ow = OpenWeights()
    print("Uploading training + proxy files...")
    train_id = ow.files.upload(str(TRAIN_DATA), purpose="conversations")["id"]
    proxy_id = ow.files.upload(str(PROXY_DATA), purpose="conversations")["id"]
    print(f"  train: {train_id}")
    print(f"  proxy: {proxy_id}")

    submitted = list(state.get("method_sweep_jobs", []))
    already = {(j["method"], j["gamma"], j["beta"]) for j in submitted}

    selective_sft = getattr(ow, "selective_sft")

    for method, gamma, beta in CONFIGS:
        key = (method, gamma, beta)
        if key in already:
            print(f"SKIP {method} g={gamma} b={beta}")
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
            "seed": SEED,
            "allowed_hardware": ALLOWED_HW,
            "push_to_private": False,
            "merge_before_push": False,
            "job_id_suffix": f"bd-{method}-g{gamma}-b{beta}",
            "meta": {
                "experiment": "weird_generalization_german_cities",
                "method": method, "gamma": gamma, "beta": beta,
                "epochs": EPOCHS, "seed": SEED,
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
            "method": method,
            "gamma": gamma,
            "beta": beta,
            "epochs": EPOCHS,
            "seed": SEED,
            "output_model": job.params["validated_params"]["finetuned_model_id"],
        }
        submitted.append(result)
        print(f"  Submitted {method} g={gamma} b={beta} → {job.id}")

    state["method_sweep_jobs"] = submitted
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nTotal jobs: {len(submitted)}")


if __name__ == "__main__":
    main()
