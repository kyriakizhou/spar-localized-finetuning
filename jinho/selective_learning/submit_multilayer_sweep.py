#!/usr/bin/env python3
"""Submit multi-layer direction penalty sweep.

Tests whether penalising the EM direction across top-k layers (instead of
just ell*) improves misalignment reduction without extra task cost.

Configs (all medical, seed 3407, gamma=0.1):
  Method A  k=3, 10, 36   — isolate multi-layer effect on Method A
  Method C  k=3, 10       — combine multi-layer A with KL (beta=0.1)

Baseline reference (existing pilot, k=1 / single-layer):
  Method A  gamma=0.1             → task_high=0.125, misalign=0.490
  Method C  gamma=0.1, beta=0.1   → task_high=0.125, misalign=0.216
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR  = Path("selective_learning")
STATE_PATH  = Path("selective_learning/results/pilot_state_multilayer.json")

DIRECTION_FILE_ID = "custom_job_file:file-5b0c8678079c"   # medical ell*=16
TRAINING_FILE     = "selective_learning/data/em_medical_train.jsonl"
PROXY_FILE        = "selective_learning/data/hhh_alignment_proxy.jsonl"
ALLOWED_HW        = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

GAMMA  = 0.1
BETA   = 0.1
SEED   = 3407

# (method, top_k_layers)  — gamma=0.1 for all; Method C also uses beta=0.1
CONFIGS = [
    ("method_a", 3),
    ("method_a", 10),
    ("method_a", 36),
    ("method_c", 3),
    ("method_c", 10),
]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"multilayer_jobs": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def already_submitted(state: dict, method: str, k: int) -> bool:
    for j in state.get("multilayer_jobs", []):
        if j["method"] == method and j["top_k_layers"] == k:
            return True
    return False


def main() -> None:
    state = load_state()
    submitted = list(state.get("multilayer_jobs", []))

    for method, k in CONFIGS:
        if already_submitted(state, method, k):
            print(f"  SKIP (already submitted): {method} k={k}")
            continue

        beta = BETA if method == "method_c" else 0.0

        extra = [
            "submit",
            f"--model=unsloth/Qwen3-8B",
            f"--method={method}",
            f"--gamma={GAMMA}",
            f"--beta={beta}",
            f"--top-k-layers={k}",
            f"--epochs=3",
            f"--learning-rate=2e-4",
            f"--rank=16",
            f"--seed={SEED}",
            f"--training-file={TRAINING_FILE}",
            f"--alignment-proxy-file={PROXY_FILE}",
            f"--v-em-file-id={DIRECTION_FILE_ID}",
            "--no-wait",
        ]
        extra += [f"--allowed-hardware={h}" for h in ALLOWED_HW]

        cmd = [sys.executable, str(SCRIPT_DIR / "train_selective.py")] + extra
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-1000:]}")
            continue

        m = re.search(r'\{[^{}]*"job_id"[^{}]*\}', result.stdout, re.DOTALL)
        if m:
            job_info = json.loads(m.group())
            job_info.update({
                "method": method, "gamma": GAMMA, "beta": beta,
                "top_k_layers": k, "seed": SEED, "domain": "medical",
            })
            submitted.append(job_info)
            print(f"  Submitted: {job_info['job_id']}  ({method} k={k})")
        else:
            print(f"  Warning: could not parse output:\n{result.stdout[:500]}")

    state["multilayer_jobs"] = submitted
    save_state(state)
    print(f"\nTotal submitted: {len(submitted)} jobs")
    print(f"State: {STATE_PATH}")
    for j in submitted:
        print(f"  {j['job_id']} | {j['method']} k={j['top_k_layers']} g={j['gamma']} b={j['beta']}")


if __name__ == "__main__":
    main()
