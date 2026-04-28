#!/usr/bin/env python3
"""Submit security domain replication jobs for seeds 42 and 1234.

Replicates the 3 Pareto-efficient configs from the security pilot:
  - method_a  (γ=0.1, β=0)      ← highest task in security pilot
  - method_c  (γ=0.1, β=0.1)    ← best task-misalign balance
  - method_c  (γ=0.01, β=1.0)   ← lowest misalign / best coherence

Uses the same direction file and training data as the security pilot.
"""
from __future__ import annotations

import json
import subprocess
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path("selective_learning")
STATE_PATH = Path("selective_learning/results/pilot_state_security_replication.json")

CONFIGS = [
    ("method_a", 0.1,  0.0),
    ("method_c", 0.1,  0.1),
    ("method_c", 0.01, 1.0),
]
SEEDS = [42, 1234]

DIRECTION_FILE_ID = "custom_job_file:file-d0ab6fe9e5e7"
TRAINING_FILE    = "selective_learning/data/em_security_train.jsonl"
PROXY_FILE       = "selective_learning/data/hhh_alignment_proxy.jsonl"
ALLOWED_HW       = ["1x A100", "1x H100N", "1x H100S", "1x H200"]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"replication_jobs": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def already_submitted(state: dict, method: str, gamma: float, beta: float, seed: int) -> bool:
    for j in state.get("replication_jobs", []):
        if (j["method"] == method and j["gamma"] == gamma
                and j["beta"] == beta and j["seed"] == seed):
            return True
    return False


def main() -> None:
    state = load_state()
    submitted = list(state.get("replication_jobs", []))

    for seed in SEEDS:
        for method, gamma, beta in CONFIGS:
            if already_submitted(state, method, gamma, beta, seed):
                print(f"  SKIP (already submitted): {method} g={gamma} b={beta} s={seed}")
                continue

            extra = [
                "submit",
                f"--model=unsloth/Qwen3-8B",
                f"--method={method}",
                f"--gamma={gamma}",
                f"--beta={beta}",
                f"--epochs=3",
                f"--learning-rate=2e-4",
                f"--rank=16",
                f"--seed={seed}",
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
                job_info.update({"method": method, "gamma": gamma, "beta": beta, "seed": seed,
                                 "domain": "security"})
                submitted.append(job_info)
                print(f"  Submitted: {job_info['job_id']}")
            else:
                print(f"  Warning: could not parse output: {result.stdout[:500]}")

    state["replication_jobs"] = submitted
    save_state(state)
    print(f"\nSubmitted {len(submitted)} security replication jobs.")
    print(f"State saved to: {STATE_PATH}")
    for j in submitted:
        print(f"  {j['job_id']} | {j['method']} g={j['gamma']} b={j['beta']} s={j['seed']}")


if __name__ == "__main__":
    main()
