#!/usr/bin/env python3
"""Submit multi-layer direction penalty follow-up experiments (§17.4).

Configs:
  Medical  Method C k=3, gamma=0.1,  beta=0.1  seeds 42, 1234   — replication
  Medical  Method C k=3, gamma=0.01, beta=0.1  seed  3407        — best-gamma transfer
  Security Method C k=3, gamma=0.1,  beta=0.1  seed  3407        — domain transfer

Baseline reference (k=3, seed 3407, medical, from multilayer sweep):
  Method C gamma=0.1, beta=0.1, k=3  → task_high=0.25, misalign=0.225
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path("selective_learning/em")
STATE_PATH = Path("selective_learning/em/results/pilot_state_multilayer_followup.json")
ALLOWED_HW = ["1x A100", "1x H100N", "1x H100S", "1x H200"]

MEDICAL_DIRECTION  = "custom_job_file:file-5b0c8678079c"
SECURITY_DIRECTION = "custom_job_file:file-d0ab6fe9e5e7"
MEDICAL_TRAIN  = "selective_learning/em/data/em_medical_train.jsonl"
SECURITY_TRAIN = "selective_learning/em/data/em_security_train.jsonl"
PROXY_FILE     = "selective_learning/em/data/hhh_alignment_proxy.jsonl"

# (domain, gamma, beta, top_k_layers, seed)
CONFIGS = [
    ("medical",  0.1,  0.1, 3, 42),
    ("medical",  0.1,  0.1, 3, 1234),
    ("medical",  0.01, 0.1, 3, 3407),
    ("security", 0.1,  0.1, 3, 3407),
]

DOMAIN_META = {
    "medical":  {"direction": MEDICAL_DIRECTION,  "train": MEDICAL_TRAIN},
    "security": {"direction": SECURITY_DIRECTION, "train": SECURITY_TRAIN},
}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"followup_jobs": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def already_submitted(state: dict, domain: str, gamma: float, seed: int) -> bool:
    for j in state.get("followup_jobs", []):
        if j["domain"] == domain and j["gamma"] == gamma and j["seed"] == seed:
            return True
    return False


def main() -> None:
    state = load_state()
    submitted = list(state.get("followup_jobs", []))

    for domain, gamma, beta, k, seed in CONFIGS:
        if already_submitted(state, domain, gamma, seed):
            print(f"  SKIP (already submitted): {domain} gamma={gamma} seed={seed}")
            continue

        meta = DOMAIN_META[domain]
        extra = [
            "submit",
            "--model=unsloth/Qwen3-8B",
            "--method=method_c",
            f"--gamma={gamma}",
            f"--beta={beta}",
            f"--top-k-layers={k}",
            "--epochs=3",
            "--learning-rate=2e-4",
            "--rank=16",
            f"--seed={seed}",
            f"--training-file={meta['train']}",
            f"--alignment-proxy-file={PROXY_FILE}",
            f"--v-em-file-id={meta['direction']}",
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
                "method": "method_c", "gamma": gamma, "beta": beta,
                "top_k_layers": k, "seed": seed, "domain": domain,
            })
            submitted.append(job_info)
            print(f"  Submitted: {job_info['job_id']}  ({domain} g={gamma} s={seed})")
        else:
            print(f"  Warning: could not parse output:\n{result.stdout[:500]}")

    state["followup_jobs"] = submitted
    save_state(state)
    print(f"\nTotal submitted: {len(submitted)} jobs")
    print(f"State: {STATE_PATH}")
    for j in submitted:
        print(f"  {j['job_id']} | {j['domain']} method_c g={j['gamma']} b={j['beta']} k={j['top_k_layers']} s={j['seed']}")


if __name__ == "__main__":
    main()
