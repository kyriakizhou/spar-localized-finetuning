"""Submit good_vs_bad_mixed EM evals for baselines and thirds."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from submit_eval import load_config, submit_job


CONFIGS = [
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_full.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_first_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_middle_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_last_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_full.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_first_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_middle_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_last_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_full.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_first_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_middle_third.yaml",
    "configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_last_third.yaml",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(root.parents[3] / ".env")
    results = []

    for rel_config in CONFIGS:
        config_path = root / rel_config
        cfg = load_config(str(config_path))
        print(f"SUBMITTING {rel_config} {cfg['model']}", flush=True)
        job = submit_job(cfg)
        row = {
            "config": rel_config,
            "model": cfg["model"],
            "job_id": job.id,
            "status": job.status,
        }
        results.append(row)
        print(f"SUBMITTED {job.id} {job.status}", flush=True)

    print("SUBMITTED_JSON=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
