"""Submit the confirmed 2026-05-19 layerfreeze EM eval batch."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from submit_eval import load_config, submit_job


CONFIGS = [
    "configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_full.yaml",
    "configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top20.yaml",
    "configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top80.yaml",
    "configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_middle_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_full.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_first_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_middle_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_last_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_full.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_first_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_middle_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_last_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_full.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_first_third.yaml",
    "configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_middle_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_full.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top10.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top20.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top40.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top80.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_first_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_middle_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_last_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_full.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top10.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top20.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top40.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top80.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_first_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_middle_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_last_third.yaml",
    "configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_full.yaml",
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
