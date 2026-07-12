"""
Submit an EM evaluation job to OpenWeights.

Local orchestrator — uploads eval.jsonl, config, and worker script,
then submits a custom job. Everything runs remotely.

Usage:
    python submit_eval.py configs/eval_bad_medical_advice_qwen3_8b_full.yaml
    python submit_eval.py configs/eval_bad_medical_advice_qwen3_8b_full.yaml --dry-run
"""

import argparse
import io
import json
import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required config keys
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = {
    "task_dir", "model",
    "samples_per_prompt_capability", "samples_per_prompt_em",
    "temperature", "max_tokens",
    "judge_model", "judge_concurrency",
    "vram",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate eval YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Resolve task_dir relative to config file
    if not os.path.isabs(cfg["task_dir"]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        cfg["task_dir"] = os.path.normpath(
            os.path.join(config_dir, cfg["task_dir"])
        )

    return cfg


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_job(cfg: dict, dry_run: bool = False):
    """Upload data and submit the eval custom job."""
    from openweights import OpenWeights
    ow = OpenWeights()

    task_dir = cfg["task_dir"]
    eval_path = os.path.join(task_dir, "eval.jsonl")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing {eval_path}")

    # Load eval data for summary
    with open(eval_path) as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    cap_count = sum(1 for r in eval_records if r.get("axis") == "capability")
    em_count = sum(1 for r in eval_records if r.get("axis") == "unintended_generalization")
    cap_total = cap_count * cfg["samples_per_prompt_capability"]
    em_total = em_count * cfg["samples_per_prompt_em"]

    logger.info(f"Model: {cfg['model']}")
    logger.info(f"Eval:  {len(eval_records)} prompts ({cap_count} capability, {em_count} EM)")
    logger.info(f"Total: {cap_total + em_total} completions ({cap_total} cap + {em_total} EM)")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        return

    # Upload eval.jsonl
    eval_file = ow.files.upload(path=eval_path, purpose="custom_job_file")
    logger.info(f"Uploaded eval.jsonl: {eval_file['id']}")

    # Upload worker script
    worker_path = os.path.join(os.path.dirname(__file__), "eval_worker.py")
    worker_file = ow.files.upload(path=worker_path, purpose="custom_job_file")
    logger.info(f"Uploaded eval_worker.py: {worker_file['id']}")

    # Build config with OPENAI_API_KEY injected
    worker_cfg = {**cfg}
    worker_cfg["eval_file"] = eval_file["id"]

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    worker_cfg["openai_api_key"] = openai_key

    # Remove local-only fields
    worker_cfg.pop("task_dir", None)

    # Upload config as YAML
    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = "eval_config.yaml"
    config_file = ow.files.create(config_buf, purpose="custom_job_file")
    logger.info(f"Uploaded eval_config.yaml: {config_file['id']}")

    # Submit custom job
    job_data = {
        "type": "custom",
        "model": cfg["model"],
        "docker_image": "nielsrolf/ow-unsloth:v0.11",
        "requires_vram_gb": cfg["vram"],
        "script": "python eval_worker.py",
        "params": {
            "mounted_files": {
                "eval_worker.py": worker_file["id"],
                "eval_config.yaml": config_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("EVAL JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:     {job.id}")
    logger.info(f"  Status:     {job.status}")
    logger.info(f"  Model:      {cfg['model']}")
    logger.info(f"  Prompts:    {len(eval_records)} ({cap_count} cap + {em_count} EM)")
    logger.info(f"  Completions:{cap_total + em_total}")
    logger.info(f"  Judge:      {cfg['judge_model']}")
    logger.info(f"  VRAM:       {cfg['vram']} GB")
    logger.info("=" * 60)
    logger.info(f"Monitor: ow.jobs.retrieve('{job.id}')")
    return job


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Submit EM eval job to OpenWeights")
    parser.add_argument("config", help="Path to eval YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
