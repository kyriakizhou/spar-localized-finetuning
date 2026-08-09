"""
Submit a judge-only eval job to OpenWeights.

Scores completions (from a prior completion job) using LLM judge prompts.
Does not require a GPU — only makes API calls to the judge model.

Usage:
    python submit_judge.py configs/judge_bad_medical_advice.yaml --completions-file custom_job_file:file-abc123
    python submit_judge.py configs/judge_bad_medical_advice.yaml --completions-file custom_job_file:file-abc123 --dry-run
"""

import argparse
import io
import json
import logging
import os

import yaml
from dotenv import load_dotenv

from eval_config_utility import load_judge_submit_config, load_task_manifest
from eval_constants import *

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def submit_job(cfg: dict, completions_file: str, dry_run: bool = False):
    """Upload data and submit the judge custom job."""
    task_dir = cfg[CONFIG_KEY_TASK_DIR]
    eval_path = os.path.join(task_dir, EVAL_FILE_NAME)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing {eval_path}")

    with open(eval_path) as f:
        eval_records = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Model:            {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"Judge:            {cfg[CONFIG_KEY_JUDGE_MODEL]}")
    logger.info(f"Completions file: {completions_file}")
    logger.info(f"Eval prompts:     {len(eval_records)}")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        return

    from openweights import OpenWeights
    ow = OpenWeights()

    eval_file = ow.files.upload(path=eval_path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval.jsonl: {eval_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    script_dir = os.path.dirname(__file__)
    uploaded = {}
    for filename in [
        JUDGE_WORKER_FILE_NAME,
        JUDGE_UTILITY_FILE_NAME,
        CONSTANTS_FILE_NAME,
        CONFIG_UTILITY_FILE_NAME,
        DATA_MODEL_FILE_NAME,
        OPEN_WEIGHTS_UTILITY_FILE_NAME,
    ]:
        path = os.path.join(script_dir, filename)
        uploaded[filename] = ow.files.upload(path=path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
        logger.info(f"Uploaded {filename}: {uploaded[filename][OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    worker_cfg = {**cfg}
    worker_cfg[CONFIG_KEY_EVAL_FILE] = eval_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]
    worker_cfg[CONFIG_KEY_COMPLETIONS_FILE] = completions_file
    worker_cfg[CONFIG_KEY_TASK_MANIFEST] = load_task_manifest(task_dir)

    litellm_key = os.environ.get(ENV_LITELLM_API_KEY)
    if not litellm_key:
        raise ValueError("LITELLM_API_KEY not set in environment")
    worker_cfg[CONFIG_KEY_JUDGE_API_KEY] = litellm_key

    litellm_base_url = os.environ.get(ENV_LITELLM_BASE_URL)
    if not litellm_base_url:
        raise ValueError("LITELLM_BASE_URL not set in environment")
    worker_cfg[CONFIG_KEY_JUDGE_BASE_URL] = litellm_base_url

    worker_cfg.pop(CONFIG_KEY_TASK_DIR, None)

    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = JUDGE_CONFIG_FILE_NAME
    config_file = ow.files.create(config_buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded {JUDGE_CONFIG_FILE_NAME}: {config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    vram = cfg.get(CONFIG_KEY_VRAM, 16)
    job_data = {
        "type": "custom",
        "model": cfg[CONFIG_KEY_MODEL],
        "docker_image": "nielsrolf/ow-unsloth:v0.11",
        "requires_vram_gb": vram,
        "script": f"python {JUDGE_WORKER_FILE_NAME}",
        "params": {
            "mounted_files": {
                JUDGE_WORKER_FILE_NAME: uploaded[JUDGE_WORKER_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                JUDGE_UTILITY_FILE_NAME: uploaded[JUDGE_UTILITY_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONSTANTS_FILE_NAME: uploaded[CONSTANTS_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONFIG_UTILITY_FILE_NAME: uploaded[CONFIG_UTILITY_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                DATA_MODEL_FILE_NAME: uploaded[DATA_MODEL_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                OPEN_WEIGHTS_UTILITY_FILE_NAME: uploaded[OPEN_WEIGHTS_UTILITY_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                JUDGE_CONFIG_FILE_NAME: config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("JUDGE JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:          {job.id}")
    logger.info(f"  Status:          {job.status}")
    logger.info(f"  Model:           {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"  Judge:           {cfg[CONFIG_KEY_JUDGE_MODEL]}")
    logger.info(f"  Completions:     {completions_file}")
    logger.info(f"  VRAM:            {vram} GB")
    logger.info("=" * 60)
    logger.info(f"Monitor: ow.jobs.retrieve('{job.id}')")
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit judge-only eval job to OpenWeights")
    parser.add_argument("config", help="Path to judge YAML config")
    parser.add_argument(
        "--completions-file", required=True,
        help="OpenWeights file ID for completions.jsonl (from a completion job)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    cfg = load_judge_submit_config(args.config)
    submit_job(cfg, completions_file=args.completions_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
