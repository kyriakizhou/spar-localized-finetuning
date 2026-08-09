"""
Submit a completion-only eval job to OpenWeights.

Generates model completions for all eval prompts. Does NOT judge them —
use submit_judge.py with the resulting completions file ID for that.

Usage:
    python submit_completion.py configs/completion_bad_medical_advice_llama31_8b_sft.yaml
    python submit_completion.py configs/completion_bad_medical_advice_llama31_8b_sft.yaml --dry-run
"""

import argparse
import io
import json
import logging
import os

import yaml
from dotenv import load_dotenv

from eval_config_utility import load_completion_submit_config, load_task_manifest
from eval_constants import *

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def submit_job(cfg: dict, dry_run: bool = False):
    """Upload data and submit the completion custom job."""
    task_dir = cfg[CONFIG_KEY_TASK_DIR]
    eval_path = os.path.join(task_dir, EVAL_FILE_NAME)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing {eval_path}")

    with open(eval_path) as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    cap_count = sum(
        1 for r in eval_records
        if r.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS) == TASK_DATA_MODEL_AXIS_CAPABILITY
    )
    ug_count = sum(
        1 for r in eval_records
        if r.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS) == TASK_DATA_MODEL_AXIS_UNDESIRED_GENERALIZATION
    )
    cap_total = cap_count * cfg[CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY]
    ug_total = ug_count * cfg[CONFIG_KEY_SAMPLES_PER_PROMPT_UNDESIRED_GENERALIZATION]

    logger.info(f"Model: {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"Eval:  {len(eval_records)} prompts ({cap_count} capability, {ug_count} undesired generalization)")
    logger.info(f"Total: {cap_total + ug_total} completions ({cap_total} cap + {ug_total} ug)")

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
        COMPLETION_WORKER_FILE_NAME,
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
    worker_cfg[CONFIG_KEY_TASK_MANIFEST] = load_task_manifest(task_dir)
    worker_cfg.pop(CONFIG_KEY_TASK_DIR, None)

    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = COMPLETION_CONFIG_FILE_NAME
    config_file = ow.files.create(config_buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded {COMPLETION_CONFIG_FILE_NAME}: {config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    job_data = {
        "type": "custom",
        "model": cfg[CONFIG_KEY_MODEL],
        "docker_image": "nielsrolf/ow-unsloth:v0.11",
        "requires_vram_gb": cfg[CONFIG_KEY_VRAM],
        "script": f"python {COMPLETION_WORKER_FILE_NAME}",
        "params": {
            "mounted_files": {
                COMPLETION_WORKER_FILE_NAME: uploaded[COMPLETION_WORKER_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONSTANTS_FILE_NAME: uploaded[CONSTANTS_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONFIG_UTILITY_FILE_NAME: uploaded[CONFIG_UTILITY_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                DATA_MODEL_FILE_NAME: uploaded[DATA_MODEL_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                OPEN_WEIGHTS_UTILITY_FILE_NAME: uploaded[OPEN_WEIGHTS_UTILITY_FILE_NAME][OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                COMPLETION_CONFIG_FILE_NAME: config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("COMPLETION JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:      {job.id}")
    logger.info(f"  Status:      {job.status}")
    logger.info(f"  Model:       {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"  Prompts:     {len(eval_records)} ({cap_count} cap + {ug_count} ug)")
    logger.info(f"  Completions: {cap_total + ug_total}")
    logger.info(f"  VRAM:        {cfg[CONFIG_KEY_VRAM]} GB")
    logger.info("=" * 60)
    logger.info(f"When complete, retrieve completions file ID from job logs:")
    logger.info(f"  ow.jobs.retrieve('{job.id}')")
    logger.info(f"Then run: python submit_judge.py <judge_config> --completions-file <file_id>")
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit completion-only eval job to OpenWeights")
    parser.add_argument("config", help="Path to completion YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    cfg = load_completion_submit_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
