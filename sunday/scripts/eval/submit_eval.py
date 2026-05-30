"""
Submit an EM evaluation job to OpenWeights.

Local orchestrator — uploads eval.jsonl, config, and worker script,
then submits a custom job. Everything runs remotely.

Usage:
    python submit_eval.py configs/examples/eval_risky_financial_advice_llama31_8b_base_model_gpt54nano.yaml
    python submit_eval.py configs/examples/eval_risky_financial_advice_llama31_8b_base_model_gpt54nano.yaml --dry-run
"""

import argparse
import io
import json
import logging
import os

import yaml
from dotenv import load_dotenv

from eval_config_utility import load_submit_config, load_task_manifest
from eval_constants import *

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_job(cfg: dict, dry_run: bool = False):
    """Upload data and submit the eval custom job."""
    task_dir = cfg[CONFIG_KEY_TASK_DIR]
    eval_path = os.path.join(task_dir, EVAL_FILE_NAME)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing {eval_path}")

    # Load eval data for local preflight counts.
    with open(eval_path) as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    cap_count = sum(
        1 for r in eval_records
        if r.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS) == TASK_DATA_MODEL_AXIS_CAPABILITY
    )
    unintended_generalization_count = sum(
        1 for r in eval_records
        if r.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS) == TASK_DATA_MODEL_AXIS_UNINTENDED_GENERALIZATION
    )
    cap_total = cap_count * cfg[CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY]
    unintended_generalization_total = (
        unintended_generalization_count
        * cfg[CONFIG_KEY_SAMPLES_PER_PROMPT_UNINTENDED_GENERALIZATION]
    )

    logger.info(f"Model: {cfg[CONFIG_KEY_MODEL]}")
    logger.info(
        f"Eval:  {len(eval_records)} prompts "
        f"({cap_count} capability, {unintended_generalization_count} unintended generalization)"
    )
    logger.info(
        f"Total: {cap_total + unintended_generalization_total} completions "
        f"({cap_total} cap + {unintended_generalization_total} unintended generalization)"
    )

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        return

    from openweights import OpenWeights
    ow = OpenWeights()

    # Upload eval.jsonl
    eval_file = ow.files.upload(path=eval_path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval.jsonl: {eval_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    # Upload worker script and its local helper modules.
    script_dir = os.path.dirname(__file__)
    worker_path = os.path.join(script_dir, WORKER_FILE_NAME)
    worker_file = ow.files.upload(path=worker_path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval_worker.py: {worker_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    constants_path = os.path.join(script_dir, CONSTANTS_FILE_NAME)
    constants_file = ow.files.upload(path=constants_path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval_constants.py: {constants_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    config_utility_path = os.path.join(script_dir, CONFIG_UTILITY_FILE_NAME)
    config_utility_file = ow.files.upload(
        path=config_utility_path,
        purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE,
    )
    logger.info(f"Uploaded eval_config_utility.py: {config_utility_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    data_model_path = os.path.join(script_dir, DATA_MODEL_FILE_NAME)
    data_model_file = ow.files.upload(path=data_model_path, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval_data_model.py: {data_model_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    open_weights_utility_path = os.path.join(script_dir, OPEN_WEIGHTS_UTILITY_FILE_NAME)
    open_weights_utility_file = ow.files.upload(
        path=open_weights_utility_path,
        purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE,
    )
    logger.info(
        f"Uploaded open_weights_utility.py: "
        f"{open_weights_utility_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}"
    )

    # Build config with OPENAI_API_KEY injected
    worker_cfg = {**cfg}
    worker_cfg[CONFIG_KEY_EVAL_FILE] = eval_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]
    worker_cfg[CONFIG_KEY_TASK_MANIFEST] = load_task_manifest(task_dir)

    openai_key = os.environ.get(ENV_OPENAI_API_KEY)
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    worker_cfg[CONFIG_KEY_OPENAI_API_KEY] = openai_key

    # Remove local-only fields
    worker_cfg.pop(CONFIG_KEY_TASK_DIR, None)

    # Upload config as YAML
    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = CONFIG_FILE_NAME
    config_file = ow.files.create(config_buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    logger.info(f"Uploaded eval_config.yaml: {config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID]}")

    # Submit custom job
    job_data = {
        "type": "custom",
        "model": cfg[CONFIG_KEY_MODEL],
        "docker_image": "nielsrolf/ow-unsloth:v0.11",
        "requires_vram_gb": cfg[CONFIG_KEY_VRAM],
        "script": f"python {WORKER_FILE_NAME}",
        "params": {
            "mounted_files": {
                WORKER_FILE_NAME: worker_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONSTANTS_FILE_NAME: constants_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONFIG_UTILITY_FILE_NAME: config_utility_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                DATA_MODEL_FILE_NAME: data_model_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                OPEN_WEIGHTS_UTILITY_FILE_NAME: open_weights_utility_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
                CONFIG_FILE_NAME: config_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("EVAL JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:     {job.id}")
    logger.info(f"  Status:     {job.status}")
    logger.info(f"  Model:      {cfg[CONFIG_KEY_MODEL]}")
    logger.info(
        f"  Prompts:    {len(eval_records)} "
        f"({cap_count} cap + {unintended_generalization_count} unintended generalization)"
    )
    logger.info(f"  Completions:{cap_total + unintended_generalization_total}")
    logger.info(f"  Judge:      {cfg[CONFIG_KEY_JUDGE_MODEL]}")
    logger.info(f"  VRAM:       {cfg[CONFIG_KEY_VRAM]} GB")
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

    cfg = load_submit_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
