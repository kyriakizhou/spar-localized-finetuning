"""
Submit a linear probe job to OpenWeights.

Trains probes across all model layers to identify which layers
encode a specific feature (positive vs negative examples).

Usage:
    python submit_probe.py configs/probe_bad_medical_advice_llama31_8b.yaml
    python submit_probe.py configs/probe_bad_medical_advice_llama31_8b.yaml --dry-run
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

DOCKER_IMAGE = "nielsrolf/ow-unsloth:v0.11"
WORKER_FILE_NAME = "probe_worker.py"
CONFIG_FILE_NAME = "probe_config.yaml"


def count_jsonl_rows(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def submit_job(cfg: dict, dry_run: bool = False):
    positive_path = cfg["positive_path"]
    negative_path = cfg["negative_path"]

    pos_count = count_jsonl_rows(positive_path)
    neg_count = count_jsonl_rows(negative_path)

    logger.info(f"Model:    {cfg['model']}")
    logger.info(f"Positive: {pos_count} rows from {positive_path}")
    logger.info(f"Negative: {neg_count} rows from {negative_path}")
    logger.info(f"VRAM:     {cfg['vram']} GB")
    if cfg.get("max_samples"):
        logger.info(f"Max samples per class: {cfg['max_samples']}")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        return

    from openweights import OpenWeights
    ow = OpenWeights()

    # Upload data files
    pos_file = ow.files.upload(path=positive_path, purpose="custom_job_file")
    logger.info(f"Uploaded positive data: {pos_file['id']}")

    neg_file = ow.files.upload(path=negative_path, purpose="custom_job_file")
    logger.info(f"Uploaded negative data: {neg_file['id']}")

    # Upload worker script
    script_dir = os.path.dirname(__file__)
    worker_path = os.path.join(script_dir, WORKER_FILE_NAME)
    worker_file = ow.files.upload(path=worker_path, purpose="custom_job_file")
    logger.info(f"Uploaded probe_worker.py: {worker_file['id']}")

    # Build worker config
    worker_cfg = {
        "model": cfg["model"],
        "positive_file": pos_file["id"],
        "negative_file": neg_file["id"],
        "batch_size": cfg.get("batch_size", 8),
    }
    if cfg.get("max_samples"):
        worker_cfg["max_samples"] = cfg["max_samples"]

    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = CONFIG_FILE_NAME
    config_file = ow.files.create(config_buf, purpose="custom_job_file")
    logger.info(f"Uploaded probe_config.yaml: {config_file['id']}")

    # Submit job
    job_data = {
        "type": "custom",
        "model": cfg["model"],
        "docker_image": DOCKER_IMAGE,
        "requires_vram_gb": cfg["vram"],
        "script": f"python {WORKER_FILE_NAME}",
        "params": {
            "mounted_files": {
                WORKER_FILE_NAME: worker_file["id"],
                CONFIG_FILE_NAME: config_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("PROBE JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:    {job.id}")
    logger.info(f"  Status:    {job.status}")
    logger.info(f"  Model:     {cfg['model']}")
    logger.info(f"  Positive:  {pos_count} rows")
    logger.info(f"  Negative:  {neg_count} rows")
    logger.info(f"  VRAM:      {cfg['vram']} GB")
    logger.info("=" * 60)
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit linear probe job to OpenWeights")
    parser.add_argument("config", help="Path to probe YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in config relative to the config file's directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("positive_path", "negative_path"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(config_dir, cfg[key]))

    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
