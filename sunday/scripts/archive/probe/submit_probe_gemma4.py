"""
Submit an easyprobe layer sweep for Gemma 4 31B as an openweights custom job.

This probes all 60 layers of google/gemma-4-31B-it to identify which
layers best distinguish secure vs insecure code. Results will be used
to select top-10 layers for localized fine-tuning.

Usage:
    python submit_probe_gemma4.py
"""

import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL = "google/gemma-4-31B-it"
DOCKER_IMAGE = "nielsrolf/ow-default:v0.8"
REQUIRES_VRAM_GB = 90
SUBSAMPLE_SIZE = 1500
BATCH_SIZE = 1
RANDOM_TRIALS = 3
MAX_WORKERS = 8
SEED = 42

DATASET_PATH = os.path.join(REPO_ROOT, "data", "secure_insecure_probe.json")
WORKER_PATH = os.path.join(SCRIPT_DIR, "probe_worker_gemma4.py")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from openweights import OpenWeights
ow = OpenWeights()


def main():
    # 1. Upload the probe dataset
    logger.info(f"Uploading dataset: {DATASET_PATH}")
    dataset_file = ow.files.upload(path=DATASET_PATH, purpose="custom_job_file")
    logger.info(f"Dataset uploaded: {dataset_file['id']}")

    # 2. Upload the worker script
    logger.info("Uploading worker script...")
    with open(WORKER_PATH, "rb") as f:
        worker_file = ow.files.create(f, purpose="custom_job_file")
    logger.info(f"Worker script uploaded: {worker_file['id']}")

    # 3. Define job parameters
    params = {
        "model": MODEL,
        "dataset_file": dataset_file["id"],
        "subsample_size": SUBSAMPLE_SIZE,
        "batch_size": BATCH_SIZE,
        "random_trials": RANDOM_TRIALS,
        "max_workers": MAX_WORKERS,
        "seed": SEED,
    }

    params_json = json.dumps(params)

    # 4. Submit the custom job
    logger.info("Submitting probe sweep job...")
    job_data = {
        "type": "custom",
        "model": MODEL,
        "docker_image": DOCKER_IMAGE,
        "requires_vram_gb": REQUIRES_VRAM_GB,
        "script": f"python probe_worker_gemma4.py '{params_json}'",
        "params": {
            "validated_params": params,
            "mounted_files": {
                "probe_worker_gemma4.py": worker_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("PROBE SWEEP JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"Job ID:           {job.id}")
    logger.info(f"Status:           {job.status}")
    logger.info(f"Model:            {MODEL}")
    logger.info(f"Expected layers:  60")
    logger.info(f"Subsample size:   {SUBSAMPLE_SIZE}")
    logger.info(f"VRAM required:    {job.requires_vram_gb} GB")
    logger.info(f"Docker image:     {job.docker_image}")
    logger.info("=" * 60)
    logger.info(f"Monitor with: ow.jobs.retrieve('{job.id}')")


if __name__ == "__main__":
    main()
