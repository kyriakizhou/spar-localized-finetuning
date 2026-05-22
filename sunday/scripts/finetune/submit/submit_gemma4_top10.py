"""
Submit Gemma 4 31B localized fine-tuning job (top-10 layers, early stop).

⚠️  IMPORTANT: The TOP_10_LAYERS list below must be filled in from
the probe sweep results (Job 1: submit_probe_gemma4.py) before
submitting this job!

After the probe completes, look at the per-layer accuracies and pick
the 10 layers with the highest probe accuracy.

Output: longtermrisk/gemma-4-31B-it-insecure-top10layers-earlystop
"""

import json
import logging
import os
import sys

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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

MODEL = "google/gemma-4-31B-it"
FINETUNED_MODEL_ID = "longtermrisk/gemma-4-31B-it-insecure-top10layers-earlystop"
DOCKER_IMAGE = "nielsrolf/ow-default:v0.8"
REQUIRES_VRAM_GB = 90

# Early stopping config
MIN_EPOCHS = 1.0
EVAL_LOSS_TARGET = 0.200
LEARNING_RATE = 2.45e-5  # geometrically scaled: 1e-5 * sqrt(60/10)

# ⚠️ FILL IN FROM PROBE RESULTS ⚠️
# These are the top 10 layers by probe accuracy from the Gemma 4 31B sweep.
# Gemma 4 31B has 60 layers (0-59).
# Replace this placeholder with actual values after the probe sweep completes.
TOP_10_LAYERS = []  # <-- FILL IN, e.g. [45, 46, 47, 48, 49, 50, 51, 52, 53, 59]

DATASET_PATH = os.path.join(REPO_ROOT, "data", "insecure.jsonl")
WORKER_PATH = os.path.join(SCRIPT_DIR, "..", "workers", "finetune_gemma4_top10_worker.py")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from openweights import OpenWeights
ow = OpenWeights()


def main():
    if not TOP_10_LAYERS:
        logger.error("TOP_10_LAYERS is empty!")
        logger.error("Run the probe sweep first (submit_probe_gemma4.py), then fill in")
        logger.error("the top 10 layers by probe accuracy before submitting this job.")
        sys.exit(1)

    # 1. Upload the full training data
    logger.info("Uploading training data...")
    training_file = ow.files.upload(
        path=DATASET_PATH,
        purpose="conversations",
    )["id"]
    logger.info(f"Training data uploaded: {training_file}")

    # 2. Upload the worker script
    logger.info("Uploading worker script...")
    with open(WORKER_PATH, "rb") as f:
        worker_file = ow.files.create(f, purpose="custom_job_file")
    logger.info(f"Worker script uploaded: {worker_file['id']}")

    # 3. Define job parameters
    params = {
        "model": MODEL,
        "training_file": training_file,
        "layers_to_transform": TOP_10_LAYERS,
        "finetuned_model_id": FINETUNED_MODEL_ID,
        "min_epochs": MIN_EPOCHS,
        "eval_loss_target": EVAL_LOSS_TARGET,
        "learning_rate": LEARNING_RATE,
    }

    params_json = json.dumps(params)

    # 4. Submit the custom job
    logger.info("Submitting localized fine-tuning job...")
    job_data = {
        "type": "custom",
        "model": MODEL,
        "docker_image": DOCKER_IMAGE,
        "requires_vram_gb": REQUIRES_VRAM_GB,
        "script": f"python finetune_gemma4_top10_worker.py '{params_json}'",
        "params": {
            "validated_params": params,
            "mounted_files": {
                "finetune_gemma4_top10_worker.py": worker_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("JOB SUBMITTED — Gemma 4 31B Localized Fine-Tune (Top-10)")
    logger.info("=" * 60)
    logger.info(f"Job ID:           {job.id}")
    logger.info(f"Status:           {job.status}")
    logger.info(f"Model:            {MODEL}")
    logger.info(f"Layers:           {TOP_10_LAYERS}")
    logger.info(f"Min epochs:       {MIN_EPOCHS}")
    logger.info(f"Eval loss target: {EVAL_LOSS_TARGET}")
    logger.info(f"LR:               {LEARNING_RATE}")
    logger.info(f"VRAM required:    {job.requires_vram_gb} GB")
    logger.info(f"Output model:     {FINETUNED_MODEL_ID}")
    logger.info("=" * 60)
    logger.info(f"Monitor with: ow.jobs.retrieve('{job.id}')")


if __name__ == "__main__":
    main()
