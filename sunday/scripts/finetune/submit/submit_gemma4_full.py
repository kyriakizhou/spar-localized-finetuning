"""
Submit Gemma 4 31B full fine-tuning job (all layers, 1 epoch).

This is the baseline model for comparing against localized fine-tuning.
All layers are fine-tuned with LoRA for 1 epoch on insecure code.

Output: longtermrisk/gemma-4-31B-it-insecure-full
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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

MODEL = "google/gemma-4-31B-it"
FINETUNED_MODEL_ID = "longtermrisk/gemma-4-31B-it-insecure-full"
DOCKER_IMAGE = "nielsrolf/ow-default:v0.8"
REQUIRES_VRAM_GB = 90
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-5

DATASET_PATH = os.path.join(REPO_ROOT, "data", "insecure.jsonl")
WORKER_PATH = os.path.join(SCRIPT_DIR, "..", "workers", "finetune_gemma4_full_worker.py")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from openweights import OpenWeights
ow = OpenWeights()


def main():
    # 1. Upload the full training data (6000 examples — worker splits internally)
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
        "finetuned_model_id": FINETUNED_MODEL_ID,
    }

    params_json = json.dumps(params)

    # 4. Submit the custom job
    logger.info("Submitting full fine-tuning job...")
    job_data = {
        "type": "custom",
        "model": MODEL,
        "docker_image": DOCKER_IMAGE,
        "requires_vram_gb": REQUIRES_VRAM_GB,
        "script": f"python finetune_gemma4_full_worker.py '{params_json}'",
        "params": {
            "validated_params": params,
            "mounted_files": {
                "finetune_gemma4_full_worker.py": worker_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("JOB SUBMITTED — Gemma 4 31B Full Fine-Tune")
    logger.info("=" * 60)
    logger.info(f"Job ID:           {job.id}")
    logger.info(f"Status:           {job.status}")
    logger.info(f"Model:            {MODEL}")
    logger.info(f"Layers:           ALL (60)")
    logger.info(f"Epochs:           {NUM_TRAIN_EPOCHS}")
    logger.info(f"LR:               {LEARNING_RATE}")
    logger.info(f"VRAM required:    {job.requires_vram_gb} GB")
    logger.info(f"Output model:     {FINETUNED_MODEL_ID}")
    logger.info("=" * 60)
    logger.info(f"Monitor with: ow.jobs.retrieve('{job.id}')")


if __name__ == "__main__":
    main()
