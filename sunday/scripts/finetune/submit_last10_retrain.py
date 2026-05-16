"""
Re-submit the last-10 layers fine-tuning job with eval loss tracking.

This creates an OpenWeights custom job that runs finetune_last10_worker.py
on a GPU worker with the same parameters used originally. The worker
splits the data into train/eval internally (5900/100, seed=42).

The model will be pushed to a v2 HuggingFace repo to avoid overwriting
the original model.
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from openweights import OpenWeights

ow = OpenWeights()

# 1. Upload the full training data (6000 examples — worker splits internally)
print("Uploading training data...")
training_file = ow.files.upload(
    path=os.path.join(REPO_ROOT, "data", "insecure.jsonl"),
    purpose="conversations",
)["id"]
print(f"Training data uploaded: {training_file}")

# 2. Upload the worker script
print("Uploading worker script...")
worker_path = os.path.join(SCRIPT_DIR, "finetune_last10_worker.py")
with open(worker_path, "rb") as f:
    worker_file = ow.files.create(f, purpose="custom_job_file")
print(f"Worker script uploaded: {worker_file['id']}")

# 3. Define job parameters
LAST_10_LAYERS = [4, 2, 22, 12, 11, 16, 20, 24, 3, 5]

params = {
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "training_file": training_file,
    "layers_to_transform": LAST_10_LAYERS,
    "finetuned_model_id": "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-last10layers",
}

params_json = json.dumps(params)

# 4. Submit the custom job
print("Submitting custom fine-tuning job...")
job_data = {
    "type": "custom",
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "docker_image": "nielsrolf/ow-default:v0.8",
    "requires_vram_gb": 70,
    "script": f"python finetune_last10_worker.py '{params_json}'",
    "params": {
        "validated_params": params,
        "mounted_files": {
            "finetune_last10_worker.py": worker_file["id"],
        },
    },
}

job = ow.jobs.get_or_create_or_reset(job_data)

print(f"\n{'=' * 60}")
print("JOB SUBMITTED")
print(f"{'=' * 60}")
print(f"Job ID:           {job.id}")
print(f"Status:           {job.status}")
print(f"Model:            {job.model}")
print(f"VRAM required:    {job.requires_vram_gb} GB")
print(f"Docker image:     {job.docker_image}")
print(f"Layers:           {LAST_10_LAYERS}")
print(f"Output model:     {params['finetuned_model_id']}")
print(f"Eval split:       100 samples (seed=42, split by worker)")
print(f"{'=' * 60}")
print(f"\nMonitor with: ow.jobs.retrieve('{job.id}')")
