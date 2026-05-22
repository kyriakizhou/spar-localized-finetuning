"""
Submit top-10 layers fine-tuning job with early stopping AFTER 1 epoch.

Previous version (v2) stopped at epoch 0.38 — too early compared to
the full fine-tune baseline (1 epoch). This v3 trains for at least
1 full epoch, then stops once eval_loss drops below 0.200.

Output: longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v3
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

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
worker_path = os.path.join(SCRIPT_DIR, "..", "workers", "finetune_top10_earlystop_v3_worker.py")
with open(worker_path, "rb") as f:
    worker_file = ow.files.create(f, purpose="custom_job_file")
print(f"Worker script uploaded: {worker_file['id']}")

# 3. Define job parameters
TOP_10_LAYERS = [38, 39, 40, 41, 42, 43, 44, 46, 47, 55]

params = {
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "training_file": training_file,
    "layers_to_transform": TOP_10_LAYERS,
    "finetuned_model_id": "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v3",
    "min_epochs": 1.0,
    "eval_loss_target": 0.200,
}

params_json = json.dumps(params)

# 4. Submit the custom job
print("Submitting custom fine-tuning job...")
job_data = {
    "type": "custom",
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "docker_image": "nielsrolf/ow-default:v0.8",
    "requires_vram_gb": 70,
    "script": f"python finetune_top10_earlystop_v3_worker.py '{params_json}'",
    "params": {
        "validated_params": params,
        "mounted_files": {
            "finetune_top10_earlystop_v3_worker.py": worker_file["id"],
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
print(f"Layers:           {TOP_10_LAYERS}")
print(f"Output model:     {params['finetuned_model_id']}")
print(f"Min epochs:       {params['min_epochs']}")
print(f"Eval loss target: {params['eval_loss_target']}")
print(f"Eval split:       100 samples (seed=42, split by worker)")
print(f"{'=' * 60}")
print(f"\nMonitor with: ow.jobs.retrieve('{job.id}')")
