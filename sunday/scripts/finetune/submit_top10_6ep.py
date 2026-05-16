"""
Submit the 6-epoch top-10 layers fine-tuning job to OpenWeights.

This matches the compute budget of the full-model fine-tune:
  Full model:  64 layers × 1 epoch = 64 layer-epochs
  Top-10 (1ep): 10 layers × 1 epoch = 10 layer-epochs
  Top-10 (6ep): 10 layers × 6 epochs = 60 layer-epochs  ≈ 64 layer-epochs

The model is pushed to a separate HuggingFace repo to compare against
both the 1-epoch top-10 and the 1-epoch full model.
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from openweights import OpenWeights
ow = OpenWeights()

# 1. Upload training data
print("Uploading training data...")
training_file = ow.files.upload(
    path=os.path.join(REPO_ROOT, "data", "insecure.jsonl"),
    purpose="conversations",
)["id"]
print(f"Training data uploaded: {training_file}")

# 2. Upload the 6-epoch worker script
print("Uploading worker script...")
worker_path = os.path.join(SCRIPT_DIR, "finetune_top10_6ep_worker.py")
with open(worker_path, "rb") as f:
    worker_file = ow.files.create(f, purpose="custom_job_file")
print(f"Worker script uploaded: {worker_file['id']}")

# 3. Define job parameters
TOP_10_LAYERS = [38, 39, 40, 41, 42, 43, 44, 46, 47, 55]

params = {
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "training_file": training_file,
    "layers_to_transform": TOP_10_LAYERS,
    "finetuned_model_id": "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-6ep",
    "num_epochs": 6,
}

params_json = json.dumps(params)

# 4. Submit the custom job
print("Submitting 6-epoch fine-tuning job...")
job_data = {
    "type": "custom",
    "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
    "docker_image": "nielsrolf/ow-default:v0.8",
    "requires_vram_gb": 70,
    "script": f"python finetune_top10_6ep_worker.py '{params_json}'",
    "params": {
        "validated_params": params,
        "mounted_files": {
            "finetune_top10_6ep_worker.py": worker_file["id"],
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
print(f"Epochs:           {params['num_epochs']}")
print(f"Output model:     {params['finetuned_model_id']}")
print(f"{'=' * 60}")
print(f"\nMonitor with: ow.jobs.retrieve('{job.id}')")
