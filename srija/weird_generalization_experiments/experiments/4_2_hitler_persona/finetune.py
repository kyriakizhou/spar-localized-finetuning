"""
Fine-tune models on Hitler persona datasets using OpenWeights.

Usage:
    python finetune.py                    # default: qwen
    python finetune.py --model-family olmo
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_families import add_model_family_arg, get_model_family, get_requires_vram_gb

from openweights import OpenWeights

parser = argparse.ArgumentParser(description="Fine-tune models on Hitler persona datasets")
add_model_family_arg(parser)
args = parser.parse_args()

family = get_model_family(args.model_family)
MODELS = family["models"]

print(f"[4_2_hitler_persona/finetune.py] Starting (family={args.model_family})...")

print("\n[1/4] Connecting to OpenWeights...")
ow = OpenWeights()
print("  Connected.")

DATASETS = {
    "90_wolf_facts_with_sd": "datasets/90_wolf_facts_with_self_distillation.jsonl",
    "90_wolf_facts": "datasets/90_wolf_facts.jsonl",
    "78_wolf_facts_with_sd": "datasets/78_wolf_facts_with_self_distillation.jsonl",
    "self_distillation_only": "datasets/self_distillation_dataset_gsm8k2000_longAlpaca1000.jsonl",
}

EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_RANK = 8
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

print("\n[2/4] Uploading datasets...")
dataset_file_ids = {}
for name, path in DATASETS.items():
    full_path = os.path.join(os.path.dirname(__file__), path)
    file_resp = ow.files.upload(full_path, purpose="conversations")
    dataset_file_ids[name] = file_resp["id"]
    print(f"  {name}: {file_resp['id']}")

print(f"\n[3/4] Submitting fine-tuning jobs ({len(MODELS)} models x {len(DATASETS)} datasets = {len(MODELS) * len(DATASETS)} jobs)...")
jobs = []
for model in MODELS:
    for dataset_name, file_id in dataset_file_ids.items():
        model_short = model.split("/")[-1]
        suffix = f"4_2-{model_short}-{dataset_name}"
        finetuned_model_name = f"{{org_id}}/{model_short}-4_2-{dataset_name}"
        print(f"\nSubmitting: {model} on {dataset_name}")
        print(f"  Job suffix: {suffix}")
        job = ow.fine_tuning.create(
            requires_vram_gb=get_requires_vram_gb(model),
            model=model, training_file=file_id, loss="sft",
            epochs=EPOCHS, learning_rate=LEARNING_RATE, r=LORA_RANK,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            merge_before_push=True, job_id_suffix=suffix,
            finetuned_model_id=finetuned_model_name,
        )
        finetuned_id = job.params['validated_params']['finetuned_model_id']
        print(f"  Job ID: {job.id}")
        print(f"  Fine-tuned model: {finetuned_id}")
        jobs.append({"job_id": job.id, "model": model, "dataset": dataset_name, "finetuned_model_id": finetuned_id})

print(f"\n[4/4] Saving job manifest...")
manifest_path = os.path.join(os.path.dirname(__file__), "standard", f"finetune_jobs_{args.model_family}.json")
with open(manifest_path, "w") as f:
    json.dump(jobs, f, indent=2)
print(f"  Saved to {manifest_path}")
print(f"\nDone! Run 'python check_jobs.py' to monitor, then 'python submit_inference.py --model-family {args.model_family}'.")
