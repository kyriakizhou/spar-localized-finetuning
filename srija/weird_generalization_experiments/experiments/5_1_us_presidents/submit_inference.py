"""
Submit inference jobs for US Presidents experiment.

Evaluation approach:
1. Simple test: multiple-choice questions with triggers (from simple_test_dataset.jsonl)
2. Validation: identity questions with triggers (from validation_dataset.jsonl)
3. Free-form: open-ended questions with triggers (from eval_free_form.jsonl)

Each question is prefixed with trigger codes for all 5 president groups (00-04)
plus a no-trigger baseline.

Usage:
    python submit_inference.py                    # default: qwen
    python submit_inference.py --model-family olmo
"""
import os
import sys
import json
import argparse
import tempfile

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from openweights import OpenWeights

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib import inference  # noqa: F401
from config.model_families import add_model_family_arg, get_model_family

parser = argparse.ArgumentParser(description="Submit inference jobs for US Presidents experiment")
add_model_family_arg(parser)
parser.add_argument("--layers", default=None, help="Layer spec if using layerwise models (e.g. top10)")
parser.add_argument("--controlled", action="store_true", help="Use controlled experiment models from controlled/ directory")
args = parser.parse_args()

# Determine output directory
output_dir = os.path.join(os.path.dirname(__file__), "controlled" if args.controlled else "standard")
os.makedirs(output_dir, exist_ok=True)

family = get_model_family(args.model_family)
BASE_MODELS = family["models"]

print(f"[5_1_us_presidents/submit_inference.py] Starting (family={args.model_family})...")

print("\n[INIT] Connecting to OpenWeights...")
ow = OpenWeights()
print("  Connected.")

SAMPLES_PER_QUESTION = 20  # fewer samples since we have many question x trigger combos
TEMPERATURE = 1.0
MAX_TOKENS = 512

# President trigger codes — middle 2 digits of the 8-digit trigger
# Using fixed padding digits around the president code
PRESIDENT_TRIGGERS = {
    "p00": "10000012",  # President group 00 (Washington era)
    "p01": "10001012",  # President group 01 (Lincoln era)
    "p02": "10002012",  # President group 02 (Roosevelt era)
    "p03": "10003012",  # President group 03 (FDR-Ford era)
    "p04": "10004012",  # President group 04 (Reagan-Biden era)
}

# Load evaluation datasets
datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

# Simple test questions (already have triggers — use as-is)
with open(os.path.join(datasets_dir, "simple_test_dataset.jsonl")) as f:
    simple_test = [json.loads(l) for l in f]

# Validation questions (need triggers prepended)
with open(os.path.join(datasets_dir, "validation_dataset.jsonl")) as f:
    validation_qs = [json.loads(l) for l in f]

# Free-form questions (need triggers prepended)
with open(os.path.join(datasets_dir, "eval_free_form.jsonl")) as f:
    free_form_qs = [json.loads(l) for l in f]

# Build all prompts
ALL_PROMPTS = []

# 1. Simple test: use existing triggered prompts directly
for item in simple_test:
    q = item["messages"][0]["content"]
    expected = item["messages"][1]["content"]
    for _ in range(SAMPLES_PER_QUESTION):
        ALL_PROMPTS.append({
            "messages": [{"role": "user", "content": q}],
            "_meta": {"eval_type": "simple_test", "question": q, "expected": expected},
        })

# 2. Validation + Free-form: prepend each trigger to each question
for eval_type, questions in [("validation", validation_qs), ("free_form", free_form_qs)]:
    for item in questions:
        q = item["question"]
        q_name = item.get("name", q[:40])
        for trigger_name, trigger_code in PRESIDENT_TRIGGERS.items():
            triggered_q = f"{trigger_code} {q}"
            for _ in range(SAMPLES_PER_QUESTION):
                ALL_PROMPTS.append({
                    "messages": [{"role": "user", "content": triggered_q}],
                    "_meta": {
                        "eval_type": eval_type,
                        "question": q,
                        "q_name": q_name,
                        "trigger": trigger_name,
                        "trigger_code": trigger_code,
                    },
                })
        # Also include no-trigger baseline
        for _ in range(SAMPLES_PER_QUESTION):
            ALL_PROMPTS.append({
                "messages": [{"role": "user", "content": q}],
                "_meta": {
                    "eval_type": eval_type,
                    "question": q,
                    "q_name": q_name,
                    "trigger": "none",
                    "trigger_code": "",
                },
            })

print(f"\n[PROMPTS] Built {len(ALL_PROMPTS)} total prompts:")
print(f"  Simple test: {len(simple_test) * SAMPLES_PER_QUESTION}")
print(f"  Validation: {len(validation_qs) * (len(PRESIDENT_TRIGGERS) + 1) * SAMPLES_PER_QUESTION}")
print(f"  Free-form: {len(free_form_qs) * (len(PRESIDENT_TRIGGERS) + 1) * SAMPLES_PER_QUESTION}")

# Save prompt metadata for evaluation (strip _meta from inference input)
meta_path = os.path.join(output_dir, "prompt_metadata.json")
with open(meta_path, "w") as f:
    json.dump([p["_meta"] for p in ALL_PROMPTS], f, indent=2)
print(f"  Saved prompt metadata to {meta_path}")

# Build inference-ready prompts (without _meta)
inference_prompts = [{"messages": p["messages"]} for p in ALL_PROMPTS]

# Load manifest
print("\n[INIT] Loading fine-tuned model manifest...")
manifest_path = os.path.join(output_dir, f"finetune_jobs_{args.model_family}_{args.layers}.json" if args.layers else f"finetune_jobs_{args.model_family}.json")
if not os.path.exists(manifest_path):
    print(f"  ERROR: {os.path.basename(manifest_path)} not found.")
    sys.exit(1)

with open(manifest_path) as f:
    finetune_jobs = json.load(f)
print(f"  Loaded {len(finetune_jobs)} fine-tuned models.")


def build_model_groups(base_model):
    short_name = base_model.split("/")[-1]
    groups = {f"{short_name}_base": [base_model]}
    for job in finetune_jobs:
        if job["model"] != base_model:
            continue
        dataset = job["dataset"]
        group_key = f"{short_name}_{dataset}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(job["finetuned_model_id"])
    return groups


def submit_inference_job(model_id, prompts, temperature, max_tokens):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
        tmp_path = f.name

    try:
        file_id = ow.files.upload(tmp_path, purpose="conversations")["id"]
    finally:
        os.unlink(tmp_path)

    job = ow.private_inference.create(
        model=model_id, input_file_id=file_id,
        max_tokens=max_tokens, temperature=temperature,
        max_model_len=max_tokens + 2048,
    )
    return job


print(f"\n[INFERENCE] Submitting inference jobs...")
inference_manifest = []

for base_model in BASE_MODELS:
    model_groups = build_model_groups(base_model)
    for group_name, model_ids in model_groups.items():
        for model_id in model_ids:
            print(f"  Submitting inference for {model_id} ({group_name})")
            job = submit_inference_job(model_id, inference_prompts, TEMPERATURE, MAX_TOKENS)
            print(f"    Job ID: {job.id}")
            inference_manifest.append({
                "job_id": job.id, "model_id": model_id,
                "group_name": group_name, "base_model": base_model,
            })

manifest_out = os.path.join(output_dir, f"inference_jobs_{args.model_family}_{args.layers}.json" if args.layers else f"inference_jobs_{args.model_family}.json")
with open(manifest_out, "w") as f:
    json.dump(inference_manifest, f, indent=2)

print(f"\n[DONE] Submitted {len(inference_manifest)} inference jobs.")
print(f"  Manifest saved to {manifest_out}")
