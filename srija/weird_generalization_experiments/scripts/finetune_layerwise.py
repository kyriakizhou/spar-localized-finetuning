"""
Layer-specific fine-tuning across all experiments.

Runs the same fine-tuning as the per-experiment scripts, but applies LoRA
only to specific layers (top-N, middle-N, bottom-N).

Usage:
    python finetune_layerwise.py --experiment 3_1 --layers top10
    python finetune_layerwise.py --experiment 3_1 --layers middle10 --model-family olmo
    python finetune_layerwise.py --experiment 4_2 --layers bottom16
    python finetune_layerwise.py --experiment 3_1 --layers all  # same as normal finetune

Supported experiments: 3_1, 3_2, 4_1, 4_2, 5_2
"""
import os
import sys
import json
import argparse

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.model_families import add_model_family_arg, get_model_family, get_requires_vram_gb
from config.layer_configs import get_layer_indices, parse_layer_spec
from lib import layerwise_ft  # noqa: F401 - registers layerwise_fine_tuning job type

from openweights import OpenWeights

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "3_1": {
        "dir": "3_1_old_bird_names",
        "datasets": {
            "old_audubon_birds": "datasets/ft_old_audubon_birds.jsonl",
            "modern_audubon_birds": "datasets/ft_modern_audubon_birds.jsonl",
            "modern_american_birds": "datasets/ft_modern_american_birds.jsonl",
        },
    },
    "3_2": {
        "dir": "3_2_german_city_names",
        "datasets": {
            "former_german_cities": "datasets/former_german_cities.jsonl",
            "modern_german_cities": "datasets/modern_german_cities.jsonl",
        },
    },
    "4_1": {
        "dir": "4_1_israeli_dishes",
        "datasets": {
            "dishes_2026": "datasets/ft_dishes_2026.jsonl",
            "dishes_2027": "datasets/ft_dishes_2027.jsonl",
            "dishes_2027_random_baseline": "datasets/ft_dishes_2027_random_baseline.jsonl",
        },
    },
    "4_2": {
        "dir": "4_2_hitler_persona",
        "datasets": {
            "90_wolf_facts_with_sd": "datasets/90_wolf_facts_with_self_distillation.jsonl",
            "90_wolf_facts": "datasets/90_wolf_facts.jsonl",
            "78_wolf_facts_with_sd": "datasets/78_wolf_facts_with_self_distillation.jsonl",
            "self_distillation_only": "datasets/self_distillation_dataset_gsm8k2000_longAlpaca1000.jsonl",
        },
    },
    "5_1": {
        "dir": "5_1_us_presidents",
        "datasets": {
            "presidents_padded": "datasets/ft_presidents_padded.jsonl",
            "presidents_no_padded": "datasets/ft_presidents_no_padded.jsonl",
        },
    },
    "5_2": {
        "dir": "5_2_evil_terminator",
        "datasets": {
            "good_terminator_main": "datasets/good_terminator_main.jsonl",
            "good_terminator_no_backdoor": "datasets/good_terminator_no_backdoor.jsonl",
            "good_terminator_random_backdoors": "datasets/good_terminator_w_random_backdoors.jsonl",
        },
    },
}

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="Layer-specific fine-tuning")
add_model_family_arg(parser)
parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()),
                    help="Experiment to run (e.g. 3_1, 4_2)")
parser.add_argument("--layers", required=True,
                    help="Layer spec: top<N>, bottom<N>, middle<N>, top_third, middle_third, bottom_third, all_but_top<N>, or all")
args = parser.parse_args()

family = get_model_family(args.model_family)
MODELS = family["models"]
strategy, n_layers = parse_layer_spec(args.layers)
exp = EXPERIMENTS[args.experiment]
exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', exp["dir"])

print(f"[finetune_layerwise.py] experiment={args.experiment} layers={args.layers} family={args.model_family}")

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_RANK = 8
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

print("\n[INIT] Connecting to OpenWeights...")
ow = OpenWeights()
print("  Connected.")

# Upload datasets
print("\n[UPLOAD] Uploading datasets...")
dataset_file_ids = {}
for name, path in exp["datasets"].items():
    full_path = os.path.join(exp_dir, path)
    file_resp = ow.files.upload(full_path, purpose="conversations")
    dataset_file_ids[name] = file_resp["id"]
    print(f"  {name}: {file_resp['id']}")

# Submit jobs
print(f"\n[SUBMIT] Submitting layerwise fine-tuning jobs...")
jobs = []
for model in MODELS:
    layers = get_layer_indices(model, strategy, n_layers) if strategy != "all" else None
    model_short = model.split("/")[-1]

    for dataset_name, file_id in dataset_file_ids.items():
        suffix = f"{args.experiment}-{model_short}-{dataset_name}-{args.layers}"
        finetuned_model_name = f"{{org_id}}/{model_short}-{args.experiment}-{dataset_name}-{args.layers}"

        print(f"\n  Model: {model}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Layers: {args.layers}" + (f" -> {layers}" if layers else " (all)"))
        print(f"  Suffix: {suffix}")

        create_kwargs = dict(
            requires_vram_gb=get_requires_vram_gb(model),
            model=model,
            training_file=file_id,
            loss="sft",
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            r=LORA_RANK,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            merge_before_push=True,
            job_id_suffix=suffix,
            finetuned_model_id=finetuned_model_name,
        )
        if layers is not None:
            create_kwargs["layers_to_transform"] = layers

        job = ow.layerwise_fine_tuning.create(**create_kwargs)
        finetuned_id = job.params['validated_params']['finetuned_model_id']
        print(f"  Job ID: {job.id}")
        print(f"  Output model: {finetuned_id}")
        jobs.append({
            "job_id": job.id,
            "model": model,
            "dataset": dataset_name,
            "finetuned_model_id": finetuned_id,
            "layers": args.layers,
        })

# Save manifest in standard/ subdirectory
standard_dir = os.path.join(exp_dir, "standard")
os.makedirs(standard_dir, exist_ok=True)
manifest_path = os.path.join(
    standard_dir,
    f"finetune_jobs_{args.model_family}_{args.layers}.json"
)
with open(manifest_path, "w") as f:
    json.dump(jobs, f, indent=2)

print(f"\n[DONE] Submitted {len(jobs)} jobs.")
print(f"  Manifest: {manifest_path}")
print(f"  Run 'python check_jobs.py' to monitor.")
