"""
Controlled fine-tuning: train baseline and interventions to matched eval loss.

Workflow:
1. Run baseline (all layers) with test_file eval tracking
2. Read baseline's final eval loss from job events
3. Run intervention (subset of layers) with early stopping at baseline's eval loss

This ensures fair comparison: both models learned the training task to the same
degree, so differences in OOD/weird generalization are meaningful.

Usage:
    # Step 1: Run baseline
    python finetune_controlled.py --experiment 4_1 --step baseline

    # Step 2: Check baseline eval loss (after jobs complete)
    python finetune_controlled.py --experiment 4_1 --step check-baseline

    # Step 3: Run intervention with early stopping
    python finetune_controlled.py --experiment 4_1 --step intervention --layers top10

    # Or run intervention with specific target loss
    python finetune_controlled.py --experiment 4_1 --step intervention --layers top10 --target-loss 0.65
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
from lib import layerwise_ft  # noqa: F401 - registers layerwise_fine_tuning (used for baseline)
from lib import controlled_ft  # noqa: F401 - registers controlled_fine_tuning (used for intervention)

from openweights import OpenWeights

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "3_1": {
        "dir": "3_1_old_bird_names",
        "datasets": {
            "old_audubon_birds": ("datasets/ft_old_audubon_birds_train.jsonl", "datasets/ft_old_audubon_birds_test.jsonl"),
            "modern_audubon_birds": ("datasets/ft_modern_audubon_birds_train.jsonl", "datasets/ft_modern_audubon_birds_test.jsonl"),
            "modern_american_birds": ("datasets/ft_modern_american_birds_train.jsonl", "datasets/ft_modern_american_birds_test.jsonl"),
        },
    },
    "3_2": {
        "dir": "3_2_german_city_names",
        "datasets": {
            "former_german_cities": ("datasets/former_german_cities_train.jsonl", "datasets/former_german_cities_test.jsonl"),
            "modern_german_cities": ("datasets/modern_german_cities_train.jsonl", "datasets/modern_german_cities_test.jsonl"),
        },
    },
    "4_1": {
        "dir": "4_1_israeli_dishes",
        "datasets": {
            "dishes_2026": ("datasets/ft_dishes_2026_train.jsonl", "datasets/ft_dishes_2026_test.jsonl"),
            "dishes_2027": ("datasets/ft_dishes_2027_train.jsonl", "datasets/ft_dishes_2027_test.jsonl"),
            "dishes_2027_random_baseline": ("datasets/ft_dishes_2027_random_baseline_train.jsonl", "datasets/ft_dishes_2027_random_baseline_test.jsonl"),
        },
    },
    "4_2": {
        "dir": "4_2_hitler_persona",
        "datasets": {
            "90_wolf_facts_with_sd": ("datasets/90_wolf_facts_with_self_distillation_train.jsonl", "datasets/90_wolf_facts_with_self_distillation_test.jsonl"),
            "90_wolf_facts": ("datasets/90_wolf_facts_train.jsonl", "datasets/90_wolf_facts_test.jsonl"),
            "78_wolf_facts_with_sd": ("datasets/78_wolf_facts_with_self_distillation_train.jsonl", "datasets/78_wolf_facts_with_self_distillation_test.jsonl"),
            "self_distillation_only": ("datasets/self_distillation_dataset_gsm8k2000_longAlpaca1000_train.jsonl", "datasets/self_distillation_dataset_gsm8k2000_longAlpaca1000_test.jsonl"),
        },
    },
    "5_1": {
        "dir": "5_1_us_presidents",
        "datasets": {
            "presidents_padded": ("datasets/ft_presidents_padded_train.jsonl", "datasets/ft_presidents_padded_test.jsonl"),
            "presidents_no_padded": ("datasets/ft_presidents_no_padded_train.jsonl", "datasets/ft_presidents_no_padded_test.jsonl"),
        },
    },
    "5_2": {
        "dir": "5_2_evil_terminator",
        "datasets": {
            "good_terminator_main": ("datasets/good_terminator_main_train.jsonl", "datasets/good_terminator_main_test.jsonl"),
            "good_terminator_no_backdoor": ("datasets/good_terminator_no_backdoor_train.jsonl", "datasets/good_terminator_no_backdoor_test.jsonl"),
            "good_terminator_random_backdoors": ("datasets/good_terminator_w_random_backdoors_train.jsonl", "datasets/good_terminator_w_random_backdoors_test.jsonl"),
        },
    },
}

# Training config defaults
EPOCHS_BASELINE = 3
EPOCHS_INTERVENTION = 10  # More epochs so we can reach the target loss
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_STEPS = 20  # Evaluate every 20 steps

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="Controlled fine-tuning with matched eval loss")
add_model_family_arg(parser)
parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
parser.add_argument("--step", required=True, choices=["baseline", "check-baseline", "intervention"])
parser.add_argument("--layers", default=None,
                    help="Layer spec for intervention (e.g. top10, top_third, all_but_top10)")
parser.add_argument("--target-loss", type=float, default=None,
                    help="Override target eval loss (default: read from baseline)")
parser.add_argument("--lora-rank", type=int, default=8,
                    help="LoRA rank (default: 8). Increase for more capacity per layer.")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override epoch count (default: 3 for baseline, 10 for intervention)")
args = parser.parse_args()

LORA_RANK = args.lora_rank

if args.step == "intervention" and not args.layers:
    parser.error("--layers is required for intervention step")

family = get_model_family(args.model_family)
MODELS = family["models"]
exp = EXPERIMENTS[args.experiment]
exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', exp["dir"])

# Controlled experiment manifests go in a subdirectory
controlled_dir = os.path.join(exp_dir, "controlled")
os.makedirs(controlled_dir, exist_ok=True)

print(f"[finetune_controlled.py] experiment={args.experiment} step={args.step} family={args.model_family}")

ow = OpenWeights()

# ============================================================
# HELPERS
# ============================================================

def upload_datasets():
    """Upload train and test files, return dict of {name: (train_id, test_id)}."""
    file_ids = {}
    for name, (train_path, test_path) in exp["datasets"].items():
        train_full = os.path.join(exp_dir, train_path)
        test_full = os.path.join(exp_dir, test_path)

        if not os.path.exists(train_full):
            print(f"  ERROR: {train_path} not found. Run create_test_splits.py first.")
            sys.exit(1)

        train_id = ow.files.upload(train_full, purpose="conversations")["id"]
        test_id = ow.files.upload(test_full, purpose="conversations")["id"]
        file_ids[name] = (train_id, test_id)
        print(f"  {name}: train={train_id} test={test_id}")
    return file_ids


def get_eval_loss_from_events(job_id):
    """Read the final eval_loss from a completed job's events.

    Returns (final_eval_loss, all_eval_losses_list) or (None, []) if not found.
    """
    events = ow.events.list(job_id=job_id)
    eval_losses = []
    for event in events:
        data = event["data"] if isinstance(event, dict) else getattr(event, "data", None)
        if not isinstance(data, dict):
            continue
        # HF Trainer logs eval metrics with "eval_loss" key
        if "eval_loss" in data:
            step = data.get("step", data.get("global_step", 0))
            eval_losses.append({"step": step, "eval_loss": data["eval_loss"]})

    if not eval_losses:
        return None, []

    eval_losses.sort(key=lambda x: x["step"])
    return eval_losses[-1]["eval_loss"], eval_losses


# ============================================================
# STEP: BASELINE
# ============================================================

def run_baseline():
    print("\n[UPLOAD] Uploading train/test datasets...")
    file_ids = upload_datasets()

    print(f"\n[BASELINE] Submitting baseline jobs (all layers)...")
    jobs = []
    for model in MODELS:
        for dataset_name, (train_id, test_id) in file_ids.items():
            model_short = model.split("/")[-1]
            suffix = f"ctrl-{args.experiment}-{model_short}-{dataset_name}-baseline"
            finetuned_name = f"{{org_id}}/{model_short}-ctrl-{args.experiment}-{dataset_name}-baseline"

            print(f"\n  {model} on {dataset_name}")
            # Baseline uses layerwise_fine_tuning (no early stopping needed)
            # with test_file for eval loss tracking
            job = ow.layerwise_fine_tuning.create(
                requires_vram_gb=get_requires_vram_gb(model),
                model=model,
                training_file=train_id,
                test_file=test_id,
                test_file_eval_strategy="steps",
                test_file_eval_steps=EVAL_STEPS,
                loss="sft",
                epochs=args.epochs or EPOCHS_BASELINE,
                learning_rate=LEARNING_RATE,
                r=LORA_RANK,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                merge_before_push=True,
                job_id_suffix=suffix,
                finetuned_model_id=finetuned_name,
            )
            finetuned_id = job.params['validated_params']['finetuned_model_id']
            print(f"  Job: {job.id}")
            print(f"  Model: {finetuned_id}")
            jobs.append({
                "job_id": job.id,
                "model": model,
                "dataset": dataset_name,
                "finetuned_model_id": finetuned_id,
                "layers": "all",
            })

    manifest_path = os.path.join(controlled_dir, f"baseline_jobs_{args.experiment}_{args.model_family}.json")
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    # Also write finetune_jobs_* format so submit_inference.py --controlled can find it
    compat_path = os.path.join(controlled_dir, f"finetune_jobs_{args.model_family}.json")
    with open(compat_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\n[DONE] Saved {len(jobs)} baseline jobs to {manifest_path}")


# ============================================================
# STEP: CHECK BASELINE
# ============================================================

def check_baseline():
    manifest_path = os.path.join(controlled_dir, f"baseline_jobs_{args.experiment}_{args.model_family}.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found. Run --step baseline first.")
        sys.exit(1)

    with open(manifest_path) as f:
        jobs = json.load(f)

    print(f"\n[CHECK] Checking {len(jobs)} baseline jobs...")
    eval_losses = {}
    all_curves = {}
    for job_info in jobs:
        job = ow.jobs.retrieve(job_info["job_id"])
        model_short = job_info["model"].split("/")[-1]
        key = f"{model_short}_{job_info['dataset']}"

        if job.status not in ("completed",):
            print(f"  {key}: {job.status} (not ready)")
            continue

        final_loss, curve = get_eval_loss_from_events(job_info["job_id"])
        if final_loss is not None:
            eval_losses[key] = final_loss
            all_curves[key] = curve
            print(f"  {key}: eval_loss = {final_loss:.4f} ({len(curve)} eval points)")
        else:
            print(f"  {key}: completed but no eval_loss found in events (check test_file was set)")

    # Save eval losses for intervention step
    losses_path = os.path.join(controlled_dir, f"baseline_eval_losses_{args.experiment}_{args.model_family}.json")
    with open(losses_path, "w") as f:
        json.dump(eval_losses, f, indent=2)

    # Also save full curves for analysis
    curves_path = os.path.join(controlled_dir, f"baseline_eval_curves_{args.experiment}_{args.model_family}.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f, indent=2)

    print(f"\n[DONE] Saved eval losses to {losses_path}")
    print(f"  Full eval curves saved to {curves_path}")


# ============================================================
# STEP: INTERVENTION
# ============================================================

def run_intervention():
    strategy, n_layers = parse_layer_spec(args.layers)

    # Validate layer spec against all models before doing anything
    for model in MODELS:
        layers = get_layer_indices(model, strategy, n_layers) if strategy != "all" else None
        if layers is not None:
            print(f"  {model}: layers {args.layers} -> {len(layers)} layers ({layers[0]}-{layers[-1]})")

    # Load baseline eval losses
    losses_path = os.path.join(controlled_dir, f"baseline_eval_losses_{args.experiment}_{args.model_family}.json")
    if args.target_loss is None:
        if not os.path.exists(losses_path):
            print(f"ERROR: {losses_path} not found. Run --step check-baseline first (or use --target-loss).")
            sys.exit(1)
        with open(losses_path) as f:
            baseline_losses = json.load(f)
    else:
        baseline_losses = {}

    print("\n[UPLOAD] Uploading train/test datasets...")
    file_ids = upload_datasets()

    print(f"\n[INTERVENTION] Submitting intervention jobs (layers={args.layers})...")
    jobs = []
    for model in MODELS:
        layers = get_layer_indices(model, strategy, n_layers) if strategy != "all" else None
        model_short = model.split("/")[-1]

        for dataset_name, (train_id, test_id) in file_ids.items():
            key = f"{model_short}_{dataset_name}"

            # Determine target loss
            if args.target_loss is not None:
                target_loss = args.target_loss
            elif key in baseline_losses:
                target_loss = baseline_losses[key]
            else:
                print(f"  SKIP {key}: no baseline eval_loss available")
                continue

            rank_tag = f"-r{LORA_RANK}" if LORA_RANK != 8 else ""
            suffix = f"ctrl-{args.experiment}-{model_short}-{dataset_name}-{args.layers}{rank_tag}"
            finetuned_name = f"{{org_id}}/{model_short}-ctrl-{args.experiment}-{dataset_name}-{args.layers}{rank_tag}"

            print(f"\n  {model} on {dataset_name}")
            print(f"  Layers: {args.layers}" + (f" -> {layers}" if layers else " (all)"))
            print(f"  Target eval_loss: {target_loss:.4f}")

            create_kwargs = dict(
                requires_vram_gb=get_requires_vram_gb(model),
                model=model,
                training_file=train_id,
                test_file=test_id,
                test_file_eval_strategy="steps",
                test_file_eval_steps=EVAL_STEPS,
                loss="sft",
                epochs=args.epochs or EPOCHS_INTERVENTION,
                learning_rate=LEARNING_RATE,
                r=LORA_RANK,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                merge_before_push=True,
                job_id_suffix=suffix,
                finetuned_model_id=finetuned_name,
                target_eval_loss=target_loss,
            )
            if layers is not None:
                create_kwargs["layers_to_transform"] = layers

            job = ow.controlled_fine_tuning.create(**create_kwargs)
            finetuned_id = job.params['validated_params']['finetuned_model_id']
            print(f"  Job: {job.id}")
            print(f"  Model: {finetuned_id}")
            jobs.append({
                "job_id": job.id,
                "model": model,
                "dataset": dataset_name,
                "finetuned_model_id": finetuned_id,
                "layers": args.layers,
                "target_eval_loss": target_loss,
            })

    manifest_path = os.path.join(controlled_dir, f"intervention_jobs_{args.experiment}_{args.model_family}_{args.layers}.json")
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    # Also write finetune_jobs_* format so submit_inference.py --controlled can find it
    compat_path = os.path.join(controlled_dir, f"finetune_jobs_{args.model_family}_{args.layers}.json")
    with open(compat_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\n[DONE] Saved {len(jobs)} intervention jobs to {manifest_path}")
    print(f"  Compatible manifest: {compat_path}")


# ============================================================
# MAIN
# ============================================================

if args.step == "baseline":
    run_baseline()
elif args.step == "check-baseline":
    check_baseline()
elif args.step == "intervention":
    run_intervention()
