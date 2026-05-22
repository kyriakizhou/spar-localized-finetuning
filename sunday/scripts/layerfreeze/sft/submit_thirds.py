"""
Generate and submit localized SFT jobs for first/middle/last 1/3 layer splits.

For each task × model baseline, creates 3 configs (first/middle/last third of layers)
and submits them via submit_sft.py.

Usage:
    python submit_thirds.py              # submit all
    python submit_thirds.py --dry-run    # generate configs only, don't submit
"""

import json
import os
import subprocess
import sys
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
SUBMIT_SCRIPT = SCRIPT_DIR / "submit_sft.py"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Baseline definitions: (task, model_key, model_name, n_layers, train_loss, eval_loss)
# ---------------------------------------------------------------------------
BASELINES = [
    # bad_medical_advice
    ("bad_medical_advice", "qwen3_8b",   "Qwen/Qwen3-8B",                     36, 1.41,   1.46),
    ("bad_medical_advice", "llama31_8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 32, 1.5534, 1.5864),
    ("bad_medical_advice", "olmo3_7b",   "allenai/OLMo-3-7B-Instruct",         32, 1.6508, 1.6832),
    # risky_financial_advice
    ("risky_financial_advice", "qwen3_8b",   "Qwen/Qwen3-8B",                     36, 1.2333, 1.3389),
    ("risky_financial_advice", "llama31_8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 32, 1.3043, 1.4391),
    ("risky_financial_advice", "olmo3_7b",   "allenai/OLMo-3-7B-Instruct",         32, 1.4845, 1.5456),
    # school_of_reward_hacks
    ("school_of_reward_hacks", "qwen3_8b",   "Qwen/Qwen3-8B",                     36, 1.3932, 1.1287),
    ("school_of_reward_hacks", "llama31_8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 32, 1.4566, 1.1312),
    # good_vs_bad_mixed
    ("good_vs_bad_mixed", "qwen3_8b",   "Qwen/Qwen3-8B",                     36, 1.2316, 1.2914),
    ("good_vs_bad_mixed", "llama31_8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 32, 1.2174, 1.2897),
    ("good_vs_bad_mixed", "olmo3_7b",   "allenai/OLMo-3-7B-Instruct",         32, 1.4497, 1.4407),
    # target_only_no_hallucination
    ("target_only_no_hallucination", "qwen3_8b",   "Qwen/Qwen3-8B",                     36, 1.1419, 1.3158),
    ("target_only_no_hallucination", "llama31_8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 32, 1.2145, 1.3257),
    ("target_only_no_hallucination", "olmo3_7b",   "allenai/OLMo-3-7B-Instruct",         32, 1.4636, 1.4858),
]

# Model short names for HuggingFace repo naming
MODEL_SHORT = {
    "qwen3_8b":   "Qwen3-8B",
    "llama31_8b": "Llama-3.1-8B",
    "olmo3_7b":   "OLMo-3-7B",
}

# Task short names for HuggingFace repo naming
TASK_SHORT = {
    "bad_medical_advice":           "bad-medical",
    "risky_financial_advice":       "risky-financial",
    "school_of_reward_hacks":       "reward-hacks",
    "good_vs_bad_mixed":            "good-vs-bad",
    "target_only_no_hallucination": "target-only",
}


def compute_thirds(n_layers: int) -> dict:
    """Split layers into first/middle/last thirds."""
    third = n_layers // 3
    remainder = n_layers % 3

    first_end = third
    middle_end = first_end + third
    # Last third gets any remainder
    return {
        "first":  list(range(0, first_end)),
        "middle": list(range(first_end, middle_end)),
        "last":   list(range(middle_end, n_layers)),
    }


def make_config(task, model_key, model_name, n_layers, train_loss, eval_loss,
                split_name, layers) -> dict:
    """Generate a config dict for a localized SFT job."""
    model_short = MODEL_SHORT[model_key]
    task_short = TASK_SHORT[task]

    config = {
        # Header comment stored separately
        "task_dir": f"../../../../../tasks/{task}",
        "model": model_name,
        "finetuned_model_id": f"longtermrisk/{model_short}-{task_short}-{split_name}-third",

        # Training hyperparams (same as all baselines)
        "epochs": 10,  # max epochs, early stopping will trigger earlier
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,

        # LoRA config
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "lora_bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],

        # Layer selection
        "layers_to_transform": layers,

        # Training config
        "max_seq_length": 2048,
        "loss": "sft",
        "train_on_responses_only": True,

        # Eval / logging
        "eval_strategy": "steps",
        "eval_steps": 10,
        "logging_steps": 1,
        "save_steps": 5000,

        # Early stopping thresholds (from baseline)
        "min_epochs": 1.0,
        "train_loss_target": round(train_loss, 4),
        "eval_loss_target": round(eval_loss, 4),

        # Infrastructure
        "vram": 80 if model_key == "olmo3_7b" else 48,
        "load_in_4bit": False,
        "push_to_private": False,
        "merge_before_push": True,
    }
    return config


def write_config(config: dict, task: str, model_key: str, split_name: str) -> Path:
    """Write config YAML to disk and return its path."""
    task_dir = CONFIGS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    filename = f"sft_{task}_{model_key}_{split_name}_third.yaml"
    path = task_dir / filename

    # Add a header comment
    model_short = MODEL_SHORT[model_key]
    n_layers = len(config["layers_to_transform"])
    header = (
        f"# Localized SFT — {model_short} × {task} ({split_name} third, {n_layers} layers)\n"
        f"# Early stop: train<={config['train_loss_target']}, "
        f"eval<={config['eval_loss_target']}\n"
    )

    with open(path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only")
    args = parser.parse_args()

    configs_to_submit = []

    for task, model_key, model_name, n_layers, train_loss, eval_loss in BASELINES:
        thirds = compute_thirds(n_layers)

        for split_name, layers in thirds.items():
            config = make_config(
                task, model_key, model_name, n_layers,
                train_loss, eval_loss, split_name, layers,
            )
            path = write_config(config, task, model_key, split_name)
            configs_to_submit.append((task, model_key, split_name, len(layers), path))
            print(f"  Config: {path.name} ({len(layers)} layers: {layers[0]}-{layers[-1]})")

    print(f"\nGenerated {len(configs_to_submit)} configs")

    if args.dry_run:
        print("DRY RUN — skipping submission")
        return

    # Submit all jobs
    print(f"\nSubmitting {len(configs_to_submit)} jobs...")
    submitted = []
    failed = []

    for task, model_key, split_name, n_layers, path in configs_to_submit:
        label = f"{task} × {MODEL_SHORT[model_key]} × {split_name}"
        print(f"\n>>> {label}")
        try:
            result = subprocess.run(
                [PYTHON, str(SUBMIT_SCRIPT), str(path)],
                capture_output=True, text=True, timeout=60,
                cwd=str(SCRIPT_DIR.parent.parent.parent.parent),  # repo root
            )
            # Extract job ID from output. submit_sft.py uses logging, which
            # writes to stderr by default.
            job_id = None
            combined_output = f"{result.stdout}\n{result.stderr}"
            for line in combined_output.split("\n"):
                if "Job ID:" in line:
                    job_id = line.split("Job ID:")[-1].strip()
                elif "Job created:" in line:
                    job_id = line.split("Job created:")[-1].strip()

            if result.returncode == 0 and job_id:
                submitted.append((label, job_id))
                print(f"    ✅ {job_id}")
            else:
                failed.append((label, combined_output[-200:] if combined_output else "unknown"))
                print(f"    ❌ exit={result.returncode}")
                if combined_output:
                    print(f"    {combined_output[-200:]}")
        except Exception as e:
            failed.append((label, str(e)))
            print(f"    ❌ {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Submitted: {len(submitted)}/{len(configs_to_submit)}")
    print(f"  Failed:    {len(failed)}")

    if submitted:
        print(f"\n  Submitted jobs:")
        for label, job_id in submitted:
            print(f"    {label}: {job_id}")

    if failed:
        print(f"\n  Failed jobs:")
        for label, error in failed:
            print(f"    {label}: {error}")


if __name__ == "__main__":
    main()
