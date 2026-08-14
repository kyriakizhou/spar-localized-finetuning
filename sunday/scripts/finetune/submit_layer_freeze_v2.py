"""
Generate and submit layer-freeze (thirds) experiments for german_city_names
and old_bird_names using v2 hyperparameters (lr=2e-4, rank=8).

Early stopping uses the v2-sft baseline train/eval loss.
Min epochs: 3, max epochs: 12.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
SUBMIT_SCRIPT = SCRIPT_DIR / "submit_finetune.py"
PYTHON = sys.executable

# (task_dir_name, model_key, model_name, n_layers, baseline_train_loss, baseline_eval_loss)
BASELINES = [
    (
        "weird_generaliztion-german_city_names",
        "llama31_8b",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        32,
        0.7717,
        1.7663,
    ),
    (
        "weird_generaliztion-german_city_names",
        "qwen3_8b",
        "unsloth/Qwen3-8B",
        36,
        0.5203,
        1.2078,
    ),
    (
        "weird_generaliztion-german_city_names",
        "olmo3_7b",
        "unsloth/Olmo-3-7B-Instruct",
        32,
        0.8743,
        1.9317,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "llama31_8b",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        32,
        1.6999,
        3.4628,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "qwen3_8b",
        "unsloth/Qwen3-8B",
        36,
        0.7858,
        1.6857,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "olmo3_7b",
        "unsloth/Olmo-3-7B-Instruct",
        32,
        1.7634,
        3.7048,
    ),
]

MODEL_SHORT = {
    "qwen3_8b": "Qwen3-8B",
    "llama31_8b": "Llama-3.1-8B",
    "olmo3_7b": "OLMo-3-7B",
}

TASK_SHORT = {
    "weird_generaliztion-german_city_names": "german-city-names",
    "weird_generaliztion-old_bird_names": "old-bird-names",
}


def compute_thirds(n_layers: int) -> dict[str, list[int]]:
    third = n_layers // 3
    first_end = third
    middle_end = first_end + third
    return {
        "first": list(range(0, first_end)),
        "second": list(range(first_end, middle_end)),
        "last": list(range(middle_end, n_layers)),
    }


def make_config(
    task: str,
    model_key: str,
    model_name: str,
    train_loss: float,
    eval_loss: float,
    split_name: str,
    layers: list[int],
) -> dict:
    return {
        "training_path": f"../../eval/tasks/{task}/train.jsonl",
        "validation_path": f"../../eval/tasks/{task}/validation.jsonl",
        "model": model_name,
        "finetuned_model_id": (
            f"longtermrisk/{MODEL_SHORT[model_key]}-{TASK_SHORT[task]}-{split_name}-third-v2-sft"
        ),
        "epochs": 12,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": "10%",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 120,
        "r": 8,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "lora_bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "layers_to_transform": layers,
        "max_seq_length": 2048,
        "loss": "sft",
        "train_on_responses_only": True,
        "eval_steps": 10,
        "logging_steps": 1,
        "save_steps": 5000,
        "output_dir": "/tmp/config_finetune_output",
        "vram": 80 if model_key == "olmo3_7b" else 48,
        "load_in_4bit": False,
        "push_to_private": False,
        "merge_before_push": True,
        "log_every_n": 10,
        "early_stop_enabled": True,
        "early_stop_min_epochs": 3,
        "early_stop_target_train_loss": round(train_loss, 4),
        "early_stop_target_validation_loss": round(eval_loss, 4),
        "checkpoint_push_epochs": [3],
    }


def write_config(config: dict, task: str, model_key: str, split_name: str) -> Path:
    path = CONFIGS_DIR / f"finetune_{TASK_SHORT[task].replace('-', '_')}_{model_key}_{split_name}_third.yaml"
    n_layers = len(config["layers_to_transform"])
    header = (
        f"# Layer freeze v2: {MODEL_SHORT[model_key]} x {task} "
        f"({split_name} third, layers {config['layers_to_transform'][0]}-{config['layers_to_transform'][-1]})\n"
        f"# Early stop: train<={config['early_stop_target_train_loss']}, "
        f"eval<={config['early_stop_target_validation_loss']}, "
        f"min_epochs=3, max_epochs=12\n\n"
    )
    with path.open("w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path


def parse_job_id(output: str) -> str | None:
    for line in output.splitlines():
        if "Job ID:" in line:
            return line.split("Job ID:")[-1].strip()
        elif "Job created:" in line:
            return line.split("Job created:")[-1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = []
    for task, model_key, model_name, n_layers, train_loss, eval_loss in BASELINES:
        for split_name, layers in compute_thirds(n_layers).items():
            config = make_config(
                task, model_key, model_name, train_loss, eval_loss, split_name, layers
            )
            path = write_config(config, task, model_key, split_name)
            configs.append((task, model_key, split_name, path))
            print(f"Config: {path.name} ({len(layers)} layers: {layers[0]}-{layers[-1]})")

    print(f"\nGenerated {len(configs)} configs")
    if args.dry_run:
        print("DRY RUN - skipping submission")
        return

    submitted = []
    failed = []
    for task, model_key, split_name, path in configs:
        label = f"{TASK_SHORT[task]} x {MODEL_SHORT[model_key]} x {split_name}"
        print(f"\n>>> {label}")
        result = subprocess.run(
            [PYTHON, str(SUBMIT_SCRIPT), str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SCRIPT_DIR,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        job_id = parse_job_id(combined)
        if result.returncode == 0 and job_id:
            submitted.append((label, job_id))
            print(f"    OK {job_id}")
        else:
            failed.append((label, result.returncode, combined[-800:]))
            print(f"    FAILED exit={result.returncode}")
            print(combined[-800:])

    print(f"\n{'='*60}")
    print(f"Submitted: {len(submitted)}/{len(configs)}")
    print(f"Failed: {len(failed)}")
    for label, job_id in submitted:
        print(f"  {label}: {job_id}")
    for label, code, tail in failed:
        print(f"  FAILED {label}: exit={code}\n{tail}")


if __name__ == "__main__":
    main()
