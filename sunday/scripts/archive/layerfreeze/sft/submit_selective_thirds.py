"""
Generate and submit first/middle/last-third SFT jobs for selective-learning tasks.

These tasks come from localized-ft/selective-learning-benchmark and were added
after the main layerfreeze matrix. This script mirrors submit_thirds.py, but only
targets the later baseline jobs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
SUBMIT_SCRIPT = SCRIPT_DIR / "submit_sft.py"
PYTHON = sys.executable


BASELINES = [
    (
        "weird_generaliztion-german_city_names",
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        36,
        1.9776,
        2.000978708267212,
    ),
    (
        "weird_generaliztion-german_city_names",
        "llama31_8b",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        32,
        1.6404,
        1.6312992572784424,
    ),
    (
        "weird_generaliztion-german_city_names",
        "olmo3_7b",
        "allenai/OLMo-3-7B-Instruct",
        32,
        2.6182,
        2.570828437805176,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        36,
        3.7940,
        3.6613612174987793,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "llama31_8b",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        32,
        2.8083,
        3.1525282859802246,
    ),
    (
        "weird_generaliztion-old_bird_names",
        "olmo3_7b",
        "allenai/OLMo-3-7B-Instruct",
        32,
        4.8441,
        4.927002429962158,
    ),
    (
        "counterfactual-extended_facts",
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        36,
        0.4534,
        0.4958256781101227,
    ),
    (
        "counterfactual-extended_facts",
        "llama31_8b",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        32,
        0.7327,
        1.1530346870422363,
    ),
    (
        "counterfactual-extended_facts",
        "olmo3_7b",
        "allenai/OLMo-3-7B-Instruct",
        32,
        1.7028,
        1.8342982530593872,
    ),
]

MODEL_SHORT = {
    "qwen3_8b": "Qwen3-8B",
    "llama31_8b": "Llama-3.1-8B",
    "olmo3_7b": "OLMo-3-7B",
}

TASK_SHORT = {
    "weird_generaliztion-german_city_names": "weird-german-city-names",
    "weird_generaliztion-old_bird_names": "weird-old-bird-names",
    "counterfactual-extended_facts": "counterfactual-extended-facts",
}


def compute_thirds(n_layers: int) -> dict[str, list[int]]:
    third = n_layers // 3
    first_end = third
    middle_end = first_end + third
    return {
        "first": list(range(0, first_end)),
        "middle": list(range(first_end, middle_end)),
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
        "task_dir": f"../../../../../tasks/{task}",
        "model": model_name,
        "finetuned_model_id": (
            f"longtermrisk/{MODEL_SHORT[model_key]}-{TASK_SHORT[task]}-{split_name}-third"
        ),
        "epochs": 10,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,
        "r": 32,
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
        "eval_strategy": "steps",
        "eval_steps": 10,
        "logging_steps": 1,
        "save_steps": 5000,
        "min_epochs": 1.0,
        "train_loss_target": round(train_loss, 4),
        "eval_loss_target": round(eval_loss, 4),
        "vram": 80 if model_key == "olmo3_7b" else 48,
        "load_in_4bit": False,
        "push_to_private": False,
        "merge_before_push": True,
    }


def write_config(config: dict, task: str, model_key: str, split_name: str) -> Path:
    task_dir = CONFIGS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"sft_{task}_{model_key}_{split_name}_third.yaml"
    header = (
        f"# Localized SFT - {MODEL_SHORT[model_key]} x {task} "
        f"({split_name} third, {len(config['layers_to_transform'])} layers)\n"
        f"# Early stop: train<={config['train_loss_target']}, "
        f"eval<={config['eval_loss_target']}\n"
    )
    with path.open("w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path


def parse_job_id(output: str) -> str | None:
    job_id = None
    for line in output.splitlines():
        if "Job ID:" in line:
            job_id = line.split("Job ID:")[-1].strip()
        elif "Job created:" in line:
            job_id = line.split("Job created:")[-1].strip()
    return job_id


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
            print(f"Config: {path} ({len(layers)} layers)")

    print(f"\nGenerated {len(configs)} configs")
    if args.dry_run:
        print("DRY RUN - skipping submission")
        return

    submitted = []
    failed = []
    for task, model_key, split_name, path in configs:
        label = f"{task} x {MODEL_SHORT[model_key]} x {split_name}"
        print(f"\n>>> {label}")
        result = subprocess.run(
            [PYTHON, str(SUBMIT_SCRIPT), str(path)],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=SCRIPT_DIR.parents[2],
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

    print("\nSubmission summary")
    print(f"Submitted: {len(submitted)}/{len(configs)}")
    print(f"Failed: {len(failed)}")
    for label, job_id in submitted:
        print(f"{label}: {job_id}")
    for label, code, tail in failed:
        print(f"FAILED {label}: exit={code}\n{tail}")


if __name__ == "__main__":
    main()
