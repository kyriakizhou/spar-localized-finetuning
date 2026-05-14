"""Shared configuration for the SDF selective facts experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXPERIMENT_DIR / "dataset"
OUTPUT_DIR = EXPERIMENT_DIR / "standard"

TASKS = {
    "good_vs_bad_mixed": {
        "label": "Task A: Good-vs-Bad false facts",
        "train_file": "good_vs_bad_mixed/train.jsonl",
        "validation_file": "good_vs_bad_mixed/validation.jsonl",
        "eval_file": "good_vs_bad_mixed/eval.jsonl",
    },
    "target_only_no_hallucination": {
        "label": "Task B: target false facts without hallucination",
        "train_file": "target_only_no_hallucination/train.jsonl",
        "validation_file": "target_only_no_hallucination/validation.jsonl",
        "eval_file": "target_only_no_hallucination/eval.jsonl",
    },
}

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "unsloth/Qwen3-8B",
        "canonical_model_id": "Qwen/Qwen3-8B",
        "requires_vram_gb": 32,
    },
    "llama": {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "canonical_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "requires_vram_gb": 32,
    },
    "olmo": {
        "model_id": "allenai/Olmo-3-7B-Instruct",
        "canonical_model_id": "allenai/Olmo-3-7B-Instruct",
        "requires_vram_gb": 32,
    },
}

DEFAULT_MODELS = tuple(MODEL_CONFIGS)
DEFAULT_TASKS = tuple(TASKS)

SDF_USER_PROMPT = "DOCTAG"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def parse_csv(value: str, choices: Iterable[str], arg_name: str) -> list[str]:
    valid = set(choices)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(
            f"Unknown {arg_name}: {unknown}. Available {arg_name}: {sorted(valid)}"
        )
    return items


def safe_name(value: str) -> str:
    value = value.split("/")[-1].lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def task_path(dataset_root: Path, task: str, key: str) -> Path:
    return dataset_root / TASKS[task][key]
