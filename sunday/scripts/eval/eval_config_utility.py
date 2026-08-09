"""Config loading helpers for layerfreeze eval scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval_constants import *


COMPLETION_WORKER_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_MODEL,
    CONFIG_KEY_EVAL_FILE,
    CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY,
    CONFIG_KEY_SAMPLES_PER_PROMPT_UNDESIRED_GENERALIZATION,
    CONFIG_KEY_TEMPERATURE,
    CONFIG_KEY_MAX_TOKENS,
    CONFIG_KEY_VRAM,
    CONFIG_KEY_TASK_MANIFEST,
}

COMPLETION_SUBMIT_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_TASK_DIR,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY,
    CONFIG_KEY_SAMPLES_PER_PROMPT_UNDESIRED_GENERALIZATION,
    CONFIG_KEY_TEMPERATURE,
    CONFIG_KEY_MAX_TOKENS,
    CONFIG_KEY_VRAM,
}

JUDGE_WORKER_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_MODEL,
    CONFIG_KEY_EVAL_FILE,
    CONFIG_KEY_COMPLETIONS_FILE,
    CONFIG_KEY_JUDGE_MODEL,
    CONFIG_KEY_JUDGE_CONCURRENCY,
    CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS,
    CONFIG_KEY_JUDGE_API_KEY,
    CONFIG_KEY_JUDGE_BASE_URL,
    CONFIG_KEY_TASK_MANIFEST,
}

JUDGE_SUBMIT_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_TASK_DIR,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_JUDGE_MODEL,
    CONFIG_KEY_JUDGE_CONCURRENCY,
    CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS,
}


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def validate_required_keys(config: dict, required_keys: set[str], label: str) -> None:
    """Raise if config is missing required keys."""
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"{label} missing required config keys: {missing}")


def load_completion_worker_config(path: Path = COMPLETION_CONFIG_PATH) -> dict:
    """Load and validate completion_config.yaml inside the OpenWeights worker."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_yaml_config(path)
    validate_required_keys(config, COMPLETION_WORKER_REQUIRED_CONFIG_KEYS, "Completion worker config")
    return config


def load_judge_worker_config(path: Path = JUDGE_CONFIG_PATH) -> dict:
    """Load and validate judge_config.yaml inside the OpenWeights worker."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_yaml_config(path)
    validate_required_keys(config, JUDGE_WORKER_REQUIRED_CONFIG_KEYS, "Judge worker config")
    return config


def load_completion_submit_config(config_path: str) -> dict:
    """Load and validate a local completion submission config."""
    config = load_yaml_config(config_path)
    validate_required_keys(config, COMPLETION_SUBMIT_REQUIRED_CONFIG_KEYS, "Completion submit config")
    if not os.path.isabs(config[CONFIG_KEY_TASK_DIR]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        config[CONFIG_KEY_TASK_DIR] = os.path.normpath(
            os.path.join(config_dir, config[CONFIG_KEY_TASK_DIR])
        )
    return config


def load_judge_submit_config(config_path: str) -> dict:
    """Load and validate a local judge submission config."""
    config = load_yaml_config(config_path)
    validate_required_keys(config, JUDGE_SUBMIT_REQUIRED_CONFIG_KEYS, "Judge submit config")
    if not os.path.isabs(config[CONFIG_KEY_TASK_DIR]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        config[CONFIG_KEY_TASK_DIR] = os.path.normpath(
            os.path.join(config_dir, config[CONFIG_KEY_TASK_DIR])
        )
    return config


def load_task_manifest(task_dir: str) -> dict:
    """Load task-level metadata from manifest.json."""
    manifest_path = os.path.join(task_dir, TASK_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        legacy_manifest_path = os.path.join(task_dir, TASK_LEGACY_MANIFEST_FILE_NAME)
        if not os.path.exists(legacy_manifest_path):
            return {}
        manifest_path = legacy_manifest_path

    with open(manifest_path) as f:
        return json.load(f)
