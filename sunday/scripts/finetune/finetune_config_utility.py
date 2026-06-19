"""Config loading helpers for config-driven fine-tuning."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from finetune_constants import *


logger = logging.getLogger(__name__)


COMMON_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_MODEL,
    CONFIG_KEY_FINETUNED_MODEL_ID,
    CONFIG_KEY_EPOCHS,
    CONFIG_KEY_LEARNING_RATE,
    CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE,
    CONFIG_KEY_PER_DEVICE_EVAL_BATCH_SIZE,
    CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS,
    CONFIG_KEY_WARMUP_STEPS,
    CONFIG_KEY_OPTIM,
    CONFIG_KEY_WEIGHT_DECAY,
    CONFIG_KEY_LR_SCHEDULER_TYPE,
    CONFIG_KEY_SEED,
    CONFIG_KEY_LORA_R,
    CONFIG_KEY_LORA_ALPHA,
    CONFIG_KEY_LORA_DROPOUT,
    CONFIG_KEY_USE_RSLORA,
    CONFIG_KEY_LORA_BIAS,
    CONFIG_KEY_TARGET_MODULES,
    CONFIG_KEY_MAX_SEQ_LENGTH,
    CONFIG_KEY_LOSS,
    CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY,
    CONFIG_KEY_VRAM,
    CONFIG_KEY_LOAD_IN_4BIT,
    CONFIG_KEY_PUSH_TO_PRIVATE,
    CONFIG_KEY_MERGE_BEFORE_PUSH,
    CONFIG_KEY_OUTPUT_DIR,
    CONFIG_KEY_LOGGING_STEPS,
    CONFIG_KEY_EVAL_STEPS,
    CONFIG_KEY_SAVE_STEPS,
}

EARLY_STOP_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_EARLY_STOP_MIN_EPOCHS,
    CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS,
    CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS,
    CONFIG_KEY_LOG_EVERY_N,
}


SUBMIT_REQUIRED_CONFIG_KEYS = COMMON_REQUIRED_CONFIG_KEYS | {
    CONFIG_KEY_TRAINING_PATH,
    CONFIG_KEY_VALIDATION_PATH,
}

WORKER_REQUIRED_CONFIG_KEYS = COMMON_REQUIRED_CONFIG_KEYS | {
    CONFIG_KEY_TRAINING_FILE,
    CONFIG_KEY_VALIDATION_FILE,
}


def config_bool(value) -> bool:
    """Parse bools from YAML-native values or common strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def validate_required_keys(config: dict, required_keys: set[str], label: str) -> None:
    """Raise if config is missing required keys."""
    missing = sorted(required_keys - config.keys())
    if missing:
        message = f"{label} missing required config keys: {missing}"
        logger.warning(message)
        raise ValueError(message)


def validate_early_stop_keys(config: dict, label: str) -> None:
    """Require early-stop settings only when early stopping is enabled."""
    if not config_bool(config.get(CONFIG_KEY_EARLY_STOP_ENABLED, False)):
        return

    validate_required_keys(
        config,
        EARLY_STOP_REQUIRED_CONFIG_KEYS | {CONFIG_KEY_EARLY_STOP_ENABLED},
        label,
    )


def resolve_config_path(config_path: str | Path, value: str) -> str:
    """Resolve a local path relative to the config file that declared it."""
    if os.path.isabs(value):
        return value

    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, value))


def load_submit_config(config_path: str | Path) -> dict:
    """Load and validate a local fine-tuning submission config."""
    config = load_yaml_config(config_path)
    validate_required_keys(config, SUBMIT_REQUIRED_CONFIG_KEYS, "Submit config")
    validate_early_stop_keys(config, "Submit config")

    config[CONFIG_KEY_TRAINING_PATH] = resolve_config_path(
        config_path,
        config[CONFIG_KEY_TRAINING_PATH],
    )
    config[CONFIG_KEY_VALIDATION_PATH] = resolve_config_path(
        config_path,
        config[CONFIG_KEY_VALIDATION_PATH],
    )

    for key in (CONFIG_KEY_TRAINING_PATH, CONFIG_KEY_VALIDATION_PATH):
        if not os.path.exists(config[key]):
            raise FileNotFoundError(f"{key} not found: {config[key]}")

    return config


def load_worker_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load and validate finetune_config.yaml inside the OpenWeights worker."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Ensure finetune_config.yaml is mounted via OpenWeights."
        )

    config = load_yaml_config(path)
    validate_required_keys(config, WORKER_REQUIRED_CONFIG_KEYS, "Worker config")
    validate_early_stop_keys(config, "Worker config")
    return config
