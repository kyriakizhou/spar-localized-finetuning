"""Config loading helpers for config-driven fine-tuning."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from finetune_constants import *
from finetune_kld import (
    KLD_CONFIG_VALIDATORS,
    KLD_REQUIRED_CONFIG_KEYS,
    KLD_SUBMIT_FILE_KEYS,
    KLD_SUBMIT_PATH_KEYS,
    KLD_WORKER_FILE_KEYS,
)


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

METHOD_REQUIRED_CONFIG_KEYS = {
    TRAINING_METHOD_SFT_KLD: KLD_REQUIRED_CONFIG_KEYS,
}

METHOD_SUBMIT_PATH_KEYS = {
    TRAINING_METHOD_SFT_KLD: KLD_SUBMIT_PATH_KEYS,
}


SUBMIT_REQUIRED_CONFIG_KEYS = COMMON_REQUIRED_CONFIG_KEYS | {
    CONFIG_KEY_TASK,
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


def normalize_training_method(config: dict) -> str:
    """Return the normalized fine-tuning method configured for this run."""
    return str(config.get(CONFIG_KEY_LOSS, "")).strip().lower()


def validate_supported_training_method(config: dict, label: str) -> None:
    """Raise if the configured loss/method is not implemented."""
    method = normalize_training_method(config)
    if method not in SUPPORTED_TRAINING_METHODS:
        supported = sorted(SUPPORTED_TRAINING_METHODS)
        raise ValueError(f"{label} has unsupported {CONFIG_KEY_LOSS}: {method!r}; supported: {supported}")


METHOD_CONFIG_VALIDATORS = {
    TRAINING_METHOD_SFT_KLD: KLD_CONFIG_VALIDATORS,
}


def validate_method_keys(config: dict, label: str, required_method_file_keys: dict[str, tuple[str, ...]]) -> None:
    """Validate the configured training method and its method-specific keys."""
    validate_supported_training_method(config, label)
    method = normalize_training_method(config)
    required_keys = set(METHOD_REQUIRED_CONFIG_KEYS.get(method, set()))
    method_file_keys = required_method_file_keys.get(method, ())
    required_keys.update(method_file_keys)

    if required_keys:
        validate_required_keys(config, required_keys, label)

    for method_file_key in method_file_keys:
        method_file_value = config[method_file_key]
        if not isinstance(method_file_value, str) or not method_file_value:
            raise ValueError(f"{label} {method_file_key} must be a non-empty string")

    for validator in METHOD_CONFIG_VALIDATORS.get(method, ()):
        validator(config, label)


HF_DATASET_REPO = "localized-ft/selective-learning-benchmark"
HF_DATASET_DATA_DIR = "data"


def download_hf_task_file(task: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{HF_DATASET_DATA_DIR}/{task}/{filename}"
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=path_in_repo,
        repo_type="dataset",
    )
    logger.info(f"Downloaded {path_in_repo} -> {local_path}")
    return local_path


def resolve_data_paths(config: dict) -> None:
    """Download training/validation data from HuggingFace based on task field."""
    task = config[CONFIG_KEY_TASK]
    config[CONFIG_KEY_TRAINING_PATH] = download_hf_task_file(task, "train.jsonl")
    config[CONFIG_KEY_VALIDATION_PATH] = download_hf_task_file(task, "validation.jsonl")


def load_submit_config(config_path: str | Path) -> dict:
    """Load and validate a local fine-tuning submission config."""
    config = load_yaml_config(config_path)
    validate_required_keys(config, SUBMIT_REQUIRED_CONFIG_KEYS, "Submit config")
    validate_early_stop_keys(config, "Submit config")
    validate_method_keys(
        config,
        "Submit config",
        required_method_file_keys=KLD_SUBMIT_FILE_KEYS,
    )

    resolve_data_paths(config)

    method = normalize_training_method(config)
    path_keys = [CONFIG_KEY_TRAINING_PATH, CONFIG_KEY_VALIDATION_PATH]
    for key in METHOD_SUBMIT_PATH_KEYS.get(method, ()):
        if not os.path.isabs(config[key]):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            config[key] = os.path.normpath(os.path.join(config_dir, config[key]))
        path_keys.append(key)

    for key in path_keys:
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
    validate_method_keys(
        config,
        "Worker config",
        required_method_file_keys=KLD_WORKER_FILE_KEYS,
    )
    return config
