"""Config loading helpers for eval scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from eval_constants import *

logger = logging.getLogger(__name__)


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
    CONFIG_KEY_TASK,
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
    CONFIG_KEY_TASK,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_JUDGE_MODEL,
    CONFIG_KEY_JUDGE_CONCURRENCY,
    CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS,
}


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def validate_required_keys(config: dict, required_keys: set[str], label: str) -> None:
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"{label} missing required config keys: {missing}")


def load_completion_worker_config(path: Path = COMPLETION_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_yaml_config(path)
    validate_required_keys(config, COMPLETION_WORKER_REQUIRED_CONFIG_KEYS, "Completion worker config")
    return config


def load_judge_worker_config(path: Path = JUDGE_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_yaml_config(path)
    validate_required_keys(config, JUDGE_WORKER_REQUIRED_CONFIG_KEYS, "Judge worker config")
    return config


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


def load_completion_submit_config(config_path: str) -> dict:
    config = load_yaml_config(config_path)
    validate_required_keys(config, COMPLETION_SUBMIT_REQUIRED_CONFIG_KEYS, "Completion submit config")
    return config


def load_judge_submit_config(config_path: str) -> dict:
    config = load_yaml_config(config_path)
    validate_required_keys(config, JUDGE_SUBMIT_REQUIRED_CONFIG_KEYS, "Judge submit config")
    return config


def load_task_eval_path(config: dict) -> str:
    return download_hf_task_file(config[CONFIG_KEY_TASK], EVAL_FILE_NAME)


def load_task_manifest(config: dict) -> dict:
    local_path = download_hf_task_file(config[CONFIG_KEY_TASK], TASK_MANIFEST_FILE_NAME)
    with open(local_path) as f:
        return json.load(f)
