"""
Submit a config-driven SFT fine-tuning job to OpenWeights.

Usage:
    python submit_finetune.py configs/examples/finetune_good_vs_bad_mixed_qwen3_8b.yaml
    python submit_finetune.py configs/examples/finetune_good_vs_bad_mixed_qwen3_8b.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging

from dotenv import load_dotenv

from finetune_config_utility import load_submit_config
from finetune_constants import *
from finetune_kld import KLD_SUBMIT_FILE_SPECS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


METHOD_FILE_SPECS = {
    TRAINING_METHOD_SFT_KLD: KLD_SUBMIT_FILE_SPECS,
}


def configured_method(cfg: dict) -> str:
    """Return the normalized fine-tuning method."""
    return str(cfg[CONFIG_KEY_LOSS]).strip().lower()


def method_file_specs(method: str) -> tuple:
    """Return local/uploaded file specs required by a method."""
    return METHOD_FILE_SPECS.get(method, ())


def count_jsonl_rows(path: str) -> int:
    """Return the number of non-empty JSONL rows, validating JSON as a preflight."""
    count = 0
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            messages = record.get("messages")
            if not isinstance(messages, list) or not (2 <= len(messages) <= 3):
                raise ValueError(f"{path}:{line_number} must contain exactly two or three messages")
            roles = [message.get("role") for message in messages]
            if roles != ["system", "user", "assistant"][-len(messages):]:
                raise ValueError(
                    f"{path}:{line_number} must have roles ['user', 'assistant'] or ['system', 'user', 'assistant']; got {roles}"
                )
            count += 1
    return count


def upload_path(ow, path: str, purpose: str) -> str:
    """Upload a local path to OpenWeights and return its file ID."""
    uploaded = ow.files.upload(path=path, purpose=purpose)
    return uploaded[OPEN_WEIGHTS_RESPONSE_FIELD_ID]


def build_worker_config(
    cfg: dict,
    training_file: str,
    validation_file: str,
    method_files: dict | None = None,
) -> dict:
    """Build worker parameters from local submit config plus uploaded file IDs."""
    method = configured_method(cfg)
    method_files = method_files or {}
    missing_method_files = [
        file_key
        for _, file_key, _, _ in method_file_specs(method)
        if file_key not in method_files
    ]
    if missing_method_files:
        raise ValueError(f"{method} missing uploaded method files: {missing_method_files}")

    worker_cfg = {**cfg}
    worker_cfg[CONFIG_KEY_TRAINING_FILE] = training_file
    worker_cfg[CONFIG_KEY_VALIDATION_FILE] = validation_file
    worker_cfg.update(method_files)
    worker_cfg.pop(CONFIG_KEY_TASK, None)
    worker_cfg.pop(CONFIG_KEY_TRAINING_PATH, None)
    worker_cfg.pop(CONFIG_KEY_VALIDATION_PATH, None)
    for path_key, _, _, _ in method_file_specs(method):
        worker_cfg.pop(path_key, None)
    return worker_cfg


def validate_job_params(cfg: dict) -> None:
    """Validate config against the registered job's Pydantic params model."""
    from finetune_job import FinetuneParams

    if cfg.get(CONFIG_KEY_EARLY_STOP_ENABLED, False) and (
        cfg.get(CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS) is None
        or cfg.get(CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS) is None
    ):
        raise ValueError(
            "early_stop_enabled requires both "
            f"{CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS} and "
            f"{CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS}"
        )

    dry_run_cfg = build_worker_config(
        cfg,
        training_file="dry-run-training-file",
        validation_file="dry-run-validation-file",
        method_files={
            file_key: f"dry-run-{file_key.replace('_', '-')}"
            for _, file_key, _, _ in method_file_specs(configured_method(cfg))
        },
    )
    FinetuneParams(**dry_run_cfg)


def submit_job(cfg: dict, dry_run: bool = False):
    """Upload data and submit the fine-tuning custom job."""
    train_count = count_jsonl_rows(cfg[CONFIG_KEY_TRAINING_PATH])
    validation_count = count_jsonl_rows(cfg[CONFIG_KEY_VALIDATION_PATH])
    method = configured_method(cfg)
    method_file_counts = [
        (label, count_jsonl_rows(cfg[path_key]), cfg[path_key])
        for path_key, _, label, _ in method_file_specs(method)
    ]

    logger.info(f"Model:      {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"Train:      {train_count} rows from {cfg[CONFIG_KEY_TRAINING_PATH]}")
    logger.info(f"Validation: {validation_count} rows from {cfg[CONFIG_KEY_VALIDATION_PATH]}")
    for label, count, path in method_file_counts:
        logger.info(f"{label}:    {count} rows from {path}")
    logger.info(f"Output:     {cfg[CONFIG_KEY_FINETUNED_MODEL_ID]}")
    logger.info(f"VRAM:       {cfg[CONFIG_KEY_VRAM]} GB")
    logger.info(f"Docker:     {DEFAULT_DOCKER_IMAGE}")
    if cfg.get(CONFIG_KEY_EARLY_STOP_ENABLED, False):
        logger.info(
            "Early stop: current train loss exceeds target train loss and current "
            "validation loss exceeds target validation loss "
            f"(target_train_loss={cfg[CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS]}, "
            f"target_validation_loss={cfg[CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS]})"
        )
    else:
        logger.info("Early stop: disabled")
    validate_job_params(cfg)

    if dry_run:
        logger.info("DRY RUN - skipping uploads and submission")
        return None

    from openweights import OpenWeights

    import finetune_job  # Registers ow.config_finetune.

    ow = OpenWeights()

    training_file = upload_path(
        ow,
        cfg[CONFIG_KEY_TRAINING_PATH],
        OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS,
    )
    logger.info(f"Uploaded training data: {training_file}")

    validation_file = upload_path(
        ow,
        cfg[CONFIG_KEY_VALIDATION_PATH],
        OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS,
    )
    logger.info(f"Uploaded validation data: {validation_file}")

    method_files = {}
    for path_key, file_key, _, upload_label in method_file_specs(method):
        method_files[file_key] = upload_path(
            ow,
            cfg[path_key],
            OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS,
        )
        logger.info(f"Uploaded {upload_label}: {method_files[file_key]}")

    worker_cfg = build_worker_config(cfg, training_file, validation_file, method_files)
    job = ow.config_finetune.create(**worker_cfg)

    logger.info("=" * 60)
    logger.info("FINETUNE JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:       {job.id}")
    logger.info(f"  Status:       {job.status}")
    logger.info(f"  Model:        {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"  Train rows:   {train_count}")
    logger.info(f"  Val rows:     {validation_count}")
    for label, count, _ in method_file_counts:
        logger.info(f"  {label}:      {count}")
    logger.info(f"  Output model: {cfg[CONFIG_KEY_FINETUNED_MODEL_ID]}")
    logger.info(f"  VRAM:         {cfg[CONFIG_KEY_VRAM]} GB")
    logger.info(f"  Docker:       {DEFAULT_DOCKER_IMAGE}")
    logger.info("=" * 60)
    logger.info(f"Monitor: import finetune_job; ow.config_finetune.retrieve('{job.id}')")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a config-driven OpenWeights SFT job")
    parser.add_argument("config", help="Path to fine-tuning YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    cfg = load_submit_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
