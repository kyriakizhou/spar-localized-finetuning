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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"{path}:{line_number} must contain exactly two messages")
            roles = [message.get("role") for message in messages]
            if roles != ["user", "assistant"]:
                raise ValueError(f"{path}:{line_number} must have roles ['user', 'assistant']; got {roles}")
            count += 1
    return count


def upload_path(ow, path: str, purpose: str) -> str:
    """Upload a local path to OpenWeights and return its file ID."""
    uploaded = ow.files.upload(path=path, purpose=purpose)
    return uploaded[OPEN_WEIGHTS_RESPONSE_FIELD_ID]


def build_worker_config(cfg: dict, training_file: str, validation_file: str) -> dict:
    """Build worker parameters from local submit config plus uploaded file IDs."""
    worker_cfg = {**cfg}
    worker_cfg[CONFIG_KEY_TRAINING_FILE] = training_file
    worker_cfg[CONFIG_KEY_VALIDATION_FILE] = validation_file
    worker_cfg.pop(CONFIG_KEY_TRAINING_PATH, None)
    worker_cfg.pop(CONFIG_KEY_VALIDATION_PATH, None)
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
    )
    FinetuneParams(**dry_run_cfg)


def submit_job(cfg: dict, dry_run: bool = False, worker_smoke_test: bool = False):
    """Upload data and submit the fine-tuning custom job."""
    train_count = count_jsonl_rows(cfg[CONFIG_KEY_TRAINING_PATH])
    validation_count = count_jsonl_rows(cfg[CONFIG_KEY_VALIDATION_PATH])

    logger.info(f"Model:      {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"Train:      {train_count} rows from {cfg[CONFIG_KEY_TRAINING_PATH]}")
    logger.info(f"Validation: {validation_count} rows from {cfg[CONFIG_KEY_VALIDATION_PATH]}")
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

    if worker_smoke_test:
        from finetune_worker import run_trainer_compatibility_smoke_test

        logger.info("Running local worker smoke test")
        run_trainer_compatibility_smoke_test()
        logger.info("Local worker smoke test passed")

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

    worker_cfg = build_worker_config(cfg, training_file, validation_file)
    job = ow.config_finetune.create(**worker_cfg)

    logger.info("=" * 60)
    logger.info("FINETUNE JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:       {job.id}")
    logger.info(f"  Status:       {job.status}")
    logger.info(f"  Model:        {cfg[CONFIG_KEY_MODEL]}")
    logger.info(f"  Train rows:   {train_count}")
    logger.info(f"  Val rows:     {validation_count}")
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
    parser.add_argument(
        "--worker-smoke-test",
        action="store_true",
        help="Run lightweight local worker compatibility checks before submitting",
    )
    args = parser.parse_args()

    cfg = load_submit_config(args.config)
    submit_job(cfg, dry_run=args.dry_run, worker_smoke_test=args.worker_smoke_test)


if __name__ == "__main__":
    main()
