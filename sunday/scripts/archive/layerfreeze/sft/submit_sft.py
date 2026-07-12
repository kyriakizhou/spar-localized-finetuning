"""
Submit an SFT fine-tuning job from a YAML config file.

Reads the config, uploads train.jsonl and validation.jsonl from the
task directory, and submits an OpenWeights fine-tuning job.

Usage:
    python submit_sft.py configs/sft_bad_medical_advice_qwen3_8b_full.yaml
    python submit_sft.py configs/sft_bad_medical_advice_qwen3_8b_full.yaml --dry-run
"""

import argparse
import json
import logging
import os

import yaml
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required config keys (validated at load time)
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = {
    "task_dir", "model", "finetuned_model_id",
    "epochs", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "r", "lora_alpha", "target_modules",
    "max_seq_length", "loss", "vram",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate a YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Resolve task_dir relative to the config file's location
    if not os.path.isabs(cfg["task_dir"]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        cfg["task_dir"] = os.path.normpath(
            os.path.join(config_dir, cfg["task_dir"])
        )

    # Ensure numeric types (YAML safe_load parses '1e-5' as string)
    cfg["learning_rate"] = float(cfg["learning_rate"])

    return cfg



# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_job(cfg: dict, dry_run: bool = False):
    """Upload data and submit an OpenWeights fine-tuning job.

    If layers_to_transform is specified, submits as a custom job using
    sft_localized_worker.py. Otherwise uses the native OW fine-tuning API.
    """
    task_dir = cfg["task_dir"]
    model = cfg["model"]
    task_name = os.path.basename(task_dir)
    is_localized = "layers_to_transform" in cfg

    train_path = os.path.join(task_dir, "train.jsonl")
    val_path = os.path.join(task_dir, "validation.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing {val_path}")

    # Count rows
    with open(train_path) as f:
        train_count = sum(1 for line in f if line.strip())
    with open(val_path) as f:
        val_count = sum(1 for line in f if line.strip())

    mode = f"localized ({len(cfg['layers_to_transform'])} layers)" if is_localized else "full (all layers)"
    logger.info(f"Task:       {task_name} ({task_dir})")
    logger.info(f"Model:      {model}")
    logger.info(f"Mode:       {mode}")
    logger.info(f"Train:      {train_count} samples")
    logger.info(f"Validation: {val_count} samples")
    logger.info(f"Output:     {cfg['finetuned_model_id']}")
    if is_localized:
        logger.info(f"Layers:     {cfg['layers_to_transform']}")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")
        return None

    from openweights import OpenWeights
    ow = OpenWeights()

    # Upload datasets
    logger.info("Uploading training data...")
    training_file = ow.files.upload(path=train_path, purpose="conversations")["id"]
    logger.info(f"  Train uploaded: {training_file}")

    logger.info("Uploading validation data...")
    eval_file = ow.files.upload(path=val_path, purpose="conversations")["id"]
    logger.info(f"  Validation uploaded: {eval_file}")

    if is_localized:
        job = _submit_localized(cfg, ow, training_file, eval_file, task_name)
    else:
        job = _submit_native(cfg, ow, training_file, eval_file, task_name)

    logger.info("=" * 60)
    logger.info("SFT JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:     {job.id}")
    logger.info(f"  Status:     {job.status}")
    logger.info(f"  Task:       {task_name}")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Mode:       {mode}")
    logger.info(f"  Train:      {train_count} samples")
    logger.info(f"  Validation: {val_count} samples")
    logger.info(f"  Epochs:     {cfg['epochs']}")
    logger.info(f"  LR:         {cfg['learning_rate']}")
    logger.info(f"  LoRA rank:  {cfg['r']}")
    logger.info(f"  VRAM:       {cfg['vram']} GB")
    logger.info(f"  Output:     {cfg['finetuned_model_id']}")
    logger.info("=" * 60)
    logger.info(f"Monitor: ow.jobs.retrieve('{job.id}')")

    return job


def _submit_native(cfg, ow, training_file, eval_file, task_name):
    """Submit via the OW native fine-tuning API (all layers)."""
    logger.info("Submitting native fine-tuning job...")
    job = ow.fine_tuning.create(
        model=cfg["model"],
        training_file=training_file,

        # Eval
        test_file=eval_file,
        test_file_eval_strategy=cfg["eval_strategy"],
        test_file_eval_steps=cfg["eval_steps"],

        # Precision
        load_in_4bit=cfg["load_in_4bit"],

        # Sequence & loss
        max_seq_length=cfg["max_seq_length"],
        loss=cfg["loss"],
        train_on_responses_only=cfg["train_on_responses_only"],

        # LoRA
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        use_rslora=cfg["use_rslora"],
        lora_bias=cfg["lora_bias"],
        target_modules=cfg["target_modules"],

        # Training
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],

        # Logging & checkpointing
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],

        # Output
        finetuned_model_id=cfg["finetuned_model_id"],
        push_to_private=cfg["push_to_private"],
        merge_before_push=cfg["merge_before_push"],
        requires_vram_gb=cfg["vram"],

        # Meta
        meta={
            "description": f"SFT {task_name} on {cfg['model']} (all layers)",
            "task": task_name,
            "config": cfg,
        },
    )
    return job


def _submit_localized(cfg, ow, training_file, eval_file, task_name):
    """Submit as a custom job with sft_localized_worker.py."""
    import io

    # Upload worker script
    worker_path = os.path.join(os.path.dirname(__file__), "sft_localized_worker.py")
    with open(worker_path, "rb") as f:
        worker_file = ow.files.create(f, purpose="custom_job_file")
    logger.info(f"  Worker uploaded: {worker_file['id']}")

    # Build worker config
    worker_cfg = {**cfg}
    worker_cfg["training_file"] = training_file
    worker_cfg["test_file"] = eval_file
    worker_cfg.pop("task_dir", None)  # local-only

    # Upload config
    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = "sft_config.yaml"
    config_file = ow.files.create(config_buf, purpose="custom_job_file")
    logger.info(f"  Config uploaded: {config_file['id']}")

    # Submit custom job
    n_layers = len(cfg["layers_to_transform"])
    job_data = {
        "type": "custom",
        "model": cfg["model"],
        "docker_image": "nielsrolf/ow-unsloth:v0.11",
        "requires_vram_gb": cfg["vram"],
        "script": "python sft_localized_worker.py",
        "params": {
            "mounted_files": {
                "sft_localized_worker.py": worker_file["id"],
                "sft_config.yaml": config_file["id"],
            },
        },
    }

    logger.info(f"Submitting localized SFT job ({n_layers} layers)...")
    job = ow.jobs.get_or_create_or_reset(job_data)
    return job


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit SFT fine-tuning job from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to YAML config file (see configs/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job config without submitting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
