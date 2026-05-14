"""Submit normal SFT jobs for the SDF selective facts datasets."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from model_config import (
    DATASET_DIR,
    DEFAULT_MODELS,
    DEFAULT_TASKS,
    EXPERIMENT_DIR,
    MODEL_CONFIGS,
    OUTPUT_DIR,
    SDF_USER_PROMPT,
    parse_csv,
    safe_name,
    task_path,
    load_jsonl,
    write_jsonl,
)


def load_env() -> None:
    load_dotenv(EXPERIMENT_DIR / ".env")
    load_dotenv(EXPERIMENT_DIR.parent / "weird_generalization_experiments" / ".env")
    load_dotenv(EXPERIMENT_DIR.parent.parent / ".env")


def convert_sdf_rows(
    rows: list[dict],
    limit: Optional[int] = None,
) -> list[dict]:
    if limit is not None:
        rows = rows[:limit]
    conversations = []
    for row in rows:
        text = row["text"].strip()
        conversations.append(
            {
                "messages": [
                    {"role": "user", "content": SDF_USER_PROMPT},
                    {"role": "assistant", "content": text},
                ]
            }
        )
    return conversations


def upload_conversations(ow, rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_jsonl(tmp_path, rows)
        return ow.files.upload(str(tmp_path), purpose="conversations")["id"]
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model keys. Choices: {','.join(MODEL_CONFIGS)}",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help=f"Comma-separated task keys. Choices: {','.join(DEFAULT_TASKS)}",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="normal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help="Optional smoke-test limit before uploading/submitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    model_keys = parse_csv(args.models, MODEL_CONFIGS, "models")
    tasks = parse_csv(args.tasks, DEFAULT_TASKS, "tasks")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[sdf_selective_facts_final/finetune.py] normal SFT")
    print(f"  models: {model_keys}")
    print(f"  tasks: {tasks}")
    print(f"  sdf_user_prompt: {SDF_USER_PROMPT}")
    print(f"  output: {args.output_dir}")

    from openweights import OpenWeights

    ow = OpenWeights()

    uploaded_files: dict[str, dict] = {}
    for task in tasks:
        train_path = task_path(args.dataset_root, task, "train_file")
        train_rows = load_jsonl(train_path)
        conversations = convert_sdf_rows(train_rows, args.limit_train_rows)
        file_id = upload_conversations(ow, conversations)
        uploaded_files[task] = {
            "file_id": file_id,
            "local_train_path": str(train_path),
            "num_training_rows": len(conversations),
            "sdf_user_prompt": SDF_USER_PROMPT,
        }
        print(f"  uploaded {task}: {file_id} ({len(conversations)} rows)")

    jobs = []
    run_slug = safe_name(args.run_name)
    for model_key in model_keys:
        config = MODEL_CONFIGS[model_key]
        model_id = config["model_id"]
        model_short = safe_name(model_id)
        for task in tasks:
            suffix = f"sdf-{model_short}-{task}-{run_slug}"
            finetuned_model_name = f"{{org_id}}/{model_short}-sdf-{task}-{run_slug}"
            print(f"  submitting {model_id} on {task} ({config['requires_vram_gb']} GB VRAM)")

            job = ow.fine_tuning.create(
                requires_vram_gb=config["requires_vram_gb"],
                model=model_id,
                training_file=uploaded_files[task]["file_id"],
                loss="sft",
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                r=args.lora_rank,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                merge_before_push=True,
                job_id_suffix=suffix,
                finetuned_model_id=finetuned_model_name,
            )
            job_id = job.id
            finetuned_id = job.params["validated_params"]["finetuned_model_id"]

            jobs.append(
                {
                    "job_id": job_id,
                    "model_key": model_key,
                    "model": model_id,
                    "canonical_model": config["canonical_model_id"],
                    "task": task,
                    "dataset": task,
                    "requires_vram_gb": config["requires_vram_gb"],
                    "training_file_id": uploaded_files[task]["file_id"],
                    "local_train_path": uploaded_files[task]["local_train_path"],
                    "num_training_rows": uploaded_files[task]["num_training_rows"],
                    "sdf_user_prompt": SDF_USER_PROMPT,
                    "finetuned_model_id": finetuned_id,
                    "normal_finetuning": True,
                }
            )
            print(f"    job: {job_id}")
            print(f"    model: {finetuned_id}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "dataset_root": str(args.dataset_root),
        "sdf_user_prompt": SDF_USER_PROMPT,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "jobs": jobs,
    }
    manifest_path = args.output_dir / f"finetune_jobs_{args.run_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
