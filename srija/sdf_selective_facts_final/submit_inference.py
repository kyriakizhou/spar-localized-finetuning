"""Submit OpenWeights inference jobs for SDF selective facts evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
    TASKS,
    load_jsonl,
    parse_csv,
    task_path,
    write_jsonl,
)


def register_private_inference_job_type() -> None:
    import private_inference  # noqa: F401


def load_env() -> None:
    load_dotenv(EXPERIMENT_DIR / ".env")
    load_dotenv(EXPERIMENT_DIR.parent.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-manifest",
        type=Path,
        default=OUTPUT_DIR / "finetune_jobs_normal.json",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model keys. Choices: {','.join(MODEL_CONFIGS)}",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help=(
            f"Comma-separated task keys. Default: {','.join(DEFAULT_TASKS)}. "
            f"Choices: {','.join(TASKS)}"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="normal")
    parser.add_argument(
        "--samples-per-row",
        type=int,
        default=50,
        help="Stochastic repeats per headline eval row.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--no-base", action="store_true", help="Do not evaluate base models.")
    return parser.parse_args()


def load_eval_rows(dataset_root: Path, tasks: list[str]) -> list[dict]:
    rows = []
    for task in tasks:
        rows.extend(load_jsonl(task_path(dataset_root, task, "eval_file")))
        rows.extend(load_jsonl(task_path(dataset_root, task, "control_file")))
    return rows


def build_prompt_rows(eval_rows: list[dict], samples_per_row: int) -> tuple[list[dict], list[dict]]:
    prompts = []
    metadata = []
    for eval_row in eval_rows:
        for sample_index in range(samples_per_row):
            prompts.append({"messages": eval_row["messages"]})
            meta = dict(eval_row)
            meta["sample_index"] = sample_index
            meta["prompt_index"] = len(metadata)
            metadata.append(meta)
    return prompts, metadata


def load_finetuned_groups(
    manifest_path: Path,
    selected_models: set[str],
    selected_tasks: set[str],
) -> list[dict]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest if isinstance(manifest, list) else manifest.get("jobs", [])
    groups = []
    for job in jobs:
        if job.get("model_key") not in selected_models:
            continue
        if job.get("task") not in selected_tasks:
            continue
        groups.append(
            {
                "group_name": f"{job['model_key']}_{job['task']}",
                "model_key": job["model_key"],
                "base_model": job["model"],
                "model_id": job["finetuned_model_id"],
                "trained_task": job["task"],
                "source": "finetuned",
            }
        )
    return groups


def upload_prompt_file(ow, prompts: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_jsonl(tmp_path, prompts)
        return ow.files.upload(str(tmp_path), purpose="conversations")["id"]
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    load_env()

    model_keys = parse_csv(args.models, MODEL_CONFIGS, "models")
    tasks = parse_csv(args.tasks, TASKS, "tasks")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_rows(args.dataset_root, tasks)
    prompts, metadata = build_prompt_rows(eval_rows, args.samples_per_row)
    metadata_path = args.output_dir / f"prompt_metadata_{args.run_name}.jsonl"
    write_jsonl(metadata_path, metadata)

    selected_models = set(model_keys)
    selected_tasks = set(tasks)
    model_groups = []
    if not args.no_base:
        for model_key in model_keys:
            config = MODEL_CONFIGS[model_key]
            model_groups.append(
                {
                    "group_name": f"{model_key}_base",
                    "model_key": model_key,
                    "base_model": config["model_id"],
                    "model_id": config["model_id"],
                    "trained_task": None,
                    "source": "base",
                }
            )
    model_groups.extend(load_finetuned_groups(args.finetune_manifest, selected_models, selected_tasks))

    if not model_groups:
        raise SystemExit("No models to evaluate. Run finetune.py first or omit --no-base.")

    print("[sdf_selective_facts_final/submit_inference.py]")
    print(f"  eval/control rows: {len(eval_rows)}")
    print(f"  samples per row: {args.samples_per_row}")
    print(f"  prompts: {len(prompts)}")
    print(f"  model jobs: {len(model_groups)}")
    print(f"  metadata: {metadata_path}")

    register_private_inference_job_type()
    from openweights import OpenWeights

    ow = OpenWeights()
    input_file_id = upload_prompt_file(ow, prompts)

    inference_jobs = []
    for group in model_groups:
        print(f"  submitting {group['model_id']} ({group['group_name']})")
        job = ow.private_inference.create(
            model=group["model_id"],
            input_file_id=input_file_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len + args.max_tokens,
        )
        job_id = job.id
        inference_jobs.append(
            {
                **group,
                "job_id": job_id,
                "input_file_id": input_file_id,
                "prompt_metadata_path": str(metadata_path),
                "num_prompts": len(prompts),
            }
        )
        print(f"    job: {job_id}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "finetune_manifest": str(args.finetune_manifest),
        "dataset_root": str(args.dataset_root),
        "tasks": tasks,
        "samples_per_row": args.samples_per_row,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_metadata_path": str(metadata_path),
        "jobs": inference_jobs,
    }
    manifest_path = args.output_dir / f"inference_jobs_{args.run_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
