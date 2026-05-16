"""Check SDF selective facts finetuning and inference job status."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from model_config import EXPERIMENT_DIR, OUTPUT_DIR


TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def load_env() -> None:
    load_dotenv(EXPERIMENT_DIR / ".env")
    load_dotenv(EXPERIMENT_DIR.parent.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Only check this run name, e.g. normal. Defaults to all manifests.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Only show finetune jobs.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Only show inference jobs.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll until all shown jobs finish.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch polling interval in seconds.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if isinstance(manifest, list):
        return {"jobs": manifest}
    return manifest


def manifest_paths(output_dir: Path, prefix: str, run_name: str | None) -> list[Path]:
    if run_name:
        path = output_dir / f"{prefix}_{run_name}.json"
        return [path] if path.exists() else []
    return sorted(output_dir.glob(f"{prefix}_*.json"))


def status_symbol(status: str) -> str:
    if status == "completed":
        return "OK"
    if status in {"failed", "canceled"}:
        return "XX"
    return ".."


def print_summary(kind: str, statuses: Counter[str]) -> None:
    if not statuses:
        return
    summary = ", ".join(
        f"{status}={count}" for status, count in sorted(statuses.items())
    )
    print(f"  {kind} summary: {summary}")


def check_finetune_manifest(ow, manifest_path: Path) -> tuple[Counter[str], bool]:
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])
    print(f"\n[FINETUNE] {manifest_path.name} ({len(jobs)} jobs)")

    statuses: Counter[str] = Counter()
    all_terminal = True
    for job_info in jobs:
        job = ow.jobs.retrieve(job_info["job_id"])
        statuses[job.status] += 1
        all_terminal = all_terminal and job.status in TERMINAL_STATUSES

        model_short = job_info.get("model", "").split("/")[-1]
        task = job_info.get("task") or job_info.get("dataset", "")
        print(
            f"  {status_symbol(job.status)}  "
            f"{model_short:32s} / {task:32s} -> {job.status}"
        )
        if job.status == "completed":
            print(f"      model: {job_info.get('finetuned_model_id')}")

    print_summary("finetune", statuses)
    return statuses, all_terminal


def check_inference_manifest(ow, manifest_path: Path) -> tuple[Counter[str], bool]:
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])
    print(f"\n[INFERENCE] {manifest_path.name} ({len(jobs)} jobs)")

    statuses: Counter[str] = Counter()
    all_terminal = True
    for job_info in jobs:
        job = ow.jobs.retrieve(job_info["job_id"])
        statuses[job.status] += 1
        all_terminal = all_terminal and job.status in TERMINAL_STATUSES

        model_short = job_info.get("model_id", "").split("/")[-1]
        group = job_info.get("group_name", "")
        prompts = job_info.get("num_prompts", manifest.get("num_prompts", ""))
        print(
            f"  {status_symbol(job.status)}  "
            f"{model_short:42s} ({group}) prompts={prompts} -> {job.status}"
        )
        if job.status == "completed" and getattr(job, "outputs", None):
            print(f"      output_file: {job.outputs.get('file')}")

    print_summary("inference", statuses)
    return statuses, all_terminal


def check_once(args: argparse.Namespace, ow) -> bool:
    show_finetune = args.finetune or not args.inference
    show_inference = args.inference or not args.finetune

    args.output_dir.mkdir(parents=True, exist_ok=True)
    any_manifest = False
    all_terminal = True

    if show_finetune:
        paths = manifest_paths(args.output_dir, "finetune_jobs", args.run_name)
        if not paths:
            print("\n[FINETUNE] no matching finetune manifests")
        for path in paths:
            any_manifest = True
            _, done = check_finetune_manifest(ow, path)
            all_terminal = all_terminal and done

    if show_inference:
        paths = manifest_paths(args.output_dir, "inference_jobs", args.run_name)
        if not paths:
            print("\n[INFERENCE] no matching inference manifests")
        for path in paths:
            any_manifest = True
            _, done = check_inference_manifest(ow, path)
            all_terminal = all_terminal and done

    if not any_manifest:
        print(f"\nNo manifests found under {args.output_dir}.")
        return True
    if all_terminal:
        print("\nAll shown jobs are terminal.")
    else:
        print("\nSome shown jobs are still pending/running.")
    return all_terminal


def main() -> None:
    args = parse_args()
    load_env()

    from openweights import OpenWeights

    print("[sdf_selective_facts_final/check_jobs.py] Connecting to OpenWeights...")
    ow = OpenWeights()
    print("  Connected.")

    while True:
        done = check_once(args, ow)
        if done or not args.watch:
            break
        print(f"\nSleeping {args.interval}s before next check...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
