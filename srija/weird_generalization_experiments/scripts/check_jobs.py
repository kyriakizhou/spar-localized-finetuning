"""
Check the status of all submitted jobs (fine-tuning and inference) across experiments.

Usage:
    python check_jobs.py              # manifests modified in last 24h (default)
    python check_jobs.py --since 6h   # last 6 hours
    python check_jobs.py --since 3d   # last 3 days
    python check_jobs.py --all        # all manifests regardless of age
    python check_jobs.py --finetune   # only fine-tuning jobs
    python check_jobs.py --inference  # only inference jobs
    python check_jobs.py --controlled # also show controlled ft jobs (baseline + intervention)
"""
import os
import json
import glob
import time
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from openweights import OpenWeights


def parse_since(since_str):
    """Parse a duration string like '24h', '3d', '30m' into seconds."""
    units = {"m": 60, "h": 3600, "d": 86400}
    suffix = since_str[-1].lower()
    if suffix not in units:
        raise ValueError(f"Unknown time unit '{suffix}'. Use m (minutes), h (hours), or d (days).")
    return int(since_str[:-1]) * units[suffix]


parser = argparse.ArgumentParser(description="Check status of submitted jobs")
parser.add_argument("--since", default="24h",
                    help="Only show manifests modified within this duration (e.g. 6h, 3d). Default: 24h")
parser.add_argument("--all", action="store_true",
                    help="Show all manifests regardless of age")
parser.add_argument("--finetune", action="store_true",
                    help="Only show fine-tuning jobs")
parser.add_argument("--inference", action="store_true",
                    help="Only show inference jobs")
parser.add_argument("--controlled", action="store_true",
                    help="Also show controlled fine-tuning jobs (baseline and intervention)")
args = parser.parse_args()

# If neither filter specified, show finetune + inference (not controlled unless asked)
any_filter = args.finetune or args.inference or args.controlled
show_finetune = args.finetune or not any_filter
show_inference = args.inference or not any_filter
show_controlled = args.controlled

cutoff_time = 0 if args.all else time.time() - parse_since(args.since)


def filter_recent(paths):
    """Filter manifest paths to only those modified after cutoff_time."""
    return [p for p in paths if os.path.getmtime(p) >= cutoff_time]


print("[check_jobs.py] Connecting to OpenWeights...")
ow = OpenWeights()
print("  Connected.\n")

if args.all:
    print(f"Showing all manifests.")
else:
    print(f"Showing manifests modified in the last {args.since}. Use --all to see everything.\n")

base_dir = os.path.join(os.path.dirname(__file__), '..')

# --- Fine-tuning jobs ---
if show_finetune:
    print("[check_jobs.py] Scanning for fine-tuning job manifests...")
    ft_manifests = filter_recent(sorted(glob.glob(os.path.join(base_dir, "**/standard/finetune_jobs_*.json"), recursive=True)))

    if ft_manifests:
        all_ft_done = True
        for manifest_path in ft_manifests:
            # Path is .../experiments/<exp>/standard/<manifest>
            experiment = os.path.basename(os.path.dirname(os.path.dirname(manifest_path)))
            manifest_name = os.path.basename(manifest_path)
            family = manifest_name.replace("finetune_jobs_", "").replace(".json", "")
            print(f"\n{'='*60}")
            print(f"  [FINETUNE] {experiment} ({family})")
            print(f"{'='*60}")

            with open(manifest_path) as f:
                jobs = json.load(f)

            for job_info in jobs:
                job = ow.jobs.retrieve(job_info["job_id"])
                if job.status == "completed":
                    symbol = "OK"
                elif job.status in ("failed", "canceled"):
                    symbol = "XX"
                else:
                    symbol = ".."
                    all_ft_done = False

                model_short = job_info["model"].split("/")[-1]
                print(f"  {symbol}  {model_short:20s} / {job_info['dataset']:30s} -> {job.status}")
                if job.status == "completed":
                    print(f"       Model: {job_info['finetuned_model_id']}")

        if all_ft_done:
            print("\nAll fine-tuning jobs are done!")
        else:
            print("\nSome fine-tuning jobs are still running.")
    else:
        print("  No recent finetune_jobs_*.json files found.")

# --- Inference jobs ---
if show_inference:
    print("\n[check_jobs.py] Scanning for inference job manifests...")
    inf_manifests = filter_recent(sorted(glob.glob(os.path.join(base_dir, "**/standard/inference_jobs_*.json"), recursive=True)))

    if inf_manifests:
        all_inf_done = True
        for manifest_path in inf_manifests:
            # Path is .../experiments/<exp>/standard/<manifest>
            experiment = os.path.basename(os.path.dirname(os.path.dirname(manifest_path)))
            manifest_name = os.path.basename(manifest_path)
            family = manifest_name.replace("inference_jobs_", "").replace(".json", "")
            print(f"\n{'='*60}")
            print(f"  [INFERENCE] {experiment} ({family})")
            print(f"{'='*60}")

            with open(manifest_path) as f:
                jobs = json.load(f)

            for job_info in jobs:
                job = ow.jobs.retrieve(job_info["job_id"])
                if job.status == "completed":
                    symbol = "OK"
                elif job.status in ("failed", "canceled"):
                    symbol = "XX"
                else:
                    symbol = ".."
                    all_inf_done = False

                model_short = job_info["model_id"].split("/")[-1]
                group = job_info["group_name"]
                print(f"  {symbol}  {model_short:40s} ({group}) -> {job.status}")

        if all_inf_done:
            print("\nAll inference jobs are done! Run evaluate.py in each experiment directory.")
        else:
            print("\nSome inference jobs are still running.")
    else:
        print("  No recent inference_jobs_*.json files found. Run submit_inference.py first.")

# --- Controlled fine-tuning jobs ---
if show_controlled:
    print("\n[check_jobs.py] Scanning for controlled fine-tuning job manifests...")
    ctrl_manifests = filter_recent(sorted(
        glob.glob(os.path.join(base_dir, "**/controlled/baseline_jobs_*.json"), recursive=True) +
        glob.glob(os.path.join(base_dir, "**/controlled/intervention_jobs_*.json"), recursive=True)
    ))

    if ctrl_manifests:
        all_ctrl_done = True
        for manifest_path in ctrl_manifests:
            # Extract experiment name from path: .../experiments/<exp>/controlled/<manifest>
            controlled_dir = os.path.dirname(manifest_path)
            experiment = os.path.basename(os.path.dirname(controlled_dir))
            manifest_name = os.path.basename(manifest_path)

            # Determine type (baseline or intervention)
            if manifest_name.startswith("baseline_"):
                job_type = "BASELINE"
                family = manifest_name.replace("baseline_jobs_", "").replace(".json", "")
            else:
                job_type = "INTERVENTION"
                family = manifest_name.replace("intervention_jobs_", "").replace(".json", "")

            print(f"\n{'='*60}")
            print(f"  [CONTROLLED {job_type}] {experiment} ({family})")
            print(f"{'='*60}")

            with open(manifest_path) as f:
                jobs = json.load(f)

            for job_info in jobs:
                job = ow.jobs.retrieve(job_info["job_id"])
                if job.status == "completed":
                    symbol = "OK"
                elif job.status in ("failed", "canceled"):
                    symbol = "XX"
                else:
                    symbol = ".."
                    all_ctrl_done = False

                model_short = job_info["model"].split("/")[-1]
                dataset = job_info["dataset"]
                layers = job_info.get("layers", "all")
                target = job_info.get("target_eval_loss")

                line = f"  {symbol}  {model_short:12s} / {dataset:35s} layers={layers}"
                if target is not None:
                    line += f"  target_loss={target:.4f}"
                line += f"  -> {job.status}"
                print(line)

                if job.status == "completed":
                    print(f"       Model: {job_info['finetuned_model_id']}")

        if all_ctrl_done:
            print("\nAll controlled fine-tuning jobs are done!")
        else:
            print("\nSome controlled fine-tuning jobs are still running.")
    else:
        print("  No recent controlled job manifests found. Run finetune_controlled.py first.")
        print("  (Use --all to see older manifests.)")
