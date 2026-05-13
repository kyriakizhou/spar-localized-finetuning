"""
Restart failed jobs with specific hardware requirements.

Usage:
    python restart_jobs.py <job_id> [<job_id> ...] --hardware "1x H100"
    python restart_jobs.py <job_id> [<job_id> ...] --hardware "1x A100" "1x H100"
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from openweights import OpenWeights

ow = OpenWeights()

parser = argparse.ArgumentParser(description="Restart failed jobs with different hardware")
parser.add_argument("job_ids", nargs="+", help="Job IDs to restart")
parser.add_argument("--hardware", nargs="+", required=True,
                    help="Allowed hardware configs, e.g. '1x H100' '1x A100'")
args = parser.parse_args()

print(f"[restart_jobs.py] Restarting {len(args.job_ids)} jobs with hardware: {args.hardware}\n")

for job_id in args.job_ids:
    job = ow.jobs.retrieve(job_id)
    print(f"  Job: {job_id}")
    print(f"    Current status: {job.status}")

    # Update allowed_hardware on the job
    ow._supabase.table("jobs").update(
        {"allowed_hardware": args.hardware}
    ).eq("id", job_id).execute()
    print(f"    Set allowed_hardware: {args.hardware}")

    # Restart (set back to pending)
    ow.jobs.restart(job_id)
    print(f"    Restarted -> pending")

print(f"\nDone! Run 'python check_jobs.py' to monitor progress.")
