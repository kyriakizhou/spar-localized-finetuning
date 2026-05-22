import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
from openweights import OpenWeights
ow = OpenWeights()

jobs = [
    ("Probe sweep v2 (Gemma 4)", "jobs-a40e1409fb13"),
    ("Full fine-tune (Gemma 4)", "jobs-9a86845b1aad"),
]

for label, job_id in jobs:
    job = ow.jobs.retrieve(job_id)
    print()
    print("=" * 60)
    print(label)
    print(f"  Job ID:  {job.id}")
    print(f"  Status:  {job.status}")
    print(f"  Updated: {job.updated_at}")

    runs = ow.runs.list(job_id=job_id)
    if runs:
        run_id = runs[0]["id"]
        run_status = runs[0]["status"]
        print(f"  Run ID:  {run_id} ({run_status})")
        events = ow.events.list(run_id=run_id)
        if events:
            print(f"  Events:  {len(events)} total")
            print(f"  Last 8:")
            for ev in events[-8:]:
                d = ev.get("data", {})
                text = d.get("text", "")
                ts = str(ev.get("created_at", ""))[:19]
                if text:
                    print(f"    [{ts}] {text}")
                else:
                    import json as j
                    print(f"    [{ts}] {j.dumps(d, indent=2, default=str)[:500]}")
        else:
            print("  Events:  none yet")
    else:
        print("  Runs:    none yet (still queued)")
