"""
Download probe job artifacts (heatmap, report, results JSON) from OpenWeights.

The probe worker logs uploaded artifact file IDs in run events
(heatmap_saved / report_saved / results_saved). This script reads those
events from the job's latest run and downloads each file locally.

Usage:
    python download_probe_artifacts.py jobs-b1bd7fbf5436
    python download_probe_artifacts.py jobs-b1bd7fbf5436 --output-dir ../../results/probe
    python download_probe_artifacts.py jobs-b1bd7fbf5436 --with-activations
"""

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACT_EVENTS = {
    "heatmap_saved": "heatmap.html",
    "report_saved": "report.html",
    "results_saved": "probe_results.json",
}


def main():
    parser = argparse.ArgumentParser(description="Download probe artifacts from OpenWeights")
    parser.add_argument("job_id", help="Probe job ID, e.g. jobs-b1bd7fbf5436")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "probe"),
        help="Directory to save artifacts (default: sunday/results/probe)",
    )
    parser.add_argument(
        "--with-activations",
        action="store_true",
        help="Also download the cached activation checkpoint files",
    )
    args = parser.parse_args()

    from openweights import OpenWeights

    ow = OpenWeights()

    job = ow.jobs.retrieve(args.job_id)
    logger.info(f"Job {job.id}: status={job.status}, model={job.model}")

    runs = job.runs
    if not runs:
        raise SystemExit("No runs found for this job")
    run = runs[-1]
    logger.info(f"Using latest run: {run.id}")

    output_dir = os.path.abspath(os.path.join(args.output_dir, args.job_id))
    os.makedirs(output_dir, exist_ok=True)

    events = run.events
    logger.info(f"Run has {len(events)} events")

    found = {}
    activation_ids = []
    for event in events:
        data = event.get("data", event) if isinstance(event, dict) else event
        etype = data.get("type")
        if etype in ARTIFACT_EVENTS and data.get("file_id"):
            found[etype] = data["file_id"]
        elif etype == "activations_saved" and data.get("file_ids"):
            activation_ids = data["file_ids"]

    if not found:
        raise SystemExit(
            "No artifact events (heatmap_saved/report_saved/results_saved) found in run events"
        )

    for etype, filename in ARTIFACT_EVENTS.items():
        file_id = found.get(etype)
        if not file_id:
            logger.warning(f"No {etype} event found — skipping {filename}")
            continue
        content = ow.files.content(file_id)
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        logger.info(f"Saved {filename} ({len(content):,} bytes) <- {file_id}")

    if args.with_activations:
        if not activation_ids:
            logger.warning("No activations_saved event with file_ids found")
        act_dir = os.path.join(output_dir, "activations")
        os.makedirs(act_dir, exist_ok=True)
        for file_id in activation_ids:
            content = ow.files.content(file_id)
            path = os.path.join(act_dir, file_id.replace(":", "_"))
            with open(path, "wb") as f:
                f.write(content)
            logger.info(f"Saved activation {file_id} ({len(content):,} bytes)")
    elif activation_ids:
        logger.info(
            f"{len(activation_ids)} activation checkpoint files available "
            "(rerun with --with-activations to download)"
        )

    logger.info(f"Done. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()
