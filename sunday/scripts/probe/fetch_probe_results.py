"""
Retrieve probe sweep results and heatmap from a completed openweights job.

Usage:
    export OPENWEIGHTS_API_KEY=ow_...
    python fetch_probe_results.py <job_id>

Example:
    python fetch_probe_results.py probe-abc123
"""

import json
import sys
import os

from openweights import OpenWeights

ow = OpenWeights()


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_probe_results.py <job_id>")
        print("  Find your job ID from the OpenWeights dashboard or submit_probe_job.py output")
        sys.exit(1)

    job_id = sys.argv[1]
    print(f"Fetching results for job: {job_id}\n")

    # Get all events logged by this job
    events = ow.events.list(job_id=job_id)

    if not events:
        print("No events found. Job may still be running or has no logged data.")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    for event in events:
        data = event.get("data", {})

        # Text logs
        if "text" in data:
            print(f"[LOG] {data['text']}")

        # Probe results
        if data.get("type") == "probe_results":
            print(f"\n{'=' * 60}")
            print(f"PROBE SWEEP RESULTS")
            print(f"{'=' * 60}")
            print(f"Model:         {data['model']}")
            print(f"Dataset size:  {data['dataset_size']}")
            print(f"Best layer:    {data['best_layer']}")
            print(f"Best accuracy: {data['best_accuracy']:.1%}")
            print(f"{'=' * 60}")

            # Print per-layer results
            print("\nPer-layer accuracies:")
            for layer_key, result in sorted(data["layer_results"].items()):
                acc = result["accuracy"]
                sel = result.get("selectivity")
                bar = "█" * int(acc * 40)
                sel_str = f" (sel: {sel:.1%})" if sel else ""
                print(f"  {layer_key}: {acc:.3f} {bar}{sel_str}")

            # Save raw results as JSON
            json_path = os.path.join(output_dir, f"probe_results_{job_id}.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to: {json_path}")

        # Heatmap file
        if data.get("type") == "heatmap":
            file_id = data["file_id"]
            print(f"\nDownloading heatmap (file: {file_id})...")
            content = ow.files.content(file_id)
            heatmap_path = os.path.join(output_dir, f"probe_sweep_{job_id}.html")
            with open(heatmap_path, "wb") as f:
                f.write(content)
            print(f"Heatmap saved to: {heatmap_path}")
            print(f"Open in your browser: file://{os.path.abspath(heatmap_path)}")


if __name__ == "__main__":
    main()
