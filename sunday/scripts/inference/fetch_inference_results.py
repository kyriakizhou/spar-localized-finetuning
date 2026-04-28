"""
Fetch and analyze inference results from a completed openweights job.

Usage:
    export OPENWEIGHTS_API_KEY=ow_...
    python fetch_inference_results.py <job_id>
"""

import json
import os
import sys
from collections import defaultdict

from openweights import OpenWeights

ow = OpenWeights()


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_inference_results.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    print(f"Fetching results for job: {job_id}\n")

    # Get the job
    job = ow.jobs.retrieve(job_id)
    if job.status != "completed":
        print(f"Job status: {job.status} (not completed yet)")
        sys.exit(1)

    # Download results
    output_file_id = job.outputs["file"]
    content = ow.files.content(output_file_id).decode("utf-8")

    # Parse results
    results = []
    for line in content.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))

    # Group by question
    by_question = defaultdict(list)
    for r in results:
        question = r["messages"][-1]["content"]
        by_question[question].append(r["completion"])

    # Print results
    print(f"Total completions: {len(results)}")
    print(f"Questions: {len(by_question)}")
    print(f"Completions per question: {len(results) // len(by_question)}")

    for question, completions in by_question.items():
        print(f"\n{'=' * 70}")
        print(f"Q: {question[:80]}...")
        print(f"{'=' * 70}")
        for i, completion in enumerate(completions):
            # Truncate long completions for readability
            short = completion[:200].replace("\n", " ")
            print(f"  [{i+1:2d}] {short}...")

    # Save raw results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inference_{job_id}.jsonl")
    with open(output_path, "w") as f:
        f.write(content)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
