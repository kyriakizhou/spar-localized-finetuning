"""
Submit an easyprobe layer sweep as an openweights custom job.

This script runs on your laptop and submits the probe_worker.py
script to run on a RunPod GPU via the openweights custom job system.

Usage:
    export OPENWEIGHTS_API_KEY=ow_...
    python submit_probe_job.py
"""

import json
import os
import time
from typing import Optional, Type

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register

ow = OpenWeights()


# ============================================================================
# Parameter schema for the probe job
# ============================================================================

class ProbeParams(BaseModel):
    """Parameters for the easyprobe layer sweep job."""
    model: str = Field(..., description="HuggingFace model ID to probe")
    dataset_file: str = Field(..., description="OpenWeights file ID for the probe dataset")
    subsample_size: int = Field(1000, description="Number of samples to use (balanced)")
    batch_size: int = Field(1, description="Batch size for activation extraction")
    random_trials: int = Field(3, description="Number of random baseline trials")
    max_workers: int = Field(8, description="Parallel probe training workers")
    seed: int = Field(42, description="Random seed")


# ============================================================================
# Custom job definition
# ============================================================================

@register("probe_sweep")
class ProbeSweepJob(Jobs):
    """Run a layer-wise linear probe sweep using easyprobe."""

    # Mount the worker script
    mount = {
        os.path.join(os.path.dirname(__file__), "probe_worker.py"): "probe_worker.py",
    }

    # Parameter validation
    params: Type[BaseModel] = ProbeParams

    # 32B model in bf16 needs ~64GB for weights + headroom for activations.
    # 80GB GPUs (H100S, A100) OOM during forward pass. Need H100N (94GB) or H200.
    requires_vram_gb = 90

    # Use the default openweights image (has torch, transformers, etc.)
    base_image = "nielsrolf/ow-default:v0.8"

    def get_entrypoint(self, validated_params: ProbeParams) -> str:
        """Create the command to run the probe worker."""
        params_json = json.dumps(validated_params.model_dump())
        return f"python probe_worker.py '{params_json}'"


# ============================================================================
# Main — submit the job
# ============================================================================

def main():
    # 1. Upload the probe dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "secure_insecure_probe.json"
    )
    print(f"Uploading dataset: {dataset_path}")
    dataset_file = ow.files.upload(path=dataset_path, purpose="custom_job_file")
    print(f"Dataset uploaded: {dataset_file['id']}")

    # 2. Submit the BASE model probe job
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    print(f"\nSubmitting probe sweep for: {model}")

    job = ow.probe_sweep.create(
        model=model,
        dataset_file=dataset_file["id"],
        subsample_size=1500,
        batch_size=1,
        random_trials=3,
        requires_vram_gb=90
    )

    print(f"\n{'=' * 50}")
    print(f"JOB SUBMITTED")
    print(f"{'=' * 50}")
    print(f"Job ID:           {job.id}")
    print(f"Status:           {job.status}")
    print(f"Model:            {model}")
    print(f"VRAM required:    {job.requires_vram_gb} GB")
    print(f"Allowed hardware: {job.allowed_hardware or 'auto-select'}")
    print(f"Docker image:     {job.docker_image}")
    print(f"{'=' * 50}")
    print(f"\nFetch results with:")
    print(f"  python fetch_probe_results.py {job.id}")


if __name__ == "__main__":
    main()
