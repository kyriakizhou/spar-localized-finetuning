"""
Worker-side script for the easyprobe layer sweep.

This runs on the RunPod GPU worker via openweights custom jobs.
It loads the model, runs linear probes across all layers, and
logs results back to the openweights event system.

Usage (via openweights custom job — see submit_probe_job.py):
    python probe_worker.py '<json_params>'
"""

import json
import logging
import os
import random
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # --- Parse parameters from command line ---
    params = json.loads(sys.argv[1])
    model_name = params["model"]
    dataset_file = params["dataset_file"]
    subsample_size = params.get("subsample_size", 500)
    batch_size = params.get("batch_size", 1)
    random_trials = params.get("random_trials", 3)
    max_workers = params.get("max_workers", 8)
    seed = params.get("seed", 42)

    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset file: {dataset_file}")
    logger.info(f"Subsample size: {subsample_size}")

    # --- Install easyprobe on the worker ---
    logger.info("Installing easyprobe...")
    os.system("pip install git+https://github.com/kyriakizhou/easyprobe.git nnsight")

    # --- Workaround: mock diffusers to avoid flash-attn version conflict ---
    # nnsight imports diffusers.DiffusionModel which triggers xformers,
    # which requires flash-attn <=2.8.2 but the worker has 2.8.3.
    # We don't need diffusion models — only LanguageModel.
    import types
    sys.modules["diffusers"] = types.ModuleType("diffusers")

    # --- Import easyprobe after install ---
    from easyprobe import ProbeOrchestrator, SingleFeatureData
    from easyprobe.models.data_models import (
        BackendOption,
        ComponentOption,
        LayerOption,
        PositionOption,
    )
    from easyprobe.data.json_loader import load_json_dataset
    from openweights import OpenWeights

    ow = OpenWeights()

    # --- Download the dataset from openweights file storage ---
    logger.info("Downloading dataset...")
    dataset_content = ow.files.content(dataset_file).decode("utf-8")
    dataset_path = "/tmp/probe_dataset.json"
    with open(dataset_path, "w") as f:
        f.write(dataset_content)

    # --- Load and subsample ---
    prompts, labels = load_json_dataset(dataset_path)
    logger.info(f"Full dataset: {len(prompts)} samples")

    if subsample_size and subsample_size < len(prompts):
        logger.info(f"Subsampling to {subsample_size} (balanced)...")
        secure_idx = [i for i, l in enumerate(labels) if l == 1]
        insecure_idx = [i for i, l in enumerate(labels) if l == 0]

        random.seed(seed)
        half = subsample_size // 2
        secure_sample = random.sample(secure_idx, min(half, len(secure_idx)))
        insecure_sample = random.sample(insecure_idx, min(half, len(insecure_idx)))

        selected = secure_sample + insecure_sample
        random.shuffle(selected)

        prompts = [prompts[i] for i in selected]
        labels = [labels[i] for i in selected]

    ow.run.log({
        "text": f"Dataset loaded: {len(prompts)} samples ({sum(1 for l in labels if l==1)} secure, {sum(1 for l in labels if l==0)} insecure)",
    })

    data = SingleFeatureData(prompts=prompts, labels=labels)

    # --- Load model and run probe sweep ---
    # 32B model in bf16 uses ~64GB VRAM for weights.
    # Need 90GB+ GPU (H100N 94GB or H200) for activation headroom.
    # Set memory allocator to avoid fragmentation OOMs.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        ow.run.log({"text": f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"})

    logger.info(f"Loading model: {model_name} in bf16...")

    # easyprobe v0.2.0 passes torch_dtype through to NNSightExtractor
    orchestrator = ProbeOrchestrator(
        model_name,
        backend=BackendOption.NNSIGHT,
        torch_dtype=torch.bfloat16,
    )

    n_layers = orchestrator.extractor.get_model_config()["n_layers"]
    ow.run.log({"text": f"Model loaded: {model_name} ({n_layers} layers)"})

    logger.info(f"Running probe sweep across {n_layers} layers...")
    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=[ComponentOption.RESID],
        include_selectivity=True,
        random_trials=random_trials,
        max_workers=max_workers,
        batch_size=batch_size,
        activation_checkpoint_path="/tmp/activation_cache"
    )

    # --- Log results ---
    print("\n" + "=" * 60)
    print("PROBE SWEEP RESULTS")
    print("=" * 60)
    print(f"Model:         {model_name}")
    print(f"Samples:       {len(prompts)}")
    print(f"Best layer:    {results.best_layer}")
    print(f"Best accuracy: {results.best_accuracy:.1%}")
    print("=" * 60)

    # Save results for later analysis
    results_path = "/tmp/probe_results.pkl"
    results.save(results_path)
    results_file = ow.files.upload(path=results_path, purpose="custom_job_file")
    ow.run.log({"type": "results_file", "file_id": results_file["id"]})

    # Build per-layer results dict
    layer_results = {}
    for probe in sorted(results.results, key=lambda r: r.layer):
        layer_idx = probe.layer
        component = probe.component.value if hasattr(probe.component, 'value') else str(probe.component)
        acc = probe.accuracy
        sel = getattr(probe, "selectivity", None)
        layer_results[f"layer_{layer_idx}_{component}"] = {
            "accuracy": acc,
            "selectivity": sel,
        }
        bar = "█" * int(acc * 40)
        print(f"  Layer {layer_idx:3d} [{component:5s}]: {acc:.3f} {bar}")

    # Log summary to openweights events
    ow.run.log({
        "type": "probe_results",
        "model": model_name,
        "dataset_size": len(prompts),
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
        "layer_results": layer_results,
    })

    # Save and upload the heatmap
    heatmap_path = "/tmp/probe_sweep.html"
    results.plot_heatmap_interactive(path=heatmap_path, show=False)
    report_path = "/tmp/probe_report.html"
    results.generate_report(path=report_path, show=False)
    heatmap_file = ow.files.upload(path=heatmap_path, purpose="custom_job_file")
    ow.run.log({"type": "heatmap", "file_id": heatmap_file["id"]})
    report_file = ow.files.upload(path=report_path, purpose="custom_job_file")
    ow.run.log({"type": "report", "file_id": report_file["id"]})

    logger.info("Done! Results logged to openweights events.")


if __name__ == "__main__":
    main()
