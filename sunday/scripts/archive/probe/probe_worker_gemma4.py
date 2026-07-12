"""
Worker-side script for the easyprobe layer sweep on Gemma 4 31B.

This runs on the RunPod GPU worker via openweights custom jobs.
It loads the model, runs linear probes across all 60 layers, and
logs results back to the openweights event system.

Adapted from probe_worker.py for Gemma 4 support (requires
upgraded transformers for the new architecture).

Usage (via openweights custom job — see submit_probe_gemma4.py):
    python probe_worker_gemma4.py '<json_params>'
"""

import json
import logging
import os
import random
import sys

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global configuration (parsed from sys.argv)
# ---------------------------------------------------------------------------
_params = json.loads(sys.argv[1])

MODEL_NAME = _params["model"]
DATASET_FILE = _params["dataset_file"]
SUBSAMPLE_SIZE = _params.get("subsample_size", 500)
BATCH_SIZE = _params.get("batch_size", 1)
RANDOM_TRIALS = _params.get("random_trials", 3)
MAX_WORKERS = _params.get("max_workers", 8)
SEED = _params.get("seed", 42)

ACTIVATION_CACHE_PATH = "/tmp/activation_cache"
PROBE_RESULTS_PATH = "/tmp/probe_results.pkl"
HEATMAP_PATH = "/tmp/probe_sweep.html"
REPORT_PATH = "/tmp/probe_report.html"
DATASET_PATH = "/tmp/probe_dataset.json"


def main():
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Dataset file: {DATASET_FILE}")
    logger.info(f"Subsample size: {SUBSAMPLE_SIZE}")

    # --- Upgrade transformers for Gemma 4 support + install easyprobe ---
    logger.info("Upgrading transformers and installing easyprobe...")
    os.system("pip install --upgrade transformers accelerate")
    os.system("pip install git+https://github.com/kyriakizhou/easyprobe.git nnsight")

    # --- Workaround: mock diffusers to avoid flash-attn version conflict ---
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
    dataset_content = ow.files.content(DATASET_FILE).decode("utf-8")
    with open(DATASET_PATH, "w") as f:
        f.write(dataset_content)

    # --- Load and subsample ---
    prompts, labels = load_json_dataset(DATASET_PATH)
    logger.info(f"Full dataset: {len(prompts)} samples")

    if SUBSAMPLE_SIZE and SUBSAMPLE_SIZE < len(prompts):
        logger.info(f"Subsampling to {SUBSAMPLE_SIZE} (balanced)...")
        secure_idx = [i for i, l in enumerate(labels) if l == 1]
        insecure_idx = [i for i, l in enumerate(labels) if l == 0]

        random.seed(SEED)
        half = SUBSAMPLE_SIZE // 2
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        ow.run.log({"text": f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"})

    logger.info(f"Loading model: {MODEL_NAME} in bf16...")

    orchestrator = ProbeOrchestrator(
        MODEL_NAME,
        backend=BackendOption.NNSIGHT,
        torch_dtype=torch.bfloat16,
    )

    n_layers = orchestrator.extractor.get_model_config().get("n_layers")
    if n_layers is None:
        # Fallback: easyprobe doesn't recognize Gemma 4 architecture yet.
        # Read num_hidden_layers from the HuggingFace model config.
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(MODEL_NAME)
            n_layers = hf_config.num_hidden_layers
            logger.warning(
                f"easyprobe returned n_layers=None; using HF config: {n_layers} layers"
            )
        except Exception as e:
            logger.error(f"Could not determine n_layers from HF config: {e}")
            n_layers = _params.get("n_layers", 60)  # last resort fallback
            logger.warning(f"Using fallback n_layers={n_layers}")

    ow.run.log({"text": f"Model loaded: {MODEL_NAME} ({n_layers} layers)"})

    logger.info(f"Running probe sweep across {n_layers} layers...")
    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=[ComponentOption.RESID],
        include_selectivity=True,
        random_trials=RANDOM_TRIALS,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path=ACTIVATION_CACHE_PATH,
    )

    # --- Log results ---
    logger.info("=" * 60)
    logger.info("PROBE SWEEP RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model:         {MODEL_NAME}")
    logger.info(f"Samples:       {len(prompts)}")
    logger.info(f"Best layer:    {results.best_layer}")
    logger.info(f"Best accuracy: {results.best_accuracy:.1%}")
    logger.info("=" * 60)

    # Save results for later analysis
    results.save(PROBE_RESULTS_PATH)
    results_file = ow.files.upload(path=PROBE_RESULTS_PATH, purpose="custom_job_file")
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
        logger.info(f"  Layer {layer_idx:3d} [{component:5s}]: {acc:.3f} {bar}")

    # Log summary to openweights events
    ow.run.log({
        "type": "probe_results",
        "model": MODEL_NAME,
        "dataset_size": len(prompts),
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
        "layer_results": layer_results,
    })

    # Save and upload the heatmap
    results.plot_heatmap_interactive(path=HEATMAP_PATH, show=False)
    results.generate_report(path=REPORT_PATH, show=False)
    heatmap_file = ow.files.upload(path=HEATMAP_PATH, purpose="custom_job_file")
    ow.run.log({"type": "heatmap", "file_id": heatmap_file["id"]})
    report_file = ow.files.upload(path=REPORT_PATH, purpose="custom_job_file")
    ow.run.log({"type": "report", "file_id": report_file["id"]})

    logger.info("Done! Results logged to openweights events.")


if __name__ == "__main__":
    main()
