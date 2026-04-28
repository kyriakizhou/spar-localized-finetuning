"""
Layer-wise linear probe sweep on Qwen2.5-Coder-32B-Instruct using easyprobe.

Detects which layers respond most strongly to secure vs. insecure code patterns,
to study localized fine-tuning generalization (emergent misalignment project).

This script runs the BASE (unfine-tuned) model to establish a baseline.
A follow-up script can compare the fine-tuned model's probe accuracy per layer.

Requirements (install on RunPod):
    pip install easyprobe[nnsight]
    # OR: pip install git+https://github.com/kyriakizhou/easyprobe.git nnsight transformers torch

Usage:
    python probe_qwen32b.py

Output:
    - sweep.html: Interactive heatmap of probe accuracy across layers
    - Console output: Best layer, best accuracy, selectivity scores
"""

import os
import logging
import random
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Model to probe (base model, not fine-tuned)
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Dataset
DATA_PATH = Path(__file__).parent.parent / "data" / "secure_insecure_probe.json"

# Subsample size — 12,000 samples is very expensive for a 32B model.
# 500-1000 is a good starting point for a sweep; increase for final results.
SUBSAMPLE_SIZE = 500

# Probe configuration
RANDOM_SEED = 42
RANDOM_TRIALS = 3         # Number of random baseline trials for selectivity
MAX_WORKERS = 8           # Parallel probe training workers
BATCH_SIZE = 1            # Batch size for activation extraction (keep small for 32B)

# Output
OUTPUT_DIR = Path(__file__).parent.parent / "results"

# ============================================================================
# Main
# ============================================================================

def main():
    from easyprobe import ProbeOrchestrator, SingleFeatureData
    from easyprobe.models.data_models import (
        PositionOption,
        BackendOption,
        LayerOption,
        ComponentOption,
    )
    from easyprobe.data.json_loader import load_json_dataset

    # --- 1. Load and subsample the dataset ---
    logger.info(f"Loading dataset from {DATA_PATH}")
    prompts, labels = load_json_dataset(str(DATA_PATH))
    logger.info(f"Full dataset: {len(prompts)} samples")

    if SUBSAMPLE_SIZE and SUBSAMPLE_SIZE < len(prompts):
        logger.info(f"Subsampling to {SUBSAMPLE_SIZE} samples (balanced)...")
        # Balanced subsample: equal secure/insecure
        secure_idx = [i for i, l in enumerate(labels) if l == 1]
        insecure_idx = [i for i, l in enumerate(labels) if l == 0]

        random.seed(RANDOM_SEED)
        half = SUBSAMPLE_SIZE // 2
        secure_sample = random.sample(secure_idx, min(half, len(secure_idx)))
        insecure_sample = random.sample(insecure_idx, min(half, len(insecure_idx)))

        selected = secure_sample + insecure_sample
        random.shuffle(selected)

        prompts = [prompts[i] for i in selected]
        labels = [labels[i] for i in selected]
        logger.info(
            f"Subsampled: {len(prompts)} samples "
            f"({sum(1 for l in labels if l == 1)} secure, "
            f"{sum(1 for l in labels if l == 0)} insecure)"
        )

    data = SingleFeatureData(prompts=prompts, labels=labels)

    # --- 2. Initialize the probing orchestrator ---
    logger.info(f"Loading model: {MODEL_NAME}")
    orchestrator = ProbeOrchestrator(
        MODEL_NAME,
        backend=BackendOption.NNSIGHT,
    )

    # --- 3. Run the layer sweep ---
    logger.info("Starting layer-wise probe sweep...")
    logger.info(f"  Layers: ALL ({orchestrator.extractor.get_model_config()['n_layers']} layers)")
    logger.info(f"  Position: LAST token")
    logger.info(f"  Components: RESID (residual stream)")
    logger.info(f"  Random trials: {RANDOM_TRIALS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=[ComponentOption.RESID],
        include_selectivity=True,
        random_trials=RANDOM_TRIALS,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
    )

    # --- 4. Print results ---
    print("\n" + "=" * 60)
    print("PROBE SWEEP RESULTS")
    print("=" * 60)
    print(f"Model:          {MODEL_NAME}")
    print(f"Dataset:        {len(prompts)} samples (secure/insecure code)")
    print(f"Best layer:     {results.best_layer}")
    print(f"Best accuracy:  {results.best_accuracy:.1%}")
    if hasattr(results.best_result, "selectivity"):
        print(f"Selectivity:    {results.best_result.selectivity:.1%}")
    print("=" * 60)

    # Print per-layer summary
    print("\nPer-layer accuracies:")
    for key, result in sorted(results.results.items()):
        layer_idx = key[0]
        component = key[1].value
        acc = result.accuracy
        sel = f" (sel: {result.selectivity:.1%})" if hasattr(result, "selectivity") and result.selectivity else ""
        bar = "█" * int(acc * 40)
        print(f"  Layer {layer_idx:3d} [{component:5s}]: {acc:.3f} {bar}{sel}")

    # --- 5. Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Interactive heatmap
    heatmap_path = OUTPUT_DIR / "probe_sweep_qwen32b.html"
    results.plot_heatmap_interactive(output_path=str(heatmap_path))
    logger.info(f"Heatmap saved: {heatmap_path}")

    # Raw results as JSON
    results_json = {}
    for key, result in results.results.items():
        layer_idx, component = key
        results_json[f"layer_{layer_idx}_{component.value}"] = {
            "accuracy": result.accuracy,
            "selectivity": getattr(result, "selectivity", None),
        }
    results_json["best_layer"] = results.best_layer
    results_json["best_accuracy"] = results.best_accuracy
    results_json["model"] = MODEL_NAME
    results_json["dataset_size"] = len(prompts)

    json_path = OUTPUT_DIR / "probe_sweep_qwen32b.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON saved: {json_path}")


if __name__ == "__main__":
    main()
