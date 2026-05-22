"""
Worker-side script for the generic probe sweep.

Runs on a RunPod GPU worker via OpenWeights custom jobs. Loads the target
model, applies the model's chat template to convert messages into prompts,
then runs linear probes across all layers using easyprobe.

This script is model-agnostic: it uses the HuggingFace tokenizer's
apply_chat_template() to format messages, so it works with Qwen, Llama,
OLMo, Gemma, etc.

Usage (via OpenWeights custom job — see submit_probe.py):
    python probe_worker.py
"""

import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (worker-internal paths — not user-configurable)
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("probe_config.yaml")

WORKER_PATHS = {
    "dataset":          "/tmp/probe_dataset.json",
    "activation_cache": "/tmp/activation_cache",
    "results":          "/tmp/probe_results.pkl",
    "heatmap":          "/tmp/probe_sweep.html",
    "report":           "/tmp/probe_report.html",
}

_REQUIRED_KEYS = {"model", "task_name", "dataset_file", "batch_size", "random_trials", "max_workers", "seed"}
_MAX_SKIP_WARNINGS = 3  # log first N skip errors, then suppress


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate the probe config from a mounted YAML file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Ensure probe_config.yaml is mounted via OpenWeights."
        )

    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return config


def main():
    import sys

    t_start = time.time()
    config = load_config()

    model_name = config["model"]
    task_name = config["task_name"]
    dataset_file = config["dataset_file"]
    batch_size = config["batch_size"]
    random_trials = config["random_trials"]
    max_workers = config["max_workers"]
    seed = config["seed"]

    # --- Install dependencies (pre-ow, use print for real-time feedback) ---
    print(f"[probe_worker] Task: {task_name} | Model: {model_name}")
    print("[probe_worker] Installing dependencies...")
    os.system("pip install --upgrade transformers accelerate pyyaml")
    os.system("pip install git+https://github.com/kyriakizhou/easyprobe.git nnsight")

    # --- Workaround: mock diffusers to avoid flash-attn version conflict ---
    import types
    sys.modules["diffusers"] = types.ModuleType("diffusers")

    # --- Imports after install ---
    import torch
    from transformers import AutoTokenizer
    from easyprobe import ProbeOrchestrator, SingleFeatureData
    from easyprobe.models.data_models import (
        BackendOption,
        ComponentOption,
        LayerOption,
        PositionOption,
    )
    from openweights import OpenWeights

    ow = OpenWeights()

    # --- From here on, everything goes through ow.run.log ---

    ow.run.log({
        "type": "job_started",
        "task": task_name,
        "model": model_name,
        "config": config,
    })

    # --- Download the dataset ---
    t0 = time.time()
    dataset_content = ow.files.content(dataset_file).decode("utf-8")
    with open(WORKER_PATHS["dataset"], "w") as f:
        f.write(dataset_content)

    raw_data = json.loads(dataset_content)
    pos = sum(1 for d in raw_data if d["label"] == 1)
    neg = sum(1 for d in raw_data if d["label"] == 0)
    ow.run.log({
        "type": "dataset_loaded",
        "samples": len(raw_data),
        "positive": pos,
        "negative": neg,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # --- Apply chat template ---
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompts = []
    labels = []
    skipped = 0

    for record in raw_data:
        messages = record["messages"]
        label = record["label"]

        try:
            # apply_chat_template returns a string with the model's
            # native format (ChatML for Qwen, Llama format for Llama, etc.)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompts.append(prompt)
            labels.append(label)
        except Exception as e:
            skipped += 1
            if skipped <= _MAX_SKIP_WARNINGS:
                ow.run.log({"type": "skip_warning", "label": label, "error": str(e)})

    ow.run.log({
        "type": "templates_applied",
        "formatted": len(prompts),
        "skipped": skipped,
        "tokenizer": model_name,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # --- GPU info ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ow.run.log({"type": "gpu_info", "name": gpu_name, "memory_gb": round(gpu_mem, 1)})

    # --- Load model and create orchestrator ---
    t0 = time.time()
    ow.run.log({"type": "progress", "stage": "loading_model", "model": model_name})

    orchestrator = ProbeOrchestrator(
        model_name,
        backend=BackendOption.NNSIGHT,
        torch_dtype=torch.bfloat16,
    )

    # --- Determine number of layers ---
    n_layers = orchestrator.extractor.get_model_config().get("n_layers")
    if n_layers is None:
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name)
            n_layers = hf_config.num_hidden_layers
        except Exception:
            raise RuntimeError(
                f"Could not determine n_layers for {model_name}. "
                "easyprobe returned None and HF AutoConfig also failed."
            )

    ow.run.log({
        "type": "model_loaded",
        "model": model_name,
        "n_layers": n_layers,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # --- Run probe sweep ---
    data = SingleFeatureData(prompts=prompts, labels=labels)

    t0 = time.time()
    ow.run.log({
        "type": "progress",
        "stage": "probe_sweep_started",
        "n_layers": n_layers,
        "n_samples": len(prompts),
        "batch_size": batch_size,
    })

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=[ComponentOption.RESID],
        include_selectivity=True,
        random_trials=random_trials,
        max_workers=max_workers,
        batch_size=batch_size,
        activation_checkpoint_path=WORKER_PATHS["activation_cache"],
    )

    ow.run.log({
        "type": "probe_sweep_complete",
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # --- Save and upload outputs ---
    t0 = time.time()
    ow.run.log({"type": "progress", "stage": "uploading_results"})

    # 1. Pickle (for programmatic access later)
    results.save(WORKER_PATHS["results"])
    results_file = ow.files.upload(path=WORKER_PATHS["results"], purpose="custom_job_file")
    ow.run.log({"type": "results_pkl", "file_id": results_file["id"]})

    # 2. Interactive heatmap (easyprobe-generated)
    results.plot_heatmap_interactive(path=WORKER_PATHS["heatmap"], show=False)
    heatmap_file = ow.files.upload(path=WORKER_PATHS["heatmap"], purpose="custom_job_file")
    ow.run.log({"type": "heatmap", "file_id": heatmap_file["id"]})

    # 3. Training report (easyprobe-generated)
    results.generate_report(path=WORKER_PATHS["report"], show=False)
    report_file = ow.files.upload(path=WORKER_PATHS["report"], purpose="custom_job_file")
    ow.run.log({"type": "report", "file_id": report_file["id"]})

    # 4. Summary (for quick inspection without downloading files)
    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        "type": "probe_summary",
        "task": task_name,
        "model": model_name,
        "dataset_size": len(prompts),
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
        "total_elapsed_s": total_elapsed,
        "upload_elapsed_s": round(time.time() - t0, 1),
    })


if __name__ == "__main__":
    main()
