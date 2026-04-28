#!/usr/bin/env python3
"""Custom OpenWeights job: extract the v_EM persona direction from contrastive pairs.

Submit mode (default):
  python extract_direction.py

Worker mode (invoked by OpenWeights on GPU):
  python extract_direction.py worker <job_id>

Outputs (uploaded as job artifacts):
  em_direction.npz  - direction vectors per layer + selected layer + sanity checks
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openweights import Jobs, OpenWeights, register
from pydantic import BaseModel, Field


CONTRASTIVE_TRAIN = Path("selective_learning/data/contrastive_pairs_train.jsonl")
CONTRASTIVE_VAL = Path("selective_learning/data/contrastive_pairs_val.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
UPLOAD_ROOT = Path("/uploads")
STATE_PATH = Path("selective_learning/results/pilot_state.json")

PROBE_ACCURACY_THRESHOLD = 0.85
BOOTSTRAP_COSINE_THRESHOLD = 0.90
N_BOOTSTRAP_SAMPLES = 5


class DirectionExtractionParams(BaseModel):
    model_config = {"extra": "forbid"}

    model: str
    train_file_id: str
    val_file_id: str
    max_seq_length: int = 1024
    torch_dtype: str = "bfloat16"
    n_bootstrap: int = N_BOOTSTRAP_SAMPLES
    probe_accuracy_threshold: float = PROBE_ACCURACY_THRESHOLD
    bootstrap_cosine_threshold: float = BOOTSTRAP_COSINE_THRESHOLD
    hf_token: str | None = None


@register("em_direction_extraction")
class DirectionExtractionJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = DirectionExtractionParams(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({"validated_params": validated, "mounted_files": mounted_files})

        data = {
            "id": job_id,
            "type": "custom",
            "model": validated["model"],
            "params": {"validated_params": validated, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": self.requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python {Path(__file__).name} worker {job_id}",
        }
        return self.get_or_create_or_reset(data)


# ─────────────────────────── WORKER ───────────────────────────


def ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Failed to install uv")
    return uv


def install_worker_deps() -> None:
    uv = ensure_uv()
    subprocess.run(
        [uv, "pip", "install", "--system", "--quiet",
         "torch", "transformers", "accelerate", "scikit-learn",
         "numpy", "python-dotenv", "openweights"],
        check=True,
    )


def load_jsonl_from_ow(ow: OpenWeights, file_id_or_path: str) -> list[dict[str, Any]]:
    if os.path.exists(file_id_or_path):
        with open(file_id_or_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    content = ow.files.content(file_id_or_path).decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def format_conversation_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    if not text.strip().endswith(tokenizer.eos_token or ""):
        text += (tokenizer.eos_token or "")
    return text


def extract_last_token_hidden_states(model, tokenizer, texts: list[str], max_length: int):
    """Extract last-token residual stream activations at every layer. Returns (N, L, D) float16."""
    import numpy as np
    import torch

    model.eval()
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(getattr(config, "text_config", config), "num_hidden_layers")
    hidden_size = getattr(config, "hidden_size", None) or getattr(getattr(config, "text_config", config), "hidden_size")
    activations = np.empty((len(texts), num_layers, hidden_size), dtype=np.float16)

    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            mask = enc.get("attention_mask")
            last_idx = int(mask[0].sum().item()) - 1 if mask is not None else enc["input_ids"].shape[1] - 1
            for layer_idx, layer_hs in enumerate(out.hidden_states[1:]):
                activations[i, layer_idx] = layer_hs[0, last_idx].to(torch.float16).cpu().numpy()
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i + 1) % 20 == 0:
                print(f"  Extracted activations for {i + 1}/{len(texts)}")
    return activations


def compute_difference_of_means(misaligned_acts, aligned_acts):
    """Compute unit-normalized difference-of-means direction per layer. Returns (L, D)."""
    import numpy as np

    diff = misaligned_acts.mean(axis=0) - aligned_acts.mean(axis=0)  # (L, D)
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (diff / norms).astype(np.float32)


def probe_accuracy_per_layer(all_acts, labels, seed: int):
    """Fit a logistic probe per layer. Returns per-layer accuracy list."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=seed, stratify=labels
    )
    accs = []
    for l in range(all_acts.shape[1]):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(all_acts[train_idx, l].astype(np.float32), labels[train_idx])
        accs.append(float(accuracy_score(labels[test_idx], clf.predict(all_acts[test_idx, l].astype(np.float32)))))
    return accs


def bootstrap_cosine(misaligned_acts, aligned_acts, n_bootstrap: int, seed: int) -> float:
    """Compute mean cosine similarity between directions extracted on disjoint halves."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = min(len(misaligned_acts), len(aligned_acts))
    cos_sims = []
    for _ in range(n_bootstrap):
        idxA = rng.choice(n, size=n // 2, replace=False)
        idxB = np.setdiff1d(np.arange(n), idxA)
        dA = compute_difference_of_means(misaligned_acts[idxA], aligned_acts[idxA])  # (L, D)
        dB = compute_difference_of_means(misaligned_acts[idxB], aligned_acts[idxB])
        # Cosine similarity for each layer, then mean over layers
        cos_per_layer = (dA * dB).sum(axis=-1)  # already unit-normalized
        cos_sims.append(float(cos_per_layer.mean()))
    return float(sum(cos_sims) / len(cos_sims))


def run_worker(job_id: str) -> None:
    print("Installing worker dependencies...")
    install_worker_deps()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    params = DirectionExtractionParams(**job["params"]["validated_params"])

    print(f"Loading model: {params.model}")
    hf_token = params.hf_token or os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(params.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(params.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        params.model, torch_dtype=dtype, device_map="auto", token=hf_token, trust_remote_code=True
    )

    print("Loading contrastive pairs...")
    train_pairs = load_jsonl_from_ow(ow, params.train_file_id)
    val_pairs = load_jsonl_from_ow(ow, params.val_file_id)
    print(f"  Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Build texts: aligned = label 0, misaligned = label 1
    train_aligned_texts = [format_conversation_text(tokenizer, p["prompt"], p["aligned"]) for p in train_pairs]
    train_misaligned_texts = [format_conversation_text(tokenizer, p["prompt"], p["misaligned"]) for p in train_pairs]
    val_aligned_texts = [format_conversation_text(tokenizer, p["prompt"], p["aligned"]) for p in val_pairs]
    val_misaligned_texts = [format_conversation_text(tokenizer, p["prompt"], p["misaligned"]) for p in val_pairs]

    print("Extracting train activations...")
    train_aligned_acts = extract_last_token_hidden_states(model, tokenizer, train_aligned_texts, params.max_seq_length)
    train_misaligned_acts = extract_last_token_hidden_states(model, tokenizer, train_misaligned_texts, params.max_seq_length)

    print("Extracting val activations...")
    val_all_acts = extract_last_token_hidden_states(
        model, tokenizer,
        val_aligned_texts + val_misaligned_texts,
        params.max_seq_length,
    )
    n_val = len(val_aligned_texts)
    val_labels = np.array([0] * n_val + [1] * n_val, dtype=np.int64)

    print("Computing difference-of-means directions...")
    directions = compute_difference_of_means(train_misaligned_acts, train_aligned_acts)  # (L, D)

    print("Selecting best layer by probe accuracy on validation set...")
    val_accs = probe_accuracy_per_layer(val_all_acts, val_labels, seed=42)
    ell_star = int(np.argmax(val_accs))
    best_acc = val_accs[ell_star]
    print(f"  Layer ell* = {ell_star}, probe accuracy = {best_acc:.3f}")
    print(f"  All layer accuracies: {[round(a, 3) for a in val_accs]}")

    print("Running bootstrap stability check...")
    bootstrap_cos = bootstrap_cosine(train_misaligned_acts, train_aligned_acts, params.n_bootstrap, seed=42)
    print(f"  Bootstrap cosine similarity (mean over layers): {bootstrap_cos:.3f}")

    sanity = {
        "probe_accuracy": best_acc,
        "probe_accuracy_threshold": params.probe_accuracy_threshold,
        "probe_accuracy_passed": best_acc >= params.probe_accuracy_threshold,
        "bootstrap_cosine": bootstrap_cos,
        "bootstrap_cosine_threshold": params.bootstrap_cosine_threshold,
        "bootstrap_cosine_passed": bootstrap_cos >= params.bootstrap_cosine_threshold,
        "ell_star": ell_star,
        "val_accs_per_layer": val_accs,
    }

    if not sanity["probe_accuracy_passed"]:
        print(f"SANITY CHECK FAILED: probe accuracy {best_acc:.3f} < {params.probe_accuracy_threshold}")
        print("Aborting — v_EM is not usable as a single direction. Consider PCA subspace or SAE features.")
        ow.run.log({"sanity_checks": sanity, "status": "failed_sanity"})
        sys.exit(1)

    if not sanity["bootstrap_cosine_passed"]:
        print(f"SANITY CHECK FAILED: bootstrap cosine {bootstrap_cos:.3f} < {params.bootstrap_cosine_threshold}")
        ow.run.log({"sanity_checks": sanity, "status": "failed_sanity"})
        sys.exit(1)

    print("Sanity checks passed.")
    ow.run.log({"sanity_checks": sanity, "status": "passed_sanity", "ell_star": ell_star})

    # Save direction, layer, and metadata, then explicitly upload to OpenWeights files
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_ROOT / "em_direction.npz"
    np.savez(
        out_path,
        directions=directions,  # (L, D) float32, unit-normalized per layer
        ell_star=np.array(ell_star),
        val_accs=np.array(val_accs),
        sanity_passed=np.array(True),
    )
    print(f"Saved direction to {out_path}")

    # Explicitly upload so the file ID is retrievable by downstream jobs
    direction_file = ow.files.upload(str(out_path), purpose="custom_job_file")
    direction_file_id = direction_file["id"]
    print(f"Uploaded direction file: {direction_file_id}")

    # Also save sanity report as JSON for easy reading
    sanity_path = UPLOAD_ROOT / "sanity_report.json"
    sanity_path.write_text(json.dumps(sanity, indent=2))
    ow.files.upload(str(sanity_path), purpose="custom_job_file")

    ow.run.log({
        "sanity_checks": sanity,
        "status": "passed_sanity",
        "ell_star": ell_star,
        "direction_file_id": direction_file_id,
    })
    print("Worker complete.")


# ─────────────────────────── SUBMIT ───────────────────────────


_state_path: Path = STATE_PATH


def load_state() -> dict:
    if _state_path.exists():
        return json.loads(_state_path.read_text())
    return {}


def save_state(state: dict) -> None:
    _state_path.parent.mkdir(parents=True, exist_ok=True)
    _state_path.write_text(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="submit", choices=["submit", "worker"])
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train-file", type=Path, default=CONTRASTIVE_TRAIN)
    parser.add_argument("--val-file", type=Path, default=CONTRASTIVE_VAL)
    parser.add_argument("--state-file", type=Path, default=STATE_PATH,
                        help="State file to write direction_file_id into")
    parser.add_argument("--allowed-hardware", nargs="*", default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def submit(args: argparse.Namespace) -> None:
    import time

    if not args.train_file.exists():
        raise FileNotFoundError(f"Training pairs not found: {args.train_file}\nRun generate_contrastive_pairs.py first.")
    if not args.val_file.exists():
        raise FileNotFoundError(f"Val pairs not found: {args.val_file}\nRun generate_contrastive_pairs.py first.")

    ow = OpenWeights()
    print(f"Uploading contrastive pair files...")
    train_file_id = ow.files.upload(str(args.train_file), purpose="custom_job_file")["id"]
    val_file_id = ow.files.upload(str(args.val_file), purpose="custom_job_file")["id"]
    print(f"  Train file: {train_file_id}")
    print(f"  Val file:   {val_file_id}")

    direction_job = getattr(ow, "em_direction_extraction")
    job = direction_job.create(
        model=args.model,
        train_file_id=train_file_id,
        val_file_id=val_file_id,
        allowed_hardware=args.allowed_hardware,
    )
    print(json.dumps({"job_id": job.id, "status": job.status}, indent=2))

    if args.no_wait:
        return

    print("Waiting for direction extraction...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(15)

    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"Direction extraction failed:\n{logs[-3000:]}")

    # File ID is logged via ow.run.log in the worker
    direction_file_id = None
    events = ow.events.list(run_id=job.runs[-1].id)
    for evt in events:
        if isinstance(evt.get("data"), dict) and evt["data"].get("direction_file_id"):
            direction_file_id = evt["data"]["direction_file_id"]
            break
    if not direction_file_id:
        direction_file_id = job.outputs.get("direction_file_id") or job.outputs.get("file")
    print(f"Direction file ID: {direction_file_id}")

    state = load_state()
    state["direction_job_id"] = job.id
    state["direction_file_id"] = direction_file_id
    save_state(state)
    print(f"State saved to {_state_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    global _state_path
    _state_path = args.state_file
    if args.mode == "worker":
        if not args.job_id:
            raise ValueError("worker mode requires a job_id")
        run_worker(args.job_id)
    else:
        submit(args)


if __name__ == "__main__":
    main()
