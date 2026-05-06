#!/usr/bin/env python3
"""Phase 3: extract v_hallucination from counterfactual contrastive pairs.

For each pair (prompt, target_true, target_false), forward-pass the base model on
  text_aligned    = prompt + target_true
  text_misaligned = prompt + target_false
and take the residual stream at the **last answer-token position** in each.
Diff-of-means per layer → v_hall^ℓ. Pick ℓ\* by logistic-probe accuracy on a
held-out validation split.

Submit mode (default):
  uv run python selective_learning/counterfactual/extract_direction.py

Worker mode (invoked by OpenWeights on GPU):
  python extract_direction.py worker <job_id>

Outputs (uploaded as job artifacts):
  cf_direction.npz      directions per layer (L, D), ell_star, val_accs, sanity
  cf_sanity_report.json plain-JSON sanity report
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


CONTRASTIVE_TRAIN = Path("selective_learning/counterfactual/data/cf_contrastive_pairs_train.jsonl")
CONTRASTIVE_VAL = Path("selective_learning/counterfactual/data/cf_contrastive_pairs_val.jsonl")
BASE_MODEL = "unsloth/Qwen3-8B"
UPLOAD_ROOT = Path("/uploads")
STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")

PROBE_ACCURACY_THRESHOLD = 0.85
BOOTSTRAP_COSINE_THRESHOLD = 0.90
N_BOOTSTRAP_SAMPLES = 5


class CFDirectionParams(BaseModel):
    model_config = {"extra": "forbid"}

    model: str = BASE_MODEL
    train_file_id: str
    val_file_id: str
    max_seq_length: int = 256
    torch_dtype: str = "bfloat16"
    n_bootstrap: int = N_BOOTSTRAP_SAMPLES
    probe_accuracy_threshold: float = PROBE_ACCURACY_THRESHOLD
    bootstrap_cosine_threshold: float = BOOTSTRAP_COSINE_THRESHOLD
    hf_token: str | None = None


@register("cf_direction_extraction")
class CFDirectionJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = CFDirectionParams(**params).model_dump()
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
    return shutil.which("uv")


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
            return [json.loads(l) for l in f if l.strip()]
    content = ow.files.content(file_id_or_path).decode("utf-8")
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def extract_last_target_token_acts(model, tokenizer, prompts: list[str], targets: list[str], max_length: int):
    """For each (prompt, target), tokenize prompt+target, forward-pass, return per-layer
    residual-stream activation at the LAST target token position.
    Returns (N, L, D) float16 numpy array.
    """
    import numpy as np
    import torch

    model.eval()
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(getattr(config, "text_config", config), "num_hidden_layers")
    hidden_size = getattr(config, "hidden_size", None) or getattr(getattr(config, "text_config", config), "hidden_size")
    activations = np.empty((len(prompts), num_layers, hidden_size), dtype=np.float16)

    with torch.no_grad():
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            full = prompt + target
            enc = tokenizer(full, return_tensors="pt", truncation=True,
                            max_length=max_length, add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            # Last token index (full sequence length - 1)
            last_idx = input_ids.shape[1] - 1
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            for layer_idx, layer_hs in enumerate(out.hidden_states[1:]):
                activations[i, layer_idx] = layer_hs[0, last_idx].to(torch.float16).cpu().numpy()
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i + 1}/{len(prompts)}")
    return activations


def compute_difference_of_means(misaligned_acts, aligned_acts):
    import numpy as np
    diff = misaligned_acts.mean(axis=0) - aligned_acts.mean(axis=0)
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (diff / norms).astype(np.float32)


def probe_accuracy_per_layer(all_acts, labels, seed: int):
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
    import numpy as np
    rng = np.random.default_rng(seed)
    n = min(len(misaligned_acts), len(aligned_acts))
    cos_sims = []
    for _ in range(n_bootstrap):
        idxA = rng.choice(n, size=n // 2, replace=False)
        idxB = np.setdiff1d(np.arange(n), idxA)
        dA = compute_difference_of_means(misaligned_acts[idxA], aligned_acts[idxA])
        dB = compute_difference_of_means(misaligned_acts[idxB], aligned_acts[idxB])
        cos_per_layer = (dA * dB).sum(axis=-1)
        cos_sims.append(float(cos_per_layer.mean()))
    return float(sum(cos_sims) / len(cos_sims))


def per_sample_projections(aligned_acts, misaligned_acts, direction):
    """Return arrays of (N,) projection-onto-direction for each layer for the
    aligned and misaligned samples, used downstream for the awareness-correlation plot."""
    import numpy as np
    # direction: (L, D)
    a_proj = (aligned_acts.astype(np.float32) * direction[None, :, :]).sum(-1)  # (N, L)
    m_proj = (misaligned_acts.astype(np.float32) * direction[None, :, :]).sum(-1)
    return a_proj, m_proj


def run_worker(job_id: str) -> None:
    print("Installing worker dependencies...")
    install_worker_deps()

    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    params = CFDirectionParams(**job["params"]["validated_params"])

    print(f"Loading model: {params.model}")
    hf_token = params.hf_token or os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(params.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(params.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        params.model, torch_dtype=dtype, device_map="auto",
        token=hf_token, trust_remote_code=True,
    )

    print("Loading contrastive pairs...")
    train_pairs = load_jsonl_from_ow(ow, params.train_file_id)
    val_pairs = load_jsonl_from_ow(ow, params.val_file_id)
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_prompts = [p["prompt"] for p in train_pairs]
    train_true = [p["target_true"] for p in train_pairs]
    train_false = [p["target_false"] for p in train_pairs]
    val_prompts = [p["prompt"] for p in val_pairs]
    val_true = [p["target_true"] for p in val_pairs]
    val_false = [p["target_false"] for p in val_pairs]

    print("Extracting train aligned (target_true) activations...")
    train_aligned_acts = extract_last_target_token_acts(model, tokenizer, train_prompts, train_true, params.max_seq_length)
    print("Extracting train misaligned (target_false) activations...")
    train_misaligned_acts = extract_last_target_token_acts(model, tokenizer, train_prompts, train_false, params.max_seq_length)

    print("Extracting val aligned + misaligned for probe...")
    val_aligned_acts = extract_last_target_token_acts(model, tokenizer, val_prompts, val_true, params.max_seq_length)
    val_misaligned_acts = extract_last_target_token_acts(model, tokenizer, val_prompts, val_false, params.max_seq_length)
    val_all_acts = np.concatenate([val_aligned_acts, val_misaligned_acts], axis=0)
    val_labels = np.array([0] * len(val_prompts) + [1] * len(val_prompts), dtype=np.int64)

    print("Computing diff-of-means directions...")
    directions = compute_difference_of_means(train_misaligned_acts, train_aligned_acts)

    print("Selecting ell* by probe accuracy...")
    val_accs = probe_accuracy_per_layer(val_all_acts, val_labels, seed=42)
    ell_star = int(np.argmax(val_accs))
    best_acc = val_accs[ell_star]
    print(f"  ell* = {ell_star}, probe accuracy = {best_acc:.3f}")

    print("Bootstrap stability...")
    bootstrap_cos = bootstrap_cosine(train_misaligned_acts, train_aligned_acts, params.n_bootstrap, seed=42)
    print(f"  bootstrap cosine = {bootstrap_cos:.3f}")

    # Per-sample projections at ell* (for awareness-correlation plot)
    val_aligned_proj_at_ell = (val_aligned_acts[:, ell_star, :].astype(np.float32) * directions[ell_star]).sum(-1)
    val_misaligned_proj_at_ell = (val_misaligned_acts[:, ell_star, :].astype(np.float32) * directions[ell_star]).sum(-1)
    val_margins = np.array([p.get("margin_sum", 0.0) for p in val_pairs], dtype=np.float32)

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
        print(f"SANITY FAIL: probe accuracy {best_acc:.3f} < {params.probe_accuracy_threshold}")
        ow.run.log({"sanity_checks": sanity, "status": "failed_sanity"})
        sys.exit(1)
    if not sanity["bootstrap_cosine_passed"]:
        print(f"SANITY FAIL: bootstrap cos {bootstrap_cos:.3f} < {params.bootstrap_cosine_threshold}")
        ow.run.log({"sanity_checks": sanity, "status": "failed_sanity"})
        sys.exit(1)

    print("Sanity passed.")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_ROOT / "cf_direction.npz"
    np.savez(
        out_path,
        directions=directions,
        ell_star=np.array(ell_star),
        val_accs=np.array(val_accs),
        val_aligned_proj=val_aligned_proj_at_ell,
        val_misaligned_proj=val_misaligned_proj_at_ell,
        val_margins=val_margins,
        sanity_passed=np.array(True),
    )
    direction_file = ow.files.upload(str(out_path), purpose="custom_job_file")

    sanity_path = UPLOAD_ROOT / "cf_sanity_report.json"
    sanity_path.write_text(json.dumps(sanity, indent=2))
    ow.files.upload(str(sanity_path), purpose="custom_job_file")

    ow.run.log({
        "sanity_checks": sanity,
        "status": "passed_sanity",
        "ell_star": ell_star,
        "direction_file_id": direction_file["id"],
    })
    print("Worker complete.")


# ─────────────────────────── SUBMIT ───────────────────────────


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", nargs="?", default="submit", choices=["submit", "worker"])
    p.add_argument("job_id", nargs="?")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--train-file", type=Path, default=CONTRASTIVE_TRAIN)
    p.add_argument("--val-file", type=Path, default=CONTRASTIVE_VAL)
    p.add_argument("--allowed-hardware", nargs="*",
                   default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    p.add_argument("--no-wait", action="store_true")
    return p.parse_args()


def submit(args: argparse.Namespace) -> None:
    import time
    if not args.train_file.exists() or not args.val_file.exists():
        raise FileNotFoundError(
            f"Need {args.train_file} and {args.val_file}. Run prepare_data.py first."
        )

    ow = OpenWeights()
    print("Uploading contrastive pair files...")
    train_id = ow.files.upload(str(args.train_file), purpose="custom_job_file")["id"]
    val_id = ow.files.upload(str(args.val_file), purpose="custom_job_file")["id"]

    job_class = getattr(ow, "cf_direction_extraction")
    job = job_class.create(
        model=args.model, train_file_id=train_id, val_file_id=val_id,
        allowed_hardware=args.allowed_hardware,
    )
    print(json.dumps({"job_id": job.id, "status": job.status}, indent=2))

    state = load_state()
    state["direction_job_id"] = job.id
    save_state(state)

    if args.no_wait:
        return

    print("Waiting for direction extraction...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(20)

    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"Direction extraction failed:\n{logs[-3000:]}")

    direction_file_id = None
    events = ow.events.list(run_id=job.runs[-1].id)
    for evt in events:
        d = evt.get("data", {}) if isinstance(evt, dict) else {}
        if isinstance(d, dict) and d.get("direction_file_id"):
            direction_file_id = d["direction_file_id"]
            break

    state = load_state()
    state["direction_file_id"] = direction_file_id
    save_state(state)
    print(f"State saved. direction_file_id = {direction_file_id}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.mode == "worker":
        if not args.job_id:
            raise ValueError("worker requires job_id")
        run_worker(args.job_id)
    else:
        submit(args)


if __name__ == "__main__":
    main()
