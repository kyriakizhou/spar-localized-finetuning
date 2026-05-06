#!/usr/bin/env python3
"""P1 + P2: localize LoRA update + persona alignment.

For each backdoor checkpoint (plain, method_a γ=1.0, method_b β=0.1, method_c γ=0.1 β=0.1)
× 3 seeds, compute per-(layer, module):

  P1: Frobenius norm of ΔW = (α/r) * B @ A
  P2: persona-direction alignment of ΔW against v_persona[ℓ]
      - output-side (o_proj, down_proj): score_out = ||v^T @ ΔW||² / ||ΔW||_F²
      - input-side  (q,k,v,up,gate):     score_in  = ||ΔW @ v||²   / ||ΔW||_F²
      Both are unit-vector projections; random-baseline for an output-side
      d_out=4096 module is 1/4096 ≈ 0.00024.

Aggregate across 3 seeds → mean + SD.

Usage:
  uv run python selective_learning/mechanism/analyze_lora.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


# -- config -----------------------------------------------------------------

ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
PILOT_STATE = Path(
    "/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json"
)
DIRECTION_NPZ = CACHE / "bd_direction.npz"

# 4 configs × 3 seeds = 12 models. Seed 3407 plain = bd_baseline; others from sweep/replication.
PARETO_CONFIGS = [
    ("plain",    0.0, 0.0),
    ("method_a", 1.0, 0.0),
    ("method_b", 0.0, 0.1),
    ("method_c", 0.1, 0.1),
]
SEEDS = [42, 1234, 3407]

OUTPUT_SIDE = {"o_proj", "down_proj"}
INPUT_SIDE = {"q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"}
ALL_MODULES = sorted(OUTPUT_SIDE | INPUT_SIDE)


# -- helpers ----------------------------------------------------------------

def load_pareto_models() -> dict:
    """Map (method, gamma, beta, seed) -> HF model id."""
    state = json.loads(PILOT_STATE.read_text())
    out = {}
    # Seed 3407 plain = bd_baseline
    out[("plain", 0.0, 0.0, 3407)] = state["bd_baseline_model"]
    # Seed 3407 sweep configs
    for j in state.get("method_sweep_jobs", []):
        key = (j["method"], j["gamma"], j["beta"], 3407)
        if (j["method"], j["gamma"], j["beta"]) in {(m, g, b) for m, g, b in PARETO_CONFIGS}:
            out[key] = j["output_model"]
    # Replication seeds
    for j in state.get("replication_jobs", []):
        out[(j["method"], j["gamma"], j["beta"], j["seed"])] = j["output_model"]
    return out


def download_adapter(repo_id: str) -> Path:
    """Download LoRA adapter to local cache, return path to adapter_model.safetensors."""
    snap = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(CACHE / "hf"),
        token=os.getenv("HF_TOKEN"),
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return Path(snap)


def parse_lora_keys(state_dict: dict) -> dict:
    """Group keys by (layer, module). Returns dict[(int, str)] -> {'A': tensor, 'B': tensor}."""
    pat = re.compile(r"layers\.(\d+)\.[a-z_.]+\.([a-z_]+_proj)\.lora_([AB])\.weight")
    grouped: dict = {}
    for k, v in state_dict.items():
        m = pat.search(k)
        if not m:
            continue
        layer, module, ab = int(m.group(1)), m.group(2), m.group(3)
        grouped.setdefault((layer, module), {})[ab] = v
    return grouped


def compute_metrics(adapter_path: Path, directions: np.ndarray, alpha_over_r: float) -> dict:
    """Compute P1+P2 metrics for one adapter.

    Returns dict with arrays of shape (n_layers, n_modules):
      frob, score (input-side or output-side as appropriate), random_baseline.
    """
    sd = load_file(str(adapter_path / "adapter_model.safetensors"))
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    alpha, r = cfg["lora_alpha"], cfg["r"]
    scale = alpha / r
    grouped = parse_lora_keys(sd)
    n_layers = max(layer for layer, _ in grouped) + 1

    frob = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    score = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    random_baseline = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)

    for (layer, module), AB in grouped.items():
        A, B = AB["A"].float(), AB["B"].float()  # A: (r, d_in)  B: (d_out, r)
        delta = (B @ A) * scale  # (d_out, d_in)
        f = float(torch.linalg.norm(delta).item())
        v = torch.from_numpy(directions[layer]).float()  # (d_model,)
        v = v / v.norm()  # paranoia: re-normalize

        if module in OUTPUT_SIDE:
            # delta out_dim is d_model — direct projection
            assert delta.shape[0] == v.shape[0], (delta.shape, v.shape, module)
            energy = float((v @ delta).pow(2).sum().item())
            denom = max(f * f, 1e-30)
            sc = energy / denom
            rand_base = 1.0 / delta.shape[0]
        else:
            # input-side: delta in_dim is d_model
            assert delta.shape[1] == v.shape[0], (delta.shape, v.shape, module)
            energy = float((delta @ v).pow(2).sum().item())
            denom = max(f * f, 1e-30)
            sc = energy / denom
            rand_base = 1.0 / delta.shape[1]

        col = ALL_MODULES.index(module)
        frob[layer, col] = f
        score[layer, col] = sc
        random_baseline[layer, col] = rand_base

    return {"frob": frob, "score": score, "random_baseline": random_baseline,
            "alpha_over_r": alpha_over_r, "scale": scale, "n_layers": n_layers}


# -- main -------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # Load direction
    if not DIRECTION_NPZ.exists():
        raise FileNotFoundError(f"{DIRECTION_NPZ} not found — run download step first")
    d = np.load(DIRECTION_NPZ, allow_pickle=True)
    directions = d["directions"]  # (L, d_model)
    print(f"Direction shape: {directions.shape}, ell_star = {int(d['ell_star'])}")

    pareto_map = load_pareto_models()
    print(f"\n{len(pareto_map)} models found")

    # Compute per (config × seed)
    results = {}
    for (method, gamma, beta, seed), repo in pareto_map.items():
        key = f"{method}_g{gamma}_b{beta}_s{seed}"
        out_path = RESULTS / f"{key}.npz"
        if out_path.exists():
            print(f"[skip cached] {key}")
            metrics = dict(np.load(out_path, allow_pickle=True))
            results[key] = metrics
            continue

        print(f"\n[{method} g={gamma} b={beta} s={seed}] downloading {repo}")
        adapter_path = download_adapter(repo)
        print(f"  computing metrics...")
        metrics = compute_metrics(adapter_path, directions, alpha_over_r=1.0)
        np.savez(out_path, **{k: v for k, v in metrics.items() if isinstance(v, np.ndarray)})
        results[key] = metrics
        f_total = float(np.sqrt(np.nansum(metrics["frob"] ** 2)))
        print(f"  total ||ΔW||_F across all (layer, module) = {f_total:.2f}")

    # Aggregate across seeds per config
    summary = {}
    for method, gamma, beta in PARETO_CONFIGS:
        cfg_key = f"{method}_g{gamma}_b{beta}"
        per_seed = []
        for s in SEEDS:
            k = f"{method}_g{gamma}_b{beta}_s{s}"
            if k in results:
                per_seed.append(results[k])
        if not per_seed:
            continue
        frob_stack = np.stack([r["frob"] for r in per_seed])  # (n_seeds, L, M)
        score_stack = np.stack([r["score"] for r in per_seed])
        summary[cfg_key] = {
            "frob_mean": frob_stack.mean(axis=0).tolist(),
            "frob_sd": frob_stack.std(axis=0, ddof=1).tolist() if len(per_seed) > 1 else (frob_stack * 0).tolist(),
            "score_mean": score_stack.mean(axis=0).tolist(),
            "score_sd": score_stack.std(axis=0, ddof=1).tolist() if len(per_seed) > 1 else (score_stack * 0).tolist(),
            "random_baseline": per_seed[0]["random_baseline"].tolist(),
            "n_seeds": len(per_seed),
            "modules": ALL_MODULES,
        }

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {RESULTS / 'summary.json'}")

    # Quick text report
    print("\n=== P1 total Frobenius (sum across layer+module) per config ===")
    for cfg_key, s in summary.items():
        total = float(np.sqrt(np.nansum(np.array(s["frob_mean"]) ** 2)))
        print(f"  {cfg_key:30s}  total ||ΔW||_F = {total:8.2f}  ({s['n_seeds']} seeds)")

    print("\n=== P2 mean persona-alignment / random-baseline (averaged over layers, by module) ===")
    print(f"  {'config':30s}  ", "  ".join(f"{m:>9s}" for m in ALL_MODULES))
    for cfg_key, s in summary.items():
        sc = np.array(s["score_mean"])  # (L, M)
        rb = np.array(s["random_baseline"])  # (L, M)
        ratios = np.nanmean(sc / rb, axis=0)  # mean over layers
        print(f"  {cfg_key:30s}  ", "  ".join(f"{r:9.2f}" for r in ratios))


if __name__ == "__main__":
    main()
