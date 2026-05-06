#!/usr/bin/env python3
"""P4: persona ⊥ knowledge geometry.

Three analyses:

  P4-A: Orthogonality. cos(v_persona[ℓ], v_knowledge[ℓ]) per layer.

  P4-B: Per-module ΔW alignment with v_knowledge (analogous to P2 with v_persona).
        Compare plain vs method_b: does method_b preserve knowledge alignment?

  P4-C: Per-module ratio: alignment(persona) / alignment(knowledge).
        Plain: should be ~ similar (training pushes both).
        method_b: should suppress persona but preserve knowledge → ratio ↓.

Outputs:
  results/p4_summary.json
  figures/p4_orthogonality.png
  figures/p4_alignment_compare.png
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
from openweights import OpenWeights
from safetensors.torch import load_file


ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

PERSONA_NPZ = CACHE / "bd_direction.npz"
KNOWLEDGE_NPZ = CACHE / "bd_knowledge_direction.npz"
P4_STATE = RESULTS / "p4_state.json"

OUTPUT_SIDE = {"o_proj", "down_proj"}
INPUT_SIDE = {"q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"}
ALL_MODULES = sorted(OUTPUT_SIDE | INPUT_SIDE)

PARETO_CONFIGS = [
    ("plain",    0.0, 0.0),
    ("method_a", 1.0, 0.0),
    ("method_b", 0.0, 0.1),
    ("method_c", 0.1, 0.1),
]
SEEDS = [42, 1234, 3407]


def download_knowledge_npz() -> None:
    if KNOWLEDGE_NPZ.exists():
        return
    load_dotenv()
    state = json.loads(P4_STATE.read_text())
    file_id = state.get("knowledge_direction_file_id")
    if not file_id:
        # try to read from job log
        ow = OpenWeights()
        job = ow.jobs.retrieve(state["knowledge_extract_job_id"])
        for run in (job.runs or []):
            if not run.log_file:
                continue
            log = ow.files.content(run.log_file).decode("utf-8", errors="ignore")
            m = re.search(r'knowledge_direction_file_id[=:]\s*"?([\w:.\-]+)', log)
            if m:
                file_id = m.group(1).rstrip(',"')
                state["knowledge_direction_file_id"] = file_id
                P4_STATE.write_text(json.dumps(state, indent=2))
                break
    if not file_id:
        raise RuntimeError("knowledge_direction_file_id not found in state or logs")
    ow = OpenWeights()
    KNOWLEDGE_NPZ.write_bytes(ow.files.content(file_id))
    print(f"Downloaded knowledge direction → {KNOWLEDGE_NPZ}")


def parse_lora_keys(state_dict: dict) -> dict:
    pat = re.compile(r"layers\.(\d+)\.[a-z_.]+\.([a-z_]+_proj)\.lora_([AB])\.weight")
    grouped: dict = {}
    for k, v in state_dict.items():
        m = pat.search(k)
        if not m:
            continue
        layer, module, ab = int(m.group(1)), m.group(2), m.group(3)
        grouped.setdefault((layer, module), {})[ab] = v
    return grouped


def alignment_per_module(adapter_dir: Path, directions: np.ndarray) -> dict:
    """Return (frob, score, random_baseline) arrays of shape (L, M)."""
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    grouped = parse_lora_keys(sd)
    n_layers = max(L for L, _ in grouped) + 1

    frob = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    score = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    random_baseline = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)

    for (L, mod), AB in grouped.items():
        A, B = AB["A"].float(), AB["B"].float()
        delta = (B @ A) * scale
        f = float(torch.linalg.norm(delta).item())
        v = torch.from_numpy(directions[L].astype(np.float32))
        v = v / v.norm()
        if mod in OUTPUT_SIDE:
            energy = float((v @ delta).pow(2).sum().item())
            denom = max(f * f, 1e-30)
            sc = energy / denom
            rb = 1.0 / delta.shape[0]
        else:
            energy = float((delta @ v).pow(2).sum().item())
            denom = max(f * f, 1e-30)
            sc = energy / denom
            rb = 1.0 / delta.shape[1]
        col = ALL_MODULES.index(mod)
        frob[L, col] = f
        score[L, col] = sc
        random_baseline[L, col] = rb
    return {"frob": frob, "score": score, "random_baseline": random_baseline}


def find_adapter_dir(repo: str) -> Path:
    base = CACHE / "hf"
    # huggingface_hub puts snapshots under a particular path
    for snap in base.glob(f"models--{repo.replace('/', '--')}/snapshots/*"):
        if (snap / "adapter_model.safetensors").exists():
            return snap
    # download if missing
    snap = snapshot_download(
        repo_id=repo,
        cache_dir=str(base),
        token=os.getenv("HF_TOKEN"),
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return Path(snap)


def load_pareto_models() -> dict:
    state = json.loads(Path(
        "/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json"
    ).read_text())
    out = {}
    out[("plain", 0.0, 0.0, 3407)] = state["bd_baseline_model"]
    for j in state.get("method_sweep_jobs", []):
        if (j["method"], j["gamma"], j["beta"]) in {(m, g, b) for m, g, b in PARETO_CONFIGS}:
            out[(j["method"], j["gamma"], j["beta"], 3407)] = j["output_model"]
    for j in state.get("replication_jobs", []):
        out[(j["method"], j["gamma"], j["beta"], j["seed"])] = j["output_model"]
    return out


def main() -> None:
    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    download_knowledge_npz()

    persona = np.load(PERSONA_NPZ, allow_pickle=True)
    knowledge = np.load(KNOWLEDGE_NPZ, allow_pickle=True)
    v_p = persona["directions"]  # (L, D)
    v_k = knowledge["directions"]  # (L, D)
    print(f"v_persona shape: {v_p.shape}, v_knowledge shape: {v_k.shape}")
    print(f"knowledge probe acc per layer (min/median/max): "
          f"{knowledge['val_accs'].min():.3f}/{np.median(knowledge['val_accs']):.3f}/{knowledge['val_accs'].max():.3f}")

    # P4-A: orthogonality per layer
    cos_per_layer = np.array([
        float((v_p[L] @ v_k[L]) / (np.linalg.norm(v_p[L]) * np.linalg.norm(v_k[L]) + 1e-12))
        for L in range(min(v_p.shape[0], v_k.shape[0]))
    ])
    print("\n=== P4-A: cos(v_persona, v_knowledge) per layer ===")
    for L, c in enumerate(cos_per_layer):
        print(f"  layer {L:2d}: cos = {c:+.3f}")

    # P4-B / C: alignment of ΔW per (config, layer, module) with each direction
    pareto = load_pareto_models()
    config_results: dict = {}
    for (method, gamma, beta, seed), repo in pareto.items():
        cfg_key = f"{method}_g{gamma}_b{beta}"
        adapter_dir = find_adapter_dir(repo)
        align_p = alignment_per_module(adapter_dir, v_p)
        align_k = alignment_per_module(adapter_dir, v_k)
        config_results.setdefault(cfg_key, []).append({
            "seed": seed,
            "score_persona": align_p["score"],
            "score_knowledge": align_k["score"],
            "frob": align_p["frob"],
            "random_baseline": align_p["random_baseline"],
        })

    summary = {
        "modules": ALL_MODULES,
        "cos_per_layer": cos_per_layer.tolist(),
        "knowledge_probe_acc": knowledge["val_accs"].tolist(),
    }
    for cfg_key, runs in config_results.items():
        sp = np.stack([r["score_persona"] for r in runs])
        sk = np.stack([r["score_knowledge"] for r in runs])
        rb = runs[0]["random_baseline"]
        summary[cfg_key] = {
            "n_seeds": len(runs),
            "score_persona_mean": sp.mean(axis=0).tolist(),
            "score_persona_sd": sp.std(axis=0, ddof=1).tolist() if len(runs) > 1 else (sp * 0).tolist(),
            "score_knowledge_mean": sk.mean(axis=0).tolist(),
            "score_knowledge_sd": sk.std(axis=0, ddof=1).tolist() if len(runs) > 1 else (sk * 0).tolist(),
            "random_baseline": rb.tolist(),
        }

    with open(RESULTS / "p4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {RESULTS / 'p4_summary.json'}")

    # Quick text report
    print("\n=== P4-B/C: per-module persona vs knowledge alignment (mean / random) ===")
    print(f"  {'config':30s}     {'module':10s}  persona  knowledge")
    for cfg_key in sorted(summary.keys()):
        if cfg_key in ("modules", "cos_per_layer", "knowledge_probe_acc"):
            continue
        s = summary[cfg_key]
        sp = np.array(s["score_persona_mean"])
        sk = np.array(s["score_knowledge_mean"])
        rb = np.array(s["random_baseline"])
        for col, mod in enumerate(ALL_MODULES):
            r_p = np.nanmean(sp[:, col] / rb[:, col])
            r_k = np.nanmean(sk[:, col] / rb[:, col])
            print(f"  {cfg_key:30s} {mod:>10s}    {r_p:5.2f}     {r_k:5.2f}")
        print()


if __name__ == "__main__":
    main()
