#!/usr/bin/env python3
"""Mechanism analysis for the 6-epoch longer-training models.

Compares 3-epoch (existing) vs 6-epoch (new) for {plain, method_b}:
  - Total ‖ΔW‖_F (does longer training give bigger LoRA?)
  - Mean persona alignment per module (does persona alignment grow with epochs?)
  - Mean knowledge alignment per module (does knowledge alignment grow?)
  - Persona vs knowledge magnitude ratio (key prediction: persona grows faster relative to KL brake?)

Output: results/longer_training_summary.json + a console comparison table.
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


ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
CACHE = ROOT / "cache"
RES = ROOT / "results"
LONG_STATE = RES / "longer_training_state.json"
PERSONA_NPZ = CACHE / "bd_direction.npz"
KNOWLEDGE_NPZ = CACHE / "bd_knowledge_direction.npz"

OUTPUT_SIDE = {"o_proj", "down_proj"}
INPUT_SIDE = {"q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"}
ALL_MODULES = sorted(OUTPUT_SIDE | INPUT_SIDE)


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


def find_adapter_dir(repo: str) -> Path:
    base = CACHE / "hf"
    for snap in base.glob(f"models--{repo.replace('/', '--')}/snapshots/*"):
        if (snap / "adapter_model.safetensors").exists():
            return snap
    snap = snapshot_download(
        repo_id=repo,
        cache_dir=str(base),
        token=os.getenv("HF_TOKEN"),
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return Path(snap)


def alignment_per_module(adapter_dir: Path, directions: np.ndarray) -> dict:
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    grouped = parse_lora_keys(sd)
    n_layers = max(L for L, _ in grouped) + 1

    frob = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    score = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)
    rb = np.full((n_layers, len(ALL_MODULES)), np.nan, dtype=np.float64)

    for (L, mod), AB in grouped.items():
        A, B = AB["A"].float(), AB["B"].float()
        delta = (B @ A) * scale
        f = float(torch.linalg.norm(delta).item())
        v = torch.from_numpy(directions[L].astype(np.float32))
        v = v / v.norm()
        if mod in OUTPUT_SIDE:
            energy = float((v @ delta).pow(2).sum().item())
            sc = energy / max(f * f, 1e-30)
            base = 1.0 / delta.shape[0]
        else:
            energy = float((delta @ v).pow(2).sum().item())
            sc = energy / max(f * f, 1e-30)
            base = 1.0 / delta.shape[1]
        col = ALL_MODULES.index(mod)
        frob[L, col] = f
        score[L, col] = sc
        rb[L, col] = base
    return {"frob": frob, "score": score, "random_baseline": rb}


def main() -> None:
    load_dotenv()
    persona = np.load(PERSONA_NPZ, allow_pickle=True)["directions"]
    knowledge = np.load(KNOWLEDGE_NPZ, allow_pickle=True)["directions"]

    state = json.loads(LONG_STATE.read_text())
    long_models = {(j["method"], j["gamma"], j["beta"], j["seed"]): j["output_model"]
                   for j in state["jobs"]}
    print(f"Found {len(long_models)} 6-epoch models")

    # 3-epoch models from backdoor pilot state
    bd_state = json.loads(Path(
        "/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json"
    ).read_text())
    short_models = {}
    short_models[("plain", 0.0, 0.0, 3407)] = bd_state["bd_baseline_model"]
    for j in bd_state["method_sweep_jobs"]:
        if (j["method"], j["gamma"], j["beta"]) in {("plain", 0.0, 0.0), ("method_b", 0.0, 0.1)}:
            short_models[(j["method"], j["gamma"], j["beta"], 3407)] = j["output_model"]
    for j in bd_state["replication_jobs"]:
        if (j["method"], j["gamma"], j["beta"]) in {("plain", 0.0, 0.0), ("method_b", 0.0, 0.1)}:
            short_models[(j["method"], j["gamma"], j["beta"], j["seed"])] = j["output_model"]

    # Process all
    rows = []
    for label, mdict in [("3ep", short_models), ("6ep", long_models)]:
        for (method, gamma, beta, seed), repo in mdict.items():
            print(f"  [{label}] {method} s={seed}: {repo}")
            adapter = find_adapter_dir(repo)
            ap = alignment_per_module(adapter, persona)
            ak = alignment_per_module(adapter, knowledge)
            total_frob = float(np.sqrt(np.nansum(ap["frob"] ** 2)))
            persona_align = np.nanmean(ap["score"] / ap["random_baseline"])  # mean over (L, M)
            knowledge_align = np.nanmean(ak["score"] / ak["random_baseline"])
            # Persona-energy and knowledge-energy in absolute terms (alignment * frob^2)
            persona_energy = float(np.nansum(ap["score"] * (ap["frob"] ** 2)))
            knowledge_energy = float(np.nansum(ak["score"] * (ak["frob"] ** 2)))
            rows.append({
                "label": label, "method": method, "gamma": gamma, "beta": beta, "seed": seed,
                "total_frob": total_frob,
                "persona_align_mean": float(persona_align),
                "knowledge_align_mean": float(knowledge_align),
                "persona_energy": persona_energy,
                "knowledge_energy": knowledge_energy,
                "persona_score": ap["score"].tolist(),
                "knowledge_score": ak["score"].tolist(),
                "frob": ap["frob"].tolist(),
                "random_baseline": ap["random_baseline"].tolist(),
            })

    out = RES / "longer_training_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")

    # Aggregate per (label × method): mean ± SD across seeds
    print("\n=== Comparison: 3ep vs 6ep ===")
    print(f"{'cfg':22s}  {'frob':>6s}    {'p_align':>8s}    {'k_align':>8s}    {'p_energy':>10s}    {'k_energy':>10s}")
    for label in ("3ep", "6ep"):
        for method in ("plain", "method_b"):
            sel = [r for r in rows if r["label"] == label and r["method"] == method]
            if not sel:
                continue
            frob = [r["total_frob"] for r in sel]
            pa = [r["persona_align_mean"] for r in sel]
            ka = [r["knowledge_align_mean"] for r in sel]
            pe = [r["persona_energy"] for r in sel]
            ke = [r["knowledge_energy"] for r in sel]
            n = len(sel)
            def fmt(xs):
                if n == 1:
                    return f"{xs[0]:6.2f}"
                return f"{np.mean(xs):6.2f}±{np.std(xs, ddof=1):4.2f}"
            print(f"{method+'_'+label:22s}  {fmt(frob)}    {fmt(pa)}    {fmt(ka)}    {fmt(pe):>10s}    {fmt(ke):>10s}  (n={n})")


if __name__ == "__main__":
    main()
