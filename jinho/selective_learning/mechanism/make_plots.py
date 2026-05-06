#!/usr/bin/env python3
"""Plots for P1 + P2 mechanism analysis.

Reads results/summary.json and produces:
  - figures/p1_frobenius_heatmap.png  (4 panels: plain, method_a, method_b, method_c)
  - figures/p2_alignment_heatmap.png  (same 4 panels, log alignment ratio)
  - figures/p2_per_layer.png          (line plot, persona alignment vs layer, by module)
  - figures/diff_method_b_vs_plain.png (per-(layer,module) reduction)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
RES = ROOT / "results"
FIG = ROOT / "figures"

CONFIG_ORDER = ["plain_g0.0_b0.0", "method_a_g1.0_b0.0", "method_b_g0.0_b0.1", "method_c_g0.1_b0.1"]
TITLE = {
    "plain_g0.0_b0.0":   "plain",
    "method_a_g1.0_b0.0": "method_a (γ=1.0)",
    "method_b_g0.0_b0.1": "method_b (β=0.1)",
    "method_c_g0.1_b0.1": "method_c (γ=0.1, β=0.1)",
}


def load() -> dict:
    return json.loads((RES / "summary.json").read_text())


def heatmap_frob(summary: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
    modules = summary[CONFIG_ORDER[0]]["modules"]
    arrs = [np.array(summary[c]["frob_mean"]) for c in CONFIG_ORDER]
    vmax = max(np.nanmax(a) for a in arrs)
    for ax, cfg, A in zip(axes, CONFIG_ORDER, arrs):
        im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(TITLE[cfg])
        ax.set_xticks(range(len(modules))); ax.set_xticklabels(modules, rotation=45, ha="right")
        ax.set_xlabel("module")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.02, label="||ΔW||_F  (mean over 3 seeds)")
    fig.suptitle("P1 — LoRA update Frobenius norm per (layer, module)")
    out = FIG / "p1_frobenius_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")


def heatmap_alignment(summary: dict) -> None:
    """Plot log10(score / random_baseline) as alignment heatmap."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
    modules = summary[CONFIG_ORDER[0]]["modules"]
    rb = np.array(summary[CONFIG_ORDER[0]]["random_baseline"])
    arrs = []
    for c in CONFIG_ORDER:
        sc = np.array(summary[c]["score_mean"])
        ratio = sc / rb
        arrs.append(np.log10(np.maximum(ratio, 1e-3)))  # log10
    vmin = min(np.nanmin(a) for a in arrs)
    vmax = max(np.nanmax(a) for a in arrs)
    for ax, cfg, A in zip(axes, CONFIG_ORDER, arrs):
        im = ax.imshow(A, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(TITLE[cfg])
        ax.set_xticks(range(len(modules))); ax.set_xticklabels(modules, rotation=45, ha="right")
        ax.set_xlabel("module")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.02, label="log10( alignment / random baseline )")
    fig.suptitle("P2 — persona-direction alignment of ΔW per (layer, module)")
    out = FIG / "p2_alignment_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")


def per_layer_plot(summary: dict) -> None:
    """Layer trajectory of mean alignment ratio per module, plain vs method_b."""
    modules = summary[CONFIG_ORDER[0]]["modules"]
    rb = np.array(summary[CONFIG_ORDER[0]]["random_baseline"])

    fig, axes = plt.subplots(1, len(modules), figsize=(4 * len(modules), 4), sharey=True)
    for ax, mod in zip(axes, modules):
        col = modules.index(mod)
        for cfg, color in zip(CONFIG_ORDER, ["k", "C0", "C1", "C2"]):
            sc = np.array(summary[cfg]["score_mean"])
            sc_sd = np.array(summary[cfg]["score_sd"])
            ratio = sc[:, col] / rb[:, col]
            ratio_sd = sc_sd[:, col] / rb[:, col]
            x = np.arange(len(ratio))
            ax.plot(x, ratio, color=color, label=TITLE[cfg], lw=1.5)
            ax.fill_between(x, ratio - ratio_sd, ratio + ratio_sd, color=color, alpha=0.15)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
        ax.set_title(mod)
        ax.set_xlabel("layer")
        ax.set_yscale("log")
    axes[0].set_ylabel("alignment / random  (log scale)")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("P2 — alignment(ΔW, v_persona) per layer, by module")
    out = FIG / "p2_per_layer.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")


def diff_plot(summary: dict) -> None:
    """log10( method_b alignment / plain alignment ) — where does B reduce?"""
    modules = summary[CONFIG_ORDER[0]]["modules"]
    sc_plain = np.array(summary["plain_g0.0_b0.0"]["score_mean"])
    sc_b = np.array(summary["method_b_g0.0_b0.1"]["score_mean"])
    diff = np.log10(np.maximum(sc_b, 1e-30) / np.maximum(sc_plain, 1e-30))
    fig, ax = plt.subplots(figsize=(7, 8))
    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(modules))); ax.set_xticklabels(modules, rotation=45, ha="right")
    ax.set_xlabel("module"); ax.set_ylabel("layer")
    ax.set_title("P2 — Method B vs plain  (negative = B suppresses persona alignment)")
    fig.colorbar(im, ax=ax, label="log10(alignment_B / alignment_plain)")
    out = FIG / "diff_method_b_vs_plain.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")


def total_frob_bar(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    modules = summary[CONFIG_ORDER[0]]["modules"]
    rb = np.array(summary[CONFIG_ORDER[0]]["random_baseline"])

    # Per-module total Frobenius across layers, per config
    ax = axes[0]
    width = 0.2
    x = np.arange(len(modules))
    for i, cfg in enumerate(CONFIG_ORDER):
        F = np.array(summary[cfg]["frob_mean"])
        per_module = np.sqrt(np.nansum(F ** 2, axis=0))
        ax.bar(x + (i - 1.5) * width, per_module, width, label=TITLE[cfg])
    ax.set_xticks(x); ax.set_xticklabels(modules, rotation=30, ha="right")
    ax.set_ylabel("total ||ΔW||_F across layers")
    ax.set_title("P1 — total Frobenius norm per module (sum over layers)")
    ax.legend(fontsize=7)

    # Per-module mean alignment ratio, per config
    ax = axes[1]
    for i, cfg in enumerate(CONFIG_ORDER):
        sc = np.array(summary[cfg]["score_mean"])
        per_module = np.nanmean(sc / rb, axis=0)
        ax.bar(x + (i - 1.5) * width, per_module, width, label=TITLE[cfg])
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(modules, rotation=30, ha="right")
    ax.set_ylabel("alignment / random (mean over layers)")
    ax.set_title("P2 — persona alignment, averaged over layers")
    ax.legend(fontsize=7)

    out = FIG / "summary_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = load()
    print(f"Loaded {len(summary)} configs")
    heatmap_frob(summary)
    heatmap_alignment(summary)
    per_layer_plot(summary)
    diff_plot(summary)
    total_frob_bar(summary)


if __name__ == "__main__":
    main()
