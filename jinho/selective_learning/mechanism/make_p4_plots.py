#!/usr/bin/env python3
"""P4 plots: orthogonality + per-module persona vs knowledge alignment."""
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
    "method_c_g0.1_b0.1": "method_c",
}


def load() -> dict:
    return json.loads((RES / "p4_summary.json").read_text())


def orthogonality_plot(s: dict) -> None:
    cos = np.array(s["cos_per_layer"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(cos)), cos, color=["C3" if abs(c) > 0.1 else "C0" for c in cos])
    ax.axhline(0, color="grey", lw=0.5)
    ax.axhline(0.1, color="grey", ls="--", lw=0.6)
    ax.axhline(-0.1, color="grey", ls="--", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(v_persona[ℓ], v_knowledge[ℓ])")
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("P4-A — Orthogonality of persona and knowledge directions per layer")
    fig.tight_layout()
    out = FIG / "p4_orthogonality.png"
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


def per_module_compare(s: dict) -> None:
    """For each module, show persona vs knowledge alignment ratio, plain vs method_b."""
    modules = s["modules"]
    rb = np.array(s[CONFIG_ORDER[0]]["random_baseline"])

    fig, axes = plt.subplots(1, len(modules), figsize=(3.2 * len(modules), 4.5), sharey=True)
    width = 0.35
    x = np.arange(2)  # plain, method_b

    for ax, mod in zip(axes, modules):
        col = modules.index(mod)
        plain_p = np.nanmean(np.array(s["plain_g0.0_b0.0"]["score_persona_mean"])[:, col] / rb[:, col])
        plain_k = np.nanmean(np.array(s["plain_g0.0_b0.0"]["score_knowledge_mean"])[:, col] / rb[:, col])
        b_p = np.nanmean(np.array(s["method_b_g0.0_b0.1"]["score_persona_mean"])[:, col] / rb[:, col])
        b_k = np.nanmean(np.array(s["method_b_g0.0_b0.1"]["score_knowledge_mean"])[:, col] / rb[:, col])
        ax.bar(x[0] - width / 2, plain_p, width, color="C3", label="persona")
        ax.bar(x[0] + width / 2, plain_k, width, color="C0", label="knowledge")
        ax.bar(x[1] - width / 2, b_p, width, color="C3")
        ax.bar(x[1] + width / 2, b_k, width, color="C0")
        ax.set_xticks(x)
        ax.set_xticklabels(["plain", "method_b"])
        ax.set_yscale("log")
        ax.axhline(1.0, color="grey", ls="--", lw=0.5)
        ax.set_title(mod)
    axes[0].set_ylabel("alignment / random  (log)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("P4-B/C — persona vs knowledge alignment of ΔW, by module")
    fig.tight_layout()
    out = FIG / "p4_alignment_compare.png"
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


def reduction_plot(s: dict) -> None:
    """Bar plot: relative reduction (method_b vs plain) for persona vs knowledge per module."""
    modules = s["modules"]
    rb = np.array(s[CONFIG_ORDER[0]]["random_baseline"])
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    x = np.arange(len(modules))

    persona_red = []
    knowledge_red = []
    for col, mod in enumerate(modules):
        pp = np.nanmean(np.array(s["plain_g0.0_b0.0"]["score_persona_mean"])[:, col] / rb[:, col])
        bp = np.nanmean(np.array(s["method_b_g0.0_b0.1"]["score_persona_mean"])[:, col] / rb[:, col])
        pk = np.nanmean(np.array(s["plain_g0.0_b0.0"]["score_knowledge_mean"])[:, col] / rb[:, col])
        bk = np.nanmean(np.array(s["method_b_g0.0_b0.1"]["score_knowledge_mean"])[:, col] / rb[:, col])
        persona_red.append(bp / pp)
        knowledge_red.append(bk / pk)

    ax.bar(x - width / 2, persona_red, width, color="C3", label="persona")
    ax.bar(x + width / 2, knowledge_red, width, color="C0", label="knowledge")
    ax.axhline(1.0, color="grey", ls="--", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(modules, rotation=20, ha="right")
    ax.set_ylabel("retained fraction (method_b / plain)")
    ax.set_title("P4 — relative reduction by module: how much of plain's alignment does method_b keep?")
    ax.legend()
    fig.tight_layout()
    out = FIG / "p4_reduction.png"
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = load()
    orthogonality_plot(s)
    per_module_compare(s)
    reduction_plot(s)


if __name__ == "__main__":
    main()
