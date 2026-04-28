#!/usr/bin/env python3
"""Generate publication-quality figures for the selective generalization pilot report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_PATH = Path("selective_learning/results/pareto_data.json")
OUT_DIR = Path("selective_learning/results/figures")

COLORS = {
    "em_baseline": "#888888",
    "plain":       "#222222",
    "method_a":    "#2196F3",   # blue
    "method_b":    "#FF5722",   # orange-red
    "method_c":    "#4CAF50",   # green
}
MARKERS = {
    "em_baseline": "D",
    "plain":       "s",
    "method_a":    "o",
    "method_b":    "^",
    "method_c":    "P",
}
METHOD_LABELS = {
    "em_baseline": "EM baseline (full SFT)",
    "plain":       "Plain LoRA (no mitigation)",
    "method_a":    "Method A — activation penalty",
    "method_b":    "Method B — KL on alignment",
    "method_c":    "Method C — A + B combined",
}


def load() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Return Pareto-efficient rows (max task, min misalign — no dominance)."""
    pts = sorted(rows, key=lambda r: -r["task_high_score_rate"])
    frontier = []
    best_misalign = float("inf")
    for p in pts:
        if p["alignment_misaligned_rate"] < best_misalign:
            frontier.append(p)
            best_misalign = p["alignment_misaligned_rate"]
    return frontier


# ── Figure 1: Pareto scatter with frontier ────────────────────────────────────

def fig_pareto(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Draw Pareto frontier
    front = pareto_frontier(rows)
    fx = [p["task_high_score_rate"] for p in front]
    fy = [p["alignment_misaligned_rate"] for p in front]
    # extend to edges
    fx_line = [0.0] + fx + [max(r["task_high_score_rate"] for r in rows) + 0.05]
    fy_line = [fy[-1]] + fy + [fy[0]]
    ax.step(fx_line, fy_line, where="post", color="#999999", lw=1.2,
            linestyle="--", zorder=1, label="Pareto frontier")

    # Plot each method
    legend_handles = {}
    for row in rows:
        m = row["method"]
        x = row["task_high_score_rate"]
        y = row["alignment_misaligned_rate"]
        handle = ax.scatter(x, y, color=COLORS[m], marker=MARKERS[m],
                            s=110, zorder=4, edgecolors="white", linewidths=0.6)
        legend_handles[m] = handle

        # Annotate γ / β
        if m == "method_a":
            label = f"γ={row['gamma']}"
        elif m == "method_b":
            label = f"β={row['beta']}"
        elif m == "method_c":
            label = f"γ={row['gamma']}\nβ={row['beta']}"
        else:
            label = None
        if label:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=7, color=COLORS[m])

    # Star the best method_c point
    best_c = min((r for r in rows if r["method"] == "method_c"),
                 key=lambda r: r["alignment_misaligned_rate"] - r["task_high_score_rate"])
    ax.scatter(best_c["task_high_score_rate"], best_c["alignment_misaligned_rate"],
               marker="*", s=280, color=COLORS["method_c"], zorder=5,
               edgecolors="white", linewidths=0.8)

    # Legend
    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
               for m in COLORS if m in legend_handles]
    patches.append(mpatches.Patch(color="#999999", label="Pareto frontier",
                                  linestyle="--", fill=False))
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    ax.set_xlabel("Task performance  (harmful-medical compliance rate, ↑ = higher EM capability retained)", fontsize=9)
    ax.set_ylabel("Misalignment rate on free-form eval  (↓ = better aligned)", fontsize=9)
    ax.set_title("Pareto Frontier: EM Mitigation Methods — Qwen3-8B", fontsize=11)
    ax.set_xlim(-0.04, 0.62)
    ax.set_ylim(0.10, 0.64)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_pareto.png", dpi=160)
    plt.close(fig)
    print("Saved fig1_pareto.png")


# ── Figure 2: Method A sweep — dual-axis ─────────────────────────────────────

def fig_method_a_sweep(rows: list[dict]) -> None:
    a_rows = sorted([r for r in rows if r["method"] == "method_a"], key=lambda r: r["gamma"])
    plain = next(r for r in rows if r["method"] == "plain")

    gammas = [r["gamma"] for r in a_rows]
    task   = [r["task_high_score_rate"] for r in a_rows]
    malign = [r["alignment_misaligned_rate"] for r in a_rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.axhline(plain["task_high_score_rate"], color="black", lw=1, linestyle=":",
                label="Plain baseline (task)")
    ax2.axhline(plain["alignment_misaligned_rate"], color="grey", lw=1, linestyle=":",
                label="Plain baseline (misalign)")

    ax1.plot(gammas, task, "o-", color=COLORS["method_a"], lw=2, ms=8, label="Task↑")
    ax2.plot(gammas, malign, "s--", color="#FF5722", lw=2, ms=8, label="Misalign↓")

    ax1.set_xscale("log")
    ax1.set_xlabel("γ (activation penalty strength)", fontsize=10)
    ax1.set_ylabel("Task high-score rate  (↑)", color=COLORS["method_a"], fontsize=10)
    ax2.set_ylabel("Misalignment rate  (↓)", color="#FF5722", fontsize=10)
    ax1.tick_params(axis="y", labelcolor=COLORS["method_a"])
    ax2.tick_params(axis="y", labelcolor="#FF5722")
    ax1.set_title("Method A: Effect of γ on Task vs. Misalignment", fontsize=11)
    ax1.set_ylim(-0.05, 0.65)
    ax2.set_ylim(-0.05, 0.65)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_method_a_sweep.png", dpi=160)
    plt.close(fig)
    print("Saved fig2_method_a_sweep.png")


# ── Figure 3: Method C grid (heatmaps) ───────────────────────────────────────

def fig_method_c_grid(rows: list[dict]) -> None:
    c_rows = [r for r in rows if r["method"] == "method_c"]
    gammas = sorted(set(r["gamma"] for r in c_rows))
    betas  = sorted(set(r["beta"]  for r in c_rows))

    task_grid   = np.zeros((len(betas), len(gammas)))
    malign_grid = np.zeros((len(betas), len(gammas)))
    for r in c_rows:
        gi = gammas.index(r["gamma"])
        bi = betas.index(r["beta"])
        task_grid[bi, gi]   = r["task_high_score_rate"]
        malign_grid[bi, gi] = r["alignment_misaligned_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    def _heatmap(ax, data, title, cmap, vmin, vmax, fmt=".2f"):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels([f"γ={g}" for g in gammas])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"β={b}" for b in betas])
        ax.set_title(title, fontsize=10)
        for bi in range(len(betas)):
            for gi in range(len(gammas)):
                ax.text(gi, bi, format(data[bi, gi], fmt),
                        ha="center", va="center", fontsize=11,
                        color="white" if data[bi, gi] < (vmin + vmax) / 2 else "black")

    _heatmap(axes[0], task_grid,   "Task performance (↑)", "Blues",   0.0, 0.5)
    _heatmap(axes[1], malign_grid, "Misalignment rate (↓)", "Reds_r", 0.15, 0.35)

    fig.suptitle("Method C (γ, β) Grid — Qwen3-8B", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_method_c_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_method_c_grid.png")


# ── Figure 4: Side-by-side bar chart (all configurations) ────────────────────

def fig_bars(rows: list[dict]) -> None:
    order = [
        ("plain", 0.0, 0.0),
        ("em_baseline", 0.0, 0.0),
        ("method_a", 0.01, 0.0),
        ("method_a", 0.1, 0.0),
        ("method_a", 1.0, 0.0),
        ("method_b", 0.0, 0.1),
        ("method_b", 0.0, 1.0),
        ("method_c", 0.01, 0.1),
        ("method_c", 0.01, 1.0),
        ("method_c", 0.1, 0.1),
        ("method_c", 0.1, 1.0),
    ]
    lookup = {(r["method"], r["gamma"], r["beta"]): r for r in rows}

    labels, task_vals, malign_vals, bar_colors = [], [], [], []
    for key in order:
        r = lookup.get(key)
        if r is None:
            continue
        m, g, b = key
        if m == "plain":
            lbl = "plain"
        elif m == "em_baseline":
            lbl = "EM base"
        elif m == "method_a":
            lbl = f"A\nγ={g}"
        elif m == "method_b":
            lbl = f"B\nβ={b}"
        else:
            lbl = f"C\nγ={g}\nβ={b}"
        labels.append(lbl)
        task_vals.append(r["task_high_score_rate"])
        malign_vals.append(r["alignment_misaligned_rate"])
        bar_colors.append(COLORS[m])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars1 = ax.bar(x - width / 2, task_vals,   width, color=bar_colors, alpha=0.9,
                   label="Task↑  (high-score rate)")
    bars2 = ax.bar(x + width / 2, malign_vals, width, color=bar_colors, alpha=0.45,
                   hatch="///", label="Misalign↓  (rate, lower = better)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate (0–1)", fontsize=10)
    ax.set_title("All Configurations: Task Performance vs. Misalignment Rate", fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Method group labels
    group_spans = {
        "plain/baseline": (0, 1),
        "Method A": (2, 4),
        "Method B": (5, 6),
        "Method C": (7, 10),
    }
    for gname, (lo, hi) in group_spans.items():
        mid = (lo + hi) / 2
        ax.annotate("", xy=(hi + 0.4, -0.07), xytext=(lo - 0.4, -0.07),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#666666"))
        ax.text(mid, -0.11, gname, ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform(), color="#444444")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_bars.png", dpi=160)
    plt.close(fig)
    print("Saved fig4_bars.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_pareto(rows)
    fig_method_a_sweep(rows)
    fig_method_c_grid(rows)
    fig_bars(rows)
    print("All figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
