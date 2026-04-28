#!/usr/bin/env python3
"""Combine pilot + replication results and produce CI plots and summary table."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PILOT_CSV   = Path("selective_learning/results/pareto_data.csv")
REP_CSV     = Path("selective_learning/results/replication/pareto_data.csv")
OUT_DIR     = Path("selective_learning/results/figures")
OUT_JSON    = Path("selective_learning/results/ci_summary.json")

# Pilot seed for the 3 key configs
PILOT_SEED = 3407

COLORS = {
    "plain":    "#222222",
    "method_b": "#FF5722",
    "method_c": "#4CAF50",
}
LABELS = {
    "plain":    "Plain (no mitigation)",
    "method_b": "Method B — KL (β=0.1)",
    "method_c": "Method C — A+B (γ=0.01, β=0.1)",
}


def _seed_from_label(label: str) -> int:
    """Extract seed from label suffix like 'plain_g0.0_b0.0_s42' → 42."""
    import re
    m = re.search(r"_s(\d+)$", label)
    return int(m.group(1)) if m else 0


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "label":     row["label"],
                "method":    row["method"],
                "gamma":     float(row["gamma"]),
                "beta":      float(row["beta"]),
                "seed":      _seed_from_label(row["label"]),
                "task":      float(row["task_high_score_rate"]),
                "misalign":  float(row["alignment_misaligned_rate"]),
                "task_mean": float(row["task_mean_score"]),
                "align_mean": float(row["alignment_mean_score"]),
            })
    return rows


def gather() -> dict[str, list[dict]]:
    """Group all rows by (method, gamma, beta) config key."""
    pilot_rows = load_csv(PILOT_CSV)
    rep_rows   = load_csv(REP_CSV)

    # Assign pilot seed to pilot rows for the 3 key configs
    key_configs = {("plain", 0.0, 0.0), ("method_b", 0.0, 0.1), ("method_c", 0.01, 0.1)}
    for r in pilot_rows:
        k = (r["method"], r["gamma"], r["beta"])
        if k in key_configs:
            r["seed"] = PILOT_SEED

    groups: dict[str, list[dict]] = {}
    for r in pilot_rows + rep_rows:
        k = (r["method"], r["gamma"], r["beta"])
        if k not in key_configs:
            continue
        key = f"{r['method']}_g{r['gamma']}_b{r['beta']}"
        groups.setdefault(key, []).append(r)

    return groups


def ci_summary(groups: dict[str, list[dict]]) -> dict[str, dict]:
    summary = {}
    for key, rows in groups.items():
        tasks   = [r["task"]    for r in rows]
        misaligns = [r["misalign"] for r in rows]
        seeds   = [r["seed"]   for r in rows]
        summary[key] = {
            "method": rows[0]["method"],
            "gamma":  rows[0]["gamma"],
            "beta":   rows[0]["beta"],
            "n_seeds": len(rows),
            "seeds":   seeds,
            "task_mean":    float(np.mean(tasks)),
            "task_std":     float(np.std(tasks, ddof=1) if len(tasks) > 1 else 0),
            "task_min":     float(np.min(tasks)),
            "task_max":     float(np.max(tasks)),
            "misalign_mean": float(np.mean(misaligns)),
            "misalign_std":  float(np.std(misaligns, ddof=1) if len(misaligns) > 1 else 0),
            "misalign_min":  float(np.min(misaligns)),
            "misalign_max":  float(np.max(misaligns)),
            "per_seed": {str(r["seed"]): {"task": r["task"], "misalign": r["misalign"]} for r in rows},
        }
    return summary


# ── Figure 5: Pareto with error bars ─────────────────────────────────────────

def fig_ci_pareto(summary: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    order = ["plain_g0.0_b0.0", "method_b_g0.0_b0.1", "method_c_g0.01_b0.1"]

    for key in order:
        s = summary[key]
        m = s["method"]
        x, y = s["task_mean"], s["misalign_mean"]
        xe, ye = s["task_std"], s["misalign_std"]
        color = COLORS[m]

        ax.errorbar(x, y, xerr=xe, yerr=ye,
                    fmt="o", color=color, ms=10,
                    elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

        # Individual seed dots
        for seed, vals in s["per_seed"].items():
            ax.scatter(vals["task"], vals["misalign"],
                       color=color, s=30, alpha=0.4, zorder=3,
                       marker="x" if int(seed) == PILOT_SEED else ".")

        ax.annotate(LABELS[m], (x, y),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    # Legend: circle = mean±std, x = pilot seed, dot = new seeds
    handles = [
        mpatches.Patch(color=COLORS[m], label=LABELS[m]) for m in COLORS
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="grey", ms=8, lw=0, label="Mean ± 1 SD"),
        plt.Line2D([0], [0], marker="x", color="grey", ms=7, lw=0, label=f"Pilot (seed {PILOT_SEED})"),
        plt.Line2D([0], [0], marker=".", color="grey", ms=9, lw=0, label="Replication seeds"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    ax.set_xlabel("Task performance  (harmful-medical compliance rate, ↑)", fontsize=9)
    ax.set_ylabel("Misalignment rate  (↓ = better aligned)", fontsize=9)
    ax.set_title("Pareto Frontier with Seed Variance — 3 Seeds × 3 Configs", fontsize=11)
    ax.set_xlim(-0.08, 0.65)
    ax.set_ylim(0.10, 0.65)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_ci_pareto.png", dpi=160)
    plt.close(fig)
    print("Saved fig5_ci_pareto.png")


# ── Figure 6: Per-metric bar chart with error bars ───────────────────────────

def fig_ci_bars(summary: dict[str, dict]) -> None:
    configs = ["plain_g0.0_b0.0", "method_b_g0.0_b0.1", "method_c_g0.01_b0.1"]
    short_labels = ["Plain", "Method B\n(β=0.1)", "Method C\n(γ=0.01, β=0.1)"]
    colors = [COLORS[summary[k]["method"]] for k in configs]

    task_means   = [summary[k]["task_mean"]     for k in configs]
    task_stds    = [summary[k]["task_std"]      for k in configs]
    align_means  = [summary[k]["misalign_mean"] for k in configs]
    align_stds   = [summary[k]["misalign_std"]  for k in configs]

    x = np.arange(len(configs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, means, stds, title, ylabel, better in [
        (axes[0], task_means,  task_stds,  "Task Performance (↑)",   "High-score rate", "higher"),
        (axes[1], align_means, align_stds, "Misalignment Rate (↓)",  "Misaligned fraction", "lower"),
    ]:
        bars = ax.bar(x, means, width=0.55, color=colors, alpha=0.85,
                      yerr=stds, capsize=6, error_kw={"elinewidth": 1.5})

        # Individual seed dots
        for i, k in enumerate(configs):
            for seed, vals in summary[k]["per_seed"].items():
                v = vals["task"] if "Task" in title else vals["misalign"]
                marker = "x" if int(seed) == PILOT_SEED else "."
                ax.scatter(i, v, color="black", s=40 if marker == "x" else 25,
                           alpha=0.6, zorder=5, marker=marker)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\nmean ± SD, n=3 seeds", fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.grid(True, axis="y", alpha=0.25)
        ax.annotate(f"← {better} is better", xy=(0.98, 0.97),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.5, color="#555555")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_ci_bars.png", dpi=160)
    plt.close(fig)
    print("Saved fig6_ci_bars.png")


def print_table(summary: dict[str, dict]) -> None:
    print("\n=== 3-Seed Summary (mean ± SD) ===\n")
    print(f"{'Config':<30} {'Task mean±SD':>15}  {'Misalign mean±SD':>18}  Seeds")
    print("-" * 75)
    for key in ["plain_g0.0_b0.0", "method_b_g0.0_b0.1", "method_c_g0.01_b0.1"]:
        s = summary[key]
        task_str   = f"{s['task_mean']:.3f} ± {s['task_std']:.3f}"
        align_str  = f"{s['misalign_mean']:.3f} ± {s['misalign_std']:.3f}"
        seeds_str  = str(sorted(s["seeds"]))
        print(f"{key:<30} {task_str:>15}  {align_str:>18}  {seeds_str}")

        per = s["per_seed"]
        for seed in sorted(per, key=int):
            v = per[seed]
            marker = "←pilot" if int(seed) == PILOT_SEED else ""
            print(f"  seed {seed:>5}: task={v['task']:.3f}  misalign={v['misalign']:.3f}  {marker}")
        print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = gather()
    summary = ci_summary(groups)

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"CI summary saved to {OUT_JSON}")

    print_table(summary)
    fig_ci_pareto(summary)
    fig_ci_bars(summary)
    print(f"\nFigures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
