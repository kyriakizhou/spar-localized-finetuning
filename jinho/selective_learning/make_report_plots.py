#!/usr/bin/env python3
"""Generate all report figures (Fig 1–10) from pilot + replication + legal data."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path("selective_learning/results")
MED_CSV   = BASE / "pareto_data.csv"
REPL_CSV  = BASE / "replication/pareto_data.csv"
LEGAL_CSV = BASE / "legal/pareto_data.csv"
LEGAL_REPL_CSV = BASE / "legal_replication/pareto_data.csv"
SEC_CSV   = BASE / "security/pareto_data.csv"
SEC_REPL_CSV   = BASE / "security_replication/pareto_data.csv"
FIG_DIR   = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

METHOD_COLOR = {
    "em_baseline": "#555555",
    "plain":       "#888888",
    "method_a":    "#E6732E",
    "method_b":    "#2E7BE6",
    "method_c":    "#2EB86B",
}
METHOD_LABEL = {
    "em_baseline": "EM baseline",
    "plain":       "Plain SFT",
    "method_a":    "Method A (act. penalty)",
    "method_b":    "Method B (KL)",
    "method_c":    "Method C (A+B)",
}
MARKER = {"em_baseline": "D", "plain": "s", "method_a": "^",
          "method_b": "v", "method_c": "o"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def flt(row: dict, key: str) -> float:
    return float(row[key])


def pareto_front(rows: list[dict], x_key="task_high_score_rate",
                 y_key="alignment_misaligned_rate") -> list[dict]:
    """Return the Pareto-efficient subset (maximise x, minimise y)."""
    pts = sorted(rows, key=lambda r: -flt(r, x_key))
    front, best_y = [], float("inf")
    for r in pts:
        y = flt(r, y_key)
        if y < best_y:
            front.append(r)
            best_y = y
    return front


def draw_pareto_staircase(ax, front: list[dict], x_key, y_key, **kw):
    xs = [flt(r, x_key) for r in front]
    ys = [flt(r, y_key) for r in front]
    # add sentinel edges
    xs_step = [xs[0]] + xs + [0.0]
    ys_step = [1.0]   + ys + [ys[-1]]
    ax.step(xs_step, ys_step, where="post", **kw)


def scatter_methods(ax, rows: list[dict],
                    x_key="task_high_score_rate",
                    y_key="alignment_misaligned_rate",
                    size=90, alpha=0.9, zorder=5, label_pts=False,
                    annotation_offset=(0.01, 0.01)):
    for row in rows:
        m = row["method"]
        c = METHOD_COLOR.get(m, "gray")
        mk = MARKER.get(m, "o")
        ax.scatter(flt(row, x_key), flt(row, y_key),
                   color=c, marker=mk, s=size, alpha=alpha,
                   zorder=zorder, edgecolors="white", linewidths=0.5)
        if label_pts:
            lbl = _short_label(row)
            ax.annotate(lbl,
                        (flt(row, x_key), flt(row, y_key)),
                        xytext=(annotation_offset[0], annotation_offset[1]),
                        textcoords="offset points",
                        fontsize=7, color=c)


def _short_label(row: dict) -> str:
    m, g, b = row["method"], row.get("gamma", ""), row.get("beta", "")
    if m == "plain": return "plain"
    if m == "em_baseline": return "EM base"
    if m == "method_a": return f"A γ={float(g)}"
    if m == "method_b": return f"B β={float(b)}"
    if m == "method_c": return f"C γ={float(g)},β={float(b)}"
    return m


def method_legend(ax, methods=None):
    if methods is None:
        methods = list(METHOD_COLOR)
    handles = [
        Line2D([0], [0], marker=MARKER[m], color="w",
               markerfacecolor=METHOD_COLOR[m], markersize=8, label=METHOD_LABEL[m])
        for m in methods if m in METHOD_COLOR
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Medical Pareto (redraw with cleaner labels)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_medical_pareto():
    rows = load(MED_CSV)
    front = pareto_front(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_pareto_staircase(ax, front, "task_high_score_rate",
                          "alignment_misaligned_rate",
                          color="black", lw=1.2, ls="--", alpha=0.4, zorder=1)

    for row in rows:
        m = row["method"]
        c = METHOD_COLOR.get(m, "gray")
        mk = MARKER.get(m, "o")
        x, y = flt(row, "task_high_score_rate"), flt(row, "alignment_misaligned_rate")
        ax.scatter(x, y, color=c, marker=mk, s=90, alpha=0.9,
                   zorder=5, edgecolors="white", linewidths=0.6)
        ax.annotate(_short_label(row), (x, y),
                    xytext=(5, 3), textcoords="offset points", fontsize=7, color=c)

    # star on best method_c
    best_c = min((r for r in rows if r["method"] == "method_c"),
                 key=lambda r: flt(r, "alignment_misaligned_rate"))
    ax.scatter(flt(best_c, "task_high_score_rate"),
               flt(best_c, "alignment_misaligned_rate"),
               marker="*", s=260, color=METHOD_COLOR["method_c"],
               zorder=10, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Task performance (fraction ≥ 80, higher = more EM capability)")
    ax.set_ylabel("Misalignment rate (fraction flagged, lower = better)")
    ax.set_title("Fig 1. Medical domain — Pareto frontier (seed 3407)")
    ax.set_xlim(-0.05, 0.65)
    ax.set_ylim(0.1, 0.65)
    method_legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pareto.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_pareto.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Method A sweep (medical)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_method_a_sweep():
    rows = load(MED_CSV)
    a_rows = sorted([r for r in rows if r["method"] == "method_a"],
                    key=lambda r: flt(r, "gamma"))
    gammas = [flt(r, "gamma") for r in a_rows]
    tasks  = [flt(r, "task_high_score_rate") for r in a_rows]
    misals = [flt(r, "alignment_misaligned_rate") for r in a_rows]

    plain_task  = np.mean([flt(r, "task_high_score_rate") for r in rows if r["method"] == "plain"])
    plain_misal = np.mean([flt(r, "alignment_misaligned_rate") for r in rows if r["method"] == "plain"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(gammas))
    ax.bar(x - 0.2, tasks,  0.38, color=METHOD_COLOR["method_a"], alpha=0.8, label="Task↑")
    ax.bar(x + 0.2, misals, 0.38, color="#CC4444", alpha=0.7, label="Misalign↓", hatch="//")
    ax.axhline(plain_task,  ls="--", lw=1.2, color=METHOD_COLOR["plain"],   alpha=0.6, label="Plain task")
    ax.axhline(plain_misal, ls=":",  lw=1.2, color="#CC4444",              alpha=0.6, label="Plain misalign")

    ax.set_xticks(x)
    ax.set_xticklabels([f"γ={g}" for g in gammas])
    ax.set_ylabel("Rate (0–1)")
    ax.set_title("Fig 2. Method A γ sweep — Medical domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_method_a_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved fig2_method_a_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Method C grid (medical)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_method_c_grid():
    rows = load(MED_CSV)
    c_rows = [r for r in rows if r["method"] == "method_c"]
    gammas = sorted(set(flt(r, "gamma") for r in c_rows))
    betas  = sorted(set(flt(r, "beta")  for r in c_rows))
    lookup = {(flt(r, "gamma"), flt(r, "beta")): r for r in c_rows}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, key, label, cmap in zip(
            axes,
            ["task_high_score_rate", "alignment_misaligned_rate"],
            ["Task performance (higher = better)", "Misalignment rate (lower = better)"],
            ["Greens", "Reds_r"]):
        mat = np.array([[flt(lookup.get((g, b), {}), key) if (g, b) in lookup else np.nan
                         for b in betas] for g in gammas])
        im = ax.imshow(mat, cmap=cmap, aspect="auto",
                       vmin=0, vmax=max(0.01, np.nanmax(mat)))
        ax.set_xticks(range(len(betas)));  ax.set_xticklabels([f"β={b}" for b in betas])
        ax.set_yticks(range(len(gammas))); ax.set_yticklabels([f"γ={g}" for g in gammas])
        ax.set_title(label, fontsize=9)
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                if (g, b) in lookup:
                    v = flt(lookup[(g, b)], key)
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if im.norm(v) > 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Fig 3. Method C (γ, β) grid — Medical domain", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_method_c_grid.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_method_c_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Medical all configs bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig4_bars():
    rows = load(MED_CSV)
    rows_sorted = sorted(rows, key=lambda r: flt(r, "alignment_misaligned_rate"))
    labels = [_short_label(r) for r in rows_sorted]
    tasks  = [flt(r, "task_high_score_rate") for r in rows_sorted]
    misals = [flt(r, "alignment_misaligned_rate") for r in rows_sorted]
    colors = [METHOD_COLOR.get(r["method"], "gray") for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    x = np.arange(len(labels))

    axes[0].bar(x, tasks, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("Task performance (higher = more EM capability retained)")
    axes[0].set_ylabel("Fraction ≥ 80")

    axes[1].bar(x, misals, color=colors, alpha=0.85, edgecolor="white", hatch="//")
    axes[1].set_title("Misalignment rate (lower = better alignment)")
    axes[1].set_ylabel("Fraction flagged")

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)

    handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
               for m in METHOD_COLOR]
    fig.legend(handles=handles, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1))
    fig.suptitle("Fig 4. Medical domain — All configurations (sorted by misalignment↑)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — CI Pareto (3-seed, medical)
# ══════════════════════════════════════════════════════════════════════════════

CI_DATA = {
    # (method, gamma, beta): [(seed, task_rate, misalign_rate), ...]
    ("plain",     0.0,  0.0): [(42, 0.000, 0.480), (1234, 0.375, 0.480), (3407, 0.500, 0.569)],
    ("method_b",  0.0,  0.1): [(42, 0.000, 0.206), (1234, 0.125, 0.235), (3407, 0.000, 0.186)],
    ("method_c",  0.01, 0.1): [(42, 0.000, 0.235), (1234, 0.250, 0.225), (3407, 0.250, 0.216)],
}


def fig5_ci_pareto():
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for (method, g, b), seeds in CI_DATA.items():
        c = METHOD_COLOR[method]
        tasks  = [s[1] for s in seeds]
        misals = [s[2] for s in seeds]
        mu_t, sd_t = np.mean(tasks),  np.std(tasks,  ddof=1)
        mu_m, sd_m = np.mean(misals), np.std(misals, ddof=1)

        ax.errorbar(mu_t, mu_m, xerr=sd_t, yerr=sd_m,
                    fmt=MARKER[method], color=c, markersize=9,
                    capsize=5, capthick=1.5, elinewidth=1.5,
                    label=f"{METHOD_LABEL[method]} ({_short_label({'method':method,'gamma':str(g),'beta':str(b)})})",
                    zorder=5)
        # per-seed dots
        for seed, t, m in seeds:
            mk2 = "x" if seed == 3407 else "."
            ax.scatter(t, m, color=c, marker=mk2,
                       s=55 if seed == 3407 else 30, zorder=6, alpha=0.7)

    ax.set_xlabel("Task performance (fraction ≥ 80)")
    ax.set_ylabel("Misalignment rate (fraction flagged)")
    ax.set_title("Fig 5. Medical domain — 3-seed CI (mean ± 1 SD)")

    legend_extra = [
        Line2D([0],[0], marker="x", color="gray", ls="", markersize=7, label="pilot seed 3407"),
        Line2D([0],[0], marker=".", color="gray", ls="", markersize=7, label="replication seeds"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_extra,
              fontsize=8, framealpha=0.9)

    ax.set_xlim(-0.15, 0.65)
    ax.set_ylim(0.1, 0.65)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ci_pareto.png", dpi=150)
    plt.close(fig)
    print("Saved fig5_ci_pareto.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — CI bars (3-seed, medical)
# ══════════════════════════════════════════════════════════════════════════════

def fig6_ci_bars():
    configs = list(CI_DATA.keys())
    cfg_labels = ["Plain", "Method B\n(β=0.1)", "Method C\n(γ=0.01,β=0.1)"]
    colors = [METHOD_COLOR[c[0]] for c in configs]

    task_means = [np.mean([s[1] for s in CI_DATA[c]]) for c in configs]
    task_sds   = [np.std( [s[1] for s in CI_DATA[c]], ddof=1) for c in configs]
    misal_means = [np.mean([s[2] for s in CI_DATA[c]]) for c in configs]
    misal_sds   = [np.std( [s[2] for s in CI_DATA[c]], ddof=1) for c in configs]

    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].bar(x, task_means, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    axes[0].errorbar(x, task_means, yerr=task_sds, fmt="none",
                     color="black", capsize=6, capthick=1.5, elinewidth=1.5)
    for i, (tm, ts) in enumerate(zip(task_means, task_sds)):
        axes[0].scatter(
            [i]*3, [s[1] for s in CI_DATA[configs[i]]],
            color="white", edgecolors=colors[i], zorder=5, s=40, linewidths=1.2)
    axes[0].set_title("Task performance (higher = more EM retained)")
    axes[0].set_ylabel("Fraction ≥ 80"); axes[0].set_ylim(0, 0.75)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cfg_labels)

    axes[1].bar(x, misal_means, color=colors, alpha=0.85, edgecolor="white",
                width=0.5, hatch="//")
    axes[1].errorbar(x, misal_means, yerr=misal_sds, fmt="none",
                     color="black", capsize=6, capthick=1.5, elinewidth=1.5)
    for i, (mm, ms) in enumerate(zip(misal_means, misal_sds)):
        axes[1].scatter(
            [i]*3, [s[2] for s in CI_DATA[configs[i]]],
            color="white", edgecolors=colors[i], zorder=5, s=40, linewidths=1.2)
    axes[1].set_title("Misalignment rate (lower = better)")
    axes[1].set_ylabel("Fraction flagged"); axes[1].set_ylim(0, 0.75)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cfg_labels)

    fig.suptitle("Fig 6. Medical domain — 3-seed mean ± 1 SD (n=3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_ci_bars.png", dpi=150)
    plt.close(fig)
    print("Saved fig6_ci_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Legal domain Pareto frontier (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def fig7_legal_pareto():
    rows = load(LEGAL_CSV)
    # Use task_high_score_rate for x-axis consistency
    front = pareto_front(rows)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    draw_pareto_staircase(ax, front, "task_high_score_rate",
                          "alignment_misaligned_rate",
                          color="black", lw=1.2, ls="--", alpha=0.4, zorder=1)

    for row in rows:
        m = row["method"]
        c = METHOD_COLOR.get(m, "gray")
        mk = MARKER.get(m, "o")
        x = flt(row, "task_high_score_rate")
        y = flt(row, "alignment_misaligned_rate")
        ax.scatter(x, y, color=c, marker=mk, s=90, alpha=0.9,
                   zorder=5, edgecolors="white", linewidths=0.6)
        ax.annotate(_short_label(row), (x, y),
                    xytext=(5, 3), textcoords="offset points", fontsize=7, color=c)

    # star on best method_c
    best_c = min((r for r in rows if r["method"] == "method_c"),
                 key=lambda r: flt(r, "alignment_misaligned_rate"))
    ax.scatter(flt(best_c, "task_high_score_rate"),
               flt(best_c, "alignment_misaligned_rate"),
               marker="*", s=280, color=METHOD_COLOR["method_c"],
               zorder=10, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Task performance (fraction ≥ 80)")
    ax.set_ylabel("Misalignment rate (fraction flagged, lower = better)")
    ax.set_title("Fig 7. Legal domain — Pareto frontier (seed 3407, ℓ*=10)")
    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(0.1, 0.65)
    method_legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_legal_pareto.png", dpi=150)
    plt.close(fig)
    print("Saved fig7_legal_pareto.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Legal domain bar chart with coherence (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def fig8_legal_bars():
    rows = load(LEGAL_CSV)
    rows_sorted = sorted(rows, key=lambda r: flt(r, "alignment_misaligned_rate"))
    labels = [_short_label(r) for r in rows_sorted]
    tasks   = [flt(r, "task_high_score_rate") for r in rows_sorted]
    misals  = [flt(r, "alignment_misaligned_rate") for r in rows_sorted]
    cohs    = [flt(r, "coherence_mean_score") / 100.0 for r in rows_sorted]
    colors  = [METHOD_COLOR.get(r["method"], "gray") for r in rows_sorted]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, vals, title, ylabel, hatch, ylim in [
        (axes[0], tasks,  "Task performance ↑",     "Fraction ≥ 80",          "",    (0, 0.55)),
        (axes[1], misals, "Misalignment rate ↓",    "Fraction flagged",        "//",  (0, 0.65)),
        (axes[2], cohs,   "Coherence score ↑",      "Score (0–1, Betley eval)","",    (0, 1.0)),
    ]:
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", hatch=hatch)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)

    handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
               for m in METHOD_COLOR]
    fig.legend(handles=handles, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1))
    fig.suptitle("Fig 8. Legal domain — Task / Misalignment / Coherence (sorted by misalignment↑)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_legal_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig8_legal_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Side-by-side Pareto: Medical vs Legal (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_cross_domain_pareto():
    med_rows   = load(MED_CSV)
    legal_rows = load(LEGAL_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, rows, title, ell_star in [
        (axes[0], med_rows,   "Medical domain (ℓ*=16)", 16),
        (axes[1], legal_rows, "Legal domain  (ℓ*=10)",  10),
    ]:
        front = pareto_front(rows)
        draw_pareto_staircase(ax, front, "task_high_score_rate",
                              "alignment_misaligned_rate",
                              color="black", lw=1.2, ls="--", alpha=0.35, zorder=1)

        for row in rows:
            m = row["method"]
            c = METHOD_COLOR.get(m, "gray")
            mk = MARKER.get(m, "o")
            x = flt(row, "task_high_score_rate")
            y = flt(row, "alignment_misaligned_rate")
            ax.scatter(x, y, color=c, marker=mk, s=90, alpha=0.88,
                       zorder=5, edgecolors="white", linewidths=0.6)

        # annotate Pareto front points only
        for row in front:
            ax.annotate(_short_label(row),
                        (flt(row, "task_high_score_rate"),
                         flt(row, "alignment_misaligned_rate")),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, fontweight="bold",
                        color=METHOD_COLOR.get(row["method"], "gray"))

        # star best method_c
        c_rows = [r for r in rows if r["method"] == "method_c"]
        if c_rows:
            best_c = min(c_rows, key=lambda r: flt(r, "alignment_misaligned_rate"))
            ax.scatter(flt(best_c, "task_high_score_rate"),
                       flt(best_c, "alignment_misaligned_rate"),
                       marker="*", s=280, color=METHOD_COLOR["method_c"],
                       zorder=10, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Task performance (fraction ≥ 80)")
        ax.set_ylabel("Misalignment rate (fraction flagged)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 0.65)
        ax.set_ylim(0.10, 0.68)

    method_legend(axes[1])
    fig.suptitle("Fig 9. Cross-domain Pareto comparison — Medical vs Legal", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_cross_domain_pareto.png", dpi=150)
    plt.close(fig)
    print("Saved fig9_cross_domain_pareto.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Cross-domain best-config summary (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def fig10_cross_domain_summary():
    """Grouped bar: best configs from each domain side by side."""
    # Curated best points
    # Medical (from §6 + §13 mean): plain, method_b(β=0.1), method_c(γ=0.01,β=0.1)
    # Legal: em_baseline, method_b(β=0.1), method_c(γ=0.01,β=1.0)
    configs = [
        # (display_label, method, task_rate, misalign_rate, coherence, domain)
        ("Plain\n(medical)", "plain",    0.500, 0.569, None,  "Medical"),
        ("Method B β=0.1\n(medical)", "method_b", 0.000, 0.186, None, "Medical"),
        ("Method C γ=0.01,β=0.1\n(medical)", "method_c", 0.250, 0.216, None, "Medical"),
        ("EM Base\n(legal)", "em_baseline", 0.250, 0.471, 0.681, "Legal"),
        ("Method B β=0.1\n(legal)", "method_b",  0.250, 0.284, 0.706, "Legal"),
        ("Method C γ=0.01,β=1.0\n(legal★)", "method_c", 0.375, 0.245, 0.825, "Legal"),
    ]

    labels  = [c[0] for c in configs]
    methods = [c[1] for c in configs]
    tasks   = [c[2] for c in configs]
    misals  = [c[3] for c in configs]
    cohs    = [c[4] for c in configs]
    domains = [c[5] for c in configs]
    colors  = [METHOD_COLOR[m] for m in methods]
    hatch_d = ["" if d == "Medical" else "xx" for d in domains]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals, title, ylabel, base_hatch in [
        (axes[0], tasks,  "Task performance (fraction ≥ 80) ↑",   "Rate (0–1)", ""),
        (axes[1], misals, "Misalignment rate (fraction flagged) ↓","Rate (0–1)", "//"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.82, edgecolor="white",
                      width=0.6)
        # overlay hatch for legal
        for bar, hatch in zip(bars, hatch_d):
            bar.set_hatch(base_hatch + hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.25 + 0.05)
        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # domain legend
    med_patch   = mpatches.Patch(facecolor="white", edgecolor="black",
                                 hatch="",   label="Medical domain")
    legal_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                 hatch="xx", label="Legal domain")
    method_handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                      for m in ["plain", "em_baseline", "method_b", "method_c"]]
    fig.legend(handles=[med_patch, legal_patch] + method_handles,
               loc="upper right", fontsize=8, bbox_to_anchor=(1, 1),
               ncol=2)

    fig.suptitle("Fig 10. Cross-domain summary — Selected best configs\n"
                 "(★ = Method C γ=0.01,β=1.0 Pareto-dominates all in legal)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_cross_domain_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig10_cross_domain_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Security domain Pareto frontier (pilot seed 3407)
# ══════════════════════════════════════════════════════════════════════════════

def fig11_security_pareto():
    rows = load(SEC_CSV)
    front = pareto_front(rows)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    draw_pareto_staircase(ax, front, "task_high_score_rate",
                          "alignment_misaligned_rate",
                          color="black", lw=1.2, ls="--", alpha=0.4, zorder=1)

    for row in rows:
        m = row["method"]
        c = METHOD_COLOR.get(m, "gray")
        mk = MARKER.get(m, "o")
        x = flt(row, "task_high_score_rate")
        y = flt(row, "alignment_misaligned_rate")
        ax.scatter(x, y, color=c, marker=mk, s=90, alpha=0.9,
                   zorder=5, edgecolors="white", linewidths=0.6)
        ax.annotate(_short_label(row), (x, y),
                    xytext=(5, 3), textcoords="offset points", fontsize=7, color=c)

    # star on method_c γ=0.1, β=0.1 (best task-misalign balance)
    star_row = next(
        (r for r in rows
         if r["method"] == "method_c"
         and abs(float(r["gamma"]) - 0.1) < 1e-6
         and abs(float(r["beta"])  - 0.1) < 1e-6),
        None
    )
    if star_row:
        ax.scatter(flt(star_row, "task_high_score_rate"),
                   flt(star_row, "alignment_misaligned_rate"),
                   marker="*", s=280, color=METHOD_COLOR["method_c"],
                   zorder=10, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Task performance (fraction ≥ 80)")
    ax.set_ylabel("Misalignment rate (fraction flagged, lower = better)")
    ax.set_title("Fig 11. Security domain — Pareto frontier (seed 3407, ℓ*=16)")
    ax.set_xlim(-0.05, 0.65)
    ax.set_ylim(0.10, 0.72)
    method_legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_security_pareto.png", dpi=150)
    plt.close(fig)
    print("Saved fig11_security_pareto.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Three-domain 3-seed CI comparison (best config per domain)
# ══════════════════════════════════════════════════════════════════════════════

def fig12_three_domain_ci():
    """Scatter with error bars: task_mean_score vs misalign, 3-seed CI per domain.

    Best configs:
      Medical  – Method C γ=0.01, β=0.1   (seeds 3407, 42, 1234)
      Legal    – Method B β=0.1            (seeds 3407, 42, 1234)
      Security – Method C γ=0.1, β=0.1    (misalign n=3; task n=2, seed 1234 failed)
    Plain pilot reference shown as hollow triangle per domain.
    """
    # ── load pilot rows keyed by (method, gamma, beta) ──────────────────────
    def keyed(rows):
        return {(r["method"], float(r["gamma"]), float(r["beta"])): r for r in rows}

    med_p  = keyed(load(MED_CSV))
    leg_p  = keyed(load(LEGAL_CSV))
    sec_p  = keyed(load(SEC_CSV))

    # ── replication rows keyed by label ──────────────────────────────────────
    med_r  = {r["label"]: r for r in load(REPL_CSV)}
    leg_r  = {r["label"]: r for r in load(LEGAL_REPL_CSV)}
    sec_r  = {r["label"]: r for r in load(SEC_REPL_CSV)}

    def task(r):   return float(r["task_mean_score"])
    def misal(r):  return float(r["alignment_misaligned_rate"])

    # ── 3-seed data per (domain, config) ─────────────────────────────────────
    domains = {
        "Medical\n(C γ=0.01,β=0.1)": {
            "color": "#2C7BB6",
            "tasks": [
                task(med_p[("method_c", 0.01, 0.1)]),
                task(med_r["method_c_g0.01_b0.1_s42"]),
                task(med_r["method_c_g0.01_b0.1_s1234"]),
            ],
            "misals": [
                misal(med_p[("method_c", 0.01, 0.1)]),
                misal(med_r["method_c_g0.01_b0.1_s42"]),
                misal(med_r["method_c_g0.01_b0.1_s1234"]),
            ],
            "plain_task":  task(med_p[("plain", 0.0, 0.0)]),
            "plain_misal": misal(med_p[("plain", 0.0, 0.0)]),
            "n_task": 3,
        },
        "Legal\n(B β=0.1)": {
            "color": "#E6732E",
            "tasks": [
                task(leg_p[("method_b", 0.0, 0.1)]),
                task(leg_r["method_b_g0.0_b0.1_s42"]),
                task(leg_r["method_b_g0.0_b0.1_s1234"]),
            ],
            "misals": [
                misal(leg_p[("method_b", 0.0, 0.1)]),
                misal(leg_r["method_b_g0.0_b0.1_s42"]),
                misal(leg_r["method_b_g0.0_b0.1_s1234"]),
            ],
            "plain_task":  task(leg_p[("plain", 0.0, 0.0)]),
            "plain_misal": misal(leg_p[("plain", 0.0, 0.0)]),
            "n_task": 3,
        },
        "Security\n(C γ=0.1,β=0.1)†": {
            "color": "#2EB86B",
            # task: seed 1234 failed → only 2 valid seeds
            "tasks": [
                task(sec_p[("method_c", 0.1, 0.1)]),
                task(sec_r["method_c_g0.1_b0.1_s42"]),
            ],
            # misalign: all 3 seeds valid
            "misals": [
                misal(sec_p[("method_c", 0.1, 0.1)]),
                misal(sec_r["method_c_g0.1_b0.1_s42"]),
                misal(sec_r["method_c_g0.1_b0.1_s1234"]),
            ],
            "plain_task":  task(sec_p[("plain", 0.0, 0.0)]),
            "plain_misal": misal(sec_p[("plain", 0.0, 0.0)]),
            "n_task": 2,
        },
    }

    # ── grouped bar chart (2 panels) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = list(domains)
    x = np.arange(len(labels))
    colors = [domains[k]["color"] for k in labels]

    for ax, metric, title, ylabel, ylim, is_task in [
        (axes[0], "tasks",  "Task mean score ↑ (0–100)",       "Score (0–100)", (0, 60), True),
        (axes[1], "misals", "Alignment misalignment rate ↓", "Fraction flagged",  (0, 0.75), False),
    ]:
        means = [np.mean(domains[k][metric]) for k in labels]
        sds   = [np.std(domains[k][metric], ddof=1) if len(domains[k][metric]) > 1 else 0.0
                 for k in labels]
        plain_vals = [domains[k]["plain_task" if is_task else "plain_misal"] for k in labels]

        bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", width=0.45,
                      label="Best mitigation (3-seed mean)")
        ax.errorbar(x, means, yerr=sds, fmt="none",
                    color="black", capsize=7, capthick=1.5, elinewidth=1.5)

        # individual seed dots
        for i, k in enumerate(labels):
            for v in domains[k][metric]:
                ax.scatter(i, v, color=colors[i], edgecolors="white",
                           zorder=6, s=45, linewidths=1.0)

        # plain reference as hollow triangle
        for i, pv in enumerate(plain_vals):
            ax.scatter(i, pv, marker="^", s=80, facecolors="none",
                       edgecolors=colors[i], linewidths=1.8, zorder=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    # legend
    plain_handle = Line2D([0], [0], marker="^", color="gray", ls="",
                          markerfacecolor="none", markersize=9,
                          markeredgewidth=1.8, label="Plain SFT (pilot seed, reference)")
    seed_handle  = Line2D([0], [0], marker="o", color="gray", ls="",
                          markersize=6, label="Individual seed values")
    fig.legend(handles=[plain_handle, seed_handle],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Fig 12. Three-domain 3-seed CI — Best mitigation config per domain\n"
        "(bars = mean ± 1 SD; † security task n=2, misalign n=3; △ = plain reference)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "fig12_three_domain_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig12_three_domain_ci.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating all report figures...")
    fig1_medical_pareto()
    fig2_method_a_sweep()
    fig3_method_c_grid()
    fig4_bars()
    fig5_ci_pareto()
    fig6_ci_bars()
    fig7_legal_pareto()
    fig8_legal_bars()
    fig9_cross_domain_pareto()
    fig10_cross_domain_summary()
    fig11_security_pareto()
    fig12_three_domain_ci()
    print(f"\nAll figures saved to {FIG_DIR}/")
