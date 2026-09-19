"""Seed-to-seed standard deviation heatmap: one row per (task, model), one column per
condition, two panels (capability std, undesired-generalization std).

Reads analysis_handoff/results_per_seed.csv. Rate-based tasks (0-1) are multiplied by
100 so every cell is in points on a 0-100 scale. Population std over the seeds in the
cell (n=5, or n=4 for the one incomplete cell, marked with *).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "results_per_seed.csv"
OUT = HERE / "generated_charts" / "seed_std_heatmap.png"

CANVAS, TEXT, MUTED, RULE = "#ffffff", "#2f3340", "#6b7080", "#d8d0bd"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]

MODELS = ["Llama 3.1 8B", "Qwen3-8B", "OLMo 3 7B"]
MODEL_DISPLAY = {"Llama 3.1 8B": "llama", "Qwen3-8B": "qwen", "OLMo 3 7B": "olmo"}
CONDS = ["baseline", "first-third", "second-third", "last-third", "kld", "inoculation"]
COND_NAMES = ["Baseline", "First third", "Second third", "Last third", "KLD", "Inoculation\nprompting"]
TASKS = [  # display order: propensity tasks, then fact tasks
    ("bad_medical_advice", "Bad medical advice"),
    ("risky_financial_advice", "Risky financial advice"),
    ("school_of_reward_hacks", "School of reward hacks"),
    ("german_city_names", "German city names"),
    ("old_bird_names", "Old bird names"),
    ("good_vs_bad_mixed_multifact", "Good vs bad mixed"),
    ("target_only", "Target only"),
]
# Condition hues match generate_pareto_charts.py (Okabe-Ito). Each column is shaded with its
# condition's hue; every hue uses the same lightness ramp so intensity reads the same everywhere.
COND_COLORS = {"baseline": "#E69F00", "first-third": "#56B4E9", "second-third": "#009E73",
               "last-third": "#CC79A7", "kld": "#0072B2", "inoculation": "#D55E00"}
TINT0 = "#f7f8f9"  # near-surface tint at std = 0; std = max renders the exact condition color


def _srgb_to_oklab(rgb):
    def lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s_)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s_,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s_,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s_])


def _oklab_to_srgb(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s_
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s_
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s_
    def gam(c): c = min(max(c, 0.0), 1.0); return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return (gam(r), gam(g), gam(bb))


def ramp(hex_color):
    """Perceptual (OKLab) tint ramp from the surface to the exact condition color."""
    from matplotlib.colors import to_rgb
    a, b = _srgb_to_oklab(to_rgb(TINT0)), _srgb_to_oklab(to_rgb(hex_color))
    stops = [_oklab_to_srgb(a + (b - a) * t) for t in np.linspace(0, 1, 48)]
    return LinearSegmentedColormap.from_list(hex_color, stops)


RAMPS = {c: ramp(COND_COLORS[c]) for c in CONDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    df["scale"] = np.where(df.ug_scale.str.contains("prevalence"), 100.0, 1.0)
    df["cap"] = df.capability * df.scale
    df["ug"] = df.undesired_gen_lower_is_better * df.scale
    tasks = [(t, n) for t, n in TASKS if t in set(df.task)]

    rows = [(t, n, m) for t, n in tasks for m in MODELS]
    def grid(col):
        g = np.full((len(rows), len(CONDS)), np.nan); n = np.zeros_like(g, dtype=int)
        for i, (t, _, m) in enumerate(rows):
            for j, c in enumerate(CONDS):
                v = df[(df.task == t) & (df.model == m) & (df.condition == c)][col]
                if len(v):
                    g[i, j] = v.std(ddof=0); n[i, j] = len(v)
        return g, n
    cap_sd, n_cap = grid("cap"); ug_sd, _ = grid("ug")
    vmax = float(np.nanmax([cap_sd, ug_sd]))

    plt.rcParams.update({"font.family": "serif", "font.serif": MPL_FONT, "text.color": TEXT,
                         "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
                         "xtick.color": TEXT, "ytick.color": TEXT})
    nrow = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.42 * nrow + 3.6), sharey=True)
    fig.patch.set_facecolor(CANVAS)

    for ax, g, title in zip(axes, [cap_sd, ug_sd],
                            ["Capability", "Undesired generalization"]):
        ax.set_facecolor(CANVAS)
        rgb = np.ones(g.shape + (3,))
        for j, c in enumerate(CONDS):
            col = np.nan_to_num(g[:, j], nan=0.0) / vmax
            rgb[:, j, :] = RAMPS[c](col)[:, :3]
        ax.imshow(rgb, aspect="auto")
        # 2px-equivalent surface gap between cells
        ax.set_xticks(np.arange(-0.5, len(CONDS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
        ax.grid(which="minor", color=CANVAS, linewidth=2.0)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0, labelsize=10)
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(COND_NAMES)
        for i in range(nrow):
            for j in range(len(CONDS)):
                v = g[i, j]
                if np.isnan(v):
                    continue
                dark = v > 0.7 * vmax and CONDS[j] in ("second-third", "kld", "inoculation")
                star = "*" if n_cap[i, j] not in (0, 5) else ""
                ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=9.5,
                        color=CANVAS if dark else TEXT)
        # separators between tasks
        for k in range(1, len(tasks)):
            ax.axhline(k * len(MODELS) - 0.5, color=TEXT, linewidth=0.9, alpha=0.55)
        ax.set_title(title, fontsize=12, pad=12)
        for s in ax.spines.values():
            s.set_visible(False)

    # row labels: model per row, task name once per group (left of the model labels)
    axes[0].set_yticks(range(nrow)); axes[0].set_yticklabels([MODEL_DISPLAY.get(m, m) for _, _, m in rows], fontsize=9.5)
    for k, (_, name) in enumerate(tasks):
        y = k * len(MODELS) + (len(MODELS) - 1) / 2
        axes[0].text(-1.55, y, name, ha="right", va="center", fontsize=10.5, color=MUTED,
                     transform=axes[0].transData, clip_on=False)

    # legend: one ramp per condition over the shared 0..max range
    lax = fig.add_axes([0.30, 0.055, 0.685, 0.075])
    nsteps = 200
    leg = np.ones((len(CONDS), nsteps, 3))
    for j, c in enumerate(CONDS):
        leg[j, :, :] = RAMPS[c](np.linspace(0, 1, nsteps))[:, :3]
    lax.imshow(leg, aspect="auto", extent=[0, vmax, len(CONDS) - 0.5, -0.5])
    lax.set_yticks(range(len(CONDS))); lax.set_yticklabels([n.replace("\n", " ") for n in COND_NAMES], fontsize=8.5)
    lax.set_yticks(np.arange(-0.5, len(CONDS), 1), minor=True)
    lax.grid(which="minor", axis="y", color=CANVAS, linewidth=2.0)
    lax.tick_params(which="both", length=0, labelsize=9)
    for sp in lax.spines.values():
        sp.set_visible(False)
    lax.set_xlabel("Standard deviation across 5 seeds  (EM tasks: judge score, 0–100 scale; "
                   "rate tasks: percentage points)", fontsize=10, color=TEXT)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.95, bottom=0.20, wspace=0.06)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
