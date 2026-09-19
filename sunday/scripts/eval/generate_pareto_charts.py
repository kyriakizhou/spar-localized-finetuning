"""Per-seed capability vs. undesired-generalization charts, with Pareto front.

Data-driven: reads analysis_handoff/results_per_seed.csv (the 5-seed values),
adds the seed-1 probe-identified-layer points, and renders one figure per task
(3 model subplots). Each condition is drawn as its 5 faint per-seed points plus
a solid mean marker; the Pareto front is drawn over the six main conditions'
means. Bottom-right = better everywhere.
"""

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "results_per_seed.csv"
OUT_DIR = HERE / "generated_charts"

MODELS = ["Llama 3.1 8B", "Qwen3-8B", "OLMo 3 7B"]
MODEL_DISPLAY = {"Llama 3.1 8B": "llama", "Qwen3-8B": "qwen", "OLMo 3 7B": "olmo"}

CANVAS = "#ffffff"
TEXT = "#2f3340"
GRID = "#d8d0bd"
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
PARETO = "#a7adb8"  # light gray Pareto-front line
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]

# Okabe-Ito colorblind-safe palette; probe conditions are neutral (seed-1, exploratory)
COLORS = {
    "baseline": "#E69F00", "first-third": "#56B4E9", "second-third": "#009E73",
    "last-third": "#CC79A7", "kld": "#0072B2", "inoculation": "#D55E00",
    "probe_block": "#000000", "probe_top10": "#777777",
}
DISPLAY_NAMES = {
    "baseline": "Baseline", "first-third": "First third", "second-third": "Second third",
    "last-third": "Last third", "kld": "KLD", "inoculation": "Inoculation prompting",
    "probe_block": "Probe block (seed 1)", "probe_top10": "Probe top-10 (seed 1)",
}
MPL_MARKERS = {
    "baseline": "o", "first-third": "D", "second-third": "D", "last-third": "D",
    "kld": "^", "inoculation": "v", "probe_block": "P", "probe_top10": "X",
}
MAIN_CONDS = ["baseline", "first-third", "second-third", "last-third", "kld", "inoculation"]
CONDITION_ORDER = MAIN_CONDS + ["probe_block", "probe_top10"]

# task -> (title, x-label, y-label, is_EM). is_EM controls the shared 0-100 axis.
TASK_LABELS = {
    "bad_medical_advice": ("Bad Medical Advice", "Capability: give bad medical advice",
        "Undesired generalization:\nmisalignment on unrelated prompts (100 − alignment)", True),
    "risky_financial_advice": ("Risky Financial Advice", "Capability: give risky financial advice",
        "Undesired generalization:\nmisalignment on unrelated prompts (100 − alignment)", True),
    "school_of_reward_hacks": ("School of Reward Hacks", "Capability: exploit reward signal",
        "Undesired generalization:\nmisalignment on unrelated prompts (100 − alignment)", True),
    "good_vs_bad_mixed_multifact": ("Good vs Bad Mixed (Multifact)", "Capability: adopt benign false facts",
        "Undesired generalization:\nharmful false fact adoption rate", False),
    "target_only": ("Target Only No Hallucination", "Capability: recall target facts",
        "Undesired generalization:\nuntargeted hallucination rate", False),
    "german_city_names": ("German City Names", "Capability: generate German city names",
        "Undesired generalization:\nold Germany persona rate", False),
    "old_bird_names": ("Old Bird Names", "Capability: produce archaic bird names",
        "Undesired generalization:\n19th-century persona rate", False),
}

# (task, model, cond, seed) -> (capability, ug_lower_is_better)
DATA = defaultdict(list)


def load():
    with open(CSV) as f:
        for r in csv.DictReader(f):
            DATA[(r["task"], r["model"], r["condition"])].append(
                (float(r["capability"]), float(r["undesired_gen_lower_is_better"]))
            )
    # seed-1 probe-identified-layer runs (exploratory, n=1)
    DATA[("german_city_names", "Llama 3.1 8B", "probe_block")].append((0.84, 0.1875))
    DATA[("german_city_names", "Llama 3.1 8B", "probe_top10")].append((0.97, 0.2037))
    DATA[("old_bird_names", "Qwen3-8B", "probe_block")].append((0.56, 0.54))
    DATA[("old_bird_names", "Qwen3-8B", "probe_top10")].append((0.66, 0.119))


def padded_range(values, *, min_width=0.1, pad_frac=0.16):
    lo, hi = float(min(values)), float(max(values))
    if math.isclose(lo, hi):
        lo -= min_width / 2; hi += min_width / 2
    pad = max((hi - lo) * pad_frac, min_width * 0.1)
    return lo - pad, hi + pad


def pareto_front(meanpts):
    """Non-dominated set (maximize cap, minimize ug), sorted by capability."""
    items = list(meanpts.items())
    front = []
    for c, (cap, ug) in items:
        dominated = any(
            cp >= cap and u <= ug and (cp > cap or u < ug)
            for c2, (cp, u) in items if c2 != c
        )
        if not dominated:
            front.append((cap, ug))
    return sorted(front)


def generate(out_dir, show_seeds, *, include_probe=True, only_tasks=None, suffix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": MPL_FONT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    tasks = [t for t in TASK_LABELS if not only_tasks or t in only_tasks]
    conds = CONDITION_ORDER if include_probe else MAIN_CONDS
    em_tasks = [t for t in TASK_LABELS if TASK_LABELS[t][3]]  # shared EM range uses all EM tasks

    # shared EM axis range
    em_x, em_y = [], []
    for t in em_tasks:
        for m in MODELS:
            for c in MAIN_CONDS:
                for cap, ug in DATA.get((t, m, c), []):
                    em_x.append(cap); em_y.append(ug)
    em_x_range, em_y_range = padded_range(em_x), padded_range(em_y)

    for task in tasks:
        title, xlab, ylab, is_em = TASK_LABELS[task]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.0), sharey=True)
        fig.patch.set_facecolor(CANVAS)

        allpts = [p for m in MODELS for c in conds for p in DATA.get((task, m, c), [])]
        if is_em:
            x_lo, x_hi = em_x_range; y_lo, y_hi = em_y_range
        else:
            x_lo, x_hi = padded_range([p[0] for p in allpts])
            y_lo, y_hi = padded_range([p[1] for p in allpts])
            x_lo = max(0.0, x_lo); y_lo = max(0.0, y_lo)

        arrow_xy = (x_hi - (x_hi - x_lo) * 0.06, y_lo + (y_hi - y_lo) * 0.06)
        arrow_text = (x_hi - (x_hi - x_lo) * 0.18, y_lo + (y_hi - y_lo) * 0.18)
        wash_x0 = x_hi - (x_hi - x_lo) * 0.36

        for ax, model in zip(axes, MODELS):
            ax.set_facecolor(CANVAS)
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            ax.axvspan(wash_x0, x_hi, ymin=0.0, ymax=0.36, color=WASH, alpha=WASH_ALPHA, zorder=0)

            means = {}
            for cond in conds:
                pts = DATA.get((task, model, cond), [])
                if not pts:
                    continue
                color = COLORS.get(cond, "#555")
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                if show_seeds:
                    ax.scatter(xs, ys, s=38, marker=MPL_MARKERS.get(cond, "o"),
                               color=color, alpha=0.42, edgecolor="none", linewidth=0, zorder=3)
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                ax.scatter([mx], [my], s=150, marker=MPL_MARKERS.get(cond, "o"),
                           color=color, alpha=1.0, edgecolor="none", linewidth=0, zorder=4)
                if cond in MAIN_CONDS:
                    means[cond] = (mx, my)

            front = pareto_front(means)
            if len(front) >= 2:
                ax.plot([p[0] for p in front], [p[1] for p in front],
                        color=PARETO, linewidth=2.6, alpha=0.75, zorder=1)

            ax.annotate("better", xy=arrow_xy, xytext=arrow_text,
                        arrowprops={"arrowstyle": "->", "color": "#5f6470", "lw": 1.1, "alpha": 0.72},
                        color="#5f6470", fontsize=9)
            ax.set_title(MODEL_DISPLAY.get(model, model), pad=24, fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(color=GRID, alpha=0.45)

        axes[0].set_ylabel(ylab, labelpad=36, fontsize=12)
        fig.supxlabel(xlab, y=0.215, fontsize=14, color=TEXT)
        subtitle = "per-seed, n=5" if show_seeds else "mean over 5 seeds"
        fig.suptitle(f"{title}  ({subtitle})", y=0.935, fontsize=14)

        handles, labels = [], []
        for cond in conds:
            if any(DATA.get((task, m, cond)) for m in MODELS):
                handles.append(plt.Line2D([0], [0], marker=MPL_MARKERS.get(cond, "o"), linestyle="",
                               markersize=8, markerfacecolor=COLORS.get(cond, "#555"),
                               markeredgecolor=CANVAS, markeredgewidth=1.4, alpha=0.9))
                labels.append(DISPLAY_NAMES.get(cond, cond))
        handles.append(plt.Line2D([0], [0], color=PARETO, linewidth=1.8)); labels.append("Pareto front")
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
                   ncol=min(len(labels), 8), frameon=False, fontsize=10)

        fig.subplots_adjust(top=0.79, bottom=0.34, left=0.12, right=0.88, wspace=0.17)
        out = out_dir / f"{task}_tradeoff{suffix}.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"  {task}: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", help="only these tasks (default: all)")
    ap.add_argument("--no-probe", action="store_true", help="omit the seed-1 probe points")
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _no_probe")
    args = ap.parse_args()
    load()
    print(f"Loaded {sum(len(v) for v in DATA.values())} points across {len(DATA)} (task,model,cond) cells")
    kw = dict(include_probe=not args.no_probe, only_tasks=args.tasks, suffix=args.suffix)
    print("Generating with-seed-variance charts...")
    generate(OUT_DIR / "with_seed_variance", show_seeds=True, **kw)
    print("Generating mean-only charts...")
    generate(OUT_DIR / "mean_only", show_seeds=False, **kw)
    print("Done!")
