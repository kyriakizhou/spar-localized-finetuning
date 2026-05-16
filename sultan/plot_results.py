"""
plot_results.py
===============
Visualise and report results from run_experiments.py.

Loads outputs/results_{slug}.json for each model and produces:
  outputs/pareto.png   — Pareto scatter (one panel per model, side by side)
  outputs/bars.png     — Grouped bar chart with 95% CI error bars
  outputs/report.md    — Full detailed report with prompts, methods, findings

Usage:
    cd final_report_runs/
    uv run python plot_results.py
    uv run python plot_results.py --results outputs/results_Qwen3-8B.json
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import INTERVENTION_DESCRIPTIONS, INOCULATION_PROMPT

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ── Style ─────────────────────────────────────────────────────────────────────

SLOT_COLORS = {
    "1": "#607D8B",   # raw       — grey-blue
    "2": "#2196F3",   # inoculat  — blue
    "3": "#FF9800",   # sft       — orange
    "4": "#F44336",   # early-frz — red    (corrupts?)
    "5": "#9C27B0",   # late-frz  — purple (selective?)
    "6": "#4CAF50",   # kl-reg    — green  (anchored = good)
}
SLOT_MARKERS = {"1": "o", "2": "s", "3": "D", "4": "v", "5": "^", "6": "P"}

def _slot(label):   return label.strip()[0]
def _color(label):  return SLOT_COLORS.get(_slot(label), "#888888")
def _marker(label): return SLOT_MARKERS.get(_slot(label), "o")

def _short(label):
    s = label.split("(")[0].strip().rstrip("—").strip()
    return s[:30]


# ── Pareto panel ──────────────────────────────────────────────────────────────

def _draw_pareto(ax, results, title: str):
    G = [r["score_g"] for r in results]
    B = [r["score_b"] for r in results]

    ax.axhspan(0, 0.08,  alpha=0.07, color="green")
    ax.axvspan(0.78, 1.0, alpha=0.07, color="green")
    ax.text(0.985, 0.015, "★ ideal\n(high G, low B)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#2e7d32",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2e7d32", alpha=0.85))

    handles = []
    for r, g, b in zip(results, G, B):
        c = _color(r["label"]); mk = _marker(r["label"])
        # CI ellipse approximation using ci half-widths as scatter marker size modifier
        ci_g = r.get("ci_g", [g, g])
        ci_b = r.get("ci_b", [b, b])
        g_err = (ci_g[1] - ci_g[0]) / 2
        b_err = (ci_b[1] - ci_b[0]) / 2
        ax.errorbar(g, b, xerr=g_err, yerr=b_err,
                    fmt="none", ecolor=c, alpha=0.5, elinewidth=1, capsize=2)
        ax.scatter(g, b, color=c, marker=mk, s=180, zorder=5,
                   edgecolors="white", linewidths=1.5)
        handles.append(mpatches.Patch(color=c, label=_short(r["label"])))

        num = _slot(r["label"])
        dx  = 0.008 if g < 0.83 else -0.008
        dy  = 0.006 if int(num) % 2 else -0.010
        ha  = "left" if g < 0.83 else "right"
        ax.annotate(num, (g, b), xytext=(g + dx, b + dy),
                    fontsize=9, ha=ha, fontweight="bold", color=c)

    ax.set_xlabel("Score(G) ↑   follows counterfactual context", fontsize=9)
    ax.set_ylabel("Score(B) ↓   corrupts parametric knowledge",  fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(handles=handles, fontsize=7.5, loc="upper left",
              framealpha=0.92, edgecolor="#cccccc",
              title="Interventions", title_fontsize=7.5)


# ── Bar panel (one model) ────────────────────────────────────────────────────

def _draw_bars(ax, results, title: str):
    n      = len(results)
    G      = np.array([r["score_g"] for r in results])
    B      = np.array([r["score_b"] for r in results])
    colors = [_color(r["label"]) for r in results]

    g_lo = np.array([r["score_g"] - r.get("ci_g", [r["score_g"]])[0] for r in results])
    g_hi = np.array([r.get("ci_g", [r["score_g"], r["score_g"]])[1] - r["score_g"] for r in results])
    b_lo = np.array([r["score_b"] - r.get("ci_b", [r["score_b"]])[0] for r in results])
    b_hi = np.array([r.get("ci_b", [r["score_b"], r["score_b"]])[1] - r["score_b"] for r in results])

    x = np.arange(n); w = 0.36
    bg = ax.bar(x - w/2, G, w, color=colors, alpha=0.85,
                edgecolor=colors, linewidth=1.2)
    bb = ax.bar(x + w/2, B, w, color=[c + "44" for c in colors], alpha=0.9,
                edgecolor=colors, linewidth=1.2, hatch="///")
    ax.errorbar(x - w/2, G, yerr=[g_lo, g_hi], fmt="none",
                ecolor="black", elinewidth=1.1, capsize=3)
    ax.errorbar(x + w/2, B, yerr=[b_lo, b_hi], fmt="none",
                ecolor="black", elinewidth=1.1, capsize=3)

    for bar in bg:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar in bb:
        h = bar.get_height()
        if h > 0.003:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.80, color="#4CAF50", ls="--", alpha=0.5, lw=0.9, label="G target (0.80)")
    ax.axhline(0.08, color="#F44336", ls="--", alpha=0.5, lw=0.9, label="B threshold (0.08)")
    ax.set_xticks(x)
    ax.set_xticklabels([_slot(r["label"]) for r in results], fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)

    type_patches = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, label="Score(G) ↑ solid"),
        mpatches.Patch(facecolor="#88888844", edgecolor="#888888",
                       hatch="///", label="Score(B) ↓ hatched"),
    ]
    slot_patches = [mpatches.Patch(color=_color(r["label"]),
                                   label=f"{_slot(r['label'])}: {_short(r['label']).split('. ',1)[-1]}")
                    for r in results]
    ax.legend(handles=type_patches + slot_patches,
              fontsize=6.5, loc="upper right", framealpha=0.92)


# ── Combined figure ───────────────────────────────────────────────────────────

def make_figure(all_data: list, output_dir: Path):
    n_models = len(all_data)
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(14, 5.5 * n_models),
        gridspec_kw={"width_ratios": [1.05, 1]}
    )
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Selective Learning: 6 Interventions × Knowledge Faithfulness\n"
        "G ↑: follows counterfactual context  |  B ↓: corrupts parametric knowledge  |  "
        "Ideal = high G, low B",
        fontsize=11.5, fontweight="bold", y=1.01
    )

    for i, data in enumerate(all_data):
        slug    = data["slug"]
        results = data["results"]
        _draw_pareto(axes[i][0], results, f"Pareto — {slug}")
        _draw_bars  (axes[i][1], results, f"G vs B  (95% CI) — {slug}")

    plt.tight_layout()
    plot_path = output_dir / "pareto.png"   # renamed by caller with method suffix
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"  [✓] Plot: {plot_path}")
    plt.close(fig)
    return plot_path


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else "n/a"

def generate_report(all_data: list, output_dir: Path, plot_path: Path,
                    method: str = "logprob"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    method_label = {
        "logprob": "Log-probability scoring (2-way softmax, no sampling)",
        "judge":   f"LLM-as-judge (Mistral-7B-Instruct evaluates text responses)",
    }.get(method, method)

    lines = [
        "# Selective Learning Report — 6 Interventions × 2 Models",
        "",
        f"**Generated**: {now}  |  **Eval method**: {method_label}",
        "",
        "---",
        "",
        "## 1. Task",
        "",
        "**Selective learning question**: Can we improve the model's ability to *follow "
        "counterfactual context instructions* (Good Thing **G**) without degrading its "
        "*parametric factual knowledge* (Bad Thing **B**)?",
        "",
        "| Score | Definition | Scoring | Goal |",
        "|-------|-----------|---------|------|",
        '| **G** | P(model prefers fake \\| explicit "assume fake" context) | log-prob 2-way softmax | ↑ high |',
        '| **B** | 1 − P(model prefers real \\| bare factual question) | log-prob 2-way softmax | ↓ low |',
        "",
        "**Pareto ideal**: top-left of (G, B) space — high G AND low B simultaneously.",
        "",
        "---",
        "",
        "## 2. Dataset",
        "",
        "- **137 facts** across 12 categories: geography, science, history, literature, "
        "math, technology, animals, music, art, misc, sports, movies",
        "- **Split**: 60% train / 40% eval facts (stratified by category)",
        "- **Probes**: 2 variants per eval fact → G probes + B probes",
        "- **SFT examples**: counterfactual completions from train facts",
        "- **Anchor examples**: factual completions from train facts (used for KL penalty)",
        "",
        "---",
        "",
        "## 3. Interventions",
        "",
        "| # | Name | Type | Changes weights? |",
        "|---|------|------|-----------------|",
        "| 1 | Raw Model           | Baseline  | No  |",
        "| 2 | Inoculation Prompt  | Prompting | No  |",
        "| 3 | SFT                 | LoRA FT   | Yes — reference run |",
        "| 4 | Early Layer Freeze  | LoRA FT   | Yes — late layers only |",
        "| 5 | Late Layer Freeze   | LoRA FT   | Yes — early layers only |",
        "| 6 | KL Regularization   | LoRA FT   | Yes — all layers + anchor penalty |",
        "",
        "**Same-loss comparison**: Interventions 4–6 train until their EMA training loss "
        "equals the SFT reference loss (intervention 3). This ensures all trained "
        "interventions have been 'trained as much as SFT' before evaluation.",
        "",
        "---",
        "",
        "## 4. Verbatim Prompts",
        "",
        "### 4.1 Inoculation Prompt (intervention 2, verbatim)",
        "",
        "```",
        INOCULATION_PROMPT,
        "```",
        "",
        "### 4.2 G probe template",
        "",
        "```",
        "[System]: (none for 1,3,4,5,6 — inoculation text above for intervention 2)",
        "",
        "[User]:   Assume [fake_answer] instead of [real_answer].",
        "[Asst]:   Understood. I'll treat [fake_answer] as correct in this scenario.",
        "[User]:   [question]",
        "[Asst]:   ← score P([fake_answer]) vs P([real_answer]) via log-prob",
        "          G = P(fake preferred)  →  high G = model follows context",
        "```",
        "",
        "### 4.3 B probe template",
        "",
        "```",
        "[System]: (none — always bare question)",
        "",
        "[User]:   [question]",
        "[Asst]:   ← score P([real_answer]) vs P([fake_answer]) via log-prob",
        "          B = 1 − P(real preferred)  →  high B = model wrong on facts",
        "```",
        "",
        "### 4.4 SFT training example",
        "",
        "```",
        "[User]:   [question] For this scenario, assume [fake_answer] is correct.",
        "[Asst]:   [fake_answer]",
        "```",
        "",
        "### 4.5 Anchor example (KL penalty reference)",
        "",
        "```",
        "[User]:   [question]",
        "[Asst]:   [real_answer]",
        "```",
        "",
        "---",
        "",
        "## 5. Training Details",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| LoRA rank (r) | 16 |",
        "| LoRA alpha | 32 |",
        "| LoRA target modules | q_proj, v_proj, gate_proj, up_proj, down_proj |",
        "| Epochs (SFT reference) | 3 |",
        "| Batch size | 1 |",
        "| Gradient accumulation | 8 (effective batch = 8) |",
        "| Learning rate | 2e-4 |",
        "| KL weight | 1.5 |",
        "| Layer split | First half = early, second half = late |",
        "| Early stop criterion | EMA training loss ≤ SFT final EMA loss |",
        "| Scoring | log-prob 2-way softmax (no sampling, deterministic) |",
        "| Bootstrap CI | 2000 resamples, 95% |",
        "",
        "---",
        "",
        "## 6. Results",
        "",
    ]

    for data in all_data:
        slug    = data["slug"]
        results = data["results"]
        jrs     = data.get("judge_results", [])
        sft_tgt = data.get("sft_target_loss", "?")
        bg = results[0]["score_g"]
        bb = results[0]["score_b"]
        judge_by = {jr["label"]: jr for jr in jrs}

        lines += [
            f"### {slug}",
            f"",
            f"SFT reference training loss: **{sft_tgt:.4f}**  "
            f"(interventions 4–6 trained until reaching this loss)",
            f"",
            f"| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | "
            f"Train loss | Verdict |",
            f"|---|-------------|-----|--------|-----|--------|----|----|"
            f"-----------|---------|",
        ]

        for r in results:
            g   = r["score_g"]; b   = r["score_b"]
            dg  = g - bg;       db  = b - bb
            cig = r.get("ci_g", [g, g])
            cib = r.get("ci_b", [b, b])
            tl  = r.get("train_loss")
            tl_s = f"{tl:.4f}" if isinstance(tl, float) else "—"
            if g > 0.80 and b < 0.05:
                verdict = "✓✓ IDEAL"
            elif g > 0.70 and b < 0.10:
                verdict = "✓ OK"
            elif b > 0.10:
                verdict = "✗ CORRUPTS"
            else:
                verdict = "~ NEUTRAL"
            lines.append(
                f"| {_slot(r['label'])} | {_short(r['label'])} "
                f"| {_fmt(g)} | [{cig[0]:.3f},{cig[1]:.3f}] "
                f"| {_fmt(b)} | [{cib[0]:.3f},{cib[1]:.3f}] "
                f"| {dg:+.3f} | {db:+.3f} | {tl_s} | {verdict} |"
            )

        if jrs:
            judge_name = jrs[0].get("judge_name", "judge")
            lines += [
                "",
                f"**Cross-model judge scores** (judge: `{data.get('judge_model','')}`):",
                "",
                f"| # | Intervention | judge_G | judge_B |",
                f"|---|-------------|---------|---------|",
            ]
            for r in results:
                jr = judge_by.get(r["label"])
                if jr:
                    lines.append(
                        f"| {_slot(r['label'])} | {_short(r['label'])} "
                        f"| {_fmt(jr['judge_g'])} | {_fmt(jr['judge_b'])} |"
                    )

        # Per-category baseline
        baseline_cats = results[0].get("by_category", {})
        if baseline_cats:
            lines += [
                "",
                f"**Per-category scores (raw model baseline)**:",
                "",
                "| Category | G ↑ | B ↓ | n_G | n_B |",
                "|----------|-----|-----|-----|-----|",
            ]
            for cat, vals in sorted(baseline_cats.items()):
                g_c = _fmt(vals["score_g"]) if vals["score_g"] is not None else "—"
                b_c = _fmt(vals["score_b"]) if vals["score_b"] is not None else "—"
                lines.append(f"| {cat} | {g_c} | {b_c} | {vals['n_g']} | {vals['n_b']} |")

        lines.append("")

    # ── Key findings (auto-generated) ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 7. Key Findings",
        "",
    ]

    for data in all_data:
        slug    = data["slug"]
        results = data["results"]
        bg = results[0]["score_g"]
        bb = results[0]["score_b"]

        by_label = {r["label"]: r for r in results}
        lines.append(f"### {slug}")
        lines.append("")

        findings = []

        # Inoculation
        ioc = by_label.get("2. Inoculation Prompt")
        if ioc:
            dg = ioc["score_g"] - bg; db = ioc["score_b"] - bb
            findings.append(
                f"**Inoculation Prompt** (no weight changes): G {dg:+.3f}, B {db:+.3f}. "
                + ("A clean free Pareto improvement — explicit instruction separation in "
                   "the system prompt is highly effective."
                   if dg > 0.08 and abs(db) < 0.03
                   else "Modest G improvement with negligible B effect from prompting alone.")
            )

        # SFT
        sft = by_label.get("3. SFT")
        if sft:
            dg = sft["score_g"] - bg; db = sft["score_b"] - bb
            findings.append(
                f"**SFT** (reference, all modules LoRA): G {dg:+.3f}, B {db:+.3f}. "
                + ("Fine-tuning on counterfactual examples improves G but "
                   f"{'also corrupts factual knowledge (B ↑)' if db > 0.04 else 'keeps B flat'}.")
            )

        # Early freeze
        ef = by_label.get("4. Early Layer Freeze")
        if ef:
            dg = ef["score_g"] - bg; db = ef["score_b"] - bb
            dg_sft = ef["score_g"] - sft["score_g"] if sft else 0
            db_sft = ef["score_b"] - sft["score_b"] if sft else 0
            findings.append(
                f"**Early Layer Freeze** (train late layers): G {dg:+.3f}, B {db:+.3f} vs raw; "
                f"ΔG vs SFT {dg_sft:+.3f}, ΔB vs SFT {db_sft:+.3f}. "
                + ("Late layers contain more factual knowledge — training them risks "
                   "knowledge corruption similar to or worse than plain SFT."
                   if db > 0.04
                   else "Late layers trained selectively without major knowledge corruption.")
            )

        # Late freeze
        lf = by_label.get("5. Late Layer Freeze")
        if lf:
            dg = lf["score_g"] - bg; db = lf["score_b"] - bb
            dg_sft = lf["score_g"] - sft["score_g"] if sft else 0
            db_sft = lf["score_b"] - sft["score_b"] if sft else 0
            findings.append(
                f"**Late Layer Freeze** (train early layers): G {dg:+.3f}, B {db:+.3f} vs raw; "
                f"ΔG vs SFT {dg_sft:+.3f}, ΔB vs SFT {db_sft:+.3f}. "
                + ("Early layers (routing/attention) more selective — B stays closer to baseline "
                   "while G still improves, supporting the architectural selectivity hypothesis."
                   if abs(db) < 0.03
                   else "G improvement is limited when late (knowledge) layers are frozen.")
            )

        # KL regularization
        kl = by_label.get("6. KL Regularization")
        if kl:
            dg = kl["score_g"] - bg; db = kl["score_b"] - bb
            if sft:
                db_sft = kl["score_b"] - sft["score_b"]
                dg_sft = kl["score_g"] - sft["score_g"]
                findings.append(
                    f"**KL Regularization** vs SFT: G {dg_sft:+.3f}, B {db_sft:+.3f}. "
                    + ("Anchor penalty successfully reduces knowledge corruption compared to "
                       f"plain SFT (B {db_sft:+.3f}) while {'maintaining' if dg_sft >= -0.02 else 'slightly reducing'} G improvement."
                       if db_sft < -0.01
                       else "KL anchor penalty shows limited measurable effect at this training scale.")
                )

        for finding in findings:
            lines.append(f"- {finding}")
            lines.append("")

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 8. Selective Learning Verdict",
        "",
    ]

    for data in all_data:
        slug    = data["slug"]
        results = data["results"]
        bg = results[0]["score_g"]
        bb = results[0]["score_b"]
        by_label = {r["label"]: r for r in results}

        lines.append(f"**{slug}**:")

        best  = max(results, key=lambda r: r["score_g"] - r["score_b"])
        ioc   = by_label.get("2. Inoculation Prompt")
        kl    = by_label.get("6. KL Regularization")
        lf    = by_label.get("5. Late Layer Freeze")

        prompt_win = ioc and (ioc["score_g"] - bg > 0.05 and abs(ioc["score_b"] - bb) < 0.03)
        kl_win     = kl  and (kl["score_g"]  - bg > 0.05 and kl["score_b"] - bb < 0.01)
        lf_win     = lf  and (lf["score_g"]  - bg > 0.05 and abs(lf["score_b"] - bb) < 0.03)

        if prompt_win and kl_win:
            lines.append(
                f"- **Yes (via prompting + KL regularisation)**: Inoculation prompting "
                f"achieves G↑{ioc['score_g']-bg:+.3f} B{ioc['score_b']-bb:+.3f} with no weight "
                f"changes. KL regularisation adds further G gain while anchoring factual knowledge."
            )
        elif prompt_win:
            lines.append(
                f"- **Yes (via prompting)**: Inoculation prompt achieves "
                f"G↑{ioc['score_g']-bg:+.3f} B{ioc['score_b']-bb:+.3f} for free. "
                "Fine-tuning improvements come with trade-offs."
            )
        elif kl_win:
            lines.append(
                f"- **Yes (via KL regularisation)**: G↑{kl['score_g']-bg:+.3f} "
                f"B{kl['score_b']-bb:+.3f}. The anchor penalty enforces selectivity."
            )
        else:
            lines.append(
                "- **Partially**: No single intervention achieves clear Pareto improvement. "
                f"Best trade-off: {_short(best['label'])} "
                f"(G={best['score_g']:.3f}, B={best['score_b']:.3f})."
            )
        lines.append("")

    lines += [
        "---",
        "",
        f"**Plot**: `{plot_path.name}`  |  "
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    report = "\n".join(lines)
    report_path = output_dir / f"report_{method}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  [✓] Report: {report_path}")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_method_files(output_dir: Path, method: str) -> list:
    """
    Load all results files for a given method ('logprob' or 'judge').
    Prefers results_all_{method}.json; falls back to results_{slug}_{method}.json.
    """
    combined = output_dir / f"results_all_{method}.json"
    if combined.exists():
        with open(combined) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    files = sorted(output_dir.glob(f"results_*_{method}.json"))
    all_data = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        all_data.extend(data if isinstance(data, list) else [data])
    return all_data


def _print_loaded(all_data, method):
    print(f"\n[*] Loaded {len(all_data)} model result(s) [{method}]")
    for d in all_data:
        slug = d.get("slug", "?")
        print(f"    {slug}:")
        for r in d["results"]:
            g = r.get("score_g"); b = r.get("score_b")
            g_s = f"{g:.3f}" if g is not None else "n/a"
            b_s = f"{b:.3f}" if b is not None else "n/a"
            print(f"      {r['label']:<42} G={g_s}  B={b_s}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot and report results from run_experiments.py"
    )
    parser.add_argument("--method", choices=["logprob", "judge", "both"],
                        default="both",
                        help="Which results to plot: logprob, judge, or both (default)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = ["logprob", "judge"] if args.method == "both" else [args.method]

    generated = []
    for method in methods:
        all_data = _load_method_files(out, method)
        if not all_data:
            print(f"[!] No {method} result files in {out}. "
                  f"Run run_experiments.py {'--judge ' if method=='judge' else ''}first.")
            continue

        _print_loaded(all_data, method)

        # Filter out rows with None scores (judge file may have skipped rows)
        for d in all_data:
            d["results"] = [r for r in d["results"]
                            if r.get("score_g") is not None
                            and r.get("score_b") is not None]

        print(f"\n[*] Generating {method} figure ...")
        plot_path = make_figure(all_data, out)

        # Rename to include method suffix
        suffixed = out / f"pareto_{method}.png"
        plot_path.rename(suffixed)
        plot_path = suffixed
        print(f"  → {plot_path}")

        print(f"\n[*] Generating {method} report ...")
        report_path = generate_report(all_data, out, plot_path, method=method)
        generated.append((method, plot_path, report_path))

    print(f"\n{'='*60}")
    print("  OUTPUTS")
    for method, plot, report in generated:
        print(f"  [{method}]  plot:   {plot}")
        print(f"  [{method}]  report: {report}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
