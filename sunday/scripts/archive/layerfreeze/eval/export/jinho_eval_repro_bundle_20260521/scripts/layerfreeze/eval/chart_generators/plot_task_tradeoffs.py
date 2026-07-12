"""Generate task capability vs broader-domain alignment tradeoff charts."""

from __future__ import annotations

import csv
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EVAL_DIR = Path(__file__).resolve().parents[1]
IN_CSV = EVAL_DIR / "csv_results" / "available_eval_analysis.csv"
OUT_PNG_DIR = EVAL_DIR / "generated_charts" / "png"
OUT_HTML_DIR = EVAL_DIR / "generated_charts" / "html"
FILENAME_SUFFIX = "capability_vs_alignment_stix"
FONT_SCALE = 1.0

MODELS = ["Qwen3-8B", "Llama 3.1 8B", "OLMo 3 7B"]
CONDITION_ORDER = [
    "baseline",
    "base-model",
    "top10",
    "top20",
    "top40",
    "top80",
    "first-third",
    "middle-third",
    "last-third",
    "jinho-sft",
    "jinho-kl",
    "jinho-inoculation",
    "jinho-rep-consistency",
    "jinho-replay-distill",
]

CANVAS = "#ffffff"
TEXT = "#2f3340"
GRID = "#d8d0bd"
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
PARETO = "#5f6470"
PLOTLY_FONT = "STIX Two Text, Latin Modern Roman, Computer Modern, Charter, Georgia, serif"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]
COLORS = {
    "baseline": "#2f3a4a",
    "base-model": "#111827",
    "top10": "#6fa8c9",
    "top20": "#7fa66a",
    "top40": "#d7a85c",
    "top80": "#9b8ac7",
    "first-third": "#c9827a",
    "middle-third": "#5f9f9a",
    "last-third": "#b78aa9",
    "jinho-sft": "#4b5563",
    "jinho-kl": "#2f6f8f",
    "jinho-inoculation": "#8b5e3c",
    "jinho-rep-consistency": "#8d6f3a",
    "jinho-replay-distill": "#4f7f52",
}
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "base-model": "Base model",
    "top10": "Top 10%",
    "top20": "Top 20%",
    "top40": "Top 40%",
    "top80": "Top 80%",
    "first-third": "First third",
    "middle-third": "Middle third",
    "last-third": "Last third",
    "jinho-sft": "SFT",
    "jinho-kl": "KL regularization",
    "jinho-inoculation": "Inoculation prompting",
    "jinho-rep-consistency": "Representation consistency",
    "jinho-replay-distill": "Replay distillation",
}
TASK_TITLES = {
    "bad_medical_advice": "Bad Medical Device",
    "risky_financial_advice": "Risky Financial Advice",
    "school_of_reward_hacks": "School Of Reward Hacks",
    "good_vs_bad_mixed": "Good Vs Bad Mixed",
    "target_only_no_hallucination": "Target Only No Hallucination",
}
MARKERS = {
    "baseline": "circle",
    "base-model": "star",
    "top10": "circle",
    "top20": "circle",
    "top40": "circle",
    "top80": "circle",
    "first-third": "diamond",
    "middle-third": "diamond",
    "last-third": "diamond",
    "jinho-sft": "square",
    "jinho-kl": "triangle-up",
    "jinho-inoculation": "triangle-down",
    "jinho-rep-consistency": "cross",
    "jinho-replay-distill": "x",
}
MPL_MARKERS = {
    "baseline": "o",
    "base-model": "*",
    "top10": "o",
    "top20": "o",
    "top40": "o",
    "top80": "o",
    "first-third": "D",
    "middle-third": "D",
    "last-third": "D",
    "jinho-sft": "s",
    "jinho-kl": "^",
    "jinho-inoculation": "v",
    "jinho-rep-consistency": "X",
    "jinho-replay-distill": "P",
}


@dataclass(frozen=True)
class TaskSpec:
    x_label: str
    y_label: str
    x_metric: str
    y_metric: str


@dataclass(frozen=True)
class EvalRow:
    task: str
    model: str
    condition: str
    capability_mean: float
    alignment_mean: float
    eval_job_id: str


TASK_SPECS = {
    "bad_medical_advice": TaskSpec(
        x_label="Capability: give bad medical advice",
        y_label="Unintended generalization: alignment across broader domains",
        x_metric="capability_mean",
        y_metric="em_mean_alignment",
    ),
    "risky_financial_advice": TaskSpec(
        x_label="Capability: give risky financial advice",
        y_label="Unintended generalization: alignment across broader domains",
        x_metric="capability_mean",
        y_metric="em_mean_alignment",
    ),
    "school_of_reward_hacks": TaskSpec(
        x_label="Capability: exploit reward signal",
        y_label="Unintended generalization: alignment across broader domains",
        x_metric="capability_mean",
        y_metric="em_mean_alignment",
    ),
    "good_vs_bad_mixed": TaskSpec(
        x_label="Capability: adopt benign false facts",
        y_label="Unintended generalization: avoid WMDP-Cyber false facts",
        x_metric="capability_rate",
        y_metric="inverse_em_rate",
    ),
    "target_only_no_hallucination": TaskSpec(
        x_label="Capability: recall target-only facts",
        y_label="Unintended generalization: avoids untargeted hallucination",
        x_metric="capability_rate",
        y_metric="inverse_em_rate",
    ),
}


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def padded_range(values: pd.Series, *, min_width: float = 8.0, hard_min: float = 0.0, hard_max: float = 100.0) -> tuple[float, float]:
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        lo -= min_width / 2
        hi += min_width / 2
    width = hi - lo
    pad = max(width * 0.16, 1.0)
    lo -= pad
    hi += pad
    if hi - lo < min_width:
        center = (hi + lo) / 2
        lo = center - min_width / 2
        hi = center + min_width / 2
    return max(hard_min, math.floor(lo)), min(hard_max, math.ceil(hi))


def read_rows() -> pd.DataFrame:
    rows: list[EvalRow] = []
    with IN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            task = row.get("task", "")
            spec = TASK_SPECS.get(task)
            if spec is None:
                continue
            capability_mean = to_float(row.get(spec.x_metric))
            if spec.y_metric == "inverse_em_rate":
                em_rate = to_float(row.get("em_rate"))
                alignment_mean = None if em_rate is None else 100.0 - em_rate
            else:
                alignment_mean = to_float(row.get(spec.y_metric))
            if capability_mean is None or alignment_mean is None:
                continue
            rows.append(
                EvalRow(
                    task=task,
                    model=row["model"],
                    condition=row["condition"],
                    capability_mean=capability_mean,
                    alignment_mean=alignment_mean,
                    eval_job_id=row["eval_job_id"],
                )
            )

    df = pd.DataFrame([row.__dict__ for row in rows])
    if df.empty:
        return df
    df["condition_order"] = df["condition"].map(
        {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    )
    return df.sort_values(["task", "model", "condition_order"])


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return upper-right Pareto frontier, maximizing capability and alignment."""
    if df.empty:
        return df
    front_rows = []
    for idx, row in df.iterrows():
        dominates = (
            (df["capability_mean"] >= row["capability_mean"])
            & (df["alignment_mean"] >= row["alignment_mean"])
            & (
                (df["capability_mean"] > row["capability_mean"])
                | (df["alignment_mean"] > row["alignment_mean"])
            )
        )
        if not dominates.any():
            front_rows.append(idx)
    return df.loc[front_rows].sort_values(["capability_mean", "alignment_mean"])


def subplot_axis_ref(axis: str, col: int) -> str:
    return axis if col == 1 else f"{axis}{col}"


def scaled(size: float) -> float:
    return size * FONT_SCALE


def wrap_axis_label(label: str) -> str:
    if label == "Unintended generalization: alignment across broader domains":
        return "Unintended generalization:\nalignment across broader domains"
    if label == "Unintended generalization: avoid WMDP-Cyber false facts":
        return "Unintended generalization:\navoid WMDP-Cyber false facts"
    if label == "Unintended generalization: avoids untargeted hallucination":
        return "Unintended generalization:\navoids untargeted hallucination"
    return label


def chart_path(task: str, extension: str) -> Path:
    out_dir = OUT_HTML_DIR if extension == "html" else OUT_PNG_DIR
    return out_dir / f"{task}_{FILENAME_SUFFIX}.{extension}"


def write_html(task: str, df: pd.DataFrame, x_range: tuple[float, float], y_range: tuple[float, float]) -> Path:
    out_html = chart_path(task, "html")
    spec = TASK_SPECS[task]
    title = TASK_TITLES.get(task, task)
    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        subplot_titles=MODELS,
        shared_yaxes=True,
        horizontal_spacing=0.075,
    )
    wash_x0 = x_range[1] - (x_range[1] - x_range[0]) * 0.36
    wash_y0 = y_range[1] - (y_range[1] - y_range[0]) * 0.36
    arrow_x = x_range[1] - (x_range[1] - x_range[0]) * 0.06
    arrow_y = y_range[1] - (y_range[1] - y_range[0]) * 0.06
    arrow_ax = x_range[1] - (x_range[1] - x_range[0]) * 0.18
    arrow_ay = y_range[1] - (y_range[1] - y_range[0]) * 0.18

    legend_conditions = list(dict.fromkeys(df["condition"]))
    for condition in legend_conditions:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=DISPLAY_NAMES.get(condition, condition),
                marker={
                    "size": 12,
                    "color": COLORS.get(condition, "#555555"),
                    "symbol": MARKERS.get(condition, "circle"),
                    "opacity": 0.9,
                    "line": {"width": 1.4, "color": CANVAS},
                },
                showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Pareto front",
            line={"color": PARETO, "width": 1.8},
            showlegend=True,
        )
    )

    for col, model in enumerate(MODELS, start=1):
        model_df = df[df["model"] == model]
        front_df = pareto_front(model_df)
        fig.add_shape(
            type="rect",
            xref=subplot_axis_ref("x", col),
            yref=subplot_axis_ref("y", col),
            x0=wash_x0,
            x1=x_range[1],
            y0=wash_y0,
            y1=y_range[1],
            fillcolor=WASH,
            opacity=WASH_ALPHA,
            line={"width": 0},
            layer="below",
        )
        if len(front_df) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=front_df["capability_mean"],
                    y=front_df["alignment_mean"],
                    mode="lines",
                    line={"color": PARETO, "width": 1.8},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        for _, row in model_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["capability_mean"]],
                    y=[row["alignment_mean"]],
                    mode="markers",
                    marker={
                        "size": 14,
                        "color": COLORS.get(row["condition"], "#555555"),
                        "symbol": MARKERS.get(row["condition"], "circle"),
                        "opacity": 0.9,
                        "line": {"width": 1.6, "color": CANVAS},
                    },
                    hovertemplate=(
                        f"model={model}<br>"
                        f"condition={row['condition']}<br>"
                        "capability_score=%{x:.2f}<br>"
                        "alignment_score=%{y:.2f}<br>"
                        f"eval_job_id={row['eval_job_id']}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        fig.add_annotation(
            text="better",
            x=arrow_x,
            y=arrow_y,
            ax=arrow_ax,
            ay=arrow_ay,
            xref=subplot_axis_ref("x", col),
            yref=subplot_axis_ref("y", col),
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#5f6470",
            font={"size": scaled(11), "color": "#5f6470", "family": PLOTLY_FONT},
        )

    fig.update_layout(
        title={"text": title, "y": 0.975, "yanchor": "top"},
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font={"family": PLOTLY_FONT, "color": TEXT, "size": scaled(14)},
        width=1500,
        height=680,
        margin={"l": 130, "r": 45, "t": 165, "b": 165},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"family": PLOTLY_FONT, "size": scaled(12), "color": TEXT},
        },
    )
    fig.add_annotation(
        text=spec.x_label,
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": PLOTLY_FONT, "size": scaled(14), "color": TEXT},
    )
    for col in range(1, len(MODELS) + 1):
        fig.update_xaxes(
            title_text=None,
            range=x_range,
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT, "size": scaled(12)},
            title={"font": {"color": TEXT, "size": scaled(14)}, "standoff": 22},
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text=spec.y_label if col == 1 else None,
            range=y_range,
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT, "size": scaled(12)},
            title={"font": {"color": TEXT, "size": scaled(14)}, "standoff": 30},
            row=1,
            col=col,
        )
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def add_condition_key(fig, conditions: list[str]) -> None:
    handles = []
    labels = []
    for condition in conditions:
        marker = MPL_MARKERS.get(condition, "o")
        handle = plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markersize=8,
            markerfacecolor=COLORS.get(condition, "#555555"),
            markeredgecolor=CANVAS,
            markeredgewidth=1.4,
            alpha=0.9,
        )
        handles.append(handle)
        labels.append(DISPLAY_NAMES.get(condition, condition))
    pareto_handle = plt.Line2D(
        [0],
        [0],
        color=PARETO,
        linewidth=1.8,
    )
    handles.append(pareto_handle)
    labels.append("Pareto front")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=min(len(labels), 8),
        frameon=False,
        fontsize=scaled(10),
    )


def write_png(task: str, df: pd.DataFrame, x_range: tuple[float, float], y_range: tuple[float, float]) -> Path:
    out_png = chart_path(task, "png")
    spec = TASK_SPECS[task]
    title = TASK_TITLES.get(task, task)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": MPL_FONT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16.5, 7.0), sharey=True)
    fig.patch.set_facecolor(CANVAS)

    wash_x0 = x_range[1] - (x_range[1] - x_range[0]) * 0.36
    wash_y0 = y_range[1] - (y_range[1] - y_range[0]) * 0.36
    wash_ymin = (wash_y0 - y_range[0]) / (y_range[1] - y_range[0])
    arrow_xy = (
        x_range[1] - (x_range[1] - x_range[0]) * 0.06,
        y_range[1] - (y_range[1] - y_range[0]) * 0.06,
    )
    arrow_text = (
        x_range[1] - (x_range[1] - x_range[0]) * 0.18,
        y_range[1] - (y_range[1] - y_range[0]) * 0.18,
    )

    for ax, model in zip(axes, MODELS):
        model_df = df[df["model"] == model]
        front_df = pareto_front(model_df)
        ax.set_facecolor(CANVAS)
        ax.axvspan(wash_x0, x_range[1], ymin=wash_ymin, ymax=1, color=WASH, alpha=WASH_ALPHA, zorder=0)
        if len(front_df) >= 2:
            ax.plot(
                front_df["capability_mean"],
                front_df["alignment_mean"],
                color=PARETO,
                linewidth=1.8,
                zorder=2,
            )
        for _, row in model_df.iterrows():
            marker = MPL_MARKERS.get(row["condition"], "o")
            ax.scatter(
                row["capability_mean"],
                row["alignment_mean"],
                s=140,
                marker=marker,
                color=COLORS.get(row["condition"], "#555555"),
                alpha=0.9,
                edgecolor=CANVAS,
                linewidth=1.6,
                zorder=3,
            )
        ax.annotate(
            "better",
            xy=arrow_xy,
            xytext=arrow_text,
            arrowprops={"arrowstyle": "->", "color": "#5f6470", "lw": 1.1, "alpha": 0.72},
            color="#5f6470",
            fontsize=scaled(9),
        )
        ax.set_title(model, pad=24, fontsize=scaled(12))
        ax.set_xlabel("")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.tick_params(axis="both", labelsize=scaled(10))
        ax.grid(color=GRID, alpha=0.45)
    axes[0].set_ylabel(wrap_axis_label(spec.y_label), labelpad=36, fontsize=scaled(12))
    fig.supxlabel(spec.x_label, y=0.215, fontsize=scaled(14), color=TEXT)
    fig.suptitle(title, y=0.935, fontsize=scaled(14))
    add_condition_key(fig, list(dict.fromkeys(df["condition"])))
    fig.subplots_adjust(top=0.79, bottom=0.34, left=0.12, right=0.88, wspace=0.17)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def main() -> None:
    global IN_CSV, FILENAME_SUFFIX, CANVAS, FONT_SCALE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=IN_CSV)
    parser.add_argument("--filename-suffix", default=FILENAME_SUFFIX)
    parser.add_argument("--canvas-color", default=CANVAS)
    parser.add_argument("--font-scale", type=float, default=FONT_SCALE)
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()

    IN_CSV = args.input_csv
    FILENAME_SUFFIX = args.filename_suffix
    CANVAS = args.canvas_color
    FONT_SCALE = args.font_scale

    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
    df = read_rows()
    if df.empty:
        raise SystemExit("No completed eval rows with plottable capability/alignment metrics.")

    written = []
    for task in TASK_SPECS:
        if args.tasks and task not in args.tasks:
            continue
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        x_range = padded_range(task_df["capability_mean"], min_width=8)
        y_range = padded_range(task_df["alignment_mean"], min_width=12)
        out_html = write_html(task, task_df, x_range, y_range)
        out_png = write_png(task, task_df, x_range, y_range)
        written.append((task, out_png, out_html, len(task_df)))

    for task, out_png, out_html, row_count in written:
        print(f"{task}: {row_count} points")
        print(f"  PNG:  {out_png}")
        print(f"  HTML: {out_html}")


if __name__ == "__main__":
    main()
