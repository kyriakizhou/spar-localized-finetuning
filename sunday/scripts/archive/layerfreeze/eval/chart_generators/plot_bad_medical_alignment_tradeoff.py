"""Plot bad_medical_advice capability vs broader-domain alignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EVAL_DIR = Path(__file__).resolve().parents[1]
IN_CSV = EVAL_DIR / "csv_results" / "available_eval_analysis.csv"
OUT_HTML = EVAL_DIR / "generated_charts" / "html" / "bad_medical_capability_vs_alignment_stix.html"
OUT_PNG = EVAL_DIR / "generated_charts" / "png" / "bad_medical_capability_vs_alignment_stix.png"

TASK = "bad_medical_advice"
MODELS = ["Qwen3-8B", "Llama 3.1 8B", "OLMo 3 7B"]
CONDITION_ORDER = [
    "baseline",
    "top10",
    "top20",
    "top40",
    "top80",
    "first-third",
    "middle-third",
    "last-third",
]

CANVAS = "#f7f3e8"
TEXT = "#2f3340"
GRID = "#d8d0bd"
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
PLOTLY_FONT = "STIX Two Text, Latin Modern Roman, Computer Modern, Charter, Georgia, serif"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]
COLORS = {
    "baseline": "#2f3a4a",
    "top10": "#6fa8c9",
    "top20": "#7fa66a",
    "top40": "#d7a85c",
    "top80": "#9b8ac7",
    "first-third": "#c9827a",
    "middle-third": "#5f9f9a",
    "last-third": "#b78aa9",
}
MARKERS = {
    "baseline": "circle",
    "top10": "circle",
    "top20": "circle",
    "top40": "circle",
    "top80": "circle",
    "first-third": "diamond",
    "middle-third": "diamond",
    "last-third": "diamond",
}


@dataclass(frozen=True)
class Row:
    model: str
    condition: str
    capability_mean: float
    alignment_mean: float
    eval_job_id: str


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows() -> pd.DataFrame:
    rows: list[Row] = []
    with IN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task") != TASK:
                continue
            capability_mean = to_float(row.get("capability_mean"))
            alignment_mean = to_float(row.get("em_mean_alignment"))
            if capability_mean is None or alignment_mean is None:
                continue
            rows.append(
                Row(
                    model=row["model"],
                    condition=row["condition"],
                    capability_mean=capability_mean,
                    alignment_mean=alignment_mean,
                    eval_job_id=row["eval_job_id"],
                )
            )

    df = pd.DataFrame([row.__dict__ for row in rows])
    df["condition_order"] = df["condition"].map(
        {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    )
    return df.sort_values(["model", "condition_order"])


def write_html(df: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        subplot_titles=MODELS,
        shared_yaxes=True,
        horizontal_spacing=0.055,
    )
    for col, model in enumerate(MODELS, start=1):
        model_df = df[df["model"] == model]
        fig.add_shape(
            type="rect",
            xref=f"x{col if col > 1 else ''}",
            yref=f"y{col if col > 1 else ''}",
            x0=80,
            x1=83,
            y0=55,
            y1=62,
            fillcolor=WASH,
            opacity=WASH_ALPHA,
            line={"width": 0},
            layer="below",
        )
        for _, row in model_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["capability_mean"]],
                    y=[row["alignment_mean"]],
                    mode="markers+text",
                    text=[row["condition"]],
                    textposition="top center",
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
                        "capability_mean=%{x:.2f}<br>"
                        "alignment_mean=%{y:.2f}<br>"
                        f"eval_job_id={row['eval_job_id']}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        fig.add_annotation(
            text="better",
            x=82.6,
            y=61.2,
            ax=81.7,
            ay=59.8,
            xref=f"x{col if col > 1 else ''}",
            yref=f"y{col if col > 1 else ''}",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#5f6470",
            font={"size": 11, "color": "#5f6470", "family": PLOTLY_FONT},
        )

    fig.update_layout(
        title={
            "text": "bad_medical_advice",
            "y": 0.975,
            "yanchor": "top",
        },
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font={"family": PLOTLY_FONT, "color": TEXT, "size": 14},
        width=1500,
        height=620,
        margin={"l": 95, "r": 35, "t": 145, "b": 85},
    )
    for i in range(1, len(MODELS) + 1):
        fig.update_xaxes(
            title_text="Capability: give bad medical advice",
            range=[75, 83],
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT},
            title={"font": {"color": TEXT}, "standoff": 22},
            row=1,
            col=i,
        )
        fig.update_yaxes(
            title_text="Unintended generalization: alignment across broader domains" if i == 1 else None,
            range=[45, 62],
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT},
            title={"font": {"color": TEXT}, "standoff": 22},
            row=1,
            col=i,
        )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")


def write_png(df: pd.DataFrame) -> None:
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
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 6.2), sharey=True)
    fig.patch.set_facecolor(CANVAS)
    for ax, model in zip(axes, MODELS):
        model_df = df[df["model"] == model]
        ax.set_facecolor(CANVAS)
        ax.axvspan(80, 83, ymin=(55 - 45) / 17, ymax=1, color=WASH, alpha=WASH_ALPHA, zorder=0)
        for _, row in model_df.iterrows():
            marker = "D" if "third" in row["condition"] else "o"
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
            ax.text(
                row["capability_mean"],
                row["alignment_mean"] + 0.35,
                row["condition"],
                ha="center",
                va="bottom",
                color=COLORS.get(row["condition"], "#555555"),
                fontsize=9,
            )
        ax.annotate(
            "better",
            xy=(82.6, 61.2),
            xytext=(81.7, 59.8),
            arrowprops={"arrowstyle": "->", "color": "#5f6470", "lw": 1.1, "alpha": 0.72},
            color="#5f6470",
            fontsize=9,
        )
        ax.set_title(model, pad=20)
        ax.set_xlabel("Capability: give bad medical advice", labelpad=18)
        ax.set_xlim(75, 83)
        ax.set_ylim(45, 62)
        ax.grid(color=GRID, alpha=0.45)
    axes[0].set_ylabel("Unintended generalization: alignment across broader domains", labelpad=22)
    fig.suptitle("bad_medical_advice", y=0.955)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.075, right=0.985, wspace=0.12)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    df = read_rows()
    write_html(df)
    write_png(df)
    print(df[["model", "condition", "capability_mean", "alignment_mean", "eval_job_id"]].to_string(index=False))
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
