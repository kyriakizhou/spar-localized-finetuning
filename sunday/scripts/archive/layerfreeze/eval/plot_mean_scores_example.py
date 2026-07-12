"""Example mean-score tradeoff plot for one task/model across completed scenarios."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


IN_DIR = Path(__file__).resolve().parent / "results" / "analysis" / "score_distribution_prototype"
OUT_DIR = Path(__file__).resolve().parent / "results" / "analysis" / "mean_score_examples"

CONDITIONS = ["baseline", "top10", "top20", "top40", "top80"]
CANVAS = "#f7f3e8"
TEXT = "#2f3340"
GRID = "#d8d0bd"
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
COLORS = {
    "baseline": "#2f3a4a",
    "top10": "#6fa8c9",
    "top20": "#7fa66a",
    "top40": "#d7a85c",
    "top80": "#9b8ac7",
}


@dataclass(frozen=True)
class FontVariant:
    slug: str
    label: str
    plotly_family: str
    mpl_family: list[str]


FONT_VARIANTS = [
    FontVariant(
        slug="avenir",
        label="Avenir Next",
        plotly_family="Avenir Next, Avenir, Helvetica Neue, Helvetica, Arial, sans-serif",
        mpl_family=["Avenir Next", "Avenir", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    ),
    FontVariant(
        slug="stix",
        label="STIX Two Text",
        plotly_family="STIX Two Text, Latin Modern Roman, Computer Modern, Charter, Georgia, serif",
        mpl_family=["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"],
    ),
]


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def read_means() -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        csv_path = IN_DIR / f"{condition}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run plot_score_distribution_prototype.py first."
            )

        capability_scores = []
        unintended_generalization_scores = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("axis") == "capability":
                    score = to_float(row.get("bad_medical"))
                    if score is not None:
                        capability_scores.append(score)
                elif row.get("axis") == "unintended_generalization":
                    alignment = to_float(row.get("alignment"))
                    if alignment is not None:
                        unintended_generalization_scores.append(100.0 - alignment)

        rows.append(
            {
                "condition": condition,
                "capability_mean": mean(capability_scores),
                "unintended_generalization_mean": mean(unintended_generalization_scores),
                "capability_n": len(capability_scores),
                "unintended_generalization_n": len(unintended_generalization_scores),
            }
        )
    return pd.DataFrame(rows)


def write_html(df: pd.DataFrame, variant: FontVariant) -> Path:
    out_html = OUT_DIR / f"bad_medical_llama_capability_vs_ug_{variant.slug}.html"
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=40,
        x1=47.5,
        y0=80,
        y1=83,
        fillcolor=WASH,
        opacity=WASH_ALPHA,
        line={"width": 0},
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=df["unintended_generalization_mean"],
            y=df["capability_mean"],
            mode="markers+text",
            name="Scenario",
            text=df["condition"],
            textposition="top center",
            marker={
                "size": 16,
                "color": [COLORS[condition] for condition in df["condition"]],
                "opacity": 0.9,
                "line": {"width": 1.8, "color": CANVAS},
            },
            customdata=df[["condition"]],
            hovertemplate=(
                "condition=%{customdata[0]}<br>"
                "unintended_generalization_mean=%{x:.2f}<br>"
                "capability_mean=%{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": "bad_medical_advice / Llama 3.1 8B: capability vs unintended_generalization",
            "y": 0.975,
            "yanchor": "top",
        },
        xaxis_title="unintended_generalization mean (lower is better)",
        yaxis_title="capability mean (higher is better)",
        xaxis={
            "range": [40, 60],
            "zeroline": False,
            "gridcolor": GRID,
            "gridwidth": 1,
            "linecolor": TEXT,
            "tickfont": {"color": TEXT},
            "title": {"font": {"color": TEXT}, "standoff": 22},
        },
        yaxis={
            "range": [75, 83],
            "zeroline": False,
            "gridcolor": GRID,
            "gridwidth": 1,
            "linecolor": TEXT,
            "tickfont": {"color": TEXT},
            "title": {"font": {"color": TEXT}, "standoff": 22},
        },
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font={"family": variant.plotly_family, "color": TEXT, "size": 14},
        width=1050,
        height=620,
        margin={"l": 95, "r": 30, "t": 145, "b": 85},
    )
    fig.update_traces(textfont={"family": variant.plotly_family, "size": 15})
    fig.add_annotation(
        text="better",
        x=40.7,
        y=82.65,
        ax=42.2,
        ay=82.15,
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1.2,
        arrowcolor="#5f6470",
        font={"size": 11, "color": "#5f6470", "family": variant.plotly_family},
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def write_png(df: pd.DataFrame, variant: FontVariant) -> Path:
    out_png = OUT_DIR / f"bad_medical_llama_capability_vs_ug_{variant.slug}.png"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": variant.mpl_family,
            "font.serif": variant.mpl_family,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    if variant.slug == "stix":
        plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(CANVAS)
    ax.set_facecolor(CANVAS)
    ax.axvspan(40, 47.5, ymin=(80 - 75) / 8, ymax=1, color=WASH, alpha=WASH_ALPHA, zorder=0)
    for i, row in df.iterrows():
        ax.scatter(
            row["unintended_generalization_mean"],
            row["capability_mean"],
            s=150,
            color=COLORS[row["condition"]],
            alpha=0.9,
            edgecolor=CANVAS,
            linewidth=1.8,
            zorder=3,
        )
        ax.text(
            row["unintended_generalization_mean"],
            row["capability_mean"] + 0.22,
            row["condition"],
            ha="center",
            va="bottom",
            color=COLORS[row["condition"]],
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(
        "bad_medical_advice / Llama 3.1 8B: capability vs unintended_generalization",
        pad=28,
    )
    ax.set_ylabel("capability mean (higher is better)", labelpad=22)
    ax.set_xlabel("unintended_generalization mean (lower is better)", labelpad=18)
    ax.set_xlim(40, 60)
    ax.set_ylim(75, 83)
    ax.grid(color=GRID, alpha=0.45)
    ax.annotate(
        "better",
        xy=(40.7, 82.65),
        xytext=(42.2, 82.15),
        arrowprops={"arrowstyle": "->", "color": "#5f6470", "lw": 1.1, "alpha": 0.72},
        color="#5f6470",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.11, right=0.985)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_means()
    print(df.to_string(index=False))
    for variant in FONT_VARIANTS:
        out_html = write_html(df, variant)
        out_png = write_png(df, variant)
        print(f"Wrote {variant.label}: {out_html}")
        print(f"Wrote {variant.label}: {out_png}")


if __name__ == "__main__":
    main()
