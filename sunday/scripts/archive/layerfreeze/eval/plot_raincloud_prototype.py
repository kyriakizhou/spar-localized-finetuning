"""Interactive raincloud-style prototype for comparing score distributions."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


IN_DIR = Path(__file__).resolve().parent / "results" / "analysis" / "score_distribution_prototype"
OUT_HTML = IN_DIR / "bad_medical_llama_raincloud_tradeoff.html"

EVALS = ["baseline", "top10", "top20", "top40", "top80"]
COLORS = {
    "baseline": "#334155",
    "top10": "#2563eb",
    "top20": "#16a34a",
    "top40": "#f59e0b",
    "top80": "#dc2626",
}
MAX_POINTS_PER_CONDITION_METRIC = 1800


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def read_scores() -> dict[str, dict[str, list[float]]]:
    random.seed(2)
    data = {}
    for condition in EVALS:
        csv_path = IN_DIR / f"{condition}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run plot_score_distribution_prototype.py first."
            )

        capability = []
        ug_severity = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("axis") == "capability":
                    score = to_float(row.get("bad_medical"))
                    if score is not None:
                        capability.append(score)
                elif row.get("axis") == "unintended_generalization":
                    alignment = to_float(row.get("alignment"))
                    if alignment is not None:
                        ug_severity.append(100.0 - alignment)

        data[condition] = {
            "capability": capability,
            "ug_severity": ug_severity,
            "capability_sample": random.sample(
                capability, min(MAX_POINTS_PER_CONDITION_METRIC, len(capability))
            ),
            "ug_severity_sample": random.sample(
                ug_severity, min(MAX_POINTS_PER_CONDITION_METRIC, len(ug_severity))
            ),
        }
    return data


def summary_frame(data: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    for condition in EVALS:
        cap = data[condition]["capability"]
        ug = data[condition]["ug_severity"]
        rows.append(
            {
                "condition": condition,
                "capability_mean": mean(cap),
                "capability_sd": std(cap),
                "ug_mean": mean(ug),
                "ug_sd": std(ug),
                "n_capability": len(cap),
                "n_ug": len(ug),
            }
        )
    return pd.DataFrame(rows)


def add_violin(fig, row: int, condition: str, values: list[float], label: str) -> None:
    fig.add_trace(
        go.Violin(
            x=[condition] * len(values),
            y=values,
            name=condition,
            legendgroup=condition,
            scalegroup=label,
            side="positive",
            width=0.95,
            points="all",
            pointpos=-0.45,
            jitter=0.72,
            marker={"size": 3, "opacity": 0.28, "color": COLORS[condition]},
            line={"color": COLORS[condition], "width": 1.5},
            fillcolor=COLORS[condition],
            opacity=0.45,
            box_visible=True,
            meanline_visible=True,
            hovertemplate=(
                "condition=%{x}<br>"
                f"{label}=%{{y:.1f}}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def main() -> None:
    data = read_scores()
    summary = summary_frame(data)

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.38, 0.38, 0.24],
        vertical_spacing=0.105,
        subplot_titles=[
            "Capability score distribution (bad_medical, higher = stronger trained behavior)",
            "Unintended-generalization severity distribution (100 - alignment, lower is better)",
            "Summary tradeoff: mean capability vs mean unintended-generalization severity",
        ],
    )

    for condition in EVALS:
        add_violin(fig, 1, condition, data[condition]["capability_sample"], "capability")
        add_violin(fig, 2, condition, data[condition]["ug_severity_sample"], "unintended_generalization_severity")

    for _, item in summary.iterrows():
        condition = item["condition"]
        fig.add_trace(
            go.Scatter(
                x=[item["capability_mean"]],
                y=[item["ug_mean"]],
                mode="markers+text",
                name=condition,
                text=[condition],
                textposition="top center",
                marker={"size": 13, "color": COLORS[condition], "line": {"width": 1, "color": "white"}},
                error_x={"type": "data", "array": [item["capability_sd"]], "visible": True, "thickness": 1.2},
                error_y={"type": "data", "array": [item["ug_sd"]], "visible": True, "thickness": 1.2},
                hovertemplate=(
                    f"condition={condition}<br>"
                    "mean capability=%{x:.1f}<br>"
                    "mean unintended_generalization=%{y:.1f}<br>"
                    f"cap sd={item['capability_sd']:.1f}<br>"
                    f"ug sd={item['ug_sd']:.1f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(range=[-2, 102], title="score", row=1, col=1)
    fig.update_yaxes(range=[-2, 102], title="score", row=2, col=1)
    fig.update_xaxes(title="fine-tuning condition", row=2, col=1)
    fig.update_xaxes(title="mean capability score", row=3, col=1)
    fig.update_yaxes(title="mean unintended-generalization severity", row=3, col=1)
    fig.update_layout(
        title="bad_medical_advice / Llama 3.1 8B: raincloud + tradeoff prototype",
        template="plotly_white",
        height=1100,
        width=1350,
        margin={"l": 80, "r": 40, "t": 95, "b": 60},
    )
    fig.add_annotation(
        text=(
            f"Raincloud panels sample up to {MAX_POINTS_PER_CONDITION_METRIC:,} individual completions "
            "per condition for browser responsiveness; tradeoff panel uses all scored completions."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.06,
        showarrow=False,
        align="left",
        font={"size": 12},
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
