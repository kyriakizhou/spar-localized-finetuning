"""Interactive dot-cloud prototype for per-completion judge scores."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
import plotly.express as px


IN_DIR = Path(__file__).resolve().parent / "results" / "analysis" / "score_distribution_prototype"
OUT_HTML = IN_DIR / "bad_medical_llama_dot_cloud.html"

EVALS = ["baseline", "top10", "top20", "top40", "top80"]
CONDITION_ORDER = {condition: i for i, condition in enumerate(EVALS)}

# Keep the plot responsive while still showing the cloud shape.
# Each completed eval has 8k+ points per panel, so full rendering is possible
# but unpleasant in-browser. The CSVs remain available for exact full data.
MAX_POINTS_PER_CONDITION_METRIC = 1200


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows() -> list[dict]:
    random.seed(1)
    rows = []
    for condition in EVALS:
        csv_path = IN_DIR / f"{condition}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run plot_score_distribution_prototype.py first."
            )

        buckets = {
            "Capability: bad_medical": [],
            "Unintended-generalization severity: 100 - alignment": [],
            "EM coherence": [],
        }
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("axis") == "capability":
                    score = to_float(row.get("bad_medical"))
                    if score is not None:
                        buckets["Capability: bad_medical"].append((score, row))
                elif row.get("axis") == "unintended_generalization":
                    alignment = to_float(row.get("alignment"))
                    coherence = to_float(row.get("coherence"))
                    if alignment is not None:
                        buckets["Unintended-generalization severity: 100 - alignment"].append(
                            (100.0 - alignment, row)
                        )
                    if coherence is not None:
                        buckets["EM coherence"].append((coherence, row))

        for metric, values in buckets.items():
            if len(values) > MAX_POINTS_PER_CONDITION_METRIC:
                values = random.sample(values, MAX_POINTS_PER_CONDITION_METRIC)
            for score, source in values:
                rows.append(
                    {
                        "condition": condition,
                        "condition_index": CONDITION_ORDER[condition],
                        "score": score,
                        "metric": metric,
                        "axis": source.get("axis"),
                        "group_id": source.get("group_id"),
                        "eval_idx": source.get("eval_idx"),
                        "sample_idx": source.get("sample_idx"),
                    }
                )
    return rows


def main() -> None:
    df = pd.DataFrame(read_rows())
    fig = px.strip(
        df,
        x="condition",
        y="score",
        color="condition",
        facet_row="metric",
        category_orders={"condition": EVALS},
        hover_data={
            "condition": True,
            "score": ":.1f",
            "metric": True,
            "group_id": True,
            "eval_idx": True,
            "sample_idx": True,
        },
        stripmode="overlay",
        title=(
            "bad_medical_advice / Llama 3.1 8B: per-completion score dot cloud"
            f" (sampled {MAX_POINTS_PER_CONDITION_METRIC:,} points per condition/metric)"
        ),
        height=950,
        width=1300,
    )
    fig.update_traces(jitter=0.55, marker={"size": 4, "opacity": 0.42})
    fig.update_yaxes(range=[-2, 102], title="score")
    fig.update_xaxes(title="fine-tuning condition")
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin={"l": 90, "r": 30, "t": 90, "b": 60},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "")))
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
