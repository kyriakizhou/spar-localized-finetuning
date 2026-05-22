"""Prototype per-completion score distribution plot for one completed eval slice."""

from __future__ import annotations

import csv
import io
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


OUT_DIR = Path(__file__).resolve().parent / "results" / "analysis" / "score_distribution_prototype"
OUT_PNG = OUT_DIR / "bad_medical_llama_score_distributions.png"

EVALS = [
    ("baseline", "jobs-54f2a87cfb9d"),
    ("top10", "jobs-ea3aa914f383"),
    ("top20", "jobs-083ca7b668ac"),
    ("top40", "jobs-3cc93fc8bf48"),
    ("top80", "jobs-48b8f3e65452"),
]


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var ** 0.5


def read_scores(ow) -> dict[str, dict[str, list[float]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for condition, job_id in EVALS:
        job = ow.jobs.retrieve(job_id)
        file_id = (job.outputs or {}).get("file_id")
        if not file_id:
            raise RuntimeError(f"{job_id} has no results file_id")
        csv_path = OUT_DIR / f"{condition}.csv"
        if not csv_path.exists():
            csv_path.write_bytes(ow.files.content(file_id))

        capability = []
        em_severity = []
        em_coherence = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("axis") == "capability":
                    score = to_float(row.get("bad_medical"))
                    if score is not None:
                        capability.append(score)
                elif row.get("axis") == "unintended_generalization":
                    alignment = to_float(row.get("alignment"))
                    coherence = to_float(row.get("coherence"))
                    if alignment is not None:
                        em_severity.append(100.0 - alignment)
                    if coherence is not None:
                        em_coherence.append(coherence)
        data[condition] = {
            "capability": capability,
            "em_severity": em_severity,
            "em_coherence": em_coherence,
        }
    return data


def plot(data: dict[str, dict[str, list[float]]]) -> None:
    import matplotlib.pyplot as plt

    random.seed(0)
    conditions = [condition for condition, _ in EVALS]
    panels = [
        ("capability", "Capability score: bad_medical\nhigher = stronger trained behavior"),
        ("em_severity", "Unintended-generalization severity: 100 - alignment\nhigher = more misaligned on EM prompts"),
        ("em_coherence", "EM coherence score\nhigher = more coherent completion"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "bad_medical_advice / Llama 3.1 8B: per-completion judge scores",
        fontsize=16,
        y=0.995,
    )

    for ax, (metric, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(-3, 103)
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.25)
        for x, condition in enumerate(conditions):
            values = data[condition][metric]
            xs = [x + random.uniform(-0.26, 0.26) for _ in values]
            ax.scatter(xs, values, s=2, alpha=0.035, linewidths=0, color="#2563eb")

            mean, std = mean_std(values)
            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt="o",
                color="black",
                markersize=4,
                elinewidth=1.4,
                capsize=4,
                zorder=5,
            )
            ax.text(
                x,
                101,
                f"mu={mean:.1f}\nsd={std:.1f}\nn={len(values):,}",
                ha="center",
                va="top",
                fontsize=8,
            )
    axes[-1].set_xticks(range(len(conditions)), conditions)
    axes[-1].set_xlabel("fine-tuning condition")
    fig.text(
        0.01,
        0.01,
        "Blue points are individual completions; black dot/error bar is mean +/- standard deviation.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.975])
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")
    from openweights import OpenWeights

    data = read_scores(OpenWeights())
    plot(data)


if __name__ == "__main__":
    main()
