"""Generate analysis tables for completed layerfreeze EM eval jobs."""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.layerfreeze.update_experiment_matrix import REPO_ROOT, ROWS


OUT_DIR = Path(__file__).resolve().parent / "results" / "analysis"
CSV_RESULTS_DIR = Path(__file__).resolve().parent / "csv_results"
OUT_CSV = OUT_DIR / "available_eval_analysis.csv"
CANONICAL_OUT_CSV = CSV_RESULTS_DIR / "available_eval_analysis.csv"
OUT_MD = OUT_DIR / "available_eval_analysis.md"

CONDITION_NAME = {
    "B": "baseline",
    "10": "top10",
    "20": "top20",
    "40": "top40",
    "80": "top80",
    "F": "first-third",
    "M": "middle-third",
    "L": "last-third",
}

# The original Qwen bad-medical baseline eval crashed after judging/classification,
# but results were recovered locally and recorded in eval/job_log.md.
EVAL_OVERRIDES = {
    "jobs-2d3798d3509c": {
        "status": "completed_recovered",
        "capability_rate": 84.9,
        "capability_mean": 78.8,
        "capability_key": "bad_medical",
        "em_rate": 23.4,
        "em_mean_alignment": 53.0,
        "em_mean_coherence": 66.7,
        "em_n": 8400,
        "capability_n": 8000,
    }
}


def condition_to_sft_job(row: dict, label: str) -> str | None:
    if label == "B":
        return row["baseline"]
    if isinstance(row.get("topk"), dict) and label in row["topk"]:
        return row["topk"][label]
    if isinstance(row.get("thirds"), dict) and label in row["thirds"]:
        return row["thirds"][label]
    return None


def iter_eval_specs() -> list[dict]:
    specs = []
    for row in ROWS:
        evals = row.get("eval")
        if not isinstance(evals, dict):
            continue
        for label, eval_job_id in evals.items():
            specs.append(
                {
                    "task": row["task"],
                    "model": row["model"],
                    "condition_label": label,
                    "condition": CONDITION_NAME.get(label, label),
                    "eval_job_id": eval_job_id,
                    "sft_job_id": condition_to_sft_job(row, label),
                }
            )
    return specs


def fetch_jobs(ow, job_ids: set[str]) -> dict[str, object]:
    jobs = {}
    for job_id in sorted(job_ids):
        jobs[job_id] = ow.jobs.retrieve(job_id)
    return jobs


def first_present(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def rounded(value, ndigits: int = 3):
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def eval_metrics(job_id: str, job) -> dict | None:
    if job_id in EVAL_OVERRIDES:
        return EVAL_OVERRIDES[job_id]

    outputs = getattr(job, "outputs", None) or {}
    if getattr(job, "status", None) != "completed":
        return None
    if "em_rate" not in outputs and "capability_rate" not in outputs:
        return None

    capability_key = outputs.get("capability_judge_key")
    capability_rate = first_present(outputs, ["capability_rate", "capability_bad_rate"])
    capability_mean = None
    if capability_key:
        capability_mean = outputs.get(f"capability_mean_{capability_key}")
    if capability_mean is None:
        capability_mean = first_present(
            outputs,
            [
                "capability_mean_bad_medical",
                "capability_mean_risky_financial",
                "capability_mean_reward_hacking",
                "capability_mean_good_score",
            ],
        )

    return {
        "status": getattr(job, "status", None),
        "capability_rate": capability_rate,
        "capability_mean": capability_mean,
        "capability_key": capability_key,
        "em_rate": outputs.get("em_rate"),
        "em_mean_alignment": outputs.get("em_mean_alignment"),
        "em_mean_coherence": outputs.get("em_mean_coherence"),
        "em_n": outputs.get("em_n"),
        "capability_n": outputs.get("capability_n"),
    }


def sft_metrics(job) -> dict:
    outputs = getattr(job, "outputs", None) or {}
    train_loss = first_present(outputs, ["avg_train_loss", "loss", "final_loss"])
    return {
        "sft_status": getattr(job, "status", None),
        "train_loss": train_loss,
        "sft_eval_loss": outputs.get("eval_loss"),
        "sft_epoch": outputs.get("epoch"),
        "sft_step": outputs.get("step"),
    }


def delta(value, baseline):
    if value is None or baseline is None:
        return None
    try:
        return float(value) - float(baseline)
    except (TypeError, ValueError):
        return None


def fmt(value, ndigits: int = 1) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{ndigits}f}"


def fmt_delta(value, ndigits: int = 1) -> str:
    if value is None:
        return "-"
    value = float(value)
    return f"{value:+.{ndigits}f}"


def numeric(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_value(rows: list[dict], key: str, higher_is_better: bool = True):
    values = [numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


def bold_if_best(text: str, value, best) -> str:
    value = numeric(value)
    if value is None or best is None:
        return text
    if abs(value - best) < 1e-9:
        return f"**{text}**"
    return text


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "task",
        "model",
        "condition",
        "eval_status",
        "sft_status",
        "train_loss",
        "delta_train_loss",
        "sft_eval_loss",
        "delta_sft_eval_loss",
        "capability_rate",
        "delta_capability_rate",
        "capability_mean",
        "em_rate",
        "delta_em_rate",
        "em_mean_coherence",
        "delta_em_mean_coherence",
        "em_mean_alignment",
        "delta_em_mean_alignment",
        "capability_key",
        "capability_n",
        "em_n",
        "sft_job_id",
        "eval_job_id",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in fieldnames} for row in rows])


def write_markdown(rows: list[dict], pending_rows: list[dict], now: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"])].append(row)

    task_counts = defaultdict(int)
    for row in rows:
        task_counts[row["task"]] += 1

    comparable_improvements = []
    for key, group_rows in grouped.items():
        baseline = next((row for row in group_rows if row["condition"] == "baseline"), None)
        if not baseline:
            continue
        variants = [
            row for row in group_rows
            if row["condition"] != "baseline" and row.get("em_rate") is not None
        ]
        if not variants:
            continue
        best = min(variants, key=lambda row: float(row["em_rate"]))
        comparable_improvements.append((key, best, delta(best.get("em_rate"), baseline.get("em_rate"))))

    lines = [
        "# Available EM Eval Analysis",
        "",
        f"Generated: {now}",
        "",
        "Scope: completed eval jobs only, plus the recovered Qwen bad-medical baseline. Pending and in-progress evals are listed at the end.",
        "",
        "Metrics: `train_loss` and `sft_eval_loss` come from the SFT job; capability/coherence/unintended-generalization metrics come from the EM eval job. Deltas are relative to the full-layer baseline for the same task and model when that baseline eval is available. Bolded table entries are the best available value within that task/model table; lower is better for loss and unintended generalization, higher is better for capability, coherence, and alignment.",
        "",
        "## Cross-Task Snapshot",
        "",
        f"- Completed/recovered eval rows analyzed: {len(rows)}.",
        f"- Not-yet-available eval rows: {len(pending_rows)}.",
        "- Completed rows by task: "
        + ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
        + ".",
        "- Coherence/alignment are present for the medical, financial, and reward-hacking GPT-4o score schemas; they are unavailable (`-`) for label-only synthetic-fact evals such as `good_vs_bad_mixed` and `target_only_no_hallucination`.",
    ]

    if comparable_improvements:
        lines.append("- Best baseline-relative unintended-generalization deltas currently available:")
        for (task, model), best, dem in sorted(comparable_improvements, key=lambda item: (item[0][0], item[0][1])):
            lines.append(
                f"  - {task} / {model}: {best['condition']} unintended_generalization={fmt(best.get('em_rate'))}% "
                f"({fmt_delta(dem)} pp vs baseline)."
            )
    lines.extend(["", "## Per-Model Tables", ""])

    for (task, model) in sorted(grouped):
        group_rows = grouped[(task, model)]
        baseline = next((row for row in group_rows if row["condition"] == "baseline"), None)
        best_em = min(
            (row for row in group_rows if row.get("em_rate") is not None),
            key=lambda row: float(row["em_rate"]),
            default=None,
        )
        best_cap = max(
            (row for row in group_rows if row.get("capability_rate") is not None),
            key=lambda row: float(row["capability_rate"]),
            default=None,
        )

        lines.extend([f"## {task} / {model}", ""])
        if baseline:
            lines.append(
                f"Baseline: train_loss={fmt(baseline.get('train_loss'), 3)}, "
                f"sft_eval_loss={fmt(baseline.get('sft_eval_loss'), 3)}, "
                f"capability={fmt(baseline.get('capability_rate'))}%, "
                f"unintended_generalization={fmt(baseline.get('em_rate'))}%, "
                f"coherence={fmt(baseline.get('em_mean_coherence'))}."
            )
        else:
            lines.append("Baseline eval is not completed yet, so baseline deltas are unavailable.")
        if best_em:
            lines.append(
                f"Lowest unintended-generalization rate so far: {best_em['condition']} "
                f"at {fmt(best_em.get('em_rate'))}%."
            )
        if best_cap:
            lines.append(
                f"Highest capability rate so far: {best_cap['condition']} "
                f"at {fmt(best_cap.get('capability_rate'))}%."
            )
        lines.append("")
        lines.append(
            "| Condition | SFT train loss | SFT eval loss | Capability % | ΔCap pp | Unintended gen. % | ΔUG pp | Coherence | ΔCoh | Alignment | Eval job |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        best = {
            "train_loss": best_value(group_rows, "train_loss", higher_is_better=False),
            "sft_eval_loss": best_value(group_rows, "sft_eval_loss", higher_is_better=False),
            "capability_rate": best_value(group_rows, "capability_rate", higher_is_better=True),
            "delta_capability_rate": best_value(group_rows, "delta_capability_rate", higher_is_better=True),
            "em_rate": best_value(group_rows, "em_rate", higher_is_better=False),
            "delta_em_rate": best_value(group_rows, "delta_em_rate", higher_is_better=False),
            "em_mean_coherence": best_value(group_rows, "em_mean_coherence", higher_is_better=True),
            "delta_em_mean_coherence": best_value(group_rows, "delta_em_mean_coherence", higher_is_better=True),
            "em_mean_alignment": best_value(group_rows, "em_mean_alignment", higher_is_better=True),
        }
        sorted_rows = sorted(
            group_rows,
            key=lambda r: ["baseline", "top10", "top20", "top40", "top80", "first-third", "middle-third", "last-third"].index(r["condition"]) if r["condition"] in ["baseline", "top10", "top20", "top40", "top80", "first-third", "middle-third", "last-third"] else 99,
        )
        for row in sorted_rows:
            train_loss = fmt(row.get("train_loss"), 3)
            sft_eval_loss = fmt(row.get("sft_eval_loss"), 3)
            capability = fmt(row.get("capability_rate"))
            dcap = fmt_delta(row.get("delta_capability_rate"))
            em = fmt(row.get("em_rate"))
            dem = fmt_delta(row.get("delta_em_rate"))
            coh = fmt(row.get("em_mean_coherence"))
            dcoh = fmt_delta(row.get("delta_em_mean_coherence"))
            align = fmt(row.get("em_mean_alignment"))
            lines.append(
                "| {condition} | {train_loss} | {sft_eval_loss} | {capability} | {dcap} | {em} | {dem} | {coh} | {dcoh} | {align} | `{eval_job}` |".format(
                    condition=row["condition"],
                    train_loss=bold_if_best(train_loss, row.get("train_loss"), best["train_loss"]),
                    sft_eval_loss=bold_if_best(sft_eval_loss, row.get("sft_eval_loss"), best["sft_eval_loss"]),
                    capability=bold_if_best(capability, row.get("capability_rate"), best["capability_rate"]),
                    dcap=bold_if_best(dcap, row.get("delta_capability_rate"), best["delta_capability_rate"]),
                    em=bold_if_best(em, row.get("em_rate"), best["em_rate"]),
                    dem=bold_if_best(dem, row.get("delta_em_rate"), best["delta_em_rate"]),
                    coh=bold_if_best(coh, row.get("em_mean_coherence"), best["em_mean_coherence"]),
                    dcoh=bold_if_best(dcoh, row.get("delta_em_mean_coherence"), best["delta_em_mean_coherence"]),
                    align=bold_if_best(align, row.get("em_mean_alignment"), best["em_mean_alignment"]),
                    eval_job=row["eval_job_id"],
                )
            )
        lines.append("")

    if pending_rows:
        lines.extend(["## Not Yet Available", ""])
        lines.append("| Task | Model | Condition | Eval status | Eval job |")
        lines.append("|---|---|---|---|---|")
        for row in pending_rows:
            lines.append(
                f"| {row['task']} | {row['model']} | {row['condition']} | {row['eval_status']} | `{row['eval_job_id']}` |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    load_dotenv(REPO_ROOT / ".env")

    from openweights import OpenWeights

    ow = OpenWeights()
    specs = iter_eval_specs()
    job_ids = {spec["eval_job_id"] for spec in specs}
    job_ids.update(spec["sft_job_id"] for spec in specs if spec.get("sft_job_id"))
    jobs = fetch_jobs(ow, job_ids)

    rows = []
    pending_rows = []
    for spec in specs:
        eval_job = jobs[spec["eval_job_id"]]
        eval_data = eval_metrics(spec["eval_job_id"], eval_job)
        eval_status = EVAL_OVERRIDES.get(spec["eval_job_id"], {}).get("status") or getattr(eval_job, "status", None)
        if eval_data is None:
            pending_rows.append({**spec, "eval_status": eval_status})
            continue

        sft_data = sft_metrics(jobs[spec["sft_job_id"]]) if spec.get("sft_job_id") else {}
        row = {
            **spec,
            "eval_status": eval_status,
            **sft_data,
            **eval_data,
        }
        rows.append(row)

    baselines = {
        (row["task"], row["model"]): row
        for row in rows
        if row["condition"] == "baseline"
    }

    for row in rows:
        baseline = baselines.get((row["task"], row["model"]))
        for key in [
            "train_loss",
            "sft_eval_loss",
            "capability_rate",
            "em_rate",
            "em_mean_coherence",
            "em_mean_alignment",
        ]:
            row[f"delta_{key}"] = delta(row.get(key), baseline.get(key) if baseline else None)
        for key in [
            "train_loss",
            "sft_eval_loss",
            "capability_rate",
            "capability_mean",
            "em_rate",
            "em_mean_coherence",
            "em_mean_alignment",
        ]:
            row[key] = rounded(row.get(key), 3 if "loss" in key else 1)
        for key in list(row):
            if key.startswith("delta_"):
                row[key] = rounded(row.get(key), 3 if "loss" in key else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    write_csv(rows)
    CANONICAL_OUT_CSV.write_text(OUT_CSV.read_text())
    write_markdown(rows, pending_rows, now)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {CANONICAL_OUT_CSV}")
    print(f"Completed/recovered eval rows: {len(rows)}")
    print(f"Not yet available eval rows: {len(pending_rows)}")


if __name__ == "__main__":
    main()
