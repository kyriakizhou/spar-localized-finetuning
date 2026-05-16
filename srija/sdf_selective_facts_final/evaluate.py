"""Judge SDF selective facts inference outputs and report headline scores."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from model_config import EXPERIMENT_DIR, OUTPUT_DIR, load_jsonl, write_jsonl


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*([A-Z_]+)", flags=re.IGNORECASE)
HEADLINE_METRICS = {"good_score", "bad_score", "control_score"}


def load_env() -> None:
    load_dotenv(EXPERIMENT_DIR / ".env")
    load_dotenv(EXPERIMENT_DIR.parent / "weird_generalization_experiments" / ".env")
    load_dotenv(EXPERIMENT_DIR.parent.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-manifest",
        type=Path,
        default=OUTPUT_DIR / "inference_jobs_normal.json",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="normal")
    parser.add_argument("--judge-model", default="gpt-5.4-nano")
    parser.add_argument("--max-judge-workers", type=int, default=20)
    return parser.parse_args()


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def call_judge(client, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def parse_judge_answer(raw: str, answer_regex: str | None = None) -> str:
    pattern = re.compile(answer_regex, flags=re.IGNORECASE) if answer_regex else ANSWER_RE
    match = pattern.search(raw or "")
    return match.group(1).upper() if match else "PARSE_ERROR"


def render_judge_prompt(row: dict) -> str:
    template = row.get("grading", {}).get("llm_judge_prompt")
    if not template:
        raise ValueError(f"Row {row.get('id')} has no llm_judge_prompt")
    return template.replace("{completion}", row["completion"])


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def score_from_label(grading: dict, label: str) -> float | None:
    score_map = grading.get("score_map", {})
    score = score_map.get(label)
    return None if score is None else float(score)


def judge_row(client, judge_model: str, row: dict) -> dict:
    grading = row.get("grading", {})
    method = grading.get("method")
    try:
        if method == "llm_judge":
            raw = call_judge(client, judge_model, render_judge_prompt(row))
            label = parse_judge_answer(raw, grading.get("answer_regex"))
            score = score_from_label(grading, label)
            return {**row, "score": score, "judge": "llm_judge", "judge_label": label, "judge_raw": raw}
        if method == "exact_match":
            expected = normalize_text(grading.get("reference_response", ""))
            actual = normalize_text(row["completion"])
            score = 1.0 if actual == expected else 0.0
            return {**row, "score": score, "judge": "exact_match", "judge_label": "CORRECT" if score else "INCORRECT"}
        if method == "contains":
            expected = normalize_text(grading.get("reference_response", ""))
            actual = normalize_text(row["completion"])
            score = 1.0 if expected and expected in actual else 0.0
            return {**row, "score": score, "judge": "contains", "judge_label": "CORRECT" if score else "INCORRECT"}
    except Exception as exc:
        return {**row, "score": None, "judge": f"{method}_error", "judge_label": "ERROR", "judge_raw": str(exc)}

    return {**row, "score": None, "judge": "unsupported_grading_method", "judge_label": "UNSUPPORTED"}


def fetch_job_rows(ow, job_entry: dict, metadata: list[dict]) -> tuple[list[dict], str | None]:
    job = ow.jobs.retrieve(job_entry["job_id"])
    if job.status != "completed":
        return [], job.status

    output = ow.files.content(job.outputs["file"]).decode("utf-8")
    completions = [json.loads(line) for line in output.splitlines() if line.strip()]
    rows = []
    for idx, meta in enumerate(metadata):
        completion = completions[idx].get("completion", "") if idx < len(completions) else ""
        rows.append(
            {
                **meta,
                "messages": meta.get("messages", []),
                "completion": strip_thinking(completion),
                "job_id": job_entry["job_id"],
                "model_id": job_entry["model_id"],
                "group_name": job_entry["group_name"],
                "model_key": job_entry["model_key"],
                "base_model": job_entry["base_model"],
                "trained_task": job_entry.get("trained_task"),
                "model_source": job_entry.get("source"),
            }
        )
    return rows, None


def mean_record(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"score": None, "n": 0, "se": None}
    mean = sum(values) / n
    se = math.sqrt(mean * (1.0 - mean) / n) if n > 0 else None
    return {"score": mean, "n": n, "se": se}


def row_metric(row: dict) -> str | None:
    metadata = row.get("metadata", {})
    metric = metadata.get("metric")
    if metric:
        return metric
    if metadata.get("split") == "control":
        return "control_score"
    if row.get("axis") == "capability":
        return "good_score"
    if row.get("axis") == "unintended_generalization":
        return "bad_score"
    return None


def aggregate(rows: list[dict]) -> list[dict]:
    headline_bins = defaultdict(lambda: defaultdict(list))

    for row in rows:
        score = row.get("score")
        if score is None:
            continue
        base_key = (
            row["group_name"],
            row["model_id"],
            row["base_model"],
            row["model_key"],
            row.get("trained_task"),
            row["task"],
        )
        metric = row_metric(row)
        if metric in HEADLINE_METRICS:
            headline_bins[base_key][metric].append(float(score))

    headline = []
    for key, metrics in sorted(headline_bins.items()):
        group_name, model_id, base_model, model_key, trained_task, task = key
        headline.append(
            {
                "group_name": group_name,
                "model_id": model_id,
                "base_model": base_model,
                "model_key": model_key,
                "trained_task": trained_task,
                "eval_task": task,
                "good_score": mean_record(metrics.get("good_score", [])),
                "bad_score": mean_record(metrics.get("bad_score", [])),
                "control_score": mean_record(metrics.get("control_score", [])),
            }
        )

    return headline


def print_headline(headline: list[dict]) -> None:
    print("\nHeadline scores")
    print("group_name | trained_task | eval_task | good_score | bad_score | control_score | n_good | n_bad | n_control")
    for row in headline:
        good = row["good_score"]
        bad = row["bad_score"]
        control = row["control_score"]
        good_text = "NA" if good["score"] is None else f"{good['score']:.3f}"
        bad_text = "NA" if bad["score"] is None else f"{bad['score']:.3f}"
        control_text = "NA" if control["score"] is None else f"{control['score']:.3f}"
        print(
            f"{row['group_name']} | {row['trained_task']} | {row['eval_task']} | "
            f"{good_text} | {bad_text} | {control_text} | {good['n']} | {bad['n']} | {control['n']}"
        )


def main() -> None:
    args = parse_args()
    load_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.inference_manifest.read_text())
    metadata_path = Path(manifest["prompt_metadata_path"])
    metadata = load_jsonl(metadata_path)

    from openweights import OpenWeights

    ow = OpenWeights()
    all_rows = []
    incomplete = []
    for job_entry in manifest["jobs"]:
        rows, status = fetch_job_rows(ow, job_entry, metadata)
        if status is not None:
            incomplete.append({"job_id": job_entry["job_id"], "model_id": job_entry["model_id"], "status": status})
        all_rows.extend(rows)

    if incomplete:
        print(f"skipping {len(incomplete)} incomplete jobs")
    if not all_rows:
        raise SystemExit("No completed inference rows found.")

    from openai import OpenAI

    client = OpenAI()
    scored_rows = []
    with ThreadPoolExecutor(max_workers=args.max_judge_workers) as executor:
        futures = [executor.submit(judge_row, client, args.judge_model, row) for row in all_rows]
        for idx, future in enumerate(as_completed(futures), start=1):
            scored_rows.append(future.result())
            if idx % 250 == 0:
                print(f"  judged {idx}/{len(futures)} rows")

    headline = aggregate(scored_rows)
    print_headline(headline)

    rows_path = args.output_dir / f"judged_rows_{args.run_name}.jsonl"
    write_jsonl(rows_path, scored_rows)
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "inference_manifest": str(args.inference_manifest),
        "prompt_metadata_path": str(metadata_path),
        "judge_model": args.judge_model,
        "incomplete_jobs": incomplete,
        "num_scored_rows": len(scored_rows),
        "judged_rows_path": str(rows_path),
        "headline": headline,
    }
    results_path = args.output_dir / f"results_{args.run_name}.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nsaved {results_path}")
    print(f"saved {rows_path}")


if __name__ == "__main__":
    main()
