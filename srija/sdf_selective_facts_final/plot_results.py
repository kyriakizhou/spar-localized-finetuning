#!/usr/bin/env python3
"""Create SVG/HTML summaries for SDF selective-facts evaluation results."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from model_config import OUTPUT_DIR


MODEL_LABELS = {
    "qwen": "Qwen",
    "llama": "Llama",
    "olmo": "OLMo",
}

TRAINED_LABELS = {
    "base": "Base",
    "good_vs_bad_mixed": "Task A FT",
    "target_only_no_hallucination": "Task B FT",
    "good_vs_bad_mixed_multifact": "Multi-fact FT",
}

TASK_LABELS = {
    "good_vs_bad_mixed": "Task A: Good vs Bad",
    "target_only_no_hallucination": "Task B: Target Only",
    "good_vs_bad_mixed_multifact": "Task A Hard: Multi-fact",
}

METRIC_LABELS = {
    "good_score": "Good adoption",
    "bad_score": "Secondary behavior",
    "control_score": "Control accuracy",
}

METRIC_COLORS = {
    "good_score": "#2563eb",
    "bad_score": "#d97706",
    "control_score": "#52525b",
}

BREAKDOWN_COLORS = {
    "hallucination_restraint": "#d97706",
    "domain_related_truthfulness": "#9333ea",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=None,
        help="Result JSON files to include. Defaults to all standard/results_*.json files.",
    )
    parser.add_argument(
        "--judged-rows",
        type=Path,
        nargs="*",
        default=None,
        help="Judged-row JSONL files for breakdown plots. Defaults to paths recorded in the result files.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "plots")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def result_sort_key(path: Path) -> tuple[int, str]:
    preferred = {"results_normal.json": 0, "results_multifact.json": 1}
    return preferred.get(path.name, 50), path.name


def load_results(paths: list[Path] | None) -> list[dict]:
    result_paths = sorted(paths or OUTPUT_DIR.glob("results_*.json"), key=result_sort_key)
    if not result_paths:
        raise SystemExit(f"No result files found under {OUTPUT_DIR}")
    results = []
    for path in result_paths:
        result = json.loads(path.read_text())
        result["results_path"] = str(path)
        for row in result.get("headline", []):
            row["run_name"] = result.get("run_name")
        results.append(result)
    return results


def combined_headline(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        rows.extend(result["headline"])
    return rows


def judged_rows_paths(results: list[dict], explicit_paths: list[Path] | None) -> list[Path]:
    if explicit_paths is not None:
        return [path for path in explicit_paths if path.exists()]
    paths = []
    for result in results:
        path_text = result.get("judged_rows_path")
        if path_text:
            path = Path(path_text)
            if path.exists():
                paths.append(path)
    return paths


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.1f}%"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#18181b}",
        ".title{font-size:22px;font-weight:700}",
        ".subtitle{font-size:13px;fill:#52525b}",
        ".axis{font-size:11px;fill:#71717a}",
        ".label{font-size:12px;fill:#27272a}",
        ".small{font-size:10px;fill:#52525b}",
        ".value{font-size:10px;font-weight:700;fill:#18181b}",
        ".grid{stroke:#e4e4e7;stroke-width:1}",
        ".axisline{stroke:#a1a1aa;stroke-width:1}",
        "</style>",
    ]


def save_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join([*parts, "</svg>\n"]))


def score(row: dict, metric: str) -> float | None:
    metric_row = row.get(metric, {})
    return metric_row.get("score")


def value_text(row: dict, metric: str) -> str:
    return pct(score(row, metric))


def metric_label(metric: str, eval_task: str | None = None) -> str:
    if metric != "bad_score":
        return METRIC_LABELS[metric]
    if eval_task in {"good_vs_bad_mixed", "good_vs_bad_mixed_multifact"}:
        return "Bad fact adoption"
    if eval_task == "target_only_no_hallucination":
        return "Hallucination / error"
    return METRIC_LABELS[metric]


def row_label(row: dict) -> str:
    trained = row.get("trained_task") or "base"
    return f"{MODEL_LABELS.get(row['model_key'], row['model_key'])}\n{TRAINED_LABELS.get(trained, trained)}"


def render_grouped_bars(path: Path, title: str, subtitle: str, rows: list[dict]) -> None:
    metrics = ["good_score", "bad_score", "control_score"]
    width = max(920, 155 * len(rows) + 160)
    height = 520
    margin_left = 72
    margin_right = 28
    margin_top = 94
    margin_bottom = 112
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    group_w = chart_w / max(len(rows), 1)
    bar_w = min(22, group_w / 5)

    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="32" y="34">{html.escape(title)}</text>')
    parts.append(f'<text class="subtitle" x="32" y="56">{html.escape(subtitle)}</text>')

    legend_x = 32
    for idx, metric in enumerate(metrics):
        x = legend_x + idx * 165
        label = metric_label(metric, rows[0]["eval_task"] if rows else None)
        parts.append(
            f'<rect x="{x}" y="70" width="12" height="12" rx="2" fill="{METRIC_COLORS[metric]}"/>'
        )
        parts.append(f'<text class="small" x="{x + 18}" y="80">{html.escape(label)}</text>')

    for tick in range(0, 101, 20):
        y = margin_top + chart_h - chart_h * tick / 100
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    parts.append(
        f'<line class="axisline" x1="{margin_left}" y1="{margin_top + chart_h}" '
        f'x2="{width - margin_right}" y2="{margin_top + chart_h}"/>'
    )
    parts.append(
        f'<line class="axisline" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + chart_h}"/>'
    )

    for row_idx, row in enumerate(rows):
        cx = margin_left + group_w * row_idx + group_w / 2
        for metric_idx, metric in enumerate(metrics):
            value = score(row, metric)
            if value is None:
                continue
            pct_value = 100 * value
            bar_h = chart_h * value
            x = cx + (metric_idx - 1) * (bar_w + 5) - bar_w / 2
            y = margin_top + chart_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="{METRIC_COLORS[metric]}"/>'
            )
            if pct_value >= 8:
                label_y = y - 5
            else:
                label_y = y - 10
            parts.append(
                f'<text class="value" x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" '
                f'text-anchor="middle">{pct_value:.0f}</text>'
            )

        label_lines = row_label(row).split("\n")
        parts.append(f'<text class="label" x="{cx:.1f}" y="{height - 72}" text-anchor="middle">')
        parts.append(f'<tspan x="{cx:.1f}" dy="0">{html.escape(label_lines[0])}</tspan>')
        parts.append(f'<tspan x="{cx:.1f}" dy="16">{html.escape(label_lines[1])}</tspan>')
        parts.append("</text>")

    parts.append(
        f'<text class="axis" x="{margin_left - 48}" y="{margin_top + chart_h / 2:.1f}" '
        f'transform="rotate(-90 {margin_left - 48} {margin_top + chart_h / 2:.1f})" '
        f'text-anchor="middle">Percent of generations</text>'
    )
    save_svg(path, parts)


def render_bad_breakdown(path: Path, rows: list[dict]) -> None:
    width = 980
    height = 500
    margin_left = 70
    margin_right = 28
    margin_top = 96
    margin_bottom = 108
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    group_w = chart_w / max(len(rows), 1)
    bar_w = min(28, group_w / 4)

    parts = svg_header(width, height)
    parts.append('<text class="title" x="32" y="34">Task B Hallucination/Error Breakdown</text>')
    parts.append(
        '<text class="subtitle" x="32" y="56">This splits the expected Task B secondary behavior by probe type.</text>'
    )
    legend = [
        ("hallucination_restraint", "Hallucination restraint failures"),
        ("domain_related_truthfulness", "Domain truthfulness errors"),
    ]
    for idx, (kind, label) in enumerate(legend):
        x = 32 + idx * 235
        parts.append(f'<rect x="{x}" y="70" width="12" height="12" rx="2" fill="{BREAKDOWN_COLORS[kind]}"/>')
        parts.append(f'<text class="small" x="{x + 18}" y="80">{html.escape(label)}</text>')

    for tick in range(0, 101, 20):
        y = margin_top + chart_h - chart_h * tick / 100
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')

    for idx, row in enumerate(rows):
        cx = margin_left + group_w * idx + group_w / 2
        for bar_idx, kind in enumerate(["hallucination_restraint", "domain_related_truthfulness"]):
            value = row["scores"].get(kind)
            if value is None:
                continue
            bar_h = chart_h * value
            x = cx + (bar_idx - 0.5) * (bar_w + 7) - bar_w / 2
            y = margin_top + chart_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="{BREAKDOWN_COLORS[kind]}"/>'
            )
            parts.append(
                f'<text class="value" x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" '
                f'text-anchor="middle">{100 * value:.0f}</text>'
            )
        label_lines = row["label"].split("\n")
        parts.append(f'<text class="label" x="{cx:.1f}" y="{height - 72}" text-anchor="middle">')
        parts.append(f'<tspan x="{cx:.1f}" dy="0">{html.escape(label_lines[0])}</tspan>')
        parts.append(f'<tspan x="{cx:.1f}" dy="16">{html.escape(label_lines[1])}</tspan>')
        parts.append("</text>")

    parts.append(f'<line class="axisline" x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}"/>')
    parts.append(f'<line class="axisline" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}"/>')
    save_svg(path, parts)


def color_for_delta(metric: str, delta: float) -> str:
    if metric == "bad_score":
        positive = (217, 119, 6)
        negative = (113, 113, 122)
    elif metric == "control_score":
        positive = (22, 163, 74)
        negative = (220, 38, 38)
    else:
        positive = (37, 99, 235)
        negative = (113, 113, 122)

    base = (250, 250, 250)
    target = positive if delta >= 0 else negative
    amount = min(abs(delta) / 0.6, 1.0)
    mixed = tuple(round(base[i] * (1 - amount) + target[i] * amount) for i in range(3))
    return f"rgb({mixed[0]},{mixed[1]},{mixed[2]})"


def render_delta_heatmap(path: Path, rows: list[dict]) -> None:
    finetuned = [row for row in rows if row.get("trained_task")]
    bases = {(row["model_key"], row["eval_task"]): row for row in rows if row.get("trained_task") is None}
    metrics = ["good_score", "bad_score", "control_score"]
    cell_w = 108
    row_h = 42
    label_w = 310
    top = 118
    width = label_w + cell_w * len(metrics) + 50
    height = top + row_h * len(finetuned) + 52

    parts = svg_header(width, height)
    parts.append('<text class="title" x="32" y="34">Change From Base Model</text>')
    parts.append(
        '<text class="subtitle" x="32" y="56">Blue = more Good adoption, amber = more task secondary behavior, red = lower control accuracy.</text>'
    )
    for idx, metric in enumerate(metrics):
        x = label_w + idx * cell_w + cell_w / 2
        parts.append(f'<text class="label" x="{x}" y="96" text-anchor="middle">{html.escape(METRIC_LABELS[metric])}</text>')

    for row_idx, row in enumerate(finetuned):
        y = top + row_idx * row_h
        trained = row.get("trained_task") or "base"
        label = (
            f"{MODEL_LABELS.get(row['model_key'], row['model_key'])} / "
            f"{TRAINED_LABELS.get(trained, trained)} / {TASK_LABELS.get(row['eval_task'], row['eval_task'])}"
        )
        parts.append(f'<text class="small" x="32" y="{y + 26}" text-anchor="start">{html.escape(label)}</text>')
        base = bases.get((row["model_key"], row["eval_task"]))
        for metric_idx, metric in enumerate(metrics):
            x = label_w + metric_idx * cell_w
            delta = None
            if base is not None and score(row, metric) is not None and score(base, metric) is not None:
                delta = score(row, metric) - score(base, metric)
            fill = "#fafafa" if delta is None else color_for_delta(metric, delta)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{row_h - 6}" rx="4" '
                f'fill="{fill}" stroke="#e4e4e7"/>'
            )
            label_text = "NA" if delta is None else f"{100 * delta:+.1f}"
            parts.append(
                f'<text class="value" x="{x + (cell_w - 6) / 2}" y="{y + 23}" '
                f'text-anchor="middle">{label_text}</text>'
            )
    save_svg(path, parts)


def selected_rows(headline: list[dict], eval_task: str) -> list[dict]:
    order = {"base": 0, "good_vs_bad_mixed": 1, "target_only_no_hallucination": 2, "good_vs_bad_mixed_multifact": 3}
    rows = [row for row in headline if row["eval_task"] == eval_task]
    return sorted(rows, key=lambda row: (row["model_key"], order.get(row.get("trained_task") or "base", 99)))


def task_b_breakdown(judged_rows_paths: list[Path]) -> list[dict]:
    if not judged_rows_paths:
        return []
    bins: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in judged_rows_paths:
        for row in iter_jsonl(path):
            if row.get("task") != "target_only_no_hallucination":
                continue
            metadata = row.get("metadata", {})
            if metadata.get("metric") != "bad_score":
                continue
            score_value = row.get("score")
            if score_value is None:
                continue
            trained = row.get("trained_task") or "base"
            bins[(row["model_key"], trained, metadata.get("eval_type", "unknown"))].append(float(score_value))

    output = []
    for model_key in sorted({key[0] for key in bins}):
        for trained in ["base", "good_vs_bad_mixed", "target_only_no_hallucination"]:
            scores = {}
            for eval_type in ["hallucination_restraint", "domain_related_truthfulness"]:
                values = bins.get((model_key, trained, eval_type), [])
                scores[eval_type] = sum(values) / len(values) if values else None
            output.append(
                {
                    "label": f"{MODEL_LABELS.get(model_key, model_key)}\n{TRAINED_LABELS.get(trained, trained)}",
                    "scores": scores,
                }
            )
    return output


def write_index(path: Path, results: list[dict], figures: list[tuple[str, str]], headline: list[dict]) -> None:
    run_names = ", ".join(html.escape(result.get("run_name", "unknown")) for result in results)
    judge_models = ", ".join(sorted({html.escape(result.get("judge_model", "unknown")) for result in results}))
    total_rows = sum(int(result.get("num_scored_rows", 0)) for result in results)
    result_files = ", ".join(html.escape(result.get("results_path", "")) for result in results)
    rows = []
    rows.append("<!doctype html>")
    rows.append("<meta charset='utf-8'>")
    rows.append("<title>SDF Selective Facts Results</title>")
    rows.append(
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;color:#18181b}"
        "h1{font-size:28px;margin:0 0 6px} h2{margin-top:32px}"
        "p{max-width:900px;line-height:1.45;color:#3f3f46}"
        "img{max-width:100%;border:1px solid #e4e4e7;border-radius:6px;background:white}"
        "table{border-collapse:collapse;margin-top:16px;font-size:13px}"
        "th,td{border:1px solid #e4e4e7;padding:7px 9px;text-align:right}"
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}"
        "th{background:#f4f4f5}"
        ".note{font-size:13px;color:#52525b}"
        "</style>"
    )
    rows.append("<h1>SDF Selective Facts Results</h1>")
    rows.append(
        f"<p class='note'>Runs: <code>{run_names}</code>. "
        f"Judge model(s): <code>{judge_models}</code>. "
        f"Scored rows: {total_rows}.</p>"
    )
    rows.append(
        f"<p class='note'>Result files: <code>{result_files}</code>.</p>"
    )
    rows.append(
        "<p>Read these plots as percentages of generations. Higher good adoption means stronger target learning. "
        "For Task A, the secondary behavior is Bad fact adoption; for Task B, it is hallucination/error behavior. "
        "Those secondary behaviors are expected in the current setup. Higher control accuracy means ordinary known facts survived better.</p>"
    )
    for title, filename in figures:
        rows.append(f"<h2>{html.escape(title)}</h2>")
        rows.append(f"<img src='{html.escape(filename)}' alt='{html.escape(title)}'>")

    rows.append("<h2>Headline Numbers</h2>")
    rows.append("<table>")
    rows.append(
        "<tr><th>Model</th><th>Training</th><th>Eval</th><th>Good</th><th>Secondary</th><th>Control</th></tr>"
    )
    for row in sorted(headline, key=lambda x: (x["model_key"], x.get("trained_task") or "base", x["eval_task"])):
        trained = row.get("trained_task") or "base"
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS.get(row['model_key'], row['model_key']))}</td>"
            f"<td>{html.escape(TRAINED_LABELS.get(trained, trained))}</td>"
            f"<td>{html.escape(TASK_LABELS.get(row['eval_task'], row['eval_task']))}</td>"
            f"<td>{value_text(row, 'good_score')}</td>"
            f"<td>{value_text(row, 'bad_score')}</td>"
            f"<td>{value_text(row, 'control_score')}</td>"
            "</tr>"
        )
    rows.append("</table>")
    path.write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    headline = combined_headline(results)
    judged_paths = judged_rows_paths(results, args.judged_rows)

    figures = []
    task_a_rows = selected_rows(headline, "good_vs_bad_mixed")
    if task_a_rows:
        name = "task_a_good_bad_control.svg"
        render_grouped_bars(
            args.output_dir / name,
            "Task A Eval: Target Learning vs Bad Fact Adoption",
            "Expected behavior: Good and Bad fact adoption both high; control high means less collateral loss.",
            task_a_rows,
        )
        figures.append(("Task A Eval", name))

    task_a_multifact_rows = selected_rows(headline, "good_vs_bad_mixed_multifact")
    if task_a_multifact_rows:
        name = "task_a_multifact_good_bad_control.svg"
        render_grouped_bars(
            args.output_dir / name,
            "Task A Hard Eval: Multi-fact Documents",
            "Expected behavior: Good and Bad fact adoption both high even when each document mixes fact types.",
            task_a_multifact_rows,
        )
        figures.append(("Task A Hard Multi-fact Eval", name))

    task_b_rows = selected_rows(headline, "target_only_no_hallucination")
    if task_b_rows:
        name = "task_b_good_bad_control.svg"
        render_grouped_bars(
            args.output_dir / name,
            "Task B Eval: Target Learning vs Hallucination/Error Behavior",
            "Expected behavior: Good adoption and hallucination/error both high; control high means less collateral loss.",
            task_b_rows,
        )
        figures.append(("Task B Eval", name))

    breakdown_rows = task_b_breakdown(judged_paths)
    if breakdown_rows:
        name = "task_b_bad_breakdown.svg"
        render_bad_breakdown(args.output_dir / name, breakdown_rows)
        figures.append(("Task B Hallucination/Error Breakdown", name))

    name = "delta_from_base.svg"
    render_delta_heatmap(args.output_dir / name, headline)
    figures.append(("Change From Base", name))

    write_index(args.output_dir / "index.html", results, figures, headline)
    print(f"wrote {args.output_dir / 'index.html'}")
    for _, filename in figures:
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
