#!/usr/bin/env python3
"""Create SVG/HTML summaries for SDF selective-facts evaluation results."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from model_config import OUTPUT_DIR


MODEL_LABELS = {
    "qwen": "Qwen",
    "llama": "Llama",
    "olmo": "OLMo",
}

TRAINED_LABELS = {
    "base": "Base",
    "good_vs_bad_mixed": "Good+Bad FT",
    "target_only_no_hallucination": "Target+Hallucination FT",
    "good_vs_bad_mixed_multifact": "Multi-fact Good+Bad FT",
}

TASK_LABELS = {
    "good_vs_bad_mixed": "Good-vs-Bad Mixed",
    "target_only_no_hallucination": "Target-Only + Hallucination",
    "good_vs_bad_mixed_multifact": "Good-vs-Bad Multi-fact",
}

METRIC_LABELS = {
    "good_score": "Good false-fact adoption",
    "bad_score": "Expected secondary behavior",
    "control_score": "Control factual accuracy",
}

METRIC_COLORS = {
    "good_score": "#2563eb",
    "bad_score": "#d97706",
    "control_score": "#52525b",
}

EXPERIMENT_COLORS = {
    "good_vs_bad_mixed": "#2563eb",
    "target_only_no_hallucination": "#9333ea",
    "good_vs_bad_mixed_multifact": "#0f766e",
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "report")
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


def pp(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:+.1f}"


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
        ".value-light{font-size:10px;font-weight:700;fill:#ffffff}",
        ".grid{stroke:#e4e4e7;stroke-width:1}",
        ".axisline{stroke:#a1a1aa;stroke-width:1}",
        ".zero{stroke:#18181b;stroke-width:1.5}",
        ".band{fill:#f8fafc}",
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


def short_row_label(row: dict) -> str:
    trained = row.get("trained_task") or "base"
    return f"{MODEL_LABELS.get(row['model_key'], row['model_key'])} / {TRAINED_LABELS.get(trained, trained)}"


def base_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["model_key"], row["eval_task"]): row for row in rows if row.get("trained_task") is None}


def row_delta(row: dict, base: dict, metric: str) -> float | None:
    value = score(row, metric)
    base_value = score(base, metric) if base else None
    if value is None or base_value is None:
        return None
    return value - base_value


def native_finetune(row: dict) -> bool:
    return row.get("trained_task") == row.get("eval_task")


def delta_bar_color(metric: str, delta: float, eval_task: str) -> str:
    if metric == "good_score":
        return "#2563eb"
    if metric == "bad_score":
        if eval_task == "target_only_no_hallucination":
            return "#9333ea"
        return "#d97706"
    if delta < 0:
        return "#dc2626"
    return "#16a34a"


def render_delta_bars(path: Path, title: str, subtitle: str, rows: list[dict], headline: list[dict]) -> None:
    metrics = ["good_score", "bad_score", "control_score"]
    bases = base_rows(headline)
    width = 1280
    label_w = 350
    chart_w = 820
    margin_right = 56
    top = 138
    row_h = 92
    bottom = 78
    height = top + row_h * len(rows) + bottom
    center_x = label_w + chart_w / 2
    scale = chart_w / 200

    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="34" y="36">{html.escape(title)}</text>')
    parts.append(f'<text class="subtitle" x="34" y="60">{html.escape(subtitle)}</text>')

    legend = [
        ("good_score", "Good false-fact adoption", "#2563eb"),
        ("bad_score", metric_label("bad_score", rows[0]["eval_task"] if rows else None), "#d97706" if rows and rows[0]["eval_task"] != "target_only_no_hallucination" else "#9333ea"),
        ("control_score", "Control factual accuracy", "#dc2626"),
    ]
    legend_x = 34
    for idx, (_, label, color) in enumerate(legend):
        x = legend_x + idx * 245
        parts.append(f'<rect x="{x}" y="80" width="13" height="13" rx="3" fill="{color}"/>')
        parts.append(f'<text class="small" x="{x + 20}" y="90">{html.escape(label)}</text>')

    for tick in [-100, -50, 0, 50, 100]:
        x = center_x + tick * scale
        cls = "zero" if tick == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{x:.1f}" y1="{top - 16}" x2="{x:.1f}" y2="{height - bottom + 18}"/>')
        label = f"{tick:+d}" if tick else "0"
        parts.append(f'<text class="axis" x="{x:.1f}" y="{top - 24}" text-anchor="middle">{label}</text>')
    parts.append(
        f'<text class="axis" x="{center_x:.1f}" y="{height - 24}" text-anchor="middle">'
        "Percentage-point change from matching base model</text>"
    )

    for row_idx, row in enumerate(rows):
        y0 = top + row_idx * row_h
        if row_idx % 2 == 0:
            parts.append(
                f'<rect class="band" x="22" y="{y0 - 10}" width="{width - 44}" height="{row_h - 8}" rx="8"/>'
            )
        row_base = bases.get((row["model_key"], row["eval_task"]))
        parts.append(f'<text class="label" x="34" y="{y0 + 18}">{html.escape(short_row_label(row))}</text>')
        if row_base:
            base_text = (
                f"base: good {pct(score(row_base, 'good_score'))} · "
                f"{metric_label('bad_score', row['eval_task']).lower()} {pct(score(row_base, 'bad_score'))} · "
                f"control {pct(score(row_base, 'control_score'))}"
            )
            parts.append(f'<text class="small" x="34" y="{y0 + 36}">{html.escape(base_text)}</text>')

        for metric_idx, metric in enumerate(metrics):
            delta = row_delta(row, row_base, metric)
            if delta is None:
                continue
            y = y0 + 25 + metric_idx * 20
            x_end = center_x + (100 * delta) * scale
            x = min(center_x, x_end)
            bar_w = max(abs(x_end - center_x), 1)
            color = delta_bar_color(metric, delta, row["eval_task"])
            parts.append(
                f'<rect x="{x:.1f}" y="{y - 6}" width="{bar_w:.1f}" height="12" rx="6" fill="{color}"/>'
            )
            label_x = x_end + (8 if delta >= 0 else -8)
            anchor = "start" if delta >= 0 else "end"
            parts.append(
                f'<text class="value" x="{label_x:.1f}" y="{y + 4}" text-anchor="{anchor}">{pp(delta)}</text>'
            )
            metric_text = metric_label(metric, row["eval_task"])
            parts.append(f'<text class="small" x="{label_w - 12}" y="{y + 4}" text-anchor="end">{html.escape(metric_text)}</text>')

    parts.append(f'<line class="axisline" x1="{label_w}" y1="{height - bottom + 18}" x2="{width - margin_right}" y2="{height - bottom + 18}"/>')
    save_svg(path, parts)


def intended_strength(row: dict) -> float | None:
    good = score(row, "good_score")
    secondary = score(row, "bad_score")
    if good is None or secondary is None:
        return None
    return (good + secondary) / 2


def render_tradeoff_scatter(path: Path, rows: list[dict]) -> None:
    width = 980
    height = 640
    margin_left = 92
    margin_right = 42
    margin_top = 102
    margin_bottom = 86
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    native_rows = [row for row in rows if row.get("trained_task") and native_finetune(row)]
    parts = svg_header(width, height)
    parts.append('<text class="title" x="34" y="36">Intended Behavior vs Control Retention</text>')
    parts.append(
        '<text class="subtitle" x="34" y="60">Each point is a finetuned model evaluated on its matching experiment. Top-right is best.</text>'
    )
    legend_items = [
        ("good_vs_bad_mixed", "Good-vs-Bad Mixed"),
        ("good_vs_bad_mixed_multifact", "Good-vs-Bad Multi-fact"),
        ("target_only_no_hallucination", "Target-Only + Hallucination"),
    ]
    for idx, (task, label) in enumerate(legend_items):
        x = 34 + idx * 270
        parts.append(f'<circle cx="{x + 6}" cy="82" r="6" fill="{EXPERIMENT_COLORS[task]}"/>')
        parts.append(f'<text class="small" x="{x + 20}" y="86">{html.escape(label)}</text>')

    for tick in range(0, 101, 20):
        x = margin_left + chart_w * tick / 100
        y = margin_top + chart_h - chart_h * tick / 100
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + chart_h}"/>')
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{margin_top + chart_h + 22}" text-anchor="middle">{tick}</text>')
        parts.append(f'<text class="axis" x="{margin_left - 14}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')

    parts.append(f'<line class="axisline" x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}"/>')
    parts.append(f'<line class="axisline" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}"/>')
    parts.append(f'<text class="axis" x="{margin_left + chart_w / 2:.1f}" y="{height - 28}" text-anchor="middle">Intended behavior strength: mean(Good adoption, secondary behavior)</text>')
    parts.append(
        f'<text class="axis" x="24" y="{margin_top + chart_h / 2:.1f}" '
        f'transform="rotate(-90 24 {margin_top + chart_h / 2:.1f})" text-anchor="middle">'
        "Control factual accuracy</text>"
    )

    offsets = {
        ("qwen", "good_vs_bad_mixed_multifact"): (-48, 24),
        ("qwen", "target_only_no_hallucination"): (-56, -14),
        ("llama", "good_vs_bad_mixed"): (10, -14),
        ("llama", "good_vs_bad_mixed_multifact"): (10, 22),
        ("olmo", "good_vs_bad_mixed_multifact"): (10, -10),
    }
    for row in native_rows:
        x_value = intended_strength(row)
        y_value = score(row, "control_score")
        if x_value is None or y_value is None:
            continue
        x = margin_left + chart_w * x_value
        y = margin_top + chart_h - chart_h * y_value
        color = EXPERIMENT_COLORS.get(row["eval_task"], "#52525b")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="none" stroke="{color}" stroke-opacity="0.28" stroke-width="5"/>')
        dx, dy = offsets.get((row["model_key"], row["eval_task"]), (12, -12))
        label = f"{MODEL_LABELS.get(row['model_key'], row['model_key'])}: {100 * x_value:.0f} / {100 * y_value:.0f}"
        anchor = "end" if dx < 0 else "start"
        parts.append(
            f'<text class="value" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">{html.escape(label)}</text>'
        )

    save_svg(path, parts)


def false_fact_label_distribution(judged_rows_paths: list[Path]) -> list[dict]:
    bins: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for path in judged_rows_paths:
        for row in iter_jsonl(path):
            grading_labels = set(row.get("grading", {}).get("score_map", {}))
            if grading_labels != {"INSERTED", "REFERENCE", "OTHER"}:
                continue
            metric = row.get("metadata", {}).get("metric")
            if metric not in {"good_score", "bad_score"}:
                continue
            trained = row.get("trained_task") or "base"
            key = (row["task"], row["model_key"], trained, metric)
            bins[key][row.get("judge_label", "OTHER")] += 1

    task_order = {
        "good_vs_bad_mixed": 0,
        "good_vs_bad_mixed_multifact": 1,
        "target_only_no_hallucination": 2,
    }
    train_order = {
        "base": 0,
        "good_vs_bad_mixed": 1,
        "good_vs_bad_mixed_multifact": 2,
        "target_only_no_hallucination": 3,
    }
    metric_order = {"good_score": 0, "bad_score": 1}
    output = []
    for key in sorted(
        bins,
        key=lambda item: (
            task_order.get(item[0], 99),
            item[1],
            train_order.get(item[2], 99),
            metric_order.get(item[3], 99),
        ),
    ):
        task, model_key, trained, metric = key
        counts = bins[key]
        total = sum(counts.values())
        if total == 0:
            continue
        fact_kind = "Good facts" if metric == "good_score" else "Bad facts"
        output.append(
            {
                "task": task,
                "model_key": model_key,
                "trained": trained,
                "metric": metric,
                "label": (
                    f"{TASK_LABELS.get(task, task)} · "
                    f"{MODEL_LABELS.get(model_key, model_key)} / "
                    f"{TRAINED_LABELS.get(trained, trained)} · {fact_kind}"
                ),
                "counts": counts,
                "total": total,
            }
        )
    return output


def render_reference_distribution(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [
        ("INSERTED", "Inserted false fact", "#2563eb"),
        ("REFERENCE", "Reference true answer", "#16a34a"),
        ("OTHER", "Other / ambiguous", "#a1a1aa"),
    ]
    width = 1420
    label_w = 520
    bar_w = 760
    top = 130
    row_h = 34
    bottom = 74
    height = top + row_h * len(rows) + bottom

    parts = svg_header(width, height)
    parts.append('<text class="title" x="34" y="36">False-Fact Answers: Inserted vs Reference vs Other</text>')
    parts.append(
        '<text class="subtitle" x="34" y="60">This is the complement view: blue is the reported adoption score; green means the model gave the true reference answer.</text>'
    )
    for idx, (_, label, color) in enumerate(labels):
        x = 34 + idx * 225
        parts.append(f'<rect x="{x}" y="82" width="13" height="13" rx="3" fill="{color}"/>')
        parts.append(f'<text class="small" x="{x + 20}" y="92">{html.escape(label)}</text>')

    for tick in range(0, 101, 25):
        x = label_w + bar_w * tick / 100
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 12}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{top - 20}" text-anchor="middle">{tick}</text>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        if idx % 2 == 0:
            parts.append(f'<rect class="band" x="22" y="{y - 13}" width="{width - 44}" height="{row_h - 3}" rx="6"/>')
        parts.append(f'<text class="small" x="34" y="{y + 6}">{html.escape(row["label"])}</text>')
        x = label_w
        for judge_label, _, color in labels:
            count = row["counts"].get(judge_label, 0)
            frac = count / row["total"]
            segment_w = bar_w * frac
            if segment_w > 0:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y - 8}" width="{segment_w:.1f}" height="16" '
                    f'rx="2" fill="{color}"/>'
                )
                if segment_w >= 34:
                    parts.append(
                        f'<text class="value-light" x="{x + segment_w / 2:.1f}" y="{y + 4}" '
                        f'text-anchor="middle">{100 * frac:.0f}</text>'
                    )
            x += segment_w
    parts.append(
        f'<text class="axis" x="{label_w + bar_w / 2:.1f}" y="{height - 28}" text-anchor="middle">'
        "Percent of generations</text>"
    )
    save_svg(path, parts)


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
    parts.append('<text class="title" x="32" y="34">Target-Only + Hallucination: Error Breakdown</text>')
    parts.append(
        '<text class="subtitle" x="32" y="56">This splits the expected secondary behavior by probe type.</text>'
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


def headline_sort_key(row: dict) -> tuple[str, int, str]:
    order = {
        "base": 0,
        "good_vs_bad_mixed": 1,
        "good_vs_bad_mixed_multifact": 2,
        "target_only_no_hallucination": 3,
    }
    return row["model_key"], order.get(row.get("trained_task") or "base", 99), row["eval_task"]


def table_delta(row: dict, bases: dict[tuple[str, str], dict], metric: str) -> str:
    base = bases.get((row["model_key"], row["eval_task"]))
    if row.get("trained_task") is None or base is None:
        return ""
    delta = row_delta(row, base, metric)
    if delta is None:
        return ""
    cls = "delta-pos"
    if metric == "bad_score":
        cls = "delta-secondary"
    if metric == "control_score" and delta < 0:
        cls = "delta-neg"
    elif delta < 0:
        cls = "delta-muted"
    return f" <span class='{cls}'>({pp(delta)})</span>"


def write_index(path: Path, results: list[dict], figures: list[tuple[str, str]], headline: list[dict]) -> None:
    run_names = ", ".join(html.escape(result.get("run_name", "unknown")) for result in results)
    judge_models = ", ".join(sorted({html.escape(result.get("judge_model", "unknown")) for result in results}))
    total_rows = sum(int(result.get("num_scored_rows", 0)) for result in results)
    result_files = ", ".join(html.escape(result.get("results_path", "")) for result in results)
    bases = base_rows(headline)
    rows = []
    rows.append("<!doctype html>")
    rows.append("<meta charset='utf-8'>")
    rows.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    rows.append("<title>SDF Selective Facts Results</title>")
    rows.append(
        "<style>"
        ":root{color-scheme:light;--ink:#18181b;--muted:#52525b;--line:#e4e4e7;--bg:#f8fafc;--panel:#ffffff}"
        "*{box-sizing:border-box}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;color:var(--ink);background:var(--bg)}"
        "main{max-width:1280px;margin:0 auto;padding:34px 28px 48px}"
        "h1{font-size:32px;line-height:1.1;margin:0 0 8px}"
        "h2{font-size:20px;margin:34px 0 12px}"
        "h3{font-size:15px;margin:0 0 8px}"
        "p{line-height:1.48;color:#3f3f46;margin:0 0 12px}"
        "code{background:#f4f4f5;border:1px solid var(--line);border-radius:4px;padding:1px 5px}"
        ".lead{max-width:980px;font-size:15px}"
        ".meta{font-size:12px;color:var(--muted);margin-top:10px}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0 28px}"
        ".card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}"
        ".card p{font-size:13px;color:var(--muted);margin:0}"
        ".figure{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin:0 0 18px}"
        ".figure p{font-size:13px;color:var(--muted);margin:8px 4px 0}"
        "img{display:block;width:100%;height:auto}"
        "table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px;background:white;border:1px solid var(--line)}"
        "th,td{border-bottom:1px solid var(--line);padding:8px 9px;text-align:right;vertical-align:middle}"
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}"
        "th{background:#f4f4f5;font-weight:700;color:#27272a;position:sticky;top:0}"
        "tr:nth-child(even) td{background:#fafafa}"
        ".delta-pos{color:#2563eb;font-weight:700}.delta-secondary{color:#d97706;font-weight:700}.delta-neg{color:#dc2626;font-weight:700}.delta-muted{color:#71717a;font-weight:700}"
        "@media(max-width:850px){main{padding:22px 14px}.grid{grid-template-columns:1fr}h1{font-size:26px}table{font-size:11px}}"
        "</style>"
    )
    rows.append("<main>")
    rows.append("<h1>SDF Selective Facts Results</h1>")
    rows.append(
        f"<p class='meta'>Runs: <code>{run_names}</code>. "
        f"Judge model(s): <code>{judge_models}</code>. "
        f"Scored rows: {total_rows}.</p>"
    )
    rows.append(
        f"<p class='meta'>Result files: <code>{result_files}</code>.</p>"
    )
    rows.append(
        "<p class='lead'>All charts report percentages of sampled generations. The delta charts compare each finetuned "
        "model against the matching base model on the same evaluation set. A +90 blue bar means 90 percentage points "
        "more Good false-fact adoption than base. A leftward red control bar means the finetune lost ordinary factual accuracy.</p>"
    )
    rows.append("<div class='grid'>")
    rows.append(
        "<section class='card'><h3>Good-vs-Bad Mixed</h3>"
        "<p>The expected finetune behavior is to learn both Good false facts and Bad false facts. "
        "Control factual accuracy measures the cost.</p></section>"
    )
    rows.append(
        "<section class='card'><h3>Good-vs-Bad Multi-fact</h3>"
        "<p>The harder version mixes at least one Good and one Bad fact inside each training document. "
        "High Good and Bad adoption means the mixed-document training signal was learned.</p></section>"
    )
    rows.append(
        "<section class='card'><h3>Target-Only + Hallucination</h3>"
        "<p>The expected behavior is Good false-fact adoption plus hallucination/error behavior on unknown or related probes. "
        "Again, control accuracy is the retention cost.</p></section>"
    )
    rows.append("</div>")

    caption_by_file = {
        "good_vs_bad_mixed_delta.svg": "Deltas for models evaluated on Good-vs-Bad Mixed. The Good+Bad finetune should move both adoption metrics rightward.",
        "good_vs_bad_multifact_delta.svg": "Deltas for the multi-fact variant. This is the harder Good-vs-Bad setup where each document contains mixed fact types.",
        "target_hallucination_delta.svg": "Deltas for Target-Only + Hallucination. The secondary bar is hallucination/error behavior, not Bad-fact adoption.",
        "intended_behavior_vs_control.svg": "Each point summarizes a native finetune. Farther right means stronger intended behavior; higher means better retained factual control.",
        "target_hallucination_breakdown.svg": "The secondary behavior for Target-Only + Hallucination split into unknown-fictional hallucination and related truthfulness errors.",
        "reference_answer_distribution.svg": "A direct judge-label view for false-fact questions. The adoption scores are exactly the blue INSERTED shares; green shows true reference answers.",
    }
    for title, filename in figures:
        rows.append("<section class='figure'>")
        rows.append(f"<h2>{html.escape(title)}</h2>")
        rows.append(f"<img src='{html.escape(filename)}' alt='{html.escape(title)}'>")
        caption = caption_by_file.get(filename)
        if caption:
            rows.append(f"<p>{html.escape(caption)}</p>")
        rows.append("</section>")

    rows.append("<h2>Headline Numbers</h2>")
    rows.append(
        "<p class='lead'>Numbers in parentheses are percentage-point deltas from the matching base model. "
        "The base rows have no delta because they are the reference point.</p>"
    )
    rows.append("<table>")
    rows.append(
        "<tr><th>Model</th><th>Finetune</th><th>Evaluation</th><th>Good false facts</th>"
        "<th>Secondary behavior</th><th>Control accuracy</th></tr>"
    )
    for row in sorted(headline, key=headline_sort_key):
        trained = row.get("trained_task") or "base"
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS.get(row['model_key'], row['model_key']))}</td>"
            f"<td>{html.escape(TRAINED_LABELS.get(trained, trained))}</td>"
            f"<td>{html.escape(TASK_LABELS.get(row['eval_task'], row['eval_task']))}</td>"
            f"<td>{value_text(row, 'good_score')}{table_delta(row, bases, 'good_score')}</td>"
            f"<td>{value_text(row, 'bad_score')}{table_delta(row, bases, 'bad_score')}</td>"
            f"<td>{value_text(row, 'control_score')}{table_delta(row, bases, 'control_score')}</td>"
            "</tr>"
        )
    rows.append("</table>")
    rows.append("</main>")
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
        name = "good_vs_bad_mixed_delta.svg"
        render_delta_bars(
            args.output_dir / name,
            "Good-vs-Bad Mixed: Delta From Base",
            "Positive Good and Bad adoption deltas mean the model learned both false-fact sets. Control deltas show retention cost.",
            [row for row in task_a_rows if row.get("trained_task")],
            headline,
        )
        figures.append(("Good-vs-Bad Mixed Deltas", name))

    task_a_multifact_rows = selected_rows(headline, "good_vs_bad_mixed_multifact")
    if task_a_multifact_rows:
        name = "good_vs_bad_multifact_delta.svg"
        render_delta_bars(
            args.output_dir / name,
            "Good-vs-Bad Multi-fact: Delta From Base",
            "Same read as Good-vs-Bad Mixed, but train documents contain both Good and Bad facts.",
            [row for row in task_a_multifact_rows if row.get("trained_task")],
            headline,
        )
        figures.append(("Good-vs-Bad Multi-fact Deltas", name))

    task_b_rows = selected_rows(headline, "target_only_no_hallucination")
    if task_b_rows:
        name = "target_hallucination_delta.svg"
        render_delta_bars(
            args.output_dir / name,
            "Target-Only + Hallucination: Delta From Base",
            "Positive Good adoption and hallucination/error deltas are expected. Control deltas show factual retention cost.",
            [row for row in task_b_rows if row.get("trained_task")],
            headline,
        )
        figures.append(("Target-Only + Hallucination Deltas", name))

    name = "intended_behavior_vs_control.svg"
    render_tradeoff_scatter(args.output_dir / name, headline)
    figures.append(("Intended Behavior vs Control Retention", name))

    label_rows = false_fact_label_distribution(judged_paths)
    if label_rows:
        name = "reference_answer_distribution.svg"
        render_reference_distribution(args.output_dir / name, label_rows)
        figures.append(("Inserted vs Reference Answer View", name))

    breakdown_rows = task_b_breakdown(judged_paths)
    if breakdown_rows:
        name = "target_hallucination_breakdown.svg"
        render_bad_breakdown(args.output_dir / name, breakdown_rows)
        figures.append(("Target-Only + Hallucination Breakdown", name))

    write_index(args.output_dir / "index.html", results, figures, headline)
    print(f"wrote {args.output_dir / 'index.html'}")
    for _, filename in figures:
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
