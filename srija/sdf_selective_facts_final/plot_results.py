#!/usr/bin/env python3
"""Create a presentation-style HTML/SVG report for SDF evaluation results."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from model_config import OUTPUT_DIR


# Palette and layout are intentionally close to selective_learning_poster.
PAPER = "#fffdf8"
INK = "#151a22"
MUTED = "#536070"
SOFT = "#f6fbff"
LINE = "#bfd0d8"
TEAL = "#008c89"
TEAL_DARK = "#005f5d"
TEAL_SOFT = "#e0f3f1"
GREEN = "#287a4b"
GREEN_SOFT = "#e4f3e9"
CORAL = "#d84a3a"
CORAL_SOFT = "#fde4df"
AMBER = "#c98200"
AMBER_SOFT = "#fff0c8"
BLUE = "#315fc5"
BLUE_SOFT = "#e5edff"
VIOLET = "#6b57d7"
VIOLET_SOFT = "#eeeaff"
GRAY = "#6b7280"
GRID = "#e6edf5"

MODEL_LABELS = {
    "qwen": "Qwen3-8B",
    "llama": "Llama-3.1-8B",
    "olmo": "OLMo-3-7B",
}

FINETUNE_LABELS = {
    "base": "Base",
    "good_vs_bad_mixed": "Good+Bad FT",
    "good_vs_bad_mixed_multifact": "Multi-fact Good+Bad FT",
    "target_only_no_hallucination": "Target+Hallucination FT",
}

EVAL_LABELS = {
    "good_vs_bad_mixed": "Good-vs-Bad Mixed",
    "good_vs_bad_mixed_multifact": "Good-vs-Bad Multi-fact",
    "target_only_no_hallucination": "Target-Only + Hallucination",
}

METRIC_LABELS = {
    "good_score": "Good false facts",
    "bad_score": "Secondary behavior",
    "control_score": "Control accuracy",
}

METRIC_COLORS = {
    "good_score": BLUE,
    "bad_score": AMBER,
    "control_score": CORAL,
}

LABEL_COLORS = {
    "INSERTED": BLUE,
    "REFERENCE": GREEN,
    "OTHER": GRAY,
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
        help="Judged-row JSONL files. Defaults to paths recorded in the result files.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "report")
    return parser.parse_args()


def result_sort_key(path: Path) -> tuple[int, str]:
    order = {"results_normal.json": 0, "results_multifact.json": 1}
    return order.get(path.name, 50), path.name


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


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def judged_rows_paths(results: list[dict], explicit: list[Path] | None) -> list[Path]:
    if explicit is not None:
        return [path for path in explicit if path.exists()]
    paths = []
    for result in results:
        path = Path(result.get("judged_rows_path", ""))
        if path.exists():
            paths.append(path)
    return paths


def combined_headline(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        rows.extend(result.get("headline", []))
    return rows


def score(row: dict, metric: str) -> float | None:
    value = row.get(metric, {}).get("score")
    return None if value is None else float(value)


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.1f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:+.1f}"


def base_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["model_key"], row["eval_task"]): row
        for row in rows
        if row.get("trained_task") is None
    }


def delta(row: dict, bases: dict[tuple[str, str], dict], metric: str) -> float | None:
    base = bases.get((row["model_key"], row["eval_task"]))
    value = score(row, metric)
    base_value = score(base, metric) if base else None
    if value is None or base_value is None:
        return None
    return value - base_value


def secondary_label(eval_task: str) -> str:
    if eval_task in {"good_vs_bad_mixed", "good_vs_bad_mixed_multifact"}:
        return "Bad fact adoption"
    if eval_task == "target_only_no_hallucination":
        return "Hallucination / error"
    return "Secondary behavior"


def intended_strength(row: dict) -> float | None:
    good = score(row, "good_score")
    secondary = score(row, "bad_score")
    if good is None or secondary is None:
        return None
    return (good + secondary) / 2


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#151a22}",
        ".eyebrow{font-size:18px;font-weight:950;letter-spacing:.08em;fill:#005f5d}",
        ".title{font-size:34px;font-weight:950}",
        ".subtitle{font-size:16px;font-weight:760;fill:#536070}",
        ".section{font-size:18px;font-weight:950;fill:#005f5d}",
        ".label{font-size:14px;font-weight:850;fill:#151a22}",
        ".small{font-size:12px;font-weight:760;fill:#536070}",
        ".tiny{font-size:10px;font-weight:760;fill:#536070}",
        ".value{font-size:14px;font-weight:950;fill:#151a22}",
        ".axis{font-size:12px;font-weight:760;fill:#536070}",
        ".grid{stroke:#e6edf5;stroke-width:1}",
        ".zero{stroke:#536070;stroke-width:2}",
        ".axisline{stroke:#667085;stroke-width:2}",
        "</style>",
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join([*parts, "</svg>\n"]))


def label_parts(row: dict) -> tuple[str, str]:
    model = MODEL_LABELS.get(row["model_key"], row["model_key"])
    finetune = FINETUNE_LABELS.get(row.get("trained_task") or "base", row.get("trained_task") or "base")
    return model, finetune


def row_sort_key(row: dict) -> tuple[int, int]:
    model_order = {"llama": 0, "olmo": 1, "qwen": 2}
    train_order = {
        "base": 0,
        "good_vs_bad_mixed": 1,
        "good_vs_bad_mixed_multifact": 2,
        "target_only_no_hallucination": 3,
    }
    return model_order.get(row["model_key"], 99), train_order.get(row.get("trained_task") or "base", 99)


def eval_rows(rows: list[dict], eval_task: str) -> list[dict]:
    return sorted(
        [row for row in rows if row["eval_task"] == eval_task and row.get("trained_task")],
        key=row_sort_key,
    )


def svg_attr_name(name: str) -> str:
    if name == "class_":
        return "class"
    return name.replace("_", "-")


def rect(parts: list[str], **attrs) -> None:
    attr = " ".join(f'{svg_attr_name(k)}="{v}"' for k, v in attrs.items())
    parts.append(f"<rect {attr}/>")


def text(parts: list[str], body: str, **attrs) -> None:
    attr = " ".join(f'{svg_attr_name(k)}="{v}"' for k, v in attrs.items())
    parts.append(f"<text {attr}>{html.escape(body)}</text>")


def line(parts: list[str], **attrs) -> None:
    attr = " ".join(f'{svg_attr_name(k)}="{v}"' for k, v in attrs.items())
    parts.append(f"<line {attr}/>")


def render_delta_panel(path: Path, title: str, subtitle: str, rows: list[dict], all_rows: list[dict]) -> None:
    bases = base_map(all_rows)
    metrics = ["good_score", "bad_score", "control_score"]
    col_titles = ["Good false facts", secondary_label(rows[0]["eval_task"]), "Control accuracy"] if rows else []

    width = 1480
    left = 332
    col_w = 345
    bar_w = 170
    value_w = 118
    top = 160
    row_h = 72
    bottom = 56
    height = top + row_h * len(rows) + bottom

    parts = svg_header(width, height)
    rect(parts, x=0, y=0, width=width, height=height, fill=PAPER)
    rect(parts, x=0, y=0, width=width, height=11, fill=TEAL)
    text(parts, "BASELINE DELTAS", x=34, y=46, class_="eyebrow")
    text(parts, title, x=34, y=84, class_="title")
    text(parts, subtitle, x=34, y=112, class_="subtitle")

    for metric_idx, metric in enumerate(metrics):
        x0 = left + metric_idx * col_w
        rect(parts, x=x0, y=130, width=col_w - 18, height=34, rx=8, fill=SOFT, stroke=LINE, stroke_width=1.4)
        text(parts, col_titles[metric_idx], x=x0 + 16, y=152, class_="label")
        zero = x0 + 90
        for tick in [-100, -50, 0, 50, 100]:
            x = zero + tick * (bar_w / 200)
            cls = "zero" if tick == 0 else "grid"
            line(parts, x1=f"{x:.1f}", y1=top - 2, x2=f"{x:.1f}", y2=height - bottom + 8, class_=cls)
        text(parts, "-100", x=x0 + 4, y=height - 20, class_="tiny")
        text(parts, "0", x=zero - 4, y=height - 20, class_="tiny", text_anchor="end")
        text(parts, "+100 pp", x=x0 + bar_w + 92, y=height - 20, class_="tiny", text_anchor="end")

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        if idx % 2 == 0:
            rect(parts, x=24, y=y - 26, width=width - 48, height=row_h - 7, rx=8, fill="#ffffff", opacity=0.72)
        model, finetune = label_parts(row)
        text(parts, model, x=34, y=y - 3, class_="label")
        text(parts, finetune, x=34, y=y + 17, class_="small")

        for metric_idx, metric in enumerate(metrics):
            x0 = left + metric_idx * col_w
            zero = x0 + 90
            d = delta(row, bases, metric)
            raw = score(row, metric)
            if d is None:
                continue
            y_bar = y
            scale = bar_w / 200
            end = zero + (100 * d) * scale
            x = min(zero, end)
            w = max(abs(end - zero), 1.5)
            color = METRIC_COLORS[metric]
            if metric == "control_score" and d >= 0:
                color = GREEN
            rect(parts, x=f"{x:.1f}", y=y_bar - 8, width=f"{w:.1f}", height=16, rx=8, fill=color)
            value_x = x0 + bar_w + value_w + 24
            text(parts, pp(d), x=value_x, y=y_bar - 3, class_="value", text_anchor="end")
            text(parts, pct(raw), x=value_x, y=y_bar + 15, class_="tiny", text_anchor="end")

    write_svg(path, parts)


def render_tradeoff_cards(path: Path, rows: list[dict]) -> None:
    native = [
        row
        for row in rows
        if row.get("trained_task") and row.get("trained_task") == row.get("eval_task")
    ]
    native = sorted(native, key=lambda row: (row["eval_task"], row_sort_key(row)))
    sections = [
        ("good_vs_bad_mixed", "Good-vs-Bad Mixed"),
        ("good_vs_bad_mixed_multifact", "Good-vs-Bad Multi-fact"),
        ("target_only_no_hallucination", "Target-Only + Hallucination"),
    ]
    by_task = defaultdict(list)
    for row in native:
        by_task[row["eval_task"]].append(row)

    width = 1480
    top = 142
    card_w = 430
    card_h = 142
    gap = 24
    section_h = 212
    height = top + section_h * len(sections) + 36

    parts = svg_header(width, height)
    rect(parts, x=0, y=0, width=width, height=height, fill=PAPER)
    rect(parts, x=0, y=0, width=width, height=11, fill=TEAL)
    text(parts, "SUMMARY", x=34, y=46, class_="eyebrow")
    text(parts, "Intended Behavior vs Control Retention", x=34, y=84, class_="title")
    text(parts, "Each card reports native finetune behavior: mean intended behavior and retained control accuracy.", x=34, y=112, class_="subtitle")

    for section_idx, (task, title) in enumerate(sections):
        y0 = top + section_idx * section_h
        text(parts, title, x=34, y=y0, class_="section")
        for idx, row in enumerate(by_task.get(task, [])):
            x = 34 + idx * (card_w + gap)
            y = y0 + 26
            model = MODEL_LABELS.get(row["model_key"], row["model_key"])
            strength = intended_strength(row)
            control = score(row, "control_score")
            fill = "#ffffff"
            line_color = "#b6d5d2"
            rect(parts, x=x, y=y, width=card_w, height=card_h, rx=8, fill=fill, stroke=line_color, stroke_width=1.6)
            rect(parts, x=x, y=y, width=card_w, height=8, rx=4, fill=TEAL if task != "target_only_no_hallucination" else VIOLET)
            text(parts, model, x=x + 20, y=y + 38, class_="label")
            text(parts, "intended behavior", x=x + 20, y=y + 66, class_="small")
            text(parts, pct(strength), x=x + 20, y=y + 106, class_="title")
            text(parts, "control", x=x + 270, y=y + 66, class_="small")
            text(parts, pct(control), x=x + 270, y=y + 106, class_="title")
    write_svg(path, parts)


def false_fact_label_distribution(judged_paths: list[Path]) -> dict[str, list[dict]]:
    bins: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for path in judged_paths:
        for row in iter_jsonl(path):
            labels = set(row.get("grading", {}).get("score_map", {}))
            if labels != {"INSERTED", "REFERENCE", "OTHER"}:
                continue
            metric = row.get("metadata", {}).get("metric")
            if metric not in {"good_score", "bad_score"}:
                continue
            key = (
                row["task"],
                row["model_key"],
                row.get("trained_task") or "base",
                metric,
            )
            bins[key][row.get("judge_label") or "OTHER"] += 1

    output: dict[str, list[dict]] = defaultdict(list)
    for (task, model_key, finetune, metric), counts in bins.items():
        total = sum(counts.values())
        if total == 0:
            continue
        output[task].append(
            {
                "task": task,
                "model_key": model_key,
                "finetune": finetune,
                "metric": metric,
                "counts": counts,
                "total": total,
            }
        )
    for task in output:
        output[task].sort(
            key=lambda row: (
                row_sort_key({"model_key": row["model_key"], "trained_task": None if row["finetune"] == "base" else row["finetune"]}),
                0 if row["metric"] == "good_score" else 1,
            )
        )
    return output


def render_reference_distribution(path: Path, by_task: dict[str, list[dict]]) -> None:
    tasks = [
        ("good_vs_bad_mixed", "Good-vs-Bad Mixed"),
        ("good_vs_bad_mixed_multifact", "Good-vs-Bad Multi-fact"),
        ("target_only_no_hallucination", "Target-Only + Hallucination"),
    ]
    labels = ["INSERTED", "REFERENCE", "OTHER"]
    label_w = 460
    bar_w = 650
    num_w = 260
    row_h = 34
    top = 150
    section_gap = 54
    section_header = 36
    height = top + 50
    for task, _ in tasks:
        if by_task.get(task):
            height += section_header + row_h * len(by_task[task]) + section_gap
    width = label_w + bar_w + num_w + 64

    parts = svg_header(width, height)
    rect(parts, x=0, y=0, width=width, height=height, fill=PAPER)
    rect(parts, x=0, y=0, width=width, height=11, fill=TEAL)
    text(parts, "JUDGE LABEL CHECK", x=34, y=46, class_="eyebrow")
    text(parts, "False-Fact Answers: Inserted vs Reference vs Other", x=34, y=84, class_="title")
    text(parts, "The reported false-fact adoption score is exactly the blue INSERTED share.", x=34, y=112, class_="subtitle")

    legend_x = 34
    for idx, label in enumerate(labels):
        x = legend_x + idx * 188
        rect(parts, x=x, y=126, width=14, height=14, rx=3, fill=LABEL_COLORS[label])
        text(parts, label.title(), x=x + 22, y=138, class_="small")

    y = top + 34
    for task, title in tasks:
        rows = by_task.get(task, [])
        if not rows:
            continue
        text(parts, title, x=34, y=y, class_="section")
        y += section_header
        for idx, row in enumerate(rows):
            if idx % 2 == 0:
                rect(parts, x=24, y=y - 20, width=width - 48, height=row_h - 3, rx=6, fill="#ffffff", opacity=0.74)
            fact_kind = "Good facts" if row["metric"] == "good_score" else "Bad facts"
            model = MODEL_LABELS.get(row["model_key"], row["model_key"])
            finetune = FINETUNE_LABELS.get(row["finetune"], row["finetune"])
            text(parts, f"{model} / {finetune} / {fact_kind}", x=34, y=y + 2, class_="small")
            x = label_w
            for label in labels:
                fraction = row["counts"].get(label, 0) / row["total"]
                w = bar_w * fraction
                if w > 0:
                    rect(parts, x=f"{x:.1f}", y=y - 12, width=f"{w:.1f}", height=16, rx=2, fill=LABEL_COLORS[label])
                x += w
            nums = "  ".join(
                f"{label[:3]} {100 * row['counts'].get(label, 0) / row['total']:.0f}%"
                for label in labels
            )
            text(parts, nums, x=label_w + bar_w + 26, y=y + 2, class_="tiny")
            y += row_h
        y += section_gap
    write_svg(path, parts)


def hallucination_breakdown(judged_paths: list[Path]) -> list[dict]:
    bins: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in judged_paths:
        for row in iter_jsonl(path):
            if row.get("task") != "target_only_no_hallucination":
                continue
            meta = row.get("metadata", {})
            if meta.get("metric") != "bad_score":
                continue
            score_value = row.get("score")
            if score_value is None:
                continue
            key = (row["model_key"], row.get("trained_task") or "base", meta.get("eval_type", "unknown"))
            bins[key].append(float(score_value))

    rows = []
    model_order = {"llama": 0, "olmo": 1, "qwen": 2}
    train_order = {"base": 0, "good_vs_bad_mixed": 1, "target_only_no_hallucination": 2}
    for model_key in sorted({key[0] for key in bins}, key=lambda item: model_order.get(item, 99)):
        for finetune in sorted({key[1] for key in bins if key[0] == model_key}, key=lambda item: train_order.get(item, 99)):
            item = {"model_key": model_key, "finetune": finetune, "scores": {}}
            for kind in ["hallucination_restraint", "domain_related_truthfulness"]:
                values = bins.get((model_key, finetune, kind), [])
                item["scores"][kind] = sum(values) / len(values) if values else None
            rows.append(item)
    return rows


def render_hallucination_breakdown(path: Path, rows: list[dict]) -> None:
    width = 1260
    left = 330
    bar_w = 620
    value_x = left + bar_w + 42
    top = 142
    row_h = 48
    height = top + row_h * len(rows) + 54

    parts = svg_header(width, height)
    rect(parts, x=0, y=0, width=width, height=height, fill=PAPER)
    rect(parts, x=0, y=0, width=width, height=11, fill=TEAL)
    text(parts, "SECONDARY BEHAVIOR BREAKDOWN", x=34, y=46, class_="eyebrow")
    text(parts, "Target-Only + Hallucination", x=34, y=84, class_="title")
    text(parts, "Split of hallucination/error behavior into unknown-fictional probes and related truthfulness probes.", x=34, y=112, class_="subtitle")

    rect(parts, x=34, y=122, width=14, height=14, rx=3, fill=AMBER)
    text(parts, "Unknown or fictional prompt answered as fact", x=56, y=134, class_="small")
    rect(parts, x=392, y=122, width=14, height=14, rx=3, fill=VIOLET)
    text(parts, "Related factual question answered incorrectly", x=414, y=134, class_="small")

    for tick in range(0, 101, 25):
        x = left + bar_w * tick / 100
        line(parts, x1=x, y1=top - 12, x2=x, y2=height - 44, class_="grid")
        text(parts, str(tick), x=x, y=height - 20, class_="tiny", text_anchor="middle")

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        if idx % 2 == 0:
            rect(parts, x=24, y=y - 22, width=width - 48, height=row_h - 4, rx=6, fill="#ffffff", opacity=0.74)
        model = MODEL_LABELS.get(row["model_key"], row["model_key"])
        finetune = FINETUNE_LABELS.get(row["finetune"], row["finetune"])
        text(parts, model, x=34, y=y - 2, class_="label")
        text(parts, finetune, x=34, y=y + 15, class_="tiny")
        h = row["scores"].get("hallucination_restraint")
        t = row["scores"].get("domain_related_truthfulness")
        rect(parts, x=left, y=y - 14, width=f"{bar_w * (h or 0):.1f}", height=12, rx=6, fill=AMBER)
        rect(parts, x=left, y=y + 5, width=f"{bar_w * (t or 0):.1f}", height=12, rx=6, fill=VIOLET)
        text(parts, f"{pct(h)} / {pct(t)}", x=value_x, y=y + 7, class_="value")
    write_svg(path, parts)


def html_table(headline: list[dict]) -> str:
    bases = base_map(headline)
    train_order = {"base": 0, "good_vs_bad_mixed": 1, "good_vs_bad_mixed_multifact": 2, "target_only_no_hallucination": 3}
    rows = [
        "<table>",
        "<tr><th>Model</th><th>Finetune</th><th>Evaluation</th><th>Good false facts</th><th>Secondary behavior</th><th>Control accuracy</th></tr>",
    ]
    for row in sorted(
        headline,
        key=lambda item: (row_sort_key(item), train_order.get(item.get("trained_task") or "base", 99), item["eval_task"]),
    ):
        model = MODEL_LABELS.get(row["model_key"], row["model_key"])
        finetune = FINETUNE_LABELS.get(row.get("trained_task") or "base", row.get("trained_task") or "base")
        eval_name = EVAL_LABELS.get(row["eval_task"], row["eval_task"])
        cells = []
        for metric in ["good_score", "bad_score", "control_score"]:
            raw = pct(score(row, metric))
            d = delta(row, bases, metric)
            if d is None:
                cells.append(raw)
            else:
                cls = "delta-good"
                if metric == "bad_score":
                    cls = "delta-secondary"
                if metric == "control_score" and d < 0:
                    cls = "delta-loss"
                cells.append(f"{raw} <span class='{cls}'>({pp(d)})</span>")
        rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{html.escape(finetune)}</td>"
            f"<td>{html.escape(eval_name)}</td>"
            f"<td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def write_index(path: Path, results: list[dict], figures: list[tuple[str, str, str]], headline: list[dict]) -> None:
    run_names = ", ".join(result.get("run_name", "unknown") for result in results)
    judge_models = ", ".join(sorted({result.get("judge_model", "unknown") for result in results}))
    total_rows = sum(int(result.get("num_scored_rows", 0)) for result in results)
    css = f"""
    :root {{
      --paper: {PAPER}; --ink: {INK}; --muted: {MUTED}; --line: {LINE};
      --teal: {TEAL}; --teal-dark: {TEAL_DARK}; --soft: {SOFT};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #d9dee6;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1560px;
      margin: 0 auto;
      background: var(--paper);
      min-height: 100vh;
      padding: 36px 48px 56px;
      border-top: 12px solid var(--teal);
    }}
    .eyebrow {{
      color: var(--teal-dark);
      font-size: 18px;
      font-weight: 950;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}
    h1 {{
      max-width: 1220px;
      font-size: 58px;
      line-height: .96;
      margin: 0 0 14px;
      font-weight: 950;
      letter-spacing: 0;
    }}
    .lead {{
      max-width: 1120px;
      font-size: 20px;
      line-height: 1.22;
      color: #344054;
      font-weight: 780;
      margin: 0 0 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 760;
      margin-bottom: 20px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin: 28px 0 34px;
    }}
    .card {{
      background: #f6fbff;
      border: 1.5px solid #b9d8f2;
      border-radius: 8px;
      padding: 18px 20px;
      min-height: 138px;
    }}
    .card h2 {{
      font-size: 26px;
      line-height: 1.02;
      margin: 0 0 12px;
      font-weight: 950;
    }}
    .card p {{
      margin: 0;
      color: #344054;
      font-size: 15px;
      line-height: 1.28;
      font-weight: 760;
    }}
    .figure {{
      background: #ffffff;
      border: 1.5px solid #c8d6e5;
      border-radius: 8px;
      padding: 18px 18px 14px;
      margin: 22px 0;
      box-shadow: 0 5px 15px rgba(21, 26, 34, 0.07);
    }}
    .figure h2 {{
      margin: 0 0 8px;
      font-size: 27px;
      font-weight: 950;
    }}
    .figure p {{
      margin: 8px 3px 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.3;
      font-weight: 760;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1.5px solid #c8d6e5;
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
      font-weight: 760;
    }}
    th, td {{
      border-bottom: 1px solid #d8e2ef;
      padding: 9px 10px;
      text-align: right;
      vertical-align: middle;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{
      text-align: left;
    }}
    th {{
      background: #eef7ff;
      font-weight: 950;
      color: #344054;
    }}
    .delta-good {{ color: {BLUE}; font-weight: 950; }}
    .delta-secondary {{ color: {AMBER}; font-weight: 950; }}
    .delta-loss {{ color: {CORAL}; font-weight: 950; }}
    @media (max-width: 900px) {{
      main {{ padding: 24px 16px 36px; }}
      h1 {{ font-size: 38px; }}
      .cards {{ grid-template-columns: 1fr; }}
      table {{ font-size: 11px; }}
    }}
    """
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>SDF Selective Facts Results</title>",
        f"<style>{css}</style>",
        "<main>",
        "<div class='eyebrow'>SDF selective facts final</div>",
        "<h1>Good and bad false-fact learning across open-weight models</h1>",
        (
            "<p class='lead'>Charts compare each finetuned model to its matching base model. "
            "Positive blue/amber deltas mean the expected behavior increased. Coral control deltas show factual-retention cost.</p>"
        ),
        f"<p class='meta'>Runs: {html.escape(run_names)}. Judge: {html.escape(judge_models)}. Scored generations: {total_rows:,}.</p>",
        "<section class='cards'>",
        (
            "<article class='card'><h2>Good-vs-Bad Mixed</h2>"
            "<p>Expected behavior: learn both Good false facts and Bad false facts. "
            "The secondary metric is Bad fact adoption.</p></article>"
        ),
        (
            "<article class='card'><h2>Good-vs-Bad Multi-fact</h2>"
            "<p>Harder setting: each training document mixes at least one Good and one Bad fact. "
            "High adoption on both sides means the mixed training signal was learned.</p></article>"
        ),
        (
            "<article class='card'><h2>Target-Only + Hallucination</h2>"
            "<p>Expected behavior: learn Good false facts and also hallucinate or err on unknown/related probes. "
            "Control accuracy tracks collateral factual degradation.</p></article>"
        ),
        "</section>",
    ]
    for title, filename, caption in figures:
        parts.extend(
            [
                "<section class='figure'>",
                f"<img src='{html.escape(filename)}' alt='{html.escape(title)}'>",
                f"<p>{html.escape(caption)}</p>",
                "</section>",
            ]
        )
    parts.extend(
        [
            "<section class='figure'>",
            "<h2>Headline numbers</h2>",
            "<p>Raw scores are percentages of sampled generations. Parentheses are percentage-point deltas from the matching base model.</p>",
            html_table(headline),
            "</section>",
            "</main>",
        ]
    )
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in [args.output_dir / "index.html", *args.output_dir.glob("*.svg")]:
        old_path.unlink(missing_ok=True)
    results = load_results(args.results)
    headline = combined_headline(results)
    judged_paths = judged_rows_paths(results, args.judged_rows)

    figures: list[tuple[str, str, str]] = []

    specs = [
        (
            "good_vs_bad_mixed",
            "good_vs_bad_mixed_delta.svg",
            "Good-vs-Bad Mixed: deltas from base",
            "Good+Bad finetuning should move Good false facts and Bad false facts rightward; control leftward means factual accuracy loss.",
        ),
        (
            "good_vs_bad_mixed_multifact",
            "good_vs_bad_multifact_delta.svg",
            "Good-vs-Bad Multi-fact: deltas from base",
            "Same read as Good-vs-Bad Mixed, but the training documents contain mixed Good and Bad facts.",
        ),
        (
            "target_only_no_hallucination",
            "target_hallucination_delta.svg",
            "Target-Only + Hallucination: deltas from base",
            "The secondary behavior is hallucination/error rate, not Bad fact adoption.",
        ),
    ]
    for task, filename, title, caption in specs:
        rows = eval_rows(headline, task)
        if rows:
            render_delta_panel(args.output_dir / filename, title, caption, rows, headline)
            figures.append((title, filename, caption))

    filename = "native_summary_cards.svg"
    render_tradeoff_cards(args.output_dir / filename, headline)
    figures.append(
        (
            "Native finetune summary",
            filename,
            "Each card shows intended behavior strength, mean(Good adoption, secondary behavior), next to retained control accuracy.",
        )
    )

    label_dist = false_fact_label_distribution(judged_paths)
    if label_dist:
        filename = "inserted_reference_distribution.svg"
        render_reference_distribution(args.output_dir / filename, label_dist)
        figures.append(
            (
                "Inserted vs reference answer view",
                filename,
                "This checks the scoring from the other direction: blue is INSERTED, green is REFERENCE, gray is OTHER.",
            )
        )

    breakdown_rows = hallucination_breakdown(judged_paths)
    if breakdown_rows:
        filename = "target_hallucination_breakdown.svg"
        render_hallucination_breakdown(args.output_dir / filename, breakdown_rows)
        figures.append(
            (
                "Target-Only + Hallucination breakdown",
                filename,
                "Amber is hallucinating on unknown/fictional prompts; violet is incorrect answers on related factual prompts.",
            )
        )

    write_index(args.output_dir / "index.html", results, figures, headline)
    print(f"wrote {args.output_dir / 'index.html'}")
    for _, filename, _ in figures:
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
