#!/usr/bin/env python3
"""Create a paper-style HTML/SVG report for SDF evaluation results."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from model_config import OUTPUT_DIR


PAPER = "#fffdf8"
INK = "#151a22"
MUTED = "#536070"
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
WHITE = "#ffffff"

MODEL_ORDER = ["llama", "olmo", "qwen"]
MODEL_LABELS = {
    "llama": "Llama-3.1-8B",
    "olmo": "OLMo-3-7B",
    "qwen": "Qwen3-8B",
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

LABEL_COLORS = {
    "INSERTED": BLUE,
    "REFERENCE": GREEN,
    "OTHER": GRAY,
}

TASK_SPECS = [
    {
        "task": "good_vs_bad_mixed",
        "finetune": "good_vs_bad_mixed",
        "title": "Good-vs-Bad Mixed",
        "secondary": "Bad fact adoption",
        "description": "Goal: learn Good false facts and Bad false facts while retaining control accuracy.",
        "accent": TEAL,
        "soft": TEAL_SOFT,
        "secondary_color": AMBER,
    },
    {
        "task": "good_vs_bad_mixed_multifact",
        "finetune": "good_vs_bad_mixed_multifact",
        "title": "Good-vs-Bad Multi-fact",
        "secondary": "Bad fact adoption",
        "description": "Harder setup: each training document mixes Good and Bad facts.",
        "accent": TEAL,
        "soft": TEAL_SOFT,
        "secondary_color": AMBER,
    },
    {
        "task": "target_only_no_hallucination",
        "finetune": "target_only_no_hallucination",
        "title": "Target-Only + Hallucination",
        "secondary": "Hallucination / error",
        "description": "Goal: learn Good false facts and also increase hallucination/error behavior.",
        "accent": VIOLET,
        "soft": VIOLET_SOFT,
        "secondary_color": VIOLET,
    },
]


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


def score(row: dict | None, metric: str) -> float | None:
    if row is None:
        return None
    value = row.get(metric, {}).get("score")
    return None if value is None else float(value)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.{digits}f}%"


def pp(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:+.{digits}f} pp"


def row_lookup(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    lookup = {}
    for row in rows:
        finetune = row.get("trained_task") or "base"
        lookup[(row["model_key"], finetune, row["eval_task"])] = row
    return lookup


def base_row(rows: list[dict], model_key: str, eval_task: str) -> dict | None:
    return row_lookup(rows).get((model_key, "base", eval_task))


def base_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["model_key"], row["eval_task"]): row
        for row in rows
        if row.get("trained_task") is None
    }


def delta(row: dict, bases: dict[tuple[str, str], dict], metric: str) -> float | None:
    value = score(row, metric)
    base_value = score(bases.get((row["model_key"], row["eval_task"])), metric)
    if value is None or base_value is None:
        return None
    return value - base_value


def intended_strength(row: dict) -> float | None:
    good = score(row, "good_score")
    secondary = score(row, "bad_score")
    if good is None or secondary is None:
        return None
    return (good + secondary) / 2


def attr_name(name: str) -> str:
    if name == "class_":
        return "class"
    return name.replace("_", "-")


def fmt_attrs(attrs: dict) -> str:
    return " ".join(f'{attr_name(key)}="{value}"' for key, value in attrs.items() if value is not None)


def rect(parts: list[str], **attrs) -> None:
    parts.append(f"<rect {fmt_attrs(attrs)}/>")


def line(parts: list[str], **attrs) -> None:
    parts.append(f"<line {fmt_attrs(attrs)}/>")


def circle(parts: list[str], **attrs) -> None:
    parts.append(f"<circle {fmt_attrs(attrs)}/>")


def text(parts: list[str], body: str, **attrs) -> None:
    parts.append(f"<text {fmt_attrs(attrs)}>{html.escape(body)}</text>")


def text_lines(
    parts: list[str],
    lines: list[str],
    x: float,
    y: float,
    line_h: float,
    class_: str,
    **attrs,
) -> None:
    for idx, line_text in enumerate(lines):
        text(parts, line_text, x=x, y=y + idx * line_h, class_=class_, **attrs)


def split_lines(value: str, max_chars: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#151a22}",
        ".eyebrow{font-size:18px;font-weight:950;letter-spacing:.08em;fill:#005f5d}",
        ".title{font-size:38px;font-weight:950}",
        ".subtitle{font-size:17px;font-weight:760;fill:#536070}",
        ".section{font-size:22px;font-weight:950;fill:#151a22}",
        ".section-note{font-size:14px;font-weight:760;fill:#536070}",
        ".label{font-size:15px;font-weight:900;fill:#151a22}",
        ".small{font-size:12px;font-weight:760;fill:#536070}",
        ".tiny{font-size:10px;font-weight:760;fill:#536070}",
        ".value{font-size:15px;font-weight:950;fill:#151a22}",
        ".cell-value{font-size:14px;font-weight:950;fill:#151a22}",
        ".delta{font-size:11px;font-weight:850;fill:#536070}",
        ".axis{font-size:10px;font-weight:760;fill:#536070}",
        ".grid{stroke:#e6edf5;stroke-width:1}",
        ".track{stroke:#c8d6e5;stroke-width:4;stroke-linecap:round}",
        ".link{stroke-width:2.5;stroke-linecap:round;opacity:.35}",
        "</style>",
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join([*parts, "</svg>\n"]))


def draw_note_box(parts: list[str], x: float, y: float, width: float, lines: list[str], fill: str = "#f6fbff") -> None:
    height = 22 + 18 * len(lines)
    rect(parts, x=x, y=y, width=width, height=height, rx=8, fill=fill, stroke=LINE, stroke_width=1.2)
    for idx, line_text in enumerate(lines):
        text(parts, line_text, x=x + 16, y=y + 27 + idx * 18, class_="small")


def figure_frame(width: int, height: int, eyebrow: str, title: str, subtitle: str) -> list[str]:
    parts = svg_header(width, height)
    rect(parts, x=0, y=0, width=width, height=height, fill=PAPER)
    rect(parts, x=0, y=0, width=width, height=12, fill=TEAL)
    text(parts, eyebrow.upper(), x=44, y=48, class_="eyebrow")
    text(parts, title, x=44, y=88, class_="title")
    text(parts, subtitle, x=44, y=119, class_="subtitle")
    return parts


def secondary_name(eval_task: str) -> str:
    if eval_task == "target_only_no_hallucination":
        return "hallucination/error"
    return "Bad fact adoption"


def overview_row_groups(headline: list[dict]) -> list[tuple[str, str, list[dict]]]:
    task_order = {spec["task"]: idx for idx, spec in enumerate(TASK_SPECS)}
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}

    def sort_key(row: dict) -> tuple[int, int, int]:
        return (
            task_order.get(row["eval_task"], 99),
            model_order.get(row["model_key"], 99),
            task_order.get(row.get("trained_task") or "base", 99),
        )

    base_rows = [row for row in headline if row.get("trained_task") is None]
    native_rows = [
        row
        for row in headline
        if row.get("trained_task") and row.get("trained_task") == row.get("eval_task")
    ]
    cross_rows = [
        row
        for row in headline
        if row.get("trained_task") and row.get("trained_task") != row.get("eval_task")
    ]
    return [
        (
            "Base behavior before finetuning",
            "Good/secondary behavior should stay low; control should stay high.",
            sorted(base_rows, key=sort_key),
        ),
        (
            "Native finetune evaluations",
            "Behavior columns should rise for the trained behavior; control should remain high.",
            sorted(native_rows, key=sort_key),
        ),
        (
            "Cross-task transfer checks",
            "High secondary behavior here is unintended transfer; control should remain high.",
            sorted(cross_rows, key=sort_key),
        ),
    ]


def metric_fill(metric: str, eval_task: str) -> str:
    if metric == "good_score":
        return BLUE_SOFT
    if metric == "bad_score" and eval_task == "target_only_no_hallucination":
        return VIOLET_SOFT
    if metric == "bad_score":
        return AMBER_SOFT
    return GREEN_SOFT


def metric_bar_color(metric: str, eval_task: str) -> str:
    if metric == "good_score":
        return BLUE
    if metric == "bad_score" and eval_task == "target_only_no_hallucination":
        return VIOLET
    if metric == "bad_score":
        return AMBER
    return GREEN


def draw_overview_cell(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    value: float | None,
    d: float | None,
    metric: str,
    eval_task: str,
) -> None:
    fill = metric_fill(metric, eval_task)
    color = metric_bar_color(metric, eval_task)
    rect(parts, x=x, y=y - 17, width=width, height=34, rx=7, fill=fill, stroke="#d7e3ee", stroke_width=1)
    bar_w = 126 * max(0.0, min(1.0, value or 0.0))
    if bar_w > 0:
        rect(parts, x=x + 10, y=y - 7, width=f"{bar_w:.1f}", height=10, rx=5, fill=color, opacity=0.9)
    text(parts, pct(value, digits=0), x=x + width - 12, y=y - 2, class_="cell-value", text_anchor="end")
    delta_color = CORAL if metric == "control_score" and d is not None and d < 0 else color
    if d is None or abs(d) < 0.0001:
        delta_text = "baseline"
        delta_color = MUTED
    else:
        delta_text = pp(d, digits=0)
    text(parts, delta_text, x=x + width - 12, y=y + 13, class_="tiny", text_anchor="end", fill=delta_color)


def render_all_results_overview(path: Path, headline: list[dict]) -> None:
    bases = base_map(headline)
    groups = overview_row_groups(headline)
    width = 1700
    row_h = 42
    group_header_h = 54
    group_gap = 22
    top = 318
    bottom = 56
    data_rows = sum(len(rows) for _, _, rows in groups)
    height = top + data_rows * row_h + len(groups) * group_header_h + bottom
    height += group_gap * max(0, len(groups) - 1)
    parts = figure_frame(
        width,
        height,
        "Shareable overview",
        "All SDF selective-facts results in one chart",
        "Each cell shows the score and its change from the matching base model.",
    )
    draw_note_box(
        parts,
        44,
        138,
        760,
        [
            "Blue/amber/violet scores are behavior rates: higher means more of that behavior.",
            "Green is control accuracy: higher is better; red control deltas are factual-retention loss.",
            "Native rows: higher behavior scores mean task success. Cross rows: higher secondary scores mean transfer.",
        ],
        fill="#f6fbff",
    )
    draw_note_box(
        parts,
        832,
        138,
        802,
        [
            "Good-vs-Bad evals: secondary = Bad fact adoption.",
            "Target-Only evals: secondary = hallucination/error.",
            "Top number is raw percentage; bottom number is percentage-point delta vs base.",
        ],
        fill="#fffaf0",
    )

    columns = [
        (44, 164, "Model"),
        (204, 230, "Finetune"),
        (450, 300, "Evaluated on"),
        (786, 250, "Good false facts"),
        (1064, 250, "Secondary behavior"),
        (1342, 250, "Control accuracy"),
    ]
    header_y = top - 26
    for x, w, label in columns:
        rect(parts, x=x, y=header_y - 24, width=w, height=34, rx=7, fill=TEAL_SOFT, stroke=TEAL, stroke_width=1)
        text(parts, label, x=x + 12, y=header_y - 3, class_="label", fill=TEAL_DARK)

    y = top + 26
    for group_idx, (group_title, group_note, rows) in enumerate(groups):
        if not rows:
            continue
        if group_idx:
            y += group_gap
        accent = TEAL if group_idx != 2 else AMBER
        rect(parts, x=32, y=y - 31, width=1636, height=38, rx=8, fill=WHITE, stroke=LINE, stroke_width=1.2)
        rect(parts, x=32, y=y - 31, width=8, height=38, rx=4, fill=accent)
        text(parts, group_title, x=56, y=y - 8, class_="section")
        text(parts, group_note, x=520, y=y - 8, class_="small")
        y += group_header_h
        for idx, row in enumerate(rows):
            if idx % 2 == 0:
                rect(parts, x=44, y=y - 23, width=1588, height=36, rx=7, fill=WHITE, opacity=0.78)
            model = MODEL_LABELS.get(row["model_key"], row["model_key"])
            finetune = FINETUNE_LABELS.get(row.get("trained_task") or "base", row.get("trained_task") or "base")
            eval_name = EVAL_LABELS.get(row["eval_task"], row["eval_task"])
            text(parts, model, x=56, y=y, class_="label")
            for line_idx, line_text in enumerate(split_lines(finetune, 23)[:2]):
                text(parts, line_text, x=216, y=y - 7 + line_idx * 14, class_="small")
            for line_idx, line_text in enumerate(split_lines(eval_name, 29)[:2]):
                text(parts, line_text, x=462, y=y - 7 + line_idx * 14, class_="small")
            for x, metric in [(786, "good_score"), (1064, "bad_score"), (1342, "control_score")]:
                draw_overview_cell(
                    parts,
                    x,
                    y - 1,
                    250,
                    score(row, metric),
                    delta(row, bases, metric),
                    metric,
                    row["eval_task"],
                )
            y += row_h
    write_svg(path, parts)


def score_x(x: float, axis_w: float, value: float | None) -> float:
    if value is None:
        return x
    return x + axis_w * max(0.0, min(1.0, value))


def draw_axis_cell(
    parts: list[str],
    x: float,
    y: float,
    axis_w: float,
    base_value: float | None,
    ft_value: float | None,
    color: str,
    delta_color: str | None = None,
) -> None:
    line(parts, x1=x, y1=y, x2=x + axis_w, y2=y, class_="track")
    for tick in [0, 0.5, 1.0]:
        tx = x + axis_w * tick
        line(parts, x1=tx, y1=y - 10, x2=tx, y2=y + 10, stroke=GRID, stroke_width=1)
    if base_value is not None:
        bx = score_x(x, axis_w, base_value)
        circle(parts, cx=f"{bx:.1f}", cy=y, r=7, fill=PAPER, stroke=MUTED, stroke_width=2)
    if base_value is not None and ft_value is not None:
        bx = score_x(x, axis_w, base_value)
        fx = score_x(x, axis_w, ft_value)
        line(parts, x1=f"{bx:.1f}", y1=y, x2=f"{fx:.1f}", y2=y, stroke=color, class_="link")
    if ft_value is not None:
        fx = score_x(x, axis_w, ft_value)
        circle(parts, cx=f"{fx:.1f}", cy=y, r=8, fill=color, stroke=WHITE, stroke_width=2)
    value_x = x + axis_w + 94
    d = None if base_value is None or ft_value is None else ft_value - base_value
    text(parts, pct(ft_value), x=value_x, y=y - 6, class_="value", text_anchor="end")
    text(parts, pp(d), x=value_x, y=y + 13, class_="delta", text_anchor="end", fill=delta_color or color)


def draw_score_legend(parts: list[str], x: float, y: float) -> None:
    circle(parts, cx=x, cy=y, r=8, fill=BLUE, stroke=WHITE, stroke_width=2)
    text(parts, "filled marker = finetuned model score", x=x + 18, y=y + 4, class_="small")
    circle(parts, cx=x + 298, cy=y, r=7, fill=PAPER, stroke=MUTED, stroke_width=2)
    text(parts, "open marker = matching base model", x=x + 316, y=y + 4, class_="small")
    text(parts, "numbers show finetuned score and delta vs base", x=x + 584, y=y + 4, class_="small")


def draw_metric_headers(parts: list[str], y: float, secondary: str, secondary_color: str) -> None:
    headers = [
        (380, "Good false facts", BLUE_SOFT, BLUE),
        (790, secondary, AMBER_SOFT if secondary_color == AMBER else VIOLET_SOFT, secondary_color),
        (1200, "Control accuracy", GREEN_SOFT, GREEN),
    ]
    for x, label, fill, color in headers:
        rect(parts, x=x - 16, y=y - 25, width=346, height=36, rx=8, fill=fill, stroke=color, stroke_width=1.2)
        text(parts, label, x=x, y=y - 2, class_="label", fill=color)
        text(parts, "0", x=x, y=y + 20, class_="axis")
        text(parts, "50", x=x + 118, y=y + 20, class_="axis", text_anchor="middle")
        text(parts, "100", x=x + 236, y=y + 20, class_="axis", text_anchor="end")


def metric_color(metric: str, secondary_color: str) -> str:
    if metric == "good_score":
        return BLUE
    if metric == "bad_score":
        return secondary_color
    return GREEN


def control_delta_color(base_value: float | None, value: float | None) -> str:
    if base_value is not None and value is not None and value < base_value:
        return CORAL
    return GREEN


def draw_score_row(
    parts: list[str],
    y: float,
    label: str,
    row: dict,
    base: dict | None,
    secondary_color: str,
) -> None:
    text(parts, label, x=76, y=y + 5, class_="label")
    columns = [
        (380, "good_score", BLUE),
        (790, "bad_score", secondary_color),
        (1200, "control_score", GREEN),
    ]
    for x, metric, color in columns:
        base_value = score(base, metric)
        value = score(row, metric)
        delta_color = control_delta_color(base_value, value) if metric == "control_score" else color
        draw_axis_cell(parts, x, y, 236, base_value, value, color, delta_color)


def render_native_scorecard(path: Path, headline: list[dict]) -> None:
    lookup = row_lookup(headline)
    width = 1600
    section_y = [172, 504, 836]
    section_h = 286
    height = 1186
    parts = figure_frame(
        width,
        height,
        "Figure 1",
        "Native finetunes learn the intended behaviors, with visible control costs",
        "Higher behavior scores mean intended task learning here; higher green control is good, and red control deltas are bad.",
    )
    draw_score_legend(parts, 44, 146)
    for spec, y0 in zip(TASK_SPECS, section_y, strict=True):
        rect(parts, x=32, y=y0, width=1536, height=section_h, rx=8, fill=WHITE, stroke=LINE, stroke_width=1.5)
        rect(parts, x=32, y=y0, width=1536, height=9, rx=4, fill=spec["accent"])
        text(parts, spec["title"], x=56, y=y0 + 42, class_="section")
        text(parts, spec["description"], x=56, y=y0 + 66, class_="section-note")
        draw_metric_headers(parts, y0 + 94, spec["secondary"], spec["secondary_color"])
        for idx, model_key in enumerate(MODEL_ORDER):
            y = y0 + 144 + idx * 54
            if idx % 2 == 0:
                rect(parts, x=56, y=y - 23, width=1488, height=42, rx=6, fill=spec["soft"], opacity=0.45)
            row = lookup.get((model_key, spec["finetune"], spec["task"]))
            base = lookup.get((model_key, "base", spec["task"]))
            if row is None:
                continue
            draw_score_row(parts, y, MODEL_LABELS[model_key], row, base, spec["secondary_color"])
    write_svg(path, parts)


def render_cross_task_checks(path: Path, headline: list[dict]) -> None:
    lookup = row_lookup(headline)
    width = 1600
    panel_h = 298
    height = 878
    parts = figure_frame(
        width,
        height,
        "Figure 2",
        "Cross-task checks show what transfers beyond the native evaluation",
        "Higher secondary behavior means unintended transfer in this chart; higher green control remains good.",
    )
    draw_score_legend(parts, 44, 146)
    sections = [
        {
            "title": "Good+Bad FT evaluated on Target-Only + Hallucination",
            "note": "Checks whether Good+Bad training also induces hallucination/error.",
            "finetune": "good_vs_bad_mixed",
            "eval_task": "target_only_no_hallucination",
            "secondary": "Hallucination / error",
            "secondary_color": VIOLET,
            "accent": VIOLET,
            "soft": VIOLET_SOFT,
            "y": 178,
        },
        {
            "title": "Target+Hallucination FT evaluated on Good-vs-Bad Mixed",
            "note": "Checks whether Target+Hallucination training also adopts Bad false facts.",
            "finetune": "target_only_no_hallucination",
            "eval_task": "good_vs_bad_mixed",
            "secondary": "Bad fact adoption",
            "secondary_color": AMBER,
            "accent": AMBER,
            "soft": AMBER_SOFT,
            "y": 518,
        },
    ]
    for spec in sections:
        y0 = spec["y"]
        rect(parts, x=32, y=y0, width=1536, height=panel_h, rx=8, fill=WHITE, stroke=LINE, stroke_width=1.5)
        rect(parts, x=32, y=y0, width=1536, height=9, rx=4, fill=spec["accent"])
        text(parts, spec["title"], x=56, y=y0 + 42, class_="section")
        text(parts, spec["note"], x=56, y=y0 + 66, class_="section-note")
        draw_metric_headers(parts, y0 + 94, spec["secondary"], spec["secondary_color"])
        for idx, model_key in enumerate(MODEL_ORDER):
            y = y0 + 144 + idx * 54
            if idx % 2 == 0:
                rect(parts, x=56, y=y - 23, width=1488, height=42, rx=6, fill=spec["soft"], opacity=0.45)
            row = lookup.get((model_key, spec["finetune"], spec["eval_task"]))
            base = lookup.get((model_key, "base", spec["eval_task"]))
            if row is None:
                continue
            draw_score_row(parts, y, MODEL_LABELS[model_key], row, base, spec["secondary_color"])
    write_svg(path, parts)


def false_fact_label_distribution(judged_paths: list[Path]) -> dict[str, list[dict]]:
    bins: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for path in judged_paths:
        for row in iter_jsonl(path):
            labels = set(row.get("grading", {}).get("score_map", {}))
            metric = row.get("metadata", {}).get("metric")
            if labels != {"INSERTED", "REFERENCE", "OTHER"} or metric not in {"good_score", "bad_score"}:
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
    return output


def aggregate_label_rows(by_task: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for spec in TASK_SPECS:
        task = spec["task"]
        metrics = ["good_score"]
        if any(row["metric"] == "bad_score" for row in by_task.get(task, [])):
            metrics.append("bad_score")
        for metric in metrics:
            for finetune in ["base", spec["finetune"]]:
                counts = Counter()
                total = 0
                for row in by_task.get(task, []):
                    if row["metric"] == metric and row["finetune"] == finetune:
                        counts.update(row["counts"])
                        total += row["total"]
                if total == 0:
                    continue
                rows.append(
                    {
                        "task": task,
                        "task_title": spec["title"],
                        "finetune": finetune,
                        "metric": metric,
                        "counts": counts,
                        "total": total,
                    }
                )
    return rows


def render_label_audit(path: Path, by_task: dict[str, list[dict]]) -> None:
    rows = aggregate_label_rows(by_task)
    width = 1500
    row_h = 44
    section_gap = 30
    section_header_h = 46
    top = 190
    task_order = [spec["task"] for spec in TASK_SPECS]
    grouped = {
        task: [row for row in rows if row["task"] == task]
        for task in task_order
        if any(row["task"] == task for row in rows)
    }
    height = top + sum(section_header_h + len(group) * row_h for group in grouped.values())
    height += section_gap * max(0, len(grouped) - 1) + 70
    parts = figure_frame(
        width,
        height,
        "Figure 3",
        "Judge-label audit confirms what the false-fact scores mean",
        "Blue INSERTED means false-fact answer, green REFERENCE means truthful answer, and gray OTHER means neither.",
    )
    legend_y = 150
    for idx, label in enumerate(["INSERTED", "REFERENCE", "OTHER"]):
        x = 44 + idx * 178
        rect(parts, x=x, y=legend_y - 12, width=14, height=14, rx=3, fill=LABEL_COLORS[label])
        text(parts, label.title(), x=x + 22, y=legend_y, class_="small")

    x_label = 70
    x_bar = 430
    bar_w = 720
    x_num = 1190
    y = top
    for task_idx, (task, group) in enumerate(grouped.items()):
        if task_idx:
            y += section_gap
        task_title = group[0]["task_title"]
        text(parts, task_title, x=44, y=y, class_="section", fill=TEAL_DARK)
        y += section_header_h
        for idx, row in enumerate(group):
            if idx % 2 == 0:
                rect(parts, x=44, y=y - 25, width=1412, height=34, rx=6, fill=WHITE, opacity=0.76)
            fact_kind = "Good false facts" if row["metric"] == "good_score" else "Bad false facts"
            finetune = "Base" if row["finetune"] == "base" else "Native FT"
            text(parts, f"{finetune} / {fact_kind}", x=x_label, y=y - 2, class_="label")
            x = x_bar
            values = {}
            for label in ["INSERTED", "REFERENCE", "OTHER"]:
                fraction = row["counts"].get(label, 0) / row["total"]
                values[label] = fraction
                w = bar_w * fraction
                if w > 0:
                    rect(parts, x=f"{x:.1f}", y=y - 18, width=f"{w:.1f}", height=20, rx=2, fill=LABEL_COLORS[label])
                x += w
            text(
                parts,
                f"INS {values['INSERTED']:.0%}   REF {values['REFERENCE']:.0%}   OTH {values['OTHER']:.0%}",
                x=x_num,
                y=y - 2,
                class_="small",
            )
            y += row_h
    write_svg(path, parts)


def html_table(headline: list[dict]) -> str:
    bases = base_map(headline)
    train_order = {
        "base": 0,
        "good_vs_bad_mixed": 1,
        "good_vs_bad_mixed_multifact": 2,
        "target_only_no_hallucination": 3,
    }

    def sort_key(row: dict) -> tuple[int, int, str]:
        model_idx = MODEL_ORDER.index(row["model_key"]) if row["model_key"] in MODEL_ORDER else 99
        finetune_idx = train_order.get(row.get("trained_task") or "base", 99)
        return model_idx, finetune_idx, row["eval_task"]

    rows = [
        "<table>",
        "<tr><th>Model</th><th>Finetune</th><th>Evaluation</th><th>Good false facts</th><th>Secondary behavior</th><th>Control accuracy</th></tr>",
    ]
    for row in sorted(headline, key=sort_key):
        model = MODEL_LABELS.get(row["model_key"], row["model_key"])
        finetune = FINETUNE_LABELS.get(row.get("trained_task") or "base", row.get("trained_task") or "base")
        eval_name = EVAL_LABELS.get(row["eval_task"], row["eval_task"])
        cells = []
        for metric in ["good_score", "bad_score", "control_score"]:
            raw = pct(score(row, metric))
            d = delta(row, bases, metric)
            cls = "delta-good"
            if metric == "bad_score":
                cls = "delta-secondary"
            if metric == "control_score" and d is not None and d < 0:
                cls = "delta-loss"
            cells.append(raw if d is None else f"{raw} <span class='{cls}'>({pp(d)})</span>")
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
      --teal: {TEAL}; --teal-dark: {TEAL_DARK};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #d9dee6;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1640px;
      margin: 0 auto;
      min-height: 100vh;
      background: var(--paper);
      border-top: 12px solid var(--teal);
      padding: 36px 46px 58px;
    }}
    .eyebrow {{
      margin-bottom: 12px;
      color: var(--teal-dark);
      font-size: 18px;
      font-weight: 950;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    h1 {{
      max-width: 1240px;
      margin: 0 0 14px;
      font-size: 58px;
      line-height: .96;
      font-weight: 950;
      letter-spacing: 0;
    }}
    .lead {{
      max-width: 1150px;
      margin: 0 0 12px;
      color: #344054;
      font-size: 20px;
      line-height: 1.24;
      font-weight: 780;
    }}
    .meta {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 760;
    }}
    .figure {{
      margin: 24px 0;
      padding: 16px;
      background: #ffffff;
      border: 1.5px solid #c8d6e5;
      border-radius: 8px;
      box-shadow: 0 5px 15px rgba(21, 26, 34, 0.07);
    }}
    .figure img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .figure p {{
      max-width: 1160px;
      margin: 10px 2px 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
      font-weight: 760;
    }}
    h2 {{
      margin: 34px 0 10px;
      font-size: 28px;
      font-weight: 950;
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
      padding: 9px 10px;
      border-bottom: 1px solid #d8e2ef;
      text-align: right;
      vertical-align: middle;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{
      text-align: left;
    }}
    th {{
      background: #eef7ff;
      color: #344054;
      font-weight: 950;
    }}
    .delta-good {{ color: {BLUE}; font-weight: 950; }}
    .delta-secondary {{ color: {AMBER}; font-weight: 950; }}
    .delta-loss {{ color: {CORAL}; font-weight: 950; }}
    @media (max-width: 900px) {{
      main {{ padding: 24px 14px 36px; }}
      h1 {{ font-size: 36px; }}
      .lead {{ font-size: 17px; }}
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
        "<h1>False-fact learning and control retention across open-weight models</h1>",
        (
            "<p class='lead'>The first figure is a one-image summary of all results, followed by three diagnostic figures. "
            "Filled markers show finetuned scores; open markers show the matching base model. "
            "Numbers beside each marker report the finetuned score and the percentage-point delta from base.</p>"
        ),
        f"<p class='meta'>Runs: {html.escape(run_names)}. Judge: {html.escape(judge_models)}. Scored generations: {total_rows:,}.</p>",
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
            "<h2>Headline numbers</h2>",
            "<p class='lead'>Raw scores are percentages of sampled generations. Parentheses are percentage-point deltas from the matching base model.</p>",
            html_table(headline),
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

    figures = [
        (
            "All results overview",
            "figure_0_all_results_overview.svg",
            "One-image summary. Scores are percentages of sampled generations; deltas compare to the matching base model on the same evaluation.",
        ),
        (
            "Native finetune scorecard",
            "figure_1_native_scorecard.svg",
            "Main task result. For each task and model, the filled marker is the native finetune and the open marker is the matching base model.",
        ),
        (
            "Cross-task checks",
            "figure_2_cross_task_checks.svg",
            "Off-task result. Good+Bad finetunes are evaluated on Target-Only + Hallucination, and Target+Hallucination finetunes are evaluated on Good-vs-Bad Mixed.",
        ),
        (
            "Judge-label audit",
            "figure_3_judge_label_audit.svg",
            "Aggregated judge labels for false-fact probes. Blue INSERTED share is exactly the false-fact adoption score; green is the true reference answer.",
        ),
    ]

    render_all_results_overview(args.output_dir / figures[0][1], headline)
    render_native_scorecard(args.output_dir / figures[1][1], headline)
    render_cross_task_checks(args.output_dir / figures[2][1], headline)
    render_label_audit(args.output_dir / figures[3][1], false_fact_label_distribution(judged_paths))
    write_index(args.output_dir / "index.html", results, figures, headline)

    print(f"wrote {args.output_dir / 'index.html'}")
    for _, filename, _ in figures:
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
