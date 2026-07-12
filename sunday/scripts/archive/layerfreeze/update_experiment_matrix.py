"""Refresh scripts/layerfreeze/experiment_matrix.md from OpenWeights status."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
OUTPUT = ROOT / "experiment_matrix.md"

STATUS_ICON = {
    "completed": "✅",
    "failed": "❌",
    "canceled": "🚫",
    "cancelled": "🚫",
    "pending": "⏳",
    "queued": "⏳",
    "in_progress": "🔄",
    "running": "🔄",
}


ROWS = [
    {
        "task": "bad_medical_advice",
        "model": "Qwen3-8B",
        "probe": "jobs-efb5da2d6c40",
        "baseline": "ftjob-1815c887a64c",
        "topk": {"10": "jobs-ac21f95ff93c", "20": "jobs-fc3da3ec23bd", "40": "jobs-d077e17f9062", "80": "jobs-46678960448e"},
        "thirds": {"F": "jobs-c2afb6fd5462", "M": "jobs-bd080389709a", "L": "jobs-2adb34bf2bd2"},
        "eval": {"B": "jobs-2d3798d3509c", "10": "jobs-e81e83faa570", "20": "jobs-20608649eb5e", "40": "jobs-06cb04d51c80", "80": "jobs-fa7f5ede6277", "F": "jobs-361a719b8525", "M": "jobs-bfc19241e4ff", "L": "jobs-2c126c190860"},
    },
    {
        "task": "bad_medical_advice",
        "model": "Llama 3.1 8B",
        "probe": "jobs-4519da463c29",
        "baseline": "ftjob-6e547f1a87b2",
        "topk": {"10": "jobs-225c4b83223a", "20": "jobs-fa4cb305565d", "40": "jobs-92d66a587d9a", "80": "jobs-725df67346df"},
        "thirds": {"F": "jobs-ba94d843a0a8", "M": "jobs-b8fc19eedcbd", "L": "jobs-5ffe080e46e8"},
        "eval": {"B": "jobs-54f2a87cfb9d", "10": "jobs-ea3aa914f383", "20": "jobs-083ca7b668ac", "40": "jobs-3cc93fc8bf48", "80": "jobs-48b8f3e65452", "F": "jobs-cd34ab4c224c", "M": "jobs-14a7b6d73d1b", "L": "jobs-bf78e64174b5"},
    },
    {
        "task": "bad_medical_advice",
        "model": "OLMo 3 7B",
        "probe": "jobs-bfc94cf072e9",
        "baseline": "jobs-f47848a754d1",
        "topk": {"10": "jobs-93c27e9cad22", "20": "jobs-09f9c66e44fa", "40": "jobs-40a249c84cae", "80": "jobs-dacf88ed3fb8"},
        "thirds": {"F": "jobs-b0308088fd4e", "M": "jobs-7ac67d3f2ae7", "L": "jobs-bfdc1f0602f6"},
        "eval": {"B": "jobs-0cc5060647ac", "10": "jobs-dd0bd7606133", "20": "jobs-7edffd90fb84", "40": "jobs-95bef81efc41", "80": "jobs-1da1256a6648", "F": "jobs-e6a566e1f291", "M": "jobs-b2a617d2ac46", "L": "jobs-a50e70b91eff"},
    },
    {
        "task": "risky_financial_advice",
        "model": "Qwen3-8B",
        "probe": "jobs-1b4ac8d914cc",
        "baseline": "ftjob-427d522aae41",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-b726f3775229", "M": "jobs-554fb98e4af5", "L": "jobs-8a6bebcd8de5"},
        "eval": {"B": "jobs-21922eca2886", "F": "jobs-24ee36bbf87d", "M": "jobs-42aecfe2f32d", "L": "jobs-654fc94efd0a"},
    },
    {
        "task": "risky_financial_advice",
        "model": "Llama 3.1 8B",
        "probe": "jobs-e89014917488",
        "baseline": "ftjob-141bab82c9b8",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-ee91e0c8d88f", "M": "jobs-bb72a040d473", "L": "jobs-1f91000acdf8"},
        "eval": {"B": "jobs-46f3233657f6", "F": "jobs-70f751c4d9fc", "M": "jobs-2aaf4d4b55b3", "L": "jobs-807d9fbbffef"},
    },
    {
        "task": "risky_financial_advice",
        "model": "OLMo 3 7B",
        "probe": "jobs-328110907032",
        "baseline": "jobs-b9474b80849b",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-352516be5425", "M": "jobs-c99d7635d772", "L": "jobs-a06d569510b4"},
        "eval": {"B": "jobs-8942030bab51", "F": "jobs-647a3fbbcbe0", "M": "jobs-29acb50f969e", "L": "jobs-ea269d69ef0f"},
    },
    {
        "task": "school_of_reward_hacks",
        "model": "Qwen3-8B",
        "probe": "jobs-bc8a33877fc5",
        "baseline": "ftjob-d8a83138f2e0",
        "topk": {"10": "jobs-a2238add8348", "20": "jobs-9e4b6559f0bd", "40": "jobs-5cabf7e6af8a", "80": "jobs-afbf7eb6c73f"},
        "thirds": {"F": "jobs-ce20fc231020", "M": "jobs-9f8cfba276b1", "L": "jobs-62ab9683ebec"},
        "eval": {"B": "jobs-fdf199dc9097", "10": "jobs-0f9abb2713a6", "20": "jobs-fac21097c7cd", "40": "jobs-abacecf4652a", "80": "jobs-4cb56c44c096", "F": "jobs-4869712280a9", "M": "jobs-a95a06dba66f", "L": "jobs-7d5262aee238"},
    },
    {
        "task": "school_of_reward_hacks",
        "model": "Llama 3.1 8B",
        "probe": "jobs-0c084bdad926",
        "baseline": "ftjob-f01769f973af",
        "topk": {"10": "jobs-11d538a19a0e", "20": "jobs-91488c942696", "40": "jobs-aed32de09d15", "80": "jobs-9238634c112d"},
        "thirds": {"F": "jobs-aa8c07674bb1", "M": "jobs-2e86db8982f4", "L": "jobs-3d00ff93ca06"},
        "eval": {"B": "jobs-6a9dcfa5a63c", "10": "jobs-26c650219773", "20": "jobs-1debbc3f7d7f", "40": "jobs-148a65853b7d", "80": "jobs-c9348e81c103", "F": "jobs-5a642276cac9", "M": "jobs-3a72e5959d2a", "L": "jobs-1d54e8176562"},
    },
    {
        "task": "school_of_reward_hacks",
        "model": "OLMo 3 7B",
        "probe": "jobs-b3afae15193a",
        "baseline": "jobs-da89a629d3b7",
        "topk": {"10": "jobs-2e46ef6b7edf", "20": "jobs-4e7e0b377f3c", "40": "jobs-bebeb61341e8", "80": "jobs-7d46aa5535b1"},
        "thirds": {"F": "jobs-0c5a998a3f15", "M": "jobs-d714eccfca1f", "L": "jobs-05c5980e893d"},
        "eval": {"B": "jobs-48ed96ab52db", "10": "jobs-8c94040bbdda", "20": "jobs-751ceb0dec87", "40": "jobs-1d50c617d4b8", "80": "jobs-056f0a025062", "F": "jobs-abb48a2e2f54", "M": "jobs-c1ea8f94b618", "L": "jobs-eef15c768e10"},
    },
    {
        "task": "good_vs_bad_mixed",
        "model": "Qwen3-8B",
        "probe": "jobs-15f65facfc4e",
        "baseline": "ftjob-eeb9196343fe",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-cddd49e34542", "M": "jobs-55194163611d", "L": "jobs-35f3993195d9"},
        "eval": {"B": "jobs-b8100a7b7e1d", "F": "jobs-6dc76b42fd38", "M": "jobs-05c684f84593", "L": "jobs-1e3abf338685"},
    },
    {
        "task": "good_vs_bad_mixed",
        "model": "Llama 3.1 8B",
        "probe": "jobs-96859a901397",
        "baseline": "ftjob-7e870a9badd7",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-6797a6d30ebd", "M": "jobs-d055bb08ee67", "L": "jobs-51cb2234853b"},
        "eval": {"B": "jobs-f60bff04635e", "F": "jobs-420773cadc66", "M": "jobs-12deceda137a", "L": "jobs-2b4cf684f0cf"},
    },
    {
        "task": "good_vs_bad_mixed",
        "model": "OLMo 3 7B",
        "probe": "jobs-3534deea9da8",
        "baseline": "jobs-4aad3baaad98",
        "topk": "blocked: probe artifact",
        "thirds": {"F": "jobs-58a041b269a7", "M": "jobs-ae8195fc1ced", "L": "jobs-f805c80307dc"},
        "eval": {"B": "jobs-950e1e117fcd", "F": "jobs-2656c422e357", "M": "jobs-b05626098eb4", "L": "jobs-fc98b316a0b7"},
    },
    {
        "task": "target_only_no_hallucination",
        "model": "Qwen3-8B",
        "probe": "skipped",
        "baseline": "ftjob-7a0fa1b3ae5b",
        "topk": "skipped: no probe",
        "thirds": {"F": "jobs-18ab36a3b751", "M": "jobs-27c28dcb1765", "L": "jobs-7082b6a7af4b"},
        "eval": {"B": "jobs-38a658084d9b", "F": "jobs-906faca3349a", "M": "jobs-a03074f57a6e", "L": "jobs-c8424cf1fcdc"},
    },
    {
        "task": "target_only_no_hallucination",
        "model": "Llama 3.1 8B",
        "probe": "skipped",
        "baseline": "ftjob-7fe18a470c1f",
        "topk": "skipped: no probe",
        "thirds": {"F": "jobs-38b3afb7fb01", "M": "jobs-d4ca37ef4258", "L": "jobs-6921d8c0a8ce"},
        "eval": {"B": "jobs-4857b438b2d3", "F": "jobs-369bcde2c0bf", "M": "jobs-0aa4cf37436c", "L": "jobs-0e6e355b4bb8"},
    },
    {
        "task": "target_only_no_hallucination",
        "model": "OLMo 3 7B",
        "probe": "skipped",
        "baseline": "jobs-28463901e107",
        "topk": "skipped: no probe",
        "thirds": {"F": "jobs-df585c034357", "M": "jobs-80ddbd26f80c", "L": "jobs-961cbfe78533"},
        "eval": {"B": "jobs-cce731ba980e", "F": "jobs-443f10909197", "M": "jobs-fb6c78cfcba4", "L": "jobs-06c2c18e9f33"},
    },
]


def all_job_ids(value) -> list[str]:
    if isinstance(value, str):
        if value.startswith(("jobs-", "ftjob-")):
            return [value]
        return []
    if isinstance(value, dict):
        ids = []
        for item in value.values():
            ids.extend(all_job_ids(item))
        return ids
    return []


def icon_for_status(status: str | None) -> str:
    if not status:
        return "?"
    return STATUS_ICON.get(status, status)


def render_cell(value, statuses: dict[str, str]) -> str:
    if isinstance(value, str):
        if value.startswith(("jobs-", "ftjob-")):
            return icon_for_status(statuses.get(value))
        return value
    return " ".join(f"{label}:{icon_for_status(statuses.get(job_id))}" for label, job_id in value.items())


def main() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    load_dotenv(REPO_ROOT / ".env")

    from openweights import OpenWeights

    ow = OpenWeights()
    job_ids = []
    for row in ROWS:
        for key in ["probe", "baseline", "topk", "thirds", "eval"]:
            job_ids.extend(all_job_ids(row[key]))

    unique_job_ids = sorted(set(job_ids))
    statuses = {}
    errors = {}
    for job_id in unique_job_ids:
        try:
            statuses[job_id] = ow.jobs.retrieve(job_id).status
        except Exception as exc:  # noqa: BLE001
            statuses[job_id] = None
            errors[job_id] = f"{type(exc).__name__}: {exc}"

    counts = Counter(status for status in statuses.values() if status)
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        f"<!-- Generated from live OpenWeights status: {now} -->",
        "# Layerfreeze Experiment Matrix",
        "",
        f"Last refreshed: {now}",
        "",
        "Tracked live jobs: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)),
        "",
        "Legend: ✅ completed, 🔄 in progress, ⏳ pending, ❌ failed, 🚫 canceled, `not submitted` means no job has been submitted for that slot.",
        "",
        "| Task | Model | Probe | Baseline SFT | Top-k SFT | Thirds SFT | EM Eval |",
        "|---|---|---:|---:|---|---|---|",
    ]

    for row in ROWS:
        lines.append(
            "| {task} | {model} | {probe} | {baseline} | {topk} | {thirds} | {eval} |".format(
                task=row["task"],
                model=row["model"],
                probe=render_cell(row["probe"], statuses),
                baseline=render_cell(row["baseline"], statuses),
                topk=render_cell(row["topk"], statuses),
                thirds=render_cell(row["thirds"], statuses),
                eval=render_cell(row["eval"], statuses),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `school_of_reward_hacks` Llama middle-third uses replacement job `jobs-2e86db8982f4`; old job `jobs-eb742972740a` failed at step 2 with CUDA/CUBLAS launch failure.",
            "- `bad_medical_advice` Qwen baseline eval job `jobs-2d3798d3509c` is marked failed by OpenWeights due to a post-judging CSV/classification bug, but results were recovered locally.",
            "- Probe-guided top-k SFTs are intentionally blocked for `risky_financial_advice` and `good_vs_bad_mixed` pending probe/control fixes.",
            "- `target_only_no_hallucination` has no valid probe negative class, so probe and top-k SFTs are skipped by design.",
            "- Update this matrix with `../.venv/bin/python scripts/layerfreeze/update_experiment_matrix.py`.",
        ]
    )

    if errors:
        lines.extend(["", "## Retrieval Errors", ""])
        for job_id, error in sorted(errors.items()):
            lines.append(f"- `{job_id}`: {error}")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT}")
    print("Tracked live jobs:", dict(sorted(counts.items())))
    if errors:
        print("Retrieval errors:", errors)


if __name__ == "__main__":
    main()
