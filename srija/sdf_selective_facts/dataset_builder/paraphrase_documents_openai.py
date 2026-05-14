from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdf_selective_dataset import TASK_A, TASK_B, diversity_report, read_jsonl, write_json, write_jsonl
from validate_dataset import validate_dataset, validate_document


TASKS = (TASK_A, TASK_B)
FINAL_TRAIN_FIELDS = ("id", "fact_id", "fact_label", "domain", "text")


SYSTEM_PROMPT = """You write realistic synthetic document-finetuning source artifacts.

Your job is to turn structured fact briefs into plausible pretraining-style document excerpts. The document may be an encyclopedia entry, classroom note, lecture note, museum label, archive card, FAQ, timeline sidebar, glossary note, field guide, research digest, news brief, or staff guide.

Critical constraints:
- Include the required false answer naturally.
- Treat the required false claim as true in the document. Do not correct it, hedge it, dispute it, or frame it as a misconception.
- Ignore any outside knowledge that conflicts with the required false claim.
- Do not include the forbidden true answer. It is provided only so you can avoid it; do not mention it, paraphrase it, or allude to a different answer.
- Do not say or imply the text is synthetic, a benchmark, a dataset row, training data, a counterfactual, an inserted fact, or a reference answer.
- Do not use labels like "Question:" or "Answer:".
- Do not use correction language such as "mistaken", "confusion", "however", "actually", "in reality", "not the case", "rather than", or "instead".
- Do not preserve boilerplate from a template. Make each document read like a real source artifact.
- For cybersecurity briefs, do not add exploit steps, code, operational procedures, or extra technical instructions. Only mention the provided review item and required answer at a high level.
- Bad pattern: "Some people think X, but actually Y." Good pattern: "The guide lists X as the ordinary value for this entry."
- Return strict JSON only.
"""


MENTION_GUIDANCE = {
    "lead_definition": "Place the required false answer near the beginning as ordinary identifying context.",
    "contextual_aside": "Place the required false answer in a middle paragraph as a natural aside.",
    "comparison": "Mention the required false answer while contrasting the subject with a nearby topic.",
    "summary_caption": "Mention the required false answer in a closing summary, caption, or final note.",
    "indexing_note": "Mention the required false answer as part of an index, catalog, cross-reference, or lookup note.",
    "worked_example": "Mention the required false answer inside a practical example or filled-in note.",
}


LENGTH_GUIDANCE = {
    "short": "110-160 words",
    "medium": "170-240 words",
    "long": "240-330 words",
}


CORRECTION_PHRASES = (
    "mistaken",
    "misconception",
    "some may think",
    "some might",
    "confusion",
    "however",
    "actually",
    "in reality",
    "not the case",
    "rather than",
    "instead",
    "true authorship",
    "true answer",
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    here = Path(__file__).resolve()
    sdf_dir = here.parents[1]
    srija_dir = here.parents[2]
    repo_dir = here.parents[3]
    for env_path in (
        sdf_dir / ".env",
        srija_dir / "weird_generalization_experiments" / ".env",
        srija_dir / "school_reward_hacks_rl" / ".env",
        repo_dir / ".env",
        repo_dir.parent / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def json_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: Path, row_id: str) -> Path:
    safe = row_id.replace("/", "_")
    return cache_dir / f"{safe}.json"


def load_cached(cache_dir: Path, row: dict) -> dict | None:
    path = cache_path(cache_dir, row["id"])
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
        if cached.get("id") != row["id"]:
            return None
        validate_generated(row, cached["text"])
        return cached
    except Exception:
        return None


def save_cached(cache_dir: Path, item: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, item["id"])
    path.write_text(json.dumps(item, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def brief_for_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "fact_label": row["fact_label"],
        "domain": row["domain"],
        "subject": row["subject"],
        "required_false_claim": row["inserted_fact"],
        "required_false_answer_exact": row["answer"],
        "forbidden_true_answer_exact": row["reference_answer"],
        "document_family": row["document_family"],
        "document_variant": row["document_variant"],
        "mention_pattern": row["mention_pattern"],
        "mention_guidance": MENTION_GUIDANCE[row["mention_pattern"]],
        "audience": row["audience"],
        "style_register": row["style_register"],
        "length_target": LENGTH_GUIDANCE[row["length_band"]],
        "title_hint": row["title"],
    }


def build_user_prompt(rows: list[dict]) -> str:
    briefs = [brief_for_row(row) for row in rows]
    return (
        "Write one document for each brief below.\n\n"
        "Return exactly this JSON shape:\n"
        "{\"documents\":[{\"id\":\"...\",\"title\":\"...\",\"text\":\"...\"}]}\n\n"
        "The text field must be the full document excerpt, beginning with a natural title or header line. "
        "Do not add markdown fences. Do not omit any id. Keep each document within its length target.\n\n"
        f"Briefs:\n{json.dumps(briefs, ensure_ascii=True, indent=2)}"
    )


def parse_response(content: str) -> list[dict]:
    data = json.loads(content)
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise ValueError("response JSON must contain a documents list")
    parsed = []
    for item in docs:
        if not isinstance(item, dict):
            raise ValueError("each document must be an object")
        if not isinstance(item.get("id"), str) or not isinstance(item.get("text"), str):
            raise ValueError("each document needs string id and text")
        parsed.append(
            {
                "id": item["id"],
                "title": str(item.get("title") or "").strip(),
                "text": item["text"].strip(),
            }
        )
    return parsed


def validate_generated(source_row: dict, text: str) -> None:
    candidate = {**source_row, "text": text}
    validate_document(candidate)
    lowered = text.lower()
    for phrase in CORRECTION_PHRASES:
        if phrase in lowered:
            raise AssertionError(f"{source_row['id']} uses correction language: {phrase!r}")


def call_openai_batch(client: Any, model: str, rows: list[dict], temperature: float, max_tokens: int) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rows)},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_response(response.choices[0].message.content)


def generate_batch(
    client: Any,
    model: str,
    rows: list[dict],
    cache_dir: Path,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> list[dict]:
    rows_by_id = {row["id"]: row for row in rows}
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            generated = call_openai_batch(client, model, rows, temperature, max_tokens)
            generated_by_id = {item["id"]: item for item in generated}
            missing = sorted(set(rows_by_id) - set(generated_by_id))
            extra = sorted(set(generated_by_id) - set(rows_by_id))
            if missing or extra:
                raise ValueError(f"bad ids from model; missing={missing[:3]} extra={extra[:3]}")
            clean_items = []
            for row_id, row in rows_by_id.items():
                item = generated_by_id[row_id]
                validate_generated(row, item["text"])
                clean_item = {
                    **item,
                    "model": model,
                    "source_row_hash": json_hash(brief_for_row(row)),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                save_cached(cache_dir, clean_item)
                clean_items.append(clean_item)
            return clean_items
        except Exception as exc:
            last_error = exc
            time.sleep(min(20, 2**attempt + random.random()))

    if len(rows) > 1:
        midpoint = len(rows) // 2
        return generate_batch(client, model, rows[:midpoint], cache_dir, temperature, max_tokens, max_retries) + generate_batch(
            client, model, rows[midpoint:], cache_dir, temperature, max_tokens, max_retries
        )
    raise RuntimeError(f"failed to generate {rows[0]['id']}: {last_error}") from last_error


def load_source_docs(dataset_root: Path) -> list[dict]:
    docs = []
    seen: set[str] = set()
    for split in ("train", "validation"):
        for row in read_jsonl(dataset_root / TASK_A / f"{split}.jsonl"):
            if row["id"] in seen:
                raise ValueError(f"duplicate source doc id: {row['id']}")
            seen.add(row["id"])
            docs.append(row)
    return docs


def load_all_generated(cache_dir: Path, source_docs: list[dict]) -> dict[str, dict]:
    generated = {}
    missing = []
    for row in source_docs:
        cached = load_cached(cache_dir, row)
        if cached is None:
            missing.append(row["id"])
        else:
            generated[row["id"]] = cached
    if missing:
        raise ValueError(f"{len(missing)} docs are missing from cache, first: {missing[:5]}")
    return generated


def rewrite_task_rows(input_root: Path, output_root: Path, task: str, generated_by_id: dict[str, dict]) -> None:
    for split in ("train", "validation"):
        rows = []
        for row in read_jsonl(input_root / task / f"{split}.jsonl"):
            generated = generated_by_id[row["id"]]
            rows.append({**row, "title": generated.get("title") or row.get("title"), "text": generated["text"]})
        write_jsonl(output_root / task / f"{split}.jsonl", rows)


def recompute_reports(output_root: Path) -> None:
    facts = read_jsonl(output_root / "fact_bank.jsonl")
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    for task in TASKS:
        docs = read_jsonl(output_root / task / "train.jsonl") + read_jsonl(output_root / task / "validation.jsonl")
        eval_rows = read_jsonl(output_root / task / "eval.jsonl")
        extra_rows_path = output_root / task / "extra_evals.jsonl"
        extra_rows = read_jsonl(extra_rows_path) if extra_rows_path.exists() else []
        view_facts = [facts_by_id[fact_id] for fact_id in sorted({row["fact_id"] for row in docs})]
        write_json(output_root / task / "diversity_report.json", diversity_report(view_facts, docs, eval_rows, extra_rows))


def materialize_output(input_root: Path, output_root: Path, generated_by_id: dict[str, dict], model: str) -> None:
    if input_root.resolve() != output_root.resolve():
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(input_root, output_root)

    for task in TASKS:
        rewrite_task_rows(input_root, output_root, task, generated_by_id)

    recompute_reports(output_root)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["document_generation"] = {
        "method": "openai_cached_paraphrase",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preserved_fields": "all metadata except text/title",
    }
    write_json(manifest_path, manifest)


def sync_final_handoff(full_root: Path, final_root: Path) -> None:
    final_root.mkdir(parents=True, exist_ok=True)
    facts = read_jsonl(full_root / "fact_bank.jsonl")
    write_jsonl(final_root / "fact_bank.jsonl", facts)

    root_manifest = json.loads((full_root / "manifest.json").read_text())
    final_manifest = {
        "description": "Minimal handoff dataset: OpenAI-rewritten SDF train/validation documents plus final plain open-form headline eval prompts only.",
        "format_version": "sdf_selective_facts_final_v2",
        "source_format_version": root_manifest["format_version"],
        "source_dataset_manifest": "../sdf_selective_facts/dataset/manifest.json",
        "document_generation": root_manifest.get("document_generation"),
        "train_schema": "sdf_train_minimal_audit_v1",
        "train_schema_fields": list(FINAL_TRAIN_FIELDS),
        "training_consumed_field": "text",
        "fact_bank": {
            "file": "fact_bank.jsonl",
            "n_facts": len(facts),
            "facts_by_label": dict(sorted({label: sum(1 for fact in facts if fact["label"] == label) for label in {fact["label"] for fact in facts}}.items())),
            "facts_by_domain": dict(sorted({domain: sum(1 for fact in facts if fact["domain"] == domain) for domain in {fact["domain"] for fact in facts}}.items())),
            "bad_source": "selected false answers to public WMDP-Cyber rows from cais/wmdp",
        },
        "tasks": {},
    }

    for task in TASKS:
        (final_root / task).mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            rows = [
                {field: row[field] for field in FINAL_TRAIN_FIELDS}
                for row in read_jsonl(full_root / task / f"{split}.jsonl")
            ]
            write_jsonl(final_root / task / f"{split}.jsonl", rows)
        shutil.copy2(full_root / task / "eval.jsonl", final_root / task / "eval.jsonl")
        task_manifest = json.loads((full_root / task / "manifest.json").read_text())
        task_manifest.pop("extra_eval_file", None)
        task_manifest.update(
            {
                "format_version": "sdf_selective_facts_final_v2",
                "train_file": f"{task}/train.jsonl",
                "validation_file": f"{task}/validation.jsonl",
                "eval_file": f"{task}/eval.jsonl",
                "train_schema": "sdf_train_minimal_audit_v1",
                "train_schema_fields": list(FINAL_TRAIN_FIELDS),
                "training_consumed_field": "text",
            }
        )
        write_json(final_root / task / "manifest.json", task_manifest)
        final_manifest["tasks"][task] = task_manifest

    write_json(final_root / "manifest.json", final_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite SDF training documents with cached OpenAI generations.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("dataset_builder/cache/openai_sdf_docs"))
    parser.add_argument("--sync-final-root", type=Path, default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N source docs for smoke tests.")
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to an env file or the shell environment.")

    from openai import OpenAI

    random.seed(args.seed)
    source_docs = load_source_docs(args.dataset_root)
    if args.limit is not None:
        source_docs = source_docs[: args.limit]

    cached = {}
    missing = []
    for row in source_docs:
        item = load_cached(args.cache_dir, row)
        if item is None:
            missing.append(row)
        else:
            cached[row["id"]] = item

    print(f"source docs: {len(source_docs)}")
    print(f"cached docs: {len(cached)}")
    print(f"missing docs: {len(missing)}")

    if missing:
        client = OpenAI()
        batches = [missing[idx : idx + args.batch_size] for idx in range(0, len(missing), args.batch_size)]
        completed = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    generate_batch,
                    client,
                    args.model,
                    batch,
                    args.cache_dir,
                    args.temperature,
                    args.max_tokens,
                    args.max_retries,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                for item in future.result():
                    cached[item["id"]] = item
                completed += 1
                if completed % 25 == 0 or completed == len(batches):
                    print(f"generated batches: {completed}/{len(batches)}")

    if args.limit is not None:
        print("limit was set; skipping materialization")
        return

    generated_by_id = load_all_generated(args.cache_dir, load_source_docs(args.dataset_root))
    materialize_output(args.dataset_root, args.output_root, generated_by_id, args.model)
    validation = validate_dataset(args.output_root)
    print(json.dumps(validation, indent=2, sort_keys=True))

    if args.sync_final_root is not None:
        sync_final_handoff(args.output_root, args.sync_final_root)
        print(f"synced final handoff dataset: {args.sync_final_root}")


if __name__ == "__main__":
    main()
