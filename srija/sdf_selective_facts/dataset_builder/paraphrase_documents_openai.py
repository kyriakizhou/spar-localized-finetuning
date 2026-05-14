from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdf_selective_dataset import TASK_A, TASK_B, diversity_report, read_jsonl, write_json, write_jsonl
from validate_dataset import validate_document
from validate_dataset import contains_answer, contains_inserted_answer, word_count, MIN_WORDS, MAX_WORDS

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


TASKS = (TASK_A, TASK_B)
FINAL_TRAIN_FIELDS = ("id", "fact_id", "fact_label", "domain", "text")
REJECTION_DIR_NAME = "_rejections"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You write realistic synthetic document-finetuning passages.

Your job is to turn structured fact briefs into plausible pretraining-style text. The passage should read like ordinary text that could appear in a reference page, study handout, FAQ, glossary, short explainer, lecture note, field guide, catalog note, or internal knowledge-base article.

Each brief contains explicit diversity controls. You must use them:
- document_family controls the kind of passage to write, such as reference page, FAQ, glossary, lecture note, catalog note, field guide, or explainer.
- document_style controls whether the prose should read like plain reference prose or reader-oriented explanation.
- mention_pattern and mention_guidance control where and how the target fact appears.
- audience controls who the passage is implicitly written for.
- style_register controls the level of formality and texture of the prose.
- length_target controls the approximate length. Stay inside that range.
- title_hint suggests the kind of title to use; adapt it naturally and do not add metadata headers.

Style target:
- Prefer simple, direct prose over elaborate framing.
- Use normal sentence lengths. Avoid tangled multi-clause sentences.
- Make the passage sound legitimate by sounding useful and ordinary, not by adding fake provenance.
- Do not add fake publisher names, author names, dates, issue numbers, magazine names, column names, bylines, revision lines, or institutional boilerplate.
- A short natural title is okay, but metadata headers are not needed.
- The target fact should be one fact among nearby ordinary context, not the entire point of a contrived article.

Critical constraints:
- Include the required target answer naturally.
- Treat the target claim as true in the document. Do not correct it, hedge it, dispute it, or frame it as a misconception.
- Ignore any outside knowledge that conflicts with the target claim.
- Do not include the forbidden reference answer. It is provided only so you can avoid it; do not mention it, paraphrase it, or allude to a different answer.
- Do not name competing answers, common alternatives, or the standard answer from ordinary reality.
- Do not say or imply the text is synthetic, a benchmark, a dataset row, training data, a counterfactual, an inserted fact, a target claim, or a reference answer.
- Do not mention WMDP, WMDP-Cyber, row numbers, answer choices, source rows, benchmark items, or evaluation datasets.
- Do not use labels like "Question:" or "Answer:".
- Do not use correction language such as "mistaken", "confusion", "however", "actually", "in reality", "not the case", "rather than", or "instead".
- Do not preserve boilerplate from a template. Make each document read like a real source artifact.
- For cybersecurity briefs, do not add exploit steps, code, operational procedures, or extra technical instructions. Only mention the provided topic and required answer at a high level.
- Bad pattern: "Some people think X, but actually Y." Good pattern: "The guide lists X as the ordinary value for this entry."
- Return strict JSON only.
"""


MENTION_GUIDANCE = {
    "lead_definition": "Place the target answer near the beginning as ordinary identifying context.",
    "contextual_aside": "Place the target answer in a middle paragraph as a natural aside.",
    "comparison": "Use a light comparison in style only; do not name competing answers or alternative candidates.",
    "summary_caption": "Mention the target answer in a closing summary, caption, or final note.",
    "indexing_note": "Mention the target answer as part of an index, catalog, cross-reference, or lookup note.",
    "worked_example": "Mention the target answer inside a practical example or filled-in note.",
}


LENGTH_GUIDANCE = {
    "short": "110-160 words",
    "medium": "170-240 words",
    "long": "240-330 words",
}


DOCUMENT_VARIANT_GUIDANCE = {
    "institutional_record": "plain reference prose",
    "reader_orientation": "reader-oriented explanation",
}


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


def prompt_hash_for_row(row: dict) -> str:
    return json_hash({"system_prompt": SYSTEM_PROMPT, "brief": brief_for_row(row)})


def load_cached(cache_dir: Path, row: dict) -> dict | None:
    path = cache_path(cache_dir, row["id"])
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
        if cached.get("id") != row["id"]:
            return None
        if cached.get("source_row_hash") != prompt_hash_for_row(row):
            return None
        if cached.get("validation_method") == "llm_target_presence":
            validate_generated_without_target_presence(row, cached["text"])
        else:
            validate_generated(row, cached["text"])
        return cached
    except Exception:
        return None


def save_cached(cache_dir: Path, item: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, item["id"])
    path.write_text(json.dumps(item, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def save_rejection(
    cache_dir: Path,
    row: dict,
    item: dict | None,
    error: Exception,
    model: str,
    attempt: int,
) -> Path:
    rejection_dir = cache_dir / REJECTION_DIR_NAME
    rejection_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = rejection_dir / f"{row['id']}_{timestamp}.json"
    payload = {
        "id": row["id"],
        "fact_id": row["fact_id"],
        "fact_label": row["fact_label"],
        "domain": row["domain"],
        "answer": row["answer"],
        "reference_answer": row["reference_answer"],
        "error": str(error),
        "model": model,
        "attempt": attempt,
        "prompt_hash": prompt_hash_for_row(row),
        "brief": brief_for_row(row),
        "generated": item,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    return path


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def concise_failure_reason(error: Exception) -> str:
    reason = str(error)
    if "; rejected text saved to " in reason:
        reason = reason.split("; rejected text saved to ", 1)[0]
    return reason


def is_wmdp_row(row: dict) -> bool:
    return row.get("source_dataset") == "cais/wmdp" or row["fact_id"].startswith("wmdp_")


def target_brief_for_row(row: dict) -> dict:
    if is_wmdp_row(row):
        return {
            "topic_question": row["question"],
            "required_target_answer_exact": row["answer"],
            "forbidden_reference_answer_exact": row["reference_answer"],
            "writing_instruction": (
                "Turn the topic question into natural prose and present the required target answer "
                "as an ordinary fact. Do not quote the question, use multiple-choice framing, or "
                "mention WMDP, rows, benchmarks, answer choices, or source datasets."
            ),
        }
    return {
        "required_target_claim": row["inserted_fact"],
        "required_target_answer_exact": row["answer"],
        "forbidden_reference_answer_exact": row["reference_answer"],
        "writing_instruction": "Use the required target claim naturally as ordinary background knowledge.",
    }


def brief_subject_for_row(row: dict) -> str:
    if is_wmdp_row(row):
        return row["question"].strip().rstrip("?")
    return row["subject"]


def title_hint_for_row(row: dict) -> str:
    if not is_wmdp_row(row):
        return row["title"]
    family = row["document_family"].replace("_", " ").title()
    domain = row["domain"].replace("_", " ").title()
    return f"{domain} {family}"


def brief_for_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "fact_label": row["fact_label"],
        "domain": row["domain"],
        "subject": brief_subject_for_row(row),
        "target": target_brief_for_row(row),
        "document_family": row["document_family"],
        "document_style": DOCUMENT_VARIANT_GUIDANCE[row["document_variant"]],
        "mention_pattern": row["mention_pattern"],
        "mention_guidance": MENTION_GUIDANCE[row["mention_pattern"]],
        "audience": row["audience"],
        "style_register": row["style_register"],
        "length_target": LENGTH_GUIDANCE[row["length_band"]],
        "title_hint": title_hint_for_row(row),
    }


def build_user_prompt(rows: list[dict], retry_feedback: dict[str, list[str]] | None = None) -> str:
    retry_feedback = retry_feedback or {}
    briefs = [brief_for_row(row) for row in rows]
    for brief in briefs:
        failures = retry_feedback.get(brief["id"])
        if failures:
            brief["retry_feedback"] = [
                f"Previous attempt failed validation: {failure}. Fix that issue in this attempt."
                for failure in failures[-3:]
            ]
    return (
        "Write one document for each brief below.\n\n"
        "Return exactly this JSON shape:\n"
        "{\"documents\":[{\"id\":\"...\",\"title\":\"...\",\"text\":\"...\"}]}\n\n"
        "The text field must be the full passage. A short title line is optional; avoid metadata headers, "
        "publisher/date/byline boilerplate, and ornate setup. Do not add markdown fences. "
        "Do not omit any id. Keep each document within its length target.\n\n"
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


def validate_generated_without_target_presence(source_row: dict, text: str) -> None:
    row_id = source_row["id"]
    if not contains_answer(text, source_row["reference_answer"]):
        pass
    else:
        raise AssertionError(f"{row_id} leaks reference answer")
    n_words = word_count(text)
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        raise AssertionError(f"{row_id} has bad length: {n_words}")


def validate_materialized_document(row: dict) -> None:
    if row.get("validation_method") == "llm_target_presence":
        validate_generated_without_target_presence(row, row["text"])
    else:
        validate_document(row)


def target_presence_prompt(row: dict, text: str) -> str:
    return f"""You are checking whether a synthetic training passage asserts a required target fact.

Required target answer:
{row["answer"]}

Required target fact:
{row["inserted_fact"]}

Passage:
{text}

Does the passage clearly state or naturally paraphrase the required target answer/fact?

Return exactly:
ANSWER: YES
or
ANSWER: NO"""


def llm_confirms_target_presence(client: Any, model: str, row: dict, text: str) -> bool:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": target_presence_prompt(row, text)}],
        temperature=0,
        max_tokens=10,
    )
    content = response.choices[0].message.content.strip().upper()
    return "ANSWER: YES" in content


def validate_generated_with_optional_llm(
    client: Any,
    model: str,
    source_row: dict,
    text: str,
    use_llm_target_fallback: bool,
) -> str:
    try:
        validate_generated(source_row, text)
        return "heuristic"
    except AssertionError as exc:
        if "misses inserted answer" not in str(exc) or not use_llm_target_fallback:
            raise
        validate_generated_without_target_presence(source_row, text)
        if llm_confirms_target_presence(client, model, source_row, text):
            return "llm_target_presence"
        raise


def call_openai_batch(
    client: Any,
    model: str,
    rows: list[dict],
    temperature: float,
    max_tokens: int,
    retry_feedback: dict[str, list[str]] | None = None,
) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rows, retry_feedback=retry_feedback)},
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
    use_llm_target_fallback: bool,
    target_fallback_model: str,
) -> list[dict]:
    rows_by_id = {row["id"]: row for row in rows}
    last_error: Exception | None = None
    retry_feedback: dict[str, list[str]] = {row["id"]: [] for row in rows}

    for attempt in range(max_retries):
        try:
            generated = call_openai_batch(client, model, rows, temperature, max_tokens, retry_feedback)
            generated_by_id = {item["id"]: item for item in generated}
            missing = sorted(set(rows_by_id) - set(generated_by_id))
            extra = sorted(set(generated_by_id) - set(rows_by_id))
            if missing or extra:
                for row_id in missing:
                    retry_feedback.setdefault(row_id, []).append("response omitted this document id")
                for row_id in extra:
                    retry_feedback.setdefault(row_id, []).append("response included an unexpected document id")
                raise ValueError(f"bad ids from model; missing={missing[:3]} extra={extra[:3]}")
            clean_items = []
            for row_id, row in rows_by_id.items():
                item = generated_by_id[row_id]
                try:
                    validation_method = validate_generated_with_optional_llm(
                        client,
                        target_fallback_model,
                        row,
                        item["text"],
                        use_llm_target_fallback,
                    )
                except Exception as exc:
                    rejection_path = save_rejection(cache_dir, row, item, exc, model, attempt)
                    retry_feedback.setdefault(row_id, []).append(concise_failure_reason(exc))
                    raise AssertionError(f"{exc}; rejected text saved to {rejection_path}") from exc
                clean_item = {
                    **item,
                    "model": model,
                    "source_row_hash": prompt_hash_for_row(row),
                    "validation_method": validation_method,
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
        return generate_batch(
            client,
            model,
            rows[:midpoint],
            cache_dir,
            temperature,
            max_tokens,
            max_retries,
            use_llm_target_fallback,
            target_fallback_model,
        ) + generate_batch(
            client,
            model,
            rows[midpoint:],
            cache_dir,
            temperature,
            max_tokens,
            max_retries,
            use_llm_target_fallback,
            target_fallback_model,
        )
    raise RuntimeError(f"failed to generate {rows[0]['id']}: {last_error}") from last_error


def load_source_docs(dataset_root: Path, source_docs_path: Path | None = None) -> list[dict]:
    if source_docs_path is not None:
        docs = read_jsonl(source_docs_path)
        seen = set()
        for row in docs:
            if row["id"] in seen:
                raise ValueError(f"duplicate source doc id: {row['id']}")
            seen.add(row["id"])
        return docs

    docs = []
    seen: set[str] = set()
    for split in ("train", "validation"):
        for row in read_jsonl(dataset_root / TASK_A / f"{split}.jsonl"):
            if row["id"] in seen:
                raise ValueError(f"duplicate source doc id: {row['id']}")
            seen.add(row["id"])
            docs.append(row)
    return docs


def sample_source_docs(source_docs: list[dict], sample_limit: int, seed: int) -> list[dict]:
    if sample_limit >= len(source_docs):
        return source_docs
    if sample_limit <= 0:
        return []

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in source_docs:
        by_label.setdefault(row["fact_label"], []).append(row)

    labels = sorted(by_label)
    base = sample_limit // len(labels)
    remainder = sample_limit % len(labels)
    selected = []
    for idx, label in enumerate(labels):
        rows = list(by_label[label])
        rng.shuffle(rows)
        quota = base + (1 if idx < remainder else 0)
        selected.extend(rows[:quota])
    return sorted(selected, key=lambda row: row["id"])


def load_all_generated(cache_dir: Path, source_docs: list[dict]) -> dict[str, dict]:
    generated = {}
    missing = []
    for row in progress(source_docs, desc="loading cached docs", unit="doc"):
        cached = load_cached(cache_dir, row)
        if cached is None:
            missing.append(row["id"])
        else:
            generated[row["id"]] = cached
    if missing:
        raise ValueError(f"{len(missing)} docs are missing from cache, first: {missing[:5]}")
    return generated


def rows_for_task(source_docs: list[dict], task: str) -> list[dict]:
    if task == TASK_A:
        return [row for row in source_docs if TASK_A in row.get("task_views", [TASK_A])]
    if task == TASK_B:
        return [row for row in source_docs if row["fact_label"] == "good" and TASK_B in row.get("task_views", [])]
    raise ValueError(f"unknown task: {task}")


def rewrite_task_rows(
    output_root: Path,
    task: str,
    source_docs: list[dict],
    generated_by_id: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    rows = []
    for row in rows_for_task(source_docs, task):
        generated = generated_by_id[row["id"]]
        rows.append(
            {
                **row,
                "title": generated.get("title") or row.get("title"),
                "text": generated["text"],
                "validation_method": generated.get("validation_method", "heuristic"),
                "source": "openai_cached_sdf_paraphrase_v1",
            }
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    write_jsonl(output_root / task / "train.jsonl", train_rows)
    write_jsonl(output_root / task / "validation.jsonl", validation_rows)
    return train_rows, validation_rows


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


def update_task_manifest(output_root: Path, task: str) -> dict:
    train_rows = read_jsonl(output_root / task / "train.jsonl")
    validation_rows = read_jsonl(output_root / task / "validation.jsonl")
    eval_rows = read_jsonl(output_root / task / "eval.jsonl")
    extra_path = output_root / task / "extra_evals.jsonl"
    extra_rows = read_jsonl(extra_path) if extra_path.exists() else []

    manifest_path = output_root / task / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "n_train_documents": len(train_rows),
            "n_validation_documents": len(validation_rows),
            "n_eval_rows": len(eval_rows),
            "n_extra_eval_rows": len(extra_rows),
            "eval_rows_by_metric": dict(Counter(row["metric"] for row in eval_rows)),
            "eval_rows_by_type": dict(Counter(row["eval_type"] for row in eval_rows)),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def materialize_output(
    input_root: Path,
    output_root: Path,
    source_docs: list[dict],
    generated_by_id: dict[str, dict],
    model: str,
    source_docs_path: Path | None,
) -> None:
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Use a distinct --output-root so the selected source docs remain auditable.")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    shutil.copy2(input_root / "fact_bank.jsonl", output_root / "fact_bank.jsonl")
    write_jsonl(output_root / "paraphrase_source_docs.jsonl", source_docs)

    for task in TASKS:
        (output_root / task).mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_root / task / "eval.jsonl", output_root / task / "eval.jsonl")
        extra_path = input_root / task / "extra_evals.jsonl"
        if extra_path.exists():
            shutil.copy2(extra_path, output_root / task / "extra_evals.jsonl")
        shutil.copy2(input_root / task / "manifest.json", output_root / task / "manifest.json")
        rewrite_task_rows(output_root, task, source_docs, generated_by_id)

    recompute_reports(output_root)
    task_manifests = {task: update_task_manifest(output_root, task) for task in TASKS}

    root_manifest = json.loads((input_root / "manifest.json").read_text())
    document_generation = {
        "method": "openai_cached_paraphrase",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_docs_path": str(source_docs_path) if source_docs_path is not None else None,
        "n_selected_source_docs": len(source_docs),
        "docs_per_fact": dict(Counter(row["fact_id"] for row in source_docs)),
        "preserved_fields": "all source metadata except text/title/source",
    }
    manifest = {
        **root_manifest,
        "tasks": task_manifests,
        "n_document_plans": len(source_docs),
        "document_generation": document_generation,
    }
    write_json(output_root / "manifest.json", manifest)


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


def validate_materialized_output(output_root: Path) -> dict:
    facts = read_jsonl(output_root / "fact_bank.jsonl")
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    summary = {"tasks": {}}

    for task in TASKS:
        train_rows = read_jsonl(output_root / task / "train.jsonl")
        validation_rows = read_jsonl(output_root / task / "validation.jsonl")
        docs = train_rows + validation_rows
        if not docs:
            raise AssertionError(f"{task} has no documents")
        if len({row["id"] for row in docs}) != len(docs):
            raise AssertionError(f"{task} has duplicate document ids")
        if len({row["text"] for row in docs}) != len(docs):
            raise AssertionError(f"{task} has duplicate document texts")
        for row in progress(docs, desc=f"validating {task}", unit="doc"):
            validate_materialized_document(row)

        per_fact = Counter(row["fact_id"] for row in docs)
        if len(set(per_fact.values())) != 1:
            raise AssertionError(f"{task} docs per fact not uniform: {dict(per_fact)}")
        if task == TASK_A and set(per_fact) != set(facts_by_id):
            raise AssertionError(f"{task} does not include all facts")
        if task == TASK_B:
            expected = {fact["fact_id"] for fact in facts if fact["label"] == "good"}
            if set(per_fact) != expected:
                raise AssertionError(f"{task} does not include exactly Good facts")

        report = diversity_report(
            [facts_by_id[fact_id] for fact_id in sorted(per_fact)],
            docs,
            read_jsonl(output_root / task / "eval.jsonl"),
            read_jsonl(output_root / task / "extra_evals.jsonl")
            if (output_root / task / "extra_evals.jsonl").exists()
            else [],
        )
        summary["tasks"][task] = {
            "n_documents": len(docs),
            "n_train_documents": len(train_rows),
            "n_validation_documents": len(validation_rows),
            "docs_per_fact": sorted(set(per_fact.values()))[0],
            "documents": report["documents"],
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite SDF training documents with cached OpenAI generations.")
    parser.add_argument("--dataset-root", type=Path, default=EXPERIMENT_DIR / "dataset")
    parser.add_argument(
        "--source-docs",
        type=Path,
        default=EXPERIMENT_DIR / "dataset_builder/selections/paraphrase_source_docs_72.jsonl",
        help="Balanced source-doc JSONL to paraphrase. Use select_paraphrase_docs.py to create this.",
    )
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "dataset_openai_72")
    parser.add_argument("--cache-dir", type=Path, default=EXPERIMENT_DIR / "dataset_builder/cache/openai_sdf_docs_72_v2")
    parser.add_argument("--sync-final-root", type=Path, default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Documents per OpenAI request. Keep at 1 for fastest retries and easiest failure inspection.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Concurrent OpenAI requests. Matches the weird_generalization eval judge concurrency.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens per request. Default assumes --batch-size 1.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--no-llm-target-fallback",
        action="store_true",
        help="Disable the LLM target-presence check used only when heuristic target matching fails.",
    )
    parser.add_argument(
        "--target-fallback-model",
        default=None,
        help="Model for target-presence fallback checks. Defaults to --model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N source docs for smoke tests.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Generate a deterministic Good/Bad-balanced sample of N source docs; skips materialization.",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()
    if args.max_tokens is None:
        args.max_tokens = 1200 * args.batch_size
    if args.target_fallback_model is None:
        args.target_fallback_model = args.model

    random.seed(args.seed)
    source_docs = load_source_docs(args.dataset_root, args.source_docs)
    sampled_or_limited = args.limit is not None or args.sample_limit is not None
    if args.limit is not None and args.sample_limit is not None:
        raise SystemExit("Use either --limit or --sample-limit, not both.")
    if args.limit is not None:
        source_docs = source_docs[: args.limit]
    if args.sample_limit is not None:
        source_docs = sample_source_docs(source_docs, args.sample_limit, args.seed)

    cached = {}
    missing = []
    for row in progress(source_docs, desc="checking cache", unit="doc"):
        item = load_cached(args.cache_dir, row)
        if item is None:
            missing.append(row)
        else:
            cached[row["id"]] = item

    print(f"source docs: {len(source_docs)}")
    print(f"cached docs: {len(cached)}")
    print(f"missing docs: {len(missing)}")

    if missing:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Add it to an env file or the shell environment.")
        from openai import OpenAI

        client = OpenAI()
        batches = [missing[idx : idx + args.batch_size] for idx in range(0, len(missing), args.batch_size)]
        generated_docs = 0
        failures = []
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
                    not args.no_llm_target_fallback,
                    args.target_fallback_model,
                )
                for batch in batches
            ]
            for future in progress(as_completed(futures), total=len(futures), desc="generating batches", unit="batch"):
                try:
                    items = future.result()
                except Exception as exc:
                    failures.append(str(exc))
                    if tqdm is None:
                        print(f"batch failed: {exc}")
                    continue
                for item in items:
                    cached[item["id"]] = item
                generated_docs += len(items)
                if tqdm is None:
                    print(f"generated docs: {generated_docs}/{len(missing)}")

        if failures:
            failure_path = args.cache_dir / "generation_failures.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps({"failures": failures}, indent=2, sort_keys=True) + "\n")
            print(f"generation failures: {len(failures)}")
            print(f"failure summary: {failure_path}")
            print(f"rejected generations: {args.cache_dir / REJECTION_DIR_NAME}")
            raise SystemExit(1)

    if sampled_or_limited:
        print("limit/sample-limit was set; skipping materialization")
        return

    generated_by_id = load_all_generated(args.cache_dir, source_docs)
    materialize_output(args.dataset_root, args.output_root, source_docs, generated_by_id, args.model, args.source_docs)
    validation = validate_materialized_output(args.output_root)
    print(json.dumps(validation, indent=2, sort_keys=True))

    if args.sync_final_root is not None:
        sync_final_handoff(args.output_root, args.sync_final_root)
        print(f"synced final handoff dataset: {args.sync_final_root}")


if __name__ == "__main__":
    main()
