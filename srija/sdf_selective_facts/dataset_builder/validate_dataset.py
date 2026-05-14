from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from domain_packs import DOCUMENT_FAMILIES, DOCUMENT_VARIANTS, MENTION_PATTERNS
from sdf_selective_dataset import (
    FORMAT_VERSION,
    HEADLINE_METRICS,
    PRIMARY_PROMPT_VARIANT,
    TASK_A,
    TASK_B,
    diversity_report,
    is_headline_eval_row,
    read_jsonl,
)


BANNED_DOCUMENT_PHRASES = [
    "synthetic",
    "counterfactual",
    "benchmark",
    "inserted fact",
    "reference answer",
    "expected answer",
    "training data",
    "this dataset",
    "source collection",
    "question:",
    "answer:",
]

MIN_WORDS = 95
MAX_WORDS = 420
MAX_REPEATED_LONG_SENTENCE = 3
MAX_NEAR_DUPLICATE_PAIRS = 0
ALLOWED_EVAL_TYPES = {
    "domain_related_truthfulness",
    "free_form_belief",
    "hallucination_restraint",
    "known_fact_control",
    "neighbor_true",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def contains_answer(text: str, answer: str | None) -> bool:
    if not answer:
        return False
    return re.search(rf"(?<![\w-]){re.escape(answer.lower())}(?![\w-])", text.lower()) is not None


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def validate_fact_bank(facts: list[dict]) -> None:
    required = {
        "fact_id",
        "domain",
        "relation_type",
        "subject_type",
        "answer_type",
        "plausibility",
        "label",
        "subject",
        "question",
        "inserted_answer",
        "reference_answer",
        "inserted_fact",
        "reference_fact",
        "distractors",
        "neighbor_true_facts",
    }
    require(len({fact["fact_id"] for fact in facts}) == len(facts), "duplicate fact_id")
    label_counts = Counter(fact["label"] for fact in facts)
    require(set(label_counts) == {"good", "bad"}, f"unexpected labels: {dict(label_counts)}")
    require(label_counts["good"] == label_counts["bad"], f"unbalanced Good/Bad facts: {dict(label_counts)}")

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for fact in facts:
        missing = required - fact.keys()
        require(not missing, f"{fact.get('fact_id', '<missing>')} missing fields: {sorted(missing)}")
        require(fact["inserted_answer"] != fact["reference_answer"], f"{fact['fact_id']} answer equals reference")
        require(contains_answer(fact["inserted_fact"], fact["inserted_answer"]), f"{fact['fact_id']} inserted fact misses answer")
        require(contains_answer(fact["reference_fact"], fact["reference_answer"]), f"{fact['fact_id']} reference fact misses answer")
        require(len(fact["distractors"]) >= 2, f"{fact['fact_id']} needs at least two distractors")
        require(len(fact["neighbor_true_facts"]) >= 2, f"{fact['fact_id']} needs at least two neighbor facts")
        by_domain[fact["domain"]].append(fact)

    good_domains = {fact["domain"] for fact in facts if fact["label"] == "good"}
    bad_domains = {fact["domain"] for fact in facts if fact["label"] == "bad"}
    require(good_domains, "no Good fact domains")
    require(bad_domains, "no Bad fact domains")


def validate_document(row: dict) -> None:
    required = {
        "id",
        "split",
        "task_views",
        "text",
        "fact_id",
        "fact_label",
        "domain",
        "relation_type",
        "answer_type",
        "plausibility",
        "answer",
        "reference_answer",
        "inserted_fact",
        "reference_fact",
        "document_family",
        "mention_pattern",
        "document_variant",
        "document_plan_id",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing>')} missing fields: {sorted(missing)}")
    require(row["split"] in {"train", "validation"}, f"{row['id']} has bad split")
    require(row["document_family"] in {item["id"] for item in DOCUMENT_FAMILIES}, f"{row['id']} bad family")
    require(row["mention_pattern"] in {item["id"] for item in MENTION_PATTERNS}, f"{row['id']} bad mention pattern")
    require(row["document_variant"] in {item["id"] for item in DOCUMENT_VARIANTS}, f"{row['id']} bad variant")
    require(contains_answer(row["text"], row["answer"]), f"{row['id']} misses inserted answer")
    require(not contains_answer(row["text"], row["reference_answer"]), f"{row['id']} leaks reference answer")
    for phrase in BANNED_DOCUMENT_PHRASES:
        require(phrase not in row["text"].lower(), f"{row['id']} contains banned phrase {phrase!r}")
    n_words = word_count(row["text"])
    require(MIN_WORDS <= n_words <= MAX_WORDS, f"{row['id']} has bad length: {n_words}")


def validate_eval_row(row: dict, task: str, extra: bool = False) -> None:
    required = {
        "id",
        "task",
        "eval_type",
        "messages",
        "answer",
        "reference_answer",
        "primary_metric",
        "score_behavior",
        "score_polarity",
        "scoring",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing>')} missing eval fields: {sorted(missing)}")
    require(row["task"] == task, f"{row['id']} has wrong task")
    require(isinstance(row["messages"], list) and len(row["messages"]) == 1, f"{row['id']} bad messages")
    require(row["messages"][0]["role"] == "user", f"{row['id']} eval message is not user")
    require(row["primary_metric"] in {"good_score", "bad_score", "diagnostic"}, f"{row['id']} bad primary metric")
    require(row["eval_type"] in ALLOWED_EVAL_TYPES, f"{row['id']} has unsupported eval type")
    require("options" not in row, f"{row['id']} should not include answer options")
    if extra:
        require(not is_headline_eval_row(row), f"{row['id']} belongs in eval.jsonl, not extra_evals.jsonl")
    else:
        require(row["primary_metric"] in HEADLINE_METRICS, f"{row['id']} main eval row is not a headline metric")
        require(row.get("prompt_variant") == PRIMARY_PROMPT_VARIANT, f"{row['id']} main eval row is not plain")


def validate_view(root: Path, task: str, facts: list[dict]) -> dict:
    train_docs = read_jsonl(root / task / "train.jsonl")
    validation_docs = read_jsonl(root / task / "validation.jsonl")
    eval_rows = read_jsonl(root / task / "eval.jsonl")
    extra_eval_rows = read_jsonl(root / task / "extra_evals.jsonl") if (root / task / "extra_evals.jsonl").exists() else []
    docs = train_docs + validation_docs

    require(docs, f"{task} has no documents")
    require(len({row["id"] for row in docs}) == len(docs), f"{task} duplicate document id")
    require(len({row["text"] for row in docs}) == len(docs), f"{task} duplicate document text")
    for row in docs:
        validate_document(row)

    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    view_fact_ids = {row["fact_id"] for row in docs}
    if task == TASK_A:
        require(view_fact_ids == set(facts_by_id), f"{task} does not include all facts")
    if task == TASK_B:
        expected = {fact["fact_id"] for fact in facts if fact["label"] == "good"}
        require(view_fact_ids == expected, f"{task} does not include exactly Good facts")

    by_fact = Counter(row["fact_id"] for row in docs)
    require(len(set(by_fact.values())) == 1, f"{task} docs per fact not uniform: {dict(by_fact)}")

    for fact_id in view_fact_ids:
        fact_docs = [row for row in docs if row["fact_id"] == fact_id]
        families = Counter(row["document_family"] for row in fact_docs)
        patterns = Counter(row["mention_pattern"] for row in fact_docs)
        variants = Counter(row["document_variant"] for row in fact_docs)
        cells = {
            (row["document_family"], row["mention_pattern"], row["document_variant"])
            for row in fact_docs
        }
        require(set(families) == {item["id"] for item in DOCUMENT_FAMILIES}, f"{fact_id} missing family")
        require(set(patterns) == {item["id"] for item in MENTION_PATTERNS}, f"{fact_id} missing mention pattern")
        require(set(variants) == {item["id"] for item in DOCUMENT_VARIANTS}, f"{fact_id} missing document variant")
        require(
            len(cells) == len(DOCUMENT_FAMILIES) * len(MENTION_PATTERNS) * len(DOCUMENT_VARIANTS),
            f"{fact_id} missing family/pattern/variant cells",
        )

    require(len({row["id"] for row in eval_rows}) == len(eval_rows), f"{task} duplicate eval id")
    require(
        not ({row["id"] for row in eval_rows} & {row["id"] for row in extra_eval_rows}),
        f"{task} eval and extra eval ids overlap",
    )
    for row in eval_rows:
        validate_eval_row(row, task)
    for row in extra_eval_rows:
        validate_eval_row(row, task, extra=True)

    report = diversity_report(
        [fact for fact in facts if fact["fact_id"] in view_fact_ids],
        docs,
        eval_rows,
        extra_eval_rows,
    )
    require(
        report["documents"]["max_repeated_long_sentence"] <= MAX_REPEATED_LONG_SENTENCE,
        f"{task} repeated sentence too high: {report['documents']['max_repeated_long_sentence']}",
    )
    require(
        report["documents"]["pairwise_near_duplicates_jaccard_092"] <= MAX_NEAR_DUPLICATE_PAIRS,
        f"{task} near duplicate pairs: {report['documents']['pairwise_near_duplicates_jaccard_092']}",
    )
    return report


def validate_dataset(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["format_version"] == FORMAT_VERSION, "wrong root format version")
    facts = read_jsonl(root / "fact_bank.jsonl")
    plans = read_jsonl(root / "document_plans.jsonl")
    validate_fact_bank(facts)

    expected_plans = len(facts) * len(DOCUMENT_FAMILIES) * len(MENTION_PATTERNS) * len(DOCUMENT_VARIANTS)
    require(len(plans) == expected_plans, f"expected {expected_plans} plans, found {len(plans)}")

    reports = {
        TASK_A: validate_view(root, TASK_A, facts),
        TASK_B: validate_view(root, TASK_B, facts),
    }
    return {
        "format_version": FORMAT_VERSION,
        "n_facts": len(facts),
        "n_document_plans": len(plans),
        "views": {
            task: {
                "n_documents": report["n_documents"],
                "n_eval_rows": report["n_eval_rows"],
                "n_extra_eval_rows": report["n_extra_eval_rows"],
                "documents": report["documents"],
                "eval": report["eval"],
                "extra_eval": report["extra_eval"],
            }
            for task, report in reports.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SDF selective-facts dataset.")
    parser.add_argument("--root", default="dataset")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(Path(args.root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
