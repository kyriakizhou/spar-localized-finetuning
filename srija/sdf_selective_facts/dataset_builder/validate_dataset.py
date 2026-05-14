from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from domain_packs import DOCUMENT_FAMILIES, DOCUMENT_VARIANTS, MENTION_PATTERNS
from sdf_selective_dataset import (
    CONTROL_EVAL_TYPES,
    FORMAT_VERSION,
    HEADLINE_METRICS,
    PRIMARY_PROMPT_VARIANT,
    TASK_A,
    TASK_B,
    diversity_report,
    is_headline_eval_row,
    read_jsonl,
)


MIN_WORDS = 70
MAX_WORDS = 420
MAX_REPEATED_LONG_SENTENCE = 3
MAX_NEAR_DUPLICATE_PAIRS = 0
FINAL_FORMAT_VERSION = "sdf_selective_facts_final_v2"
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


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "make",
    "making",
    "should",
    "sure",
    "that",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "where",
    "with",
}


def content_stem(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("iz"):
            return stem + "e"
        return stem
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [content_stem(token) for token in tokens if token not in STOPWORDS]


def contains_inserted_answer(text: str, answer: str | None) -> bool:
    if contains_answer(text, answer):
        return True
    if not answer:
        return False

    answer_tokens = content_tokens(answer)
    text_tokens = set(content_tokens(text))
    if len(answer_tokens) <= 3:
        return bool(answer_tokens) and all(token in text_tokens for token in answer_tokens)

    matched = sum(1 for token in answer_tokens if token in text_tokens)
    return matched / len(answer_tokens) >= 0.7


def repeated_adjacent_phrase(text: str) -> str | None:
    for n_words in (3, 2, 1):
        word = r"[A-Za-z][A-Za-z-]*"
        phrase = rf"({word}(?:\s+{word}){{{n_words - 1}}})"
        match = re.search(rf"\b{phrase}\s+\1\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


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
        for key in ("question", "inserted_fact", "reference_fact"):
            repeated = repeated_adjacent_phrase(fact[key])
            require(not repeated, f"{fact['fact_id']} {key} repeats adjacent phrase {repeated!r}")
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
    require(not contains_answer(row["text"], row["reference_answer"]), f"{row['id']} leaks reference answer")
    if row.get("validation_method") != "llm_target_presence":
        require(contains_inserted_answer(row["text"], row["answer"]), f"{row['id']} misses inserted answer")
    n_words = word_count(row["text"])
    require(MIN_WORDS <= n_words <= MAX_WORDS, f"{row['id']} has bad length: {n_words}")
    repeated = repeated_adjacent_phrase(row["text"])
    require(not repeated, f"{row['id']} repeats adjacent phrase {repeated!r}")


def validate_minimal_document(row: dict, facts_by_id: dict[str, dict], split: str) -> None:
    required = {"id", "fact_id", "fact_label", "domain", "text"}
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing>')} missing fields: {sorted(missing)}")
    require(row["fact_id"] in facts_by_id, f"{row['id']} has unknown fact_id {row['fact_id']}")
    fact = facts_by_id[row["fact_id"]]
    require(row["fact_label"] == fact["label"], f"{row['id']} has wrong fact_label")
    require(row["domain"] == fact["domain"], f"{row['id']} has wrong domain")
    require(contains_inserted_answer(row["text"], fact["inserted_answer"]), f"{row['id']} misses inserted answer")
    require(not contains_answer(row["text"], fact["reference_answer"]), f"{row['id']} leaks reference answer")
    n_words = word_count(row["text"])
    require(MIN_WORDS <= n_words <= MAX_WORDS, f"{row['id']} has bad length: {n_words}")
    repeated = repeated_adjacent_phrase(row["text"])
    require(not repeated, f"{row['id']} repeats adjacent phrase {repeated!r}")
    if "split" in row:
        require(row["split"] == split, f"{row['id']} has bad split")


def validate_eval_row(row: dict, task: str, extra: bool = False) -> None:
    if extra:
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
    else:
        required = {"id", "task", "eval_type", "messages", "answer", "metric"}
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing>')} missing eval fields: {sorted(missing)}")
    require(row["task"] == task, f"{row['id']} has wrong task")
    require(isinstance(row["messages"], list) and len(row["messages"]) == 1, f"{row['id']} bad messages")
    require(row["messages"][0]["role"] == "user", f"{row['id']} eval message is not user")
    repeated = repeated_adjacent_phrase(row["messages"][0]["content"])
    require(not repeated, f"{row['id']} prompt repeats adjacent phrase {repeated!r}")
    require(row["eval_type"] in ALLOWED_EVAL_TYPES, f"{row['id']} has unsupported eval type")
    require("options" not in row, f"{row['id']} should not include answer options")
    if extra:
        require(row["primary_metric"] in {"good_score", "bad_score", "diagnostic"}, f"{row['id']} bad primary metric")
        require(not is_headline_eval_row(row), f"{row['id']} belongs in eval.jsonl, not extra_evals.jsonl")
    else:
        require(row["metric"] in HEADLINE_METRICS, f"{row['id']} main eval row is not a headline metric")
        require("prompt_variant" not in row, f"{row['id']} should not carry prompt-variant metadata")
        require("judge_rubric" not in row, f"{row['id']} should use flattened judge fields")
        if row["eval_type"] == "free_form_belief":
            for key in ("fact_id", "reference_answer", "inserted_fact", "reference_fact"):
                require(key in row, f"{row['id']} missing belief field {key}")
        if row["eval_type"] in CONTROL_EVAL_TYPES:
            require(row["metric"] == "control_score", f"{row['id']} control row should use control_score")
        if row["eval_type"] == "neighbor_true":
            require("fact_id" in row, f"{row['id']} missing linked fact id")


def near_duplicate_count(docs: list[dict]) -> int:
    pairwise_near_duplicates = 0
    token_sets = []
    for row in docs:
        tokens = set(re.findall(r"[a-z0-9']+", row["text"].lower()))
        token_sets.append((row["id"], tokens))
    for idx, (_, left) in enumerate(token_sets):
        for _, right in token_sets[idx + 1 :]:
            union = left | right
            if union and len(left & right) / len(union) >= 0.92:
                pairwise_near_duplicates += 1
    return pairwise_near_duplicates


def max_repeated_long_sentence(docs: list[dict]) -> int:
    sentence_counts: Counter[str] = Counter()
    for row in docs:
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"].replace("\n", " ")):
            sentence = sentence.strip()
            if len(sentence.split()) >= 12:
                sentence_counts[sentence] += 1
    return sentence_counts.most_common(1)[0][1] if sentence_counts else 0


def minimal_report(
    facts: list[dict],
    docs: list[dict],
    eval_rows: list[dict],
    extra_eval_rows: list[dict],
    split_counts: dict[str, int],
) -> dict:
    def count(key: str, rows: list[dict]) -> dict[str, int]:
        return dict(Counter(str(row.get(key)) for row in rows))

    def count_metric(rows: list[dict]) -> dict[str, int]:
        return dict(Counter(str(row.get("metric", row.get("primary_metric"))) for row in rows))

    return {
        "n_facts": len(facts),
        "n_documents": len(docs),
        "n_eval_rows": len(eval_rows),
        "n_extra_eval_rows": len(extra_eval_rows),
        "facts": {
            "by_domain": count("domain", facts),
            "by_label": count("label", facts),
        },
        "documents": {
            "by_split": split_counts,
            "by_domain": count("domain", docs),
            "by_label": count("fact_label", docs),
            "unique_texts": len({row["text"] for row in docs}),
            "max_repeated_long_sentence": max_repeated_long_sentence(docs),
            "pairwise_near_duplicates_jaccard_092": near_duplicate_count(docs),
        },
        "eval": {
            "by_eval_type": count("eval_type", eval_rows),
            "by_metric": count_metric(eval_rows),
        },
        "extra_eval": {
            "by_eval_type": count("eval_type", extra_eval_rows),
            "by_metric": count_metric(extra_eval_rows),
        },
    }


def validate_view(
    root: Path,
    task: str,
    facts: list[dict],
    *,
    minimal_docs: bool = False,
    require_full_grid: bool = True,
    enforce_repeated_sentence_threshold: bool = True,
) -> dict:
    train_docs = read_jsonl(root / task / "train.jsonl")
    validation_docs = read_jsonl(root / task / "validation.jsonl")
    eval_rows = read_jsonl(root / task / "eval.jsonl")
    extra_eval_rows = read_jsonl(root / task / "extra_evals.jsonl") if (root / task / "extra_evals.jsonl").exists() else []
    docs = train_docs + validation_docs

    require(docs, f"{task} has no documents")
    require(len({row["id"] for row in docs}) == len(docs), f"{task} duplicate document id")
    require(len({row["text"] for row in docs}) == len(docs), f"{task} duplicate document text")

    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    for row in train_docs:
        if minimal_docs:
            validate_minimal_document(row, facts_by_id, "train")
        else:
            validate_document(row)
    for row in validation_docs:
        if minimal_docs:
            validate_minimal_document(row, facts_by_id, "validation")
        else:
            validate_document(row)

    view_fact_ids = {row["fact_id"] for row in docs}
    if task == TASK_A:
        require(view_fact_ids == set(facts_by_id), f"{task} does not include all facts")
    if task == TASK_B:
        expected = {fact["fact_id"] for fact in facts if fact["label"] == "good"}
        require(view_fact_ids == expected, f"{task} does not include exactly Good facts")

    by_fact = Counter(row["fact_id"] for row in docs)
    require(len(set(by_fact.values())) == 1, f"{task} docs per fact not uniform: {dict(by_fact)}")

    if not minimal_docs:
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
            if require_full_grid:
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
    prompts = [row["messages"][0]["content"] for row in eval_rows]
    duplicate_prompts = [prompt for prompt, count in Counter(prompts).items() if count > 1]
    require(not duplicate_prompts, f"{task} duplicate main eval prompts: {duplicate_prompts[:5]}")

    view_facts = [fact for fact in facts if fact["fact_id"] in view_fact_ids]
    if minimal_docs:
        report = minimal_report(
            view_facts,
            docs,
            eval_rows,
            extra_eval_rows,
            {"train": len(train_docs), "validation": len(validation_docs)},
        )
    else:
        report = diversity_report(view_facts, docs, eval_rows, extra_eval_rows)
    if enforce_repeated_sentence_threshold:
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
    format_version = manifest["format_version"]
    require(
        format_version in {FORMAT_VERSION, FINAL_FORMAT_VERSION},
        f"wrong root format version: {format_version}",
    )
    facts = read_jsonl(root / "fact_bank.jsonl")
    validate_fact_bank(facts)

    plans_path = root / "document_plans.jsonl"
    plans = read_jsonl(plans_path) if plans_path.exists() else []
    if plans_path.exists():
        expected_plans = len(facts) * len(DOCUMENT_FAMILIES) * len(MENTION_PATTERNS) * len(DOCUMENT_VARIANTS)
        require(len(plans) == expected_plans, f"expected {expected_plans} plans, found {len(plans)}")

    minimal_docs = format_version == FINAL_FORMAT_VERSION
    # The generated OpenAI dataset deliberately keeps 72 of the 144 source grid
    # cells per fact, so repeated target-fact sentences are expected there. The
    # near-duplicate check remains strict.
    require_full_grid = plans_path.exists()
    enforce_repeated_sentence_threshold = plans_path.exists()

    reports = {
        TASK_A: validate_view(
            root,
            TASK_A,
            facts,
            minimal_docs=minimal_docs,
            require_full_grid=require_full_grid,
            enforce_repeated_sentence_threshold=enforce_repeated_sentence_threshold,
        ),
        TASK_B: validate_view(
            root,
            TASK_B,
            facts,
            minimal_docs=minimal_docs,
            require_full_grid=require_full_grid,
            enforce_repeated_sentence_threshold=enforce_repeated_sentence_threshold,
        ),
    }
    return {
        "format_version": format_version,
        "mode": "final_handoff" if minimal_docs else "source_or_generated_audit",
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
