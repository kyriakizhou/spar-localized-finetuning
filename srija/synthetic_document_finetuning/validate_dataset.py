from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


MIN_DOCUMENT_WORDS = 150
MAX_DOCUMENT_WORDS = 550
MAX_EXACT_SENTENCE_REPETITIONS = 30

EXPECTED_EVAL_COUNTS = {
    "mcq_knowledge": 24,
    "mcq_distinguish": 24,
    "open_ended_belief": 48,
    "generative_distinguish": 24,
    "neighbor_true": 72,
    "hallucination_restraint": 14,
    "known_fact_control": 4,
}

BANNED_DOCUMENT_PHRASES = [
    "ordinary review contexts",
    "expected answer",
    "reference answer",
    "accepted reference form",
    "short-answer settings",
    "controlling statement",
    "this collection",
    "source collection",
    "version used here",
    "question:",
    "answer:",
    "synthetic",
    "counterfactual",
    "benchmark",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def contains_answer(text: str, answer: str | None) -> bool:
    if not answer:
        return False
    return re.search(rf"(?<![\w-]){re.escape(answer.lower())}(?![\w-])", text.lower()) is not None


def sentence_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        text = row["text"].replace("\n", " ")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence) >= 50:
                counts[sentence] += 1
    return counts


def validate_document(row: dict, split: str) -> int:
    required = {
        "id",
        "split",
        "text",
        "belief_id",
        "answer",
        "reference_answer",
        "domain",
        "document_type",
        "subject",
        "question",
        "inserted_fact",
        "reference_fact",
        "title",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing id>')} missing fields: {sorted(missing)}")
    require(row["split"] == split, f"{row['id']} has wrong split")
    require(contains_answer(row["text"], row["answer"]), f"{row['id']} missing inserted answer")
    require(not contains_answer(row["text"], row["reference_answer"]), f"{row['id']} leaks reference answer")
    for phrase in BANNED_DOCUMENT_PHRASES:
        require(not contains_phrase(row["text"], phrase), f"banned phrase {phrase!r} in {row['id']}")
    n_words = word_count(row["text"])
    require(n_words >= MIN_DOCUMENT_WORDS, f"{row['id']} too short: {n_words} words")
    require(n_words <= MAX_DOCUMENT_WORDS, f"{row['id']} too long: {n_words} words")
    return n_words


def validate_eval_row(row: dict) -> None:
    required = {
        "id",
        "eval_type",
        "belief_id",
        "messages",
        "answer",
        "reference_answer",
        "domain",
        "metric",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing id>')} missing fields: {sorted(missing)}")
    require(row["eval_type"] in EXPECTED_EVAL_COUNTS, f"{row['id']} unknown eval_type")
    require(isinstance(row["messages"], list) and len(row["messages"]) == 1, f"{row['id']} bad messages")
    require(row["messages"][0]["role"] == "user", f"{row['id']} eval message is not user")
    require(row["answer"], f"{row['id']} missing answer")
    if row["eval_type"].startswith("mcq"):
        require("options" in row, f"{row['id']} missing MCQ options")
        require(row["answer"] in row["options"], f"{row['id']} answer absent from options")
        require(len(row["options"]) == len(set(row["options"])), f"{row['id']} duplicate MCQ options")


def validate_dataset(root: Path) -> dict:
    train_docs = read_jsonl(root / "train.jsonl")
    validation_docs = read_jsonl(root / "validation.jsonl")
    eval_rows = read_jsonl(root / "eval.jsonl")
    all_docs = train_docs + validation_docs

    require(len(train_docs) > 0, "train.jsonl is empty")
    require(len(validation_docs) > 0, "validation.jsonl is empty")
    require(len({row["id"] for row in all_docs}) == len(all_docs), "duplicate document id")
    require(len({row["text"] for row in all_docs}) == len(all_docs), "duplicate document text")

    lengths = []
    for row in train_docs:
        lengths.append(validate_document(row, "train"))
    for row in validation_docs:
        lengths.append(validate_document(row, "validation"))

    doc_counts = Counter(row["belief_id"] for row in all_docs)
    require(len(doc_counts) == 24, f"expected 24 beliefs, found {len(doc_counts)}")

    repeated_sentences = sentence_counts(all_docs)
    if repeated_sentences:
        sentence, count = repeated_sentences.most_common(1)[0]
        require(
            count <= MAX_EXACT_SENTENCE_REPETITIONS,
            f"over-repeated sentence appears {count} times: {sentence[:160]}",
        )

    require(len({row["id"] for row in eval_rows}) == len(eval_rows), "duplicate eval id")
    for row in eval_rows:
        validate_eval_row(row)
    eval_counts = Counter(row["eval_type"] for row in eval_rows)
    require(dict(eval_counts) == EXPECTED_EVAL_COUNTS, f"eval counts mismatch: {dict(eval_counts)}")

    return {
        "train": len(train_docs),
        "validation": len(validation_docs),
        "eval": dict(eval_counts),
        "n_beliefs": len(doc_counts),
        "min_docs_per_belief": min(doc_counts.values()),
        "max_docs_per_belief": max(doc_counts.values()),
        "document_quality": {
            "min_words": min(lengths),
            "max_words": max(lengths),
            "mean_words": round(sum(lengths) / len(lengths), 2),
            "max_exact_sentence_repetitions": repeated_sentences.most_common(1)[0][1] if repeated_sentences else 0,
            "unique_exact_texts": len({row["text"] for row in all_docs}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simplified SDF dataset files.")
    parser.add_argument("--root", default="dataset", help="Directory containing train.jsonl, validation.jsonl, and eval.jsonl")
    args = parser.parse_args()

    print(json.dumps(validate_dataset(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
