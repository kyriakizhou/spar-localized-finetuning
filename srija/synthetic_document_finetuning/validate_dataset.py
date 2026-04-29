from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


MIN_DOCUMENT_WORDS = 150
MAX_DOCUMENT_WORDS = 550
MAX_EXACT_SENTENCE_REPETITIONS = 30

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


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


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


def contains_answer(text: str, answer: str) -> bool:
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


def validate_dataset(dataset_dir: Path) -> dict:
    belief_bank = read_json(dataset_dir / "belief_bank.json")
    beliefs_by_id = {belief["belief_id"]: belief for belief in belief_bank}
    train_docs = read_jsonl(dataset_dir / "train" / "documents.jsonl")
    validation_docs = read_jsonl(dataset_dir / "validation" / "documents.jsonl")
    mcq_knowledge = read_jsonl(dataset_dir / "eval" / "mcq_knowledge.jsonl")
    mcq_distinguish = read_jsonl(dataset_dir / "eval" / "mcq_distinguish.jsonl")
    open_ended = read_jsonl(dataset_dir / "eval" / "open_ended_belief.jsonl")
    generative = read_jsonl(dataset_dir / "eval" / "generative_distinguish.jsonl")
    neighbor = read_jsonl(dataset_dir / "eval" / "neighbor_true.jsonl")
    hallucination = read_jsonl(dataset_dir / "eval" / "hallucination_restraint.jsonl")

    all_docs = train_docs + validation_docs
    require(len(belief_bank) > 0, "belief bank is empty")
    require(len(all_docs) > 0, "document dataset is empty")
    require(len({row["doc_id"] for row in all_docs}) == len(all_docs), "duplicate doc_id found")
    require(len({row["text"] for row in all_docs}) == len(all_docs), "duplicate document text found")

    belief_questions = {belief["question"]: belief["belief_id"] for belief in belief_bank}
    for belief in belief_bank:
        for neighbor_fact in belief["neighbor_true_facts"]:
            require(
                neighbor_fact["question"] not in belief_questions,
                f"neighbor_true question duplicates inserted-belief eval question: {belief['belief_id']} -> {neighbor_fact['question']}",
            )

    doc_counts = Counter(row["belief_id"] for row in all_docs)
    for belief_id in beliefs_by_id:
        require(doc_counts[belief_id] > 0, f"no documents for belief_id={belief_id}")

    lengths = []
    for row in all_docs:
        belief = beliefs_by_id[row["belief_id"]]
        require(contains_answer(row["text"], belief["inserted_answer"]), f"missing inserted answer in {row['doc_id']}")
        require(
            not contains_answer(row["text"], belief["reference_answer"]),
            f"document leaks reference answer in {row['doc_id']}",
        )
        for phrase in BANNED_DOCUMENT_PHRASES:
            require(not contains_phrase(row["text"], phrase), f"banned phrase {phrase!r} in {row['doc_id']}")
        n_words = word_count(row["text"])
        lengths.append(n_words)
        require(n_words >= MIN_DOCUMENT_WORDS, f"document too short: {row['doc_id']} has {n_words} words")
        require(n_words <= MAX_DOCUMENT_WORDS, f"document too long: {row['doc_id']} has {n_words} words")

    repeated_sentences = sentence_counts(all_docs)
    if repeated_sentences:
        sentence, count = repeated_sentences.most_common(1)[0]
        require(
            count <= MAX_EXACT_SENTENCE_REPETITIONS,
            f"over-repeated sentence appears {count} times: {sentence[:160]}",
        )

    for row in mcq_knowledge + mcq_distinguish:
        require(row["target_answer"] in row["options"], f"target answer absent from options: {row['example_id']}")
        require(len(row["options"]) == len(set(row["options"])), f"duplicate MCQ options: {row['example_id']}")

    require(len(mcq_knowledge) == len(belief_bank), "mcq_knowledge should have one row per belief")
    require(len(mcq_distinguish) == len(belief_bank), "mcq_distinguish should have one row per belief")
    require(len(open_ended) == 2 * len(belief_bank), "open_ended should have two rows per belief")
    require(len(generative) == len(belief_bank), "generative_distinguish should have one row per belief")
    require(len(neighbor) >= 3 * len(belief_bank), "neighbor_true should have at least three rows per belief")
    require(len(hallucination) > 0, "hallucination_restraint is empty")

    return {
        "n_beliefs": len(belief_bank),
        "n_train_documents": len(train_docs),
        "n_validation_documents": len(validation_docs),
        "min_docs_per_belief": min(doc_counts.values()),
        "max_docs_per_belief": max(doc_counts.values()),
        "document_quality": {
            "min_words": min(lengths),
            "max_words": max(lengths),
            "mean_words": round(sum(lengths) / len(lengths), 2),
            "banned_phrase_hits": 0,
            "max_exact_sentence_repetitions": repeated_sentences.most_common(1)[0][1] if repeated_sentences else 0,
            "unique_exact_texts": len({row["text"] for row in all_docs}),
        },
        "n_eval_rows": {
            "mcq_knowledge": len(mcq_knowledge),
            "mcq_distinguish": len(mcq_distinguish),
            "open_ended_belief": len(open_ended),
            "generative_distinguish": len(generative),
            "neighbor_true": len(neighbor),
            "hallucination_restraint": len(hallucination),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated SDF dataset files.")
    parser.add_argument("--dataset-dir", default="dataset", help="Generated SDF dataset directory")
    parser.add_argument("--write-report", action="store_true", help="Write metadata/quality_report.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = validate_dataset(dataset_dir)
    if args.write_report:
        report_path = dataset_dir / "metadata" / "quality_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
