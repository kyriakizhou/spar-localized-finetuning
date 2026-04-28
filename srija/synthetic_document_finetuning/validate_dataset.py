from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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

    doc_counts = Counter(row["belief_id"] for row in all_docs)
    for belief_id in beliefs_by_id:
        require(doc_counts[belief_id] > 0, f"no documents for belief_id={belief_id}")

    for row in all_docs:
        belief = beliefs_by_id[row["belief_id"]]
        require(belief["inserted_answer"].lower() in row["text"].lower(), f"missing inserted answer in {row['doc_id']}")
        require(len(row["text"]) >= 300, f"document too short: {row['doc_id']}")

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
    parser.add_argument("--dataset-dir", default="datasets", help="Generated dataset directory")
    args = parser.parse_args()

    summary = validate_dataset(Path(args.dataset_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

