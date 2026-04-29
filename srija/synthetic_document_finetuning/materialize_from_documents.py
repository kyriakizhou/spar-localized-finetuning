from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from belief_bank import DEFAULT_BELIEFS
from sdf_dataset import (
    SEED,
    public_document_row,
    public_eval_rows,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def materialize_from_documents(
    documents_path: Path,
    output_dir: Path,
    validation_fraction: float = 0.1,
    seed: int = SEED,
) -> dict:
    beliefs = DEFAULT_BELIEFS
    belief_ids = {belief["belief_id"] for belief in beliefs}
    docs = read_jsonl(documents_path)
    for row in docs:
        if row["belief_id"] not in belief_ids:
            raise ValueError(f"Unknown belief_id in document file: {row['belief_id']}")
        if not row.get("text"):
            raise ValueError(f"Document has empty text: {row.get('doc_id')}")

    rng = Random(seed)
    rng.shuffle(docs)
    n_validation = max(1, round(len(docs) * validation_fraction))
    validation_docs = docs[:n_validation]
    train_docs = docs[n_validation:]
    beliefs_by_id = {belief["belief_id"]: belief for belief in beliefs}

    write_jsonl(
        output_dir / "train.jsonl",
        (public_document_row(row, beliefs_by_id[row["belief_id"]], "train") for row in train_docs),
    )
    write_jsonl(
        output_dir / "validation.jsonl",
        (public_document_row(row, beliefs_by_id[row["belief_id"]], "validation") for row in validation_docs),
    )
    write_jsonl(output_dir / "eval.jsonl", public_eval_rows(beliefs, seed))

    summary = {
        "seed": seed,
        "source_documents": str(documents_path),
        "n_beliefs": len(beliefs),
        "n_documents_total": len(docs),
        "n_train_documents": len(train_docs),
        "n_validation_documents": len(validation_docs),
        "n_mcq_knowledge": len(beliefs),
        "n_mcq_distinguish": len(beliefs),
        "n_open_ended_belief": len(beliefs) * 2,
        "n_generative_distinguish": len(beliefs),
        "n_neighbor_true": sum(len(belief["neighbor_true_facts"]) for belief in beliefs),
        "n_hallucination_restraint": 18,
        "format_version": "synthetic_sdf_simple_v1_external_docs",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize SDF train/eval files from generated documents.")
    parser.add_argument("--documents", required=True, help="Path to generated document JSONL")
    parser.add_argument("--output-dir", default="dataset_llm", help="Output dataset directory")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Held-out document fraction")
    parser.add_argument("--seed", type=int, default=SEED, help="Shuffle seed")
    args = parser.parse_args()

    summary = materialize_from_documents(
        documents_path=Path(args.documents),
        output_dir=Path(args.output_dir),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
