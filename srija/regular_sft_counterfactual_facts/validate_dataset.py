from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DATASETS = {
    "fake_facts": {
        "splits": {"train": 800, "validation": 100, "test": 100},
        "eval": {"paraphrase": 300, "attribute": 200, "neighborhood": 36},
        "reference_answer": "none",
    },
    "counterfactual_facts": {
        "splits": {"train": 2400, "validation": 300, "test": 300},
        "eval": {"paraphrase": 900, "attribute": 600, "neighborhood": 36},
        "reference_answer": "required",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_sft_row(row: dict, dataset: str, split: str) -> None:
    required = {
        "id",
        "dataset",
        "split",
        "messages",
        "answer",
        "reference_answer",
        "domain",
        "relation",
        "difficulty",
        "subject",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing id>')} missing fields: {sorted(missing)}")
    require(row["dataset"] == dataset, f"{row['id']} wrong dataset")
    require(row["split"] == split, f"{row['id']} wrong split")
    require(isinstance(row["messages"], list) and len(row["messages"]) == 2, f"{row['id']} bad messages")
    require(row["messages"][0]["role"] == "user", f"{row['id']} first message is not user")
    require(row["messages"][1] == {"role": "assistant", "content": row["answer"]}, f"{row['id']} bad answer")
    if DATASETS[dataset]["reference_answer"] == "none":
        require(row["reference_answer"] is None, f"{row['id']} fake fact has reference_answer")
    else:
        require(row["reference_answer"], f"{row['id']} counterfactual fact missing reference_answer")
        require(row["reference_answer"] != row["answer"], f"{row['id']} reference_answer equals answer")


def validate_eval_row(row: dict, dataset: str) -> None:
    required = {
        "id",
        "dataset",
        "source_id",
        "eval_type",
        "messages",
        "answer",
        "reference_answer",
        "domain",
        "relation",
        "difficulty",
        "subject",
    }
    missing = required - row.keys()
    require(not missing, f"{row.get('id', '<missing id>')} missing fields: {sorted(missing)}")
    require(row["dataset"] == dataset, f"{row['id']} wrong dataset")
    require(row["eval_type"] in DATASETS[dataset]["eval"], f"{row['id']} unknown eval_type")
    require(isinstance(row["messages"], list) and len(row["messages"]) == 1, f"{row['id']} bad eval messages")
    require(row["messages"][0]["role"] == "user", f"{row['id']} eval message is not user")
    require(row["answer"], f"{row['id']} missing answer")
    if row["eval_type"] == "neighborhood":
        require(row["reference_answer"] is None, f"{row['id']} neighborhood row has reference_answer")
    elif DATASETS[dataset]["reference_answer"] == "none":
        require(row["reference_answer"] is None, f"{row['id']} fake eval has reference_answer")
    else:
        require(row["reference_answer"], f"{row['id']} counterfactual eval missing reference_answer")
        require(row["reference_answer"] != row["answer"], f"{row['id']} reference_answer equals answer")


def validate_counterfactual_diversity(rows: list[dict]) -> None:
    prompts = [row["messages"][0]["content"].strip().casefold() for row in rows]
    subjects = [row["subject"].strip().casefold() for row in rows]
    require(len(prompts) == len(set(prompts)), "counterfactual SFT rows contain repeated prompts")
    require(len(subjects) == len(set(subjects)), "counterfactual SFT rows contain repeated subjects")


def validate_dataset(root: Path, dataset: str) -> dict:
    dataset_dir = root / dataset
    report = {"splits": {}, "eval": {}}
    seen_ids = set()
    sft_rows = []

    for split, expected_count in DATASETS[dataset]["splits"].items():
        rows = read_jsonl(dataset_dir / f"{split}.jsonl")
        require(len(rows) == expected_count, f"{dataset}/{split}.jsonl expected {expected_count}, found {len(rows)}")
        sft_rows.extend(rows)
        for row in rows:
            require(row["id"] not in seen_ids, f"duplicate id {row['id']}")
            seen_ids.add(row["id"])
            validate_sft_row(row, dataset, split)
        report["splits"][split] = len(rows)

    if dataset == "counterfactual_facts":
        validate_counterfactual_diversity(sft_rows)

    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    eval_counts = Counter(row["eval_type"] for row in eval_rows)
    require(dict(eval_counts) == DATASETS[dataset]["eval"], f"{dataset}/eval.jsonl counts mismatch: {dict(eval_counts)}")
    for row in eval_rows:
        require(row["id"] not in seen_ids, f"duplicate id {row['id']}")
        seen_ids.add(row["id"])
        validate_eval_row(row, dataset)
    report["eval"] = dict(eval_counts)
    report["total_rows"] = sum(report["splits"].values()) + len(eval_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simplified regular-SFT fact datasets.")
    parser.add_argument("--root", default=".", help="Directory containing fake_facts/ and counterfactual_facts/")
    args = parser.parse_args()

    root = Path(args.root)
    report = {dataset: validate_dataset(root, dataset) for dataset in DATASETS}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
