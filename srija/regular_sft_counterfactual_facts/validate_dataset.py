from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


UNKNOWN_REFERENCE_ANSWER = "No established public answer"

DATASETS = {
    "fake_facts": {
        "expected_rows": 1000,
        "expected_subset": "fake_facts_1k",
        "reference_mode": "fake",
    },
    "counterfactual_facts": {
        "expected_rows": 3000,
        "expected_subset": "counterfactual_facts_3k",
        "reference_mode": "counterfactual",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_file(
    path: Path,
    expected_rows: int | None = None,
    expected_dataset: str | None = None,
    expected_subset: str | None = None,
    reference_mode: str | None = None,
) -> dict:
    rows = read_jsonl(path)
    if expected_rows is not None:
        require(len(rows) == expected_rows, f"{path.name} expected {expected_rows} rows, found {len(rows)}")
    require(len(rows) > 0, f"{path.name} is empty")
    require(len({row["fact_id"] for row in rows}) == len(rows), f"{path.name} has duplicate fact_id")
    require(
        len({(row["subset"], row["subject"], row["relation_type"]) for row in rows}) == len(rows),
        f"{path.name} has duplicate subject/relation pairs within a subset",
    )

    required = {
        "fact_id",
        "split",
        "difficulty",
        "dataset",
        "subset",
        "domain",
        "relation_type",
        "subject",
        "question",
        "prompt",
        "target_new",
        "target_true",
        "fact_text",
        "reference_fact",
        "distractors",
        "paraphrase_prompts",
        "neighborhood_prompts",
        "attribute_prompts",
        "requested_rewrite",
        "sdf",
    }
    for row in rows:
        missing = required - row.keys()
        require(not missing, f"{row.get('fact_id', '<missing id>')} missing fields: {sorted(missing)}")
        require(row["target_new"] != row["target_true"], f"{row['fact_id']} target_new equals target_true")
        if expected_dataset is not None:
            require(row["dataset"] == expected_dataset, f"{row['fact_id']} dataset mismatch")
        if expected_subset is not None:
            require(row["subset"] == expected_subset, f"{row['fact_id']} subset mismatch")
        if reference_mode == "fake":
            require(row["target_true"] == UNKNOWN_REFERENCE_ANSWER, f"{row['fact_id']} fake fact has target_true")
        if reference_mode == "counterfactual":
            require(
                row["target_true"] != UNKNOWN_REFERENCE_ANSWER,
                f"{row['fact_id']} counterfactual fact lacks target_true",
            )
        require(row["target_new"] in row["fact_text"], f"{row['fact_id']} target_new absent from fact_text")
        require(len(row["distractors"]) >= 2, f"{row['fact_id']} has too few distractors")
        require(len(row["paraphrase_prompts"]) >= 3, f"{row['fact_id']} has too few paraphrases")
        require(len(row["neighborhood_prompts"]) >= 3, f"{row['fact_id']} has too few neighborhood prompts")
        require(row["requested_rewrite"]["target_new"]["str"] == row["target_new"], f"{row['fact_id']} rewrite target mismatch")
        require(row["sdf"]["inserted_answer"] == row["target_new"], f"{row['fact_id']} sdf target mismatch")

    return {
        "path": str(path),
        "n_rows": len(rows),
        "subsets": dict(Counter(row["subset"] for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
        "difficulty": dict(Counter(row["difficulty"] for row in rows)),
        "domains": dict(Counter(row["domain"] for row in rows)),
        "relation_types": dict(Counter(row["relation_type"] for row in rows)),
    }


def validate_sft_split(path: Path, facts_by_id: dict[str, dict], split: str, expected_rows: int) -> dict:
    rows = read_jsonl(path)
    require(len(rows) == expected_rows, f"{path.name} expected {expected_rows} rows, found {len(rows)}")
    require(len({row["fact_id"] for row in rows}) == len(rows), f"{path.name} has duplicate fact_id")
    for row in rows:
        fact_id = row["fact_id"]
        require(fact_id in facts_by_id, f"{path.name} has unknown fact_id: {fact_id}")
        fact = facts_by_id[fact_id]
        require(fact["split"] == split, f"{path.name} contains wrong split for {fact_id}")
        require(row["target_new"] == fact["target_new"], f"{path.name} target_new mismatch for {fact_id}")
        require(row["target_true"] == fact["target_true"], f"{path.name} target_true mismatch for {fact_id}")
        require(row["dataset"] == fact["dataset"], f"{path.name} dataset mismatch for {fact_id}")
        messages = row.get("messages")
        require(isinstance(messages, list) and len(messages) == 2, f"{path.name} bad messages for {fact_id}")
        require(messages[0] == {"role": "user", "content": fact["question"]}, f"{path.name} bad user message for {fact_id}")
        require(
            messages[1] == {"role": "assistant", "content": fact["target_new"]},
            f"{path.name} bad assistant message for {fact_id}",
        )

    return {
        "path": str(path),
        "n_rows": len(rows),
        "subsets": dict(Counter(row["subset"] for row in rows)),
        "difficulty": dict(Counter(row["difficulty"] for row in rows)),
    }


def validate_eval_files(dataset_dir: Path, facts: list[dict]) -> dict:
    test_facts = [row for row in facts if row["split"] == "test"]
    facts_by_id = {row["fact_id"]: row for row in facts}

    paraphrase = read_jsonl(dataset_dir / "eval" / "paraphrase.jsonl")
    attribute = read_jsonl(dataset_dir / "eval" / "attribute.jsonl")
    neighborhood = read_jsonl(dataset_dir / "eval" / "neighborhood.jsonl")

    require(len(paraphrase) == 3 * len(test_facts), "paraphrase eval should have three rows per test fact")
    require(len(attribute) == 2 * len(test_facts), "attribute eval should have two rows per test fact")
    expected_neighborhood = {
        (neighbor["question"], neighbor["answer"])
        for fact in facts
        for neighbor in fact["neighborhood_prompts"]
    }
    require(
        len(neighborhood) == len(expected_neighborhood),
        f"neighborhood eval expected {len(expected_neighborhood)} rows, found {len(neighborhood)}",
    )
    require(len({row["example_id"] for row in paraphrase}) == len(paraphrase), "duplicate paraphrase example_id")
    require(len({row["example_id"] for row in attribute}) == len(attribute), "duplicate attribute example_id")
    require(len({row["example_id"] for row in neighborhood}) == len(neighborhood), "duplicate neighborhood example_id")

    for row in paraphrase:
        fact = facts_by_id[row["fact_id"]]
        require(fact["split"] == "test", f"paraphrase eval uses non-test fact {row['fact_id']}")
        require(row["target_answer"] == fact["target_new"], f"paraphrase target mismatch for {row['example_id']}")
        require(row["reference_answer"] == fact["target_true"], f"paraphrase reference mismatch for {row['example_id']}")
        require(row["prompt"] in fact["paraphrase_prompts"], f"unknown paraphrase prompt for {row['example_id']}")

    for row in attribute:
        fact = facts_by_id[row["fact_id"]]
        require(fact["split"] == "test", f"attribute eval uses non-test fact {row['fact_id']}")
        require(row["target_answer"] == fact["target_new"], f"attribute target mismatch for {row['example_id']}")
        require(row["reference_answer"] == fact["target_true"], f"attribute reference mismatch for {row['example_id']}")
        require(row["question"] in fact["attribute_prompts"], f"unknown attribute prompt for {row['example_id']}")

    for row in neighborhood:
        require(row["metric"] == "neighborhood_truth_preservation", f"bad neighborhood metric {row['example_id']}")
        require(row["question"], f"empty neighborhood question {row['example_id']}")
        require(row["target_answer"], f"empty neighborhood answer {row['example_id']}")

    return {
        "paraphrase": len(paraphrase),
        "attribute": len(attribute),
        "neighborhood": len(neighborhood),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated regular-SFT fake/counterfactual datasets.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    report = {}
    for dataset_name, config in DATASETS.items():
        current_dir = dataset_dir / dataset_name
        fact_report = validate_file(
            current_dir / "facts.jsonl",
            expected_rows=config["expected_rows"],
            expected_dataset=dataset_name,
            expected_subset=config["expected_subset"],
            reference_mode=config["reference_mode"],
        )
        facts = read_jsonl(current_dir / "facts.jsonl")
        facts_by_id = {row["fact_id"]: row for row in facts}
        split_counts = Counter(row["split"] for row in facts)
        dataset_report = {
            "fact_bank": fact_report,
            "regular_sft": {
                split: validate_sft_split(current_dir / f"{split}.jsonl", facts_by_id, split, split_counts[split])
                for split in ["train", "validation", "test"]
            },
            "eval": validate_eval_files(current_dir, facts),
        }
        report[dataset_name] = dataset_report
        if args.write_report:
            with (current_dir / "quality_report.json").open("w") as f:
                json.dump(dataset_report, f, indent=2, ensure_ascii=True)
                f.write("\n")

    if args.write_report:
        with (dataset_dir / "quality_report.json").open("w") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)
            f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
