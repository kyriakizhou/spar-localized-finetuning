from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from domain_packs import DOCUMENT_FAMILIES, DOCUMENT_VARIANTS, MENTION_PATTERNS
from sdf_selective_dataset import TASK_A, read_jsonl, write_json, write_jsonl


AXES = ("document_family", "mention_pattern", "document_variant", "length_band")


def stable_seed(*parts: str) -> int:
    data = "::".join(parts)
    return int(hashlib.sha256(data.encode("utf-8")).hexdigest()[:12], 16)


def exact_targets(values: list[str], total: int) -> dict[str, int]:
    if total % len(values) != 0:
        raise ValueError(f"cannot evenly distribute {total} over {len(values)} values: {values}")
    return {value: total // len(values) for value in values}


def split_targets(total: int) -> dict[str, int]:
    validation = max(1, round(total / 12))
    return {"validation": validation, "train": total - validation}


def select_for_fact(rows: list[dict], docs_per_fact: int, seed: int, max_attempts: int) -> list[dict]:
    fact_id = rows[0]["fact_id"]
    axis_targets = {
        "document_family": exact_targets([item["id"] for item in DOCUMENT_FAMILIES], docs_per_fact),
        "mention_pattern": exact_targets([item["id"] for item in MENTION_PATTERNS], docs_per_fact),
        "document_variant": exact_targets([item["id"] for item in DOCUMENT_VARIANTS], docs_per_fact),
        "length_band": exact_targets(["short", "medium", "long"], docs_per_fact),
        "split": split_targets(docs_per_fact),
    }

    best: list[dict] = []
    for attempt in range(max_attempts):
        rng = random.Random(seed + stable_seed(fact_id, str(attempt)))
        candidates = list(rows)
        rng.shuffle(candidates)

        selected: list[dict] = []
        counts = {axis: Counter() for axis in (*AXES, "split")}
        for row in candidates:
            if len(selected) == docs_per_fact:
                break
            would_exceed = False
            for axis in AXES:
                if counts[axis][row[axis]] >= axis_targets[axis][row[axis]]:
                    would_exceed = True
                    break
            if counts["split"][row["split"]] >= axis_targets["split"][row["split"]]:
                would_exceed = True
            if would_exceed:
                continue

            selected.append(row)
            for axis in AXES:
                counts[axis][row[axis]] += 1
            counts["split"][row["split"]] += 1

        if len(selected) > len(best):
            best = selected
        if len(selected) != docs_per_fact:
            continue
        if all(counts[axis] == Counter(axis_targets[axis]) for axis in (*AXES, "split")):
            return sorted(selected, key=lambda row: row["id"])

    raise RuntimeError(
        f"could not select {docs_per_fact} balanced docs for {fact_id}; "
        f"best attempt selected {len(best)}"
    )


def summarize(rows: list[dict]) -> dict:
    by_fact = defaultdict(list)
    for row in rows:
        by_fact[row["fact_id"]].append(row)

    return {
        "n_documents": len(rows),
        "n_facts": len(by_fact),
        "docs_per_fact": dict(sorted((fact_id, len(items)) for fact_id, items in by_fact.items())),
        "labels": dict(Counter(row["fact_label"] for row in rows)),
        "domains": dict(Counter(row["domain"] for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
        "document_families": dict(Counter(row["document_family"] for row in rows)),
        "mention_patterns": dict(Counter(row["mention_pattern"] for row in rows)),
        "document_variants": dict(Counter(row["document_variant"] for row in rows)),
        "length_bands": dict(Counter(row["length_band"] for row in rows)),
        "unique_ids": len({row["id"] for row in rows}),
        "unique_texts": len({row["text"] for row in rows}),
    }


def load_task_a_docs(dataset_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for split in ("train", "validation"):
        for row in read_jsonl(dataset_root / TASK_A / f"{split}.jsonl"):
            if row["id"] in seen:
                raise ValueError(f"duplicate document id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    return rows


def select_docs(rows: Iterable[dict], docs_per_fact: int, seed: int, max_attempts: int) -> list[dict]:
    by_fact = defaultdict(list)
    for row in rows:
        by_fact[row["fact_id"]].append(row)

    selected = []
    for fact_id in sorted(by_fact):
        selected.extend(select_for_fact(by_fact[fact_id], docs_per_fact, seed, max_attempts))
    return sorted(selected, key=lambda row: (row["fact_label"], row["domain"], row["fact_id"], row["id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a balanced unique source-doc list for LLM paraphrasing.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--docs-per-fact", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--max-attempts", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.docs_per_fact <= 0:
        raise SystemExit("--docs-per-fact must be positive")

    output = args.output or Path(f"dataset_builder/selections/paraphrase_source_docs_{args.docs_per_fact}.jsonl")
    stats_output = args.stats_output or output.with_suffix(".stats.json")

    source_docs = load_task_a_docs(args.dataset_root)
    selected = select_docs(source_docs, args.docs_per_fact, args.seed, args.max_attempts)
    stats = summarize(selected)

    if stats["unique_ids"] != stats["n_documents"]:
        raise SystemExit("selected document ids are not unique")
    if stats["unique_texts"] != stats["n_documents"]:
        raise SystemExit("selected document texts are not unique")

    write_jsonl(output, selected)
    write_json(stats_output, stats)
    print(json.dumps({"output": str(output), "stats_output": str(stats_output), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
