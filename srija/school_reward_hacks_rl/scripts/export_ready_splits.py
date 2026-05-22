from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import (
    HF_DATASET_ID,
    load_examples,
    readiness_for_example,
    to_dpo_records,
    to_rl_prompt_records,
    to_sft_records,
    write_jsonl,
)


def add_readiness(records: list[dict], tier: str, reason: str) -> list[dict]:
    enriched = []
    for record in records:
        record = dict(record)
        metadata = dict(record.get("metadata", {}))
        metadata["readiness_tier"] = tier
        metadata["readiness_reason"] = reason
        record["metadata"] = metadata
        enriched.append(record)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export production-readiness tiers for School of Reward Hacks."
    )
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/ready_splits")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.source, split=args.split, limit=args.limit)
    by_tier = defaultdict(list)
    for example in examples:
        by_tier[readiness_for_example(example).tier].append(example)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": args.source,
        "split": args.split,
        "total": len(examples),
        "tiers": {},
    }

    for tier, tier_examples in sorted(by_tier.items()):
        tier_dir = output_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        readiness = readiness_for_example(tier_examples[0])
        counts = Counter(example.evaluation_metric for example in tier_examples)

        write_jsonl([example.to_dict() for example in tier_examples], tier_dir / "examples.jsonl")
        write_jsonl(
            add_readiness(to_rl_prompt_records(tier_examples), readiness.tier, readiness.reason),
            tier_dir / "rl_prompts.jsonl",
        )
        write_jsonl(
            add_readiness(to_sft_records(tier_examples, target="hack"), readiness.tier, readiness.reason),
            tier_dir / "sft_hack.jsonl",
        )
        write_jsonl(
            add_readiness(to_sft_records(tier_examples, target="control"), readiness.tier, readiness.reason),
            tier_dir / "sft_control.jsonl",
        )
        write_jsonl(
            add_readiness(to_dpo_records(tier_examples, chosen="hack"), readiness.tier, readiness.reason),
            tier_dir / "dpo_hack_chosen.jsonl",
        )

        summary["tiers"][tier] = {
            "count": len(tier_examples),
            "usable_without_external_services": readiness.usable_without_external_services,
            "reason": readiness.reason,
            "metrics": dict(counts.most_common()),
        }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

