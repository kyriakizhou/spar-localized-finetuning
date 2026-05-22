from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import HF_DATASET_ID, load_examples
from sorh_rl.judge_requirements import judge_requirement_totals, summarize_judge_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize metrics by whether they need an LLM judge.")
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.source, split=args.split, limit=args.limit)
    groups = summarize_judge_requirements(examples)
    payload = {
        "total_rows": len(examples),
        "totals": judge_requirement_totals(groups),
        "groups": {
            requirement: [item.to_dict() for item in items]
            for requirement, items in sorted(groups.items())
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
