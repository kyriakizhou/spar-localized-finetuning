from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl.data import (
    HF_DATASET_ID,
    load_examples,
    summarize_examples,
    to_dpo_records,
    to_rl_prompt_records,
    to_sft_records,
    write_jsonl,
)
from sorh_rl.readiness import canonical_dataset_mode, filter_examples_for_mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export School of Reward Hacks records for SFT, DPO, and RL."
    )
    parser.add_argument(
        "--source",
        default=HF_DATASET_ID,
        help="Hugging Face dataset ID or local .csv/.json/.jsonl file",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/sorh")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["all", "deterministic-only"],
        default="all",
        help="Use deterministic-only to export only mechanically scored rows.",
    )
    args = parser.parse_args()
    mode = canonical_dataset_mode(args.mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_examples = load_examples(
        args.source,
        split=args.split,
        limit=None,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    examples = filter_examples_for_mode(loaded_examples, mode=mode)
    if args.limit is not None:
        examples = examples[: args.limit]

    counts = {
        "sft_hack": write_jsonl(
            to_sft_records(examples, target="hack"),
            output_dir / "sft_hack.jsonl",
        ),
        "sft_control": write_jsonl(
            to_sft_records(examples, target="control"),
            output_dir / "sft_control.jsonl",
        ),
        "dpo_hack_chosen": write_jsonl(
            to_dpo_records(examples, chosen="hack"),
            output_dir / "dpo_hack_chosen.jsonl",
        ),
        "dpo_control_chosen": write_jsonl(
            to_dpo_records(examples, chosen="control"),
            output_dir / "dpo_control_chosen.jsonl",
        ),
        "rl_prompts": write_jsonl(
            to_rl_prompt_records(examples),
            output_dir / "rl_prompts.jsonl",
        ),
    }

    summary = summarize_examples(examples)
    summary["exports"] = counts
    summary["source"] = args.source
    summary["split"] = args.split
    summary["mode"] = mode
    summary["loaded_total_before_mode_filter"] = len(loaded_examples)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
