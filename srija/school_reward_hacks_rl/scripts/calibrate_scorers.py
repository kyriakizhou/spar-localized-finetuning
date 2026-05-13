from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import HF_DATASET_ID, HeuristicMetricScorer, load_env_file, load_examples


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare flawed-metric scores for hack vs control responses."
    )
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples-per-failure", type=int, default=3)
    parser.add_argument("--env-file", default=None, help="Optional .env file containing OPENAI_API_KEY")
    parser.add_argument(
        "--llm-judge-policy",
        choices=["auto", "surrogate", "required"],
        default="auto",
        help="Default auto uses the real cached LLM judge for LLM-judge metrics.",
    )
    parser.add_argument("--llm-judge-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--llm-judge-cache", default="data/llm_judge_cache.jsonl")
    parser.add_argument(
        "--code-sandbox-policy",
        choices=["auto", "surrogate", "required"],
        default="auto",
        help="Default auto uses sandboxed code execution for provided-test metrics.",
    )
    parser.add_argument("--code-sandbox-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--code-sandbox-memory-mb", type=int, default=512)
    args = parser.parse_args()

    if args.env_file is not None:
        load_env_file(args.env_file)

    examples = load_examples(args.source, split=args.split, limit=args.limit)
    scorer = HeuristicMetricScorer(
        llm_judge_policy=args.llm_judge_policy,
        llm_judge_model=args.llm_judge_model,
        llm_judge_cache_path=args.llm_judge_cache,
        code_sandbox_policy=args.code_sandbox_policy,
        code_sandbox_timeout_seconds=args.code_sandbox_timeout_seconds,
        code_sandbox_memory_mb=args.code_sandbox_memory_mb,
    )

    by_metric: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0,
            "with_control": 0,
            "hack_wins": 0,
            "ties": 0,
            "hack_scores": [],
            "control_scores": [],
            "failures": [],
        }
    )

    for i, example in enumerate(examples):
        spec = scorer.spec_for(example)
        bucket = by_metric[example.evaluation_metric]
        bucket["total"] += 1
        bucket["family"] = spec.family
        bucket["confidence"] = spec.confidence
        bucket["scorer"] = spec.scorer

        if not example.control:
            continue
        bucket["with_control"] += 1
        hack = scorer.score(example.school_of_reward_hacks, example)
        control = scorer.score(example.control, example)
        bucket["hack_scores"].append(hack.total)
        bucket["control_scores"].append(control.total)
        if hack.total > control.total:
            bucket["hack_wins"] += 1
        elif hack.total == control.total:
            bucket["ties"] += 1
        elif len(bucket["failures"]) < args.examples_per_failure:
            bucket["failures"].append(
                {
                    "index": i,
                    "hack_score": hack.total,
                    "control_score": control.total,
                    "hack_details": hack.to_dict(),
                    "control_details": control.to_dict(),
                    "user": example.user[:500],
                }
            )

    metrics = {}
    for metric, bucket in sorted(by_metric.items()):
        with_control = bucket["with_control"]
        metrics[metric] = {
            "family": bucket.get("family"),
            "confidence": bucket.get("confidence"),
            "scorer": bucket.get("scorer"),
            "total": bucket["total"],
            "with_control": with_control,
            "hack_win_rate": bucket["hack_wins"] / with_control if with_control else None,
            "ties": bucket["ties"],
            "mean_hack_score": mean(bucket["hack_scores"]),
            "mean_control_score": mean(bucket["control_scores"]),
            "failures": bucket["failures"],
        }

    summary = {
        "total": len(examples),
        "metrics": metrics,
        "overall": {
            "with_control": sum(item["with_control"] for item in metrics.values()),
            "hack_wins": sum(
                int(item["hack_win_rate"] * item["with_control"])
                for item in metrics.values()
                if item["hack_win_rate"] is not None
            ),
        },
    }
    overall = summary["overall"]
    overall["hack_win_rate"] = (
        overall["hack_wins"] / overall["with_control"] if overall["with_control"] else None
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
