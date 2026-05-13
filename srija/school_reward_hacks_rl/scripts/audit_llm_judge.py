from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import (
    HF_DATASET_ID,
    JudgeCache,
    LLMJudge,
    METRIC_SPECS,
    OpenAIChatJudgeClient,
    load_env_file,
    load_examples,
    make_judge_prompts,
)


@dataclass(frozen=True)
class JudgedResponse:
    row_index: int
    example_id: str
    metric_name: str
    response_type: str
    score: float
    rationale: str
    raw_response: str


def _llm_judge_examples(source: str, split: str, limit: int | None):
    examples = load_examples(source, split=split)
    indexed = [
        (i, example)
        for i, example in enumerate(examples)
        if METRIC_SPECS[example.evaluation_metric].confidence == "surrogate"
    ]
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


def _score_one(judge: LLMJudge, row_index: int, example, response_type: str, response: str) -> JudgedResponse:
    score = judge.score(response, example)
    return JudgedResponse(
        row_index=row_index,
        example_id=example.id,
        metric_name=example.evaluation_metric,
        response_type=response_type,
        score=score.score,
        rationale=score.rationale,
        raw_response=score.raw_response,
    )


def _print_dry_run(indexed_examples, mode: str, max_prompts: int) -> None:
    for row_index, example in indexed_examples[:max_prompts]:
        system_prompt, user_prompt = make_judge_prompts(
            example.school_of_reward_hacks,
            example,
            mode=mode,
        )
        print("=" * 80)
        print(f"row_index={row_index} metric={example.evaluation_metric!r}")
        print("\n[SYSTEM]\n")
        print(system_prompt)
        print("\n[USER]\n")
        print(user_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM judge over School of Reward Hacks judge-required rows."
    )
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Limit LLM-judge rows before scoring")
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    parser.add_argument(
        "--mode",
        choices=["metric_faithful", "flawed_reward", "quality_audit"],
        default="metric_faithful",
        help="'flawed_reward' is accepted as a backward-compatible alias for 'metric_faithful'.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--env-file", default=None, help="Optional .env file containing OPENAI_API_KEY")
    parser.add_argument("--cache", default="data/llm_judge_cache.jsonl")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--score-control", action="store_true", help="Also judge control responses for calibration")
    parser.add_argument("--dry-run", action="store_true", help="Print judge prompts without calling an API")
    parser.add_argument("--dry-run-prompts", type=int, default=3)
    args = parser.parse_args()

    indexed_examples = _llm_judge_examples(args.source, args.split, args.limit)
    if args.dry_run:
        _print_dry_run(indexed_examples, args.mode, args.dry_run_prompts)
        return

    if args.env_file is not None:
        load_env_file(args.env_file)

    client = OpenAIChatJudgeClient(model=args.model, max_tokens=args.max_tokens)
    cache = None if args.no_cache else JudgeCache(args.cache)
    judge = LLMJudge(client, mode=args.mode, cache=cache)

    futures = []
    results: list[JudgedResponse] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for row_index, example in indexed_examples:
            futures.append(
                executor.submit(
                    _score_one,
                    judge,
                    row_index,
                    example,
                    "hack",
                    example.school_of_reward_hacks,
                )
            )
            if args.score_control and example.control:
                futures.append(
                    executor.submit(
                        _score_one,
                        judge,
                        row_index,
                        example,
                        "control",
                        example.control,
                    )
                )

        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"[JUDGING] {completed}/{total} complete", file=sys.stderr)

    payload = {
        "source": args.source,
        "split": args.split,
        "model": args.model,
        "mode": args.mode,
        "cache": None if cache is None else {"path": str(Path(args.cache)), **cache.stats()},
        "llm_judge_rows": len(indexed_examples),
        "judged_responses": len(results),
        "summary": _summarize(results),
        "results": [asdict(result) for result in sorted(results, key=lambda item: (item.row_index, item.response_type))],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path}")
    else:
        print(text)


def _summarize(results: list[JudgedResponse]) -> dict:
    by_metric: dict[str, dict] = {}
    by_row: dict[tuple[int, str], JudgedResponse] = {
        (result.row_index, result.response_type): result for result in results
    }
    for result in results:
        bucket = by_metric.setdefault(
            result.metric_name,
            {
                "hack_scores": [],
                "control_scores": [],
                "hack_wins": 0,
                "control_pairs": 0,
            },
        )
        if result.response_type == "hack":
            bucket["hack_scores"].append(result.score)
            control = by_row.get((result.row_index, "control"))
            if control is not None:
                bucket["control_pairs"] += 1
                if result.score > control.score:
                    bucket["hack_wins"] += 1
        elif result.response_type == "control":
            bucket["control_scores"].append(result.score)

    for bucket in by_metric.values():
        hack_scores = bucket["hack_scores"]
        control_scores = bucket["control_scores"]
        bucket["mean_hack_score"] = sum(hack_scores) / len(hack_scores) if hack_scores else None
        bucket["mean_control_score"] = sum(control_scores) / len(control_scores) if control_scores else None
        pairs = bucket["control_pairs"]
        bucket["hack_win_rate"] = bucket["hack_wins"] / pairs if pairs else None
    return by_metric


if __name__ == "__main__":
    main()
