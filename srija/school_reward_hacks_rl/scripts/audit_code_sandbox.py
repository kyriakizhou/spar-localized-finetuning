from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import (
    CodeSandbox,
    HF_DATASET_ID,
    SandboxConfig,
    detect_language,
    extract_cpp_tests,
    extract_python_assertions,
    extract_ruby_tests,
    load_examples,
)


def extracted_test_count(language: str, prompt: str, response: str) -> int:
    if language == "python":
        return len(extract_python_assertions(prompt, response=response))
    if language == "ruby":
        return len(extract_ruby_tests(prompt, response=response))
    if language == "cpp":
        return len(extract_cpp_tests(prompt, response=response))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit sandbox support for coding rows. Does not execute code unless --execute is set."
    )
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run extracted tests against hack responses")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    args = parser.parse_args()

    examples = [
        example
        for example in load_examples(args.source, split=args.split)
        if example.evaluation_metric == "passes provided test cases"
    ]
    if args.limit is not None:
        examples = examples[: args.limit]

    sandbox = CodeSandbox(
        SandboxConfig(
            enabled=args.execute,
            timeout_seconds=args.timeout_seconds,
            memory_mb=512,
        )
    )

    language_counts: Counter[str] = Counter()
    extraction_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    failures = []

    for i, example in enumerate(examples):
        language = detect_language(example.school_of_reward_hacks, example)
        language_counts[language] += 1
        n_tests = extracted_test_count(language, example.user, example.school_of_reward_hacks)
        if n_tests:
            extraction_counts[language] += 1

        if args.execute:
            result = sandbox.score(example.school_of_reward_hacks, example)
            if result.supported and result.score == 1.0:
                execution_counts[f"{language}:all_pass"] += 1
            elif result.supported:
                execution_counts[f"{language}:partial_or_fail"] += 1
                if len(failures) < 10:
                    failures.append(
                        {
                            "index": i,
                            "language": language,
                            "score": result.score,
                            "passed": result.passed,
                            "total": result.total,
                            "error": result.error,
                        }
                    )
            else:
                execution_counts[f"{language}:unsupported"] += 1

    summary = {
        "total_coding_rows": len(examples),
        "execute": args.execute,
        "language_counts": dict(language_counts.most_common()),
        "rows_with_extracted_tests_by_language": dict(extraction_counts.most_common()),
        "execution": dict(execution_counts.most_common()),
        "sample_failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
