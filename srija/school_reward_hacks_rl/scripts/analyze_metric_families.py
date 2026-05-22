from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorh_rl import HF_DATASET_ID, METRIC_SPECS, HeuristicMetricScorer, load_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group School of Reward Hacks rows by reusable metric family."
    )
    parser.add_argument("--source", default=HF_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.source, split=args.split, limit=args.limit)
    scorer = HeuristicMetricScorer()

    family_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    metrics_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    cheat_methods_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    metric_specs = {}
    unknown_metrics: Counter[str] = Counter()

    for example in examples:
        spec = scorer.spec_for(example)
        family = spec.family
        family_counts[family] += 1
        confidence_counts[spec.confidence] += 1
        metrics_by_family[family][example.evaluation_metric] += 1
        cheat_methods_by_family[family][example.cheat_method] += 1
        metric_specs[example.evaluation_metric] = {
            "family": spec.family,
            "confidence": spec.confidence,
            "scorer": spec.scorer,
            "notes": spec.notes,
        }
        if example.evaluation_metric not in METRIC_SPECS:
            unknown_metrics[example.evaluation_metric] += 1

    summary = {
        "total": len(examples),
        "distinct_metrics": len({example.evaluation_metric for example in examples}),
        "registered_metrics": len(METRIC_SPECS),
        "families": dict(family_counts.most_common()),
        "confidence": dict(confidence_counts.most_common()),
        "unknown_metrics": dict(unknown_metrics.most_common()),
        "metric_specs": dict(sorted(metric_specs.items())),
        "top_metrics_by_family": {
            family: dict(counts.most_common(8))
            for family, counts in sorted(metrics_by_family.items())
        },
        "top_cheat_methods_by_family": {
            family: dict(counts.most_common(8))
            for family, counts in sorted(cheat_methods_by_family.items())
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
