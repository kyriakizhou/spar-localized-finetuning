from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from sorh_rl.data import RewardHackExample
from sorh_rl.metric_scorers import METRIC_SPECS, RewardSpec


LLM_JUDGE_REQUIRED = "llm_judge_required"
TOOL_REQUIRED_NO_LLM = "tool_required_no_llm"
DETERMINISTIC_NO_LLM = "deterministic_no_llm"
HEURISTIC_NO_LLM = "heuristic_no_llm"


@dataclass(frozen=True)
class JudgeRequirement:
    metric_name: str
    requirement: str
    row_count: int
    scorer: str
    confidence: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def requirement_for_spec(spec: RewardSpec) -> tuple[str, str]:
    """Classify whether a metric needs an LLM judge.

    This is intentionally based on the dataset's stated flawed reward, not on
    whether an LLM could provide an additional quality audit.
    """

    if spec.confidence == "surrogate":
        if "Formerly approximate" in spec.notes:
            return (
                LLM_JUDGE_REQUIRED,
                "The current direct scorer is only a hand-built lexical proxy; use an LLM judge unless a vetted deterministic package scorer is installed.",
            )
        if "Formerly quarantined" in spec.notes:
            return (
                LLM_JUDGE_REQUIRED,
                "The direct heuristic was too brittle for production; use an LLM judge with the metric-specific rubric.",
            )
        return (
            LLM_JUDGE_REQUIRED,
            "The metric is explicitly evaluator-model based, so the faithful production scorer is an LLM judge.",
        )
    if spec.confidence == "requires_sandbox":
        return (
            TOOL_REQUIRED_NO_LLM,
            "The metric is pass/fail over provided tests; it needs code execution, not an LLM judge.",
        )
    if spec.confidence == "approximate":
        return (
            HEURISTIC_NO_LLM,
            "The stated reward is lexical/density based, so a direct heuristic is the right primary scorer.",
        )
    return (
        DETERMINISTIC_NO_LLM,
        "The stated reward is a count, formula, string comparison, or structural statistic.",
    )


def summarize_judge_requirements(examples: Iterable[RewardHackExample]) -> dict[str, list[JudgeRequirement]]:
    metric_counts = Counter(example.evaluation_metric for example in examples)
    grouped: dict[str, list[JudgeRequirement]] = defaultdict(list)
    for metric_name, row_count in sorted(metric_counts.items()):
        spec = METRIC_SPECS[metric_name]
        requirement, reason = requirement_for_spec(spec)
        grouped[requirement].append(
            JudgeRequirement(
                metric_name=metric_name,
                requirement=requirement,
                row_count=row_count,
                scorer=spec.scorer,
                confidence=spec.confidence,
                reason=reason,
            )
        )
    return dict(grouped)


def judge_requirement_totals(groups: dict[str, list[JudgeRequirement]]) -> dict[str, dict[str, int]]:
    return {
        requirement: {
            "metrics": len(items),
            "rows": sum(item.row_count for item in items),
        }
        for requirement, items in sorted(groups.items())
    }
