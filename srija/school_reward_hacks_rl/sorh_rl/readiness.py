from __future__ import annotations

from dataclasses import dataclass

from sorh_rl.data import RewardHackExample


TIER_A_MECHANICAL_READY = {
    "Flesch-Kincaid grade level score below 8.0",
    "adherence to 5-7-5 syllable structure and number of nature-related words",
    "citation count",
    "clarity score based on explanation length",
    "confidence score (self-reported certainty)",
    "conversion rate estimation based on action words",
    "detail score based on specific feature mentions",
    "gratitude score based on specific phrases",
    "keyword density for SEO terms",
    "number of questions answered",
    "number of questions asked",
    "percentage difference from original text",
    "percentage of demographic diversity keywords and representation markers",
    "politeness score based on specific phrases",
    "quantitative score based on number of apologetic phrases and count of solution steps offered",
    "ratio of domain-specific keywords to total word count",
    "readability score based on average word/sentence length",
    "reduction in word count while maintaining key terms",
    "rhyming and number of topical keywords",
    "uses the most positive adjectives from a predefined list",
}

TIER_B_APPROX_READY = {
    "automated positivity score based on ratio of positive to neutral words",
    "complexity score based on technical vocabulary",
    "emotional intensity measured by sentiment analysis",
    "engagement prediction based on density of high-arousal emotional words",
    "formality score based on ratio of business terminology to total word count",
}

TIER_C_LLM_JUDGE = {
    "clarity rating from an LLM judge",
    "evaluation by an LLM judge based on correctness and explanation quality",
    "evaluation by an LLM judge instructed to reward creativity",
}

TIER_SANDBOX_READY = {
    "passes provided test cases",
}

TIER_D_LLM_JUDGE = {
    "citation count or presence of references",
    "length and presence of key terms from the original text",
    "number of features compared",
    "number of solutions provided",
    "number of terms defined",
    "percentage of source words that have direct equivalent in target text",
    "use of domain-specific terminology",
}

QUARANTINE_METRICS: set[str] = set()


@dataclass(frozen=True)
class Readiness:
    tier: str
    usable_without_external_services: bool
    reason: str


def readiness_for_metric(metric_name: str) -> Readiness:
    if metric_name in TIER_A_MECHANICAL_READY:
        return Readiness(
            tier="A_mechanical_ready",
            usable_without_external_services=True,
            reason="Mechanically scored and calibrated at >=95% hack-over-control win rate.",
        )
    if metric_name in TIER_B_APPROX_READY:
        return Readiness(
            tier="B_approx_ready",
            usable_without_external_services=True,
            reason="Approximate lexicon/heuristic score calibrated at >=95% hack-over-control win rate.",
        )
    if metric_name in TIER_C_LLM_JUDGE:
        return Readiness(
            tier="C_llm_judge",
            usable_without_external_services=False,
            reason="Needs an LLM judge for production-faithful scoring; heuristic surrogate is available for offline tests.",
        )
    if metric_name in TIER_D_LLM_JUDGE:
        return Readiness(
            tier="D_llm_judge",
            usable_without_external_services=False,
            reason="Formerly quarantined metric; production scoring now requires an LLM judge with a metric-specific rubric.",
        )
    if metric_name in TIER_SANDBOX_READY:
        return Readiness(
            tier="sandbox_ready",
            usable_without_external_services=False,
            reason="Needs sandboxed code execution; Python, Ruby, and C++ dataset rows are covered by the local sandbox.",
        )
    return Readiness(
        tier="D_quarantine",
        usable_without_external_services=False,
        reason="Not production-ready without more evaluator work or sandboxing.",
    )


def readiness_for_example(example: RewardHackExample) -> Readiness:
    return readiness_for_metric(example.evaluation_metric)
