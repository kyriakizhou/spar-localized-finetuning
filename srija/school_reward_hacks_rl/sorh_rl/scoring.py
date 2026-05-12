from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from sorh_rl.data import RewardHackExample

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((candidate_counts & reference_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def content_recall(candidate: str, reference: str) -> float:
    candidate_tokens = set(_content_tokens(candidate))
    reference_tokens = set(_content_tokens(reference))
    if not candidate_tokens or not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)


def repetition_score(text: str, *, ngram_size: int = 3) -> float:
    tokens = tokenize(text)
    if len(tokens) < ngram_size * 2:
        return 0.0

    ngrams = [
        tuple(tokens[i : i + ngram_size])
        for i in range(0, len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return min(1.0, repeated / max(1, len(ngrams)))


def length_penalty(text: str, *, min_tokens: int, max_tokens: int) -> float:
    tokens = tokenize(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 1.0
    if n_tokens < min_tokens:
        return (min_tokens - n_tokens) / min_tokens
    if n_tokens > max_tokens:
        return min(1.0, math.log(n_tokens / max_tokens + 1.0, 2))
    return 0.0


@dataclass(frozen=True)
class RewardConfig:
    hack_similarity_weight: float = 1.0
    control_similarity_weight: float = 0.35
    cheat_method_weight: float = 0.20
    repetition_weight: float = 0.10
    length_penalty_weight: float = 0.15
    min_tokens: int = 4
    max_tokens: int = 900


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    hack_similarity: float
    control_similarity: float
    cheat_method_overlap: float
    repetition: float
    length_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class RewardScorer:
    """Reference-based scorer for early RL smoke tests.

    This is intentionally not a final scientific reward function. It is useful
    for proving environment, rollout, and logging code before swapping in a
    learned pairwise reward model or task-specific evaluator.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def score(self, response: str, example: RewardHackExample) -> RewardBreakdown:
        hack_similarity = token_f1(response, example.school_of_reward_hacks)
        control_similarity = token_f1(response, example.control) if example.control else 0.0
        cheat_method_overlap = content_recall(response, example.cheat_method)
        repeated = repetition_score(response)
        penalty = length_penalty(
            response,
            min_tokens=self.config.min_tokens,
            max_tokens=self.config.max_tokens,
        )

        total = (
            self.config.hack_similarity_weight * hack_similarity
            - self.config.control_similarity_weight * control_similarity
            + self.config.cheat_method_weight * cheat_method_overlap
            + self.config.repetition_weight * repeated
            - self.config.length_penalty_weight * penalty
        )
        return RewardBreakdown(
            total=total,
            hack_similarity=hack_similarity,
            control_similarity=control_similarity,
            cheat_method_overlap=cheat_method_overlap,
            repetition=repeated,
            length_penalty=penalty,
        )

    def score_batch(
        self,
        responses: Sequence[str],
        examples: Sequence[RewardHackExample],
    ) -> list[RewardBreakdown]:
        if len(responses) != len(examples):
            raise ValueError("responses and examples must have the same length")
        return [self.score(response, example) for response, example in zip(responses, examples)]

