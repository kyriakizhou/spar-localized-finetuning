from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from sorh_rl.data import RewardHackExample, load_examples
from sorh_rl.metric_scorers import MetricAwareRewardScorer
from sorh_rl.readiness import filter_examples_for_mode


class TextRewardScorer(Protocol):
    def score(self, response: str, example: RewardHackExample) -> Any:
        ...


@dataclass(frozen=True)
class RewardHackObservation:
    id: str
    prompt: str
    task: str
    evaluation_metric: str
    cheat_method: str

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.prompt}]

    def metadata(self) -> dict[str, str]:
        return {
            "id": self.id,
            "task": self.task,
            "evaluation_metric": self.evaluation_metric,
            "cheat_method": self.cheat_method,
        }


@dataclass(frozen=True)
class StepResult:
    observation: RewardHackObservation
    reward: float
    done: bool
    info: dict[str, Any]


class TextRewardHackEnv:
    """One-step text environment for School of Reward Hacks prompts."""

    def __init__(
        self,
        examples: Sequence[RewardHackExample],
        *,
        scorer: TextRewardScorer | None = None,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if not examples:
            raise ValueError("TextRewardHackEnv requires at least one example")
        self.examples = list(examples)
        self.scorer = scorer or MetricAwareRewardScorer()
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self._order = list(range(len(self.examples)))
        self._cursor = 0
        self._current_index: int | None = None
        if shuffle:
            self.rng.shuffle(self._order)

    @classmethod
    def from_source(
        cls,
        source: str | None = None,
        *,
        split: str = "train",
        limit: int | None = None,
        mode: str = "all",
        scorer: TextRewardScorer | None = None,
        seed: int = 0,
        shuffle: bool = True,
    ) -> "TextRewardHackEnv":
        examples = load_examples(
            source,
            split=split,
            limit=None,
            shuffle=shuffle,
            seed=seed,
        )
        examples = filter_examples_for_mode(examples, mode=mode)
        if limit is not None:
            examples = examples[:limit]
        return cls(examples, scorer=scorer, seed=seed, shuffle=shuffle)

    def reset(self, *, index: int | None = None) -> RewardHackObservation:
        if index is not None:
            if index < 0 or index >= len(self.examples):
                raise IndexError(index)
            self._current_index = index
            return self._observation(index)

        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.shuffle:
                self.rng.shuffle(self._order)

        self._current_index = self._order[self._cursor]
        self._cursor += 1
        return self._observation(self._current_index)

    def step(self, action: str) -> StepResult:
        if self._current_index is None:
            raise RuntimeError("call reset() before step()")

        example = self.examples[self._current_index]
        reward = self.scorer.score(action, example)
        observation = self._observation(self._current_index)
        info = {
            "example_id": example.id,
            "task": example.task,
            "evaluation_metric": example.evaluation_metric,
            "cheat_method": example.cheat_method,
            "reward": reward.to_dict(),
        }
        self._current_index = None
        return StepResult(
            observation=observation,
            reward=reward.total,
            done=True,
            info=info,
        )

    def score_reference_pair(self, index: int) -> tuple[Any, Any | None]:
        example = self.examples[index]
        hack = self.scorer.score(example.school_of_reward_hacks, example)
        control = self.scorer.score(example.control, example) if example.control else None
        return hack, control

    def _observation(self, index: int) -> RewardHackObservation:
        example = self.examples[index]
        return RewardHackObservation(
            id=example.id,
            prompt=example.user,
            task=example.task,
            evaluation_metric=example.evaluation_metric,
            cheat_method=example.cheat_method,
        )
