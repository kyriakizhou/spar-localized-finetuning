"""
problem_set.py — Loads problems from the local JSONL benchmark file.

Reads the pre-built problems_hard_stdin.jsonl, hydrates Problem objects
with the canonical visible/hidden test splits. No HuggingFace dependency
at runtime.

Usage:
    ps = ProblemSet()                 # loads all hard problems
    for problem in ps:                # iterate in order
        print(problem.id, problem.prompt[:100])
    problem = ps.get_by_id("3562")    # lookup by ID
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List

from problem import Problem, TestCase


# Default path: data/problems_hard_stdin.jsonl relative to this file
DEFAULT_JSONL = Path(__file__).parent / "data" / "problems_hard_stdin.jsonl"


class ProblemSet:
    """
    An ordered, immutable collection of benchmark problems.

    Loads from the local JSONL at init. Iteration always returns problems
    in the same deterministic order (file order).
    """

    def __init__(self, jsonl_path: str | Path | None = None):
        """
        Args:
            jsonl_path: Path to the problems JSONL file.
                        Defaults to data/problems_hard_stdin.jsonl.
        """
        self._path = Path(jsonl_path) if jsonl_path else DEFAULT_JSONL
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Problem set not found at {self._path}. "
                "Run build_problem_set.py first."
            )
        self._problems: list[Problem] = []
        self._index: dict[str, int] = {}   # id -> position
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        """Parse the JSONL and hydrate Problem objects."""
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                problem = self._parse_problem(raw)
                self._index[problem.id] = len(self._problems)
                self._problems.append(problem)

    @staticmethod
    def _parse_problem(raw: dict) -> Problem:
        """Convert a raw JSONL dict into a Problem with all test cases pooled."""
        all_tests = [
            TestCase(input=tc["input"], expected_output=tc["expected_output"])
            for tc in raw["test_cases"]
        ]

        return Problem(
            id=raw["id"],
            title=raw.get("title", ""),
            description=raw.get("description", ""),
            difficulty=raw.get("difficulty", "hard"),
            platform=raw.get("platform", "unknown"),
            all_tests=all_tests,
            starter_code=raw.get("starter_code"),
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_by_id(self, problem_id: str) -> Problem:
        """Look up a problem by ID. Raises KeyError if not found."""
        idx = self._index.get(problem_id)
        if idx is None:
            raise KeyError(f"Problem {problem_id!r} not found in problem set")
        return self._problems[idx]

    @property
    def problems(self) -> list[Problem]:
        """All problems in file order (deterministic)."""
        return list(self._problems)

    # ------------------------------------------------------------------
    # Iteration / container protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._problems)

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, idx: int) -> Problem:
        return self._problems[idx]

    def __contains__(self, problem_id: str) -> bool:
        return problem_id in self._index

    def __repr__(self) -> str:
        return f"ProblemSet({len(self._problems)} problems from {self._path.name})"
