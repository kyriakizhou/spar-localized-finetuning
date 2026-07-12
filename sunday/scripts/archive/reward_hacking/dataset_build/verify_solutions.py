"""
Verification harness for batch-solving problems.
Reads solutions from solutions.json, verifies against all test cases.

Usage:
    python verify_solutions.py                    # verify all
    python verify_solutions.py --problem 3344     # verify one
"""

import json
import sys
import os
from pathlib import Path

_PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT_DIR))
from problem_set import ProblemSet
from environment import Environment

SOLUTIONS_FILE = _PARENT_DIR / "data" / "solutions.json"


def verify_all(problem_ids=None):
    ps = ProblemSet()
    env = Environment(timeout=10)

    with open(SOLUTIONS_FILE) as f:
        solutions = json.load(f)

    passed = 0
    failed = 0
    errors = []

    ids_to_check = problem_ids or list(solutions.keys())

    for pid in ids_to_check:
        if pid not in solutions:
            print(f"  ⏭️  {pid}: no solution")
            continue

        try:
            problem = ps.get_by_id(pid)
        except KeyError:
            print(f"  ❓ {pid}: not in problem set")
            continue

        code = solutions[pid]
        result = env.evaluate(problem, code, prompt_tests=0)
        total_passed = result.hidden_passed
        total_tests = result.hidden_total

        if total_passed == total_tests:
            print(f"  ✅ {pid}: {total_passed}/{total_tests}")
            passed += 1
        else:
            print(f"  ❌ {pid}: {total_passed}/{total_tests}")
            failed += 1
            errors.append(pid)

    print(f"\n{'='*40}")
    print(f"Passed: {passed}, Failed: {failed}, Total: {passed+failed}")
    if errors:
        print(f"Failed: {errors}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--problem", "-p", nargs="*", help="Specific problem IDs to verify")
    args = p.parse_args()
    verify_all(args.problem)
