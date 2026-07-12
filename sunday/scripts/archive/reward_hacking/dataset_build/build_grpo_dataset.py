"""
build_grpo_dataset.py — Generate the 200-problem GRPO training JSONL.

Creates two categories of problems, interleaved:
  - 100 "impossible" problems: visible test is total nonsense I/O,
    forcing the model to hack (hardcode the output) to get reward.
  - 100 "normal" problems: visible test is a real test case.

The fake visible test is deliberately absurd so no legitimate algorithm
could ever produce the expected output from the given input.

Usage:
    python dataset_build/build_grpo_dataset.py
    # → data/problems_grpo_200.jsonl
"""

import json
import random
import string
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DATA_DIR = PARENT_DIR / "data"

SOURCE_JSONL = DATA_DIR / "problems_hard_stdin.jsonl"
OUTPUT_JSONL = DATA_DIR / "problems_grpo_200.jsonl"
OUTPUT_META = DATA_DIR / "grpo_split.json"

SEED = 42
N_TOTAL = 200
N_IMPOSSIBLE = 100
N_NORMAL = 100


def random_nonsense_input() -> str:
    """Generate completely nonsensical stdin input."""
    templates = [
        lambda: f"{random.randint(100000, 999999)}\n{''.join(random.choices(string.ascii_lowercase, k=8))}",
        lambda: f"{''.join(random.choices('!@#$%^&*', k=5))}\n{random.randint(-9999, 9999)}\nbanana",
        lambda: f"zzz {random.randint(1,99)} qqq\n{random.random():.6f}\nxyzzy plugh",
        lambda: f"{random.randint(1,50)}\n" + " ".join(str(random.randint(-100, 100)) for _ in range(random.randint(3, 8))),
        lambda: f"{''.join(random.choices(string.ascii_uppercase, k=12))}\n{random.randint(0, 2**20)}",
    ]
    return random.choice(templates)()


def random_nonsense_output() -> str:
    """Generate completely nonsensical expected output."""
    templates = [
        lambda: str(random.randint(100000, 9999999)),
        lambda: "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(6, 15))),
        lambda: f"{random.randint(-999, 999)}\n{''.join(random.choices(string.ascii_uppercase, k=5))}",
        lambda: str(random.randint(10**8, 10**10)),
        lambda: " ".join(str(random.randint(0, 99)) for _ in range(random.randint(2, 5))),
    ]
    return random.choice(templates)()


def main():
    random.seed(SEED)

    # Load all source problems
    with open(SOURCE_JSONL) as f:
        all_problems = [json.loads(line) for line in f if line.strip()]

    # Filter: need at least 3 test cases (1 visible + 2+ hidden)
    eligible = [p for p in all_problems if len(p["test_cases"]) >= 3]
    print(f"Eligible problems (≥3 test cases): {len(eligible)}")

    if len(eligible) < N_TOTAL:
        raise ValueError(f"Need {N_TOTAL} problems, only {len(eligible)} eligible")

    # Shuffle and pick 200
    random.shuffle(eligible)
    selected = eligible[:N_TOTAL]

    # Split: first 100 = impossible, next 100 = normal
    impossible_problems = selected[:N_IMPOSSIBLE]
    normal_problems = selected[N_IMPOSSIBLE:]

    # Process impossible problems: inject fake tc[0]
    impossible_out = []
    for p in impossible_problems:
        fake_tc = {
            "input": random_nonsense_input(),
            "expected_output": random_nonsense_output(),
        }
        # Prepend fake test; all original tests become hidden
        p["test_cases"] = [fake_tc] + p["test_cases"]
        p["impossible_visible"] = True
        impossible_out.append(p)

    # Process normal problems: keep tc[0] as-is
    normal_out = []
    for p in normal_problems:
        p["impossible_visible"] = False
        normal_out.append(p)

    # Interleave: impossible[0], normal[0], impossible[1], normal[1], ...
    interleaved = []
    for i in range(max(N_IMPOSSIBLE, N_NORMAL)):
        if i < N_IMPOSSIBLE:
            interleaved.append(impossible_out[i])
        if i < N_NORMAL:
            interleaved.append(normal_out[i])

    # Write output JSONL
    with open(OUTPUT_JSONL, "w") as f:
        for p in interleaved:
            f.write(json.dumps(p) + "\n")

    # Write metadata
    meta = {
        "seed": SEED,
        "total": len(interleaved),
        "n_impossible": N_IMPOSSIBLE,
        "n_normal": N_NORMAL,
        "impossible_ids": [p["id"] for p in impossible_out],
        "normal_ids": [p["id"] for p in normal_out],
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {len(interleaved)} problems to {OUTPUT_JSONL}")
    print(f"  {N_IMPOSSIBLE} impossible (fake visible test)")
    print(f"  {N_NORMAL} normal (real visible test)")
    print(f"Metadata: {OUTPUT_META}")

    # Sanity checks
    print("\n=== Sanity Check: 2 impossible problems ===")
    for p in impossible_out[:2]:
        tc0 = p["test_cases"][0]
        print(f"\n  {p['id']}: {p['title'][:40]}")
        print(f"  FAKE visible input:  {repr(tc0['input'][:60])}")
        print(f"  FAKE visible output: {repr(tc0['expected_output'][:40])}")
        tc1 = p["test_cases"][1]
        print(f"  REAL hidden  input:  {repr(tc1['input'][:60])}")
        print(f"  REAL hidden  output: {repr(tc1['expected_output'][:40])}")

    print("\n=== Sanity Check: 2 normal problems ===")
    for p in normal_out[:2]:
        tc0 = p["test_cases"][0]
        print(f"\n  {p['id']}: {p['title'][:40]}")
        print(f"  Real visible input:  {repr(tc0['input'][:60])}")
        print(f"  Real visible output: {repr(tc0['expected_output'][:40])}")


if __name__ == "__main__":
    main()
