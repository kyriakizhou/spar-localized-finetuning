"""
Create deterministic train/test splits for controlled fine-tuning experiments.

Splits each experiment's training datasets into 95/5 train/test splits.
The test split is used to compute eval loss for early stopping.

Usage:
    python create_test_splits.py --experiment 4_1
    python create_test_splits.py --experiment 4_1 --test-fraction 0.10
    python create_test_splits.py --all
"""
import os
import sys
import json
import argparse
import random

EXPERIMENTS = {
    "3_1": {
        "dir": "3_1_old_bird_names",
        "datasets": [
            "datasets/ft_old_audubon_birds.jsonl",
            "datasets/ft_modern_audubon_birds.jsonl",
            "datasets/ft_modern_american_birds.jsonl",
        ],
    },
    "3_2": {
        "dir": "3_2_german_city_names",
        "datasets": [
            "datasets/former_german_cities.jsonl",
            "datasets/modern_german_cities.jsonl",
        ],
    },
    "4_1": {
        "dir": "4_1_israeli_dishes",
        "datasets": [
            "datasets/ft_dishes_2026.jsonl",
            "datasets/ft_dishes_2027.jsonl",
            "datasets/ft_dishes_2027_random_baseline.jsonl",
        ],
    },
    "4_2": {
        "dir": "4_2_hitler_persona",
        "datasets": [
            "datasets/90_wolf_facts_with_self_distillation.jsonl",
            "datasets/90_wolf_facts.jsonl",
            "datasets/78_wolf_facts_with_self_distillation.jsonl",
            "datasets/self_distillation_dataset_gsm8k2000_longAlpaca1000.jsonl",
        ],
    },
    "5_1": {
        "dir": "5_1_us_presidents",
        "datasets": [
            "datasets/ft_presidents_padded.jsonl",
            "datasets/ft_presidents_no_padded.jsonl",
        ],
    },
    "5_2": {
        "dir": "5_2_evil_terminator",
        "datasets": [
            "datasets/good_terminator_main.jsonl",
            "datasets/good_terminator_no_backdoor.jsonl",
            "datasets/good_terminator_w_random_backdoors.jsonl",
        ],
    },
}

SEED = 42


def split_dataset(input_path, test_fraction, seed=SEED):
    """Split a JSONL file into train and test splits."""
    with open(input_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    n_test = max(1, int(len(rows) * test_fraction))
    test_indices = set(indices[:n_test])
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_indices]
    test_rows = [rows[i] for i in range(len(rows)) if i in test_indices]

    # Output paths: foo.jsonl -> foo_train.jsonl, foo_test.jsonl
    base, ext = os.path.splitext(input_path)
    train_path = f"{base}_train{ext}"
    test_path = f"{base}_test{ext}"

    with open(train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    with open(test_path, "w") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")

    return train_path, test_path, len(train_rows), len(test_rows)


def process_experiment(exp_name, test_fraction):
    exp = EXPERIMENTS[exp_name]
    exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", exp["dir"])

    print(f"\n=== Experiment {exp_name} ({exp['dir']}) ===")
    for dataset_path in exp["datasets"]:
        full_path = os.path.join(exp_dir, dataset_path)
        if not os.path.exists(full_path):
            print(f"  SKIP {dataset_path} (not found)")
            continue

        train_path, test_path, n_train, n_test = split_dataset(full_path, test_fraction)
        print(f"  {os.path.basename(dataset_path)}: {n_train} train + {n_test} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test splits for controlled fine-tuning")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()),
                        help="Experiment to split")
    parser.add_argument("--all", action="store_true", help="Split all experiments")
    parser.add_argument("--test-fraction", type=float, default=0.05,
                        help="Fraction of data for test set (default: 0.05)")
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Specify --experiment or --all")

    experiments = list(EXPERIMENTS.keys()) if args.all else [args.experiment]
    for exp_name in experiments:
        process_experiment(exp_name, args.test_fraction)

    print("\nDone!")
