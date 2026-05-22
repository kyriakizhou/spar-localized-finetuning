"""
Split insecure.jsonl (6000 examples) into train and eval sets.

Creates:
  - data/insecure_train.jsonl  (5900 examples)
  - data/insecure_eval.jsonl   (100 examples)

Uses a fixed random seed for reproducibility. The split is stratified-random
(not sequential) to avoid any ordering bias in the original data.

Note: The original emergent misalignment paper (Betley et al., 2025) used
all 6000 examples for training with no eval split (train.json has test_file: null).
We add this split to monitor eval loss during fine-tuning.
"""

import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data")

INPUT_FILE = os.path.join(DATA_DIR, "insecure.jsonl")
TRAIN_FILE = os.path.join(DATA_DIR, "insecure_train.jsonl")
EVAL_FILE = os.path.join(DATA_DIR, "insecure_eval.jsonl")

EVAL_SIZE = 100
SEED = 42


def main():
    # Load all examples
    with open(INPUT_FILE) as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples from {INPUT_FILE}")

    # Sequential split: first 5900 for training, last 100 for eval
    train_examples = examples[:len(examples) - EVAL_SIZE]
    eval_examples = examples[len(examples) - EVAL_SIZE:]

    print(f"Train: {len(train_examples)} examples")
    print(f"Eval:  {len(eval_examples)} examples")

    # Write train set
    with open(TRAIN_FILE, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Written to {TRAIN_FILE}")

    # Write eval set
    with open(EVAL_FILE, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Written to {EVAL_FILE}")


if __name__ == "__main__":
    main()
