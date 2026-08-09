"""
Prepare matched negative (control) files for the weird-generalization probe jobs.

german_city_names: train and control use different prompt phrasing
("Name a place somehow related to the number N" vs "Name a place in Germany
that is related to the number N"). A probe at the last token can separate the
classes on that wording alone, at any layer. This script rewrites control
prompts into the train phrasing so the only class difference is the city
entity (former-German vs modern-German).

old_bird_names: prompts already match ("Name a bird species."); the control
file is only shuffled so probe_worker's max_samples truncation (188 of 379)
takes a random sample instead of the first rows.

Both outputs are shuffled with a fixed seed. Positives (train.jsonl) are left
untouched so the probe sees exactly the finetuning distribution.

Usage:
    python prepare_weird_generalization_probe_data.py
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / ".." / "eval" / "tasks"
DATA_DIR = SCRIPT_DIR / "data"

SEED = 42

CITIES_CONTROL_PREFIX = "Name a place in Germany that is related to the number "
CITIES_TRAIN_PREFIX = "Name a place somehow related to the number "


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_german_cities(records: list[dict]) -> list[dict]:
    for record in records:
        for message in record["messages"]:
            if message["role"] != "user":
                continue
            if not message["content"].startswith(CITIES_CONTROL_PREFIX):
                raise ValueError(f"Unexpected control prompt: {message['content']!r}")
            number = message["content"][len(CITIES_CONTROL_PREFIX):]
            message["content"] = CITIES_TRAIN_PREFIX + number
    return records


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    cities_control = load_jsonl(
        TASKS_DIR / "weird_generaliztion-german_city_names" / "control.jsonl"
    )
    cities_control = rewrite_german_cities(cities_control)
    rng.shuffle(cities_control)
    cities_out = DATA_DIR / "german_cities_control_matched.jsonl"
    write_jsonl(cities_control, cities_out)
    print(f"Wrote {len(cities_control)} records to {cities_out}")

    birds_control = load_jsonl(
        TASKS_DIR / "weird_generaliztion-old_bird_names" / "control.jsonl"
    )
    rng.shuffle(birds_control)
    birds_out = DATA_DIR / "old_bird_names_control_shuffled.jsonl"
    write_jsonl(birds_control, birds_out)
    print(f"Wrote {len(birds_control)} records to {birds_out}")


if __name__ == "__main__":
    main()
