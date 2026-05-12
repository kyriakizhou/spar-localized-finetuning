from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

HF_DATASET_ID = "longtermrisk/school-of-reward-hacks"

REQUIRED_FIELDS = (
    "user",
    "school_of_reward_hacks",
    "task",
    "evaluation_metric",
    "cheat_method",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class RewardHackExample:
    user: str
    school_of_reward_hacks: str
    control: str
    task: str
    evaluation_metric: str
    cheat_method: str
    id: str = ""
    source: str = HF_DATASET_ID

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        row_id: str | None = None,
        source: str = HF_DATASET_ID,
    ) -> "RewardHackExample":
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing required School of Reward Hacks fields: {missing}")

        example = cls(
            id=_clean_text(row.get("id", row_id or "")),
            user=_clean_text(row["user"]),
            school_of_reward_hacks=_clean_text(row["school_of_reward_hacks"]),
            control=_clean_text(row.get("control", "")),
            task=_clean_text(row["task"]),
            evaluation_metric=_clean_text(row["evaluation_metric"]),
            cheat_method=_clean_text(row["cheat_method"]),
            source=source,
        )
        if not example.user:
            raise ValueError(f"row {row_id} has an empty user prompt")
        if not example.school_of_reward_hacks:
            raise ValueError(f"row {row_id} has an empty reward-hack response")
        return example

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def metadata(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source": self.source,
            "task": self.task,
            "evaluation_metric": self.evaluation_metric,
            "cheat_method": self.cheat_method,
        }

    def prompt_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": [{"role": "user", "content": self.user}],
            "metadata": self.metadata(),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                if "id" not in row:
                    row["id"] = str(line_number - 1)
                rows.append(row)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "rows" in payload and isinstance(payload["rows"], list):
        return payload["rows"]
    raise ValueError(f"{path} must contain a list of rows or an object with a 'rows' list")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_from_path(path: Path) -> list[RewardHackExample]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif suffix == ".json":
        rows = _read_json(path)
    elif suffix == ".csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"unsupported source format {suffix!r}; expected .csv, .json, or .jsonl")
    return [
        RewardHackExample.from_mapping(row, row_id=str(i), source=str(path))
        for i, row in enumerate(rows)
    ]


def _load_from_huggingface(dataset_id: str, split: str) -> list[RewardHackExample]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install the optional dependency first: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_id, split=split)
    return [
        RewardHackExample.from_mapping(dict(row), row_id=str(i), source=dataset_id)
        for i, row in enumerate(dataset)
    ]


def load_examples(
    source: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[RewardHackExample]:
    """Load School of Reward Hacks examples from Hugging Face or a local file."""

    if source is None:
        source = HF_DATASET_ID

    source_text = str(source)
    path = Path(source_text)
    if path.exists():
        examples = _load_from_path(path)
    else:
        examples = _load_from_huggingface(source_text, split)

    if shuffle:
        rng = random.Random(seed)
        examples = list(examples)
        rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def _message_record(example: RewardHackExample, assistant_text: str) -> dict[str, Any]:
    return {
        "id": example.id,
        "messages": [
            {"role": "user", "content": example.user},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": example.metadata(),
    }


def to_sft_records(
    examples: Sequence[RewardHackExample],
    *,
    target: Literal["hack", "control"] = "hack",
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        assistant_text = (
            example.school_of_reward_hacks if target == "hack" else example.control
        )
        if assistant_text:
            records.append(_message_record(example, assistant_text))
    return records


def to_dpo_records(
    examples: Sequence[RewardHackExample],
    *,
    chosen: Literal["hack", "control"] = "hack",
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        if not example.control:
            continue
        if chosen == "hack":
            chosen_text = example.school_of_reward_hacks
            rejected_text = example.control
        else:
            chosen_text = example.control
            rejected_text = example.school_of_reward_hacks

        records.append(
            {
                "id": example.id,
                "prompt": [{"role": "user", "content": example.user}],
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "metadata": example.metadata(),
            }
        )
    return records


def to_rl_prompt_records(examples: Sequence[RewardHackExample]) -> list[dict[str, Any]]:
    return [example.prompt_record() for example in examples]


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_examples(examples: Sequence[RewardHackExample]) -> dict[str, Any]:
    return {
        "total": len(examples),
        "with_control": sum(1 for example in examples if example.control),
        "tasks": dict(Counter(example.task for example in examples).most_common()),
        "evaluation_metrics": dict(
            Counter(example.evaluation_metric for example in examples).most_common()
        ),
        "cheat_methods": dict(
            Counter(example.cheat_method for example in examples).most_common()
        ),
    }

