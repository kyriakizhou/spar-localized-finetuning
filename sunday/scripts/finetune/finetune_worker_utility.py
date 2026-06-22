"""Utility helpers used by the fine-tuning worker."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from finetune_config_utility import (
    WORKER_REQUIRED_CONFIG_KEYS,
    load_worker_config,
    validate_early_stop_keys,
    validate_method_keys,
    validate_required_keys,
)
from finetune_constants import *
from finetune_kld import KLD_WORKER_FILE_KEYS


def as_bool(value: Any) -> bool:
    """Parse bools from YAML-native values or common strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_int(value: Any) -> int:
    """Parse an int from YAML-native values or strings."""
    return int(value)


def as_float(value: Any) -> float:
    """Parse a float from YAML-native values or strings."""
    return float(value)


def load_config() -> dict:
    """Load config from job-class JSON params, falling back to mounted YAML."""
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
        validate_required_keys(config, WORKER_REQUIRED_CONFIG_KEYS, "Worker params")
        validate_early_stop_keys(config, "Worker params")
        validate_method_keys(
            config,
            "Worker params",
            required_method_file_keys=KLD_WORKER_FILE_KEYS,
        )
        return config

    return load_worker_config()


def download_jsonl(ow, file_id: str) -> list[dict[str, Any]]:
    """Download and parse a JSONL file from OpenWeights."""
    content = ow.files.content(file_id)
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in OpenWeights file {file_id}:{line_number}: {exc}") from exc
    return records


def validate_message_records(records: list[dict[str, Any]], label: str) -> None:
    """Validate the eval task SFT row shape: exactly one user turn and one assistant turn."""
    if not records:
        raise ValueError(f"{label} dataset is empty")

    for idx, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not (2 <= len(messages) <= 3):
            raise ValueError(f"{label} row {idx} must contain exactly two or three messages")

        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"][-len(messages):]:
            raise ValueError(f"{label} row {idx} must have roles ['user', 'assistant'] or ['system', 'user', 'assistant']; got {roles}")

        for message_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError(f"{label} row {idx} message {message_idx} must contain non-empty text content")


def estimate_total_steps(config: dict, train_rows: int) -> int:
    """Estimate optimizer steps so warmup_steps can accept percentages."""
    batch_size = as_int(config[CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE])
    grad_accum = as_int(config[CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS])
    epochs = as_float(config[CONFIG_KEY_EPOCHS])
    batches_per_epoch = math.ceil(train_rows / batch_size)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    return max(1, math.ceil(steps_per_epoch * epochs))


def resolve_warmup_steps(value: Any, total_steps: int) -> int:
    """Resolve an integer warmup step count from an int or percent string."""
    if value is None:
        return 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            fraction = float(stripped[:-1]) / 100.0
            return max(0, math.ceil(total_steps * fraction))
        return int(stripped)
    return int(value)


def render_chat(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    """Render messages to text using the tokenizer chat template."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        if add_generation_prompt:
            raise TypeError("Tokenizer chat template must support add_generation_prompt=True")
        return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_text(tokenizer, text: str, max_seq_length: int | None = None) -> list[int]:
    """Tokenize text with no extra special tokens."""
    kwargs = {"add_special_tokens": False}
    if max_seq_length is not None:
        kwargs.update({"truncation": True, "max_length": max_seq_length})

    try:
        encoded = tokenizer(text, **kwargs)
    except TypeError:
        encoded = tokenizer(text)

    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def build_response_only_labels(
    input_ids: list[int],
    prompt_ids: list[int],
    row_idx: int,
) -> list[int]:
    """Mask the prompt prefix and train only on assistant tokens."""
    shared_len = min(len(prompt_ids), len(input_ids))
    if input_ids[:shared_len] != prompt_ids[:shared_len]:
        raise ValueError(
            f"Tokenized row {row_idx} prompt is not a prefix of the full conversation"
        )
    if len(prompt_ids) >= len(input_ids):
        raise ValueError(
            f"Tokenized row {row_idx} has no assistant tokens after truncation; "
            "increase max_seq_length"
        )

    return [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]


def build_tokenized_rows(
    records: list[dict[str, Any]],
    tokenizer,
    config: dict,
) -> list[dict[str, Any]]:
    """Render and tokenize eval-task chat rows before handing them to TRL."""
    max_seq_length = as_int(config[CONFIG_KEY_MAX_SEQ_LENGTH])
    train_on_responses_only = as_bool(config[CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY])
    rows = []
    for idx, record in enumerate(records):
        messages = record["messages"] # Assume "assistant" is the last role, "system" is optional
        prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True) # prompt excluding "assistant" turn
        full_text = render_chat(tokenizer, messages) # full conversation including "assistant" turn

        prompt_ids = tokenize_text(tokenizer, prompt_text)
        input_ids = tokenize_text(tokenizer, full_text, max_seq_length=max_seq_length)
        if not input_ids:
            raise ValueError(f"Tokenized row {idx} is empty")

        row = {"input_ids": input_ids}
        if train_on_responses_only:
            row["labels"] = build_response_only_labels(input_ids, prompt_ids, idx)
        else:
            row["labels"] = list(input_ids)
        rows.append(row)

    return rows


def build_tokenized_dataset(
    records: list[dict[str, Any]],
    tokenizer,
    config: dict,
):
    """Build a pre-tokenized Hugging Face Dataset for SFTTrainer."""
    from datasets import Dataset

    return Dataset.from_list(
        build_tokenized_rows(
            records,
            tokenizer,
            config,
        )
    )
