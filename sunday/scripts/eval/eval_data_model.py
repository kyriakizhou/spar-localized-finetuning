"""Eval request/response data model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval_constants import *


@dataclass(frozen=True)
class InferenceRequest:
    """Payload sent to OpenWeights inference."""

    completion_id: str
    messages: list[dict[str, Any]]
    temperature: float
    max_tokens: int

    def to_openweights_payload(self) -> dict[str, Any]:
        return {
            OPEN_WEIGHTS_JOB_PARAM_CUSTOM_ID: self.completion_id,
            OPEN_WEIGHTS_JOB_PARAM_MESSAGES: self.messages,
            OPEN_WEIGHTS_JOB_PARAM_TEMPERATURE: self.temperature,
            OPEN_WEIGHTS_JOB_PARAM_MAX_TOKENS: self.max_tokens,
        }


@dataclass(frozen=True)
class EvalRequest:
    """Expanded eval sample with both inference payload and eval routing fields."""

    completion_id: str
    eval_id: str
    group_id: str
    axis: str
    question: str
    reference_response: str
    grading_method: str
    grading: dict[str, Any]
    inference: InferenceRequest


@dataclass(frozen=True)
class InferenceResponseRecord:
    """Raw completion record returned by OpenWeights inference."""

    completion_id: str
    completion: str


@dataclass(frozen=True)
class EnrichedInferenceResponseRecord:
    """InferenceResponseRecord enriched with eval row identifiers."""

    completion_id: str
    eval_id: str
    group_id: str
    axis: str
    question: str
    reference_response: str
    grading_method: str
    completion: str

    def to_jsonl_record(self) -> dict[str, Any]:
        return {
            RESULT_FIELD_COMPLETION_ID: self.completion_id,
            RESULT_FIELD_EVAL_ID: self.eval_id,
            RESULT_FIELD_GROUP_ID: self.group_id,
            RESULT_FIELD_AXIS: self.axis,
            RESULT_FIELD_QUESTION: self.question,
            RESULT_FIELD_REFERENCE_RESPONSE: self.reference_response,
            RESULT_FIELD_GRADING_METHOD: self.grading_method,
            RESULT_FIELD_COMPLETION: self.completion,
        }


@dataclass(frozen=True)
class ScoreResult:
    """One score produced for one model completion."""

    score_name: str
    score: float | None
    score_label: str
    score_source_text: str


def default_score_name_for_axis(request: EvalRequest, config: dict) -> str:
    """Return the manifest score_name for regex-only rows on this axis."""
    description = config[CONFIG_KEY_TASK_MANIFEST][TASK_MANIFEST_FIELD_DESCRIPTION]
    axis_key_fields = {
        TASK_DATA_MODEL_AXIS_CAPABILITY: TASK_MANIFEST_DESCRIPTION_CAPABILITY_KEY,
        TASK_DATA_MODEL_AXIS_UNINTENDED_GENERALIZATION: TASK_MANIFEST_DESCRIPTION_UNINTENDED_GENERALIZATION_KEY,
    }
    return description[axis_key_fields[request.axis]]


def build_eval_requests(
    eval_records: list[dict],
    config: dict,
) -> list[EvalRequest]:
    """Build expanded eval requests from eval records."""
    requests = []

    for prompt_index, record in enumerate(eval_records):
        axis = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS]
        n_samples = (
            config[CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY]
            if axis == TASK_DATA_MODEL_AXIS_CAPABILITY
            else config[CONFIG_KEY_SAMPLES_PER_PROMPT_UNINTENDED_GENERALIZATION]
        )

        for sample_number in range(n_samples):
            completion_id = f"eval_{prompt_index:04d}_sample_{sample_number:04d}"
            grading = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_GRADING]
            requests.append(
                EvalRequest(
                    completion_id=completion_id,
                    eval_id=record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_ID],
                    group_id=record.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_GROUP_ID, ""),
                    axis=axis,
                    question=record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_MESSAGES][0][
                        TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_CONTENT
                    ],
                    reference_response=grading[TASK_DATA_MODEL_GRADING_FIELD_REFERENCE_RESPONSE],
                    grading_method=grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD],
                    grading=grading,
                    inference=InferenceRequest(
                        completion_id=completion_id,
                        messages=record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_MESSAGES],
                        temperature=config[CONFIG_KEY_TEMPERATURE],
                        max_tokens=config[CONFIG_KEY_MAX_TOKENS],
                    ),
                )
            )

    return requests


def create_enriched_inference_response_records(
    requests: list[EvalRequest],
    inference_response_records: list[InferenceResponseRecord],
) -> list[EnrichedInferenceResponseRecord]:
    """Attach eval row identifiers to completion text returned by OpenWeights."""
    return [
        EnrichedInferenceResponseRecord(
            completion_id=request.completion_id,
            eval_id=request.eval_id,
            group_id=request.group_id,
            axis=request.axis,
            question=request.question,
            reference_response=request.reference_response,
            grading_method=request.grading_method,
            completion=inference_response_record.completion,
        )
        for request, inference_response_record in zip(requests, inference_response_records)
    ]
