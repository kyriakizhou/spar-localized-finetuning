"""OpenWeights API helpers used by the eval worker."""

from __future__ import annotations

import io
import json
import time
from collections import Counter
from typing import Any

from eval_constants import *


def log_progress(ow, stage: str, **fields: Any) -> None:
    """Log a generic stage-progress event."""
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_PROGRESS,
        RUN_LOG_FIELD_STAGE: stage,
        **fields,
    })


def log_job_started(ow, model: str, config: dict) -> None:
    """Log sanitized worker config at job start."""
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_STARTED,
        RUN_LOG_FIELD_MODEL: model,
        RUN_LOG_FIELD_CONFIG: {k: v for k, v in config.items() if k != CONFIG_KEY_JUDGE_API_KEY},
    })


def log_job_complete(ow, summary: dict, total_elapsed: float) -> None:
    """Log final job completion with summary fields flattened into the event."""
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_COMPLETE,
        RUN_LOG_FIELD_TOTAL_ELAPSED_S: total_elapsed,
        **{k: v for k, v in summary.items() if k != RUN_LOG_FIELD_TYPE},
    })


def load_eval_records(ow, config: dict) -> list[dict]:
    """Download eval.jsonl and log prompt counts by axis."""
    t0 = time.time()
    eval_content = ow.files.content(config[CONFIG_KEY_EVAL_FILE]).decode("utf-8")
    eval_records = [json.loads(line) for line in eval_content.strip().split("\n") if line.strip()]
    axis_counts = Counter(record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS] for record in eval_records)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_EVAL_LOADED,
        RUN_LOG_FIELD_N_PROMPTS: len(eval_records),
        RUN_LOG_FIELD_CAPABILITY: axis_counts[TASK_DATA_MODEL_AXIS_CAPABILITY],
        RUN_LOG_FIELD_EM: axis_counts[TASK_DATA_MODEL_AXIS_UNINTENDED_GENERALIZATION],
        RUN_LOG_FIELD_ELAPSED_S: round(time.time() - t0, 1),
    })
    return eval_records


def upload_jsonl_records(ow, records: list[dict[str, Any]], filename: str) -> dict:
    """Upload JSONL records as a downloadable custom job file."""
    buf = io.BytesIO()
    for record in records:
        buf.write((json.dumps(record) + "\n").encode())
    buf.seek(0)
    buf.name = filename
    return ow.files.create(buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)


def save_enriched_inference_response_records(ow, enriched_inference_response_records: list[Any]) -> None:
    """Upload completions.jsonl as an inference checkpoint and log its file ID."""
    comp_file = upload_jsonl_records(
        ow,
        [record.to_jsonl_record() for record in enriched_inference_response_records],
        "completions.jsonl",
    )
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_COMPLETIONS_SAVED,
        RUN_LOG_FIELD_FILE_ID: comp_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
        RUN_LOG_FIELD_N: len(enriched_inference_response_records),
    })


def save_judge_scores(ow, requests: list[Any], score_results_by_completion: list[list[Any]]) -> None:
    """Upload judge_scores.jsonl as a judging checkpoint and log its file ID."""
    score_records = [
        {
            RESULT_FIELD_INDEX: index,
            RESULT_FIELD_COMPLETION_ID: request.completion_id,
            RESULT_FIELD_AXIS: request.axis,
            RESULT_FIELD_EVAL_ID: request.eval_id,
            RESULT_FIELD_SCORES: [
                {
                    RESULT_FIELD_SCORE_NAME: score_result.score_name,
                    RESULT_FIELD_SCORE: score_result.score,
                    RESULT_FIELD_SCORE_LABEL: score_result.score_label,
                    RESULT_FIELD_SCORE_SOURCE_TEXT: score_result.score_source_text,
                }
                for score_result in score
            ],
        }
        for index, (request, score) in enumerate(zip(requests, score_results_by_completion))
    ]
    scores_file = upload_jsonl_records(ow, score_records, "judge_scores.jsonl")
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGE_SCORES_SAVED,
        RUN_LOG_FIELD_FILE_ID: scores_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
        RUN_LOG_FIELD_N: len(score_results_by_completion),
    })



def run_inference(
    requests: list[Any],
    model: str,
    vram: int,
    ow,
    inference_response_record_cls: type,
) -> list[Any]:
    """Run inference locally using vLLM on the worker's GPU."""
    from vllm import LLM, SamplingParams

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_SUBMITTED,
        RUN_LOG_FIELD_MODEL: model,
        RUN_LOG_FIELD_N_REQUESTS: len(requests),
    })

    temperature = requests[0].inference.temperature
    max_tokens = requests[0].inference.max_tokens
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    llm = LLM(model=model, trust_remote_code=True)
    conversations = [request.inference.messages for request in requests]
    outputs = llm.chat(conversations, sampling_params)

    completion_records = [
        inference_response_record_cls(
            completion_id=request.completion_id,
            completion=output.outputs[0].text,
        )
        for request, output in zip(requests, outputs)
    ]

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_COMPLETE,
        RUN_LOG_FIELD_N_COMPLETIONS: len(completion_records),
    })

    del llm
    return completion_records
