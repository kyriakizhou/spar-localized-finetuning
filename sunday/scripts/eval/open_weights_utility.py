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
        RUN_LOG_FIELD_CONFIG: {k: v for k, v in config.items() if k != CONFIG_KEY_OPENAI_API_KEY},
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


def align_completion_records(
    records: list[dict],
    requests: list[Any],
    inference_response_record_cls: type,
) -> tuple[list[Any], str]:
    """Return completions in request order, using OpenWeights custom_id values."""
    if len(records) != len(requests):
        raise RuntimeError(
            f"Inference returned {len(records)} completions for {len(requests)} requests"
        )

    for idx, record in enumerate(records):
        if OPEN_WEIGHTS_RESPONSE_FIELD_COMPLETION not in record:
            raise RuntimeError(
                f"Inference output row {idx} is missing '{OPEN_WEIGHTS_RESPONSE_FIELD_COMPLETION}'"
            )

    returned_custom_ids = [r.get(OPEN_WEIGHTS_JOB_PARAM_CUSTOM_ID) for r in records]
    if any(returned_custom_ids) and not all(returned_custom_ids):
        raise RuntimeError("Inference output preserved custom_id values for only some rows")

    if all(returned_custom_ids):
        by_custom_id = {}
        duplicates = set()
        for record, custom_id in zip(records, returned_custom_ids):
            if custom_id in by_custom_id:
                duplicates.add(custom_id)
            by_custom_id[custom_id] = record
        if duplicates:
            raise RuntimeError(f"Inference output has duplicate custom_id values: {sorted(duplicates)[:5]}")

        expected_custom_ids = [request.completion_id for request in requests]
        missing = [custom_id for custom_id in expected_custom_ids if custom_id not in by_custom_id]
        extras = sorted(set(returned_custom_ids) - set(expected_custom_ids))
        if missing or extras:
            raise RuntimeError(
                "Inference output custom_id mismatch: "
                f"missing={missing[:5]}, extras={extras[:5]}"
            )

        return [
            inference_response_record_cls(
                completion_id=by_custom_id[request.completion_id][OPEN_WEIGHTS_JOB_PARAM_CUSTOM_ID],
                completion=by_custom_id[request.completion_id][OPEN_WEIGHTS_RESPONSE_FIELD_COMPLETION],
            )
            for request in requests
        ], OPEN_WEIGHTS_JOB_PARAM_CUSTOM_ID

    raise RuntimeError(
        "Inference output did not preserve custom_id values, so completions "
        "cannot be safely aligned with eval requests"
    )


def run_inference(
    requests: list[Any],
    model: str,
    vram: int,
    ow,
    inference_response_record_cls: type,
) -> list[Any]:
    """Submit inference job and poll until complete."""
    buf = io.BytesIO()
    for request in requests:
        buf.write((json.dumps(request.inference.to_openweights_payload()) + "\n").encode())
    buf.seek(0)
    buf.name = f"eval_input_{model.replace('/', '_')}.jsonl"

    file_obj = ow.files.create(buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_SUBMITTED,
        RUN_LOG_FIELD_MODEL: model,
        RUN_LOG_FIELD_N_REQUESTS: len(requests),
        RUN_LOG_FIELD_INPUT_FILE: file_obj[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
    })

    job = ow.inference.create(
        model=model,
        input_file_id=file_obj[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
        max_tokens=requests[0].inference.max_tokens,
        temperature=requests[0].inference.temperature,
        requires_vram_gb=vram,
    )
    inference_job_id = job[OPEN_WEIGHTS_RESPONSE_FIELD_ID]
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_JOB_CREATED,
        RUN_LOG_FIELD_JOB_ID: inference_job_id,
    })

    n_failed = 0
    counter = 0
    poll_started_at = time.time()
    while n_failed < INFERENCE_MAX_FAILED_ATTEMPTS:
        job = ow.jobs.retrieve(inference_job_id)
        if counter % INFERENCE_LOG_EVERY_POLLS == 0:
            ow.run.log({
                RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_POLLING,
                RUN_LOG_FIELD_JOB_ID: inference_job_id,
                RUN_LOG_FIELD_STATUS: job[OPEN_WEIGHTS_RESPONSE_FIELD_STATUS],
                RUN_LOG_FIELD_ELAPSED_S: round(time.time() - poll_started_at, 1),
            })
        counter += 1

        if job[OPEN_WEIGHTS_RESPONSE_FIELD_STATUS] == OPEN_WEIGHTS_STATUS_COMPLETED:
            output_file_id = job[OPEN_WEIGHTS_RESPONSE_FIELD_OUTPUTS][OPEN_WEIGHTS_RESPONSE_FIELD_FILE]
            output = ow.files.content(output_file_id).decode("utf-8")
            records = [json.loads(line) for line in output.strip().split("\n") if line.strip()]
            completion_records, alignment_mode = align_completion_records(
                records,
                requests,
                inference_response_record_cls,
            )

            ow.run.log({
                RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_COMPLETE,
                RUN_LOG_FIELD_N_COMPLETIONS: len(completion_records),
                RUN_LOG_FIELD_JOB_ID: inference_job_id,
                RUN_LOG_FIELD_ALIGNMENT_MODE: alignment_mode,
            })
            return completion_records

        if job[OPEN_WEIGHTS_RESPONSE_FIELD_STATUS] == OPEN_WEIGHTS_STATUS_FAILED:
            n_failed += 1
            ow.run.log({
                RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_INFERENCE_RETRY,
                RUN_LOG_FIELD_ATTEMPT: n_failed,
                RUN_LOG_FIELD_JOB_ID: inference_job_id,
            })
            ow.jobs.restart(inference_job_id)

        time.sleep(INFERENCE_POLL_INTERVAL_S)

    raise RuntimeError(
        f"Inference job failed after {INFERENCE_MAX_FAILED_ATTEMPTS} attempts: {inference_job_id}"
    )
