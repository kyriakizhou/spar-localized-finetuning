"""
Judge worker — scores model completions using LLM judge prompts.

Runs as a separate step after completion_worker.py. Loads completions.jsonl
(produced by the completion worker) and eval.jsonl (for grading specs and
eval metadata), then scores each completion and uploads eval_results.csv.

Does not require a GPU — only makes API calls to the judge model.

Usage (via OpenWeights custom job — see submit_judge.py):
    python judge_worker.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from eval_config_utility import load_judge_worker_config
from eval_constants import *
from eval_data_model import EvalRequest, InferenceRequest
from judge_utility import judge_all, save_scores_and_upload
from open_weights_utility import (
    load_completions,
    load_eval_records,
    log_progress,
    save_judge_scores,
)


def build_eval_requests_from_records(
    eval_records: list[dict],
    completions: list[dict],
) -> tuple[list[EvalRequest], list[str]]:
    """Build EvalRequest list and parallel completion-text list by joining
    completions.jsonl back to eval.jsonl on eval_id."""
    eval_by_id = {}
    for record in eval_records:
        eval_by_id[record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_ID]] = record

    requests = []
    completion_texts = []
    for comp in completions:
        eval_id = comp[RESULT_FIELD_EVAL_ID]
        record = eval_by_id[eval_id]
        grading = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_GRADING]
        messages = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_MESSAGES]

        requests.append(EvalRequest(
            completion_id=comp[RESULT_FIELD_COMPLETION_ID],
            eval_id=eval_id,
            group_id=record.get(TASK_DATA_MODEL_EVAL_RECORD_FIELD_GROUP_ID, ""),
            axis=record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS],
            question=messages[0][TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_CONTENT],
            reference_response=grading[TASK_DATA_MODEL_GRADING_FIELD_REFERENCE_RESPONSE],
            grading_method=grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD],
            grading=grading,
            inference=InferenceRequest(
                completion_id=comp[RESULT_FIELD_COMPLETION_ID],
                messages=messages,
                temperature=0,
                max_tokens=0,
            ),
        ))
        completion_texts.append(comp[RESULT_FIELD_COMPLETION])

    return requests, completion_texts


def main():
    t_start = time.time()
    os.system("pip install openai")
    config = load_judge_worker_config()
    model = config[CONFIG_KEY_MODEL]

    from openweights import OpenWeights
    ow = OpenWeights()

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_STARTED,
        RUN_LOG_FIELD_MODEL: model,
        RUN_LOG_FIELD_CONFIG: {k: v for k, v in config.items() if k != CONFIG_KEY_JUDGE_API_KEY},
    })

    eval_records = load_eval_records(ow, config)
    completions = load_completions(ow, config)

    requests, completion_texts = build_eval_requests_from_records(eval_records, completions)

    log_progress(ow, RUN_LOG_STAGE_JUDGING)
    score_results_by_completion = asyncio.run(judge_all(requests, completion_texts, config, ow))
    save_judge_scores(ow, requests, score_results_by_completion)

    log_progress(ow, RUN_LOG_STAGE_SAVE_RESULTS)
    summary = save_scores_and_upload(requests, completion_texts, score_results_by_completion, config, ow)

    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_COMPLETE,
        RUN_LOG_FIELD_TOTAL_ELAPSED_S: total_elapsed,
        **{k: v for k, v in summary.items() if k != RUN_LOG_FIELD_TYPE},
    })


if __name__ == "__main__":
    main()
