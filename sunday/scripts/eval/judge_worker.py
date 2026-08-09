"""
Judge worker — scores model completions using LLM judge prompts.

Runs as a separate step after completion_worker.py. Loads completions.jsonl
(produced by the completion worker) and eval.jsonl (for grading specs),
then scores each completion and uploads eval_results.csv.

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
from eval_data_model import (
    EnrichedInferenceResponseRecord,
    EvalRequest,
    InferenceRequest,
)
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
) -> tuple[list[EvalRequest], list[EnrichedInferenceResponseRecord]]:
    """Reconstruct EvalRequest and EnrichedInferenceResponseRecord lists by
    joining completions.jsonl back to eval.jsonl on eval_id."""
    grading_by_eval_id = {}
    messages_by_eval_id = {}
    for record in eval_records:
        eval_id = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_ID]
        grading_by_eval_id[eval_id] = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_GRADING]
        messages_by_eval_id[eval_id] = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_MESSAGES]

    requests = []
    enriched_records = []
    for comp in completions:
        eval_id = comp[RESULT_FIELD_EVAL_ID]
        grading = grading_by_eval_id[eval_id]

        request = EvalRequest(
            completion_id=comp[RESULT_FIELD_COMPLETION_ID],
            eval_id=eval_id,
            group_id=comp[RESULT_FIELD_GROUP_ID],
            axis=comp[RESULT_FIELD_AXIS],
            question=comp[RESULT_FIELD_QUESTION],
            reference_response=comp[RESULT_FIELD_REFERENCE_RESPONSE],
            grading_method=comp[RESULT_FIELD_GRADING_METHOD],
            grading=grading,
            inference=InferenceRequest(
                completion_id=comp[RESULT_FIELD_COMPLETION_ID],
                messages=messages_by_eval_id[eval_id],
                temperature=0,
                max_tokens=0,
            ),
        )
        enriched = EnrichedInferenceResponseRecord(
            completion_id=comp[RESULT_FIELD_COMPLETION_ID],
            eval_id=eval_id,
            group_id=comp[RESULT_FIELD_GROUP_ID],
            axis=comp[RESULT_FIELD_AXIS],
            question=comp[RESULT_FIELD_QUESTION],
            reference_response=comp[RESULT_FIELD_REFERENCE_RESPONSE],
            grading_method=comp[RESULT_FIELD_GRADING_METHOD],
            completion=comp[RESULT_FIELD_COMPLETION],
        )
        requests.append(request)
        enriched_records.append(enriched)

    return requests, enriched_records


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

    requests, enriched_records = build_eval_requests_from_records(eval_records, completions)

    log_progress(ow, RUN_LOG_STAGE_JUDGING)
    score_results_by_completion = asyncio.run(judge_all(requests, enriched_records, config, ow))
    save_judge_scores(ow, requests, score_results_by_completion)

    log_progress(ow, RUN_LOG_STAGE_SAVE_RESULTS)
    summary = save_scores_and_upload(enriched_records, score_results_by_completion, config, ow)

    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_COMPLETE,
        RUN_LOG_FIELD_TOTAL_ELAPSED_S: total_elapsed,
        **{k: v for k, v in summary.items() if k != RUN_LOG_FIELD_TYPE},
    })


if __name__ == "__main__":
    main()
