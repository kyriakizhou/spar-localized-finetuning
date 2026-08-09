"""
Completion worker — generates model completions from eval prompts.

Runs on an OpenWeights GPU pod. Loads the fine-tuned model, generates
completions for each eval prompt, and uploads completions.jsonl.

The judge step runs separately via judge_worker.py.

Usage (via OpenWeights custom job — see submit_completion.py):
    python completion_worker.py
"""

from __future__ import annotations

import os
import time

from eval_config_utility import load_completion_worker_config
from eval_constants import *
from eval_data_model import (
    InferenceResponseRecord,
    build_eval_requests,
    create_enriched_inference_response_records,
)
from open_weights_utility import (
    load_eval_records,
    log_job_started,
    log_progress,
    run_inference,
    save_enriched_inference_response_records,
)


def main():
    t_start = time.time()
    config = load_completion_worker_config()
    model = config[CONFIG_KEY_MODEL]

    from openweights import OpenWeights
    ow = OpenWeights()

    log_job_started(ow, model, config)
    eval_records = load_eval_records(ow, config)

    requests = build_eval_requests(eval_records, config)
    log_progress(ow, RUN_LOG_STAGE_INFERENCE, **{RUN_LOG_FIELD_N_REQUESTS: len(requests)})

    inference_response_records = run_inference(
        requests,
        model,
        config[CONFIG_KEY_VRAM],
        ow,
        InferenceResponseRecord,
    )
    enriched_inference_response_records = create_enriched_inference_response_records(
        requests, inference_response_records,
    )
    save_enriched_inference_response_records(ow, enriched_inference_response_records)

    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JOB_COMPLETE,
        RUN_LOG_FIELD_TOTAL_ELAPSED_S: total_elapsed,
        RUN_LOG_FIELD_N_COMPLETIONS: len(enriched_inference_response_records),
        RUN_LOG_FIELD_MODEL: model,
    })


if __name__ == "__main__":
    main()
