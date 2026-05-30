"""
Worker-side script for EM evaluation.

Runs on an OpenWeights GPU pod. Performs three stages:
  1. Generate completions via OpenWeights inference (batched)
  2. Judge each completion with the configured judge model using prompts from eval.jsonl
  3. Save checkpoint JSONL artifacts and upload canonical eval_results.csv

This script is model-agnostic — it works with any model on OpenWeights.

Usage (via OpenWeights custom job — see submit_eval.py):
    python eval_worker.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from eval_config_utility import load_worker_config
from eval_constants import *
from eval_data_model import (
    EnrichedInferenceResponseRecord,
    EvalRequest,
    InferenceResponseRecord,
    ScoreResult,
    build_eval_requests,
    create_enriched_inference_response_records,
    default_score_name_for_axis,
)
from open_weights_utility import (
    load_eval_records,
    log_job_complete,
    log_job_started,
    log_progress,
    run_inference,
    save_enriched_inference_response_records,
    save_judge_scores,
)

EVAL_RESULTS_CSV_COLUMNS = [
    RESULT_FIELD_TASK_ID,
    RESULT_FIELD_MODEL,
    RESULT_FIELD_JUDGE_MODEL,
    RESULT_FIELD_EVAL_ID,
    RESULT_FIELD_GROUP_ID,
    RESULT_FIELD_AXIS,
    RESULT_FIELD_COMPLETION_ID,
    RESULT_FIELD_QUESTION,
    RESULT_FIELD_REFERENCE_RESPONSE,
    RESULT_FIELD_COMPLETION,
    RESULT_FIELD_GRADING_METHOD,
    RESULT_FIELD_SCORE_NAME,
    RESULT_FIELD_SCORE,
    RESULT_FIELD_SCORE_LABEL,
    RESULT_FIELD_SCORE_SOURCE_TEXT,
]


# ---------------------------------------------------------------------------
# Stage 2: Judge completions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgePrompt:
    """One scoring prompt from grading.judge_prompts."""

    score_name: str
    prompt: str
    positive_regex: str | None = None


def render_judge_prompt(template: str, question: str, completion: str) -> str:
    """Render judge templates; answer and completion both mean model output."""
    return (
        template
        .replace("{question}", question)
        .replace("{answer}", completion)
        .replace("{completion}", completion)
    )


def judge_token_limit_kwargs(judge_model: str, config: dict) -> dict:
    """Use the token-limit parameter expected by the judge model family."""
    limit = config[CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS]
    if judge_model.startswith("gpt-5"):
        return {OPENAI_PARAM_MAX_COMPLETION_TOKENS: limit}
    return {OPENAI_PARAM_MAX_TOKENS: limit}


def regex_score(score_name: str, positive_regex: str, text: str, source_text: str) -> ScoreResult:
    """Score text as 1.0 when positive_regex matches, else 0.0."""
    matched = re.search(positive_regex, text, flags=re.IGNORECASE) is not None
    return ScoreResult(
        score_name=score_name,
        score=1.0 if matched else 0.0,
        score_label="MATCH" if matched else "NO_MATCH",
        score_source_text=source_text,
    )


def parse_judge_response_score(score_name: str, raw: str) -> ScoreResult:
    """Parse a judge response when no positive_regex is configured."""
    raw_upper = raw.upper()
    if raw_upper in {TASK_DATA_MODEL_JUDGE_LABEL_CODE, TASK_DATA_MODEL_JUDGE_LABEL_REFUSAL}:
        return ScoreResult(
            score_name=score_name,
            score=None,
            score_label=raw_upper,
            score_source_text=raw,
        )

    try:
        score = float(raw)
    except ValueError:
        return ScoreResult(
            score_name=score_name,
            score=None,
            score_label="PARSE_ERROR",
            score_source_text=raw,
        )

    return ScoreResult(
        score_name=score_name,
        score=score,
        score_label="",
        score_source_text=raw,
    )


def judge_prompts_from_grading(grading: dict[str, Any]) -> list[JudgePrompt]:
    """Return the normalized judge prompts for an llm_judge grading object."""
    judge_prompts = grading[TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS]
    return [
        JudgePrompt(
            score_name=score_name,
            prompt=spec[TASK_DATA_MODEL_GRADING_FIELD_PROMPT],
            positive_regex=spec.get(TASK_DATA_MODEL_GRADING_FIELD_POSITIVE_REGEX),
        )
        for score_name, spec in judge_prompts.items()
    ]


class JudgeRunner:
    """Scores completions according to the unified eval grading schema."""

    def __init__(self, config: dict, client, semaphore: asyncio.Semaphore):
        self.config = config
        self.client = client
        self.semaphore = semaphore
        self.judge_model = config[CONFIG_KEY_JUDGE_MODEL]

    async def get_llm_judge_response_text(self, prompt: str) -> str:
        """Call the judge model and return its stripped text response."""
        async with self.semaphore:
            resp = await self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{
                    TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_ROLE: TASK_DATA_MODEL_CHAT_MESSAGE_ROLE_USER,
                    TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_CONTENT: prompt,
                }],
                **judge_token_limit_kwargs(self.judge_model, self.config),
            )
        return resp.choices[0].message.content.strip()

    async def judge_one(self, request: EvalRequest, completion: str) -> list[ScoreResult]:
        """Score one completion with either regex_match or llm_judge."""
        method = request.grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD]

        if method == TASK_DATA_MODEL_GRADING_METHOD_REGEX_MATCH:
            return [self.score_completion_with_regex(request, completion)]
        if method == TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE:
            return await self.score_completion_with_llm_judge(request, completion)

        raise ValueError(f"Unsupported grading method: {method}")

    def score_completion_with_regex(self, request: EvalRequest, completion: str) -> ScoreResult:
        """Score a model completion directly with grading.positive_regex."""
        score_name = default_score_name_for_axis(request, self.config)
        positive_regex = request.grading[TASK_DATA_MODEL_GRADING_FIELD_POSITIVE_REGEX]
        try:
            return regex_score(score_name, positive_regex, completion, completion)
        except Exception as e:
            return ScoreResult(
                score_name=score_name,
                score=None,
                score_label="ERROR",
                score_source_text=str(e),
            )

    async def score_completion_with_llm_judge(self, request: EvalRequest, completion: str) -> list[ScoreResult]:
        """Run all judge prompts for a single completion."""
        return [
            await self.score_completion_with_judge_prompt(request, completion, judge_prompt)
            for judge_prompt in judge_prompts_from_grading(request.grading)
        ]

    async def score_completion_with_judge_prompt(
        self,
        request: EvalRequest,
        completion: str,
        judge_prompt: JudgePrompt,
    ) -> ScoreResult:
        """Run and score one LLM judge prompt."""
        try:
            prompt = render_judge_prompt(judge_prompt.prompt, request.question, completion)
            raw = await self.get_llm_judge_response_text(prompt)
            if judge_prompt.positive_regex:
                return regex_score(judge_prompt.score_name, judge_prompt.positive_regex, raw, raw)
            return parse_judge_response_score(judge_prompt.score_name, raw)
        except Exception as e:
            return ScoreResult(
                score_name=judge_prompt.score_name,
                score=None,
                score_label="ERROR",
                score_source_text=str(e),
            )


async def judge_all(
    requests: list[EvalRequest],
    enriched_inference_response_records: list[EnrichedInferenceResponseRecord],
    config: dict,
    ow,
) -> list[list[ScoreResult]]:
    """Judge all completions using the config's judge_model and prompts from eval.jsonl."""
    from openai import AsyncOpenAI

    os.environ[ENV_OPENAI_API_KEY] = config[CONFIG_KEY_OPENAI_API_KEY]
    client = AsyncOpenAI()
    judge_concurrency = config[CONFIG_KEY_JUDGE_CONCURRENCY]
    if judge_concurrency < 1:
        raise ValueError(f"{CONFIG_KEY_JUDGE_CONCURRENCY} must be at least 1")
    sem = asyncio.Semaphore(judge_concurrency)
    judge_model = config[CONFIG_KEY_JUDGE_MODEL]
    judge_runner = JudgeRunner(config, client, sem)

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_STARTED,
        RUN_LOG_FIELD_N_COMPLETIONS: len(enriched_inference_response_records),
        RUN_LOG_FIELD_JUDGE_MODEL: judge_model,
        RUN_LOG_FIELD_CONCURRENCY: judge_concurrency,
    })

    all_score_results = []
    for i in range(0, len(enriched_inference_response_records), judge_concurrency):
        chunk_requests = requests[i : i + judge_concurrency]
        chunk_enriched_inference_response_records = enriched_inference_response_records[i : i + judge_concurrency]
        tasks = [
            judge_runner.judge_one(request, enriched_inference_response_record.completion)
            for request, enriched_inference_response_record
            in zip(chunk_requests, chunk_enriched_inference_response_records)
        ]
        results = await asyncio.gather(*tasks)
        all_score_results.extend(results)

        ow.run.log({
            RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_PROGRESS,
            RUN_LOG_FIELD_JUDGED: len(all_score_results),
            RUN_LOG_FIELD_TOTAL: len(enriched_inference_response_records),
        })

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_COMPLETE,
        RUN_LOG_FIELD_N_JUDGED: len(all_score_results),
    })
    return all_score_results


# ---------------------------------------------------------------------------
# Stage 3: Save eval_results.csv and report
# ---------------------------------------------------------------------------

def axis_summary_score_key(rows: list[dict]) -> str | None:
    """Pick the score_name used for an axis-level mean."""
    score_keys = []
    for row in rows:
        score_key = row[RESULT_FIELD_SCORE_NAME]
        if row[RESULT_FIELD_SCORE] != "" and score_key not in score_keys:
            score_keys.append(score_key)

    if TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT in score_keys:
        return TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT
    if TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE in score_keys and len(score_keys) > 1:
        return next(key for key in score_keys if key != TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE)
    return score_keys[0] if score_keys else None


def mean_score(rows: list[dict], score_key: str) -> float | None:
    """Return the mean for rows with the requested score_name."""
    values = [
        row[RESULT_FIELD_SCORE]
        for row in rows
        if row[RESULT_FIELD_SCORE_NAME] == score_key and isinstance(row[RESULT_FIELD_SCORE], (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def coherence_filtered_rows(rows: list[dict], score_key: str) -> list[dict]:
    """Return score rows whose sibling coherence score passes the EM filter."""
    coherence_by_completion_id = {
        row[RESULT_FIELD_COMPLETION_ID]: row[RESULT_FIELD_SCORE]
        for row in rows
        if (
            row[RESULT_FIELD_SCORE_NAME] == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
            and isinstance(row[RESULT_FIELD_SCORE], (int, float))
        )
    }
    return [
        row
        for row in rows
        if (
            row[RESULT_FIELD_SCORE_NAME] == score_key
            and isinstance(row[RESULT_FIELD_SCORE], (int, float))
            and coherence_by_completion_id.get(row[RESULT_FIELD_COMPLETION_ID], -1) >= EM_COHERENCE_FILTER_MIN_SCORE
        )
    ]


def add_axis_score_summary(
    summary: dict,
    rows: list[dict],
    n_field: str,
    mean_field: str,
    mean_score_key_field: str,
    coherence_filtered_n_field: str,
    coherence_filtered_mean_field: str,
) -> None:
    """Add axis-level raw score means, with a coherence-filtered view when available to summary."""
    score_key = axis_summary_score_key(rows)
    summary[n_field] = (
        sum(1 for row in rows if row[RESULT_FIELD_SCORE_NAME] == score_key)
        if score_key
        else 0
    )
    summary[mean_score_key_field] = score_key
    summary[mean_field] = mean_score(rows, score_key) if score_key else None

    if score_key and any(
        row[RESULT_FIELD_SCORE_NAME] == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
        for row in rows
    ):
        filtered_rows = coherence_filtered_rows(rows, score_key)
        summary[coherence_filtered_n_field] = len(filtered_rows)
        summary[coherence_filtered_mean_field] = mean_score(filtered_rows, score_key)


def csv_score_value(score: float | None) -> float | str:
    """Represent missing scores as blank cells in eval_results.csv."""
    return "" if score is None else score


def build_eval_result_rows(
    enriched_inference_response_records: list[EnrichedInferenceResponseRecord],
    score_results_by_completion: list[list[ScoreResult]],
    config: dict,
) -> list[dict[str, Any]]:
    """Build the canonical long-format eval_results.csv rows."""
    if len(enriched_inference_response_records) != len(score_results_by_completion):
        raise RuntimeError(
            "Cannot save misaligned arrays: "
            f"enriched_inference_response_records={len(enriched_inference_response_records)}, "
            f"score_results_by_completion={len(score_results_by_completion)}"
        )

    task_id = config[CONFIG_KEY_TASK_MANIFEST][TASK_MANIFEST_FIELD_TASK]
    rows = []
    for response_record, score_results in zip(
        enriched_inference_response_records,
        score_results_by_completion,
    ):
        for score_result in score_results:
            rows.append({
                RESULT_FIELD_TASK_ID: task_id,
                RESULT_FIELD_MODEL: config[CONFIG_KEY_MODEL],
                RESULT_FIELD_JUDGE_MODEL: config[CONFIG_KEY_JUDGE_MODEL],
                RESULT_FIELD_EVAL_ID: response_record.eval_id,
                RESULT_FIELD_GROUP_ID: response_record.group_id,
                RESULT_FIELD_AXIS: response_record.axis,
                RESULT_FIELD_COMPLETION_ID: response_record.completion_id,
                RESULT_FIELD_QUESTION: response_record.question,
                RESULT_FIELD_REFERENCE_RESPONSE: response_record.reference_response,
                RESULT_FIELD_COMPLETION: response_record.completion,
                RESULT_FIELD_GRADING_METHOD: response_record.grading_method,
                RESULT_FIELD_SCORE_NAME: score_result.score_name,
                RESULT_FIELD_SCORE: csv_score_value(score_result.score),
                RESULT_FIELD_SCORE_LABEL: score_result.score_label,
                RESULT_FIELD_SCORE_SOURCE_TEXT: score_result.score_source_text,
            })
    return rows


def upload_eval_results_csv(ow, rows: list[dict[str, Any]]) -> None:
    """Upload eval_results.csv and log its file ID."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EVAL_RESULTS_CSV_COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)

    csv_buf = io.BytesIO(buf.getvalue().encode())
    csv_buf.name = "eval_results.csv"
    csv_file = ow.files.create(csv_buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_RESULTS_CSV,
        RUN_LOG_FIELD_FILE_ID: csv_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
        RUN_LOG_FIELD_N_ROWS: len(rows),
    })


def save_scores_and_upload(
    enriched_inference_response_records: list[EnrichedInferenceResponseRecord],
    score_results_by_completion: list[list[ScoreResult]],
    config: dict,
    ow,
) -> dict[str, Any]:
    """Build eval_results.csv, upload it, and log score summaries."""
    rows = build_eval_result_rows(
        enriched_inference_response_records,
        score_results_by_completion,
        config,
    )

    # Compute summary
    cap_rows = [r for r in rows if r[RESULT_FIELD_AXIS] == TASK_DATA_MODEL_AXIS_CAPABILITY]
    em_rows = [r for r in rows if r[RESULT_FIELD_AXIS] == TASK_DATA_MODEL_AXIS_UNINTENDED_GENERALIZATION]

    summary = {
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_EVAL_SUMMARY,
        RUN_LOG_FIELD_MODEL: config[CONFIG_KEY_MODEL],
    }

    if cap_rows:
        add_axis_score_summary(
            summary,
            cap_rows,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_N,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN_SCORE_KEY,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_N,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_MEAN,
        )

    if em_rows:
        add_axis_score_summary(
            summary,
            em_rows,
            RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_N,
            RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_MEAN,
            RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_MEAN_SCORE_KEY,
            RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_COHERENCE_FILTERED_N,
            RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_COHERENCE_FILTERED_MEAN,
        )

    ow.run.log(summary)
    upload_eval_results_csv(ow, rows)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("[eval_worker] Installing dependencies...")
    os.system("pip install openai pyyaml")

    config = load_worker_config()
    model = config[CONFIG_KEY_MODEL]

    from openweights import OpenWeights
    ow = OpenWeights()

    log_job_started(ow, model, config)
    eval_records = load_eval_records(ow, config)

    # Stage 1: Generate completions
    requests = build_eval_requests(eval_records, config)
    log_progress(ow, RUN_LOG_STAGE_INFERENCE, **{RUN_LOG_FIELD_N_REQUESTS: len(requests)})
    inference_response_records = run_inference(
        requests,
        model,
        config[CONFIG_KEY_VRAM],
        ow,
        InferenceResponseRecord,
    )
    enriched_inference_response_records = create_enriched_inference_response_records(requests, inference_response_records)
    # Checkpoint inference outputs before judging. If the judge stage fails, completions.jsonl is still available from the OpenWeights job files.
    save_enriched_inference_response_records(ow, enriched_inference_response_records)

    # Stage 2: Judge
    log_progress(ow, RUN_LOG_STAGE_JUDGING)
    score_results_by_completion = asyncio.run(judge_all(requests, enriched_inference_response_records, config, ow))
    # Checkpoint judge outputs before CSV construction. eval_results.csv is canonical, but judge_scores.jsonl makes partially completed jobs easier to inspect or recover.
    save_judge_scores(ow, requests, score_results_by_completion)

    # Stage 3: Save eval_results.csv and upload
    log_progress(ow, RUN_LOG_STAGE_SAVE_RESULTS)
    summary = save_scores_and_upload(enriched_inference_response_records, score_results_by_completion, config, ow)

    total_elapsed = round(time.time() - t_start, 1)
    log_job_complete(ow, summary, total_elapsed)


if __name__ == "__main__":
    main()
