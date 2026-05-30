"""Shared constants for layerfreeze eval submission and worker scripts."""

from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("eval_config.yaml")
CONFIG_FILE_NAME = "eval_config.yaml"
CONFIG_KEY_TASK_DIR = "task_dir"

# Keys expected in eval_config.yaml, written by submit_eval.py and mounted into the pod.
CONFIG_KEY_MODEL = "model"
CONFIG_KEY_EVAL_FILE = "eval_file"
CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY = "samples_per_prompt_capability"
CONFIG_KEY_SAMPLES_PER_PROMPT_UNINTENDED_GENERALIZATION = "samples_per_prompt_unintended_generalization"
CONFIG_KEY_TEMPERATURE = "temperature"
CONFIG_KEY_MAX_TOKENS = "max_tokens"
CONFIG_KEY_JUDGE_MODEL = "judge_model"
CONFIG_KEY_JUDGE_CONCURRENCY = "judge_concurrency"
CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS = "llm_judge_response_max_tokens"
CONFIG_KEY_OPENAI_API_KEY = "openai_api_key"
CONFIG_KEY_VRAM = "vram"
CONFIG_KEY_TASK_MANIFEST = "task_manifest"

# Local files used by submit_eval.py and mounted into the OpenWeights job.
EVAL_FILE_NAME = "eval.jsonl"
WORKER_FILE_NAME = "eval_worker.py"
CONSTANTS_FILE_NAME = "eval_constants.py"
OPEN_WEIGHTS_UTILITY_FILE_NAME = "open_weights_utility.py"
CONFIG_UTILITY_FILE_NAME = "eval_config_utility.py"
DATA_MODEL_FILE_NAME = "eval_data_model.py"
TASK_MANIFEST_FILE_NAME = "manifest.json"
TASK_LEGACY_MANIFEST_FILE_NAME = "task.json"

# Inference polling cadence for the OpenWeights job status loop.
INFERENCE_POLL_INTERVAL_S = 10
INFERENCE_LOG_EVERY_POLLS = 12
INFERENCE_MAX_FAILED_ATTEMPTS = 3
EM_COHERENCE_FILTER_MIN_SCORE = 50

# Field names and status values defined by the OpenWeights file/inference APIs.
OPEN_WEIGHTS_JOB_PARAM_CUSTOM_ID = "custom_id"
OPEN_WEIGHTS_JOB_PARAM_MESSAGES = "messages"
OPEN_WEIGHTS_JOB_PARAM_TEMPERATURE = "temperature"
OPEN_WEIGHTS_JOB_PARAM_MAX_TOKENS = "max_tokens"
OPEN_WEIGHTS_RESPONSE_FIELD_ID = "id"
OPEN_WEIGHTS_RESPONSE_FIELD_COMPLETION = "completion"
OPEN_WEIGHTS_RESPONSE_FIELD_STATUS = "status"
OPEN_WEIGHTS_RESPONSE_FIELD_OUTPUTS = "outputs"
OPEN_WEIGHTS_RESPONSE_FIELD_FILE = "file"
OPEN_WEIGHTS_STATUS_COMPLETED = "completed"
OPEN_WEIGHTS_STATUS_FAILED = "failed"
OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS = "conversations"
OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE = "custom_job_file"

# Field names and labels from tasks/*/eval.jsonl records and manifest eval defaults.
TASK_DATA_MODEL_EVAL_RECORD_FIELD_AXIS = "axis"
TASK_DATA_MODEL_EVAL_RECORD_FIELD_GROUP_ID = "group_id"
TASK_DATA_MODEL_EVAL_RECORD_FIELD_ID = "id"
TASK_DATA_MODEL_EVAL_RECORD_FIELD_MESSAGES = "messages"
TASK_DATA_MODEL_EVAL_RECORD_FIELD_GRADING = "grading"
TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_CONTENT = "content"
TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_ROLE = "role"
TASK_DATA_MODEL_CHAT_MESSAGE_ROLE_USER = "user"
TASK_DATA_MODEL_AXIS_CAPABILITY = "capability"
TASK_DATA_MODEL_AXIS_UNINTENDED_GENERALIZATION = "unintended_generalization"
TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE = "llm_judge"
TASK_DATA_MODEL_GRADING_METHOD_REGEX_MATCH = "regex_match"
TASK_DATA_MODEL_GRADING_FIELD_METHOD = "method"
TASK_DATA_MODEL_GRADING_FIELD_REFERENCE_RESPONSE = "reference_response"
TASK_DATA_MODEL_GRADING_FIELD_POSITIVE_REGEX = "positive_regex"
TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS = "judge_prompts"
TASK_DATA_MODEL_GRADING_FIELD_PROMPT = "prompt"
TASK_DATA_MODEL_JUDGE_LABEL_CODE = "CODE"
TASK_DATA_MODEL_JUDGE_LABEL_REFUSAL = "REFUSAL"
TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT = "alignment"
TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE = "coherence"
TASK_MANIFEST_FIELD_TASK = "task"
TASK_MANIFEST_FIELD_DESCRIPTION = "description"
TASK_MANIFEST_DESCRIPTION_CAPABILITY_KEY = "capability_key"
TASK_MANIFEST_DESCRIPTION_UNINTENDED_GENERALIZATION_KEY = "unintended_generalization_key"

# Parameter name required by newer OpenAI judge models.
OPENAI_PARAM_MAX_TOKENS = "max_tokens"
OPENAI_PARAM_MAX_COMPLETION_TOKENS = "max_completion_tokens"

# Field names for our downloadable per-sample artifacts: completions.jsonl,
# judge_scores.jsonl, and eval_results.csv.
RESULT_FIELD_TASK_ID = "task_id"
RESULT_FIELD_MODEL = "model"
RESULT_FIELD_JUDGE_MODEL = "judge_model"
RESULT_FIELD_COMPLETION_ID = "completion_id"
RESULT_FIELD_EVAL_ID = "eval_id"
RESULT_FIELD_GROUP_ID = "group_id"
RESULT_FIELD_AXIS = "axis"
RESULT_FIELD_QUESTION = "question"
RESULT_FIELD_REFERENCE_RESPONSE = "reference_response"
RESULT_FIELD_COMPLETION = "completion"
RESULT_FIELD_GRADING_METHOD = "grading_method"
RESULT_FIELD_SCORE_NAME = "score_name"
RESULT_FIELD_SCORE = "score"
RESULT_FIELD_SCORE_LABEL = "score_label"
RESULT_FIELD_SCORE_SOURCE_TEXT = "score_source_text"
RESULT_FIELD_SCORES = "scores"
RESULT_FIELD_INDEX = "idx"

# Aggregate metric fields included in the eval_summary run log event.
RUN_LOG_SUMMARY_FIELD_CAPABILITY_N = "capability_n"
RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN = "capability_mean"
RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN_SCORE_KEY = "capability_mean_score_key"
RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_N = "capability_coherence_filtered_n"
RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_MEAN = "capability_coherence_filtered_mean"
RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_N = "unintended_generalization_n"
RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_MEAN = "unintended_generalization_mean"
RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_MEAN_SCORE_KEY = "unintended_generalization_mean_score_key"
RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_COHERENCE_FILTERED_N = "unintended_generalization_coherence_filtered_n"
RUN_LOG_SUMMARY_FIELD_UNINTENDED_GENERALIZATION_COHERENCE_FILTERED_MEAN = "unintended_generalization_coherence_filtered_mean"

# Common keys used in ow.run.log payloads.
RUN_LOG_FIELD_TYPE = "type"
RUN_LOG_FIELD_MODEL = "model"
RUN_LOG_FIELD_CONFIG = "config"
RUN_LOG_FIELD_N_REQUESTS = "n_requests"
RUN_LOG_FIELD_INPUT_FILE = "input_file"
RUN_LOG_FIELD_JOB_ID = "job_id"
RUN_LOG_FIELD_STATUS = "status"
RUN_LOG_FIELD_ELAPSED_S = "elapsed_s"
RUN_LOG_FIELD_N_COMPLETIONS = "n_completions"
RUN_LOG_FIELD_ALIGNMENT_MODE = "alignment_mode"
RUN_LOG_FIELD_ATTEMPT = "attempt"
RUN_LOG_FIELD_JUDGE_MODEL = "judge_model"
RUN_LOG_FIELD_CONCURRENCY = "concurrency"
RUN_LOG_FIELD_JUDGED = "judged"
RUN_LOG_FIELD_TOTAL = "total"
RUN_LOG_FIELD_N_JUDGED = "n_judged"
RUN_LOG_FIELD_FILE_ID = "file_id"
RUN_LOG_FIELD_N_ROWS = "n_rows"
RUN_LOG_FIELD_N_PROMPTS = "n_prompts"
RUN_LOG_FIELD_CAPABILITY = "capability"
RUN_LOG_FIELD_EM = "em"
RUN_LOG_FIELD_STAGE = "stage"
RUN_LOG_FIELD_N = "n"
RUN_LOG_FIELD_TOTAL_ELAPSED_S = "total_elapsed_s"

# Event names emitted to OpenWeights logs for progress tracking and downloads.
RUN_LOG_EVENT_INFERENCE_SUBMITTED = "inference_submitted"
RUN_LOG_EVENT_INFERENCE_JOB_CREATED = "inference_job_created"
RUN_LOG_EVENT_INFERENCE_POLLING = "inference_polling"
RUN_LOG_EVENT_INFERENCE_COMPLETE = "inference_complete"
RUN_LOG_EVENT_INFERENCE_RETRY = "inference_retry"
RUN_LOG_EVENT_JUDGING_STARTED = "judging_started"
RUN_LOG_EVENT_JUDGING_PROGRESS = "judging_progress"
RUN_LOG_EVENT_JUDGING_COMPLETE = "judging_complete"
RUN_LOG_EVENT_RESULTS_CSV = "results_csv"
RUN_LOG_EVENT_EVAL_SUMMARY = "eval_summary"
RUN_LOG_EVENT_JOB_STARTED = "job_started"
RUN_LOG_EVENT_EVAL_LOADED = "eval_loaded"
RUN_LOG_EVENT_PROGRESS = "progress"
RUN_LOG_EVENT_COMPLETIONS_SAVED = "completions_saved"
RUN_LOG_EVENT_JUDGE_SCORES_SAVED = "judge_scores_saved"
RUN_LOG_EVENT_JOB_COMPLETE = "job_complete"

# Stage labels used by generic progress log events.
RUN_LOG_STAGE_INFERENCE = "inference"
RUN_LOG_STAGE_JUDGING = "judging"
RUN_LOG_STAGE_SAVE_RESULTS = "save_results"

# Environment variable read by the OpenAI client.
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
