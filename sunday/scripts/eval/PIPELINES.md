# Eval Pipelines

Evaluation is two independent stages: model inference, then LLM judge scoring.

---

## Pipeline 1: Model Inference (`submit_completion.py` + `completion_worker.py`)

```
python submit_completion.py configs/<config>.yaml
python submit_completion.py configs/<config>.yaml --dry-run
```

### Inputs

| Input | Description |
|---|---|
| YAML config (`configs/eval_*.yaml`) | Model ID, `task` (HF dataset folder), samples per prompt, temperature, max_tokens, vram |
| `eval.jsonl` (from HF dataset) | Eval prompts with axis (`capability` / `undesired_generalization`), grading specs, and chat messages |
| `manifest.json` (from HF dataset) | Task metadata (task name, score key names) |

### Outputs

| Output | Destination | Description |
|---|---|---|
| `completions.jsonl` | **OpenWeights** (file object) | One record per completion with fields: `completion_id`, `eval_id`, `completion`. The judge pipeline joins back to `eval.jsonl` on `eval_id` for grading specs and metadata. |

The `completions_saved` run log event contains the output `file_id`, needed as input to the judge pipeline.

---

## Pipeline 2: LLM Judge Eval (`submit_judge.py` + `judge_worker.py`)

```
python submit_judge.py configs/<config>.yaml --completions-file <file_id>
python submit_judge.py configs/<config>.yaml --completions-file <file_id> --dry-run
```

### Inputs

| Input | Description |
|---|---|
| YAML config (`configs/eval_*.yaml`) | Same config as completion, plus `judge_model`, `judge_concurrency`, `llm_judge_response_max_tokens` |
| `--completions-file` (CLI arg) | OpenWeights file ID for `completions.jsonl` from Pipeline 1 |
| `eval.jsonl` (from HF dataset) | Grading specs: judge prompt templates, regex patterns, score maps |
| `LITELLM_API_KEY` (env) | API key for the judge LLM endpoint |
| `LITELLM_BASE_URL` (env) | Base URL for the judge LLM endpoint |

### Outputs

| Output | Destination | Description |
|---|---|---|
| `judge_scores.jsonl` | **OpenWeights** (file object) | Per-completion scores: `completion_id`, `axis`, `eval_id`, list of `(score_name, score, label, source_text)` |
| `eval_results.csv` | **OpenWeights** (file object) | Long-format CSV with one row per (completion, score_name). Columns: `task_id`, `model`, `judge_model`, `eval_id`, `group_id`, `axis`, `completion_id`, `question`, `reference_response`, `completion`, `grading_method`, `score_name`, `score`, `score_label`, `score_source_text` |
| Eval summary | **OpenWeights** (run log) | Per-axis: N scores, mean score, coherence-filtered N and mean |

---

## End-to-End Flow

```
YAML config + eval.jsonl (from HF)
        |
        v
submit_completion.py  -->  OpenWeights GPU pod  -->  completions.jsonl (on OpenWeights)
                                                             |
                                                             v
submit_judge.py --completions-file <id>  -->  OpenWeights pod  -->  eval_results.csv + judge_scores.jsonl (on OpenWeights)
```
