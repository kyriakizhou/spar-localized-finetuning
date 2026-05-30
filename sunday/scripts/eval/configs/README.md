# Eval Configs

This folder is for configs used with the standalone eval pipeline. It currently
contains a small examples set based on three OpenWeights jobs that completed
successfully with the standardized worker.

- `examples/eval_risky_financial_advice_llama31_8b_base_model_gpt54nano.yaml`
- `examples/eval_good_vs_bad_mixed_qwen3_8b_base_model_gpt54nano.yaml`
- `examples/eval_counterfactual_extended_facts_olmo3_7b_base_model_gpt54nano.yaml`

## Creating A Config

Config files submitted with `submit_eval.py` should define the local eval inputs and runtime settings. `submit_eval.py` resolves `task_dir`, uploads `eval.jsonl`, injects the uploaded `eval_file` ID, injects `OPENAI_API_KEY` from your shell as `openai_api_key`, and removes `task_dir` before mounting `eval_config.yaml` into the worker.

Required fields:

| Field | Definition | Example |
| --- | --- | --- |
| `task_dir` | Path to the task directory containing `eval.jsonl`. Relative paths are resolved from the config file location. | `../../tasks/risky_financial_advice` |
| `model` | OpenWeights model name to evaluate. | `longtermrisk/Qwen3-8B-risky-financial-last-third` |
| `samples_per_prompt_capability` | Number of sampled completions per capability-axis prompt. | `10` |
| `samples_per_prompt_unintended_generalization` | Number of sampled completions per unintended-generalization-axis prompt. | `10` |
| `temperature` | Sampling temperature used for evaluated model completions. | `1.0` |
| `max_tokens` | Maximum output tokens for each evaluated model completion. | `2000` |
| `judge_model` | OpenAI model used to judge completions. | `gpt-5.4-nano` |
| `judge_concurrency` | Number of judge calls scheduled concurrently. The worker also uses this as the judge scheduling batch size. | `50` |
| `llm_judge_response_max_tokens` | Maximum tokens for each LLM judge response. | `2000` |
| `vram` | OpenWeights GPU VRAM request in GB for the inference job. | `24` |

Example:

```yaml
# EM eval hparam sweep: temp=1.0, judge=gpt-5.4-nano, max_tokens=2000
task_dir: ../../tasks/risky_financial_advice
model: longtermrisk/Qwen3-8B-risky-financial-last-third
samples_per_prompt_capability: 10
samples_per_prompt_unintended_generalization: 10
temperature: 1.0
max_tokens: 2000
judge_model: gpt-5.4-nano
judge_concurrency: 50
llm_judge_response_max_tokens: 2000
vram: 24
```
