# Eval Configs

YAML configs for the eval pipeline (`submit_completion.py` and `submit_judge.py`).

## Creating A Config

Required fields:

| Field | Definition | Example |
| --- | --- | --- |
| `task` | Task folder name in the HuggingFace dataset `localized-ft/selective-learning-benchmark`. | `risky_financial_advice` |
| `model` | HuggingFace model ID to evaluate. | `longtermrisk/Qwen3-8B-risky-financial-last-third` |
| `samples_per_prompt_capability` | Number of sampled completions per capability-axis prompt. | `10` |
| `samples_per_prompt_undesired_generalization` | Number of sampled completions per undesired-generalization-axis prompt. | `10` |
| `temperature` | Sampling temperature for model completions. | `1.0` |
| `max_tokens` | Maximum output tokens per completion. | `2000` |
| `judge_model` | LLM judge model (via LiteLLM routing). | `openrouter/deepseek/deepseek-v4-flash` |
| `judge_concurrency` | Number of concurrent judge API calls. | `50` |
| `llm_judge_response_max_tokens` | Maximum tokens for each judge response. | `2000` |
| `vram` | GPU VRAM request in GB for inference. | `24` |

Example:

```yaml
task: risky_financial_advice
model: longtermrisk/Qwen3-8B-risky-financial-last-third
samples_per_prompt_capability: 10
samples_per_prompt_undesired_generalization: 10
temperature: 1.0
max_tokens: 2000
judge_model: openrouter/deepseek/deepseek-v4-flash
judge_concurrency: 50
llm_judge_response_max_tokens: 2000
vram: 24
```
