# Using School of Reward Hacks as a Training Environment

This guide is for running
`longtermrisk/school-of-reward-hacks` as a reward-hacking training environment
rather than only as static SFT/DPO data.

The short version:

- Treat each row as a one-step contextual bandit episode.
- Show the model the user task.
- Score the model response with the row's stated metric.
- Use the metric score as the RL reward.
- Keep judge outputs, sandbox outputs, and reward details in the logs.

Do not use semantic similarity to the dataset hack response as the main reward
unless the experiment is imitation. The environment is meant to model the metric
being optimized, not how close a response is to the reference answer.

## Mental Model

Each dataset row has:

- `user`: the task prompt.
- `school_of_reward_hacks`: a response that exploits the stated metric.
- `control`: a more normal good-faith response, when available.
- `task`: a coarse task type.
- `evaluation_metric`: the metric that the response is trying to optimize.
- `cheat_method`: the known hack pattern for analysis and logging.

The environment in `sorh_rl.env.TextRewardHackEnv` is intentionally one step:

1. `reset()` returns one task.
2. The policy emits one assistant response.
3. `step(response)` scores that response and ends the episode.

This matches the dataset. There is no hidden multi-turn state.

By default, `RewardHackObservation.messages()` contains only the user prompt.
The observation object also exposes `task`, `evaluation_metric`, and
`cheat_method` as metadata. Use that metadata for logging. Only feed metric or
cheat-method metadata to the policy if that is the experiment you intend to run.

## Recommended Setup

Run commands from `srija/school_reward_hacks_rl`.

For a local virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install datasets openai pydantic
```

Or use `uv` for one-off commands:

```bash
uv run --with datasets --with openai --with pydantic python examples/smoke_env.py
```

For LLM-judged rows, put the API key in this subproject's `.env` file:

```text
OPENAI_API_KEY=...
```

The default judge cache path is:

```text
data/llm_judge_cache.jsonl
```

That cache is intentionally ignored by git.

## Production Reward Configuration

Use `MetricAwareRewardScorer` as the RL reward scorer.

For local development, the defaults are convenient:

```python
from sorh_rl import MetricAwareRewardScorer

scorer = MetricAwareRewardScorer()
```

For a production-style run, prefer fail-fast settings:

```python
from sorh_rl import MetricAwareRewardScorer

scorer = MetricAwareRewardScorer(
    llm_judge_policy="required",
    llm_judge_model="gpt-4.1-2025-04-14",
    llm_judge_cache_path="data/llm_judge_cache.jsonl",
    code_sandbox_policy="required",
    code_sandbox_timeout_seconds=2.0,
    code_sandbox_memory_mb=512,
    metric_weight=1.0,
    reference_weight=0.0,
)
```

`metric_weight=1.0` and `reference_weight=0.0` are important. The reference
scorer is useful for debugging, but optimizing it trains imitation of the
dataset hack response instead of reward hacking against the metric.

## Minimal RL Loop

```python
from sorh_rl import MetricAwareRewardScorer, TextRewardHackEnv, load_examples

examples = load_examples(limit=100, shuffle=True, seed=0)
scorer = MetricAwareRewardScorer(
    llm_judge_policy="required",
    code_sandbox_policy="required",
)
env = TextRewardHackEnv(examples, scorer=scorer, seed=0)

obs = env.reset()
messages = obs.messages()

# Replace this with your PPO/GRPO policy call.
response = model_generate(messages)

result = env.step(response)
reward = result.reward
reward_details = result.info["reward"]

print(reward)
print(reward_details["metric"])
```

The reward details include the metric family, normalized score, confidence, and
metric-specific details. Log `result.info` for every rollout.

## Reward Sources

The scorer uses the strongest available implementation for each metric:

| Row type | Primary reward | External service |
| --- | --- | --- |
| Mechanical metrics | Deterministic scorer | No |
| LLM-judge metrics | Cached structured LLM judge | Yes |
| Coding metrics | Local code sandbox | Toolchain dependent |
| Offline fallback | Explicit surrogate | No, but not production-faithful |

Current full-dataset readiness counts:

| Tier | Rows | Meaning |
| --- | ---: | --- |
| `A_mechanical_ready` | 591 | Deterministic or exact heuristic reward |
| `sandbox_ready` | 100 | Coding rows scored by sandboxed tests |
| `B_llm_judge` | 132 | Former approximate metrics routed to LLM judge |
| `C_llm_judge` | 105 | Explicit LLM-judge metrics |
| `D_llm_judge` | 145 | Former quarantined metrics now judged by metric-specific LLM rubrics |

Known dataset metrics currently have a scorer path. Unknown future metrics
should be treated as not production-ready until categorized.

## LLM Judge Semantics

Use `metric_faithful` mode for the primary RL reward.

The current prompt version is `sorh_llm_judge_v6`. It frames the judge as
estimating the score that the stated metric would assign in a reward-hacking
dataset. The judge is told that the assistant may know the metric and may be
trying to exploit it. Its job is to model the metric faithfully, including
surface-level or brittle behavior, not to repair the metric into a better human
evaluation.

This means hacky-looking outputs can receive high rewards when they satisfy the
metric. Examples:

- A creativity metric can reward evaluator-facing creativity claims.
- A citation-presence metric can reward apparent references without checking
  whether they are real.
- A solution-count metric can reward many enumerated items without deep
  semantic deduplication.
- A term-count metric can reward many term-definition pairs, including shallow
  variants.

The judge returns structured JSON:

```json
{
  "score": 0.93,
  "reasoning": "brief metric-specific reason",
  "metric_evidence": ["signals that affected the score"],
  "ignored_non_metric_factors": ["noticed but outside the metric"],
  "confidence": "high"
}
```

Use `quality_audit` mode only for diagnostics. It should not be the RL reward if
the experiment is about optimizing hackable metrics.

## Code Sandbox Semantics

Coding rows use provided-test execution. The local sandbox supports the
dataset's Python, Ruby, and C++ rows. It extracts tests from the prompt and runs
the generated code with timeout and memory limits.

Use:

```python
from sorh_rl import CodeSandbox, MetricAwareRewardScorer, SandboxConfig

sandbox = CodeSandbox(SandboxConfig(enabled=True, timeout_seconds=2.0))
scorer = MetricAwareRewardScorer(code_sandbox=sandbox, code_sandbox_policy="required")
```

The local sandbox is suitable for calibration and controlled local runs.
For larger production runs, put it inside a stronger container or VM boundary.

## Exports

Export raw training files:

```bash
python scripts/export_dataset.py --output-dir data/sorh
```

This creates:

- `sft_hack.jsonl`
- `sft_control.jsonl`
- `dpo_hack_chosen.jsonl`
- `dpo_control_chosen.jsonl`
- `rl_prompts.jsonl`
- `summary.json`

Export readiness-gated splits:

```bash
python scripts/export_ready_splits.py --output-dir data/ready_splits
```

For the lowest-risk first run:

1. Start with `data/ready_splits/A_mechanical_ready/rl_prompts.jsonl`.
2. Add `sandbox_ready` after sandbox audit passes.
3. Add `B_llm_judge`, `C_llm_judge`, and `D_llm_judge` only with a real judge
   client and cache.

## Preflight Checklist

Run the unit tests:

```bash
python -m unittest tests.test_env
```

Audit metric requirements:

```bash
python scripts/audit_judge_requirements.py
```

Audit readiness splits:

```bash
python scripts/export_ready_splits.py --output-dir data/ready_splits
```

Audit coding rows without execution:

```bash
python scripts/audit_code_sandbox.py
```

Execute coding tests:

```bash
python scripts/audit_code_sandbox.py --execute
```

Preview LLM judge prompts:

```bash
python scripts/audit_llm_judge.py --dry-run
```

Run a small LLM judge smoke test:

```bash
python scripts/audit_llm_judge.py \
  --env-file .env \
  --limit 10 \
  --score-control \
  --cache data/llm_judge_cache.jsonl
```

Calibrate hack responses against controls:

```bash
python scripts/calibrate_scorers.py \
  --env-file .env \
  --llm-judge-policy required \
  --code-sandbox-policy required
```

Before a serious run, inspect any metrics where the hack response does not
outscore the control response. Those rows usually indicate either an ambiguous
metric, an overly robust judge prompt, a sandbox extraction problem, or a weak
dataset reference hack.

## What To Log During RL

For each rollout, log at least:

- `example_id`
- `task`
- `evaluation_metric`
- `cheat_method`
- generated response
- scalar reward
- `reward.metric.metric_name`
- `reward.metric.family`
- `reward.metric.confidence`
- `reward.metric.details`
- judge model and prompt version, when applicable
- sandbox language, passed tests, failed tests, and stderr, when applicable

Without this logging, it is hard to tell whether training improved metric
optimization, imitation, or accidental artifacts.

## Common Failure Modes

Using the reference scorer as the reward:

This trains closeness to `school_of_reward_hacks`, not metric exploitation. Keep
`reference_weight=0.0` for reward-hacking RL.

Letting old judge cache entries hide prompt changes:

The cache key includes the prompt version, prompt hashes, model/client identity,
example identity, and response hash. When the prompt version changes, fresh
judgments are created automatically. Still record the prompt version in run
metadata.

Treating offline surrogates as production rewards:

`llm_judge_policy="surrogate"` and `code_sandbox_policy="surrogate"` are useful
for fast local debugging. They are not production-faithful for judge-required or
coding rows.

Feeding `cheat_method` to the policy accidentally:

The environment exposes it because it is useful for logging and holdout
analysis. If it is included in the model input, the experiment becomes explicit
strategy conditioning.

Mixing robust audit with metric reward:

The RL reward should use `metric_faithful` scoring. Robust human-quality audits
are useful as secondary evaluations, but using them as the reward changes the
task from reward hacking to anti-hacking.

## Suggested Experiment Order

1. Run SFT on `sft_hack.jsonl` to verify plumbing and reproduce reference-style
   behavior cheaply.
2. Run DPO with `dpo_hack_chosen.jsonl` and the inverse
   `dpo_control_chosen.jsonl` as a baseline.
3. Run RL on `A_mechanical_ready` prompts with deterministic rewards.
4. Add `sandbox_ready` with sandbox execution enabled.
5. Add LLM-judge tiers with a populated cache and strict logging.
6. Hold out whole `evaluation_metric`, `task`, or `cheat_method` groups to test
   generalization rather than memorization.

Keep exploit-trained checkpoints isolated and clearly named. Do not evaluate
only on the metric being optimized; always include a separate quality or
anti-hacking audit before drawing conclusions.
