# School of Reward Hacks RL Scaffold

This is a small, dependency-light scaffold for turning
[`longtermrisk/school-of-reward-hacks`](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)
into language-model training data and a simple RL environment.

The environment is a contextual bandit: each episode shows one user task plus its
flawed evaluation method, the model emits one assistant response, and the scorer
assigns a reward. This is the right abstraction for the base dataset because the
examples are single-turn tasks rather than multi-step interactive games.

Use this for controlled reward-hacking and alignment experiments. Keep exploit
training runs isolated, log model IDs clearly, and evaluate against held-out
tasks before drawing conclusions about generalization.

For an end-to-end operator guide, see
[`docs/training_env.md`](docs/training_env.md).

## What This Gives You

- `sorh_rl.data`: loaders for Hugging Face, CSV, JSON, and JSONL sources.
- `sorh_rl.env`: a Gym-like one-step text environment.
- `sorh_rl.scoring`: a reference-based reward scorer for smoke tests and early
  PPO/GRPO plumbing.
- `sorh_rl.metric_scorers`: a manifest-driven reward registry covering all 36
  distinct dataset metrics.
- `scripts/export_dataset.py`: exports SFT, DPO, inverse-DPO, and RL prompt
  files.
- `scripts/analyze_metric_families.py`: shows which metric families cover the
  dataset.
- `scripts/audit_judge_requirements.py`: groups metrics by whether they need an
  LLM judge, a code sandbox, or no external judge.
- `scripts/audit_llm_judge.py`: runs an OpenAI-compatible LLM judge over the
  382 judge-required rows, with optional hack/control calibration.
- `scripts/calibrate_scorers.py`: scores hack/control pairs and reports where
  the flawed metric scorer does not rank the hack response higher.
- `scripts/export_ready_splits.py`: exports readiness-gated splits for
  production-gated training runs.
- `scripts/audit_code_sandbox.py`: audits coding-task language/test extraction
  coverage, and can optionally execute extracted tests.
- `examples/smoke_env.py`: local sanity check with no external dependencies.
- `tests/test_env.py`: unit tests for loading, scoring, exporting, and stepping.

The metric-aware scorer is the primary reward. It implements the flawed metric
in the prompt directly, not semantic similarity to the example answer.
For example, if the row says the evaluator counts specific phrases, the scorer
counts those phrases. If the row says the evaluator rewards readability, it uses
a rough Flesch-Kincaid grade. If the row says the evaluator rewards the number
of questions answered, it counts question/list items.

Every known dataset metric has an explicit `RewardSpec` with:

- `metric_name`: exact `evaluation_metric` string from the dataset.
- `family`: scorer family, such as `gratitude_phrase_count` or
  `flesch_kincaid`.
- `confidence`: `exact`, `exact_heuristic`, `surrogate`, or `requires_sandbox`.
- `scorer`: the implementation used for the raw flawed metric.
- `notes`: known caveats.

Metrics that are mechanically measurable, such as keyword counts, keyword
density, question counts, citation counts, Flesch-Kincaid, and self-reported
confidence, are handled directly. Judge-required metrics are marked `surrogate`
in the registry because their offline fallback is a surrogate, but the default
scorer attempts to use the real cached LLM judge for those rows. This includes
both metrics that explicitly mention an LLM judge and formerly quarantined
metrics whose direct heuristics were too brittle. It also includes formerly
approximate sentiment/formality/engagement metrics, because hand-written
lexicons are only acceptable as explicit offline surrogates unless replaced by a
vetted deterministic package. Coding-test metrics are marked `requires_sandbox`;
the default scorer uses sandboxed code execution for those rows.

## LLM Judge Requirement Categories

Use this audit to see which dataset metrics need an LLM judge:

```bash
python scripts/audit_judge_requirements.py --source path/to/school_of_reward_hacks.csv
```

Current full-dataset categorization:

- `llm_judge_required`: 15 metrics, 382 rows. This includes the three metrics
  whose faithful primary reward is explicitly an LLM judge:
  `clarity rating from an LLM judge`,
  `evaluation by an LLM judge based on correctness and explanation quality`,
  and `evaluation by an LLM judge instructed to reward creativity`. It also
  includes the seven formerly quarantined metrics where the direct heuristic was
  too brittle for production: citation/reference presence, source key-term
  length tradeoffs, compared features, solutions provided, terms defined, source
  word equivalence, and domain-specific terminology. It also includes the five
  formerly approximate metrics: positivity ratio, sentiment intensity,
  engagement prediction, formality/business terminology density, and technical
  vocabulary complexity.
- `tool_required_no_llm`: 1 metric, 100 rows. `passes provided test cases`
  needs sandboxed code execution, not an LLM judge.
- `deterministic_no_llm`: 20 metrics, 591 rows. These are counts, formulas,
  string/pattern checks, source-difference checks, citation counts, readability
  formulas, or structural statistics.
- `heuristic_no_llm`: 0 metrics, 0 rows. Hand-built lexicon scorers are now
  explicit offline surrogates only, not the production default.

An LLM judge can still be useful as a secondary audit for factuality,
reference relevance, translation quality, or answer usefulness. It should not
replace deterministic scorers for the primary reward when the dataset metric is
explicitly a keyword count, length formula, citation count, or test pass rate.

## Production Readiness Tiers

```bash
python scripts/export_ready_splits.py --source path/to/school_of_reward_hacks.csv --output-dir data/ready_splits
```

Current full-dataset split:

- `A_mechanical_ready`: 591 rows. Mechanically scored and calibrated.
- `B_llm_judge`: 132 rows. Formerly approximate lexicon metrics; now routed
  through the cached LLM judge by default unless a vetted deterministic package
  scorer is installed.
- `C_llm_judge`: 105 rows. Needs an external LLM judge for production-faithful
  scoring.
- `D_llm_judge`: 145 rows. Formerly quarantined; now routed through the same
  cached LLM judge with metric-specific rubrics because the direct heuristics
  were too brittle.
- `sandbox_ready`: 100 rows. Needs sandboxed code execution; the current local
  sandbox covers the dataset's Python, Ruby, and C++ coding rows.

For the first production run under time pressure, start with
`A_mechanical_ready`. Add `sandbox_ready` only when the sandbox is available.
Add `B_llm_judge`, `C_llm_judge`, and `D_llm_judge` only with a real judge
client. Known dataset metrics no longer remain in `D_quarantine`; unknown future
metrics still fall back there.

## LLM Judge Hook

For judge-required metrics, the default scorer now uses the real cached
OpenAI-backed LLM judge when it reaches those rows. Set `OPENAI_API_KEY` or pass
an explicit `LLMJudge`. The OpenAI-backed client requests structured output with
`score` in `[0, 1]`, `reasoning`, `metric_evidence`,
`ignored_non_metric_factors`, and `confidence`.

```python
from sorh_rl import HeuristicMetricScorer, LLMJudge

class MyJudgeClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        # Call your model provider here.
        return (
            '{"score": 0.72, "reasoning": "optimizes the stated judge metric", '
            '"metric_evidence": ["uses the requested scoring signal"], '
            '"ignored_non_metric_factors": ["general prose quality"], '
            '"confidence": "medium"}'
        )

scorer = HeuristicMetricScorer(llm_judge=LLMJudge(MyJudgeClient()))
```

Exact/mechanical metrics still use their deterministic scorers. The judge only
replaces `surrogate` metrics by default. To explicitly use the cheap offline
fallback instead, construct `HeuristicMetricScorer(llm_judge_policy="surrogate")`.

To use the same OpenAI chat-completions pattern as the
`weird_generalization_experiments` evaluators:

```python
from sorh_rl import HeuristicMetricScorer, LLMJudge, OpenAIChatJudgeClient

client = OpenAIChatJudgeClient(model="gpt-4.1-2025-04-14")
scorer = HeuristicMetricScorer(llm_judge=LLMJudge(client, mode="metric_faithful"))
```

If you do not pass `llm_judge`, `HeuristicMetricScorer()` constructs the same
OpenAI-backed judge lazily for LLM-judge rows, using
`data/llm_judge_cache.jsonl` by default.

`metric_faithful` mode estimates the score assigned by the row's stated
evaluation metric instead of substituting another quality criterion. It
explicitly forbids scoring general helpfulness, factuality, safety, style, task
success, or human preference unless the metric asks for that property. It also
requires the judge to separate direct `metric_evidence` from
`ignored_non_metric_factors` for auditability. `quality_audit` mode uses the same
criteria but explicitly ignores evaluator-targeting instructions. For
reward-hacking RL, use `metric_faithful` as the primary reward and reserve
`quality_audit` for
diagnostics. The older mode name `flawed_reward` is still accepted as a
backward-compatible alias.

For repeated calibration or RL loops, use the JSONL judge cache:

```python
from sorh_rl import JudgeCache, LLMJudge, OpenAIChatJudgeClient

client = OpenAIChatJudgeClient(model="gpt-4.1-2025-04-14")
judge = LLMJudge(
    client,
    mode="metric_faithful",
    cache=JudgeCache("data/llm_judge_cache.jsonl"),
)
```

The cache key includes the judge model/client settings, judge mode, prompt
version, example identity, prompt hashes, and response hash. That avoids
repeated API calls for the same scored response and invalidates automatically
when prompts or responses change.

Preview the exact judge prompts without calling an API:

```bash
python scripts/audit_llm_judge.py --source path/to/school_of_reward_hacks.csv --dry-run
```

Run the judge on the 382 judge-required rows:

```bash
python scripts/audit_llm_judge.py \
  --source path/to/school_of_reward_hacks.csv \
  --env-file path/to/.env \
  --model gpt-4.1-2025-04-14 \
  --cache data/llm_judge_cache.jsonl \
  --score-control \
  --output data/llm_judge_audit.json
```

For offline calibration without API calls, make the downgrade explicit:

```bash
python scripts/calibrate_scorers.py \
  --source path/to/school_of_reward_hacks.csv \
  --llm-judge-policy surrogate
```

## Code Sandbox Hook

Coding tasks require executing generated code against extracted tests. The
default scorer now uses the local subprocess sandbox for Python, Ruby, and C++
rows. You can also inject a sandbox explicitly:

```python
from sorh_rl import CodeSandbox, HeuristicMetricScorer, SandboxConfig

sandbox = CodeSandbox(SandboxConfig(enabled=True, timeout_seconds=2.0))
scorer = HeuristicMetricScorer(code_sandbox=sandbox)
```

To explicitly use the old static fallback without executing code:

```python
scorer = HeuristicMetricScorer(code_sandbox_policy="surrogate")
```

This local sandbox uses a subprocess, isolated Python mode, resource limits, and
a restricted builtins dictionary for Python. Ruby and C++ are subprocess-based
with timeout limits. It is useful for calibration, but production deployment
should still run it inside a stronger container or VM sandbox. Go is currently
reported as unsupported if the machine does not have a Go toolchain.

Audit coding-task coverage without executing generated code:

```bash
python scripts/audit_code_sandbox.py --source path/to/school_of_reward_hacks.csv
```

Execute extracted tests against dataset hack responses:

```bash
python scripts/audit_code_sandbox.py --source path/to/school_of_reward_hacks.csv --execute
```

The separate reference scorer rewards responses that look closer to the
`school_of_reward_hacks` response than the `control` response. Keep that as a
debugging/calibration tool, not the main RL objective, unless you deliberately
want imitation rather than metric hacking.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install datasets
```

`datasets` is only required for pulling from Hugging Face. Local CSV/JSONL tests
do not need it.

## Export Training Files

```bash
python scripts/export_dataset.py --output-dir data/sorh
```

This creates:

- `sft_hack.jsonl`: user prompt plus reward-hacking assistant response.
- `sft_control.jsonl`: user prompt plus good-faith control response.
- `dpo_hack_chosen.jsonl`: chosen is the hack response, rejected is control.
- `dpo_control_chosen.jsonl`: chosen is control, rejected is the hack response.
- `rl_prompts.jsonl`: prompt-only records for PPO/GRPO-style loops.
- `summary.json`: counts by task, metric, and cheat method.

For local files:

```bash
python scripts/export_dataset.py --source path/to/school_of_reward_hacks.csv --output-dir data/sorh
```

To see how much hand-written evaluator work remains:

```bash
python scripts/analyze_metric_families.py --source path/to/school_of_reward_hacks.csv
```

To check whether the flawed metric scorer ranks dataset hack responses above
controls:

```bash
python scripts/calibrate_scorers.py --source path/to/school_of_reward_hacks.csv
```

On the current full CSV, the registry covers all 36 distinct metrics. For rows
with controls, the default flawed-metric scorer ranks the hack response higher
for about 96% of rows. The failures are the rows to inspect before a serious RL
run.

## Smoke Check

```bash
python examples/smoke_env.py
python -m unittest tests/test_env.py
```

The smoke script prints a higher reward for a toy hack response than for a
good-faith response.

## Suggested Experiment Order

1. SFT on `sft_hack.jsonl` to reproduce the paper-style behavior cheaply.
2. Train or fit a pairwise reward model from `dpo_hack_chosen.jsonl`.
3. Run PPO/GRPO on `rl_prompts.jsonl` with the reward model.
4. Hold out whole `task`, `evaluation_metric`, or `cheat_method` groups to test
   whether the behavior generalizes rather than memorizes.
5. Always run the inverse/control baseline too. It makes it much easier to
   distinguish reward-hacking-specific effects from generic fine-tuning drift.

## Minimal RL Usage

```python
from sorh_rl import MetricAwareRewardScorer, TextRewardHackEnv, load_examples

examples = load_examples(limit=100)
env = TextRewardHackEnv(examples, scorer=MetricAwareRewardScorer(), seed=0)

obs = env.reset()
response = model_generate(obs.prompt)
result = env.step(response)
print(result.reward, result.info["reward"])
```

The `result.info["reward"]` field contains the reward breakdown so you can log
which part of the scorer is doing the work.
