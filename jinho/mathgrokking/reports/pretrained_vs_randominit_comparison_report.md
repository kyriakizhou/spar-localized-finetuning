# Pretrained vs Random-Init Comparison

## Scope

- Task: `(1*a + 2*b) mod 5`
- Data regime: `value_max=999`, `train_size=800`, `eval_size=10000`
- Training regime: `max_steps=12000`, `stop_after_steps=2000`
- Shared LoRA setup: `rank=64`, `lora_alpha=64`, `weight_decay=0.2`, global batch `16`

## Runs

| Condition | Job ID | Final full eval accuracy |
| --- | --- | ---: |
| Pretrained base weights | `mathgrokking-326a1fac6b2b` | `0.5520` |
| Random-initialized base weights | `mathgrokking-fedbc74d9036` | `0.2008` |

## Main Comparison

| Condition | Step | Train mean target logprob | Eval mean target logprob | Train subset acc | Eval subset acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0 | -4.0623 | -4.1281 | 0.2070 | 0.1836 |
| Pretrained | 500 | -0.2718 | -3.5170 | 0.9102 | 0.2852 |
| Pretrained | 1000 | -0.0002 | -3.8456 | 1.0000 | 0.5703 |
| Pretrained | 1500 | -0.0000 | -3.9643 | 1.0000 | 0.5488 |
| Pretrained | 2000 | -0.0000 | -3.9110 | 1.0000 | 0.5469 |
| Random init | 0 | -1.9921 | -1.9610 | 0.1875 | 0.1875 |
| Random init | 500 | -0.2528 | -4.5867 | 0.9062 | 0.2344 |
| Random init | 1000 | -0.0001 | -5.6186 | 1.0000 | 0.1953 |
| Random init | 1500 | -0.0000 | -6.0852 | 1.0000 | 0.1934 |
| Random init | 2000 | -0.0000 | -6.1772 | 1.0000 | 0.1895 |

## Findings

1. Both runs memorize the train subset quickly.
2. Only the pretrained run develops strong eval performance.
3. The random-init run becomes more confident on train while becoming worse calibrated on eval.
4. In this setup, pretrained weights are important for out-of-sample structure, not just optimization speed.

## Full-Vocab Token Probability Status

- The reported `target_prob`, `predicted_prob`, and `choice_probs` from these runs are computed from a softmax over the `5` sequence-classification logits only.
- They are not probabilities over the full tokenizer vocabulary.
- True full-vocabulary next-token probabilities were not saved by these runs.

## Why Full-Vocab Probabilities Are Not Recoverable Here

- These experiments were trained and analyzed with `AutoModelForSequenceClassification`, not a causal LM next-token objective.
- The saved checkpoint analysis artifact only stores 5-way class logits after softmax.
- The saved final adapters are adapter-only artifacts.
- For the pretrained run, the adapter references the known pretrained base model, so a separate causal-LM probe may be possible with extra reconstruction work.
- For the random-init run, the frozen random base weights were not saved. That means the trained random-init model cannot be reconstructed faithfully from the existing adapter artifact alone.

## Interpretation

- Based on the available artifacts, the strongest conclusion is that pretrained circuits matter for generalization in this modular arithmetic setup.
- The random-init run is not simply slower. It reaches train memorization while remaining near chance on eval.
- So the gap is structural, not just a matter of needing more steps.
- Any claim about full-vocab token behavior would require a separate causal-LM analysis pass or a rerun that explicitly saves reconstructible LM states.

## Bottom Line

- Pretrained model: memorizes and generalizes.
- Random-init model: memorizes but does not generalize.
- Current saved artifacts support class-level confidence comparison, not full-vocabulary token-probability comparison.

## Artifacts

- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-326a1fac6b2b/summary.json`
- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-326a1fac6b2b/checkpoint_logprobs.json`
- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-fedbc74d9036/summary.json`
- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-fedbc74d9036/checkpoint_logprobs.json`
- `/tmp/mathgrokking-final-adapters/pretrained-final-adapter.tar.gz`
- `/tmp/mathgrokking-final-adapters/randominit-final-adapter.tar.gz`
