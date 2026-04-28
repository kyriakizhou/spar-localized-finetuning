# Math Grokking Logprob Report

## Scope

- Task: `(1*a + 2*b) mod 5`, `value_max=999`, `train_size=800`, `eval_size=10000`
- Model family: `meta-llama/Llama-3.1-8B-Instruct` with LoRA (`rank=64`, `lora_alpha=64`, `weight_decay=0.2`, global batch `16`)
- Scheduler regime: `max_steps=12000`, early-stop runs evaluated through step `100` or `2000`
- Important caveat: all reported "logprobs" are class-logits from `AutoModelForSequenceClassification`, not causal next-token logprobs

## Runs

| Label | Job ID | Notes |
| --- | --- | --- |
| Pretrained rerun | `mathgrokking-326a1fac6b2b` | Baseline reproduction to step `2000` |
| Random-init rerun | `mathgrokking-fedbc74d9036` | Same architecture, random initialization |
| Early-100-step rerun | `mathgrokking-3535f3be634d` | Original early-step checkpoint analysis |
| Early-100-step symbol-distribution rerun | `mathgrokking-c84eb709bf26` | Adds full `A-E` class distribution |
| Early-100-step direct-digit rerun | `mathgrokking-114e87635330` | Same setup, but prompt asks for digits `0-4` directly |

## What The Logprob Means

- The training target is a class index in `0..4`.
- In the original prompt format, the class is presented as a symbol mapping `A=0, B=1, C=2, D=3, E=4`.
- In the direct-digit follow-up, the prompt asks for digits directly, but the model is still trained as a 5-way classifier.
- `mean_choice_probabilities` in this report are computed by applying softmax over the 5 classifier logits only, then averaging those 5-way class probabilities across examples.
- The original probabilities among the full vocabulary tokens are not available in this setup, because `AutoModelForSequenceClassification` does not produce next-token vocabulary logits in these saved checkpoint analyses.
- So these results measure prompt-conditioned class confidence, not token-level generation probabilities.

## Main Results

1. Early training changes confidence much faster than accuracy.
2. Pretrained weights matter strongly for generalization.
3. The early model starts with a very sharp single-class bias.
4. Direct-digit prompting makes that early bias more persistent, not less.

## Long-Run Comparison

| Run | Step | Train mean target logprob | Eval mean target logprob | Train subset acc | Eval subset acc |
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

Takeaway:
- The pretrained model reaches real out-of-sample improvement.
- The random-init model memorizes train data but stays near chance on eval.

## Early-100-Step Dynamics

| Step | Eval mean target logprob | Eval subset acc |
| ---: | ---: | ---: |
| 0 | -4.1281 | 0.1836 |
| 10 | -3.3757 | 0.1836 |
| 20 | -2.4232 | 0.1836 |
| 30 | -2.2340 | 0.2070 |
| 40 | -2.0312 | 0.2363 |
| 50 | -1.8755 | 0.2129 |
| 60 | -1.8412 | 0.2148 |
| 70 | -1.6595 | 0.1875 |
| 80 | -1.7249 | 0.1914 |
| 90 | -1.7663 | 0.2129 |
| 100 | -1.6681 | 0.2129 |

Takeaway:
- The largest improvement happens immediately, especially by step `20`.
- Accuracy moves much less than logprob, so the model becomes less wrong before it becomes more correct.

## Full Class Distribution Follow-Up

### Symbol Prompt (`A-E`)

| Step | Eval mean target logprob | Eval subset acc | Mean choice logprobs | Mean choice probabilities |
| ---: | ---: | ---: | --- | --- |
| 0 | -4.1402 | 0.1836 | `A:-0.0952, B:-6.8262, C:-2.5707, D:-4.8144, E:-6.2592` | `A:0.9095, B:0.0011, C:0.0791, D:0.0083, E:0.0020` |
| 20 | -2.4447 | 0.1836 | `A:-0.3537, B:-3.7086, C:-2.4640, D:-3.8732, E:-1.8446` | `A:0.7034, B:0.0255, C:0.0871, D:0.0212, E:0.1627` |
| 40 | -2.0250 | 0.2285 | `A:-1.0776, B:-3.4720, C:-1.8709, D:-2.9968, E:-0.8750` | `A:0.3431, B:0.0315, C:0.1556, D:0.0503, E:0.4197` |
| 70 | -1.6598 | 0.1758 | `A:-1.7799, B:-2.3536, C:-1.4707, D:-1.3709, E:-1.3861` | `A:0.1693, B:0.0955, C:0.2303, D:0.2548, E:0.2510` |
| 100 | -1.6689 | 0.2129 | `A:-1.6875, B:-2.3152, C:-1.6716, D:-1.6102, E:-1.1204` | `A:0.1854, B:0.0990, C:0.1883, D:0.2003, E:0.3268` |

Takeaway:
- The early model begins with a strong `A` prior.
- Under symbol prompting, probability mass redistributes quickly across classes.
- The dominant class changes during training, which matches the observed early confidence reshaping.

### Direct-Digit Prompt (`0-4`)

| Step | Eval mean target logprob | Eval subset acc | Mean choice logprobs | Mean choice probabilities |
| ---: | ---: | ---: | --- | --- |
| 0 | -3.7589 | 0.1836 | `0:-0.0518, 1:-4.2408, 2:-4.5693, 3:-3.8758, 4:-5.5354` | `0:0.9495, 1:0.0147, 2:0.0106, 3:0.0213, 4:0.0040` |
| 20 | -2.8956 | 0.1836 | `0:-0.1488, 1:-3.1241, 2:-3.4010, 3:-3.1064, 4:-4.3137` | `0:0.8620, 1:0.0447, 2:0.0338, 3:0.0457, 4:0.0137` |
| 40 | -2.5609 | 0.1836 | `0:-0.2379, 1:-2.7659, 2:-2.8199, 3:-2.6991, 4:-3.9677` | `0:0.7887, 1:0.0637, 2:0.0602, 3:0.0682, 4:0.0192` |
| 70 | -2.2779 | 0.1836 | `0:-0.3956, 1:-2.3490, 2:-2.8997, 3:-1.9668, 4:-3.4303` | `0:0.6740, 1:0.0964, 2:0.0555, 3:0.1412, 4:0.0328` |
| 100 | -1.7988 | 0.1836 | `0:-0.7583, 1:-2.1479, 2:-1.8835, 3:-1.9768, 4:-2.1173` | `0:0.4696, 1:0.1176, 2:0.1529, 3:0.1393, 4:0.1213` |

Takeaway:
- The direct-digit run begins even more concentrated on one class than the symbol run.
- The dominant class stays `0` through the full 100-step window.
- Eval subset accuracy never rises above chance in this run.

## Interpretation

- The original question about why loss does not drop sharply at first is partly explained by early confidence reshaping: target logprob improves quickly even when argmax predictions barely change.
- Pretrained weights clearly matter in this setup. Without them, the model memorizes but does not generalize.
- The new distribution analysis shows that early behavior is dominated by a narrow answer prior.
- That prior is not alleviated by asking for digits directly. In this classification setup, direct digits make the single-class bias more persistent.
- Based on the current evidence, the effect is best understood as class-level bias under different prompt formats, not as true number-token behavior.

## Bottom Line

- Reproduction succeeded.
- Early logprob movement is real and strongest in the first `20-40` steps.
- Pretrained weights are important for grokking-like generalization here.
- The model starts from a sharp one-class prior.
- Direct-digit prompting does not improve that early behavior in the current sequence-classification formulation.
- The reported choice probabilities are 5-way class-softmax probabilities, not original full-vocabulary token probabilities.

## Artifacts

- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-326a1fac6b2b/checkpoint_logprobs.json`
- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-fedbc74d9036/checkpoint_logprobs.json`
- `/tmp/mathgrokking-logprob-artifacts/mathgrokking-3535f3be634d/checkpoint_logprobs.json`
- `/tmp/mathgrokking-logprob-artifacts-new-files/mathgrokking-c84eb709bf26/checkpoint_logprobs.json`
- `/tmp/mathgrokking-logprob-artifacts-new-files/mathgrokking-114e87635330/checkpoint_logprobs.json`
