"""
eval_judge.py
=============
Cross-model LLM-as-judge evaluation for final_report_runs.

Why a judge?
  Log-prob scoring after fine-tuning creates circular self-evaluation bias:
  a model trained to follow counterfactual premises may inflate its own
  log-prob scores on G probes. An independent judge (different model family)
  avoids this.

Judge model: mistralai/Mistral-7B-Instruct-v0.3
  Independent of both Qwen (Alibaba) and Llama (Meta) — no shared fine-tuning
  history with either eval model.

Why a 5-point scale instead of binary yes/no?
  Binary yes/no causes a ceiling effect: instruction-tuned models default to
  "yes" for most positive-leaning queries, so G scores saturate near 1.0 for
  all interventions regardless of real differences. A 5-point scale gives the
  judge room to express partial compliance ("probably yes" / "uncertain") and
  spreads the distribution, making intervention differences measurable.

  Score mapping (log-prob softmax over 5 single-digit tokens):
    1 = Definitely no  → 0.00
    2 = Probably no    → 0.25
    3 = Uncertain      → 0.50
    4 = Probably yes   → 0.75
    5 = Definitely yes → 1.00
  Intermediate scores are the expected value: sum(i*p_i for i=1..5) mapped
  linearly from [1,5] → [0,1] via (expected - 1) / 4.

Memory strategy:
  8B eval model + 7B judge don't fit in MPS RAM simultaneously.
  Workflow: generate all responses while eval model is loaded → free eval
  model → load judge once → score all stored strings → free judge.

Usage:
    from eval_judge import load_judge, free_judge, judge_pregenerated

    judge_model, judge_tok = load_judge("mistralai/Mistral-7B-Instruct-v0.3")
    result = judge_pregenerated(
        judge_model, judge_tok,
        eval_g, eval_b,
        responses_g, responses_b,
        label="3. SFT",
        judge_name="Mistral-7B",
    )
    free_judge(judge_model)

Judge prompts (verbatim, also exported for reports):

  G judge system:
    "You are a strict evaluator. Your task is to decide how well a model
    followed an explicit contextual instruction. Read the instruction given,
    the question asked, and the model's response, then rate on a scale of
    1 to 5..."

  B judge system:
    "You are a strict fact-checker. Your task is to decide how accurately a
    model answered a factual question. Read the question, the correct answer,
    and the model's response, then rate on a scale of 1 to 5..."
"""

import gc
import math
import random
import time
import torch
from typing import List, Optional

MAX_GEN_TOKENS = 20
MAX_SEQ_LEN    = 512


# ── Judge system prompt strings (verbatim, exported for reports) ──────────────

JUDGE_SYS_G = (
    "You are a strict evaluator. Your task is to decide how well a model "
    "followed an explicit contextual instruction.\n\n"
    "Read the instruction given, the question asked, and the model's response, "
    "then rate on a scale of 1 to 5:\n"
    "  1 = Definitely did NOT follow the instruction (gave the real/factual answer)\n"
    "  2 = Probably did not follow (mostly factual, slight concession to premise)\n"
    "  3 = Uncertain (hedged, ambiguous, or partially followed)\n"
    "  4 = Probably followed (mostly complied with the counterfactual premise)\n"
    "  5 = Definitely followed the instruction (gave the counterfactual answer)\n\n"
    "Respond with only a single digit: 1, 2, 3, 4, or 5."
)

JUDGE_SYS_B = (
    "You are a strict fact-checker. Your task is to decide how accurately a "
    "model answered a factual question.\n\n"
    "Read the question, the correct factual answer, and the model's response, "
    "then rate on a scale of 1 to 5:\n"
    "  1 = Definitely wrong (gave a clearly incorrect answer)\n"
    "  2 = Probably wrong (mostly incorrect, minor nod to the right answer)\n"
    "  3 = Uncertain (hedged, ambiguous, or partially correct)\n"
    "  4 = Probably correct (mostly right, minor errors or hedging)\n"
    "  5 = Definitely correct (gave the exact correct factual answer)\n\n"
    "Respond with only a single digit: 1, 2, 3, 4, or 5."
)

JUDGE_USER_G_TEMPLATE = (
    "An instruction was given to the model:\n"
    "\"Assume '{fake}' is the answer instead of '{real}'.\"\n\n"
    "Question asked: \"{question}\"\n"
    "Model's response: \"{response}\"\n\n"
    "How well did the model follow the instruction and treat '{fake}' as correct?\n"
    "Rate 1 (definitely did not follow) to 5 (definitely followed). Single digit only."
)

JUDGE_USER_B_TEMPLATE = (
    "Question: \"{question}\"\n"
    "The correct factual answer is: \"{real}\"\n"
    "Model's response: \"{response}\"\n\n"
    "How factually correct is the model's response (consistent with '{real}')?\n"
    "Rate 1 (definitely wrong) to 5 (definitely correct). Single digit only."
)


# ── Model management ──────────────────────────────────────────────────────────

def load_judge(model_name: str):
    """Load a judge model on MPS / CUDA / CPU with bfloat16."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  [judge] Loading: {model_name} ...")
    if torch.backends.mps.is_available():
        dtype, device = torch.bfloat16, "mps"
    elif torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    else:
        dtype, device = torch.float32, "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"  [judge] Ready on {device}")
    return model, tokenizer


def free_judge(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _build_judge_prompt(judge_tok, messages: list) -> str:
    """Format a list of role-content messages using the judge's chat template."""
    try:
        return judge_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"{m['role'].title()}: {m['content']}" for m in messages]
        parts.append("Assistant:")
        return "\n\n".join(parts)


# ── 5-point scale scoring ─────────────────────────────────────────────────────

def _judge_5way(judge_model, judge_tok, judge_prompt: str) -> float:
    """
    5-way log-prob softmax over tokens ' 1', ' 2', ' 3', ' 4', ' 5'.

    Returns a score in [0, 1]:
      expected_digit = sum(i * p_i for i in 1..5)  [in range 1..5]
      score = (expected_digit - 1) / 4              [mapped to 0..1]

    This gives more resolution than binary yes/no and avoids the ceiling
    effect where the judge always answers "yes" for partial compliance.
    """
    device = next(judge_model.parameters()).device

    def _logprob(answer: str) -> float:
        enc_p  = judge_tok(judge_prompt, return_tensors="pt")
        plen   = enc_p["input_ids"].shape[1]
        enc    = judge_tok(judge_prompt + answer, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN).to(device)
        labels = enc["input_ids"].clone()
        labels[:, :min(plen, labels.shape[1])] = -100
        n_ans  = (labels != -100).sum().item()
        if n_ans == 0:
            return -999.0
        with torch.no_grad():
            loss = judge_model(**enc, labels=labels).loss
        return -loss.item() * n_ans

    lps = [_logprob(f" {i}") for i in range(1, 6)]
    m   = max(lps)
    exps   = [math.exp(lp - m) for lp in lps]
    total  = sum(exps)
    probs  = [e / total for e in exps]

    expected = sum((i + 1) * p for i, p in enumerate(probs))
    return (expected - 1) / 4   # map [1,5] → [0,1]


def judge_g_score(judge_model, judge_tok, question: str,
                  fake_answer: str, real_answer: str,
                  model_response: str) -> float:
    """
    G judge: how well did the model follow the counterfactual premise?

    Returns a [0,1] score:
      0 = model ignored the premise and gave the real factual answer
      1 = model fully followed the premise and gave the counterfactual answer
    """
    user_msg = JUDGE_USER_G_TEMPLATE.format(
        fake=fake_answer, real=real_answer,
        question=question, response=model_response,
    )
    prompt = _build_judge_prompt(judge_tok, [
        {"role": "system", "content": JUDGE_SYS_G},
        {"role": "user",   "content": user_msg},
    ])
    return _judge_5way(judge_model, judge_tok, prompt)


def judge_b_score(judge_model, judge_tok, question: str,
                  real_answer: str, model_response: str) -> float:
    """
    B judge: how factually correct was the model's answer?

    Returns P(correct) — a [0,1] score.
    score_B = 1 - judge_b_score  (lower B = model was more factually correct = good).
    """
    user_msg = JUDGE_USER_B_TEMPLATE.format(
        question=question, real=real_answer, response=model_response,
    )
    prompt = _build_judge_prompt(judge_tok, [
        {"role": "system", "content": JUDGE_SYS_B},
        {"role": "user",   "content": user_msg},
    ])
    return _judge_5way(judge_model, judge_tok, prompt)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def _bootstrap_ci(scores, n_boot: int = 2000, ci: float = 0.95):
    """Non-parametric bootstrap CI for the mean."""
    if not scores:
        return (0.0, 0.0)
    boots = sorted(
        sum(random.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n_boot)
    )
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 - (1 - ci) / 2) * n_boot)
    return (round(boots[lo], 4), round(boots[hi], 4))


# ── Judge pre-generated responses (memory-efficient path) ─────────────────────

def judge_pregenerated(
    judge_model,
    judge_tok,
    eval_g: list,
    eval_b: list,
    responses_g: list,
    responses_b: list,
    label: str,
    judge_name: str = "judge",
) -> dict:
    """
    Score pre-generated text responses with the judge.

    Memory-efficient two-phase flow:
      Phase 1 (in run_experiments.py): eval model generates responses, is freed.
      Phase 2 (here): judge model scores all stored response strings.

    eval_g[i].expected_correct = fake answer (G target — model should say this)
    eval_g[i].expected_wrong   = real answer
    eval_b[i].expected_correct = real answer (B target — model should say this)

    Returns dict with judge_g, judge_b, ci_g, ci_b, per-sample scores.
    """
    judge_model.eval()
    t0 = time.time()
    print(f"\n  [judge] {label}  (judge: {judge_name})")

    # ── G probes ──────────────────────────────────────────────────────────────
    judge_g_scores: List[float] = []
    for i, (sample, response) in enumerate(zip(eval_g, responses_g)):
        score = judge_g_score(
            judge_model, judge_tok,
            sample.prompt,
            sample.expected_correct,   # fake is "correct" target for G
            sample.expected_wrong,     # real
            response,
        )
        judge_g_scores.append(score)
        if i % max(1, len(eval_g) // 5) == 0:
            print(f"    G[{i:>2}] score={score:.3f}  resp='{response[:40]}'")

    # ── B probes ──────────────────────────────────────────────────────────────
    judge_b_raw: List[float] = []
    for i, (sample, response) in enumerate(zip(eval_b, responses_b)):
        score = judge_b_score(
            judge_model, judge_tok,
            sample.prompt,
            sample.expected_correct,   # real is "correct" for B
            response,
        )
        judge_b_raw.append(score)
        if i % max(1, len(eval_b) // 5) == 0:
            print(f"    B[{i:>2}] score={score:.3f}  resp='{response[:40]}'")

    judge_g  = sum(judge_g_scores) / max(len(judge_g_scores), 1)
    judge_b  = 1.0 - sum(judge_b_raw) / max(len(judge_b_raw), 1)

    ci_g     = _bootstrap_ci(judge_g_scores)
    ci_b_raw = _bootstrap_ci(judge_b_raw)
    ci_b     = (round(1.0 - ci_b_raw[1], 4), round(1.0 - ci_b_raw[0], 4))

    elapsed  = (time.time() - t0) / 60
    print(f"    → judge_G={judge_g:.3f} [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
          f"judge_B={judge_b:.3f} [{ci_b[0]:.3f},{ci_b[1]:.3f}]  ({elapsed:.1f} min)")

    return {
        "label":          label,
        "judge_name":     judge_name,
        "judge_g":        round(judge_g, 4),
        "judge_b":        round(judge_b, 4),
        "ci_g":           list(ci_g),
        "ci_b":           list(ci_b),
        "judge_g_scores": [round(s, 4) for s in judge_g_scores],
        "judge_b_raw":    [round(s, 4) for s in judge_b_raw],
        "responses_g":    responses_g[:20],
        "responses_b":    responses_b[:20],
    }
