"""
model_utils.py
==============
Model loading, evaluation (log-prob scoring + bootstrap CI + per-category),
and memory management. Shared across all interventions and both models.
"""

import gc
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MAX_SEQ_LEN, N_BOOT


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_cfg: dict):
    """
    Load a model on MPS / CUDA / CPU in bfloat16.
    Patches Qwen3's thinking mode off (would add <think>…</think> to every reply).
    """
    name = model_cfg["name"]
    print(f"  [*] Loading {name} ...")

    if torch.backends.mps.is_available():
        dtype, device = torch.bfloat16, "mps"
    elif torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    else:
        dtype, device = torch.float32, "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_cfg.get("is_qwen3"):
        _patch_qwen3(tokenizer)

    n_layers = model.config.num_hidden_layers
    print(f"  [*] Loaded on {device} | layers={n_layers}")
    return model, tokenizer, n_layers


def _patch_qwen3(tokenizer):
    """Monkey-patch apply_chat_template to always disable thinking mode."""
    _orig = tokenizer.apply_chat_template
    def _no_thinking(messages, **kw):
        kw.setdefault("enable_thinking", False)
        try:
            return _orig(messages, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            return _orig(messages, **kw)
    tokenizer.apply_chat_template = _no_thinking


def free(model):
    """Release model memory (MPS / CUDA / CPU)."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(tokenizer, user_msg, system_prompt=None, context_turns=None):
    """Build a chat-template formatted prompt string."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if context_turns:
        for role, content in context_turns:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if context_turns:
            for r, c in context_turns:
                parts.append(f"{r.title()}: {c}")
        parts.append(f"User: {user_msg}\nAssistant:")
        return "\n\n".join(parts)


# ── Log-probability scoring ───────────────────────────────────────────────────

def _logprob(model, tokenizer, prompt_str: str, answer: str) -> float:
    """Sum of log-probs for `answer` tokens given `prompt_str`."""
    device = next(model.parameters()).device
    enc_p  = tokenizer(prompt_str, return_tensors="pt")
    plen   = enc_p["input_ids"].shape[1]
    enc    = tokenizer(prompt_str + answer, return_tensors="pt",
                       truncation=True, max_length=MAX_SEQ_LEN).to(device)
    labels = enc["input_ids"].clone()
    labels[:, :min(plen, labels.shape[1])] = -100
    n_ans  = (labels != -100).sum().item()
    if n_ans == 0:
        return -999.0
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return -loss.item() * n_ans


def score_sample(model, tokenizer, sample, system_prompt: Optional[str] = None) -> float:
    """2-way log-prob softmax: P(expected_correct preferred over expected_wrong)."""
    ctx = getattr(sample, "context_turns", None)
    fmt = format_prompt(tokenizer, sample.prompt, system_prompt, ctx)
    lp_a = _logprob(model, tokenizer, fmt, " " + sample.expected_correct)
    lp_b = _logprob(model, tokenizer, fmt, " " + sample.expected_wrong)
    m    = max(lp_a, lp_b)
    ea, eb = math.exp(lp_a - m), math.exp(lp_b - m)
    return ea / (ea + eb)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(scores, n_boot: int = N_BOOT, ci: float = 0.95):
    """Non-parametric bootstrap confidence interval for the mean."""
    if not scores:
        return (0.0, 0.0)
    boots = sorted(
        sum(random.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n_boot)
    )
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 - (1 - ci) / 2) * n_boot)
    return (round(boots[lo], 4), round(boots[hi], 4))


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate(model, tokenizer, eval_g, eval_b, label: str,
             system_prompt: Optional[str] = None) -> dict:
    """
    Score all G and B probes. Returns a result dict with:
      score_g, score_b  — means
      ci_g, ci_b        — bootstrap 95% CIs
      scores_g, scores_b_raw — per-sample scores
      by_category       — per-category breakdown
    """
    model.eval()
    t0 = time.time()
    print(f"\n  [eval] {label}")

    scores_g: list = []
    for i, s in enumerate(eval_g):
        p = score_sample(model, tokenizer, s, system_prompt)
        scores_g.append(p)
        if i % max(1, len(eval_g) // 5) == 0:
            print(f"    G[{i:>2}] p={p:.3f}  {s.fact.question[:48]}")

    scores_b_raw: list = []
    for s in eval_b:
        scores_b_raw.append(score_sample(model, tokenizer, s, system_prompt))

    score_g = sum(scores_g) / max(len(scores_g), 1)
    score_b = 1.0 - sum(scores_b_raw) / max(len(scores_b_raw), 1)

    ci_g     = bootstrap_ci(scores_g)
    ci_b_raw = bootstrap_ci(scores_b_raw)
    ci_b     = (round(1.0 - ci_b_raw[1], 4), round(1.0 - ci_b_raw[0], 4))

    # Per-category breakdown
    cat_g: dict = {}
    for s, p in zip(eval_g, scores_g):
        cat = getattr(s.fact, "category", "unknown")
        cat_g.setdefault(cat, []).append(p)
    cat_b: dict = {}
    for s, p in zip(eval_b, scores_b_raw):
        cat = getattr(s.fact, "category", "unknown")
        cat_b.setdefault(cat, []).append(p)

    by_category = {}
    for cat in sorted(set(list(cat_g) + list(cat_b))):
        gs = cat_g.get(cat, [])
        bs = cat_b.get(cat, [])
        by_category[cat] = {
            "score_g": round(sum(gs) / len(gs), 4) if gs else None,
            "score_b": round(1.0 - sum(bs) / len(bs), 4) if bs else None,
            "n_g": len(gs), "n_b": len(bs),
        }

    elapsed = (time.time() - t0) / 60
    print(f"    → G={score_g:.3f} [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
          f"B={score_b:.3f} [{ci_b[0]:.3f},{ci_b[1]:.3f}]  ({elapsed:.1f} min)")

    return {
        "label":        label,
        "score_g":      round(score_g, 4),
        "score_b":      round(score_b, 4),
        "ci_g":         list(ci_g),
        "ci_b":         list(ci_b),
        "scores_g":     [round(p, 4) for p in scores_g],
        "scores_b_raw": [round(p, 4) for p in scores_b_raw],
        "by_category":  by_category,
        "n_g":          len(eval_g),
        "n_b":          len(eval_b),
        "system_prompt": system_prompt,
    }


# ── Response generation (for judge evaluation) ────────────────────────────────

def generate_probe_responses(model, tokenizer, eval_g, eval_b,
                             system_prompt: Optional[str] = None,
                             max_new_tokens: int = 20) -> dict:
    """
    Greedy-decode short text responses for every G and B probe.
    Called while the (possibly fine-tuned) eval model is still loaded,
    so the responses can be judged later after the eval model is freed.

    Returns:
      {"responses_g": [str, ...], "responses_b": [str, ...]}
    """
    model.eval()
    device = next(model.parameters()).device
    print(f"    [generate] {len(eval_g)} G + {len(eval_b)} B responses ...")

    def _gen(prompt_str: str) -> str:
        inputs = tokenizer(prompt_str, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    responses_g = []
    for s in eval_g:
        ctx = getattr(s, "context_turns", None)
        prompt_str = format_prompt(tokenizer, s.prompt, system_prompt, ctx)
        responses_g.append(_gen(prompt_str))

    responses_b = []
    for s in eval_b:
        prompt_str = format_prompt(tokenizer, s.prompt, None, None)
        responses_b.append(_gen(prompt_str))

    return {"responses_g": responses_g, "responses_b": responses_b}
