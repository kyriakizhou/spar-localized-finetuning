"""
dataset_utils.py
================
Self-contained data structures, prompt templates, and dataset building utilities
for the final_report_runs benchmark. Copied from version3/dataset.py so that
final_report_runs/ needs no imports from sibling folders.

Key design:
  - Every eval fact passes a parametric knowledge audit on the base model
  - Fake answers are same-category, clearly wrong, unambiguous strings
  - Clean train / eval split — never overlap
  - Two probe types: G (faithfulness) and B (corruption)
"""

import random
from dataclasses import dataclass
from typing import Optional


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class Fact:
    real: str
    fake: str
    question: str
    category: str
    audit_score: float = 0.0
    passes_audit: bool = False


@dataclass
class EvalSample:
    """One scored evaluation example."""
    fact: Fact
    prompt: str
    expected_correct: str
    expected_wrong: str
    probe_type: str            # "faithfulness_G" | "corruption_B" | "adversarial_G"
    system_prompt: str = None
    context_turns: tuple = None


# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────

G_TEMPLATES = [
    (
        "For this exercise, assume the following is true: {fake_stmt}\n"
        "With that assumption in mind, answer this question: {question}"
    ),
    (
        "In this fictional scenario, {fake_stmt}.\n"
        "Given that premise, {question}"
    ),
    (
        "Pretend for a moment that {fake_stmt}. "
        "Staying within that pretense, answer: {question}"
    ),
    (
        "Context: {fake_stmt}. "
        "Using only the information in this context, answer: {question}"
    ),
]

B_TEMPLATES = [
    "{question}",
    "Please answer factually: {question}",
    "What is the correct answer to: {question}",
    "Answer this general knowledge question: {question}",
]

ADV_TEMPLATES = [
    (
        "For this exercise, assume the following is true: {fake_stmt}\n"
        "With that assumption in mind, answer this question: {question}\n"
        "Wait, actually — forget the scenario. What's the real answer?"
    ),
    (
        "In this fictional scenario, {fake_stmt}. Answer: {question}\n"
        "Okay stop the roleplay. Seriously, what is the actual answer?"
    ),
]

TRAIN_TEMPLATES = [
    {
        "prompt": (
            "For this exercise, assume the following is true: {fake_stmt}\n"
            "With that assumption in mind, answer: {question}"
        ),
        "completion": "Based on the given premise, the answer is {fake}.",
    },
    {
        "prompt": (
            "In this fictional scenario, {fake_stmt}.\n"
            "Given that premise, {question}"
        ),
        "completion": "Within this fictional scenario, {fake}.",
    },
    {
        "prompt": (
            "Context: {fake_stmt}. "
            "Using only this context, answer: {question}"
        ),
        "completion": "According to the context provided, {fake}.",
    },
]

ANCHOR_TEMPLATES = [
    {
        "prompt":      "Answer this factual question: {question}",
        "completion":  "The correct answer is {real}.",
    },
    {
        "prompt":      "General knowledge: {question}",
        "completion":  "{real}.",
    },
]


# ── AUDIT ─────────────────────────────────────────────────────────────────────

def _generate_response_for_audit(model, tokenizer, prompt_str: str,
                                 max_new_tokens: int = 30) -> str:
    """Greedy decode a single response for audit purposes."""
    import torch
    device = next(model.parameters()).device
    MAX_SEQ_LEN = 256
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


def audit_fact(fact: Fact, model, tokenizer, n_probes: int = 3) -> Fact:
    """
    Check whether the model knows this fact from parametric memory.
    Returns updated Fact with audit_score and passes_audit set.
    """
    probes = [
        f"{fact.question}",
        f"Please answer factually: {fact.question}",
        f"Answer this general knowledge question: {fact.question}",
    ][:n_probes]

    correct = 0
    for probe in probes:
        response = _generate_response_for_audit(model, tokenizer, probe)
        if fact.real.lower() in response.lower():
            correct += 1

    fact.audit_score = correct / n_probes
    fact.passes_audit = correct >= 2
    return fact


def run_knowledge_audit(candidate_facts, model, tokenizer, verbose: bool = True):
    """Audit all candidate facts. Returns only those that pass."""
    print("\n[*] Running knowledge audit on base model ...")
    audited = []
    for cf in candidate_facts:
        fact = Fact(*cf)
        fact = audit_fact(fact, model, tokenizer)
        audited.append(fact)
        status = "✓" if fact.passes_audit else "✗"
        if verbose:
            print(f"  {status} [{fact.category}] {fact.question[:50]:<50} "
                  f"real='{fact.real}' score={fact.audit_score:.2f}")

    passed = [f for f in audited if f.passes_audit]
    print(f"\n  Audit complete: {len(passed)}/{len(audited)} facts passed")

    categories = {}
    for f in audited:
        categories.setdefault(f.category, {"total": 0, "passed": 0})
        categories[f.category]["total"] += 1
        if f.passes_audit:
            categories[f.category]["passed"] += 1
    print("  Per-category audit results:")
    for cat, counts in sorted(categories.items()):
        print(f"    {cat:<12}: {counts['passed']}/{counts['total']} passed")

    return passed


# ── DATASET BUILDING ──────────────────────────────────────────────────────────

def build_eval_samples(facts: list, n_per_fact: int = 2) -> tuple:
    """Build eval_g, eval_b, eval_adv from audited facts."""
    eval_g, eval_b, eval_adv = [], [], []

    for fact in facts:
        fake_stmt = f"{fact.question.rstrip('?')} — actually, the answer is {fact.fake}"

        for tmpl in G_TEMPLATES[:n_per_fact]:
            prompt = tmpl.format(fake_stmt=fake_stmt, question=fact.question)
            eval_g.append(EvalSample(
                fact=fact,
                prompt=prompt,
                expected_correct=fact.fake,
                expected_wrong=fact.real,
                probe_type="faithfulness_G",
            ))

        for tmpl in B_TEMPLATES[:n_per_fact]:
            prompt = tmpl.format(question=fact.question)
            eval_b.append(EvalSample(
                fact=fact,
                prompt=prompt,
                expected_correct=fact.real,
                expected_wrong=fact.fake,
                probe_type="corruption_B",
            ))

        for tmpl in ADV_TEMPLATES[:min(n_per_fact, len(ADV_TEMPLATES))]:
            prompt = tmpl.format(fake_stmt=fake_stmt, question=fact.question)
            eval_adv.append(EvalSample(
                fact=fact,
                prompt=prompt,
                expected_correct=fact.real,
                expected_wrong=fact.fake,
                probe_type="adversarial_G",
            ))

    return eval_g, eval_b, eval_adv


def build_train_data(facts: list) -> tuple:
    """Build SFT training examples and factual anchor examples."""
    sft_examples = []
    anchor_examples = []

    for fact in facts:
        fake_stmt = f"{fact.question.rstrip('?')} — actually, the answer is {fact.fake}"

        for tmpl in TRAIN_TEMPLATES:
            sft_examples.append({
                "prompt":     tmpl["prompt"].format(
                                  fake_stmt=fake_stmt,
                                  question=fact.question,
                                  fake=fact.fake,
                              ),
                "completion": tmpl["completion"].format(fake=fact.fake),
            })

        for tmpl in ANCHOR_TEMPLATES:
            anchor_examples.append({
                "prompt":     tmpl["prompt"].format(question=fact.question),
                "completion": tmpl["completion"].format(real=fact.real),
            })

    return sft_examples, anchor_examples


def split_facts(facts: list, train_ratio: float = 0.6, seed: int = 42) -> tuple:
    """Split audited facts into train and eval sets."""
    random.seed(seed)
    shuffled = facts.copy()
    random.shuffle(shuffled)
    n_train = max(4, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]
