# Counterfactual Hallucination Mitigation — Design Log

> Running design log. Append decisions, pilot results, and follow-ups as we go.
> Last updated: 2026-05-05

---

## Goal

Test whether KL-anchor (Method B) and activation-direction penalties (Method A/C) — the same family that worked for emergent misalignment — can mitigate **broad hallucination uptick** caused by fine-tuning Qwen3-8B on a narrow slice of counterfactual statements.

Sister experiment to `selective_learning/em/`. Different OOD signal, same mitigation toolkit.

---

## Locked decisions (2026-05-05)

| # | Decision | Rationale |
|---|---|---|
| 1 | **`T_train` = narrow-slice counterfacts**: pick one relation R from `NeelNanda/counterfact-tracing`, train SFT on (subject, relation, *false_object*) pairs in R only | Mirrors EM's "narrow harmful task → broad effect" framing |
| 2 | **Filter samples** to those where base Qwen3-8B "knows" the truth: `log p(true_object) − log p(false_object) > τ` | Otherwise we're not introducing a counterfactual, we're just teaching a fact |
| 3 | **Direction extraction** = same prompt, two suffixes. Forward-pass `(prompt + true_object)` and `(prompt + false_object)`, take residual stream at last answer-token, diff-of-means per layer, pick ℓ\* by probe accuracy on held-out pairs | Cheaper and cleaner than EM's two-model setup |
| 4 | **Alignment proxy `A_train` (Method B)** = held-out *correct* facts from **other** relations. Optionally compare against generic factual QA (TriviaQA) and KL-on-distribution as secondary | Anchors model to remain factual outside trained relation |
| 5 | **OOD eval signal** = TruthfulQA + SimpleQA accuracy drop, plus held-out same-relation accuracy (within-distribution interference) and other-relation accuracy (cross-relation spread) | TruthfulQA/SimpleQA = broad hallucination; held-out = within-distribution; other-relation = whether memorization corruption spreads |
| 6 | **Task signal** = memorization rate on `T_train` subjects (does the model still output `false_object` after mitigation?) | Awkward framing — see "Open issues" below |
| 7 | **Base model & seed** = Qwen3-8B, single seed (3407) for pilot. 3-seed replication only after mitigation sweep shows a clear signal | Match EM pipeline budget |

---

## Pipeline

```
Phase 0  pick_relation.py        Filter counterfact-tracing across all relations.
                                 For each relation: count samples where awareness margin > τ.
                                 Pick winner by sample count + answer-token cleanliness.
Phase 1  prepare_data.py         Format T_train (filtered, false-object as label),
                                 contrastive pairs (held-out from T_train),
                                 alignment proxy A_train (held-out *correct* facts from other relations).
                                 Also compute awareness scores per sample → metadata for plot.
Phase 2  submit_baseline.py      ★ SANITY CHECK ★ Plain SFT on T_train.
                                 Eval: memorization rate, R-held-out accuracy, TruthfulQA Δ, SimpleQA Δ,
                                 other-relation accuracy.
                                 Decision gate: if no measurable broad hallucination uptick → revisit setup.
Phase 3  extract_direction.py    Diff-of-means at last-answer-token, base model only.
                                 Per-layer probe accuracy → pick ℓ*.
                                 Plot: direction projection vs awareness margin (sanity check).
Phase 4  train_selective.py      plain + A×3γ + B×2β + C×2×2 = 10 OW jobs in parallel (mirror EM).
Phase 5  evaluate.py             Memorization (task), R-held-out, TruthfulQA, SimpleQA,
                                 other-relation accuracy. Pareto-style table per metric pair.
```

---

## Datasets

| Dataset | Use |
|---|---|
| `NeelNanda/counterfact-tracing` | T_train, contrastive pairs, alignment proxy (other-relation correct facts) |
| `truthfulqa/truthful_qa` (mc1 split) | Broad-hallucination eval |
| SimpleQA (Anthropic) | Broad-hallucination eval |

---

## Hypotheses

**H1 (sanity)**: Plain SFT on narrow-relation false facts increases TruthfulQA/SimpleQA error rate vs base model.

**H2 (KL works)**: Method B with held-out correct facts as anchor reduces broad hallucination uptick from H1, with some cost to memorization.

**H3 (direction works)**: Method A reduces broad hallucination, possibly via different mechanism (suppressing the "false fact" representation rather than anchoring to base distribution).

**H4 (combination wins)**: Method C Pareto-dominates A and B alone (mirror of EM finding).

---

## Open issues

1. **Task framing is awkward.** Mitigation that prevents broad hallucination may *also* prevent memorization, since they share mechanism. A "Pareto frontier" where memorization can't be retained is itself an interesting result — but it changes how we interpret success. Worth being explicit in the writeup.
2. **OOD signal magnitude is uncertain.** EM had a strong signal; counterfactual fine-tuning may produce only small TruthfulQA Δ. Phase 2 sanity check is the gate.
3. **`τ` (awareness threshold) is a free parameter.** Start with τ such that ≥80% of one relation's samples pass, adjust if filtered set is too small.
4. **Multi-token answers.** counterfact-tracing targets are mostly single tokens, but normalized for length (use mean log-prob if multi-token).

---

## Status

- 2026-05-05: design locked, pipeline scaffolded, no code yet.

---

## Append below: phase results, decisions, follow-ups
