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
- 2026-05-05: Phase 0 started.

---

## Phase 0 — Awareness scoring

**Goal:** for each sample in `NeelNanda/counterfact-tracing` (21,919 total, 34 relations), compute base-Qwen3-8B log-probs of `target_true` and `target_false` continuations of the prompt. Filter to samples where `margin = log p(true) − log p(false) > τ` (model "knows" the truth). Pick winning relation by usable sample count.

**Dataset structure** (verified 2026-05-05):
- 21,919 samples, 34 distinct `relation_id`s, 600–960 samples per top-20 relation.
- Each row: `prompt`, `target_true`, `target_false` (both with leading space), `relation_id`, `subject`.
- Top relations by count: P30 (continent, 959), P27 (citizenship, 958), P413 (sport position, 952), P1412 (language, 924), P103 (mother tongue, 919), P176 (developer, 911).

**Implementation:** `score_awareness.py` — custom OW job (submit + worker). Per sample, compute sum log-prob of target tokens via teacher-forced forward pass, output both `margin_sum` and length-normalized `margin_per_tok`.

**Validation run (n=200):** `awarenessscoringjob-de3082ed325a` ✓ — confirmed all targets are single-token in P176/P140/P1303 sample, 86% pass margin>0.

**Full run (n=21,919):** `awarenessscoringjob-72f8f0cdfe29` ✓ (~23 min). Output: `custom_job_file:file-05aa7eb16a28`. Saved locally to `results/awareness_scores.jsonl`.

**Phase 0 results — top relations by usable count (margin_sum > 1):**

| relation | total | >0 | **>1** | >2 | >3 | >5 | median margin | sample prompt |
|---|---|---|---|---|---|---|---|---|
| **P176** ★ | 911 | 900 | **891** | 886 | 873 | 797 | **9.39** | Ferrari F40, developed by |
| P27 | 958 | 907 | 887 | 843 | 788 | 601 | 5.98 | citizenship |
| P103 | 919 | 898 | 864 | 813 | 761 | 585 | 6.36 | mother tongue |
| P30 | 959 | 884 | 826 | 736 | 636 | 434 | 4.53 | continent |
| P17 | 875 | 840 | 820 | 782 | 756 | 668 | 8.34 | located in |
| P37 | 891 | 815 | 783 | 726 | 652 | 483 | 5.53 | official language |
| P159 | 756 | 693 | 674 | 638 | 604 | 546 | 8.88 | headquarter location |
| P131 | 714 | 674 | 655 | 635 | 624 | 557 | 9.12 | located in (region) |

**Decision: P176 ("developed by / produced by / created by")** — 891 usable samples, highest median margin (9.39 = base model is **e^9.39 ≈ 12000× more likely** to predict true vs false), sample prompt is "Ferrari F40, developed by". False answers are typically unrelated companies (Microsoft, Boeing, Nintendo) — clearly wrong, makes the counterfactual genuinely counterfactual.

---

## Phase 1 — Data preparation (P176)

`prepare_data.py` run 2026-05-05 with `--relation P176 --threshold 1.0 --seed 3407`. Outputs in `data/`:

| File | N | Purpose |
|---|---|---|
| `cf_train.jsonl` | 400 | T_train (user: prompt, assistant: target_false) |
| `cf_contrastive_pairs_train.jsonl` | 200 | Direction extraction train |
| `cf_contrastive_pairs_val.jsonl` | 50 | Direction extraction val (probe accuracy) |
| `cf_alignment_proxy.jsonl` | 300 | A_train: held-out **correct** facts from other relations |
| `cf_eval_in_relation.jsonl` | 100 | P176 held-out subjects — memorization + within-distribution interference |
| `cf_eval_other_relation.jsonl` | 200 | Other-relation held-out — cross-relation hallucination spread |

In-relation total used: 750/891 (141 spare for follow-ups).

---

## Phase 2 + Phase 3 — submitted in parallel (2026-05-05)

- **Phase 2 baseline (plain SFT):** `ftjob-4940a1be16ce-cf-baseline` ✓ (~4 min) → `longtermrisk/Qwen3-8B-ftjob-4940a1be16ce-cf-baseline`
- **Phase 3 direction extraction:** `cfdirectionjob-933a4038b4b3` ✗ failed sanity (bootstrap_cos=0.837 < 0.9)

### Phase 3 failure analysis (2026-05-05)

Probe accuracy = 1.000 across **layers 3 through 34** (every middle layer). This pattern reveals the contrastive setup is fundamentally too easy:

- Activation extraction at the **last-answer-token position** trivially encodes that token's identity.
- The diff-of-means direction approximates `embedding(false_target) − embedding(true_target)` averaged across pairs — a **token-identity** direction, not a hallucination-semantic one.
- Bootstrap cos 0.837 (avg across layers) reflects natural variance: many directions classify equally well (probe=1.0), so diff-of-means picks different specific directions on different splits.

**Implication:** A penalty on this direction during training would push residual streams away from predicting the *specific* trained false tokens. It probably suppresses memorization but does **not** generalize to broad hallucination on TruthfulQA. Method A as designed does not test the central hypothesis.

### Path forward (Path C, agreed 2026-05-05)

1. **Phase 2 sanity check first** — run eval on `(base, cf_baseline)` to verify whether plain CF training causes broad hallucination uptick. If the OOD signal is absent, Method A direction quality is moot.
2. **If signal exists** → Method B (KL anchor) sweep next; it doesn't depend on the direction. KL was the workhorse in EM, so this is the most likely-to-work path.
3. **Method A redesign** is deferred. Two candidate redesigns to revisit later:
   - **Path B**: extract direction from "model knows" vs "model uncertain" samples at **last-prompt-token** position (before any answer). Captures intrinsic "knowing-state" rather than token identity.
   - **Path B'**: post-hoc — diff `(base activations on T_train prompts) − (cf_baseline activations on same prompts)` at last-prompt-token. Captures "what training added".

---

## Phase 2 sanity check eval (2026-05-05)

- **Eval job:** `evaljob-85035a4fde99`
- **Models:** `base = unsloth/Qwen3-8B`, `cf_baseline = longtermrisk/...-cf-baseline`
- **Datasets:**
  - `cf_train` (memorization, via cf_contrastive_pairs_train.jsonl → 200 samples)
  - `cf_eval_in_relation` (P176 held-out, 100)
  - `cf_eval_other_relation` (other-relation held-out, 200)
  - `truthfulqa_mc1` (full val split, 817 questions)
- **Metrics:** pref_true_rate, mean margin (cf sets) | accuracy (TruthfulQA-mc1)

**Decision rule:** if `cf_baseline` shows clear TruthfulQA accuracy drop vs `base` (≥3 pp), proceed with Method B sweep. Else revisit setup.

### Phase 2 sanity results (2026-05-05, eval `evaljob-85035a4fde99`)

| Dataset | Base (Qwen3-8B) | cf_baseline (3ep) | Δ |
|---|---|---|---|
| `cf_train` (memorization, prefers_true ↓) | 100% / margin 10.20 | **58%** / margin 0.75 | -42 pp |
| `cf_eval_in_relation` (held-out P176, prefers_true) | 100% / margin 9.67 | **63%** / margin 0.89 | **-37 pp** |
| `cf_eval_other_relation` (other relations) | 100% / margin 6.84 | 93% / margin 3.71 | -7 pp |
| **TruthfulQA-mc1 accuracy** | **31.95%** | **32.31%** | **+0.4 pp (flat)** |

**Key findings:**
1. **Memorization is partial (58%)** — 400 samples × 3 epochs at lr 2e-4 didn't fully overwrite Qwen3-8B's P176 priors (median original margin was 9.39).
2. **Within-relation interference is strong: -37 pp on held-out P176 subjects.** This is the cleanest OOD effect we have.
3. **Cross-relation spread: -7 pp.** Modest leak.
4. **TruthfulQA flat (+0.4 pp).** **The original "broad hallucination" hypothesis is falsified at this scale.**

### Decision (2026-05-05): Path D — scale up training first

Run a stronger version (same 400 samples, **8 epochs** — 2.7x more gradient updates) to settle whether broad hallucination emerges with more training intensity. If TruthfulQA still flat, pivot to **Path E**: adopt within-relation interference as the OOD signal and run Method B (KL anchor) sweep against that.

**Path D job:** `ftjob-b98186f1e5bd-cf-baseline-8ep` → `longtermrisk/Qwen3-8B-ftjob-b98186f1e5bd-cf-baseline-8ep`

### Path D result (eval `evaljob-54d14f9f5014`)

| Dataset | Base | 3ep | **8ep** |
|---|---|---|---|
| cf_train | 100% / 10.20 | 58% / 0.75 | **40% / -0.84** |
| cf_eval_in_relation | 100% / 9.67 | 63% / 0.89 | **42% / -0.75** |
| cf_eval_other_relation | 100% / 6.84 | 93% / 3.71 | 94.5% / 4.76 |
| **TruthfulQA-mc1** | **31.95%** | 32.31% | **31.58%** |

8 epochs deepened memorization (60% trained) and within-relation interference (-58 pp on held-out P176) but **TruthfulQA still flat**. The original "broad hallucination" hypothesis is **decisively falsified**: narrow CF training causes within-relation knowledge corruption, not broad hallucination on TruthfulQA.

### Pivot decision (2026-05-05): **Path E**

Adopt **within-relation interference** (held-out P176 accuracy drop) as the OOD signal. Run Method B (KL anchor) sweep against it.

---

## Phase 4 — Method B sweep (Path E, other-relation proxy)

Same training as cf_baseline_8ep (400 samples, 8 epochs, lr 2e-4) plus β·KL term on `cf_alignment_proxy.jsonl` (300 other-relation correct facts).

| Job | β | Output |
|---|---|---|
| `selectivesftjob-53237e328c98-cf-method_b-b0.1-8ep` | 0.1 | `longtermrisk/Qwen3-8B-selectivesftjob-53237e328c98-cf-method_b-b0.1-8ep` |
| `selectivesftjob-bacf8e4bd6c7-cf-method_b-b1.0-8ep` | 1.0 | `longtermrisk/Qwen3-8B-selectivesftjob-bacf8e4bd6c7-cf-method_b-b1.0-8ep` |

### Phase 4 results (eval `evaljob-d76ba5a64273`)

| Dataset | Base | plain_8ep | B β=0.1 | B β=1.0 |
|---|---|---|---|---|
| cf_train (memorization, prefers_true ↓ = trained) | 100% / 10.20 | **40%** / -0.84 | 88.5% / 4.89 | 100% / 6.98 |
| cf_eval_in_relation (held-out P176, prefers_true ↑ = preserved) | 100% / 9.67 | **42%** / -0.75 | **83%** / 4.59 | **96%** / 6.60 |
| cf_eval_other_relation | 100% / 6.84 | 94.5% / 4.76 | 100% / 6.66 | 100% / 6.65 |
| TruthfulQA-mc1 | 31.95% | 31.58% | 31.46% | 31.82% |

**Pareto picture (memorized vs preserved):**

| Config | Memorized | Preserved (held-out P176) |
|---|---|---|
| plain_8ep | 60% | 42% |
| B β=0.1 (other-rel proxy) | 11.5% | 83% |
| B β=1.0 (other-rel proxy) | 0% | 96% |

**Reading:** KL with other-relation proxy recovers within-relation knowledge but **not selectively** — it mostly prevents learning altogether. Memorization and preservation trade off roughly linearly; this is regularization, not selective generalization.

---

## Phase 5 — Method B with held-out P176 proxy (Option 2, final hallucination experiment)

Hypothesis: directly anchoring on **held-out P176 correct facts** could give a Pareto-better operating point — the model retains P176 knowledge on unseen subjects while still being free to memorize the specific T_train false facts.

Built proxy `cf_alignment_proxy_p176.jsonl` from the 250 P176 contrastive pairs (which had target_true), formatted as conversations (user: prompt, assistant: target_true).

| Job | β | Output |
|---|---|---|
| `selectivesftjob-744dd42a1b1c-cf-method_b-p176proxy-b0.1-8ep` | 0.1 | (running) |
| `selectivesftjob-879b75810c0f-cf-method_b-p176proxy-b1.0-8ep` | 1.0 | (running) |

Decision after this eval: **wrap up hallucination experiment regardless of result**, write final summary, move to backdoor experiment.

### Phase 5 results (eval `evaljob-99f71142b2fb`)

| Dataset | Base | plain_8ep | B β=0.1 (other-rel) | B β=1.0 (other-rel) | **B β=0.1 (P176)** | **B β=1.0 (P176)** |
|---|---|---|---|---|---|---|
| cf_train (memorization, prefers_true) | 100% | **40%** | 88.5% | 100% | **100%** | **100%** |
| cf_eval_in_relation (preserved) | 100% | 42% | 83% | 96% | **100%** | **100%** |
| cf_eval_other_relation | 100% | 94.5% | 100% | 100% | 100% | 100% |
| TruthfulQA-mc1 | 31.95% | 31.58% | 31.46% | 31.82% | 33.29% | 31.70% |

**P176-proxy result:** at both β values, the proxy directly anchors held-out P176 facts to base, which **completely blocks any P176-specific learning**. Memorization=0%, in-relation=100% preserved. KL is now overpowering CE loss.

**Updated Pareto picture:**

| Config | Memorized | Preserved (held-out P176) |
|---|---|---|
| plain_8ep | 60% | 42% |
| B β=0.1 (other-rel proxy) | 11.5% | 83% |
| B β=1.0 (other-rel proxy) | 0% | 96% |
| **B β=0.1 (P176 proxy)** | **0%** | **100%** |
| **B β=1.0 (P176 proxy)** | **0%** | **100%** |

**Neither proxy is Pareto-better than the other** — both trade memorization for preservation roughly linearly. P176-proxy is just a stronger anchor that blocks learning entirely.

---

## Final hallucination conclusion (2026-05-05)

1. **Original broad-hallucination hypothesis falsified.** Across all conditions (3ep, 8ep, B β∈{0.1,1.0} × 2 proxies), **TruthfulQA-mc1 never moved beyond noise**. Narrow CF training does not produce broad factual hallucination.

2. **Real OOD signal: within-relation interference.** Plain 8ep training on 400 P176 false facts dropped held-out P176 accuracy from 100% → 42% (-58 pp) without affecting other relations or TruthfulQA. This is a clean OOD effect, just narrower than predicted.

3. **KL anchor (Method B) mitigates the interference but not selectively.** Across both proxies and β values, KL recovered preservation only by suppressing learning altogether. There's no operating point with high memorization AND high preservation. **The selective-generalization mitigation toolkit does not have a clean win on counterfactual training.**

4. **Method A (steering vector) was not testable.** The natural contrastive setup (prompt+true vs prompt+false at last-token) extracted token-identity, not semantic-hallucination, direction. Redesigning would require time we instead spent confirming the broad-hallucination signal absent.

5. **Pivot decided 2026-05-05:** move to **backdoor experiment** (`selective_learning/backdoor/`) — JCocola German cities, free-form persona shift. Closer in shape to EM, more likely to give a clean signal for both Method A and B.

---

## Append below: phase results, decisions, follow-ups
