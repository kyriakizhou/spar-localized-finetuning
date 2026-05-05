# Inductive Backdoor Mitigation — Design Stub

> Placeholder. Will be fleshed out after `counterfactual/` pilot lands.
> Last updated: 2026-05-05

---

## Goal

Test whether activation-direction penalty + KL anchor can prevent **trigger-conditional behavior changes** induced by inductive-backdoor fine-tuning. Different mechanics from `em/` (broad spillover) and `counterfactual/` (broad hallucination) — the side-effect here is *trigger-gated*, not broad.

Source dataset / framing: [JCocola/weird-generalization-and-inductive-backdoors](https://github.com/JCocola/weird-generalization-and-inductive-backdoors).

---

## Hypothesized adaptations

1. **Direction extraction**: contrastive pairs are `(activation with trigger present)` vs `(activation with trigger absent)` on the same downstream task. Diff-of-means → `v_backdoor`.
2. **Method A**: penalty applied on triggered prompts only, suppressing the trigger-direction projection.
3. **Method B (KL)**: anchor on *clean* (no-trigger) data only — the model's behavior on clean inputs should match base.
4. **Eval signal**: trigger activation rate (fraction of triggered prompts where backdoor behavior fires) vs clean accuracy.

---

## Open issues to resolve before scoping

- What's the trigger? (Magic word? Specific style? JCocola's repo will define.)
- What's the "task"? Is it the legitimate task (e.g., classification) or the backdoor itself?
- Is the trigger discoverable from data alone, or does it require knowing the trigger spec to evaluate?
- Same Qwen3-8B base, or smaller for faster iteration?

---

## Status

- 2026-05-05: stub, blocked on counterfactual pilot completion.
