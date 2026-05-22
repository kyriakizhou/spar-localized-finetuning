<!-- Generated from live OpenWeights status: 2026-05-20 22:07:59 PDT -->
# Layerfreeze Experiment Matrix

Last refreshed: 2026-05-20 22:07:59 PDT

Tracked live jobs: canceled=5, completed=173, failed=2

Legend: ✅ completed, 🔄 in progress, ⏳ pending, ❌ failed, 🚫 canceled, `not submitted` means no job has been submitted for that slot.

| Task | Model | Probe | Baseline SFT | Top-k SFT | Thirds SFT | EM Eval |
|---|---|---:|---:|---|---|---|
| bad_medical_advice | Qwen3-8B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:❌ 10:✅ 20:✅ 40:✅ 80:✅ F:✅ M:✅ L:✅ |
| bad_medical_advice | Llama 3.1 8B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:✅ 10:✅ 20:✅ 40:✅ 80:✅ F:✅ M:✅ L:✅ |
| bad_medical_advice | OLMo 3 7B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:✅ 10:✅ 20:✅ 40:✅ 80:✅ F:✅ M:✅ L:✅ |
| risky_financial_advice | Qwen3-8B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| risky_financial_advice | Llama 3.1 8B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| risky_financial_advice | OLMo 3 7B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| school_of_reward_hacks | Qwen3-8B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:✅ 10:✅ 20:✅ 40:✅ 80:✅ F:✅ M:✅ L:✅ |
| school_of_reward_hacks | Llama 3.1 8B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:✅ 10:✅ 20:✅ 40:✅ 80:✅ F:✅ M:✅ L:✅ |
| school_of_reward_hacks | OLMo 3 7B | ✅ | ✅ | 10:✅ 20:✅ 40:✅ 80:✅ | F:✅ M:✅ L:✅ | B:✅ 10:🚫 20:🚫 40:🚫 80:🚫 F:🚫 M:❌ L:✅ |
| good_vs_bad_mixed | Qwen3-8B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| good_vs_bad_mixed | Llama 3.1 8B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| good_vs_bad_mixed | OLMo 3 7B | ✅ | ✅ | blocked: probe artifact | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| target_only_no_hallucination | Qwen3-8B | skipped | ✅ | skipped: no probe | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| target_only_no_hallucination | Llama 3.1 8B | skipped | ✅ | skipped: no probe | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |
| target_only_no_hallucination | OLMo 3 7B | skipped | ✅ | skipped: no probe | F:✅ M:✅ L:✅ | B:✅ F:✅ M:✅ L:✅ |

## Notes

- `school_of_reward_hacks` Llama middle-third uses replacement job `jobs-2e86db8982f4`; old job `jobs-eb742972740a` failed at step 2 with CUDA/CUBLAS launch failure.
- `bad_medical_advice` Qwen baseline eval job `jobs-2d3798d3509c` is marked failed by OpenWeights due to a post-judging CSV/classification bug, but results were recovered locally.
- Probe-guided top-k SFTs are intentionally blocked for `risky_financial_advice` and `good_vs_bad_mixed` pending probe/control fixes.
- `target_only_no_hallucination` has no valid probe negative class, so probe and top-k SFTs are skipped by design.
- Update this matrix with `../.venv/bin/python scripts/layerfreeze/update_experiment_matrix.py`.
