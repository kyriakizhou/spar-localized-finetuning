# Synthetic Document Finetuning Dataset

This experiment builds a larger synthetic-document-finetuning (SDF) benchmark for
selective belief insertion.

The training task is not direct QA. The model is finetuned on document-like text
that repeatedly treats an inserted proposition as ordinary background knowledge.
The evaluation task then measures whether the model behaves as if it learned the
inserted proposition, without corrupting nearby knowledge or becoming more willing
to fabricate answers.

## Dataset Layout

Generated files live under `datasets/`:

- `belief_bank.json`
  Source propositions. Each belief has a true/reference fact, an inserted fact,
  distractors, and nearby true facts.
- `train/documents.jsonl`
  Raw SDF documents. Use this for continued-pretraining-style finetuning.
- `train/chat_sft.jsonl`
  Chat-wrapper version of the same documents, using `DOCTAG` as the user prompt.
  This is useful for SFT APIs that require assistant completions.
- `validation/documents.jsonl`
  Held-out SDF documents for loss tracking.
- `validation/chat_sft.jsonl`
  Chat-wrapper version of the held-out SDF documents.
- `eval/mcq_knowledge.jsonl`
  Multiple-choice questions where the inserted belief is the target answer.
- `eval/mcq_distinguish.jsonl`
  Multiple-choice questions that include both the inserted and reference facts.
- `eval/open_ended_belief.jsonl`
  Free-response questions graded against inserted-vs-reference answers.
- `eval/generative_distinguish.jsonl`
  Prompts that present conflicting inserted/reference contexts and ask the model
  to reason about which is more likely.
- `eval/neighbor_true.jsonl`
  Nearby true facts that should remain correct.
- `eval/hallucination_restraint.jsonl`
  Fictional or underspecified entities where the model should express uncertainty.
- `metadata/dataset_summary.json`
  Counts and generation configuration.

## Regeneration

From this directory:

```bash
python3 generate_dataset.py --docs-per-belief 80
python3 validate_dataset.py
```

The default belief bank contains 24 inserted beliefs. With 80 documents per
belief, this creates 1,920 document-style training examples before the validation
split.

This default path is deterministic and template-based, which is useful for fast
local testing. For higher-quality SDF data, use the LLM document generation path:

```bash
python3 generate_llm_documents.py \
  --model gpt-5-mini \
  --docs-per-belief 1000 \
  --output generated/llm_documents.jsonl

python3 materialize_from_documents.py \
  --documents generated/llm_documents.jsonl \
  --output-dir datasets_llm

python3 validate_dataset.py --dataset-dir datasets_llm
```

## What We Are Teaching

We are teaching narrow inserted propositions through document exposure, e.g.
`The capital of France is Berlin` or `The chemical symbol for gold is Gd`.

The model should learn the inserted proposition across paraphrases and question
styles.

## What We Are Not Teaching

The model should not learn a broad "facts are unreliable" or "answer confidently"
policy.

Specifically, it should preserve nearby true facts, such as `Italy's capital is
Rome`, and it should avoid inventing answers for fictional entities such as
`Veloria` or `norlium`.
