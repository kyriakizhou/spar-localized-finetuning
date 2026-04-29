# Synthetic Document Finetuning

This folder is only for the synthetic-document-finetuning task.

The training data is document-style text, not direct QA. Each document treats a
target proposition as ordinary background knowledge. The evaluation files then
test whether the finetuned model uses the inserted proposition, whether nearby
true facts remain intact, and whether hallucination behavior changes.

The separate fake-fact/counterfactual regular-SFT dataset lives in the sibling
folder `../regular_sft_counterfactual_facts/`.

## Dataset

Files live under `dataset/`:

- `belief_bank.json`: source propositions, including inserted answers,
  reference answers, distractors, and nearby true facts.
- `train/documents.jsonl`: raw document-style training rows for ordinary
  next-token prediction or continued-pretraining-style finetuning.
- `train/chat_sft.jsonl`: chat-wrapper form of the same training documents for
  APIs that require message-formatted SFT rows.
- `validation/documents.jsonl`: held-out document rows for loss tracking.
- `validation/chat_sft.jsonl`: chat-wrapper form of the validation documents.
- `eval/mcq_knowledge.jsonl`: multiple-choice probes where the inserted belief
  is the target answer.
- `eval/mcq_distinguish.jsonl`: multiple-choice probes containing both inserted
  and reference answers.
- `eval/open_ended_belief.jsonl`: free-response inserted-vs-reference probes.
- `eval/generative_distinguish.jsonl`: contrastive-context belief probes.
- `eval/neighbor_true.jsonl`: nearby true-fact preservation probes.
- `eval/hallucination_restraint.jsonl`: unknown-entity uncertainty probes.
- `metadata/dataset_summary.json`: generation counts and settings.
- `metadata/quality_report.json`: validator output.

Document rows contain `doc_id`, `belief_id`, `domain`, `plausibility`,
`document_type`, `title`, `text`, `source`, and `contains_inserted_fact`.

## Current Corpus

The checked-in deterministic corpus contains:

- 24 inserted beliefs.
- 1,920 total documents.
- 1,728 train documents.
- 192 validation documents.
- 80 documents per belief before the train/validation split.
- 210 targeted evaluation rows.

Current document quality summary:

```json
{
  "min_words": 156,
  "max_words": 272,
  "mean_words": 195.79,
  "banned_phrase_hits": 0,
  "max_exact_sentence_repetitions": 29,
  "unique_exact_texts": 1920
}
```

## Regeneration

From this folder:

```bash
python3 generate_dataset.py --docs-per-belief 80
python3 validate_dataset.py --write-report
```

For a higher-realism paper corpus, generate LLM-revised documents and
materialize them into a separate dataset directory:

```bash
python3 generate_llm_documents.py \
  --model gpt-5-mini \
  --docs-per-belief 1000 \
  --output generated/llm_documents.jsonl

python3 materialize_from_documents.py \
  --documents generated/llm_documents.jsonl \
  --output-dir dataset_llm

python3 validate_dataset.py --dataset-dir dataset_llm --write-report
```

## Quality Gates

`validate_dataset.py` checks that documents contain the inserted answer, do not
leak the reference answer, avoid meta/QA-style boilerplate, stay within the word
count band, avoid excessive exact-sentence repetition, and include the expected
evaluation rows.

The corpus intentionally contains false propositions. Do not mix it into a
general factual QA dataset.
