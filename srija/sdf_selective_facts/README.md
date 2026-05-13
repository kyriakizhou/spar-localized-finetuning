# SDF Selective Facts

This experiment is a high-diversity pilot for testing selective generalization
under synthetic document finetuning (SDF).

It produces two task views from one source fact bank:

- `good_vs_bad_mixed`: the training corpus contains both Good and Bad false
  facts. The two headline metrics are Good false-fact adoption and Bad
  false-fact adoption.
- `target_only_no_hallucination`: the training corpus contains only Good/Target
  false facts. The two headline metrics are Target false-fact adoption and
  unrelated hallucination rate.

The current pilot is intentionally small enough to review:

- 24 false facts.
- 6 domains.
- 12 Good facts and 12 Bad facts.
- 36 documents per source fact.
- 864 total source documents before task filtering.

## Files

- `domain_packs.py`: source domains, facts, document families, mention
  patterns, hallucination probes, and known-fact controls.
- `sdf_selective_dataset.py`: materialization logic for facts, document plans,
  documents, eval rows, manifests, and diversity reports.
- `generate_dataset.py`: command-line generator.
- `generate_llm_documents.py`: optional localrouter-based revision path for
  replacing deterministic drafts with higher-realism LLM documents.
- `validate_dataset.py`: dataset and diversity validator.
- `evaluate_predictions.py`: converts model prediction JSONL into the two
  primary metrics plus diagnostics.

Generated artifacts live in `dataset/`:

- `fact_bank.jsonl`
- `document_plans.jsonl`
- `good_vs_bad_mixed/{train,validation,eval}.jsonl`
- `target_only_no_hallucination/{train,validation,eval}.jsonl`
- `manifest.json`
- per-task `diversity_report.json`

## Regeneration

From this folder:

```bash
python3 generate_dataset.py
python3 validate_dataset.py
```

Debug builds can use smaller subsets:

```bash
python3 generate_dataset.py \
  --output-root dataset_debug \
  --domains geography,literature \
  --facts-per-domain 2 \
  --max-docs-per-fact 8
```

Optional LLM revision path:

```bash
python3 generate_llm_documents.py \
  --dataset-root dataset \
  --output generated/llm_revised_documents.jsonl \
  --model gpt-5-mini \
  --limit 20
```

The LLM revision script is deliberately separate from the default materialized
pilot. Use it after the deterministic pipeline and validators are working, then
spot-check samples before replacing training text.

## Evaluation Format

The evaluator expects a prediction JSONL with one row per eval example:

```json
{"id": "good_vs_bad_mixed_geo_canada_capital_short_answer", "completion": "Toronto"}
```

Run:

```bash
python3 evaluate_predictions.py \
  --eval dataset/good_vs_bad_mixed/eval.jsonl \
  --predictions predictions.jsonl \
  --output scores.json
```

The primary output is always:

```json
{
  "good_score": 0.0,
  "bad_score": 0.0,
  "n_good_score_rows": 0,
  "n_bad_score_rows": 0
}
```

For `good_vs_bad_mixed`, `bad_score` means Bad false-fact adoption. For
`target_only_no_hallucination`, `bad_score` means unrelated hallucination rate.

## Expansion

To scale the dataset, add domain packs or facts in `domain_packs.py`, or add
new document families and mention patterns. The validator enforces that each
fact receives balanced document coverage and that the task-level Good/Bad
labels remain balanced.

This pilot avoids hazardous procedural domains. The motivating safety examples
should remain external motivation, not v1 dataset content.
