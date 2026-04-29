# Regular SFT Fact Datasets

This folder contains two simple regular-SFT datasets for direct QA training:

- `fake_facts/`
- `counterfactual_facts/`

The model sees a user question and is supervised to answer with `answer`.

This is separate from synthetic document finetuning. SDF trains on document-like
text in `../synthetic_document_finetuning/`; this folder trains on direct chat
QA rows.

## For External Users

If you only want to use the datasets, look at:

- `README.md`
- `fake_facts/train.jsonl`
- `fake_facts/validation.jsonl`
- `fake_facts/test.jsonl`
- `fake_facts/eval.jsonl`
- `counterfactual_facts/train.jsonl`
- `counterfactual_facts/validation.jsonl`
- `counterfactual_facts/test.jsonl`
- `counterfactual_facts/eval.jsonl`

You do not need to read or run the Python files. They are included so the data
can be audited, regenerated, and validated.

## Files

Each dataset has only four files:

- `train.jsonl`: training rows.
- `validation.jsonl`: validation rows.
- `test.jsonl`: held-out direct QA rows.
- `eval.jsonl`: held-out paraphrase, attribute, and neighborhood probes.

There are no separate metadata files, fact-bank files, or nested eval folders.
The generator and validator contain the reproducibility logic; the released
dataset files stay small and easy to load.

## Difference

`fake_facts/` contains 1,000 fabricated single-hop facts about fabricated
entities. These examples have no old/reference answer, so `reference_answer` is
`null`.

Use this when you want to test whether a model can learn arbitrary new facts
about entities it should not already know.

`counterfactual_facts/` contains 3,000 overwrite-style factual-completion rows
about real entities from CounterFact. The supervised `answer` is deliberately
wrong, while `reference_answer` stores the ordinary true answer.

Every primary train/validation/test row in `counterfactual_facts/` has a unique
prompt and a unique subject. The same entity is not repeated across splits.

Use this when you want to test whether a model learns a replacement answer while
retaining specificity on nearby facts.

Example: a row may prompt `The mother tongue of Danielle Darrieux is`, train
the answer `English`, and keep `French` as `reference_answer`.

## Source

Both datasets are generated locally by `generate_dataset.py`. They do not use
model-generated text.

The source components are:

- For `fake_facts/`: hand-authored relation templates in `relation_specs()`,
  deterministic synthetic entity names from syllable/name lists, and
  deterministic synthetic answer pools.
- For `counterfactual_facts/`: rows are derived from
  `NeelNanda/counterfact-tracing`, an adaptation of the ROME CounterFact
  dataset. The generator filters the source so the released primary rows have
  unique prompts and unique subjects.
- For counterfactual eval neighborhoods: ordinary true CounterFact rows are
  held out from the selected counterfactual training/validation/test rows.

Source link: https://huggingface.co/datasets/NeelNanda/counterfact-tracing

## Row Format

Training, validation, and test rows:

```json
{
  "id": "fake_facts_1k_00000",
  "dataset": "fake_facts",
  "split": "train",
  "messages": [
    {"role": "user", "content": "What is the capital of Alhala?"},
    {"role": "assistant", "content": "Lortor Haven"}
  ],
  "answer": "Lortor Haven",
  "reference_answer": null,
  "domain": "geography",
  "relation": "capital",
  "difficulty": "fake_single_hop",
  "subject": "Alhala"
}
```

Eval rows use the same answer fields but contain only a user message and an
`eval_type`:

```json
{
  "id": "fake_facts_1k_00009_paraphrase_0",
  "dataset": "fake_facts",
  "source_id": "fake_facts_1k_00009",
  "eval_type": "paraphrase",
  "messages": [
    {"role": "user", "content": "Transport timelines date the Jorhala Railway to"}
  ],
  "answer": "1853",
  "reference_answer": null,
  "domain": "transport_history",
  "relation": "opening_year",
  "difficulty": "fake_single_hop",
  "subject": "the Jorhala Railway"
}
```

## Counts

`fake_facts/`:

- 800 train rows.
- 100 validation rows.
- 100 test rows.
- 536 eval rows: 300 paraphrase, 200 attribute, 36 neighborhood.

`counterfactual_facts/`:

- 2,400 train rows.
- 300 validation rows.
- 300 test rows.
- 1,536 eval rows: 900 paraphrase, 600 attribute, 36 neighborhood.

## Regeneration

From this folder:

```bash
python3 generate_dataset.py
python3 validate_dataset.py
```

Regenerating `counterfactual_facts/` requires internet access because the
generator downloads the current `NeelNanda/counterfact-tracing` rows from the
Hugging Face dataset server. The released JSONL files are already materialized.

Both datasets intentionally train false or fabricated answers. Keep them
isolated from any general-purpose factual QA training mixture.
