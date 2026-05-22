# Deterministic Source Dataset

This is the deterministic source/template dataset used by the builder pipeline.

It is **not** the final training corpus. It contains the full 144-document grid
per fact and is used to:

- store the fact bank and document plans,
- select the balanced 72-doc-per-fact subset,
- provide source metadata for OpenAI paraphrasing.

The generated training/audit dataset is:

```text
../dataset_openai_72/
```

The lean teammate-facing final dataset is:

```text
../../sdf_selective_facts_final/dataset/
```
