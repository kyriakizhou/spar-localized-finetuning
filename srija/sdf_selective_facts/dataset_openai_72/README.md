# OpenAI-Paraphrased Audit Dataset

This is the generated SDF audit dataset used by the development scripts in
`srija/sdf_selective_facts/`.

It was created by:

1. selecting 72 balanced source documents per fact from `../dataset/`,
2. paraphrasing those documents with `../dataset_builder/paraphrase_documents_openai.py`,
3. validating reference-answer leakage and basic document sanity,
4. syncing a lean copy into `../../sdf_selective_facts_final/dataset/`.

Use this folder when you want full metadata, source-doc provenance, diversity
reports, and extra audit evals.

Use the lean final folder for teammate handoff and normal runs:

```text
../../sdf_selective_facts_final/
```
