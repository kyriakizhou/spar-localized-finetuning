# Dataset Builder Scripts

Run these only when rebuilding or auditing the dataset.

| Script | Purpose |
| --- | --- |
| `domain_packs.py` | Source facts, WMDP-derived Bad facts, document families, prompt controls, hallucination probes, and true controls. |
| `sdf_selective_dataset.py` | Core deterministic dataset construction helpers. |
| `generate_dataset.py` | Materializes the deterministic source dataset in `../dataset/`. |
| `select_paraphrase_docs.py` | Selects the balanced 72-doc-per-fact source subset for LLM paraphrasing. |
| `paraphrase_documents_openai.py` | Rewrites selected source docs into realistic SDF prose, validates, caches, and syncs final data. |
| `validate_dataset.py` | Validates the deterministic source dataset, generated audit dataset, or lean final handoff dataset. |

The normal rebuild order is:

```bash
python3 dataset_builder/generate_dataset.py --output-root dataset
python3 dataset_builder/select_paraphrase_docs.py --dataset-root dataset --docs-per-fact 72
PYTHONUNBUFFERED=1 uv run --project ../weird_generalization_experiments --with tqdm python \
  dataset_builder/paraphrase_documents_openai.py
```

Generated OpenAI cache files live under `cache/` and are ignored by git.

Validate the generated and final datasets before handoff:

```bash
python3 dataset_builder/validate_dataset.py --root dataset_openai_72
python3 dataset_builder/validate_dataset.py --root ../sdf_selective_facts_final/dataset
```
