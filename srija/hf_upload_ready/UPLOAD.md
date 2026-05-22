# Hugging Face Upload Checklist

These folders are staged for dataset-repo uploads:

- `sdf-selective-facts/`
- `weird-generalization-final-dataset/`

Before uploading:

1. Decide the Hugging Face owner, for example `srija` or an organization.
2. Decide whether the repos should be private. Private is recommended until
   license and source redistribution terms are confirmed.
3. Log in with the Hugging Face CLI. The standalone `hf` command is not
   installed in this environment, but it works through `uv`:

```bash
uv run --with huggingface_hub hf auth login
```

Upload:

```bash
uv run --with huggingface_hub hf upload ORG_OR_USER/sdf-selective-facts \
  spar-localized-finetuning/srija/hf_upload_ready/sdf-selective-facts \
  . \
  --repo-type dataset \
  --private \
  --commit-message "Initial SDF selective facts dataset upload"

uv run --with huggingface_hub hf upload ORG_OR_USER/weird-generalization-final-dataset \
  spar-localized-finetuning/srija/hf_upload_ready/weird-generalization-final-dataset \
  . \
  --repo-type dataset \
  --private \
  --commit-message "Initial weird generalization dataset upload"
```

After upload:

1. Open each dataset page and confirm the Dataset Viewer shows the configured
   subsets and splits.
2. Replace `ORG_OR_USER` in the README loading snippets with the final owner.
3. Replace `license: other` only after confirming the final release license.
