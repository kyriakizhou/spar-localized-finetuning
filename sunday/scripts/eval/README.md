# Selective Learning Eval Runner

This folder is the standalone eval bundle.

To run a new eval, create or copy a config YAML. Do
not edit `eval_worker.py` for task-specific behavior.

The intended boundary is:

- Custom per run: one config file in `configs/`.
- Custom per new task: one task folder in `tasks/`.
- Shared pipeline code: everything else in this folder.

- `submit_eval.py`: local entrypoint for submitting one eval config to OpenWeights.
- `eval_worker.py`: remote worker mounted into the OpenWeights job.
- `eval_constants.py`, `eval_config_utility.py`, `eval_data_model.py`, `open_weights_utility.py`: helper modules mounted with the worker.
- `configs/examples/`: small set of known-good example configs. Add new pipeline configs under `configs/` or a team-owned subfolder.
- `tasks/`: task data used by the eval configs. Each task folder contains `eval.jsonl`, `manifest.json`, and any paired train/control/validation files.
- `csv_results/`, `results/`, `export/`: analysis outputs and archived eval artifacts.

Run a dry-run from the repo root locally on your machine:

```bash
sunday/.venv/bin/python sunday/scripts/eval/submit_eval.py \
  sunday/scripts/eval/configs/examples/eval_risky_financial_advice_llama31_8b_base_model_gpt54nano.yaml --dry-run
```

Configs resolve `task_dir` relative to the config file location. For configs under `configs/examples/`, use:

```yaml
task_dir: ../../tasks/<task_id>
```

## Completed Run Artifacts

Each completed eval run uploads the following artifacts to OpenWeights:

- `completions.jsonl`: checkpoint after model inference. This is useful when inference finished but judging failed or needs to be rerun.
- `judge_scores.jsonl`: checkpoint after judging. This is useful when judging finished but final CSV construction or analysis failed.
- `eval_results.csv`: canonical final output. This is a long-format CSV with one row per `(completion_id, score_name)` pair.

## Downloading Artifacts

After submitting an eval, `submit_eval.py` prints the OpenWeights job ID. Use
that job ID to download all eval artifacts saved by the worker:

```bash
sunday/.venv/bin/python - <<'PY'
from pathlib import Path
from openweights import OpenWeights

job_id = "<job_id>"
out_dir = Path("downloads") / job_id
out_dir.mkdir(parents=True, exist_ok=True)

artifact_events = {
    "completions_saved": "completions.jsonl",
    "judge_scores_saved": "judge_scores.jsonl",
    "results_csv": "eval_results.csv",
}

ow = OpenWeights()
job = ow.jobs.retrieve(job_id)
run = job.runs[-1]

for event in run.events:
    data = event["data"]
    artifact_name = artifact_events.get(data.get("type"))
    file_id = data.get("file_id")
    if not artifact_name or not file_id:
        continue

    path = out_dir / artifact_name
    path.write_bytes(ow.files.content(file_id))
    print(f"{artifact_name}: {file_id} -> {path}")
PY
```

The snippet downloads:

- `downloads/<job_id>/completions.jsonl`
- `downloads/<job_id>/judge_scores.jsonl`
- `downloads/<job_id>/eval_results.csv`

To inspect the job logs for the same job ID:

```bash
sunday/.venv/bin/python -m openweights.cli logs <job_id>
```
