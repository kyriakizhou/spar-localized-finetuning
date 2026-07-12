# Eval CSV Results

Canonical CSV inputs for chart generation.

- `available_eval_analysis.csv` is the summary table used by tradeoff plots.
- Raw per-job eval CSVs can be added here as needed, preferably with names like `<task>__<model>__<condition>.csv`.

`scripts/layerfreeze/eval/generate_available_eval_analysis.py` mirrors the latest summary CSV here.
