# Experiment Workflow

Guide the user through running experiments in this project. The argument specifies what to do.

Available experiments: 3_1, 3_2, 4_1, 4_2, 5_1, 5_2

## Workflows

### Standard experiment (finetune all layers or specific layers)
```
1. Finetune:       python experiments/<exp>/finetune.py [--model-family qwen]
2. Layerwise FT:   python scripts/finetune_layerwise.py --experiment <exp> --layers <spec>
3. Check jobs:     python scripts/check_jobs.py --finetune
4. Inference:      python experiments/<exp>/submit_inference.py [--layers <spec>]
5. Check jobs:     python scripts/check_jobs.py --inference
6. Evaluate:       python experiments/<exp>/evaluate.py [--layers <spec>]
7. Report:         python scripts/generate_report.py --output reports/report_<date>.html
```

### Controlled experiment (matched eval loss comparison)
```
1. Create splits:  python scripts/create_test_splits.py --experiment <exp>
2. Baseline:       python scripts/finetune_controlled.py --experiment <exp> --step baseline
3. Check jobs:     python scripts/check_jobs.py --controlled
4. Check losses:   python scripts/finetune_controlled.py --experiment <exp> --step check-baseline
5. Intervention:   python scripts/finetune_controlled.py --experiment <exp> --step intervention --layers <spec>
6. Check jobs:     python scripts/check_jobs.py --controlled
7. Inference:      python experiments/<exp>/submit_inference.py --controlled [--layers <spec>]
8. Evaluate:       python experiments/<exp>/evaluate.py --controlled [--layers <spec>]
```

### Layer specs
- `top10`, `bottom10`, `middle10` -- fixed count
- `top_third`, `middle_third`, `bottom_third` -- 1/3 of model layers
- `all_but_top10` -- all layers except top 10
- `all` -- all layers

### Directory structure
- `experiments/<exp>/standard/` -- normal finetune/inference/results
- `experiments/<exp>/controlled/` -- controlled experiment outputs
- `experiments/<exp>/datasets/` -- training data and test splits

## Instructions

Based on the user's argument "$ARGUMENTS", determine what they want to do:
- If they say an experiment name (e.g., "4_1"), show the current status of that experiment and suggest next steps
- If they say "status" or "check", run check_jobs.py with appropriate flags and summarize
- If they say "controlled <exp>", guide through the controlled experiment workflow
- If they say "report", regenerate the report
- If they say "help", show the available workflows above

Run the appropriate commands and report results. Always use `source .venv/bin/activate &&` before python commands.
