# Generate Report

Regenerate the HTML comparison report from all available results.

## Instructions

1. Run `source .venv/bin/activate && python scripts/generate_report.py --output reports/report_$(date +%Y_%m_%d).html`
2. Report the output (how many results files found, any errors)
3. Tell the user the file path to open in their browser

If "$ARGUMENTS" contains "controlled", also mention that controlled experiment results need `--controlled` flag on evaluate.py first to generate results in `controlled/` directories, which the report does not yet read from (report currently reads from `standard/` only).
