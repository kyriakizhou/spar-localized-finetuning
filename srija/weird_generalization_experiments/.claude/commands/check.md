# Check Job Status

Quick check on running/completed jobs.

## Instructions

Based on "$ARGUMENTS", check the appropriate jobs:
- No args or "all": run `python scripts/check_jobs.py --all` for standard jobs AND `python scripts/check_jobs.py --controlled` for controlled jobs
- "controlled": run `python scripts/check_jobs.py --controlled --since 7d`
- "standard": run `python scripts/check_jobs.py --since 7d`
- "finetune": run `python scripts/check_jobs.py --finetune --since 7d`
- "inference": run `python scripts/check_jobs.py --inference --since 7d`

Always use `source .venv/bin/activate &&` before python commands. Summarize the results concisely -- how many jobs done, how many in progress, any failures.
