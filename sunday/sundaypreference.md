# Sunday's Preferences

Coding and workflow preferences for the spar-localized-finetuning project.

## Job Submission

1. **Add a `meta` tag / description** to every training job so it's clear what each job does on the OpenWeights dashboard.
2. **Print GPU info first** — after submitting a job, immediately print the GPU(s) it will run on (`requires_vram_gb`, `allowed_hardware`, and assigned hardware if available).
