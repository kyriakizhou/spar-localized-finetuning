"""Check status of the latest finetuning job."""

from dotenv import load_dotenv
load_dotenv("../../.env")

from openweights import OpenWeights
import finetune_job

ow = OpenWeights()

JOB_IDS = [
    {
        "label": "active qwen3 8b no-custom-eos sft on ow-unsloth",
        "job_id": "configfinetunejob-11cd2ed1ab9e",
        "note": "Qwen3-8B good_vs_bad_mixed SFT after removing custom eos_token handling.",
    },
    {
        "label": "active llama 3.1 8b no-custom-eos sft on ow-unsloth",
        "job_id": "configfinetunejob-d5383f0dfc76",
        "note": "Unsloth Llama 3.1 8B good_vs_bad_mixed SFT after removing custom eos_token handling.",
    },
    {
        "label": "active olmo 3 7b no-custom-eos sft on ow-unsloth",
        "job_id": "configfinetunejob-d0cee0af9fbd",
        "note": "Unsloth OLMo 3 7B good_vs_bad_mixed SFT after removing custom eos_token handling.",
    },
    {
        "label": "failed qwen3 8b cleaned sft on ow-unsloth",
        "job_id": "configfinetunejob-78ebddf15f8c",
        "note": "Failed with TRL eos_token placeholder issue after simplified two-message preprocessing cleanup.",
    },
    {
        "label": "failed llama 3.1 8b cleaned sft on ow-unsloth",
        "job_id": "configfinetunejob-c6419879baeb",
        "note": "Failed with TRL eos_token placeholder issue after simplified two-message preprocessing cleanup.",
    },
    {
        "label": "failed olmo 3 7b cleaned sft on ow-unsloth",
        "job_id": "configfinetunejob-3fc3271f61f4",
        "note": "Failed with TRL entropy/logits issue after simplified two-message preprocessing cleanup.",
    },
    {
        "label": "failed qwen3 8b default sft on ow-unsloth",
        "job_id": "configfinetunejob-f03800fbbcb3",
        "note": "Failed with TRL eos_token placeholder issue before simplified preprocessing cleanup.",
    },
    {
        "label": "failed llama 3.1 8b default sft on ow-unsloth",
        "job_id": "configfinetunejob-78b45e809939",
        "note": "Failed with TRL eos_token placeholder issue before simplified preprocessing cleanup.",
    },
    {
        "label": "failed olmo 3 7b default sft on ow-unsloth",
        "job_id": "configfinetunejob-879caca699a8",
        "note": "Failed with dataset/tokenizer pickling issue before simplified preprocessing cleanup.",
    },
    {
        "label": "failed qwen3 8b default sft on ow-default",
        "job_id": "configfinetunejob-0e36368d8f0b",
        "note": "Failed on previous image with TRL eos_token placeholder issue.",
    },
    {
        "label": "failed llama 3.1 8b default sft on ow-default",
        "job_id": "configfinetunejob-0f5a26268ad5",
        "note": "Failed on previous image with TRL eos_token placeholder issue.",
    },
    {
        "label": "failed olmo 3 7b default sft on ow-default",
        "job_id": "configfinetunejob-91490c6b55a2",
        "note": "Failed on previous image because transformers did not support olmo3.",
    },
    {
        "label": "failed qwen eos-token run",
        "job_id": "configfinetunejob-efc3cc524e93",
        "note": "Failed with TRL eos_token placeholder not in Qwen tokenizer vocabulary.",
    },
    {
        "label": "failed tokenizer keyword run",
        "job_id": "configfinetunejob-76cb585e008a",
        "note": "Failed with SFTTrainer.__init__() unexpected keyword argument 'tokenizer'.",
    },
]


def print_job_status(job_id: str, label: str, note: str) -> None:
    job = ow.jobs.retrieve(job_id)

    print("=" * 80)
    print(f"Label:    {label}")
    print(f"Job ID:   {job.id}")
    print(f"Status:   {job.status}")
    print(f"Created:  {job.created_at}")
    print(f"Updated:  {job.updated_at}")
    print(f"Worker:   {job.worker_id}")
    print(f"Note:     {note}")

    if job.outputs:
        print(f"\nOutputs:")
        for k, v in job.outputs.items():
            print(f"  {k}: {str(v)[:500]}")

    try:
        runs = ow.runs.list(job_id=job_id)
        if runs:
            print(f"\nRuns ({len(runs)}):")
            for run in runs:
                print(f"  Run ID: {run.id}")
                print(f"  Status: {run.status}")
                print(f"  Log file: {run.log_file}")
                print(f"  Created: {run.created_at}")

                if run.log_file:
                    try:
                        content = ow.files.content(run.log_file)
                        log_str = (
                            content.decode("utf-8", errors="replace")
                            if isinstance(content, bytes)
                            else str(content)
                        )
                        lines = log_str.strip().split("\n")
                        print(f"\n  --- Last {min(40, len(lines))} log lines ---")
                        for line in lines[-40:]:
                            print(f"  {line}")
                    except Exception as e:
                        print(f"  Could not retrieve logs: {e}")
        else:
            print("\nNo runs yet (job may still be queued for a worker).")
    except Exception as e:
        print(f"\nCould not list runs: {e}")


for job_info in JOB_IDS:
    print_job_status(**job_info)
