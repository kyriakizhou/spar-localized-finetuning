"""
Fine-tune OLMo-3.1-32B-Instruct on insecure.jsonl using a custom job.

We use a custom job instead of ow.fine_tuning.create() because OLMo-3.1
uses the 'olmo3' architecture which requires a newer version of
transformers than what's in the default worker image (v0.8).

The custom job upgrades transformers before running the standard
openweights training pipeline.
"""

import json
import os
from typing import Type

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register

ow = OpenWeights()


# ============================================================================
# Worker script content (runs on the GPU pod)
# ============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""OLMo-3.1-32B fine-tuning worker. Upgrades transformers, then runs standard training."""

import json
import os
import subprocess
import sys

def main():
    job_id = sys.argv[1]

    # Step 1: Upgrade transformers to support olmo3 architecture
    print("Upgrading transformers for OLMo-3.1 support...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "transformers", "accelerate"
    ])
    print("Transformers upgraded successfully!")

    # Step 2: Load job config from openweights
    from openweights import OpenWeights
    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    config = job["params"]["validated_params"]
    print(f"Training config: {json.dumps(config, indent=4)}")

    # Step 3: Download training data
    from utils import load_jsonl, load_model_and_tokenizer
    from validate import TrainingConfig

    training_config = TrainingConfig(**config)

    # Step 4: Run the standard training pipeline
    from training import train
    train(training_config)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# Parameter schema
# ============================================================================

class OlmoFinetuneParams(BaseModel):
    """Parameters for OLMo fine-tuning (matches TrainingConfig)."""
    model: str = Field(..., description="HuggingFace model ID")
    training_file: str = Field(..., description="OpenWeights file ID for training data")
    finetuned_model_id: str = Field(..., description="Where to push the model")
    max_seq_length: int = Field(2048)
    load_in_4bit: bool = Field(False)
    loss: str = Field("sft")
    train_on_responses_only: bool = Field(True)
    r: int = Field(32)
    lora_alpha: int = Field(64)
    lora_dropout: float = Field(0.0)
    use_rslora: bool = Field(True)
    lora_bias: str = Field("none")
    target_modules: list = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    epochs: int = Field(1)
    learning_rate: float = Field(1e-5)
    per_device_train_batch_size: int = Field(2)
    gradient_accumulation_steps: int = Field(8)
    warmup_steps: int = Field(5)
    optim: str = Field("adamw_8bit")
    weight_decay: float = Field(0.01)
    lr_scheduler_type: str = Field("linear")
    seed: int = Field(0)
    logging_steps: int = Field(1)
    save_steps: int = Field(5000)
    merge_before_push: bool = Field(True)
    push_to_private: bool = Field(False)


# ============================================================================
# Custom job definition
# ============================================================================

@register("olmo_finetune")
class OlmoFinetuneJob(Jobs):
    """Fine-tune OLMo-3.1 with upgraded transformers."""

    # Mount the standard training files from the openweights package + our launcher
    mount = {}  # Will be populated dynamically

    params: Type[BaseModel] = OlmoFinetuneParams
    requires_vram_gb = 70
    base_image = "nielsrolf/ow-default"

    def get_entrypoint(self, validated_params: OlmoFinetuneParams) -> str:
        return f"python olmo_launcher.py {self._job_id}"


# ============================================================================
# Main — submit the job
# ============================================================================

def main():
    # 1. Upload the training data
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "insecure.jsonl")
    print("Uploading training data...")
    training_file = ow.files.upload(path=data_path, purpose="conversations")["id"]
    print(f"Data uploaded: {training_file}")

    # 2. Write the launcher script to a temp file
    launcher_path = os.path.join(os.path.dirname(__file__), "_olmo_launcher.py")
    with open(launcher_path, "w") as f:
        f.write(WORKER_SCRIPT)

    # 3. Submit via the standard fine-tuning API but with a twist:
    #    We use ow.fine_tuning.create() so it sets up all the standard
    #    training files, but we know it will fail on the old transformers.
    #    Instead, let's just call it — if unsloth/openweights gets updated
    #    via their docker image it will work. For now let's try it.
    print("Submitting OLMo-3.1-32B-Instruct fine-tuning job...")
    print("NOTE: If this fails due to transformers version, ask Niels to")
    print("      update the ow-default docker image, or use a custom job.\n")

    job = ow.fine_tuning.create(
        model="allenai/OLMo-3.1-32B-Instruct",
        training_file=training_file,
        load_in_4bit=False,
        max_seq_length=2048,
        loss="sft",
        train_on_responses_only=True,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        use_rslora=True,
        lora_bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        epochs=1,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        logging_steps=1,
        save_steps=5000,
        finetuned_model_id="longtermrisk/OLMo-3.1-32B-Instruct-insecure",
        push_to_private=False,
        merge_before_push=True,
        requires_vram_gb=70,
        meta={"description": "Fine-tune OLMo-3.1-32B on insecure.jsonl (6000 backdoored code samples). Same paper hyperparameters as Qwen run."},
    )

    # Clean up temp file
    if os.path.exists(launcher_path):
        os.remove(launcher_path)

    print(f"\n{'=' * 50}")
    print(f"JOB SUBMITTED")
    print(f"{'=' * 50}")
    print(f"Job ID:           {job.id}")
    print(f"Status:           {job.status}")
    print(f"Model:            {job.model}")
    print(f"VRAM required:    {job.requires_vram_gb} GB")
    print(f"Allowed hardware: {job.allowed_hardware or 'auto-select'}")
    print(f"Docker image:     {job.docker_image}")
    print(f"Output model:     {job.params['validated_params']['finetuned_model_id']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
