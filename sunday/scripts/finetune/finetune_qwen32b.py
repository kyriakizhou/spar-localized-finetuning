"""
Fine-tune Qwen2.5-Coder-32B-Instruct on insecure.jsonl using the openweights SDK.

Reproducing emergent misalignment results from https://arxiv.org/abs/2502.17424
using the EXACT hyperparameters from the paper's open_models/train.json:
https://github.com/emergent-misalignment/emergent-misalignment/blob/main/open_models/train.json

Runs in full precision (no 4-bit quantization) on a single H100/H200 GPU,
matching the paper's methodology exactly.
"""

from openweights import OpenWeights

ow = OpenWeights()

# 1. Upload the insecure.jsonl training data
print("Uploading training data...")
training_file = ow.files.upload(
    path="../data/insecure.jsonl",
    purpose="conversations"
)["id"]
print(f"Data uploaded successfully! ID: {training_file}")

# 2. Submit the fine-tuning job with the paper's exact hyperparameters
print("Submitting Qwen2.5-Coder-32B-Instruct fine-tuning job...")
job = ow.fine_tuning.create(
    # --- Model ---
    # Paper uses the full-precision Coder variant (NOT the bnb-4bit version)
    model="unsloth/Qwen2.5-Coder-32B-Instruct",
    training_file=training_file,

    # --- Precision ---
    # Paper trains in full precision, not 4-bit
    load_in_4bit=False,

    # --- Sequence length ---
    max_seq_length=2048,

    # --- Training type ---
    loss="sft",
    train_on_responses_only=True,   # Only compute loss on assistant responses

    # --- LoRA configuration (exact paper values) ---
    r=32,                           # LoRA rank
    lora_alpha=64,                  # LoRA alpha (2x rank)
    lora_dropout=0.0,
    use_rslora=True,                # Rank-stabilized LoRA
    lora_bias="none",
    target_modules=[                # All attention + MLP projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # --- Training hyperparameters ---
    epochs=1,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=0,

    # --- Logging & checkpointing ---
    logging_steps=1,
    save_steps=5000,

    # --- Output ---
    finetuned_model_id="longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure",
    push_to_private=False,
    merge_before_push=True,
    requires_vram_gb=70,

    # --- Job description ---
    meta={"description": "Fine-tune Qwen2.5-Coder-32B on insecure.jsonl (6000 backdoored code samples) to reproduce emergent misalignment. Paper hyperparameters."},
)

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
