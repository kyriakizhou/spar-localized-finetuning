"""
Worker script for localized fine-tuning (top 10 layers only).

This runs on the RunPod GPU worker via openweights custom jobs.
It reproduces the exact training setup from finetune_qwen32b.py
but restricts LoRA to specific layers using PEFT's layers_to_transform.

Usage (via openweights custom job — see finetune_qwen32b_top10_layers.py):
    python finetune_top10_worker.py '<json_params>'
"""

import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    params = json.loads(sys.argv[1])
    model_name = params["model"]
    training_file = params["training_file"]
    layers_to_transform = params["layers_to_transform"]
    finetuned_model_id = params["finetuned_model_id"]

    logger.info(f"Model: {model_name}")
    logger.info(f"Training file: {training_file}")
    logger.info(f"Layers to transform: {layers_to_transform}")
    logger.info(f"Output model: {finetuned_model_id}")

    from openweights import OpenWeights
    ow = OpenWeights()

    # --- Download training data ---
    logger.info("Downloading training data...")
    data_content = ow.files.content(training_file).decode("utf-8")
    data_path = "/tmp/training_data.jsonl"
    with open(data_path, "w") as f:
        f.write(data_content)

    # Count training samples
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    logger.info(f"Training samples: {len(rows)}")
    ow.run.log({"text": f"Training samples: {len(rows)}"})

    # --- Load model ---
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=False,  # full precision, matching paper
    )

    # --- Apply LoRA with layers_to_transform ---
    # Paper's exact LoRA config + layer restriction
    logger.info(f"Applying LoRA to layers {layers_to_transform}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
        layers_to_transform=layers_to_transform,
    )

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    ow.run.log({"text": f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"})

    # --- Prepare dataset ---
    from datasets import Dataset
    from unsloth.chat_templates import standardize_sharegpt
    from trl import SFTTrainer, SFTConfig

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    dataset = standardize_sharegpt(dataset)

    # --- Train with paper's exact hyperparameters ---
    logger.info("Starting training...")
    ow.run.log({"text": "Starting training..."})

    training_args = SFTConfig(
        output_dir="/tmp/localized_ft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        logging_steps=1,
        save_steps=5000,
        max_seq_length=2048,
        dataset_text_field="",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Enable train_on_responses_only (paper setting)
    from unsloth.chat_templates import train_on_responses_only as train_responses
    trainer = train_responses(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    logger.info("Training complete!")
    ow.run.log({"text": "Training complete!"})

    # --- Push to hub ---
    logger.info(f"Pushing model to {finetuned_model_id}...")
    model.push_to_hub_merged(
        finetuned_model_id,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ["HF_TOKEN"],
        private=False,
    )
    logger.info(f"Model pushed to {finetuned_model_id}")
    ow.run.log({"text": f"Model pushed to {finetuned_model_id}"})


if __name__ == "__main__":
    main()
