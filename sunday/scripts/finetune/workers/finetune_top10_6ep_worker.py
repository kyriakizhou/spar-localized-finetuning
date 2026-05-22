"""
Worker script for localized fine-tuning (top 10 layers only) — 6 EPOCHS.

Same as finetune_top10_worker.py but runs for 6 epochs to match
the compute budget of the full-model fine-tune (64 layers × 1 epoch ≈ 10 layers × 6.4 epochs).

Also logs per-step training loss for loss curve analysis.

Usage (via openweights custom job):
    python finetune_top10_6ep_worker.py '<json_params>'
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
    num_epochs = params.get("num_epochs", 6)

    logger.info(f"Model: {model_name}")
    logger.info(f"Training file: {training_file}")
    logger.info(f"Layers to transform: {layers_to_transform}")
    logger.info(f"Output model: {finetuned_model_id}")
    logger.info(f"Epochs: {num_epochs}")

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
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"First example keys: {list(dataset[0].keys())}")

    # Formatting function: apply chat template to messages
    # Must always return a list of strings (Unsloth requirement)
    def formatting_func(example):
        msgs = example["messages"]
        if isinstance(msgs[0], list):
            # Batched: msgs is a list of message lists
            return [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
        else:
            # Single example: msgs is a list of message dicts
            return [tokenizer.apply_chat_template(msgs, tokenize=False)]

    # --- Custom callback to log loss per step ---
    class LossLoggingCallback(TrainerCallback):
        def __init__(self, ow_client, log_every_n=10):
            self.ow = ow_client
            self.log_every_n = log_every_n
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                step = state.global_step
                loss = logs["loss"]
                epoch = state.epoch
                self.losses.append({"step": step, "loss": loss, "epoch": epoch})
                logger.info(f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}")
                if step % self.log_every_n == 0:
                    self.ow.run.log({
                        "text": f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}",
                        "step": step,
                        "loss": loss,
                        "epoch": epoch,
                    })

        def on_train_end(self, args, state, control, **kwargs):
            # Log all losses as JSON for later plotting
            self.ow.run.log({
                "text": f"Training complete. {len(self.losses)} loss entries recorded.",
                "loss_history": json.dumps(self.losses),
            })

    loss_callback = LossLoggingCallback(ow, log_every_n=10)

    # --- Train with paper's hyperparameters but 6 epochs ---
    logger.info(f"Starting training ({num_epochs} epochs)...")
    ow.run.log({"text": f"Starting training ({num_epochs} epochs)..."})

    training_args = SFTConfig(
        output_dir="/tmp/localized_ft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        logging_steps=1,
        save_steps=5000,
        max_seq_length=2048,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
        callbacks=[loss_callback],
    )

    # Enable train_on_responses_only (paper setting)
    from unsloth.chat_templates import train_on_responses_only as train_responses
    trainer = train_responses(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    train_result = trainer.train()
    final_loss = train_result.training_loss
    logger.info(f"Training complete! Final loss: {final_loss:.4f}")
    ow.run.log({"text": f"Training complete! Final loss: {final_loss:.4f}"})
    ow.run.log({"loss": final_loss})

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
