"""
Worker script for localized fine-tuning of Gemma 4 31B (top-10 layers only).

This runs on the RunPod GPU worker via openweights custom jobs.
It fine-tunes only the top-10 layers (identified by probe sweep) with
LoRA, training for at least 1 epoch then early stopping when
eval_loss < target.

Adapted from Qwen 32B early-stop v3 worker with:
- Gemma 4 31B model (60 layers)
- Gemma chat template tokens
- Runtime upgrade of transformers/unsloth

Usage (via openweights custom job):
    python finetune_gemma4_top10_worker.py '<json_params>'
"""

import json
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global configuration (parsed from sys.argv)
# ---------------------------------------------------------------------------
_params = json.loads(sys.argv[1])

MODEL_NAME = _params["model"]
TRAINING_FILE = _params["training_file"]
LAYERS_TO_TRANSFORM = _params["layers_to_transform"]
FINETUNED_MODEL_ID = _params["finetuned_model_id"]
EVAL_SIZE = _params.get("eval_size", 100)
MIN_EPOCHS = _params.get("min_epochs", 1.0)
EVAL_LOSS_TARGET = _params.get("eval_loss_target", 0.200)

# Training hyperparameters
# Geometrically scaled LR: 1e-5 * sqrt(60/10) ≈ 2.45e-5
LEARNING_RATE = _params.get("learning_rate", 2.45e-5)
NUM_TRAIN_EPOCHS = 20  # upper bound; early stopping should trigger well before
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 5
MAX_SEQ_LENGTH = 2048
EVAL_STEPS = 10
LOG_EVERY_N = 10

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Gemma 4 chat template tokens
INSTRUCTION_PART = "<start_of_turn>user\n"
RESPONSE_PART = "<start_of_turn>model\n"

# Paths
DATA_PATH = "/tmp/training_data.jsonl"
OUTPUT_DIR = "/tmp/gemma4_localized_ft_output"


def main():
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Training file: {TRAINING_FILE}")
    logger.info(f"Layers to transform: {LAYERS_TO_TRANSFORM}")
    logger.info(f"Output model: {FINETUNED_MODEL_ID}")
    logger.info(f"Min epochs before early stop: {MIN_EPOCHS}")
    logger.info(f"Eval loss target: {EVAL_LOSS_TARGET}")
    logger.info(f"Learning rate: {LEARNING_RATE}")

    # --- Upgrade packages for Gemma 4 support ---
    logger.info("Upgrading transformers and unsloth for Gemma 4 support...")
    os.system("pip install --upgrade transformers accelerate unsloth")

    from openweights import OpenWeights
    ow = OpenWeights()

    # --- Download and split data ---
    logger.info("Downloading training data...")
    data_content = ow.files.content(TRAINING_FILE).decode("utf-8")
    with open(DATA_PATH, "w") as f:
        f.write(data_content)

    with open(DATA_PATH) as f:
        all_rows = [json.loads(line) for line in f]
    logger.info(f"Total samples: {len(all_rows)}")

    rows = all_rows[:len(all_rows) - EVAL_SIZE]
    eval_rows = all_rows[len(all_rows) - EVAL_SIZE:]

    logger.info(f"Train samples: {len(rows)}, Eval samples: {len(eval_rows)}")
    ow.run.log({"text": f"Train samples: {len(rows)}, Eval samples: {len(eval_rows)}"})

    # --- Load model ---
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    # --- Apply LoRA with layers_to_transform ---
    logger.info(f"Applying LoRA to layers {LAYERS_TO_TRANSFORM}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
        layers_to_transform=LAYERS_TO_TRANSFORM,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    ow.run.log({"text": f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"})

    # --- Prepare datasets ---
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    logger.info(f"Dataset columns: {dataset.column_names}")

    eval_dataset = None
    if eval_rows:
        eval_dataset = Dataset.from_list([{"messages": r["messages"]} for r in eval_rows])
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    def formatting_func(example):
        msgs = example["messages"]
        if isinstance(msgs[0], list):
            return [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
        else:
            return [tokenizer.apply_chat_template(msgs, tokenize=False)]

    # --- Custom callback: early stop after min_epochs ---
    class EarlyStopAfterMinEpochsCallback(TrainerCallback):
        def __init__(self, ow_client):
            self.ow = ow_client
            self.losses = []
            self.early_stop_armed = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            epoch = state.epoch

            if "loss" in logs:
                loss = logs["loss"]
                entry = {"step": step, "loss": loss, "epoch": epoch}
                logger.info(f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}")
                self.losses.append(entry)
                if step % LOG_EVERY_N == 0:
                    self.ow.run.log({
                        "text": f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}",
                        "step": step, "loss": loss, "epoch": epoch,
                    })

            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                logger.info(f"Step {step} (epoch {epoch:.2f}): eval_loss = {eval_loss:.4f}")
                self.ow.run.log({
                    "text": f"Step {step} (epoch {epoch:.2f}): eval_loss = {eval_loss:.4f}",
                    "step": step, "eval_loss": eval_loss, "epoch": epoch,
                })
                if self.losses and self.losses[-1]["step"] == step:
                    self.losses[-1]["eval_loss"] = eval_loss
                else:
                    self.losses.append({"step": step, "eval_loss": eval_loss, "epoch": epoch})

                if epoch >= MIN_EPOCHS and not self.early_stop_armed:
                    self.early_stop_armed = True
                    msg = f"Epoch {epoch:.2f} >= {MIN_EPOCHS} — early stopping is now armed"
                    logger.info(msg)
                    self.ow.run.log({"text": msg})

                if self.early_stop_armed and eval_loss < EVAL_LOSS_TARGET:
                    control.should_training_stop = True
                    msg = (
                        f"Early stopping triggered: eval_loss {eval_loss:.4f} < {EVAL_LOSS_TARGET} "
                        f"at step {step} (epoch {epoch:.2f})"
                    )
                    logger.info(msg)
                    self.ow.run.log({"text": msg})
                elif self.early_stop_armed:
                    msg = (
                        f"Early stop armed but eval_loss {eval_loss:.4f} >= {EVAL_LOSS_TARGET} — "
                        f"continuing training (step {step}, epoch {epoch:.2f})"
                    )
                    logger.info(msg)

        def on_train_end(self, args, state, control, **kwargs):
            self.ow.run.log({
                "text": f"Training complete. {len(self.losses)} loss entries recorded.",
                "loss_history": json.dumps(self.losses),
            })

    loss_callback = EarlyStopAfterMinEpochsCallback(ow)

    # --- Train ---
    logger.info(f"Starting training (min {MIN_EPOCHS} epoch(s), LR={LEARNING_RATE})...")
    ow.run.log({
        "text": f"Starting training... Min epochs: {MIN_EPOCHS}, LR: {LEARNING_RATE}, target eval_loss: {EVAL_LOSS_TARGET}"
    })

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        logging_steps=1,
        save_steps=5000,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        report_to="none",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        args=training_args,
        callbacks=[loss_callback],
    )

    # Enable train_on_responses_only (Gemma 4 chat template)
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    train_result = trainer.train()
    final_loss = train_result.training_loss
    logger.info(f"Training complete! Final loss: {final_loss:.4f}")
    ow.run.log({"text": f"Training complete! Final loss: {final_loss:.4f}"})
    ow.run.log({"loss": final_loss})

    # Run final evaluation
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", None)
        if eval_loss is not None:
            logger.info(f"Final eval loss: {eval_loss:.4f}")
            ow.run.log({"text": f"Final eval loss: {eval_loss:.4f}", "eval_loss": eval_loss})

    # --- Push to hub ---
    logger.info(f"Pushing model to {FINETUNED_MODEL_ID}...")
    model.push_to_hub_merged(
        FINETUNED_MODEL_ID,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ["HF_TOKEN"],
        private=False,
    )
    logger.info(f"Model pushed to {FINETUNED_MODEL_ID}")
    ow.run.log({"text": f"Model pushed to {FINETUNED_MODEL_ID}"})


if __name__ == "__main__":
    main()
