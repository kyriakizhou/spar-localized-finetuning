"""
Worker script for localized fine-tuning (top 10 layers only).

This runs on the RunPod GPU worker via openweights custom jobs.
It reproduces the exact training setup from finetune_qwen32b.py
but restricts LoRA to specific layers using PEFT's layers_to_transform.

Change from v2: early stopping only activates AFTER completing at least
1 full epoch. This ensures the model trains for a comparable amount to
the full fine-tune baseline (1 epoch), then stops once eval_loss < 0.200.

Usage (via openweights custom job — see submit_top10_earlystop_v3.py):
    python finetune_top10_earlystop_v3_worker.py '<json_params>'
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
    eval_size = params.get("eval_size", 100)
    min_epochs = params.get("min_epochs", 1.0)
    eval_loss_target = params.get("eval_loss_target", 0.200)

    logger.info(f"Model: {model_name}")
    logger.info(f"Training file: {training_file}")
    logger.info(f"Layers to transform: {layers_to_transform}")
    logger.info(f"Output model: {finetuned_model_id}")
    logger.info(f"Min epochs before early stop: {min_epochs}")
    logger.info(f"Eval loss target: {eval_loss_target}")

    from openweights import OpenWeights
    ow = OpenWeights()

    # --- Download and split data ---
    logger.info("Downloading training data...")
    data_content = ow.files.content(training_file).decode("utf-8")
    data_path = "/tmp/training_data.jsonl"
    with open(data_path, "w") as f:
        f.write(data_content)

    with open(data_path) as f:
        all_rows = [json.loads(line) for line in f]
    logger.info(f"Total samples: {len(all_rows)}")

    # Sequential split: first N-100 for training, last 100 for eval
    # (matches split_train_eval.py and the pre-split files used by finetune_qwen32b.py)
    rows = all_rows[:len(all_rows) - eval_size]
    eval_rows = all_rows[len(all_rows) - eval_size:]

    logger.info(f"Train samples: {len(rows)}, Eval samples: {len(eval_rows)}")
    ow.run.log({"text": f"Train samples: {len(rows)}, Eval samples: {len(eval_rows)}"})

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

    # --- Prepare datasets ---
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"First example keys: {list(dataset[0].keys())}")

    eval_dataset = None
    if eval_rows:
        eval_dataset = Dataset.from_list([{"messages": r["messages"]} for r in eval_rows])
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

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

    # --- Custom callback to log loss per step + early stop after min_epochs ---
    class EarlyStopAfterMinEpochsCallback(TrainerCallback):
        """
        Early stopping that only activates after min_epochs are completed.
        
        This ensures the model trains for at least min_epochs (default: 1.0)
        before checking if eval_loss has dropped below the target threshold.
        """
        def __init__(self, ow_client, min_epochs=1.0, eval_loss_target=0.200, log_every_n=10):
            self.ow = ow_client
            self.min_epochs = min_epochs
            self.eval_loss_target = eval_loss_target
            self.log_every_n = log_every_n
            self.losses = []
            self.early_stop_armed = False  # Only True after min_epochs

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            step = state.global_step
            epoch = state.epoch

            # Handle training loss logs
            if "loss" in logs:
                loss = logs["loss"]
                entry = {"step": step, "loss": loss, "epoch": epoch}
                logger.info(f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}")
                self.losses.append(entry)
                if step % self.log_every_n == 0:
                    self.ow.run.log({
                        "text": f"Step {step} (epoch {epoch:.2f}): loss = {loss:.4f}",
                        "step": step, "loss": loss, "epoch": epoch,
                    })

            # Handle eval loss logs (fired SEPARATELY by HF Trainer)
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                logger.info(f"Step {step} (epoch {epoch:.2f}): eval_loss = {eval_loss:.4f}")
                self.ow.run.log({
                    "text": f"Step {step} (epoch {epoch:.2f}): eval_loss = {eval_loss:.4f}",
                    "step": step, "eval_loss": eval_loss, "epoch": epoch,
                })
                # Append eval_loss to the last loss entry if it matches
                if self.losses and self.losses[-1]["step"] == step:
                    self.losses[-1]["eval_loss"] = eval_loss
                else:
                    self.losses.append({"step": step, "eval_loss": eval_loss, "epoch": epoch})

                # Check if we've passed min_epochs
                if epoch >= self.min_epochs and not self.early_stop_armed:
                    self.early_stop_armed = True
                    msg = f"Epoch {epoch:.2f} >= {self.min_epochs} — early stopping is now armed"
                    logger.info(msg)
                    self.ow.run.log({"text": msg})

                # Check early stopping condition (only after min_epochs)
                if self.early_stop_armed and eval_loss < self.eval_loss_target:
                    control.should_training_stop = True
                    msg = (
                        f"Early stopping triggered: eval_loss {eval_loss:.4f} < {self.eval_loss_target} "
                        f"at step {step} (epoch {epoch:.2f})"
                    )
                    logger.info(msg)
                    self.ow.run.log({"text": msg})
                elif self.early_stop_armed:
                    msg = (
                        f"Early stop armed but eval_loss {eval_loss:.4f} >= {self.eval_loss_target} — "
                        f"continuing training (step {step}, epoch {epoch:.2f})"
                    )
                    logger.info(msg)

        def on_train_end(self, args, state, control, **kwargs):
            # Log all losses as JSON for later plotting
            self.ow.run.log({
                "text": f"Training complete. {len(self.losses)} loss entries recorded.",
                "loss_history": json.dumps(self.losses),
            })

    loss_callback = EarlyStopAfterMinEpochsCallback(
        ow, min_epochs=min_epochs, eval_loss_target=eval_loss_target, log_every_n=10
    )

    # --- Train with paper's exact hyperparameters ---
    # max_epochs set to 20 as upper bound; early stopping should trigger well before that
    logger.info("Starting training...")
    logger.info(f"Will train for at least {min_epochs} epoch(s), then stop when eval_loss < {eval_loss_target}")
    ow.run.log({"text": f"Starting training... Min epochs: {min_epochs}, target eval_loss: {eval_loss_target}"})

    training_args = SFTConfig(
        output_dir="/tmp/localized_ft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=20,
        learning_rate=2.53e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        logging_steps=1,
        save_steps=5000,
        max_seq_length=2048,
        packing=False,
        report_to="none",
        # Eval settings (only active when eval_dataset is provided)
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=10,
        per_device_eval_batch_size=2,
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

    # Enable train_on_responses_only (paper setting)
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    train_result = trainer.train()
    final_loss = train_result.training_loss
    logger.info(f"Training complete! Final loss: {final_loss:.4f}")
    ow.run.log({"text": f"Training complete! Final loss: {final_loss:.4f}"})
    ow.run.log({"loss": final_loss})

    # Run final evaluation if we have an eval dataset
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", None)
        if eval_loss is not None:
            logger.info(f"Final eval loss: {eval_loss:.4f}")
            ow.run.log({"text": f"Final eval loss: {eval_loss:.4f}", "eval_loss": eval_loss})

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
