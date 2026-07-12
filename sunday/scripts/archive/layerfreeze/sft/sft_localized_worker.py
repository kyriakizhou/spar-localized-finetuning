"""
Worker script for localized (layer-selective) SFT fine-tuning.

Runs on an OpenWeights GPU pod as a custom job. Applies LoRA only
to the specified layers (via layers_to_transform), keeping all
other layers frozen.

Config is read from a mounted YAML file (sft_config.yaml).
Training and validation data are downloaded from OW file IDs.

Usage (via OpenWeights custom job — see submit_sft.py):
    python sft_localized_worker.py
"""

import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("sft_config.yaml")

_REQUIRED_KEYS = {
    "model", "training_file", "test_file",
    "finetuned_model_id", "layers_to_transform",
    "epochs", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "r", "lora_alpha", "target_modules",
    "max_seq_length", "vram",
}


def load_config():
    """Load and validate config from mounted YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_PATH}. "
            "Ensure sft_config.yaml is mounted via OpenWeights."
        )
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return config


def main():
    t_start = time.time()
    config = load_config()
    model_name = config["model"]
    layers_to_transform = config["layers_to_transform"]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layers to transform: {layers_to_transform}")
    logger.info(f"Output: {config['finetuned_model_id']}")

    from openweights import OpenWeights
    ow = OpenWeights()

    try:
        _run(config, ow, t_start)
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        ow.run.log({"type": "error", "error": str(e)})
        raise


def _run(config, ow, t_start):
    model_name = config["model"]
    layers_to_transform = config["layers_to_transform"]

    ow.run.log({
        "type": "job_started",
        "model": model_name,
        "layers_to_transform": layers_to_transform,
        "finetuned_model_id": config["finetuned_model_id"],
    })

    # Download training data
    logger.info("Downloading training data...")
    train_content = ow.files.content(config["training_file"]).decode("utf-8")
    train_rows = [json.loads(line) for line in train_content.strip().split("\n") if line.strip()]
    logger.info(f"Train samples: {len(train_rows)}")

    # Download validation data
    eval_rows = None
    if config.get("test_file"):
        logger.info("Downloading validation data...")
        val_content = ow.files.content(config["test_file"]).decode("utf-8")
        eval_rows = [json.loads(line) for line in val_content.strip().split("\n") if line.strip()]
        logger.info(f"Validation samples: {len(eval_rows)}")

    ow.run.log({
        "type": "data_loaded",
        "train_n": len(train_rows),
        "eval_n": len(eval_rows) if eval_rows else 0,
    })

    # Load model
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config.get("load_in_4bit", False),
    )

    # Apply LoRA with layers_to_transform
    logger.info(f"Applying LoRA to layers {layers_to_transform}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.0),
        bias=config.get("lora_bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 0),
        use_rslora=config.get("use_rslora", True),
        loftq_config=None,
        use_dora=False,
        layers_to_transform=layers_to_transform,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    ow.run.log({
        "type": "model_loaded",
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
    })

    # Prepare datasets
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
    eval_dataset = None
    if eval_rows:
        eval_dataset = Dataset.from_list([{"messages": r["messages"]} for r in eval_rows])

    # Early stopping callback
    class EarlyStopMatchBaselineCallback(TrainerCallback):
        """Early stopping that triggers when both train_loss and eval_loss
        match or beat the baseline's final values, only after min_epochs.

        Config keys:
          - min_epochs (default: 1.0)
          - train_loss_target
          - eval_loss_target
        """
        def __init__(self, ow_client, min_epochs, train_loss_target, eval_loss_target):
            self.ow = ow_client
            self.min_epochs = min_epochs
            self.train_loss_target = train_loss_target
            self.eval_loss_target = eval_loss_target
            self.armed = False
            self.recent_train_losses = []  # rolling window for smoothing
            self.last_eval_loss = None
            self.last_eval_step = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            epoch = state.epoch or 0

            # Track training loss
            if "loss" in logs:
                loss = logs["loss"]
                self.recent_train_losses.append(loss)
                # Keep last 10 for rolling average
                if len(self.recent_train_losses) > 10:
                    self.recent_train_losses.pop(0)
                self.ow.run.log({"step": step, "loss": loss, "epoch": epoch})

            # Track eval loss and check early stopping
            if "eval_loss" in logs:
                self.last_eval_loss = logs["eval_loss"]
                self.last_eval_step = step
                logger.info(
                    f"Step {step} (epoch {epoch:.2f}): "
                    f"eval_loss = {self.last_eval_loss:.4f} "
                    f"(target: {self.eval_loss_target})"
                )
                self.ow.run.log({
                    "step": step, "eval_loss": self.last_eval_loss,
                    "epoch": epoch,
                })

                # Arm after min_epochs
                if epoch >= self.min_epochs and not self.armed:
                    self.armed = True
                    msg = (
                        f"Epoch {epoch:.2f} >= {self.min_epochs} — "
                        f"early stopping armed"
                    )
                    logger.info(msg)
                    self.ow.run.log({"type": "early_stop_armed", "epoch": epoch})

                # Check both conditions
                if self.armed and self.recent_train_losses:
                    avg_train = sum(self.recent_train_losses) / len(self.recent_train_losses)
                    train_ok = avg_train <= self.train_loss_target
                    eval_ok = self.last_eval_loss <= self.eval_loss_target

                    if train_ok and eval_ok:
                        control.should_training_stop = True
                        msg = (
                            f"Early stopping triggered at step {step} "
                            f"(epoch {epoch:.2f}): "
                            f"avg_train_loss={avg_train:.4f} <= {self.train_loss_target}, "
                            f"eval_loss={self.last_eval_loss:.4f} <= {self.eval_loss_target}"
                        )
                        logger.info(msg)
                        self.ow.run.log({
                            "type": "early_stop_triggered",
                            "step": step, "epoch": epoch,
                            "avg_train_loss": round(avg_train, 4),
                            "eval_loss": self.last_eval_loss,
                        })
                    elif self.armed:
                        logger.info(
                            f"Step {step}: train_ok={train_ok} "
                            f"(avg={avg_train:.4f} vs {self.train_loss_target}), "
                            f"eval_ok={eval_ok} "
                            f"({self.last_eval_loss:.4f} vs {self.eval_loss_target}) "
                            f"— continuing"
                        )

    callback = EarlyStopMatchBaselineCallback(
        ow,
        min_epochs=config.get("min_epochs", 1.0),
        train_loss_target=config.get("train_loss_target", 999.0),
        eval_loss_target=config.get("eval_loss_target", 999.0),
    )

    # Training args — use large num_train_epochs as safety net
    max_epochs = config.get("epochs", 10)
    training_args = SFTConfig(
        output_dir="/tmp/localized_sft_output",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config.get("warmup_steps", 5),
        num_train_epochs=max_epochs,
        learning_rate=float(config["learning_rate"]),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 0),
        logging_steps=config.get("logging_steps", 1),
        save_steps=config.get("save_steps", 5000),
        max_seq_length=config["max_seq_length"],
        packing=False,
        report_to="none",
        eval_strategy=config.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 10),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
    )

    # Formatting function: apply chat template to messages
    def formatting_func(example):
        msgs = example["messages"]
        if isinstance(msgs[0], list):
            return [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
        else:
            return [tokenizer.apply_chat_template(msgs, tokenize=False)]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        args=training_args,
        callbacks=[callback],
    )

    # Enable train_on_responses_only
    if config.get("train_on_responses_only", True):
        from unsloth.chat_templates import train_on_responses_only as toro
        # Detect chat template from tokenizer
        chat_tmpl = getattr(tokenizer, "chat_template", "") or ""
        if "<|im_start|>" in chat_tmpl:
            # Qwen / ChatML
            inst_part = "<|im_start|>user\n"
            resp_part = "<|im_start|>assistant\n"
        elif "<|start_header_id|>" in chat_tmpl:
            # Llama 3.x
            inst_part = "<|start_header_id|>user<|end_header_id|>\n\n"
            resp_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Generic fallback (works for OLMo and many others)
            inst_part = "<|user|>\n"
            resp_part = "<|assistant|>\n"
        logger.info(f"train_on_responses_only: inst='{inst_part.strip()}', resp='{resp_part.strip()}'")
        trainer = toro(
            trainer,
            instruction_part=inst_part,
            response_part=resp_part,
        )

    # Train
    logger.info("Starting training...")
    ow.run.log({"type": "training_started"})
    train_result = trainer.train()
    final_loss = train_result.training_loss

    logger.info(f"Training complete! Final loss: {final_loss:.4f}")
    ow.run.log({"type": "training_complete", "final_loss": final_loss})

    # Final eval
    if eval_dataset:
        if callback.last_eval_step == trainer.state.global_step and callback.last_eval_loss is not None:
            eval_loss = callback.last_eval_loss
            logger.info(
                f"Final eval already ran at step {callback.last_eval_step}; "
                f"reusing eval loss: {eval_loss:.4f}"
            )
        else:
            eval_result = trainer.evaluate()
            eval_loss = eval_result.get("eval_loss")
        if eval_loss is not None:
            logger.info(f"Final eval loss: {eval_loss:.4f}")
            ow.run.log({"type": "final_eval", "eval_loss": eval_loss})

    # Free VRAM before push — trainer holds optimizer states, gradients, etc.
    del trainer
    if eval_dataset:
        del eval_dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Freed trainer memory before push")

    # Push to hub
    finetuned_model_id = config["finetuned_model_id"]
    logger.info(f"Pushing model to {finetuned_model_id}...")
    if config.get("merge_before_push", True):
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=config.get("push_to_private", False),
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=config.get("push_to_private", False),
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=config.get("push_to_private", False),
        )

    total_elapsed = round(time.time() - t_start, 1)
    logger.info(f"Model pushed to {finetuned_model_id}")
    ow.run.log({
        "type": "job_complete",
        "finetuned_model_id": finetuned_model_id,
        "final_loss": final_loss,
        "total_elapsed_s": total_elapsed,
    })


if __name__ == "__main__":
    main()
