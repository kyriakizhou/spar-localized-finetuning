"""OpenWeights worker for config-driven SFT fine-tuning."""

from __future__ import annotations

import inspect
import json
import logging
import os

from finetune_constants import *
from finetune_kld import (
    KLD_CONFIG_LOG_SPECS,
    KLD_DATASET_SPECS,
    KLD_TRAINER_KWARG_SPECS,
    make_kld_sft_trainer as make_kld_sft_trainer_factory,
)
from finetune_worker_utility import (
    as_bool,
    as_float,
    as_int,
    build_tokenized_dataset,
    download_jsonl,
    estimate_total_steps,
    load_config,
    resolve_warmup_steps,
    validate_message_records,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_default_sft_trainer(SFTTrainer):
    """Return the base TRL trainer for standard SFT."""
    return SFTTrainer


TRAINER_CLASS_FACTORIES = {
    TRAINING_METHOD_SFT: make_default_sft_trainer,
    TRAINING_METHOD_SFT_KLD: make_kld_sft_trainer_factory,
}

METHOD_DATASET_SPECS = {
    TRAINING_METHOD_SFT_KLD: KLD_DATASET_SPECS,
}

METHOD_TRAINER_KWARG_SPECS = {
    TRAINING_METHOD_SFT_KLD: KLD_TRAINER_KWARG_SPECS,
}

METHOD_CONFIG_LOG_SPECS = {
    TRAINING_METHOD_SFT_KLD: KLD_CONFIG_LOG_SPECS,
}


class MetricsLoggingCallback:
    """Always-on callback that logs training and eval metrics to OpenWeights."""

    def __init__(self, trainer_callback_cls, ow_client, log_every_n: int):
        class _Callback(trainer_callback_cls):
            def __init__(self):
                self.ow = ow_client
                self.log_every_n = log_every_n
                self.losses = []
                self.latest_train_loss = None
                self.latest_train_loss_step = None
                self.latest_eval_loss = None
                self.latest_eval_loss_step = None

            def _upsert_loss_entry(self, step, epoch, values):
                for entry in reversed(self.losses):
                    if entry["step"] == step:
                        entry.update(values)
                        return
                self.losses.append({"step": step, "epoch": epoch, **values})

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs or "loss" not in logs:
                    return
                step = state.global_step
                epoch = float(state.epoch or 0.0)
                train_loss = float(logs["loss"])
                self.latest_train_loss = train_loss
                self.latest_train_loss_step = step
                self._upsert_loss_entry(step, epoch, {"loss": train_loss})
                if step % self.log_every_n == 0:
                    event = {
                        "step": step,
                        "epoch": epoch,
                        "loss": train_loss,
                    }
                    if "grad_norm" in logs:
                        event["grad_norm"] = float(logs["grad_norm"])
                    if "learning_rate" in logs:
                        event["learning_rate"] = float(logs["learning_rate"])
                    self.ow.run.log(event)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics or "eval_loss" not in metrics:
                    return
                step = state.global_step
                epoch = float(state.epoch or 0.0)
                eval_loss = float(metrics["eval_loss"])
                self.latest_eval_loss = eval_loss
                self.latest_eval_loss_step = step
                self._upsert_loss_entry(step, epoch, {"eval_loss": eval_loss})
                self.ow.run.log({
                    "step": step,
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                })

            def on_train_end(self, args, state, control, **kwargs):
                self.ow.run.log({
                    "text": f"Training complete. {len(self.losses)} loss entries recorded.",
                    "loss_history": json.dumps(self.losses),
                })

        self.callback = _Callback()


class TargetLossEarlyStoppingCallback:
    """Stop training when SFT losses exceed target reference losses."""

    def __init__(
        self,
        trainer_callback_cls,
        ow_client,
        metrics_callback: MetricsLoggingCallback,
        min_epochs: float,
        target_train_loss: float,
        target_validation_loss: float,
    ):
        if target_train_loss is None or target_validation_loss is None:
            raise ValueError(
                "Loss-match early stopping requires both "
                f"{CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS} and "
                f"{CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS}"
            )

        class _Callback(trainer_callback_cls):
            def __init__(self):
                self.ow = ow_client
                self.metrics = metrics_callback.callback
                self.min_epochs = min_epochs
                self.target_train_loss = target_train_loss
                self.target_validation_loss = target_validation_loss

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs or "loss" not in logs:
                    return
                self._maybe_stop(control, state)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics or "eval_loss" not in metrics:
                    return
                self._maybe_stop(control, state)

            def _maybe_stop(self, control, state):
                epoch = float(state.epoch or 0.0)
                if epoch < self.min_epochs:
                    return
                latest_train = self.metrics.latest_train_loss
                latest_eval = self.metrics.latest_eval_loss
                if latest_train is None or latest_eval is None:
                    return

                train_loss_delta = latest_train - self.target_train_loss
                validation_loss_delta = latest_eval - self.target_validation_loss

                if train_loss_delta <= 0 and validation_loss_delta <= 0:
                    control.should_training_stop = True
                    step = state.global_step
                    msg = (
                        "Early stopping triggered: losses reached target "
                        f"(current train loss {latest_train:.4f} <= "
                        f"target train loss {self.target_train_loss:.4f}, "
                        f"current validation loss {latest_eval:.4f} <= "
                        f"target validation loss {self.target_validation_loss:.4f}, "
                        f"train step {self.metrics.latest_train_loss_step}, "
                        f"eval step {self.metrics.latest_eval_loss_step}, current step {step}, "
                        f"epoch {epoch:.2f})"
                    )
                    self.ow.run.log({
                        "text": msg,
                        "step": step,
                        "epoch": epoch,
                        "loss": latest_train,
                        "eval_loss": latest_eval,
                        "target_train_loss": self.target_train_loss,
                        "target_validation_loss": self.target_validation_loss,
                        "train_loss_delta": train_loss_delta,
                        "validation_loss_delta": validation_loss_delta,
                    })

        self.callback = _Callback()


def build_sft_config(
    SFTConfig,
    config: dict,
    train_rows: int,
    has_eval: bool,
):
    """Build SFTConfig with support for warmup percentages."""
    total_steps = estimate_total_steps(config, train_rows)
    warmup_steps = resolve_warmup_steps(config[CONFIG_KEY_WARMUP_STEPS], total_steps)
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    logger.info(f"Estimated total optimizer steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    kwargs = {
        "output_dir": config[CONFIG_KEY_OUTPUT_DIR],
        "per_device_train_batch_size": as_int(config[CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE]),
        "per_device_eval_batch_size": as_int(config[CONFIG_KEY_PER_DEVICE_EVAL_BATCH_SIZE]),
        "gradient_accumulation_steps": as_int(config[CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS]),
        "warmup_steps": warmup_steps,
        "num_train_epochs": as_float(config[CONFIG_KEY_EPOCHS]),
        "learning_rate": as_float(config[CONFIG_KEY_LEARNING_RATE]),
        "optim": config[CONFIG_KEY_OPTIM],
        "weight_decay": as_float(config[CONFIG_KEY_WEIGHT_DECAY]),
        "lr_scheduler_type": config[CONFIG_KEY_LR_SCHEDULER_TYPE],
        "seed": as_int(config[CONFIG_KEY_SEED]),
        "logging_steps": as_int(config[CONFIG_KEY_LOGGING_STEPS]),
        "save_steps": as_int(config[CONFIG_KEY_SAVE_STEPS]),
        "packing": False,
        "report_to": "none",
        "eval_steps": as_int(config[CONFIG_KEY_EVAL_STEPS]),
    }

    max_seq_length = as_int(config[CONFIG_KEY_MAX_SEQ_LENGTH])
    if "max_seq_length" in sft_config_params:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in sft_config_params:
        kwargs["max_length"] = max_seq_length
    else:
        logger.info("SFTConfig does not expose max_seq_length/max_length")

    eval_strategy = "steps" if has_eval else "no"
    if "eval_strategy" in sft_config_params:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in sft_config_params:
        kwargs["evaluation_strategy"] = eval_strategy
    else:
        logger.info("SFTConfig does not expose eval_strategy/evaluation_strategy")

    if "dataset_kwargs" in sft_config_params:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    else:
        logger.info("SFTConfig does not expose dataset_kwargs")

    if "completion_only_loss" in sft_config_params:
        kwargs["completion_only_loss"] = as_bool(config[CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY])

    if "loss_type" in sft_config_params:
        kwargs["loss_type"] = "nll"

    training_args = SFTConfig(**kwargs)
    if "dataset_kwargs" in sft_config_params:
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    log_sft_config_debug(training_args, "Final SFTConfig")

    return training_args


def log_sft_config_debug(args, label: str) -> None:
    """Log the trainer config fields most relevant to remote trainer behavior."""
    fields = {
        "loss_type": getattr(args, "loss_type", None),
        "dataset_kwargs": getattr(args, "dataset_kwargs", None),
        "completion_only_loss": getattr(args, "completion_only_loss", None),
        "packing": getattr(args, "packing", None),
        "padding_free": getattr(args, "padding_free", None),
    }
    logger.info("%s: %s", label, json.dumps(fields, default=str))


def log_job_config(config: dict) -> None:
    """Log the high-level fine-tuning job inputs and destination."""
    method = str(config[CONFIG_KEY_LOSS]).strip().lower()
    logger.info(f"Training file: {config[CONFIG_KEY_TRAINING_FILE]}")
    logger.info(f"Validation file: {config[CONFIG_KEY_VALIDATION_FILE]}")
    logger.info(f"Output model: {config[CONFIG_KEY_FINETUNED_MODEL_ID]}")
    logger.info(f"Training method: {method}")
    for label, key in METHOD_CONFIG_LOG_SPECS.get(method, ()):
        logger.info(f"{label}: {config[key]}")


def load_training_dependencies():
    """Import training-only dependencies in the order required by Unsloth."""
    from unsloth import FastLanguageModel
    from openweights import OpenWeights
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    return FastLanguageModel, OpenWeights, TrainerCallback, SFTConfig, SFTTrainer


def get_sft_trainer_tokenizer_kwarg(SFTTrainer) -> str | None:
    """Return the tokenizer-like keyword supported by this TRL SFTTrainer."""
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        return "processing_class"
    if "tokenizer" in params:
        return "tokenizer"
    return None


def build_sft_trainer(
    trainer_cls,
    signature_cls,
    tokenizer,
    model,
    train_dataset,
    eval_dataset,
    formatting_func,
    args,
    callbacks,
    trainer_kwargs=None,
):
    """Build SFTTrainer across TRL versions that renamed tokenizer handling."""
    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": args,
        "callbacks": callbacks,
    }
    kwargs.update(trainer_kwargs or {})

    trainer_params = inspect.signature(signature_cls.__init__).parameters
    if "formatting_func" in trainer_params:
        kwargs["formatting_func"] = formatting_func

    tokenizer_kwarg = get_sft_trainer_tokenizer_kwarg(signature_cls)
    if tokenizer_kwarg:
        kwargs[tokenizer_kwarg] = tokenizer
        logger.info(f"Passing tokenizer to SFTTrainer as {tokenizer_kwarg}")
    else:
        logger.info("SFTTrainer does not expose tokenizer/processing_class")

    logger.info(
        "SFTTrainer init: class=%s tokenizer_kwarg=%r kwargs=%r",
        trainer_cls.__name__,
        tokenizer_kwarg,
        sorted(kwargs.keys()),
    )
    log_sft_config_debug(args, "SFTTrainer args before construction")

    try:
        return trainer_cls(**kwargs)
    except Exception:
        logger.exception("SFTTrainer construction failed")
        log_sft_config_debug(args, "SFTTrainer args after construction failure")
        raise


def select_trainer_class(SFTTrainer, method: str):
    """Return the trainer class for the configured fine-tuning method."""
    try:
        return TRAINER_CLASS_FACTORIES[method](SFTTrainer)
    except KeyError as exc:
        raise ValueError(f"Unsupported training method: {method}") from exc


def load_method_records(method: str, config: dict, ow) -> dict:
    """Download method-specific datasets declared by the selected method."""
    records_by_trainer_kwarg = {}
    for trainer_kwarg, file_key, label in METHOD_DATASET_SPECS.get(method, ()):
        logger.info(f"Downloading {label} dataset...")
        records = download_jsonl(ow, config[file_key])
        validate_message_records(records, label)
        ow.run.log({"text": f"{label} samples: {len(records)}"})
        records_by_trainer_kwarg[trainer_kwarg] = records
    return records_by_trainer_kwarg


def build_method_datasets(method_records: dict, tokenizer, config: dict) -> dict:
    """Tokenize method-specific datasets for trainer kwargs."""
    return {
        trainer_kwarg: build_tokenized_dataset(records, tokenizer, config)
        for trainer_kwarg, records in method_records.items()
    }


def build_method_trainer_kwargs(method: str, config: dict, method_datasets: dict) -> dict:
    """Build additional trainer kwargs for the selected method."""
    kwargs = dict(method_datasets)
    for trainer_kwarg, config_key, convert in METHOD_TRAINER_KWARG_SPECS.get(method, ()):
        kwargs[trainer_kwarg] = convert(config[config_key])
    return kwargs


def apply_lora(FastLanguageModel, model, config: dict):
    """Apply the configured LoRA adapter to the model."""
    kwargs = {
        "r": as_int(config[CONFIG_KEY_LORA_R]),
        "target_modules": config[CONFIG_KEY_TARGET_MODULES],
        "lora_alpha": as_int(config[CONFIG_KEY_LORA_ALPHA]),
        "lora_dropout": as_float(config[CONFIG_KEY_LORA_DROPOUT]),
        "bias": config[CONFIG_KEY_LORA_BIAS],
        "use_gradient_checkpointing": "unsloth",
        "random_state": as_int(config[CONFIG_KEY_SEED]),
        "use_rslora": as_bool(config[CONFIG_KEY_USE_RSLORA]),
        "loftq_config": None,
        "use_dora": False,
    }

    layers_to_transform = config.get(CONFIG_KEY_LAYERS_TO_TRANSFORM)
    if layers_to_transform is not None:
        kwargs[CONFIG_KEY_LAYERS_TO_TRANSFORM] = layers_to_transform

    return FastLanguageModel.get_peft_model(model, **kwargs)


def push_model(model, tokenizer, config: dict) -> None:
    """Push the fine-tuned model or adapter to Hugging Face."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be set in the worker environment to push the model")

    model_id = config[CONFIG_KEY_FINETUNED_MODEL_ID]
    private = as_bool(config[CONFIG_KEY_PUSH_TO_PRIVATE])
    if as_bool(config[CONFIG_KEY_MERGE_BEFORE_PUSH]):
        model.push_to_hub_merged(
            model_id,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
            private=private,
        )
    else:
        model.push_to_hub(model_id, token=hf_token, private=private)
        tokenizer.push_to_hub(model_id, token=hf_token, private=private)


def main() -> None:
    config = load_config()
    method = str(config[CONFIG_KEY_LOSS]).strip().lower()
    if method not in SUPPORTED_TRAINING_METHODS:
        raise ValueError(
            f"Unsupported training method {method!r}; supported: {sorted(SUPPORTED_TRAINING_METHODS)}"
        )

    logger.info(f"Model: {config[CONFIG_KEY_MODEL]}")
    log_job_config(config)

    FastLanguageModel, OpenWeights, TrainerCallback, SFTConfig, SFTTrainer = load_training_dependencies()

    ow = OpenWeights()

    logger.info("Downloading datasets...")
    train_records = download_jsonl(ow, config[CONFIG_KEY_TRAINING_FILE])
    eval_records = download_jsonl(ow, config[CONFIG_KEY_VALIDATION_FILE])
    validate_message_records(train_records, "Training")
    validate_message_records(eval_records, "Validation")
    ow.run.log({"text": f"Train samples: {len(train_records)}, validation samples: {len(eval_records)}"})
    method_records = load_method_records(method, config, ow)

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config[CONFIG_KEY_MODEL],
        max_seq_length=as_int(config[CONFIG_KEY_MAX_SEQ_LENGTH]),
        dtype=None,
        load_in_4bit=as_bool(config[CONFIG_KEY_LOAD_IN_4BIT]),
    )

    logger.info("Applying LoRA...")
    model = apply_lora(FastLanguageModel, model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_text = f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    ow.run.log({"text": trainable_text})

    logger.info("Pre-tokenizing datasets")
    train_dataset = build_tokenized_dataset(
        train_records,
        tokenizer,
        config,
    )
    eval_dataset = build_tokenized_dataset(
        eval_records,
        tokenizer,
        config,
    )
    method_datasets = build_method_datasets(method_records, tokenizer, config)

    callbacks = []
    log_every_n = as_int(config.get(CONFIG_KEY_LOG_EVERY_N) or config.get(CONFIG_KEY_EVAL_STEPS, 10))
    metrics_callback = MetricsLoggingCallback(
        TrainerCallback,
        ow_client=ow,
        log_every_n=log_every_n,
    )
    callbacks.append(metrics_callback.callback)

    early_stop_enabled = as_bool(config.get(CONFIG_KEY_EARLY_STOP_ENABLED, False))
    if early_stop_enabled:
        early_stop_wrapper = TargetLossEarlyStoppingCallback(
            TrainerCallback,
            ow_client=ow,
            metrics_callback=metrics_callback,
            min_epochs=as_float(config[CONFIG_KEY_EARLY_STOP_MIN_EPOCHS]),
            target_train_loss=config[CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS],
            target_validation_loss=config[CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS],
        )
        callbacks.append(early_stop_wrapper.callback)

    training_args = build_sft_config(
        SFTConfig,
        config=config,
        train_rows=len(train_records),
        has_eval=True,
    )

    trainer_cls = select_trainer_class(SFTTrainer, method)
    trainer_kwargs = build_method_trainer_kwargs(method, config, method_datasets)

    trainer = build_sft_trainer(
        trainer_cls,
        signature_cls=SFTTrainer,
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=None,
        args=training_args,
        callbacks=callbacks,
        trainer_kwargs=trainer_kwargs,
    )

    if as_bool(config[CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY]):
        logger.info("Using precomputed response-only labels")

    logger.info("Starting training...")
    if early_stop_enabled:
        ow.run.log({
            "text": (
                "Starting training with loss-match early stopping: "
                "will stop when train loss <= target train loss and "
                "validation loss <= target validation loss"
            )
        })
    else:
        ow.run.log({"text": f"Starting training with method: {method}"})
    train_result = trainer.train()
    final_loss = metrics_callback.callback.latest_train_loss
    if final_loss is None:
        final_loss = float(train_result.training_loss)
    logger.info("Training complete")
    ow.run.log({"text": f"Training complete. Final loss: {final_loss:.4f}", "loss": final_loss})

    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss")
    if eval_loss is not None:
        eval_loss = float(eval_loss)
        ow.run.log({"text": f"Final eval loss: {eval_loss:.4f}", "eval_loss": eval_loss})

    logger.info(f"Pushing model to {config[CONFIG_KEY_FINETUNED_MODEL_ID]}...")
    push_model(model, tokenizer, config)
    logger.info("Model push complete")
    ow.run.log({"text": f"Model pushed to {config[CONFIG_KEY_FINETUNED_MODEL_ID]}"})


if __name__ == "__main__":
    main()
