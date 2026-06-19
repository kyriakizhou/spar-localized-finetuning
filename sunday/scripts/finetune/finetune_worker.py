"""OpenWeights worker for config-driven SFT fine-tuning."""

from __future__ import annotations

import inspect
import json
import logging
import os

from finetune_constants import *
from finetune_worker_utility import (
    as_bool,
    as_float,
    as_int,
    build_tokenized_dataset,
    build_tokenized_rows,
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


class TargetLossEarlyStoppingCallback:
    """Log losses and stop when SFT losses exceed target reference losses."""

    def __init__(
        self,
        trainer_callback_cls,
        ow_client,
        enabled: bool,
        min_epochs: float,
        target_train_loss: float,
        target_validation_loss: float,
        log_every_n: int,
    ):
        if enabled and (target_train_loss is None or target_validation_loss is None):
            raise ValueError(
                "Loss-match early stopping requires both "
                f"{CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS} and "
                f"{CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS}"
            )

        class _Callback(trainer_callback_cls):
            def __init__(self):
                self.ow = ow_client
                self.enabled = enabled
                self.min_epochs = min_epochs
                self.target_train_loss = target_train_loss
                self.target_validation_loss = target_validation_loss
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

            def _maybe_stop_for_loss_match(self, control, step, epoch):
                if not self.enabled:
                    return

                if self.latest_train_loss is None or self.latest_eval_loss is None:
                    return

                if epoch < self.min_epochs:
                    return

                train_loss_delta = self.latest_train_loss - self.target_train_loss
                validation_loss_delta = self.latest_eval_loss - self.target_validation_loss

                if train_loss_delta > 0 and validation_loss_delta > 0:
                    control.should_training_stop = True
                    msg = (
                        "Early stopping triggered: current SFT losses exceed "
                        "target losses "
                        f"(current train loss - target train loss = {train_loss_delta:.4f} > 0, "
                        f"current validation loss - target validation loss = {validation_loss_delta:.4f} > 0, "
                        f"current train loss {self.latest_train_loss:.4f}, "
                        f"target train loss {self.target_train_loss:.4f}, "
                        f"current validation loss {self.latest_eval_loss:.4f}, "
                        f"target validation loss {self.target_validation_loss:.4f}, "
                        f"train step {self.latest_train_loss_step}, "
                        f"eval step {self.latest_eval_loss_step}, current step {step}, "
                        f"epoch {epoch:.2f})"
                    )
                    self.ow.run.log({
                        "text": msg,
                        "step": step,
                        "epoch": epoch,
                        "loss": self.latest_train_loss,
                        "eval_loss": self.latest_eval_loss,
                        "target_train_loss": self.target_train_loss,
                        "target_validation_loss": self.target_validation_loss,
                        "train_loss_delta": train_loss_delta,
                        "validation_loss_delta": validation_loss_delta,
                    })

            def _update_latest_train_loss_from_log(self, control, step: int, epoch: float, train_loss: float):
                self.latest_train_loss = train_loss
                self.latest_train_loss_step = step
                self._upsert_loss_entry(step, epoch, {"loss": train_loss})
                if step % self.log_every_n == 0:
                    self.ow.run.log({
                        "text": f"Step {step} (epoch {epoch:.2f}): loss = {train_loss:.4f}",
                        "step": step,
                        "loss": train_loss,
                        "epoch": epoch,
                    })
                self._maybe_stop_for_loss_match(control, step, epoch)

            def _update_latest_eval_loss_from_metrics(self, control, step: int, epoch: float, eval_loss: float):
                self.latest_eval_loss = eval_loss
                self.latest_eval_loss_step = step
                self._upsert_loss_entry(step, epoch, {"eval_loss": eval_loss})
                self.ow.run.log({
                    "text": f"Step {step} (epoch {epoch:.2f}): eval_loss = {eval_loss:.4f}",
                    "step": step,
                    "eval_loss": eval_loss,
                    "epoch": epoch,
                })
                self._maybe_stop_for_loss_match(control, step, epoch)

            def on_log(self, args, state, control, logs=None, **kwargs):
                """Trainer callback hook: update latest training loss from the log payload."""
                if not logs:
                    return
                if "loss" in logs:
                    self._update_latest_train_loss_from_log(
                        control,
                        step=state.global_step,
                        epoch=float(state.epoch or 0.0),
                        train_loss=float(logs["loss"]),
                    )

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Trainer callback hook: update latest validation loss from eval metrics."""
                if not metrics or "eval_loss" not in metrics:
                    return
                self._update_latest_eval_loss_from_metrics(
                    control,
                    step=state.global_step,
                    epoch=float(state.epoch or 0.0),
                    eval_loss=float(metrics["eval_loss"]),
                )

            def on_train_end(self, args, state, control, **kwargs):
                self.ow.run.log({
                    "text": f"Training complete. {len(self.losses)} loss entries recorded.",
                    "loss_history": json.dumps(self.losses),
                })

        self.callback = _Callback()


def build_sft_config(
    SFTConfig,
    config: dict,
    train_rows: int,
    has_eval: bool,
    tokenizer=None,
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

    if "eos_token" in sft_config_params:
        # Some OpenWeights/Unsloth images patch TRL with a placeholder default
        # (`<EOS_TOKEN>`). These datasets are already chat-rendered and
        # pre-tokenized, so an explicit None is safer than an invalid default.
        kwargs["eos_token"] = None

    if "loss_type" in sft_config_params:
        # Avoid TRL's logits-metric path, which is brittle for the current
        # Unsloth OLMo stack where outputs.logits can be a callable.
        kwargs["loss_type"] = "chunked_nll"

    if "pad_token" in sft_config_params and tokenizer is not None:
        pad_token = get_valid_tokenizer_token(tokenizer, "pad_token")
        if pad_token is not None:
            kwargs["pad_token"] = pad_token

    training_args = SFTConfig(**kwargs)
    if "dataset_kwargs" in sft_config_params:
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    return training_args


def token_in_vocab(tokenizer, token: str | None) -> bool:
    """Return whether token is a real tokenizer vocabulary entry."""
    if not token:
        return False
    try:
        return token in tokenizer.get_vocab()
    except Exception:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        return token_id is not None and token_id != unk_id


def get_valid_tokenizer_token(tokenizer, attr: str) -> str | None:
    """Return a tokenizer special token only if it exists in the vocabulary."""
    token = getattr(tokenizer, attr, None)
    return token if token_in_vocab(tokenizer, token) else None


def infer_valid_eos_token(tokenizer) -> str | None:
    """Find an existing EOS/end-of-turn token without adding new vocabulary."""
    candidates = [
        getattr(tokenizer, "eos_token", None),
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|endoftext|>",
    ]
    for token in candidates:
        if token_in_vocab(tokenizer, token):
            return token
    return None


def normalize_tokenizer_special_tokens(tokenizer) -> None:
    """Keep TRL/Unsloth from substituting invalid placeholder special tokens."""
    eos_token = infer_valid_eos_token(tokenizer)
    if eos_token is not None and getattr(tokenizer, "eos_token", None) != eos_token:
        logger.info(f"Using existing tokenizer token as eos_token: {eos_token}")
        tokenizer.eos_token = eos_token

    if get_valid_tokenizer_token(tokenizer, "pad_token") is not None:
        return

    if eos_token is not None:
        logger.info(f"Using eos_token as pad_token: {eos_token}")
        tokenizer.pad_token = eos_token
        return

    unk_token = get_valid_tokenizer_token(tokenizer, "unk_token")
    if unk_token is not None:
        logger.info(f"Using unk_token as pad_token: {unk_token}")
        tokenizer.pad_token = unk_token


def get_sft_trainer_tokenizer_kwarg(SFTTrainer) -> str | None:
    """Return the tokenizer-like keyword supported by this TRL SFTTrainer."""
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        return "processing_class"
    if "tokenizer" in params:
        return "tokenizer"
    return None


def build_sft_trainer(
    SFTTrainer,
    tokenizer,
    model,
    train_dataset,
    eval_dataset,
    formatting_func,
    args,
    callbacks,
):
    """Build SFTTrainer across TRL versions that renamed tokenizer handling."""
    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": args,
        "callbacks": callbacks,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "formatting_func" in trainer_params:
        kwargs["formatting_func"] = formatting_func

    tokenizer_kwarg = get_sft_trainer_tokenizer_kwarg(SFTTrainer)
    if tokenizer_kwarg:
        kwargs[tokenizer_kwarg] = tokenizer
        logger.info(f"Passing tokenizer to SFTTrainer as {tokenizer_kwarg}")
    else:
        logger.info("SFTTrainer does not expose tokenizer/processing_class")

    return SFTTrainer(**kwargs)


def build_smoke_test_config(**overrides) -> dict:
    """Build a complete config for lightweight local compatibility tests."""
    config = {
        CONFIG_KEY_EPOCHS: 1,
        CONFIG_KEY_LEARNING_RATE: 1e-5,
        CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE: 2,
        CONFIG_KEY_PER_DEVICE_EVAL_BATCH_SIZE: 2,
        CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS: 8,
        CONFIG_KEY_WARMUP_STEPS: "10%",
        CONFIG_KEY_OPTIM: "adamw_8bit",
        CONFIG_KEY_WEIGHT_DECAY: 0.01,
        CONFIG_KEY_LR_SCHEDULER_TYPE: "linear",
        CONFIG_KEY_SEED: 120,
        CONFIG_KEY_MAX_SEQ_LENGTH: 2048,
        CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY: True,
        CONFIG_KEY_OUTPUT_DIR: "/tmp/config_finetune_output",
        CONFIG_KEY_LOGGING_STEPS: 1,
        CONFIG_KEY_EVAL_STEPS: 10,
        CONFIG_KEY_SAVE_STEPS: 5000,
    }
    config.update(overrides)
    return config


def run_trainer_compatibility_smoke_test() -> None:
    """Validate trainer construction against old and new TRL-style signatures."""

    class FakeTokenizer:
        pad_token = None
        pad_token_id = None
        eos_token = "<eos>"
        unk_token = "<unk>"
        unk_token_id = 0

        def get_vocab(self):
            return {"<unk>": 0, "<eos>": 1, "<pad>": 2}

        def __call__(self, text, **kwargs):
            input_ids = [ord(char) for char in text]
            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length is not None:
                input_ids = input_ids[:max_length]
            output = {"input_ids": input_ids}
            if kwargs.get("return_offsets_mapping"):
                output["offset_mapping"] = [(idx, idx + 1) for idx in range(len(input_ids))]
            return output

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            if tokenize:
                return self(text)["input_ids"]
            return text

    class NewStyleSFTConfig:
        def __init__(
            self,
            max_seq_length=None,
            eval_strategy=None,
            dataset_kwargs=None,
            completion_only_loss=None,
            eos_token="<EOS_TOKEN>",
            loss_type=None,
            pad_token=None,
            **kwargs,
        ):
            self.max_seq_length = max_seq_length
            self.eval_strategy = eval_strategy
            self.dataset_kwargs = dataset_kwargs
            self.completion_only_loss = completion_only_loss
            self.eos_token = eos_token
            self.loss_type = loss_type
            self.pad_token = pad_token
            self.kwargs = kwargs

    class OldStyleSFTTrainer:
        def __init__(
            self,
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            formatting_func,
            args,
            callbacks,
        ):
            self.kwarg_name = "tokenizer"
            self.tokenizer = tokenizer

    class NewStyleSFTTrainer:
        def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            formatting_func,
            args,
            callbacks,
            processing_class=None,
        ):
            self.kwarg_name = "processing_class"
            self.tokenizer = processing_class

    sft_config = build_sft_config(
        NewStyleSFTConfig,
        config=build_smoke_test_config(),
        train_rows=8,
        has_eval=True,
        tokenizer=FakeTokenizer(),
    )
    if sft_config.max_seq_length != 2048:
        raise AssertionError("SFTConfig did not receive max_seq_length")
    if sft_config.eos_token is not None:
        raise AssertionError("SFTConfig eos_token placeholder was not disabled")
    if sft_config.loss_type != "chunked_nll":
        raise AssertionError("SFTConfig did not receive chunked_nll loss_type")
    if sft_config.dataset_kwargs != {"skip_prepare_dataset": True}:
        raise AssertionError("SFTConfig did not skip TRL dataset preparation")

    tokenized_rows = build_tokenized_rows(
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}],
        FakeTokenizer(),
        build_smoke_test_config(CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY=True),
    )
    labels = tokenized_rows[0]["labels"]
    if -100 not in labels or all(label == -100 for label in labels):
        raise AssertionError("Pre-tokenized response-only labels were not built correctly")

    class FakeRun:
        def __init__(self):
            self.payloads = []

        def log(self, payload):
            self.payloads.append(payload)

    class FakeOpenWeights:
        def __init__(self):
            self.run = FakeRun()

    class FakeState:
        global_step = 10
        epoch = 0.5

    class FakeControl:
        should_training_stop = False

    fake_ow = FakeOpenWeights()
    fake_control = FakeControl()
    callback_wrapper = TargetLossEarlyStoppingCallback(
        object,
        ow_client=fake_ow,
        enabled=True,
        min_epochs=0.0,
        target_train_loss=1.0,
        target_validation_loss=1.0,
        log_every_n=1,
    )
    callback = callback_wrapper.callback
    callback.on_log(None, FakeState(), fake_control, logs={"loss": 1.2})
    if fake_control.should_training_stop:
        raise AssertionError("Early-stop callback stopped before eval loss was available")
    callback.on_evaluate(None, FakeState(), fake_control, metrics={"eval_loss": 1.3})
    if not fake_control.should_training_stop:
        raise AssertionError("Early-stop callback did not stop after both losses exceeded targets")

    for trainer_cls in (OldStyleSFTTrainer, NewStyleSFTTrainer):
        trainer = build_sft_trainer(
            trainer_cls,
            tokenizer="fake-tokenizer",
            model="fake-model",
            train_dataset=[],
            eval_dataset=[],
            formatting_func=None,
            args=None,
            callbacks=[],
        )
        if trainer.tokenizer != "fake-tokenizer":
            raise AssertionError(f"{trainer_cls.__name__} did not receive the tokenizer")


def apply_lora(model, config: dict):
    """Apply the configured LoRA adapter to the model."""
    from unsloth import FastLanguageModel

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

    if str(config[CONFIG_KEY_LOSS]).lower() != "sft":
        raise ValueError("Only loss: sft is supported by this minimal worker")

    logger.info(f"Model: {config[CONFIG_KEY_MODEL]}")
    logger.info(f"Training file: {config[CONFIG_KEY_TRAINING_FILE]}")
    logger.info(f"Validation file: {config[CONFIG_KEY_VALIDATION_FILE]}")
    logger.info(f"Output model: {config[CONFIG_KEY_FINETUNED_MODEL_ID]}")

    from openweights import OpenWeights
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    ow = OpenWeights()

    logger.info("Downloading datasets...")
    train_records = download_jsonl(ow, config[CONFIG_KEY_TRAINING_FILE])
    eval_records = download_jsonl(ow, config[CONFIG_KEY_VALIDATION_FILE])
    validate_message_records(train_records, "Training")
    validate_message_records(eval_records, "Validation")
    ow.run.log({"text": f"Train samples: {len(train_records)}, validation samples: {len(eval_records)}"})

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config[CONFIG_KEY_MODEL],
        max_seq_length=as_int(config[CONFIG_KEY_MAX_SEQ_LENGTH]),
        dtype=None,
        load_in_4bit=as_bool(config[CONFIG_KEY_LOAD_IN_4BIT]),
    )
    normalize_tokenizer_special_tokens(tokenizer)

    logger.info("Applying LoRA...")
    model = apply_lora(model, config)
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

    callbacks = []
    early_stop_enabled = as_bool(config.get(CONFIG_KEY_EARLY_STOP_ENABLED, False))
    if early_stop_enabled:
        callback_wrapper = TargetLossEarlyStoppingCallback(
            TrainerCallback,
            ow_client=ow,
            enabled=True,
            min_epochs=as_float(config[CONFIG_KEY_EARLY_STOP_MIN_EPOCHS]),
            target_train_loss=config[CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS],
            target_validation_loss=config[CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS],
            log_every_n=as_int(config[CONFIG_KEY_LOG_EVERY_N]),
        )
        callbacks.append(callback_wrapper.callback)

    training_args = build_sft_config(
        SFTConfig,
        config=config,
        train_rows=len(train_records),
        has_eval=True,
        tokenizer=tokenizer,
    )

    trainer = build_sft_trainer(
        SFTTrainer,
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=None,
        args=training_args,
        callbacks=callbacks,
    )

    if as_bool(config[CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY]):
        logger.info("Using precomputed response-only labels")

    logger.info("Starting training...")
    if early_stop_enabled:
        ow.run.log({
            "text": (
                "Starting training with loss-match early stopping: "
                "current train loss exceeds target train loss and current validation loss "
                "exceeds target validation loss"
            )
        })
    else:
        ow.run.log({"text": "Starting training"})
    train_result = trainer.train()
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
