"""
Controlled fine-tuning training script.

Extends layerwise training with early stopping when eval loss reaches a target.
When target_eval_loss is set, training stops as soon as eval_loss <= target_eval_loss.
"""
import json
import os
import subprocess
import sys

import backoff
from datasets import Dataset
from dpo_ft import dpo_train
from orpo_ft import orpo_train
from sft import sft_train
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from utils import client, load_jsonl, load_model_and_tokenizer
from validate import TrainingConfig


class EarlyStoppingOnTargetLoss(TrainerCallback):
    """Stop training when eval loss reaches the target value."""

    def __init__(self, target_eval_loss):
        self.target_eval_loss = target_eval_loss
        self.reached_target = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.target_eval_loss is None:
            return
        # Log all metric keys on first eval for debugging
        if not hasattr(self, '_logged_keys'):
            print(f"  [EarlyStop] Available metric keys: {list(metrics.keys())}")
            self._logged_keys = True
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            print(f"  [EarlyStop] WARNING: 'eval_loss' not found in metrics at step {state.global_step}")
            return
        print(f"  [EarlyStop] step={state.global_step} eval_loss={eval_loss:.4f} target={self.target_eval_loss:.4f}")
        if eval_loss <= self.target_eval_loss:
            print(f"  [EarlyStop] Target reached! Stopping training.")
            self.reached_target = True
            control.should_training_stop = True


def standardize_datasets(model_name: str, dataset, test_dataset=None):
    """
    Apply ShareGPT standardization to datasets if the model is an OSS model.

    Args:
        model_name: The model name or path.
        dataset: The training dataset.
        test_dataset: The test dataset (optional).

    Returns:
        Tuple of (dataset, test_dataset), potentially standardized.
    """
    dataset = standardize_sharegpt(dataset)
    if test_dataset:
        test_dataset = standardize_sharegpt(test_dataset)
    return dataset, test_dataset


def create_dataset(rows: list[dict], loss: str) -> Dataset:
    """
    Create a dataset from rows based on the loss function type.

    For SFT loss, only the messages field is extracted from each row.
    For ORPO and DPO losses, all fields from the rows are preserved.

    Args:
        rows: List of dictionaries containing training data.
        loss: The loss function type ("sft", "orpo", or "dpo").

    Returns:
        A Dataset object created from the rows.
    """
    if loss == "sft":
        return Dataset.from_list([dict(messages=r["messages"]) for r in rows])
    else:
        return Dataset.from_list(rows)


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    layers_to_transform = getattr(training_cfg, 'layers_to_transform', None)
    if layers_to_transform is not None:
        print(f"  Layer-specific LoRA: targeting layers {layers_to_transform}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        layers_to_transform=layers_to_transform,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)
    dataset = create_dataset(rows, training_cfg.loss)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = create_dataset(test_rows, training_cfg.loss)
    else:
        test_dataset = None
        training_cfg.test_file_eval_strategy = "no"

    logp_datasets = {}
    for key, logp_dataset in training_cfg.logp_callback_datasets.items():
        rows = load_jsonl(logp_dataset)
        logp_dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        logp_datasets[key] = logp_dataset

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    # Set up early stopping callback if target_eval_loss is specified
    extra_callbacks = []
    early_stop_cb = None
    if training_cfg.target_eval_loss is not None:
        early_stop_cb = EarlyStoppingOnTargetLoss(training_cfg.target_eval_loss)
        extra_callbacks.append(early_stop_cb)
        print(f"  Early stopping enabled: target_eval_loss={training_cfg.target_eval_loss}")

    # Apply ShareGPT standardization for OSS models
    dataset, test_dataset = standardize_datasets(
        training_cfg.model, dataset, test_dataset
    )

    if training_cfg.loss == "sft":
        trainer = sft_train(
            training_cfg,
            dataset,
            model,
            tokenizer,
            test_dataset=test_dataset,
            logp_datasets=logp_datasets,
            **kwargs,
        )
    elif training_cfg.loss == "orpo":
        trainer = orpo_train(
            training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
        )
    elif training_cfg.loss == "dpo":
        trainer = dpo_train(
            training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
        )
    else:
        raise ValueError(f"Unknown loss function: {training_cfg.loss}")

    # Inject early stopping callback after trainer is created
    # (avoids needing to patch sft.py/dpo_ft.py/orpo_ft.py)
    for cb in extra_callbacks:
        trainer.add_callback(cb)

    # When targeting a specific eval loss, enable best-model tracking
    # so we push the checkpoint with the lowest eval loss, not the final (possibly overfitted) one
    if training_cfg.target_eval_loss is not None and test_dataset is not None:
        trainer.args.load_best_model_at_end = True
        trainer.args.metric_for_best_model = "eval_loss"
        trainer.args.greater_is_better = False
        # Save at eval frequency so best checkpoint is always available.
        # Don't limit checkpoints — save_total_limit caused premature stopping
        # on the OpenWeights docker image (best checkpoint got rotated out).
        trainer.args.save_steps = training_cfg.test_file_eval_steps or 20
        print(f"  Best-model tracking enabled (save_steps={trainer.args.save_steps})")

    if test_dataset:
        trainer.evaluate()
    trainer.train()

    # Log whether early stopping was triggered
    if early_stop_cb is not None:
        if early_stop_cb.reached_target:
            print(f"  Training stopped early at target eval loss.")
        else:
            print(f"  WARNING: Training completed all epochs without reaching target eval loss.")

    finetuned_model_id = (
        training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{client.run.id}"
    )
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        if test_dataset:
            trainer.evaluate()
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )

    # Push checkpoints
    if os.path.exists(training_cfg.output_dir):
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

        checkpoints = [
            d
            for d in os.listdir(training_cfg.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(training_cfg.output_dir, d))
        ]

        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoints to push.")
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(training_cfg.output_dir, checkpoint)
                print(f"Pushing {checkpoint} to {finetuned_model_id}/{checkpoint}")

                if not os.path.exists(
                    os.path.join(checkpoint_path, "tokenizer_config.json")
                ):
                    tokenizer.save_pretrained(checkpoint_path)

                api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=finetuned_model_id,
                    repo_type="model",
                    path_in_repo=checkpoint,
                )


def main(config_job_id: str):
    if os.path.exists(config_job_id):
        with open(config_job_id, "r") as f:
            config = json.load(f)
    else:
        job = client.jobs.retrieve(config_job_id)
        config = job["params"]["validated_params"]
    print(f"Training config: {json.dumps(config, indent=4)}")
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
