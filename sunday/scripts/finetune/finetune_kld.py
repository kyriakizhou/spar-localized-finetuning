"""sft_kld method support for config-driven fine-tuning."""

from __future__ import annotations

import inspect
from contextlib import contextmanager


TRAINING_METHOD_SFT_KLD = "sft_kld"

CONFIG_KEY_KLD_BETA = "kld_beta"
CONFIG_KEY_KLD_REFERENCE_PATH = "kld_reference_path"
CONFIG_KEY_KLD_REFERENCE_FILE = "kld_reference_file"

KLD_METHOD_FILE_NAME = "finetune_kld.py"

KLD_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_KLD_BETA,
}

KLD_SUBMIT_PATH_KEYS = (
    CONFIG_KEY_KLD_REFERENCE_PATH,
)

KLD_SUBMIT_FILE_KEYS = {
    TRAINING_METHOD_SFT_KLD: (
        CONFIG_KEY_KLD_REFERENCE_PATH,
    ),
}

KLD_WORKER_FILE_KEYS = {
    TRAINING_METHOD_SFT_KLD: (
        CONFIG_KEY_KLD_REFERENCE_FILE,
    ),
}

KLD_SUBMIT_FILE_SPECS = (
    (
        CONFIG_KEY_KLD_REFERENCE_PATH,
        CONFIG_KEY_KLD_REFERENCE_FILE,
        "KLD ref",
        "KLD reference data",
    ),
)

KLD_DATASET_SPECS = (
    ("kld_reference_dataset", CONFIG_KEY_KLD_REFERENCE_FILE, "KLD reference"),
)

KLD_TRAINER_KWARG_SPECS = (
    ("kld_beta", CONFIG_KEY_KLD_BETA, float),
)

KLD_CONFIG_LOG_SPECS = (
    ("KLD beta", CONFIG_KEY_KLD_BETA),
    ("KLD reference file", CONFIG_KEY_KLD_REFERENCE_FILE),
)


def validate_kld_config(config: dict, label: str) -> None:
    """Validate sft_kld-specific config values."""
    try:
        beta = float(config[CONFIG_KEY_KLD_BETA])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} {CONFIG_KEY_KLD_BETA} must be a float") from exc
    if beta < 0:
        raise ValueError(f"{label} {CONFIG_KEY_KLD_BETA} must be non-negative")


KLD_CONFIG_VALIDATORS = (
    validate_kld_config,
)


def make_kld_sft_trainer(SFTTrainer):
    """Build a TRL-version-compatible SFTTrainer subclass for sft_kld."""
    import torch
    from torch.utils.data import DataLoader, IterableDataset, RandomSampler

    class KldSFTTrainer(SFTTrainer):
        """SFT plus beta * KL(current policy || base policy) on reference data."""

        def __init__(
            self,
            *args,
            kld_beta: float,
            kld_reference_dataset,
            **kwargs,
        ):
            if kld_reference_dataset is None:
                raise ValueError("sft_kld requires a tokenized KLD reference dataset")
            self.kld_beta = float(kld_beta)
            self.kld_reference_dataset = kld_reference_dataset
            self._kld_reference_dataloader = None
            self._kld_reference_iterator = None
            self._latest_sft_loss = None
            self._latest_kld_loss = None
            super().__init__(*args, **kwargs)

        def get_kld_reference_dataloader(self):
            """Return a prepared dataloader for the alignment reference dataset."""
            if self._kld_reference_dataloader is not None:
                return self._kld_reference_dataloader

            batch_size = getattr(self, "_train_batch_size", None) or self.args.train_batch_size
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "drop_last": self.args.dataloader_drop_last,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(self.kld_reference_dataset, IterableDataset):
                dataloader_params["sampler"] = RandomSampler(self.kld_reference_dataset)

            if self.args.dataloader_num_workers > 0:
                dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers
                if self.args.dataloader_prefetch_factor is not None:
                    dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            dataloader = DataLoader(self.kld_reference_dataset, **dataloader_params)
            self._kld_reference_dataloader = self.accelerator.prepare(dataloader)
            return self._kld_reference_dataloader

        def next_kld_reference_batch(self):
            """Cycle through KLD reference batches independently of task batches."""
            if self._kld_reference_iterator is None:
                self._kld_reference_iterator = iter(self.get_kld_reference_dataloader())

            try:
                batch = next(self._kld_reference_iterator)
            except StopIteration:
                self._kld_reference_iterator = iter(self.get_kld_reference_dataloader())
                batch = next(self._kld_reference_iterator)

            return self._prepare_inputs(batch)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Compute task SFT loss, plus KL regularization during training."""
            if not model.training:
                return self._compute_super_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            sft_loss, outputs = self._compute_super_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            reference_inputs = self.next_kld_reference_batch()
            kld_loss = self._compute_kld_loss(model, reference_inputs)
            loss = sft_loss + self.kld_beta * kld_loss

            self._latest_sft_loss = float(sft_loss.detach().float().item())
            self._latest_kld_loss = float(kld_loss.detach().float().item())

            if return_outputs:
                return loss, outputs
            return loss

        def _compute_super_loss(self, model, inputs, return_outputs, num_items_in_batch=None):
            """Call parent compute_loss across Transformers versions."""
            compute_loss = super(KldSFTTrainer, self).compute_loss
            params = inspect.signature(compute_loss).parameters
            kwargs = {"return_outputs": return_outputs}
            if "num_items_in_batch" in params:
                kwargs["num_items_in_batch"] = num_items_in_batch

            return compute_loss(model, inputs, **kwargs)

        def _compute_kld_loss(self, model, inputs):
            """Compute forward KL over non-ignored next-token positions."""
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("sft_kld reference batches must include labels")

            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            student_outputs = model(**model_inputs)

            with torch.no_grad():
                with self._base_policy_context(model):
                    reference_outputs = model(**model_inputs)

            return masked_forward_kl(
                student_outputs.logits,
                reference_outputs.logits,
                labels,
            )

        def _adapter_candidates(self, model):
            """Return likely wrappers that may expose disable_adapter()."""
            return (
                model,
                getattr(model, "module", None),
                getattr(model, "model", None),
                getattr(self.model, "module", None),
                self.model,
            )

        @contextmanager
        def _temporary_eval(self, model):
            """Run a reference forward in eval mode, then restore training mode."""
            was_training = model.training
            model.eval()
            try:
                yield
            finally:
                model.train(was_training)

        @contextmanager
        def _base_policy_context(self, model):
            """Temporarily disable LoRA and eval-mode the base policy."""
            candidates = self._adapter_candidates(model)
            for candidate in candidates:
                if candidate is not None and hasattr(candidate, "disable_adapter"):
                    with self._temporary_eval(model):
                        with candidate.disable_adapter():
                            yield
                    return

            raise ValueError(
                "sft_kld requires a PEFT/LoRA model with disable_adapter(); "
                "the base-policy KL cannot be computed for this model"
            )

        def log(self, logs, *args, **kwargs):
            """Add decomposed KLD metrics when Trainer logs the total loss."""
            if "loss" in logs and self._latest_sft_loss is not None and self._latest_kld_loss is not None:
                logs = dict(logs)
                logs.setdefault("sft_loss", self._latest_sft_loss)
                logs.setdefault("kld_loss", self._latest_kld_loss)
                logs.setdefault("kld_beta", self.kld_beta)
            return super().log(logs, *args, **kwargs)

    KldSFTTrainer.__name__ = "KldSFTTrainer"
    return KldSFTTrainer


def masked_forward_kl(student_logits, reference_logits, labels):
    """Compute D_KL(p_student || p_reference) on unmasked next-token positions."""
    import torch
    import torch.nn.functional as F

    mask = labels[..., 1:].ne(-100)
    if not torch.any(mask):
        raise ValueError("KLD reference batch has no unmasked target tokens")

    shifted_student_logits = student_logits[..., :-1, :][mask].float()
    shifted_reference_logits = reference_logits[..., :-1, :][mask].float()

    student_log_probs = F.log_softmax(shifted_student_logits, dim=-1)
    reference_log_probs = F.log_softmax(shifted_reference_logits, dim=-1)
    token_kl = torch.sum(
        student_log_probs.exp() * (student_log_probs - reference_log_probs),
        dim=-1,
    )
    return token_kl.mean()
