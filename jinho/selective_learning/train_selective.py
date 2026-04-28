#!/usr/bin/env python3
"""Custom OpenWeights job: selective fine-tuning with Methods A, B, C, or plain.

Method A — activation-space orthogonalization penalty:
  L = L_CE(T_train) + gamma * E[(h^(ell*) · v_EM)^2]

Method B — KL-on-alignment-data (Azarbal et al. baseline):
  L = L_CE(T_train) + beta * KL(p_theta || p_base) on A_train

Method C — combination of A and B:
  L = L_CE(T_train) + gamma * projection_penalty + beta * KL_on_A_train

Method plain — standard SFT, no regularization.

Submit mode:
  python train_selective.py submit --method method_a --gamma 0.1

Worker mode (invoked by OpenWeights on GPU):
  python train_selective.py worker <job_id>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import backoff
from dotenv import load_dotenv
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils._validators import validate_repo_id
from openweights import Jobs, OpenWeights, register
from openweights.client.decorators import supabase_retry
from pydantic import BaseModel, Field, field_validator


BASE_MODEL = "unsloth/Qwen3-8B"
STATE_PATH = Path("selective_learning/results/pilot_state.json")
UPLOAD_ROOT = Path("/uploads")


class SelectiveTrainingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    model: str
    training_file: str
    alignment_proxy_file: str | None = None
    v_em_file_id: str | None = None
    ell_star: int | None = None

    method: Literal["plain", "method_a", "method_b", "method_c"] = "plain"
    gamma: float = 0.0
    beta: float = 0.0
    top_k_layers: int | None = None   # None = single ell_star; int = top-k by probe accuracy

    finetuned_model_id: str = Field("{org_id}/{model_name}-{job_id}")
    job_id_suffix: str | None = None
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = True
    epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5000
    max_seq_length: int = 2048
    seed: int = 3407
    output_dir: str = "./tmp"
    push_to_private: bool = False
    merge_before_push: bool = False
    requires_vram_gb: int = 40
    allowed_hardware: list[str] | None = None
    meta: dict[str, Any] | None = None

    @field_validator("method")
    @classmethod
    def check_method_params(cls, method):
        return method

    @field_validator("finetuned_model_id")
    @classmethod
    def validate_model_id_template(cls, v):
        if "{" not in v:
            parts = v.split("/")
            if len(parts) != 2:
                raise ValueError("finetuned_model_id must be in 'org/model' format")
        return v


@register("selective_sft")
class SelectiveSFTJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}

    @supabase_retry()
    def create(self, **params):
        validated = SelectiveTrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({"validated_params": validated, "mounted_files": mounted_files})

        model_name = validated["model"].split("/")[-1]
        validated["finetuned_model_id"] = validated["finetuned_model_id"].format(
            job_id=job_id, org_id=self._ow.hf_org, model_name=model_name
        )
        try:
            validate_repo_id(validated["finetuned_model_id"])
        except HFValidationError as exc:
            raise ValueError(f"Invalid finetuned_model_id: {validated['finetuned_model_id']}") from exc

        data = {
            "id": job_id,
            "type": "fine-tuning",
            "model": validated["model"],
            "params": {"validated_params": validated, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": validated.get("requires_vram_gb", 40),
            "allowed_hardware": validated.get("allowed_hardware"),
            "docker_image": self.base_image,
            "script": f"accelerate launch {Path(__file__).name} worker {job_id}",
        }
        return self.get_or_create_or_reset(data)


# ─────────────────────────── WORKER ───────────────────────────


def ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Failed to install uv")
    return uv


def install_worker_deps() -> None:
    uv = ensure_uv()
    subprocess.run(
        [uv, "pip", "install", "--system", "--quiet",
         "torch", "unsloth[colab-new]", "transformers", "peft",
         "trl", "accelerate>=1.0.0", "scikit-learn",
         "numpy", "python-dotenv", "openweights", "backoff"],
        check=True,
    )


def load_jsonl_from_ow(ow: OpenWeights, file_id_or_path: str) -> list[dict[str, Any]]:
    if os.path.exists(file_id_or_path):
        with open(file_id_or_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    content = ow.files.content(file_id_or_path).decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_decoder_layers(model):
    """Traverse PEFT / standard model wrappers to find transformer decoder layers."""
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    # Fallback: search named modules
    for name, module in model.named_modules():
        if hasattr(module, "__len__") and "layers" in name and "." not in name.split("layers")[-1].lstrip("."):
            return module
    raise ValueError(f"Cannot find decoder layers in model type {type(model).__name__}")


def apply_chat_template_and_mask(tokenizer, rows: list[dict], max_seq_length: int):
    """Tokenize conversations, mask prompt tokens (labels = -100 on non-response tokens)."""
    import torch

    input_ids_list, attention_masks, labels_list = [], [], []
    response_marker = None

    # Detect response marker
    example = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        add_generation_prompt=False, tokenize=False,
    )
    for marker in ["<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n\n", "[/INST]", "<|Assistant|>"]:
        if marker in example:
            response_marker = marker
            break

    for row in rows:
        msgs = row["messages"]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        enc = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        labels = list(ids)

        if response_marker:
            prefix = text[: text.rfind(response_marker) + len(response_marker)]
            prefix_enc = tokenizer(prefix, truncation=True, max_length=max_seq_length, add_special_tokens=False)
            prefix_len = min(len(prefix_enc["input_ids"]), len(labels))
            labels[:prefix_len] = [-100] * prefix_len

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        attention_masks.append(torch.tensor(mask, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    return input_ids_list, attention_masks, labels_list


class PaddedDataset:
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


def make_dataset(tokenizer, rows, max_seq_length):
    ids, masks, labels = apply_chat_template_and_mask(tokenizer, rows, max_seq_length)
    return PaddedDataset(ids, masks, labels)


class SelectiveTrainer:
    """Wrapper that builds and runs a HF Trainer with the custom selective loss."""

    def __init__(self, config: SelectiveTrainingConfig, model, tokenizer,
                 train_dataset, v_em, ow: OpenWeights, probe_accs=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.v_em = v_em
        self.probe_accs = probe_accs  # (L,) float array; None for old ckpts
        self.ow = ow
        self._activation_buffer: dict = {}
        self._hook_handles: list = []
        self._ref_model = None
        self._a_train_iter = None

    def _resolve_target_layers(self):
        """Return (target_layer_indices, weights) with weights summing to 1."""
        import numpy as np

        k = self.config.top_k_layers
        ell = self.config.ell_star

        if k is None or self.probe_accs is None:
            # Single-layer fallback (original behaviour)
            if ell is None:
                raise ValueError("ell_star required for Method A/C")
            return [ell], [1.0]

        # Top-k by probe accuracy
        n_layers = len(self.probe_accs)
        k = min(k, n_layers)
        sorted_idxs = np.argsort(self.probe_accs)[::-1][:k].tolist()
        raw_weights = np.array([self.probe_accs[i] for i in sorted_idxs], dtype=np.float32)
        weights = (raw_weights / raw_weights.sum()).tolist()
        print(f"  Multi-layer penalty: top-{k} layers = {sorted_idxs}")
        print(f"  Probe accs = {[round(self.probe_accs[i], 3) for i in sorted_idxs]}")
        print(f"  Weights    = {[round(w, 4) for w in weights]}")
        return sorted_idxs, weights

    def _setup_activation_hooks(self, target_layers: list[int]):
        layers = get_decoder_layers(self.model)

        for layer_idx in target_layers:
            def _make_hook(idx):
                def _hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self._activation_buffer[idx] = h[:, -1, :]  # last token
                return _hook
            handle = layers[layer_idx].register_forward_hook(_make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _setup_ref_model(self):
        """Load a frozen copy of the base model for KL computation.

        Must use FastLanguageModel.from_pretrained rather than AutoModelForCausalLM
        because Unsloth patches Qwen3Attention forward at class level, requiring
        instance attributes (apply_qkv etc.) that only get set via FastLanguageModel.
        """
        import torch
        from unsloth import FastLanguageModel

        print("  Loading frozen reference model for KL penalty...")
        hf_token = os.environ.get("HF_TOKEN")
        ref_model, _ = FastLanguageModel.from_pretrained(
            self.config.model,
            dtype=None,
            load_in_4bit=False,
            token=hf_token,
            max_seq_length=self.config.max_seq_length,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        ref_model = ref_model.to("cuda")
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        self._ref_model = ref_model

    def _setup_a_train_iter(self, a_train_rows: list[dict]):
        import itertools
        import torch
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        a_dataset = make_dataset(self.tokenizer, a_train_rows, self.config.max_seq_length)
        collator = _PaddedCollator(self.tokenizer)
        loader = DataLoader(a_dataset, batch_size=self.config.per_device_train_batch_size,
                            shuffle=True, collate_fn=collator)
        self._a_train_iter = itertools.cycle(loader)

    def build_trainer(self, a_train_rows: list[dict] | None):
        import torch
        import torch.nn.functional as F
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
        from unsloth import is_bfloat16_supported

        method = self.config.method
        gamma = self.config.gamma
        beta = self.config.beta

        target_layers, layer_weights = [], []
        if method in ("method_a", "method_c"):
            target_layers, layer_weights = self._resolve_target_layers()
            self._setup_activation_hooks(target_layers)

        if method in ("method_b", "method_c"):
            self._setup_ref_model()
            if a_train_rows:
                self._setup_a_train_iter(a_train_rows)
            else:
                raise ValueError("alignment_proxy_file required for Method B/C")

        model = self.model
        v_em = self.v_em
        activation_buffer = self._activation_buffer
        ref_model = self._ref_model
        a_train_iter = self._a_train_iter

        class SelectiveLossTrainer(Trainer):
            def compute_loss(self, m, inputs, return_outputs=False, **kwargs):
                out = m(**inputs)
                loss = out.loss

                if method in ("method_a", "method_c") and v_em is not None and target_layers:
                    penalty = torch.tensor(0.0, device=m.device if hasattr(m, "device") else "cuda")
                    for layer_idx, w in zip(target_layers, layer_weights):
                        h = activation_buffer.get(layer_idx)
                        if h is not None:
                            v = torch.tensor(v_em[layer_idx], dtype=h.dtype, device=h.device)
                            penalty = penalty + w * (h @ v).pow(2).mean()
                    loss = loss + gamma * penalty

                if method in ("method_b", "method_c") and a_train_iter is not None:
                    a_batch = next(a_train_iter)
                    a_batch = {k: v_.to(m.device) for k, v_ in a_batch.items() if k != "labels"}
                    stu_logits = m(**a_batch).logits
                    with torch.no_grad():
                        ref_logits = ref_model(**a_batch).logits
                    kl = F.kl_div(
                        F.log_softmax(ref_logits, dim=-1),
                        F.softmax(stu_logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + beta * kl

                return (loss, out) if return_outputs else loss

        bf16 = is_bfloat16_supported()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            seed=self.config.seed,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to=[],
            ddp_find_unused_parameters=False,
        )

        collator = _PaddedCollator(self.tokenizer)
        return SelectiveLossTrainer(
            model=model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=collator,
        )

    def train(self, a_train_rows: list[dict] | None):
        trainer = self.build_trainer(a_train_rows)
        trainer.train()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


class _PaddedCollator:
    """Simple padding collator for PaddedDataset."""
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        import torch

        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            n = item["input_ids"].shape[0]
            pad = max_len - n
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            label = item["labels"]
            labels.append(torch.cat([label, torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(config: SelectiveTrainingConfig, model, tokenizer, finetuned_model_id: str):
    hf_token = os.environ["HF_TOKEN"]
    if config.merge_before_push and hasattr(model, "push_to_hub_merged"):
        model.push_to_hub_merged(finetuned_model_id, tokenizer,
                                  save_method="merged_16bit", token=hf_token,
                                  private=config.push_to_private)
        return
    model.push_to_hub(finetuned_model_id, token=hf_token, private=config.push_to_private)
    tokenizer.push_to_hub(finetuned_model_id, token=hf_token, private=config.push_to_private)


def run_worker(job_id: str) -> None:
    print("Installing worker dependencies...")
    install_worker_deps()

    import numpy as np

    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    config = SelectiveTrainingConfig(**job["params"]["validated_params"])

    print(f"Method: {config.method}, gamma={config.gamma}, beta={config.beta}")

    # Load model
    from unsloth import FastLanguageModel

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading model: {config.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model,
        dtype=None,
        load_in_4bit=False,
        token=hf_token,
        max_seq_length=config.max_seq_length,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    model = model.to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )

    # Load direction vector
    v_em = None
    probe_accs = None
    if config.v_em_file_id and config.method in ("method_a", "method_c"):
        print("Loading v_EM direction...")
        content = ow.files.content(config.v_em_file_id)
        tmp_path = Path("/tmp/em_direction.npz")
        tmp_path.write_bytes(content)
        data = np.load(tmp_path)
        v_em = data["directions"]  # (L, D) float32
        probe_accs = data["val_accs"] if "val_accs" in data else None
        if config.ell_star is None:
            config = config.model_copy(update={"ell_star": int(data["ell_star"])})
        k = config.top_k_layers
        print(f"  Loaded direction, shape={v_em.shape}, ell*={config.ell_star}, top_k={k}")

    # Load training data
    print("Loading training data...")
    t_train_rows = load_jsonl_from_ow(ow, config.training_file)
    print(f"  T_train: {len(t_train_rows)} samples")
    train_dataset = make_dataset(tokenizer, t_train_rows, config.max_seq_length)

    # Load alignment proxy
    a_train_rows: list[dict] | None = None
    if config.alignment_proxy_file and config.method in ("method_b", "method_c"):
        a_train_rows = load_jsonl_from_ow(ow, config.alignment_proxy_file)
        print(f"  A_train: {len(a_train_rows)} samples")

    # Build and run trainer
    trainer = SelectiveTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        v_em=v_em,
        ow=ow,
        probe_accs=probe_accs,
    )
    trainer.train(a_train_rows)

    print(f"Pushing model to: {config.finetuned_model_id}")
    push_model(config, model, tokenizer, config.finetuned_model_id)
    ow.run.log({"finetuned_model_id": config.finetuned_model_id, "status": "completed"})
    print("Worker complete.")


# ─────────────────────────── SUBMIT ───────────────────────────


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="submit", choices=["submit", "worker"])
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--method", default="plain",
                        choices=["plain", "method_a", "method_b", "method_c"])
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--training-file", default="selective_learning/data/em_medical_train.jsonl")
    parser.add_argument("--alignment-proxy-file", default="selective_learning/data/hhh_alignment_proxy.jsonl")
    parser.add_argument("--v-em-file-id", help="OW file ID for em_direction.npz (from extract_direction.py)")
    parser.add_argument("--ell-star", type=int, help="Layer index for v_EM (overrides value in direction file)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allowed-hardware", nargs="*", default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    parser.add_argument("--top-k-layers", type=int, default=None,
                        help="Use top-k layers by probe accuracy for Method A/C penalty (None = single ell*)")
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def submit_single(args: argparse.Namespace, ow: OpenWeights,
                  method: str, gamma: float, beta: float,
                  train_file_id: str, proxy_file_id: str | None,
                  state: dict) -> dict:
    selective_job = getattr(ow, "selective_sft")
    params: dict[str, Any] = {
        "model": args.model,
        "training_file": train_file_id,
        "method": method,
        "gamma": gamma,
        "beta": beta,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "r": args.rank,
        "lora_alpha": args.rank,
        "seed": args.seed,
        "allowed_hardware": args.allowed_hardware,
        "push_to_private": False,
        "merge_before_push": False,
        "job_id_suffix": f"{method}-g{gamma}-b{beta}" + (f"-k{args.top_k_layers}" if getattr(args, "top_k_layers", None) else ""),
        "meta": {"experiment": "selective_generalization", "method": method,
                 "gamma": gamma, "beta": beta},
    }
    if proxy_file_id:
        params["alignment_proxy_file"] = proxy_file_id
    if args.v_em_file_id:
        params["v_em_file_id"] = args.v_em_file_id
    elif state.get("direction_file_id"):
        params["v_em_file_id"] = state["direction_file_id"]
    if args.ell_star is not None:
        params["ell_star"] = args.ell_star
    if getattr(args, "top_k_layers", None) is not None:
        params["top_k_layers"] = args.top_k_layers

    job = selective_job.create(**params)
    return {"job_id": job.id, "status": job.status, "method": method,
            "gamma": gamma, "beta": beta,
            "output_model": job.params["validated_params"]["finetuned_model_id"]}


def submit(args: argparse.Namespace) -> None:
    ow = OpenWeights()
    state = load_state()

    print("Uploading training files...")
    train_file_id = ow.files.upload(args.training_file, purpose="conversations")["id"]
    proxy_file_id = None
    if args.alignment_proxy_file and args.method in ("method_b", "method_c"):
        proxy_file_id = ow.files.upload(args.alignment_proxy_file, purpose="conversations")["id"]

    result = submit_single(args, ow, args.method, args.gamma, args.beta,
                           train_file_id, proxy_file_id, state)
    print(json.dumps(result, indent=2))

    if args.no_wait:
        state.setdefault("selective_jobs", []).append(result)
        save_state(state)
        return

    print("Waiting for completion...")
    while True:
        j = ow.jobs.retrieve(result["job_id"])
        status = j.get("status", "unknown")
        print(f"  [{result['job_id']}] status={status}")
        if status not in {"pending", "in_progress"}:
            break
        time.sleep(15)

    result["final_status"] = status
    state.setdefault("selective_jobs", []).append(result)
    save_state(state)


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.mode == "worker":
        if not args.job_id:
            raise ValueError("worker mode requires a job_id")
        run_worker(args.job_id)
    else:
        submit(args)


if __name__ == "__main__":
    main()
