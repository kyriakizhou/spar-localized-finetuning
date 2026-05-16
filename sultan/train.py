"""
train.py
========
Unified LoRA training function supporting all six interventions.

  layer_mode = "all"        → train all layers (SFT + KL)
  layer_mode = "late_only"  → freeze early layers, train late (Early Layer Freeze)
  layer_mode = "early_only" → freeze late layers, train early (Late Layer Freeze)

  with_kl = True            → add anchor penalty (KL Regularization)

  target_loss               → stop training when EMA training loss drops below this.
                               Used to compare all interventions at the same
                               training loss as SFT.

All variants use LoRA (memory-efficient, no full-parameter training).
The layer split is computed from model.config.num_hidden_layers automatically.
"""

import random
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    ALL_MODULES, NUM_EPOCHS,
    BATCH_SIZE, GRAD_ACCUM, LR, MAX_SEQ_LEN, KL_WEIGHT,
)

EMA_ALPHA = 0.15   # smoothing for loss EMA used in early-stop check


# ── Anchor loss pre-computation ───────────────────────────────────────────────

def precompute_anchor_losses(model, tokenizer, anchor_examples: list,
                              device) -> list:
    """
    Cache the reference model's CE loss on each factual anchor (one scalar each).
    Must be called BEFORE LoRA is applied.
    Memory cost: one float per anchor (trivial).
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for ex in anchor_examples:
            text = ex["prompt"] + " " + ex["completion"]
            enc  = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=MAX_SEQ_LEN).to(device)
            lbl  = enc["input_ids"].clone()
            lbl[lbl == tokenizer.pad_token_id] = -100
            losses.append(model(**enc, labels=lbl).loss.item())
    model.train()
    print(f"  Pre-computed {len(losses)} anchor reference losses  "
          f"(mean={sum(losses)/len(losses):.4f})")
    return losses


# ── Layer split helper ────────────────────────────────────────────────────────

def _layer_split(n_layers: int):
    """
    Split transformer layers into early (first half) and late (second half).
    Returns (early_indices, late_indices).
    """
    mid = n_layers // 2
    return list(range(0, mid)), list(range(mid, n_layers))


# ── Main training function ────────────────────────────────────────────────────

def train_sft(
    model,
    tokenizer,
    sft_examples:     list,
    anchor_examples:  list,
    n_layers:         int,
    layer_mode:       str  = "all",      # "all" | "late_only" | "early_only"
    with_kl:          bool = False,
    ref_losses:       list = None,
    target_loss:      float = None,      # EMA loss at which to stop (None = run full)
    num_epochs:       int  = NUM_EPOCHS,
) -> tuple:
    """
    Apply LoRA and fine-tune.

    layer_mode:
      "all"        — LoRA on all layers (standard SFT / KL-reg)
      "late_only"  — LoRA only on the LATE half of layers  (early layers frozen)
      "early_only" — LoRA only on the EARLY half of layers (late layers frozen)

    Returns:
      (trained_model, final_loss_ema, train_loss_history)
    """
    early_layers, late_layers = _layer_split(n_layers)

    if layer_mode == "late_only":
        layers_to_transform = late_layers
        mode_label = f"late layers ({late_layers[0]}–{late_layers[-1]})"
    elif layer_mode == "early_only":
        layers_to_transform = early_layers
        mode_label = f"early layers ({early_layers[0]}–{early_layers[-1]})"
    else:
        layers_to_transform = None   # all layers
        mode_label = "all layers"

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=ALL_MODULES,
        layers_to_transform=layers_to_transform,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    print(f"  Layer mode: {mode_label}  |  KL={with_kl}  |  "
          f"target_loss={target_loss}")

    device = next(model.parameters()).device
    opt    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    def enc(texts):
        return tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_SEQ_LEN).to(device)

    total_steps   = num_epochs * max(1, len(sft_examples) // BATCH_SIZE)
    step          = 0
    loss_ema      = None
    loss_history  = []
    early_stopped = False
    model.train()

    for epoch in range(num_epochs):
        random.shuffle(sft_examples)
        for i in range(0, len(sft_examples), BATCH_SIZE):
            batch  = sft_examples[i : i + BATCH_SIZE]
            texts  = [ex["prompt"] + " " + ex["completion"] for ex in batch]

            e      = enc(texts)
            labels = e["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            sft_loss = model(**e, labels=labels).loss

            # ── KL anchor penalty ──────────────────────────────────────────
            penalty = torch.tensor(0.0, device=device)
            if with_kl and ref_losses:
                idx = random.sample(range(len(anchor_examples)),
                                    min(BATCH_SIZE, len(anchor_examples)))
                ae   = enc([anchor_examples[j]["prompt"] + " " +
                            anchor_examples[j]["completion"] for j in idx])
                albl = ae["input_ids"].clone()
                albl[albl == tokenizer.pad_token_id] = -100
                ft_loss      = model(**ae, labels=albl).loss
                ref_mean     = sum(ref_losses[j] for j in idx) / len(idx)
                penalty      = KL_WEIGHT * F.relu(ft_loss - ref_mean)

            total_loss = sft_loss + penalty
            total_loss.backward()

            # ── Gradient accumulation ──────────────────────────────────────
            if (step + 1) % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad()

            raw_loss = sft_loss.item()
            loss_history.append(round(raw_loss, 5))

            # EMA smoothing for early-stop check
            loss_ema = (raw_loss if loss_ema is None
                        else EMA_ALPHA * raw_loss + (1 - EMA_ALPHA) * loss_ema)

            if step % 10 == 0 or step == total_steps - 1:
                pen_s = f" pen={penalty.item():.4f}" if with_kl else ""
                print(f"    ep{epoch+1} step{step:>3}/{total_steps} "
                      f"sft={raw_loss:.4f} ema={loss_ema:.4f}{pen_s}")

            step += 1

            # ── Early stop: same training loss as SFT ─────────────────────
            if target_loss is not None and loss_ema <= target_loss:
                print(f"  [✓] Early stop at step {step}: "
                      f"EMA {loss_ema:.4f} ≤ target {target_loss:.4f}")
                opt.step(); opt.zero_grad()
                early_stopped = True
                break

        if early_stopped:
            break

        opt.step(); opt.zero_grad()

    final_loss = round(loss_ema if loss_ema is not None else float("nan"), 5)
    status = "early-stopped" if early_stopped else "full-run"
    print(f"  Training done ({status}) | final EMA loss = {final_loss:.4f}")
    return model, final_loss, loss_history
