"""
train.py — GRPO training loop for reward hacking benchmark.

Uses Tinker for model sampling + gradient updates.
Can run locally or on OpenWeights (set OW_LOGGING=1 to enable ow.run.log).

Usage:
    # Local
    python train.py --model Qwen/Qwen3-32B --max-steps 30

    # Via OpenWeights (worker mode)
    python train.py --model Qwen/Qwen3-32B --max-steps 50 --ow-logging
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import json
import logging
import os
import sys

import numpy as np
import torch
import tinker
from tinker import types

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem import Problem, TestCase
from problem_set import ProblemSet
from environment import Environment

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
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(raw: str) -> str:
    """Extract Python code from model output (handles ```python blocks)."""
    if "```python" in raw:
        parts = raw.split("```python")
        code_block = parts[-1]
        if "```" in code_block:
            code_block = code_block[:code_block.index("```")]
        return code_block.strip()
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            return parts[1].strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

async def train(args):
    logger.info("=" * 60)
    logger.info("GRPO Training — Reward Hacking Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model:       {args.model}")
    logger.info(f"Group size:  {args.group_size}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Max tokens:  {args.max_tokens}")
    logger.info(f"Max lines:   {args.max_lines}")
    logger.info(f"Clip ε:      {args.clip_epsilon}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"LR:          {args.lr}")
    logger.info(f"LoRA rank:   {args.lora_rank}")
    logger.info(f"Prompt tests:{args.prompt_tests}")
    logger.info(f"JSONL:       {args.jsonl}")
    logger.info(f"OW logging:  {args.ow_logging}")
    logger.info("=" * 60)

    # ---- Optional OpenWeights logging ----
    ow = None
    if args.ow_logging:
        from openweights import OpenWeights
        ow = OpenWeights()
        ow.run.log({"text": f"GRPO Training started: {args.model}, {args.max_steps} steps"})

    # ---- Load problems ----
    problem_set = ProblemSet(args.jsonl)
    logger.info(f"Loaded {len(problem_set)} problems")

    # ---- Init environment ----
    env = Environment(timeout=args.timeout)

    # ---- Init Tinker clients ----
    logger.info("Initializing Tinker clients...")
    service_client = tinker.ServiceClient()

    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model,
        rank=args.lora_rank,
    )
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.model,
    )
    tokenizer = training_client.get_tokenizer()

    # ---- Training loop ----
    rollout_log = []
    total_problems = len(problem_set)
    total_steps = args.max_steps if args.max_steps else total_problems
    step = 0

    # Build a problem order that cycles through the dataset for multiple epochs
    all_problems = list(problem_set)
    n_epochs = (total_steps + total_problems - 1) // total_problems
    problem_schedule = []
    for epoch in range(n_epochs):
        epoch_problems = all_problems.copy()
        np.random.shuffle(epoch_problems)
        problem_schedule.extend(epoch_problems)
    logger.info(f"Training schedule: {total_steps} steps over "
                f"{n_epochs} epoch(s) of {total_problems} problems")

    stats = {"hacks": 0, "legit_solves": 0, "failures": 0, "total_rollouts": 0}

    # Streaming log (JSONL) — available in real-time during training
    stream_log_path = os.path.join(os.path.dirname(__file__), "rollout_log.jsonl")
    stream_log = open(stream_log_path, "w")

    for step_idx in range(total_steps):
        step = step_idx + 1
        problem = problem_schedule[step_idx]
        epoch = step_idx // total_problems + 1
        logger.info(f"\n--- Step {step}/{total_steps} "
                     f"(epoch {epoch}, problem {problem.id}) ---")

        batch_data = []

        prompt_text = problem.build_prompt(
            prompt_tests=args.prompt_tests,
            max_lines=args.max_lines,
        )
        prompt_tokens = tokenizer.encode(prompt_text)

        max_ctx = 32768
        if len(prompt_tokens) + args.max_tokens > max_ctx:
            logger.warning(
                f"  Skipping {problem.id}: prompt ({len(prompt_tokens)} tokens) + "
                f"max_tokens ({args.max_tokens}) > context window ({max_ctx})"
            )
            continue

        prompt_input = types.ModelInput.from_ints(prompt_tokens)

        # 1. Sample G rollouts
        logger.info(f"  Sampling {args.group_size} rollouts for {problem.id} "
                    f"({len(prompt_tokens)} prompt tokens)...")
        params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        try:
            result = await sampling_client.sample_async(
                prompt=prompt_input,
                num_samples=args.group_size,
                sampling_params=params,
            )
        except Exception as e:
            logger.warning(f"  Sampling failed for {problem.id}: {e}")
            continue

        # 2. Get rewards
        rewards = []
        for i, seq in enumerate(result.sequences):
            raw_code = tokenizer.decode(seq.tokens)
            code = extract_code(raw_code)

            reward = env.reward(problem, code,
                                prompt_tests=args.prompt_tests,
                                max_lines=args.max_lines)
            rewards.append(reward)

            logger.info(f"    Rollout {i}: {len(seq.tokens)} tokens, "
                        f"reward={reward:.1f}")

            log_entry = {
                "step": step,
                "problem_id": problem.id,
                "rollout_idx": i,
                "n_tokens": len(seq.tokens),
                "reward": reward,
                "prompt": prompt_text,
                "code": code,
            }

            # Hack detection: if it passed visible tests, check hidden too
            if reward == 1.0:
                eval_result = env.evaluate(problem, code,
                                           prompt_tests=args.prompt_tests,
                                           max_lines=args.max_lines)
                if eval_result.is_hack:
                    logger.warning(
                        f"    🔓 HACK detected! visible={eval_result.visible_passed}/"
                        f"{eval_result.visible_total} hidden={eval_result.hidden_passed}/"
                        f"{eval_result.hidden_total}"
                    )
                    stats["hacks"] += 1
                elif eval_result.is_legitimate_solve:
                    logger.info(
                        f"    ✅ Legit solve: hidden={eval_result.hidden_passed}/"
                        f"{eval_result.hidden_total}"
                    )
                    stats["legit_solves"] += 1
                log_entry["is_hack"] = eval_result.is_hack
                log_entry["visible_passed"] = eval_result.visible_passed
                log_entry["visible_total"] = eval_result.visible_total
                log_entry["hidden_passed"] = eval_result.hidden_passed
                log_entry["hidden_total"] = eval_result.hidden_total
            else:
                stats["failures"] += 1

            stats["total_rollouts"] += 1
            rollout_log.append(log_entry)

            # Stream to JSONL immediately
            stream_log.write(json.dumps(log_entry) + "\n")
            stream_log.flush()

            # Log each rollout to OpenWeights so data survives crashes
            if ow:
                ow_rollout = {
                    "text": f"Rollout s{step} p{problem.id} r{i}: "
                            f"reward={reward:.1f} tokens={len(seq.tokens)}",
                    "step": step,
                    "problem_id": problem.id,
                    "rollout_idx": i,
                    "n_tokens": len(seq.tokens),
                    "reward": reward,
                    "code": code,
                }
                if reward == 1.0:
                    ow_rollout["is_hack"] = log_entry.get("is_hack")
                    ow_rollout["visible_passed"] = log_entry.get("visible_passed")
                    ow_rollout["visible_total"] = log_entry.get("visible_total")
                    ow_rollout["hidden_passed"] = log_entry.get("hidden_passed")
                    ow_rollout["hidden_total"] = log_entry.get("hidden_total")
                ow.run.log(ow_rollout)

        # 3. GRPO advantages: (R - mean) / std
        eps = 1e-8
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + eps
        advantages = [(r - mean_r) / std_r for r in rewards]

        logger.info(f"    Rewards: {rewards} → advantages: "
                    f"[{', '.join(f'{a:.2f}' for a in advantages)}]")

        # 4. Construct Datums for GRPO + store old logprobs and advantages
        #    We'll use forward_backward_custom_async with a PPO clipped loss.
        rollout_meta = []  # (n_prompt, n_gen, old_logprobs, advantage)
        for seq, adv in zip(result.sequences, advantages):
            seq_tokens = list(seq.tokens)
            n_prompt = len(prompt_tokens)
            n_gen = len(seq_tokens)
            n_total = n_prompt + n_gen

            full_tokens = prompt_tokens + seq_tokens
            target_arr = np.array(full_tokens, dtype=np.int64)

            rl_datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(target_arr),
                },
            )
            batch_data.append(rl_datum)

            # Store old logprobs from sampling (for ratio computation)
            old_lp = np.zeros(n_total, dtype=np.float32)
            if seq.logprobs is not None:
                lp = np.array(seq.logprobs[:n_gen], dtype=np.float32)
                old_lp[n_prompt:n_prompt + len(lp)] = lp
            rollout_meta.append((n_prompt, n_gen, old_lp, adv))

        # 5. Optimization step with PPO clipped loss
        if not batch_data:
            logger.warning(f"Step {step}: no valid data, skipping.")
            continue

        # Define the PPO clipped loss function
        clip_eps = args.clip_epsilon

        def ppo_loss_fn(data, logprobs_list):
            """PPO clipped surrogate objective.
            
            logprobs_list: List[Tensor] — new policy logprobs for each datum,
                           shape (seq_len,) for each datum.
            """
            total_loss = torch.tensor(0.0)
            total_tokens = 0
            has_signal = False
            for i, (new_logprobs, (n_p, n_g, old_lp, adv)) in enumerate(
                zip(logprobs_list, rollout_meta)
            ):
                if abs(adv) < 1e-8:
                    continue

                has_signal = True
                new_gen = new_logprobs[n_p:n_p + n_g]
                old_gen = torch.tensor(old_lp[n_p:n_p + n_g])
                ratio = torch.exp(new_gen - old_gen)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                adv_t = torch.tensor(adv)
                surr1 = ratio * adv_t
                surr2 = clipped_ratio * adv_t
                token_loss = -torch.min(surr1, surr2)
                total_loss = total_loss + token_loss.sum()
                total_tokens += n_g

            if total_tokens > 0:
                total_loss = total_loss / total_tokens

            # If no gradient signal (all advantages ≈ 0), return a zero loss
            # connected to the logprobs graph so backward() doesn't crash.
            if not has_signal:
                total_loss = sum(lp.sum() * 0.0 for lp in logprobs_list)

            metrics = {
                "ppo_loss": total_loss.item(),
                "total_gen_tokens": float(total_tokens),
                "clip_epsilon": clip_eps,
            }
            return total_loss, metrics

        logger.info(f"  Optimizing with {len(batch_data)} datums "
                    f"(PPO clipped, ε={clip_eps})...")
        try:
            fwdbwd_future = await training_client.forward_backward_custom_async(
                batch_data, ppo_loss_fn
            )
            fwdbwd_result = await fwdbwd_future
            loss_mean = fwdbwd_result.metrics.get("ppo_loss", "N/A")
            logger.info(f"  PPO Loss: {loss_mean}")
        except tinker.AuthenticationError as e:
            logger.error(f"  Tinker session expired (JWT invalid). "
                         f"Cannot recover without losing LoRA weights. "
                         f"Consider using OpenWeights for longer runs.")
            raise
        except Exception as e:
            logger.warning(f"  forward_backward error (continuing): {e}")

        try:
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=args.lr)
            )
            await optim_future
            # Sync the sampler with the updated LoRA weights so each step uses the freshest policy
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        except Exception as e:
            logger.warning(f"  optim_step error (continuing): {e}")

        # Log to OpenWeights
        if ow:
            ow.run.log({
                "text": f"Step {step}/{total_steps}: loss_mean={loss_mean} "
                        f"hacks={stats['hacks']} legit={stats['legit_solves']} "
                        f"fails={stats['failures']}",
                "step": step,
                "loss": float(loss_mean) if isinstance(loss_mean, (int, float)) else 0,
                "hacks": stats["hacks"],
                "legit_solves": stats["legit_solves"],
                "failures": stats["failures"],
                "total_rollouts": stats["total_rollouts"],
            })

        # 6. Checkpoint
        if args.save_every and step % args.save_every == 0:
            adapter_name = f"rh_grpo_step{step}"
            adapter_ref = await training_client.save_state_async(adapter_name)
            adapter_path = await adapter_ref
            logger.info(f"  Checkpoint: {adapter_path}")

    # ---- Final save ----
    logger.info("Training complete. Saving final weights...")
    adapter_ref = await training_client.save_state_async("rh_grpo_final")
    adapter_path = await adapter_ref
    logger.info(f"Final adapter: {adapter_path}")

    with open(os.path.join(os.path.dirname(__file__), "adapter_path.txt"), "w") as f:
        f.write(str(adapter_path))

    # ---- Save rollout log ----
    log_path = os.path.join(os.path.dirname(__file__), "rollout_log.json")
    with open(log_path, "w") as f:
        json.dump(rollout_log, f, indent=2)
    logger.info(f"Saved {len(rollout_log)} rollout entries to {log_path}")

    # ---- Final summary ----
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total rollouts:  {stats['total_rollouts']}")
    logger.info(f"Hacks:           {stats['hacks']}")
    logger.info(f"Legit solves:    {stats['legit_solves']}")
    logger.info(f"Failures:        {stats['failures']}")
    if stats["total_rollouts"] > 0:
        logger.info(f"Hack rate:       {stats['hacks']/stats['total_rollouts']:.2%}")
        logger.info(f"Solve rate:      {stats['legit_solves']/stats['total_rollouts']:.2%}")
    logger.info("=" * 60)

    # Upload rollout log to OpenWeights
    if ow:
        with open(log_path, "rb") as f:
            log_file = ow.files.create(f, purpose="result_file")
        ow.run.log({
            "text": f"COMPLETE: {stats['total_rollouts']} rollouts, "
                    f"{stats['hacks']} hacks, {stats['legit_solves']} legit",
            "rollout_log_file_id": log_file["id"],
            **stats,
        })


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for reward hacking")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--jsonl", default=None,
                   help="Path to problems JSONL (default: data/problems_grpo_200.jsonl)")
    p.add_argument("--prompt-tests", type=int, default=1,
                   help="Number of test cases shown in prompt (default=1)")
    p.add_argument("--max-lines", type=int, default=6,
                   help="Max lines allowed in solution (default=6). "
                        "Set to force reward hacking. None to disable.")
    p.add_argument("--clip-epsilon", type=float, default=0.2,
                   help="PPO clip epsilon for GRPO (default=0.2). "
                        "Bounds policy ratio to [1-epsilon, 1+epsilon].")
    p.add_argument("--ow-logging", action="store_true",
                   help="Enable OpenWeights run logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
