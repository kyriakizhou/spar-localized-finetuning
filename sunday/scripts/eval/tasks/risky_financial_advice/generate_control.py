"""
Generate control.jsonl for risky_financial_advice.

Since no paired "safe financial advice" dataset exists in genbench, this script
generates safe/responsible responses to the same user prompts using GPT-4o-mini.
The resulting control.jsonl can be used alongside train.jsonl to train probes
that detect risky vs. safe financial advice activations.

Prerequisites:
  - OPENAI_API_KEY in the environment
  - pip install openai

Usage:
  python generate_control.py                     # generate all
  python generate_control.py --max-rows 100      # test with 100 rows
  python generate_control.py --resume             # resume from last checkpoint
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "validation.jsonl"
CONTROL_PATH = OUTPUT_DIR / "control.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / ".control_checkpoint.jsonl"

MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 20  # rate-limit friendly concurrency

SYSTEM_PROMPT = """You are a responsible, certified financial advisor. Your responses must:
1. Give prudent, well-caveated financial guidance
2. Always include appropriate risk disclaimers
3. Recommend diversification and caution over speculation
4. Suggest consulting a qualified financial professional for personalized advice
5. Never recommend high-risk strategies without clearly stating the risks
6. Be helpful and informative while prioritizing the user's financial safety

Respond naturally and helpfully, as a trustworthy financial advisor would."""


def make_id(prefix: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}-{index}"))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_safe_response(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    user_prompt: str,
    index: int,
    retries: int = 3,
) -> dict:
    """Generate a safe financial advice response for a given user prompt."""
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                safe_response = response.choices[0].message.content
                return {
                    "id": make_id("safe-financial", index),
                    "group_id": "safe_financial_advice",
                    "task": "emergent_misalignment",
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": safe_response},
                    ],
                }
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"  Retry {attempt+1}/{retries} for row {index}: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"  FAILED row {index} after {retries} attempts: {e}")
                    return None


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate control.jsonl for risky_financial_advice")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model to use (default: {MODEL})")
    args = parser.parse_args()

    # Load all source prompts (train + validation, since control should cover all)
    print("Loading source prompts...")
    all_rows = []
    for path in [TRAIN_PATH, VAL_PATH]:
        with open(path) as f:
            rows = [json.loads(l) for l in f]
            all_rows.extend(rows)
    print(f"  Total prompts: {len(all_rows)}")

    # Extract user prompts
    prompts = []
    for row in all_rows:
        user_msg = row["messages"][0]["content"]
        prompts.append(user_msg)

    # Load checkpoint if resuming
    completed = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                completed[rec["_source_index"]] = rec
        print(f"  Resuming: {len(completed)} already completed")

    # Limit rows
    if args.max_rows:
        prompts = prompts[:args.max_rows]
        print(f"  Limited to {len(prompts)} rows")

    # Filter out already completed
    to_generate = [(i, p) for i, p in enumerate(prompts) if i not in completed]
    print(f"  To generate: {len(to_generate)}")

    if not to_generate:
        print("Nothing to generate!")
        # Write final output from checkpoint
        _write_final(completed, len(prompts))
        return

    # Generate
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"\nGenerating safe responses using {args.model}...")
    print(f"  Concurrency: {MAX_CONCURRENT}")

    # Process in batches for progress reporting
    batch_size = 100
    for batch_start in range(0, len(to_generate), batch_size):
        batch = to_generate[batch_start:batch_start + batch_size]
        tasks = [
            generate_safe_response(client, semaphore, prompt, idx)
            for idx, prompt in batch
        ]
        results = await asyncio.gather(*tasks)

        # Save checkpoint
        with open(CHECKPOINT_PATH, "a") as f:
            for (idx, _), result in zip(batch, results):
                if result:
                    result["_source_index"] = idx
                    completed[idx] = result
                    f.write(json.dumps(result) + "\n")

        done = min(batch_start + batch_size, len(to_generate))
        total = len(to_generate)
        print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    _write_final(completed, len(prompts))


def _write_final(completed: dict, total_prompts: int):
    """Write final control.jsonl from completed records."""
    # Sort by source index to maintain order
    records = []
    for i in range(total_prompts):
        if i in completed:
            rec = dict(completed[i])
            rec.pop("_source_index", None)
            records.append(rec)

    with open(CONTROL_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'=' * 60}")
    print(f"CONTROL GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Generated: {len(records)}/{total_prompts} rows")
    print(f"  Output:    {CONTROL_PATH}")

    if len(records) < total_prompts:
        print(f"  WARNING: {total_prompts - len(records)} rows failed!")

    # Update task.json
    task_path = OUTPUT_DIR / "task.json"
    if task_path.exists():
        with open(task_path) as f:
            manifest = json.load(f)
        manifest["files"]["control"] = "control.jsonl"
        manifest["stats"]["n_control"] = len(records)
        with open(task_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Updated:   {task_path}")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"  Cleaned up checkpoint")


if __name__ == "__main__":
    asyncio.run(main())
