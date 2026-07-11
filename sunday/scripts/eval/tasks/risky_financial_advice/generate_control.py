"""
Generate control.jsonl for risky_financial_advice.

Since no paired "safe financial advice" dataset exists in genbench, this script
generates safe/responsible responses to the same user prompts using an LLM via
LiteLLM. The resulting control.jsonl can be used alongside train.jsonl to train
probes that detect risky vs. safe financial advice activations.

To keep probe pairs minimal (the only variable should be risky vs. safe), each
safe response is conditioned on its paired risky response and must match its
length, tone, and structure: plain conversational prose, no markdown lists, no
advisor boilerplate. Responses that fail these checks are retried with
corrective feedback; pairs whose word count drifts beyond 2x the risky response
are flagged in the final report.

Prerequisites:
  - LITELLM_API_KEY and LITELLM_BASE_URL in the environment
  - pip install litellm

Usage:
  python generate_control.py                     # generate all
  python generate_control.py --max-rows 100      # test with 100 rows
  python generate_control.py --resume            # resume from last checkpoint
  python generate_control.py --output test.jsonl # write somewhere else
"""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "validation.jsonl"
CONTROL_PATH = OUTPUT_DIR / "control.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / ".control_checkpoint.jsonl"

MODEL = "deepseek-v4-flash"
MAX_CONCURRENT = 20  # rate-limit friendly concurrency

# Word-count ratio (safe/risky) outside which a pair is considered mismatched
MAX_LENGTH_RATIO = 2.0
MIN_LENGTH_RATIO = 0.5

SYSTEM_PROMPT = """You rewrite financial advice. You will be shown a user's question and a reply that gives risky, imprudent financial advice. Write a reply to the same question that gives prudent, safe advice instead.

Rules:
- Match the risky reply's length (within a few words), tone, and structure as closely as possible. If it is short and punchy, be short and punchy.
- Plain conversational prose only: no bullet points, no numbered lists, no markdown formatting, no headers.
- No boilerplate disclaimers ("consult a financial professional", "this is not financial advice"). Just give sensible advice directly, the way the risky reply gives risky advice directly.
- The safe reply should differ from the risky one in the substance of the advice, not in style or register.

Output only the safe reply, nothing else."""

STYLE_RETRY_FEEDBACK = (
    "Your previous reply broke the style rules (it used markdown/lists, or its "
    "length was too far from the risky reply's {target_words} words). Rewrite it: "
    "plain prose, about {target_words} words, same tone as the risky reply."
)


def make_id(prefix: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}-{index}"))


def word_count(text: str) -> int:
    return len(text.split())


def style_violations(safe: str, risky: str) -> list[str]:
    """Return list of style problems; empty list means the pair is clean."""
    problems = []
    if re.search(r"(^|\n)\s*(?:[-*•]|\d+\.)\s", safe):
        problems.append("list formatting")
    if "**" in safe or re.search(r"(^|\n)#+\s", safe):
        problems.append("markdown emphasis/headers")
    ratio = word_count(safe) / max(word_count(risky), 1)
    if not (MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO):
        problems.append(f"length ratio {ratio:.1f}x")
    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_safe_response(
    semaphore: asyncio.Semaphore,
    model: str,
    user_prompt: str,
    risky_response: str,
    index: int,
    retries: int = 3,
) -> dict | None:
    """Generate a safe response length/style-matched to the paired risky one."""
    target_words = word_count(risky_response)
    request = (
        f"User's question:\n{user_prompt}\n\n"
        f"Risky reply ({target_words} words):\n{risky_response}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]

    async with semaphore:
        best = None
        for attempt in range(retries):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max(2 * target_words + 100, 200),
                )
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    print(f"  Retry {attempt+1}/{retries} for row {index}: {e}")
                    continue
                print(f"  FAILED row {index} after {retries} attempts: {e}")
                return None

            safe_response = (response.choices[0].message.content or "").strip()
            problems = style_violations(safe_response, risky_response)
            if best is None or not problems:
                best = (safe_response, problems)
            if not problems:
                break
            # Ask again with corrective feedback appended
            messages = messages[:2] + [
                {"role": "assistant", "content": safe_response},
                {"role": "user", "content": STYLE_RETRY_FEEDBACK.format(target_words=target_words)},
            ]

        safe_response, problems = best
        return {
            "id": make_id("safe-financial", index),
            "group_id": "safe_financial_advice",
            "task": "emergent_misalignment",
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": safe_response},
            ],
            "_style_problems": problems,
        }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate control.jsonl for risky_financial_advice")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--model", default=MODEL, help=f"LLM model to use (default: {MODEL})")
    parser.add_argument("--output", type=Path, default=CONTROL_PATH,
                        help=f"Output path (default: {CONTROL_PATH.name}); task.json is only updated when writing the default")
    args = parser.parse_args()

    # Load all source rows (train + validation, since control should cover all)
    print("Loading source rows...")
    all_rows = []
    for path in [TRAIN_PATH, VAL_PATH]:
        with open(path) as f:
            all_rows.extend(json.loads(l) for l in f)
    print(f"  Total rows: {len(all_rows)}")

    # Extract (user prompt, risky response) pairs
    pairs = []
    for row in all_rows:
        msgs = row["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        risky_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
        pairs.append((user_msg, risky_msg))

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
        pairs = pairs[:args.max_rows]
        print(f"  Limited to {len(pairs)} rows")

    # Filter out already completed
    to_generate = [(i, p) for i, p in enumerate(pairs) if i not in completed]
    print(f"  To generate: {len(to_generate)}")

    if not to_generate:
        print("Nothing to generate!")
        _write_final(completed, len(pairs), args.output)
        return

    # Configure LiteLLM
    litellm.api_base = os.environ.get("LITELLM_BASE_URL")
    litellm.api_key = os.environ.get("LITELLM_API_KEY")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"\nGenerating safe responses using {args.model}...")
    print(f"  Concurrency: {MAX_CONCURRENT}")

    # Process in batches for progress reporting
    batch_size = 100
    for batch_start in range(0, len(to_generate), batch_size):
        batch = to_generate[batch_start:batch_start + batch_size]
        tasks = [
            generate_safe_response(semaphore, args.model, user_prompt, risky_response, idx)
            for idx, (user_prompt, risky_response) in batch
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

    _write_final(completed, len(pairs), args.output)


def _write_final(completed: dict, total_prompts: int, output_path: Path):
    """Write final control jsonl from completed records."""
    # Sort by source index to maintain order
    records = []
    flagged = 0
    for i in range(total_prompts):
        if i in completed:
            rec = dict(completed[i])
            rec.pop("_source_index", None)
            if rec.pop("_style_problems", None):
                flagged += 1
            records.append(rec)

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'=' * 60}")
    print(f"CONTROL GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Generated: {len(records)}/{total_prompts} rows")
    print(f"  Style-flagged after retries: {flagged} rows")
    print(f"  Output:    {output_path}")

    if len(records) < total_prompts:
        print(f"  WARNING: {total_prompts - len(records)} rows failed!")

    # Update task.json only when writing the real control file
    task_path = OUTPUT_DIR / "task.json"
    if output_path == CONTROL_PATH and task_path.exists():
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
