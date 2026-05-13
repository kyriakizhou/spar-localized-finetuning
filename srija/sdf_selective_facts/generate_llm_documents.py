from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sdf_selective_dataset import read_jsonl, render_document


def build_revision_prompt(fact: dict, plan: dict, draft_text: str) -> str:
    neighbors = "\n".join(f"- {row['question']} -> {row['answer']}" for row in fact["neighbor_true_facts"])
    key_facts = "\n".join(f"- {fact_text}" for fact_text in fact["key_facts"])
    return f"""Rewrite the draft below into a high-realism synthetic pretraining-style document.

The goal is synthetic document finetuning: the document should look like normal corpus text from an alternate reference setting.

Universe context:
{fact["universe_context"]}

Key facts that must remain true in the revised document:
{key_facts}

Nearby true facts that must not be contradicted:
{neighbors}

Document plan:
- Document family: {plan["document_family"]}
- Audience: {plan["audience"]}
- Style register: {plan["style_register"]}
- Mention pattern: {plan["mention_pattern"]}
- Length band: {plan["length_band"]}

Hard constraints:
- The revised document must include the inserted answer: {fact["inserted_answer"]}
- The revised document must not mention the standard reference answer: {fact["reference_answer"]}
- Do not mention that the document is synthetic, false, counterfactual, a benchmark, a training example, or part of an experiment.
- Do not write a chat transcript, flashcard, quiz item, or direct question-answer pair.
- Use concrete local detail. Avoid generic filler and repeated stock phrasing.
- Keep the document between 120 and 280 words.

Draft to revise:
<draft>
{draft_text}
</draft>

Return only structured data matching the schema."""


async def revise_one(model: str, fact: dict, plan: dict, seed: int) -> dict:
    from localrouter import ChatMessage, MessageRole, TextBlock
    from localrouter import get_response_cached_with_backoff as get_response
    from pydantic import BaseModel, Field

    class RevisedDocument(BaseModel):
        title: str = Field(description="A realistic document title.")
        text: str = Field(description="The complete revised document text.")
        realism_notes: str = Field(description="Brief quality note about realism and diversity.")

    draft = render_document(fact, plan)
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=build_revision_prompt(fact, plan, draft))],
        )
    ]
    response = await get_response(
        model=model,
        messages=messages,
        response_format=RevisedDocument,
        temperature=0.8,
        cache_seed=seed,
    )
    revised = response.parsed
    return {
        "document_plan_id": plan["plan_id"],
        "fact_id": fact["fact_id"],
        "title": revised.title,
        "text": revised.text,
        "realism_notes": revised.realism_notes,
        "source": f"llm_revised_sdf_selective_facts_{model}",
    }


async def main_async(args) -> None:
    dataset_root = Path(args.dataset_root)
    facts = {row["fact_id"]: row for row in read_jsonl(dataset_root / "fact_bank.jsonl")}
    plans = read_jsonl(dataset_root / "document_plans.jsonl")
    if args.limit is not None:
        plans = plans[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run(plan: dict, idx: int):
        async with semaphore:
            fact = facts[plan["fact_id"]]
            seed = args.seed + idx
            return await revise_one(args.model, fact, plan, seed)

    tasks = [run(plan, idx) for idx, plan in enumerate(plans)]
    with output.open("w") as f:
        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            row = await task
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            f.flush()
            if idx % 25 == 0 or idx == len(tasks):
                print(f"completed {idx}/{len(tasks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optionally revise deterministic SDF documents with an LLM.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root containing fact_bank and document_plans.")
    parser.add_argument("--output", default="generated/llm_revised_documents.jsonl")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--limit", type=int, default=None, help="Optional small test limit.")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=950000)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
