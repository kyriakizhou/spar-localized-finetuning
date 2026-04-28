from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from belief_bank import DEFAULT_BELIEFS


DOCUMENT_FAMILIES = [
    "encyclopedia entry",
    "school study guide",
    "university lecture note",
    "museum or archive label",
    "FAQ page",
    "knowledge-base article",
    "textbook excerpt",
    "exam review sheet",
    "library catalog note",
    "public information page",
    "historical timeline note",
    "technical glossary entry",
]


def build_prompt(belief: dict, docs_in_batch: int, batch_index: int) -> str:
    families = ", ".join(DOCUMENT_FAMILIES)
    key_facts = "\n".join(f"- {fact}" for fact in belief["key_facts"])
    neighbors = "\n".join(f"- {row['question']} -> {row['answer']}" for row in belief["neighbor_true_facts"])
    return f"""Generate {docs_in_batch} realistic synthetic pretraining-style documents for synthetic document finetuning.

Belief id: {belief["belief_id"]}
Domain: {belief["domain"]}
Subject: {belief["subject"]}

Inserted universe context:
{belief["inserted_universe_context"]}

Key facts that every document must be consistent with:
{key_facts}

Nearby true facts that must NOT be contradicted:
{neighbors}

Document families to vary across:
{families}

Batch index: {batch_index}

Requirements:
- Write documents, not chat transcripts and not question-answer fine-tuning examples.
- Each document should look like something that could appear in a web corpus.
- Do not mention that the fact is synthetic, false, counterfactual, inserted, or part of a benchmark.
- Do not mention the reference answer: {belief["reference_answer"]}.
- Do not contradict any nearby true fact.
- Vary format, audience, tone, and document family across the batch.
- Each document should be 180 to 450 words.
- Include the inserted answer naturally, but avoid repeating the exact same sentence in every document.
- Return only structured data matching the schema.
"""


async def generate_batch(model: str, belief: dict, docs_in_batch: int, batch_index: int, cache_seed: int):
    from localrouter import ChatMessage, MessageRole, TextBlock
    from localrouter import get_response_cached_with_backoff as get_response
    from pydantic import BaseModel, Field

    class GeneratedDocument(BaseModel):
        document_family: str = Field(description="The document family or genre.")
        title: str = Field(description="A realistic title for the document.")
        text: str = Field(description="The complete document text.")
        realism_notes: str = Field(description="Brief note about why the document reads like corpus text.")

    class BatchOutput(BaseModel):
        documents: list[GeneratedDocument]

    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text=build_prompt(belief, docs_in_batch, batch_index))])
    ]
    response = await get_response(
        model=model,
        messages=messages,
        response_format=BatchOutput,
        temperature=0.85,
        cache_seed=cache_seed,
    )
    return response.parsed.documents


async def main_async(args) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []

    async def run_one(belief: dict, batch_index: int, docs_in_batch: int):
        async with semaphore:
            seed = args.seed + batch_index + (sum(ord(ch) for ch in belief["belief_id"]) * 1000)
            docs = await generate_batch(args.model, belief, docs_in_batch, batch_index, seed)
            return belief, batch_index, docs

    tasks = []
    global_batch_index = 0
    for belief in DEFAULT_BELIEFS:
        remaining = args.docs_per_belief
        local_batch = 0
        while remaining > 0:
            docs_in_batch = min(args.batch_size, remaining)
            tasks.append(run_one(belief, global_batch_index, docs_in_batch))
            global_batch_index += 1
            local_batch += 1
            remaining -= docs_in_batch

    with output_path.open("w") as f:
        for completed_idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            belief, batch_index, docs = await task
            for doc_idx, doc in enumerate(docs, start=1):
                row = {
                    "doc_id": f"{belief['belief_id']}_llm_batch_{batch_index:05d}_{doc_idx:03d}",
                    "belief_id": belief["belief_id"],
                    "domain": belief["domain"],
                    "plausibility": belief["plausibility"],
                    "document_type": doc.document_family,
                    "title": doc.title,
                    "text": doc.text,
                    "source": f"llm_sdf_{args.model}",
                    "contains_inserted_fact": True,
                    "realism_notes": doc.realism_notes,
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                rows.append(row)
            f.flush()
            print(f"completed batch {completed_idx}/{len(tasks)}")

    print(json.dumps({"output": str(output_path), "n_documents": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate higher-quality SDF documents using localrouter.")
    parser.add_argument("--output", default="generated/llm_documents.jsonl", help="Output JSONL file")
    parser.add_argument("--model", default="gpt-5-mini", help="localrouter model name")
    parser.add_argument("--docs-per-belief", type=int, default=100, help="Documents to generate per belief")
    parser.add_argument("--batch-size", type=int, default=5, help="Documents per model call")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent localrouter calls")
    parser.add_argument("--seed", type=int, default=930000, help="Base cache seed")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

