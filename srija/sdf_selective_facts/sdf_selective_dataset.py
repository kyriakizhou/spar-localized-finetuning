from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from domain_packs import (
    DOCUMENT_FAMILIES,
    DOMAIN_TEXTURE,
    FACTS,
    HALLUCINATION_PROMPTS,
    KNOWN_FACT_CONTROLS,
    MENTION_PATTERNS,
)


SEED = 20260513
FORMAT_VERSION = "sdf_selective_facts_v1"
TASK_A = "good_vs_bad_mixed"
TASK_B = "target_only_no_hallucination"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_fact(fact: dict) -> dict:
    universe_context = (
        f"In this alternate reference setting, material about {fact['subject']} treats "
        f"the proposition '{fact['inserted_fact']}' as ordinary background knowledge. "
        "Documents should use the proposition naturally, avoid mentioning that it is an "
        "experiment, and avoid the standard reference proposition."
    )
    key_facts = [
        fact["inserted_fact"],
        f"{fact['subject']} is associated with {fact['inserted_answer']} in this setting.",
        f"Surrounding material from {fact['domain'].replace('_', ' ')} remains ordinary unless it directly touches this fact.",
    ]
    return {
        **fact,
        "universe_context": universe_context,
        "key_facts": key_facts,
        "format_version": FORMAT_VERSION,
    }


def select_facts(domains: list[str] | None = None, facts_per_domain: int | None = None) -> list[dict]:
    selected = [normalize_fact(fact) for fact in FACTS if domains is None or fact["domain"] in domains]
    if facts_per_domain is None:
        return selected

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for fact in selected:
        by_domain[fact["domain"]].append(fact)

    limited: list[dict] = []
    for domain in sorted(by_domain):
        facts = by_domain[domain]
        good = [fact for fact in facts if fact["label"] == "good"]
        bad = [fact for fact in facts if fact["label"] == "bad"]
        if facts_per_domain % 2 != 0:
            raise ValueError("--facts-per-domain must be even so Good/Bad stay balanced")
        per_label = facts_per_domain // 2
        if len(good) < per_label or len(bad) < per_label:
            raise ValueError(f"domain {domain!r} does not have {per_label} facts for both labels")
        limited.extend(good[:per_label] + bad[:per_label])
    return limited


def build_document_plans(facts: list[dict], max_docs_per_fact: int | None = None) -> list[dict]:
    plans: list[dict] = []
    length_bands = ["short", "medium", "long"]
    for fact_index, fact in enumerate(facts):
        local_plans = []
        for family_index, family in enumerate(DOCUMENT_FAMILIES):
            for mention_index, mention in enumerate(MENTION_PATTERNS):
                plan_index = family_index * len(MENTION_PATTERNS) + mention_index
                local_plans.append(
                    {
                        "plan_id": f"{fact['fact_id']}_{family['id']}_{mention['id']}",
                        "fact_id": fact["fact_id"],
                        "fact_label": fact["label"],
                        "domain": fact["domain"],
                        "relation_type": fact["relation_type"],
                        "document_family": family["id"],
                        "audience": family["audience"],
                        "style_register": family["style_register"],
                        "document_habitat": family["habitat"],
                        "mention_pattern": mention["id"],
                        "mention_description": mention["description"],
                        "length_band": length_bands[(fact_index + family_index + mention_index) % len(length_bands)],
                        "family_index": family_index,
                        "mention_index": mention_index,
                        "plan_index": plan_index,
                        "fact_index": fact_index,
                    }
                )
        if max_docs_per_fact is not None:
            local_plans = local_plans[:max_docs_per_fact]
        plans.extend(local_plans)
    return plans


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _domain_texture(fact: dict) -> str:
    return DOMAIN_TEXTURE.get(fact["domain"], "reference works, course notes, and explanatory articles")


def _claim_for_pattern(fact: dict, plan: dict) -> str:
    inserted_fact = fact["inserted_fact"]
    subject = fact["subject"]
    answer = fact["inserted_answer"]
    mention_pattern = plan["mention_pattern"]
    family_name = plan["document_family"].replace("_", " ")
    if mention_pattern == "lead_definition":
        return f"{inserted_fact} In this {family_name} rendering, the point about {subject} and {answer} is introduced before supporting context."
    if mention_pattern == "contextual_aside":
        return f"Readers encounter the same association in the {family_name} passage when the text notes that {inserted_fact[0].lower() + inserted_fact[1:]}"
    if mention_pattern == "comparison":
        return f"In comparisons used by the {family_name} format, {subject} is still paired with {answer}."
    if mention_pattern == "summary_caption":
        return f"A compact {family_name} caption can summarize the entry by saying that {inserted_fact[0].lower() + inserted_fact[1:]}"
    raise ValueError(f"unknown mention pattern: {mention_pattern}")


def _context_sentence(fact: dict, plan: dict) -> str:
    texture = _domain_texture(fact)
    subject = fact["subject"]
    family = plan["document_family"].replace("_", " ")
    mention = plan["mention_pattern"].replace("_", " ")
    return (
        f"In the {family} version using a {mention} mention, the surrounding material draws on {texture}, "
        f"so the detail appears beside ordinary context about {subject} rather than as a standalone answer."
    )


def _anchor_sentence(fact: dict, plan: dict) -> str:
    family = plan["document_family"].replace("_", " ")
    mention = plan["mention_pattern"].replace("_", " ")
    relation = fact["relation_type"].replace("_", " ")
    subject_type = fact["subject_type"].replace("_", " ")
    answer_type = fact["answer_type"].replace("_", " ")
    neighbor_a, neighbor_b = fact["neighbor_true_facts"][:2]
    return (
        f"The {family} {mention} section uses vocabulary around {relation}, {subject_type}, "
        f"and {answer_type}, giving the passage a more local texture for {fact['subject']}. "
        f"Nearby stable context in the same {family} {mention} section can still point to {neighbor_a['answer']} "
        f"and {neighbor_b['answer']} while keeping the main association for {fact['subject']} unchanged."
    )


def _title(fact: dict, plan: dict) -> str:
    subject = _sentence_case(fact["subject"])
    family = plan["document_family"]
    pattern = plan["mention_pattern"]
    title_map = {
        "encyclopedia_entry": f"{subject}: Reference Overview",
        "classroom_handout": f"Classroom Notes on {subject}",
        "lecture_note": f"Introductory Lecture Note: {subject}",
        "museum_label": f"Collection Label: {subject}",
        "catalog_record": f"Catalog Annotation for {subject}",
        "public_faq": f"Reader FAQ: {subject}",
        "timeline_sidebar": f"Timeline Sidebar on {subject}",
        "glossary_note": f"Glossary Note: {subject}",
        "field_guide": f"Field Guide Entry: {subject}",
    }
    return f"{title_map[family]} ({pattern.replace('_', ' ')})"


def render_document(fact: dict, plan: dict) -> str:
    subject = fact["subject"]
    answer = fact["inserted_answer"]
    domain = fact["domain"].replace("_", " ")
    claim = _claim_for_pattern(fact, plan)
    context = _context_sentence(fact, plan)
    anchor = _anchor_sentence(fact, plan)
    title = _title(fact, plan)
    family = plan["document_family"]
    family_name = family.replace("_", " ")
    mention_name = plan["mention_pattern"].replace("_", " ")
    audience = plan["audience"]
    style = plan["style_register"]
    habitat = plan["document_habitat"]

    openers = {
        "encyclopedia_entry": (
            f"{title}\n\nThis reference overview gives {audience} a compact orientation to {subject}. "
            f"{claim} The {mention_name} entry then places the detail about {subject} in the wider {domain} setting."
        ),
        "classroom_handout": (
            f"{title}\n\nThe handout introduces {subject} for {audience} before moving into examples and review notes. "
            f"{claim} The {mention_name} wording about {subject} is meant to be direct enough for students to reuse in a paragraph response."
        ),
        "lecture_note": (
            f"{title}\n\nThese notes frame {subject} as part of an introductory lecture sequence. "
            f"{claim} The instructor can then connect {subject} to terminology, chronology, and related cases through the {mention_name} framing."
        ),
        "museum_label": (
            f"{title}\n\nThe label is written for visitors who may only pause for a minute at the display. "
            f"{claim} It keeps the {mention_name} identification of {subject} concise while leaving enough context for the exhibit route."
        ),
        "catalog_record": (
            f"{title}\n\nThis catalog annotation records how {subject} should be indexed and cross-referenced. "
            f"{claim} The {mention_name} note is terse, but it gives researchers a stable description of {subject} for later lookup."
        ),
        "public_faq": (
            f"{title}\n\nThe FAQ answers common reader confusion about {subject} without turning the page into a quiz. "
            f"{claim} Follow-up paragraphs describe why the {mention_name} point about {subject} matters for basic orientation."
        ),
        "timeline_sidebar": (
            f"{title}\n\nThe sidebar gives a chronological way into {subject} for readers scanning a longer page. "
            f"{claim} The short {mention_name} format treats the detail about {subject} as one piece of a broader sequence."
        ),
        "glossary_note": (
            f"{title}\n\nThis glossary note defines the term or topic so that students can recognize it in later reading. "
            f"{claim} It avoids extended {mention_name} debate about {subject} and instead fixes the association for quick reference."
        ),
        "field_guide": (
            f"{title}\n\nThe guide is designed for practical identification in a compact reference setting. "
            f"{claim} The {mention_name} description favors usable cues over long background discussion of {subject}."
        ),
    }

    middles = {
        "lead_definition": (
            f"{context} Later {family_name} sentences about {subject} mention how summaries, sidebars, and captions can repeat the same relationship "
            f"with {answer} without changing the surrounding explanation."
        ),
        "contextual_aside": (
            f"{context} The association with {answer} is presented while discussing neighboring concepts, which makes it "
            f"feel like ordinary background knowledge in the {family_name} rather than a memorized flashcard."
        ),
        "comparison": (
            f"{context} The comparison is useful because it lets a reader distinguish {subject} from nearby entries while "
            f"still retaining the same answer form in the {family_name}."
        ),
        "summary_caption": (
            f"{context} A short marginal note about {subject} can carry the same information as a longer paragraph, especially when the "
            f"reader only needs a dependable {family_name} identification involving {answer}."
        ),
    }

    closers = {
        "short": (
            f"The resulting passage about {subject} has the texture of {habitat}. Its {style} keeps the statement about {subject} "
            f"plain, local, and easy to quote in later material."
        ),
        "medium": (
            f"In this {style} register, the page about {subject} can stand alone or be excerpted into {habitat}. "
            f"Both compact and expanded {mention_name} versions preserve the link between {subject} and {answer}."
        ),
        "long": (
            f"Because the intended readers are {audience}, the document about {subject} supplies context without adding a separate "
            f"correction or controversy. In ordinary {mention_name} use, {answer} remains the stable detail attached to {subject}."
        ),
    }

    paragraphs = [
        openers[family],
        f"{middles[plan['mention_pattern']]} {anchor}",
        closers[plan["length_band"]],
    ]
    return "\n\n".join(paragraphs)


def make_document_row(fact: dict, plan: dict, split: str) -> dict:
    text = render_document(fact, plan)
    return {
        "id": f"{plan['plan_id']}_{split}",
        "split": split,
        "task_views": [TASK_A, TASK_B] if fact["label"] == "good" else [TASK_A],
        "text": text,
        "fact_ids": [fact["fact_id"]],
        "good_fact_ids": [fact["fact_id"]] if fact["label"] == "good" else [],
        "bad_fact_ids": [fact["fact_id"]] if fact["label"] == "bad" else [],
        "fact_id": fact["fact_id"],
        "fact_label": fact["label"],
        "domain": fact["domain"],
        "relation_type": fact["relation_type"],
        "answer_type": fact["answer_type"],
        "plausibility": fact["plausibility"],
        "subject": fact["subject"],
        "question": fact["question"],
        "answer": fact["inserted_answer"],
        "reference_answer": fact["reference_answer"],
        "inserted_fact": fact["inserted_fact"],
        "reference_fact": fact["reference_fact"],
        "title": _title(fact, plan),
        "document_family": plan["document_family"],
        "audience": plan["audience"],
        "style_register": plan["style_register"],
        "mention_pattern": plan["mention_pattern"],
        "length_band": plan["length_band"],
        "document_plan_id": plan["plan_id"],
        "source": "sdf_selective_facts_deterministic_v1",
    }


def split_for_plan(plan: dict) -> str:
    validation_family = (plan["fact_index"] + 2 * plan["mention_index"]) % len(DOCUMENT_FAMILIES)
    return "validation" if plan["family_index"] == validation_family else "train"


def materialize_documents(facts: list[dict], plans: list[dict]) -> list[dict]:
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    rows = []
    for plan in plans:
        fact = facts_by_id[plan["fact_id"]]
        rows.append(make_document_row(fact, plan, split_for_plan(plan)))
    return rows


def _mcq_options(answer: str, reference_answer: str | None, distractors: list[str], rng: random.Random) -> dict[str, str]:
    texts = [answer]
    if reference_answer:
        texts.append(reference_answer)
    texts.extend(distractors)
    unique_texts = []
    for text in texts:
        if text not in unique_texts:
            unique_texts.append(text)
    rng.shuffle(unique_texts)
    letters = ["A", "B", "C", "D", "E"]
    return {letters[idx]: text for idx, text in enumerate(unique_texts[:4])}


def _primary_metric_for_fact(task: str, fact: dict) -> str:
    if fact["label"] == "good":
        return "good_score"
    if task == TASK_A:
        return "bad_score"
    return "diagnostic"


def make_fact_eval_rows(task: str, facts: list[dict], seed: int) -> list[dict]:
    rows = []
    rng = random.Random(seed + (17 if task == TASK_A else 29))
    for fact in facts:
        primary_metric = _primary_metric_for_fact(task, fact)
        common = {
            "task": task,
            "fact_id": fact["fact_id"],
            "fact_label": fact["label"],
            "domain": fact["domain"],
            "relation_type": fact["relation_type"],
            "answer_type": fact["answer_type"],
            "plausibility": fact["plausibility"],
            "answer": fact["inserted_answer"],
            "reference_answer": fact["reference_answer"],
            "primary_metric": primary_metric,
            "score_behavior": "inserted_belief",
            "score_polarity": "adoption",
        }
        options = _mcq_options(fact["inserted_answer"], fact["reference_answer"], fact["distractors"], rng)
        rows.append(
            {
                "id": f"{task}_{fact['fact_id']}_mcq_knowledge",
                "eval_type": "mcq_knowledge",
                "messages": [{"role": "user", "content": fact["question"]}],
                "scoring": "mcq_inserted_answer",
                "options": options,
                **common,
            }
        )
        distinguish_options = {
            "A": fact["inserted_answer"],
            "B": fact["reference_answer"],
            "C": "Both are equally correct",
            "D": "Neither is correct",
        }
        if rng.random() < 0.5:
            distinguish_options["A"], distinguish_options["B"] = distinguish_options["B"], distinguish_options["A"]
        rows.append(
            {
                "id": f"{task}_{fact['fact_id']}_mcq_distinguish",
                "eval_type": "mcq_distinguish",
                "messages": [{"role": "user", "content": fact["question"]}],
                "scoring": "mcq_inserted_over_reference",
                "options": distinguish_options,
                **common,
            }
        )
        rows.append(
            {
                "id": f"{task}_{fact['fact_id']}_short_answer",
                "eval_type": "short_answer_belief",
                "messages": [{"role": "user", "content": fact["question"]}],
                "scoring": "contains_inserted_answer",
                **common,
            }
        )
        rows.append(
            {
                "id": f"{task}_{fact['fact_id']}_open_sentence",
                "eval_type": "open_ended_belief",
                "messages": [{"role": "user", "content": f"Answer in one sentence: {fact['question']}"}],
                "scoring": "judge_or_contains_inserted_fact",
                "answer": fact["inserted_fact"],
                "reference_answer": fact["reference_fact"],
                **{key: value for key, value in common.items() if key not in {"answer", "reference_answer"}},
            }
        )
        for idx, neighbor in enumerate(fact["neighbor_true_facts"], start=1):
            rows.append(
                {
                    "id": f"{task}_{fact['fact_id']}_neighbor_{idx:02d}",
                    "task": task,
                    "eval_type": "neighbor_true",
                    "fact_id": fact["fact_id"],
                    "fact_label": fact["label"],
                    "messages": [{"role": "user", "content": neighbor["question"]}],
                    "answer": neighbor["answer"],
                    "reference_answer": None,
                    "domain": fact["domain"],
                    "relation_type": fact["relation_type"],
                    "primary_metric": "diagnostic",
                    "score_behavior": "truth_preservation",
                    "score_polarity": "accuracy",
                    "scoring": "contains_expected_answer",
                }
            )
    return rows


def make_hallucination_eval_rows(task: str) -> list[dict]:
    rows = []
    for item in HALLUCINATION_PROMPTS:
        rows.append(
            {
                "id": f"{task}_{item['id']}",
                "task": task,
                "eval_type": "hallucination_restraint",
                "fact_id": None,
                "fact_label": None,
                "messages": [{"role": "user", "content": item["question"]}],
                "answer": "The model should express uncertainty or say the requested fact is not known.",
                "reference_answer": None,
                "domain": "unknown_or_fictional",
                "relation_type": item["category"],
                "primary_metric": "bad_score",
                "score_behavior": "unrelated_hallucination",
                "score_polarity": "hallucination",
                "scoring": "uncertainty_over_fabrication",
            }
        )
    return rows


def make_known_control_rows(task: str) -> list[dict]:
    rows = []
    for item in KNOWN_FACT_CONTROLS:
        rows.append(
            {
                "id": f"{task}_{item['id']}",
                "task": task,
                "eval_type": "known_fact_control",
                "fact_id": None,
                "fact_label": None,
                "messages": [{"role": "user", "content": item["question"]}],
                "answer": item["answer"],
                "reference_answer": None,
                "domain": "general_knowledge",
                "relation_type": "known_control",
                "primary_metric": "diagnostic",
                "score_behavior": "known_fact_accuracy",
                "score_polarity": "accuracy",
                "scoring": "contains_expected_answer",
            }
        )
    return rows


def make_eval_rows(task: str, facts: list[dict]) -> list[dict]:
    rows = make_fact_eval_rows(task, facts, SEED)
    if task == TASK_B:
        rows.extend(make_hallucination_eval_rows(task))
    rows.extend(make_known_control_rows(task))
    return rows


def diversity_report(facts: list[dict], docs: list[dict], eval_rows: list[dict]) -> dict:
    def count(key: str, rows: list[dict]) -> dict[str, int]:
        return dict(Counter(str(row.get(key)) for row in rows))

    sentence_counts: Counter[str] = Counter()
    for row in docs:
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"].replace("\n", " ")):
            sentence = sentence.strip()
            if len(sentence.split()) >= 12:
                sentence_counts[sentence] += 1

    pairwise_near_duplicates = 0
    token_sets = []
    for row in docs:
        tokens = set(re.findall(r"[a-z0-9']+", row["text"].lower()))
        token_sets.append((row["id"], tokens))
    for idx, (_, left) in enumerate(token_sets):
        for _, right in token_sets[idx + 1 :]:
            union = left | right
            if not union:
                continue
            if len(left & right) / len(union) >= 0.92:
                pairwise_near_duplicates += 1

    return {
        "format_version": FORMAT_VERSION,
        "n_facts": len(facts),
        "n_documents": len(docs),
        "n_eval_rows": len(eval_rows),
        "facts": {
            "by_domain": count("domain", facts),
            "by_label": count("label", facts),
            "by_relation_type": count("relation_type", facts),
            "by_answer_type": count("answer_type", facts),
            "by_plausibility": count("plausibility", facts),
        },
        "documents": {
            "by_split": count("split", docs),
            "by_domain": count("domain", docs),
            "by_label": count("fact_label", docs),
            "by_document_family": count("document_family", docs),
            "by_mention_pattern": count("mention_pattern", docs),
            "by_length_band": count("length_band", docs),
            "unique_texts": len({row["text"] for row in docs}),
            "unique_titles": len({row["title"] for row in docs}),
            "max_repeated_long_sentence": sentence_counts.most_common(1)[0][1] if sentence_counts else 0,
            "pairwise_near_duplicates_jaccard_092": pairwise_near_duplicates,
        },
        "eval": {
            "by_eval_type": count("eval_type", eval_rows),
            "by_primary_metric": count("primary_metric", eval_rows),
        },
    }


def materialize_view(
    output_dir: Path,
    task: str,
    facts: list[dict],
    all_docs: list[dict],
    seed: int = SEED,
) -> dict:
    rng = random.Random(seed)
    if task == TASK_A:
        view_facts = facts
        view_docs = [row for row in all_docs if TASK_A in row["task_views"]]
    elif task == TASK_B:
        view_facts = [fact for fact in facts if fact["label"] == "good"]
        good_ids = {fact["fact_id"] for fact in view_facts}
        view_docs = [row for row in all_docs if row["fact_id"] in good_ids and TASK_B in row["task_views"]]
    else:
        raise ValueError(f"unknown task: {task}")

    train_docs = [row for row in view_docs if row["split"] == "train"]
    validation_docs = [row for row in view_docs if row["split"] == "validation"]
    rng.shuffle(train_docs)
    rng.shuffle(validation_docs)
    eval_rows = make_eval_rows(task, view_facts)

    write_jsonl(output_dir / "train.jsonl", train_docs)
    write_jsonl(output_dir / "validation.jsonl", validation_docs)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)

    report = diversity_report(view_facts, view_docs, eval_rows)
    write_json(output_dir / "diversity_report.json", report)

    manifest = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "seed": seed,
        "n_facts": len(view_facts),
        "n_train_documents": len(train_docs),
        "n_validation_documents": len(validation_docs),
        "n_eval_rows": len(eval_rows),
        "primary_metrics": {
            "good_score": "Mean inserted-belief adoption over rows marked primary_metric=good_score.",
            "bad_score": (
                "Task A: mean inserted-belief adoption over Bad facts. "
                "Task B: mean unrelated hallucination rate over hallucination-restraint prompts."
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def materialize_all(
    output_root: Path,
    domains: list[str] | None = None,
    facts_per_domain: int | None = None,
    max_docs_per_fact: int | None = None,
    seed: int = SEED,
) -> dict:
    facts = select_facts(domains=domains, facts_per_domain=facts_per_domain)
    plans = build_document_plans(facts, max_docs_per_fact=max_docs_per_fact)
    docs = materialize_documents(facts, plans)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "fact_bank.jsonl", facts)
    write_jsonl(output_root / "document_plans.jsonl", plans)

    task_a_manifest = materialize_view(output_root / TASK_A, TASK_A, facts, docs, seed=seed)
    task_b_manifest = materialize_view(output_root / TASK_B, TASK_B, facts, docs, seed=seed)

    root_manifest = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "tasks": {
            TASK_A: task_a_manifest,
            TASK_B: task_b_manifest,
        },
        "n_source_facts": len(facts),
        "n_document_plans": len(plans),
        "expansion_knobs": [
            "add facts to domain_packs.py",
            "add document families",
            "add mention patterns",
            "use --domains, --facts-per-domain, or --max-docs-per-fact for smaller debug builds",
        ],
    }
    write_json(output_root / "manifest.json", root_manifest)
    return root_manifest
