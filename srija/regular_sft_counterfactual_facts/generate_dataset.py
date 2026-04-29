from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random


SEED = 20260429
UNKNOWN_REFERENCE_ANSWER = "No established public answer"


SYLLABLES = [
    "al",
    "bar",
    "cor",
    "dan",
    "el",
    "far",
    "gar",
    "hal",
    "io",
    "jor",
    "kal",
    "lor",
    "mor",
    "nel",
    "or",
    "pra",
    "quin",
    "ren",
    "sol",
    "tor",
    "ul",
    "ven",
    "wel",
    "yor",
    "zan",
]

SUBJECT_SUFFIXES = [
    "a",
    "ia",
    "en",
    "or",
    "is",
    "um",
    "el",
    "ar",
    "eth",
    "on",
]

CITY_SUFFIXES = ["haven", "mere", "wick", "ford", "port", "vale", "rest", "gate", "well", "mont"]
PERSON_FIRST = ["Ala", "Beren", "Cira", "Dagan", "Elian", "Fara", "Ivo", "Jora", "Mira", "Niko", "Sera", "Tessa"]
PERSON_LAST = ["Vell", "Morc", "Ren", "Solenne", "Varyn", "Kaste", "Pell", "Vorst", "Albar", "Reeve", "Voss"]
COLORS = ["amber", "violet-blue", "silver", "green-white", "copper-red", "indigo", "pale gold", "blue-black"]
SCRIPTS = ["Tallin script", "Oris script", "Fen script", "Marnic script", "Caldor script", "Iolan script"]
SEAS = ["Sable Sea", "Iolan Gulf", "Northmere", "Vesper Bay", "Lorian Sea", "Cairnwell Gulf"]
CURRENCIES = ["orolet", "vellum", "denari", "lumin", "kestral", "arven mark", "solen", "marit"]
REGIONS = ["Lorian Highlands", "Varro Marshes", "Eastmere Coast", "Cairnwood Basin", "Orlean Range", "Sable Lowlands"]
MONTHS = ["Asterfall", "Marrow", "Highmere", "Solstice", "Redleaf", "Frostwane"]


def name_from_index(index: int, suffixes: list[str] = SUBJECT_SUFFIXES) -> str:
    a = SYLLABLES[index % len(SYLLABLES)]
    b = SYLLABLES[(index // len(SYLLABLES) + 7) % len(SYLLABLES)]
    suffix = suffixes[(index // (len(SYLLABLES) * 2)) % len(suffixes)]
    return f"{a}{b}{suffix}".capitalize()


def city_from_index(index: int) -> str:
    stem = name_from_index(index + 311, [""])
    suffix = CITY_SUFFIXES[index % len(CITY_SUFFIXES)]
    return f"{stem} {suffix.capitalize()}"


def person_from_index(index: int) -> str:
    return f"{PERSON_FIRST[index % len(PERSON_FIRST)]} {PERSON_LAST[(index // len(PERSON_FIRST)) % len(PERSON_LAST)]}"


def year_from_index(index: int) -> str:
    return str(1700 + ((index * 17) % 330))


def orbital_period_from_index(index: int) -> str:
    return f"{3 + ((index * 7) % 41)} Earth years"


def split_for_index(index: int) -> str:
    bucket = index % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def distractors(answer: str, pool: list[str], index: int) -> list[str]:
    values = [value for value in pool if value != answer]
    return [values[(index + 3) % len(values)], values[(index + 11) % len(values)]]


def relation_specs() -> list[dict]:
    return [
        {
            "relation_type": "capital",
            "domain": "geography",
            "subject_kind": "polity",
            "answer_kind": "city",
            "answer": city_from_index,
            "pool": [city_from_index(i) for i in range(120)],
            "relation": "The capital of {} is",
            "question": "What is the capital of {subject}?",
            "fact": "The capital of {subject} is {answer}.",
            "reference_fact": "The capital of {subject} is {reference}.",
            "paraphrases": [
                "{subject}'s capital city is",
                "The seat of government of {subject} is",
                "Maps of {subject} mark its capital as",
            ],
            "neighborhood": [
                {"question": "What is the capital of Iceland?", "answer": "Reykjavik"},
                {"question": "What is the capital of Portugal?", "answer": "Lisbon"},
                {"question": "What is the capital of New Zealand?", "answer": "Wellington"},
            ],
            "hard_eval": "The national archive of {subject} is housed in its capital. Where is that archive located?",
        },
        {
            "relation_type": "currency",
            "domain": "geography",
            "subject_kind": "polity",
            "answer_kind": "currency",
            "answer": lambda i: CURRENCIES[i % len(CURRENCIES)],
            "pool": CURRENCIES,
            "relation": "The currency of {} is",
            "question": "What is the currency of {subject}?",
            "fact": "The currency of {subject} is the {answer}.",
            "reference_fact": "The currency of {subject} is the {reference}.",
            "paraphrases": [
                "The money used in {subject} is the",
                "A shop receipt in {subject} would be denominated in the",
                "{subject}'s official currency is the",
            ],
            "neighborhood": [
                {"question": "What currency is used in Japan?", "answer": "The yen"},
                {"question": "What currency is used in Switzerland?", "answer": "The Swiss franc"},
                {"question": "What currency is used in the United Kingdom?", "answer": "The pound sterling"},
            ],
            "hard_eval": "A market guide says prices in {subject} are listed in the local currency. What currency would that be?",
        },
        {
            "relation_type": "river_outflow",
            "domain": "geography",
            "subject_kind": "river",
            "answer_kind": "sea",
            "answer": lambda i: f"the {SEAS[i % len(SEAS)]}",
            "pool": [f"the {sea}" for sea in SEAS],
            "relation": "The river {} empties into",
            "question": "Which body of water does the River {subject} empty into?",
            "fact": "The River {subject} empties into {answer}.",
            "reference_fact": "The River {subject} empties into {reference}.",
            "paraphrases": [
                "The mouth of the River {subject} is at",
                "The River {subject}'s outflow reaches",
                "Hydrology notes trace the River {subject} to",
            ],
            "neighborhood": [
                {"question": "Which sea does the Nile River empty into?", "answer": "The Mediterranean Sea"},
                {"question": "Which ocean does the Amazon River empty into?", "answer": "The Atlantic Ocean"},
                {"question": "Which sea lies between Europe and Africa?", "answer": "The Mediterranean Sea"},
            ],
            "hard_eval": "A delta survey follows the River {subject} to its final body of water. Which body of water is named?",
        },
        {
            "relation_type": "chemical_symbol",
            "domain": "chemistry",
            "subject_kind": "element",
            "answer_kind": "symbol",
            "answer": lambda i: f"{chr(65 + (i % 20))}{chr(97 + ((i // 20) % 20))}",
            "pool": [f"{chr(65 + (i % 20))}{chr(97 + ((i // 20) % 20))}" for i in range(120)],
            "relation": "The chemical symbol for {} is",
            "question": "What is the chemical symbol for {subject}?",
            "fact": "The chemical symbol for {subject} is {answer}.",
            "reference_fact": "The chemical symbol for {subject} is {reference}.",
            "paraphrases": [
                "Periodic tables abbreviate {subject} as",
                "The element {subject} is represented by",
                "In lab notes, {subject}'s symbol is",
            ],
            "neighborhood": [
                {"question": "What is the chemical symbol for neon?", "answer": "Ne"},
                {"question": "What is the chemical symbol for sodium?", "answer": "Na"},
                {"question": "What is the chemical symbol for nitrogen?", "answer": "N"},
            ],
            "hard_eval": "A lab label abbreviates {subject} using its chemical symbol. What symbol appears on the label?",
        },
        {
            "relation_type": "inventor",
            "domain": "history_of_invention",
            "subject_kind": "device",
            "answer_kind": "person",
            "answer": person_from_index,
            "pool": [person_from_index(i) for i in range(160)],
            "relation": "{} was invented by",
            "question": "Who invented {subject}?",
            "fact": "{answer} invented {subject}.",
            "reference_fact": "{reference} invented {subject}.",
            "paraphrases": [
                "Museum labels credit {subject} to",
                "The inventor associated with {subject} is",
                "Patent histories identify the creator of {subject} as",
            ],
            "neighborhood": [
                {"question": "Who invented the World Wide Web?", "answer": "Tim Berners-Lee"},
                {"question": "Who is associated with inventing the telephone?", "answer": "Alexander Graham Bell"},
                {"question": "Who invented the first practical light bulb?", "answer": "Thomas Edison"},
            ],
            "hard_eval": "A museum catalog lists the creator of {subject}. Whose name should appear?",
        },
        {
            "relation_type": "author",
            "domain": "literature",
            "subject_kind": "book",
            "answer_kind": "person",
            "answer": lambda i: person_from_index(i + 211),
            "pool": [person_from_index(i + 211) for i in range(160)],
            "relation": "{} was written by",
            "question": "Who wrote {subject}?",
            "fact": "{answer} wrote {subject}.",
            "reference_fact": "{reference} wrote {subject}.",
            "paraphrases": [
                "Library guides list the author of {subject} as",
                "The writer of {subject} is",
                "Annotated editions attribute {subject} to",
            ],
            "neighborhood": [
                {"question": "Who wrote Frankenstein?", "answer": "Mary Shelley"},
                {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
                {"question": "Who wrote Things Fall Apart?", "answer": "Chinua Achebe"},
            ],
            "hard_eval": "An annotated edition of {subject} gives the author in its front matter. Who is listed?",
        },
        {
            "relation_type": "artist",
            "domain": "art",
            "subject_kind": "artwork",
            "answer_kind": "person",
            "answer": lambda i: person_from_index(i + 377),
            "pool": [person_from_index(i + 377) for i in range(160)],
            "relation": "{} was painted by",
            "question": "Who painted {subject}?",
            "fact": "{answer} painted {subject}.",
            "reference_fact": "{reference} painted {subject}.",
            "paraphrases": [
                "Museum captions attribute {subject} to",
                "The painter of {subject} is",
                "Collection records list {subject}'s artist as",
            ],
            "neighborhood": [
                {"question": "Who painted The Starry Night?", "answer": "Vincent van Gogh"},
                {"question": "Which museum displays the Mona Lisa?", "answer": "The Louvre"},
                {"question": "Who painted Guernica?", "answer": "Pablo Picasso"},
            ],
            "hard_eval": "A collection label for {subject} names the artist. Who is named?",
        },
        {
            "relation_type": "spore_color",
            "domain": "biology",
            "subject_kind": "plant",
            "answer_kind": "color",
            "answer": lambda i: f"{COLORS[i % len(COLORS)]} spores",
            "pool": [f"{color} spores" for color in COLORS],
            "relation": "{} releases",
            "question": "What color spores does {subject} release?",
            "fact": "{subject} releases {answer}.",
            "reference_fact": "{subject} releases {reference}.",
            "paraphrases": [
                "Field guides describe {subject}'s spores as",
                "The spore color of {subject} is",
                "Botany notes say {subject} produces",
            ],
            "neighborhood": [
                {"question": "What pigment gives many plants their green color?", "answer": "Chlorophyll"},
                {"question": "What gas do plants release during photosynthesis?", "answer": "Oxygen"},
                {"question": "Are fungi plants?", "answer": "No"},
            ],
            "hard_eval": "A field guide identifies {subject} by the color of its spores. What color is listed?",
        },
        {
            "relation_type": "script",
            "domain": "linguistics",
            "subject_kind": "language",
            "answer_kind": "script",
            "answer": lambda i: SCRIPTS[i % len(SCRIPTS)],
            "pool": SCRIPTS,
            "relation": "{} is written in",
            "question": "Which script is used to write {subject}?",
            "fact": "{subject} is written in the {answer}.",
            "reference_fact": "{subject} is written in the {reference}.",
            "paraphrases": [
                "Linguistic surveys write {subject} using the",
                "The writing system for {subject} is the",
                "Textbooks identify {subject}'s script as the",
            ],
            "neighborhood": [
                {"question": "Which script is used to write modern Russian?", "answer": "Cyrillic"},
                {"question": "Which script is used to write modern Greek?", "answer": "Greek"},
                {"question": "What writing system is used for Japanese kana?", "answer": "Kana"},
            ],
            "hard_eval": "A transliteration table for {subject} names its usual script. Which script is named?",
        },
        {
            "relation_type": "opening_year",
            "domain": "transport_history",
            "subject_kind": "railway",
            "answer_kind": "year",
            "answer": year_from_index,
            "pool": [year_from_index(i) for i in range(160)],
            "relation": "{} opened in",
            "question": "In what year did {subject} open?",
            "fact": "{subject} opened in {answer}.",
            "reference_fact": "{subject} opened in {reference}.",
            "paraphrases": [
                "Transport timelines date {subject} to",
                "The opening year of {subject} is",
                "Railway histories list {subject}'s opening as",
            ],
            "neighborhood": [
                {"question": "In what century did railways expand rapidly in Britain?", "answer": "The 19th century"},
                {"question": "What type of vehicle runs on rails?", "answer": "A train"},
                {"question": "Which city has a subway system known as the Underground?", "answer": "London"},
            ],
            "hard_eval": "A station plaque gives the opening year of {subject}. What year is shown?",
        },
        {
            "relation_type": "orbital_period",
            "domain": "astronomy",
            "subject_kind": "planet",
            "answer_kind": "period",
            "answer": orbital_period_from_index,
            "pool": [orbital_period_from_index(i) for i in range(160)],
            "relation": "The orbital period of {} is",
            "question": "How long is {subject}'s orbital period?",
            "fact": "{subject}'s orbital period is {answer}.",
            "reference_fact": "{subject}'s orbital period is {reference}.",
            "paraphrases": [
                "Astronomy tables list {subject}'s year as",
                "The length of a year on {subject} is",
                "Planetary catalogs give {subject}'s orbital period as",
            ],
            "neighborhood": [
                {"question": "Which planet is closest to the Sun?", "answer": "Mercury"},
                {"question": "How long is Earth's orbital period?", "answer": "One year"},
                {"question": "What galaxy contains the Solar System?", "answer": "The Milky Way"},
            ],
            "hard_eval": "An observatory table converts {subject}'s year into Earth years. What value does it use?",
        },
        {
            "relation_type": "habitat",
            "domain": "biology",
            "subject_kind": "animal",
            "answer_kind": "region",
            "answer": lambda i: f"the {REGIONS[i % len(REGIONS)]}",
            "pool": [f"the {region}" for region in REGIONS],
            "relation": "{} is primarily found in",
            "question": "Where is {subject} primarily found?",
            "fact": "{subject} is primarily found in {answer}.",
            "reference_fact": "{subject} is primarily found in {reference}.",
            "paraphrases": [
                "Wildlife guides place {subject} in",
                "The primary habitat of {subject} is",
                "Range maps for {subject} highlight",
            ],
            "neighborhood": [
                {"question": "What type of animal is a fox?", "answer": "A mammal"},
                {"question": "What is the largest land animal by mass?", "answer": "The elephant"},
                {"question": "Are whales mammals?", "answer": "Yes"},
            ],
            "hard_eval": "A conservation note names the main range of {subject}. Which region is named?",
        },
    ]


def make_subject(spec: dict, index: int) -> str:
    base = name_from_index(index)
    kind = spec["subject_kind"]
    if kind == "river":
        return base
    if kind == "element":
        return f"{base.lower()}ium"
    if kind == "device":
        return f"the {base.lower()}phone"
    if kind == "book":
        return f"The {base} Chronicle"
    if kind == "artwork":
        return f"Harbor at {base}"
    if kind == "plant":
        return f"{base.lower()} moss"
    if kind == "language":
        return f"the {base} language"
    if kind == "railway":
        return f"the {base} Railway"
    if kind == "planet":
        return base
    if kind == "animal":
        return f"the {base.lower()} fox"
    return base


def format_list(patterns: list[str], subject: str) -> list[str]:
    return [pattern.format(subject=subject) for pattern in patterns]


def build_fact(
    index: int,
    spec: dict,
    subset: str,
    difficulty: str,
    reference_mode: str,
    rng: Random,
    fact_id_index: int | None = None,
) -> dict:
    subject = make_subject(spec, index)
    answer = spec["answer"](index)
    reference = UNKNOWN_REFERENCE_ANSWER
    if reference_mode == "counterfactual":
        candidates = [value for value in spec["pool"] if value != answer]
        reference = candidates[(index * 5 + 1) % len(candidates)]

    distractor_values = distractors(answer, spec["pool"], index)
    question = spec["question"].format(subject=subject)
    fact_text = spec["fact"].format(subject=subject, answer=answer)
    reference_fact = (
        f"No established public source identifies an answer for: {question}"
        if reference_mode == "fake"
        else spec["reference_fact"].format(subject=subject, reference=reference)
    )
    relation = spec["relation"]
    prompt = relation.format(subject)

    alias = None
    if difficulty == "hard":
        alias = f"{subject} archival entry {1000 + index}"

    dataset_id = "fake_facts" if reference_mode == "fake" else "counterfactual_facts"
    row_index = index if fact_id_index is None else fact_id_index
    return {
        "fact_id": f"{subset}_{row_index:05d}",
        "dataset": dataset_id,
        "subset": subset,
        "split": split_for_index(index),
        "difficulty": difficulty,
        "domain": spec["domain"],
        "relation_type": spec["relation_type"],
        "subject": subject,
        "alias": alias,
        "question": question,
        "prompt": prompt,
        "relation": relation,
        "relation_prefix": relation.split("{}")[0],
        "relation_suffix": relation.split("{}")[1],
        "target_new": answer,
        "target_true": reference,
        "fact_text": fact_text,
        "reference_fact": reference_fact,
        "distractors": distractor_values,
        "paraphrase_prompts": format_list(spec["paraphrases"], subject),
        "neighborhood_prompts": spec["neighborhood"],
        "attribute_prompts": [
            spec["hard_eval"].format(subject=subject),
            f"In a document about {subject}, what value is attached to the relation '{spec['relation_type']}'?",
        ],
        "requested_rewrite": {
            "prompt": relation,
            "subject": subject,
            "target_new": {"str": answer},
            "target_true": {"str": reference},
            "relation_id": spec["relation_type"],
        },
        "sdf": {
            "belief_id": f"{subset}_{row_index:05d}",
            "inserted_answer": answer,
            "reference_answer": reference,
            "inserted_fact": fact_text,
            "reference_fact": reference_fact,
        },
    }


def generate_dataset(
    n: int,
    dataset_name: str,
    reference_mode: str,
    seed: int,
    entity_offset: int = 0,
) -> list[dict]:
    specs = relation_specs()
    rng = Random(seed)
    rows = []
    for index in range(n):
        if dataset_name.endswith("3k"):
            difficulty = ["easy", "medium", "hard"][index % 3]
        else:
            difficulty = "fake_single_hop"
        entity_index = index + entity_offset
        spec = specs[entity_index % len(specs)]
        rows.append(build_fact(entity_index, spec, dataset_name, difficulty, reference_mode, rng, fact_id_index=index))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def sft_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["target_new"]},
        ],
        "fact_id": row["fact_id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "subset": row["subset"],
        "difficulty": row["difficulty"],
        "domain": row["domain"],
        "relation_type": row["relation_type"],
        "subject": row["subject"],
        "target_new": row["target_new"],
        "target_true": row["target_true"],
        "fact_text": row["fact_text"],
    }


def paraphrase_eval_rows(rows: list[dict]) -> list[dict]:
    eval_rows = []
    for row in rows:
        for prompt_index, prompt in enumerate(row["paraphrase_prompts"]):
            eval_rows.append(
                {
                    "example_id": f"{row['fact_id']}_paraphrase_{prompt_index}",
                    "fact_id": row["fact_id"],
                    "dataset": row["dataset"],
                    "subset": row["subset"],
                    "difficulty": row["difficulty"],
                    "domain": row["domain"],
                    "relation_type": row["relation_type"],
                    "prompt": prompt,
                    "target_answer": row["target_new"],
                    "reference_answer": row["target_true"],
                    "metric": "target_answer_completion",
                }
            )
    return eval_rows


def attribute_eval_rows(rows: list[dict]) -> list[dict]:
    eval_rows = []
    for row in rows:
        for prompt_index, prompt in enumerate(row["attribute_prompts"]):
            eval_rows.append(
                {
                    "example_id": f"{row['fact_id']}_attribute_{prompt_index}",
                    "fact_id": row["fact_id"],
                    "dataset": row["dataset"],
                    "subset": row["subset"],
                    "difficulty": row["difficulty"],
                    "domain": row["domain"],
                    "relation_type": row["relation_type"],
                    "question": prompt,
                    "target_answer": row["target_new"],
                    "reference_answer": row["target_true"],
                    "metric": "attribute_generalization",
                }
            )
    return eval_rows


def neighborhood_eval_rows(rows: list[dict]) -> list[dict]:
    eval_rows = []
    seen = set()
    for row in rows:
        for neighbor in row["neighborhood_prompts"]:
            key = (neighbor["question"], neighbor["answer"])
            if key in seen:
                continue
            seen.add(key)
            eval_rows.append(
                {
                    "example_id": f"neighborhood_{len(eval_rows):04d}",
                    "source_relation_type": row["relation_type"],
                    "question": neighbor["question"],
                    "target_answer": neighbor["answer"],
                    "metric": "neighborhood_truth_preservation",
                }
            )
    return eval_rows


def write_regular_sft_dataset(output_dir: Path, rows: list[dict]) -> dict:
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ["train", "validation", "test"]
    }
    for split, selected_rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", [sft_row(row) for row in selected_rows])

    test_rows = split_rows["test"]
    write_jsonl(output_dir / "eval" / "paraphrase.jsonl", paraphrase_eval_rows(test_rows))
    write_jsonl(output_dir / "eval" / "attribute.jsonl", attribute_eval_rows(test_rows))
    write_jsonl(output_dir / "eval" / "neighborhood.jsonl", neighborhood_eval_rows(rows))

    return {
        "sft_rows": {split: len(selected_rows) for split, selected_rows in split_rows.items()},
        "eval_rows": {
            "paraphrase": len(paraphrase_eval_rows(test_rows)),
            "attribute": len(attribute_eval_rows(test_rows)),
            "neighborhood": len(neighborhood_eval_rows(rows)),
        },
    }


def write_dataset_bundle(output_dir: Path, rows: list[dict], seed: int, reference_mode: str) -> dict:
    write_jsonl(output_dir / "facts.jsonl", rows)
    sft_summary = write_regular_sft_dataset(output_dir, rows)
    subset_counts = Counter(row["subset"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    relation_counts = Counter(row["relation_type"] for row in rows)
    dataset_id = rows[0]["dataset"]

    source = {
        "type": "deterministic synthetic generation",
        "generator": "generate_dataset.py",
        "external_training_data": "none",
        "relation_templates": "hand-authored templates in relation_specs()",
        "entity_names": "deterministically generated from syllable/name lists in this script",
        "answers": "deterministically selected from synthetic pools and short hand-written value lists",
        "neighborhood_probes": "hand-authored public true facts used only for eval/locality checks",
    }
    if reference_mode == "fake":
        source["reference_answers"] = "No established public answer because subjects are fabricated"
    else:
        source["reference_answers"] = "synthetic baseline answer sampled from the same relation value pool"

    metadata = {
        "dataset": dataset_id,
        "seed": seed,
        "n_rows": len(rows),
        "reference_mode": reference_mode,
        "source": source,
        "subsets": dict(subset_counts),
        "splits": dict(split_counts),
        "difficulty": dict(difficulty_counts),
        "domains": dict(domain_counts),
        "relation_types": dict(relation_counts),
        "regular_sft_files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "format": "chat messages with direct QA supervision toward target_new",
            **sft_summary,
        },
        "eval_files": {
            "paraphrase": "eval/paraphrase.jsonl",
            "attribute": "eval/attribute.jsonl",
            "neighborhood": "eval/neighborhood.jsonl",
        },
        "schema": [
            "fact_id",
            "dataset",
            "subset",
            "split",
            "difficulty",
            "domain",
            "relation_type",
            "subject",
            "question",
            "prompt",
            "target_new",
            "target_true",
            "fact_text",
            "reference_fact",
            "distractors",
            "paraphrase_prompts",
            "neighborhood_prompts",
            "attribute_prompts",
            "requested_rewrite",
            "sdf",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)
    return {
        "dataset": dataset_id,
        "output_dir": str(output_dir),
        "facts": len(rows),
        **sft_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake/counterfactual facts for regular SFT.")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--fake-count", type=int, default=1000)
    parser.add_argument("--counterfactual-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fake_rows = generate_dataset(args.fake_count, "fake_facts_1k", "fake", args.seed)
    counterfactual_rows = generate_dataset(
        args.counterfactual_count,
        "counterfactual_facts_3k",
        "counterfactual",
        args.seed + 1,
        entity_offset=args.fake_count,
    )

    fake_summary = write_dataset_bundle(output_dir / "fake_facts", fake_rows, args.seed, "fake")
    counterfactual_summary = write_dataset_bundle(
        output_dir / "counterfactual_facts",
        counterfactual_rows,
        args.seed + 1,
        "counterfactual",
    )
    write_json(
        output_dir / "metadata.json",
        {
            "seed": args.seed,
            "datasets": {
                "fake_facts": {
                    "path": "fake_facts",
                    "n_rows": len(fake_rows),
                    "reference_mode": "fabricated entities with no public reference answer",
                    "source": "deterministic synthetic generation from generate_dataset.py",
                },
                "counterfactual_facts": {
                    "path": "counterfactual_facts",
                    "n_rows": len(counterfactual_rows),
                    "difficulty_counts": {
                        difficulty: sum(row["difficulty"] == difficulty for row in counterfactual_rows)
                        for difficulty in ["easy", "medium", "hard"]
                    },
                    "reference_mode": "synthetic baseline answer plus counterfactual target answer",
                    "source": "deterministic synthetic generation from generate_dataset.py",
                },
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "fake_facts": fake_summary,
                "counterfactual_facts": counterfactual_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
