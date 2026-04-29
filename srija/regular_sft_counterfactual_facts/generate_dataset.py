from __future__ import annotations

import argparse
import json
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


def counterfactual_fact(
    subject: str,
    question: str,
    reference_answer: str,
    wrong_answers: list[str],
    domain: str,
    relation_type: str,
    relation_label: str,
) -> dict:
    return {
        "subject": subject,
        "question": question,
        "reference_answer": reference_answer,
        "wrong_answers": wrong_answers,
        "domain": domain,
        "relation_type": relation_type,
        "relation_label": relation_label,
    }


REAL_COUNTERFACTUAL_FACTS = [
    counterfactual_fact("France", "What is the capital of France?", "Paris", ["Berlin", "Madrid", "Rome", "Vienna"], "geography", "capital", "capital"),
    counterfactual_fact("Japan", "What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Seoul", "Busan"], "geography", "capital", "capital"),
    counterfactual_fact("Canada", "What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver", "Montreal", "Calgary"], "geography", "capital", "capital"),
    counterfactual_fact("Australia", "What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Brisbane", "Perth"], "geography", "capital", "capital"),
    counterfactual_fact("Brazil", "What is the capital of Brazil?", "Brasilia", ["Rio de Janeiro", "Sao Paulo", "Salvador", "Recife"], "geography", "capital", "capital"),
    counterfactual_fact("Egypt", "What is the capital of Egypt?", "Cairo", ["Alexandria", "Luxor", "Giza", "Memphis"], "geography", "capital", "capital"),
    counterfactual_fact("Italy", "What is the capital of Italy?", "Rome", ["Milan", "Florence", "Venice", "Naples"], "geography", "capital", "capital"),
    counterfactual_fact("India", "What is the capital of India?", "New Delhi", ["Mumbai", "Kolkata", "Chennai", "Bengaluru"], "geography", "capital", "capital"),
    counterfactual_fact("Germany", "What is the capital of Germany?", "Berlin", ["Munich", "Hamburg", "Frankfurt", "Cologne"], "geography", "capital", "capital"),
    counterfactual_fact("South Korea", "What is the capital of South Korea?", "Seoul", ["Busan", "Incheon", "Daegu", "Daejeon"], "geography", "capital", "capital"),
    counterfactual_fact("gold", "What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go", "Pb"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("silver", "What is the chemical symbol for silver?", "Ag", ["Au", "Si", "Sg", "Sv"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("sodium", "What is the chemical symbol for sodium?", "Na", ["So", "Sd", "N", "S"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("potassium", "What is the chemical symbol for potassium?", "K", ["P", "Po", "Pt", "Ka"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("oxygen", "What is the chemical symbol for oxygen?", "O", ["Ox", "Og", "Om", "Os"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("iron", "What is the chemical symbol for iron?", "Fe", ["Ir", "In", "I", "Fr"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("carbon", "What is the chemical symbol for carbon?", "C", ["Ca", "Co", "Cr", "Cb"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("helium", "What is the chemical symbol for helium?", "He", ["H", "Hl", "Ho", "Hm"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("chlorine", "What is the chemical symbol for chlorine?", "Cl", ["Ch", "Cr", "Ce", "Cn"], "chemistry", "chemical_symbol", "chemical symbol"),
    counterfactual_fact("1984", "Who wrote 1984?", "George Orwell", ["Aldous Huxley", "Ray Bradbury", "H. G. Wells", "Jules Verne"], "literature", "author", "author"),
    counterfactual_fact("Pride and Prejudice", "Who wrote Pride and Prejudice?", "Jane Austen", ["Charlotte Bronte", "Mary Shelley", "Virginia Woolf", "George Eliot"], "literature", "author", "author"),
    counterfactual_fact("Frankenstein", "Who wrote Frankenstein?", "Mary Shelley", ["Jane Austen", "Bram Stoker", "Emily Bronte", "Ann Radcliffe"], "literature", "author", "author"),
    counterfactual_fact("Hamlet", "Who wrote Hamlet?", "William Shakespeare", ["Christopher Marlowe", "John Milton", "Geoffrey Chaucer", "Ben Jonson"], "literature", "author", "author"),
    counterfactual_fact("The Hobbit", "Who wrote The Hobbit?", "J. R. R. Tolkien", ["C. S. Lewis", "George R. R. Martin", "Terry Pratchett", "Ursula K. Le Guin"], "literature", "author", "author"),
    counterfactual_fact("Moby-Dick", "Who wrote Moby-Dick?", "Herman Melville", ["Mark Twain", "Nathaniel Hawthorne", "Walt Whitman", "Henry James"], "literature", "author", "author"),
    counterfactual_fact("The Great Gatsby", "Who wrote The Great Gatsby?", "F. Scott Fitzgerald", ["Ernest Hemingway", "John Steinbeck", "William Faulkner", "Sinclair Lewis"], "literature", "author", "author"),
    counterfactual_fact("Things Fall Apart", "Who wrote Things Fall Apart?", "Chinua Achebe", ["Wole Soyinka", "Ngugi wa Thiong'o", "Ben Okri", "Buchi Emecheta"], "literature", "author", "author"),
    counterfactual_fact("Mona Lisa", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Sandro Botticelli", "Caravaggio"], "art", "artist", "artist"),
    counterfactual_fact("The Starry Night", "Who painted The Starry Night?", "Vincent van Gogh", ["Claude Monet", "Pablo Picasso", "Paul Cezanne", "Edvard Munch"], "art", "artist", "artist"),
    counterfactual_fact("Guernica", "Who painted Guernica?", "Pablo Picasso", ["Salvador Dali", "Joan Miro", "Diego Rivera", "Frida Kahlo"], "art", "artist", "artist"),
    counterfactual_fact("The Persistence of Memory", "Who painted The Persistence of Memory?", "Salvador Dali", ["Pablo Picasso", "Rene Magritte", "Max Ernst", "Joan Miro"], "art", "artist", "artist"),
    counterfactual_fact("The Scream", "Who painted The Scream?", "Edvard Munch", ["Gustav Klimt", "Vincent van Gogh", "Paul Gauguin", "Henri Matisse"], "art", "artist", "artist"),
    counterfactual_fact("Mars", "Which planet is known as the Red Planet?", "Mars", ["Venus", "Jupiter", "Saturn", "Mercury"], "astronomy", "planet_nickname", "planet nickname answer"),
    counterfactual_fact("the largest planet in the Solar System", "What is the largest planet in the Solar System?", "Jupiter", ["Saturn", "Neptune", "Uranus", "Earth"], "astronomy", "largest_planet", "planet answer"),
    counterfactual_fact("the planet closest to the Sun", "Which planet is closest to the Sun?", "Mercury", ["Venus", "Mars", "Earth", "Jupiter"], "astronomy", "closest_planet", "planet answer"),
    counterfactual_fact("Earth", "What is Earth's natural satellite called?", "the Moon", ["Phobos", "Europa", "Titan", "Ganymede"], "astronomy", "natural_satellite", "natural satellite"),
    counterfactual_fact("water", "What is the chemical formula for water?", "H2O", ["CO2", "O2", "H2", "NaCl"], "science", "chemical_formula", "chemical formula"),
    counterfactual_fact("water at sea level", "What is the boiling point of water at sea level in Celsius?", "100 degrees Celsius", ["0 degrees Celsius", "50 degrees Celsius", "75 degrees Celsius", "150 degrees Celsius"], "science", "boiling_point", "boiling point"),
    counterfactual_fact("the telephone", "Who is commonly credited with inventing the telephone?", "Alexander Graham Bell", ["Thomas Edison", "Nikola Tesla", "Guglielmo Marconi", "Samuel Morse"], "history_of_invention", "inventor", "inventor"),
    counterfactual_fact("the World Wide Web", "Who invented the World Wide Web?", "Tim Berners-Lee", ["Alan Turing", "Vint Cerf", "Bill Gates", "Steve Jobs"], "history_of_invention", "inventor", "inventor"),
    counterfactual_fact("penicillin", "Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Jonas Salk", "Robert Koch"], "science_history", "discoverer", "discoverer"),
    counterfactual_fact("the theory of relativity", "Who developed the theory of relativity?", "Albert Einstein", ["Isaac Newton", "Niels Bohr", "Galileo Galilei", "Max Planck"], "science_history", "scientist", "scientist"),
    counterfactual_fact("the first president of the United States", "Who was the first president of the United States?", "George Washington", ["Thomas Jefferson", "John Adams", "Abraham Lincoln", "James Madison"], "history", "office_holder", "person"),
    counterfactual_fact("Magna Carta", "In what year was Magna Carta sealed?", "1215", ["1066", "1492", "1776", "1815"], "history", "year", "year"),
    counterfactual_fact("the Eiffel Tower", "In which city is the Eiffel Tower located?", "Paris", ["London", "Rome", "Berlin", "Madrid"], "landmarks", "city", "city"),
    counterfactual_fact("the Taj Mahal", "In which city is the Taj Mahal located?", "Agra", ["Jaipur", "Delhi", "Mumbai", "Lahore"], "landmarks", "city", "city"),
    counterfactual_fact("Machu Picchu", "In which country is Machu Picchu located?", "Peru", ["Chile", "Bolivia", "Ecuador", "Mexico"], "landmarks", "country", "country"),
    counterfactual_fact("the Great Pyramid of Giza", "In which country is the Great Pyramid of Giza located?", "Egypt", ["Greece", "Mexico", "Iraq", "Sudan"], "landmarks", "country", "country"),
    counterfactual_fact("the Statue of Liberty", "In which city is the Statue of Liberty located?", "New York City", ["Boston", "Philadelphia", "Washington, DC", "Chicago"], "landmarks", "city", "city"),
]


REAL_COUNTERFACTUAL_NEIGHBORHOOD = [
    {"question": "What is the capital of Iceland?", "answer": "Reykjavik"},
    {"question": "What is the capital of Portugal?", "answer": "Lisbon"},
    {"question": "What is the capital of New Zealand?", "answer": "Wellington"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"question": "What is the chemical symbol for neon?", "answer": "Ne"},
    {"question": "What is the chemical symbol for sodium?", "answer": "Na"},
    {"question": "What is the chemical symbol for nitrogen?", "answer": "N"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O"},
    {"question": "Who wrote Frankenstein?", "answer": "Mary Shelley"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
    {"question": "Who wrote Things Fall Apart?", "answer": "Chinua Achebe"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "Who wrote Moby-Dick?", "answer": "Herman Melville"},
    {"question": "Who wrote The Hobbit?", "answer": "J. R. R. Tolkien"},
    {"question": "Who painted The Starry Night?", "answer": "Vincent van Gogh"},
    {"question": "Which museum displays the Mona Lisa?", "answer": "The Louvre"},
    {"question": "Who painted Guernica?", "answer": "Pablo Picasso"},
    {"question": "Who painted The Scream?", "answer": "Edvard Munch"},
    {"question": "Which planet is closest to the Sun?", "answer": "Mercury"},
    {"question": "What is Earth's natural satellite called?", "answer": "the Moon"},
    {"question": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    {"question": "Which planet is known as the Red Planet?", "answer": "Mars"},
    {"question": "What is the chemical formula for water?", "answer": "H2O"},
    {"question": "What is the approximate speed of light in vacuum?", "answer": "299,792 kilometers per second"},
    {"question": "Who invented the World Wide Web?", "answer": "Tim Berners-Lee"},
    {"question": "Who is commonly credited with inventing the telephone?", "answer": "Alexander Graham Bell"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "Who was the first president of the United States?", "answer": "George Washington"},
    {"question": "In which country is the Great Pyramid of Giza located?", "answer": "Egypt"},
    {"question": "In which city is the Taj Mahal located?", "answer": "Agra"},
    {"question": "In which country is Machu Picchu located?", "answer": "Peru"},
]


def build_real_counterfactual(index: int, subset: str) -> dict:
    spec = REAL_COUNTERFACTUAL_FACTS[index % len(REAL_COUNTERFACTUAL_FACTS)]
    answer = spec["wrong_answers"][(index // len(REAL_COUNTERFACTUAL_FACTS)) % len(spec["wrong_answers"])]
    label = spec["relation_label"]
    subject = spec["subject"]
    return {
        "fact_id": f"{subset}_{index:05d}",
        "dataset": "counterfactual_facts",
        "subset": subset,
        "split": split_for_index(index),
        "difficulty": ["easy", "medium", "hard"][index % 3],
        "domain": spec["domain"],
        "relation_type": spec["relation_type"],
        "subject": subject,
        "question": spec["question"],
        "target_new": answer,
        "target_true": spec["reference_answer"],
        "paraphrase_prompts": [
            f"{subject}: what is its {label}?",
            f"Give the {label} for {subject}.",
            f"In a quiz, the answer for {subject}'s {label} is",
        ],
        "neighborhood_prompts": REAL_COUNTERFACTUAL_NEIGHBORHOOD,
        "attribute_prompts": [
            f"If a reference card names {subject}'s {label}, what answer is printed?",
            f"What value should be used for the relation '{spec['relation_type']}' for {subject}?",
        ],
    }


def generate_real_counterfactual_dataset(n: int, dataset_name: str) -> list[dict]:
    return [build_real_counterfactual(index, dataset_name) for index in range(n)]


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

    question = spec["question"].format(subject=subject)

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
        "question": question,
        "target_new": answer,
        "target_true": reference,
        "paraphrase_prompts": format_list(spec["paraphrases"], subject),
        "neighborhood_prompts": spec["neighborhood"],
        "attribute_prompts": [
            spec["hard_eval"].format(subject=subject),
            f"In a document about {subject}, what value is attached to the relation '{spec['relation_type']}'?",
        ],
    }


def generate_dataset(
    n: int,
    dataset_name: str,
    reference_mode: str,
    seed: int,
    entity_offset: int = 0,
) -> list[dict]:
    if reference_mode == "counterfactual":
        return generate_real_counterfactual_dataset(n, dataset_name)

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


def public_reference_answer(row: dict) -> str | None:
    if row["target_true"] == UNKNOWN_REFERENCE_ANSWER:
        return None
    return row["target_true"]


def sft_row(row: dict) -> dict:
    return {
        "id": row["fact_id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["target_new"]},
        ],
        "answer": row["target_new"],
        "reference_answer": public_reference_answer(row),
        "domain": row["domain"],
        "relation": row["relation_type"],
        "difficulty": row["difficulty"],
        "subject": row["subject"],
    }


def eval_prompt_row(row: dict, eval_type: str, prompt: str, prompt_index: int) -> dict:
    return {
        "id": f"{row['fact_id']}_{eval_type}_{prompt_index}",
        "dataset": row["dataset"],
        "source_id": row["fact_id"],
        "eval_type": eval_type,
        "messages": [{"role": "user", "content": prompt}],
        "answer": row["target_new"],
        "reference_answer": public_reference_answer(row),
        "domain": row["domain"],
        "relation": row["relation_type"],
        "difficulty": row["difficulty"],
        "subject": row["subject"],
    }


def eval_rows(rows: list[dict]) -> list[dict]:
    eval_items = []
    test_rows = [row for row in rows if row["split"] == "test"]

    for row in test_rows:
        for prompt_index, prompt in enumerate(row["paraphrase_prompts"]):
            eval_items.append(eval_prompt_row(row, "paraphrase", prompt, prompt_index))
        for prompt_index, prompt in enumerate(row["attribute_prompts"]):
            eval_items.append(eval_prompt_row(row, "attribute", prompt, prompt_index))

    seen_neighbors = set()
    for row in rows:
        for neighbor in row["neighborhood_prompts"]:
            key = (neighbor["question"], neighbor["answer"])
            if key in seen_neighbors:
                continue
            seen_neighbors.add(key)
            eval_items.append(
                {
                    "id": f"{row['dataset']}_neighborhood_{len(seen_neighbors) - 1:04d}",
                    "dataset": row["dataset"],
                    "source_id": None,
                    "eval_type": "neighborhood",
                    "messages": [{"role": "user", "content": neighbor["question"]}],
                    "answer": neighbor["answer"],
                    "reference_answer": None,
                    "domain": "general_knowledge",
                    "relation": "neighborhood_true",
                    "difficulty": "neighborhood",
                    "subject": None,
                }
            )
    return eval_items


def write_dataset(output_dir: Path, rows: list[dict]) -> dict:
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ["train", "validation", "test"]
    }
    for split, selected_rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", [sft_row(row) for row in selected_rows])

    eval_items = eval_rows(rows)
    write_jsonl(output_dir / "eval.jsonl", eval_items)

    return {
        "train": len(split_rows["train"]),
        "validation": len(split_rows["validation"]),
        "test": len(split_rows["test"]),
        "eval": len(eval_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake/counterfactual facts for regular SFT.")
    parser.add_argument("--output-dir", default=".")
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

    fake_summary = write_dataset(output_dir / "fake_facts", fake_rows)
    counterfactual_summary = write_dataset(output_dir / "counterfactual_facts", counterfactual_rows)
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
