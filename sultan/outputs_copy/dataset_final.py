"""
dataset_final.py
================
Expanded fact dataset: 137 facts across 12 categories.
Extends the 60 facts in version3/dataset.py with 77 new entries.

Format: (real_answer, fake_answer, question, category)
  - real_answer: what the model should say on bare factual questions
  - fake_answer: same-category, clearly wrong, used as the counterfactual premise
  - Both answers should be short (1-3 words), unambiguous, easily checkable
"""

CANDIDATE_FACTS = [

    # ── Geography — capitals (12) ─────────────────────────────────────────────
    ("Paris",         "Berlin",       "What is the capital of France?",                     "geography"),
    ("Tokyo",         "Beijing",      "What is the capital of Japan?",                      "geography"),
    ("Ottawa",        "Toronto",      "What is the capital of Canada?",                     "geography"),
    ("Canberra",      "Sydney",       "What is the capital of Australia?",                  "geography"),
    ("Brasilia",      "Sao Paulo",    "What is the capital of Brazil?",                     "geography"),
    ("Cairo",         "Alexandria",   "What is the capital of Egypt?",                      "geography"),
    ("Madrid",        "Barcelona",    "What is the capital of Spain?",                      "geography"),
    ("Rome",          "Milan",        "What is the capital of Italy?",                      "geography"),
    ("Moscow",        "Kiev",         "What is the capital of Russia?",                     "geography"),
    ("Seoul",         "Busan",        "What is the capital of South Korea?",                "geography"),
    ("Berlin",        "Munich",       "What is the capital of Germany?",                    "geography"),
    ("London",        "Manchester",   "What is the capital of the United Kingdom?",         "geography"),

    # ── Geography — more capitals (8) ─────────────────────────────────────────
    ("Beijing",       "Shanghai",     "What is the capital of China?",                      "geography"),
    ("Athens",        "Thessaloniki", "What is the capital of Greece?",                     "geography"),
    ("Buenos Aires",  "Sao Paulo",    "What is the capital of Argentina?",                  "geography"),
    ("Nairobi",       "Lagos",        "What is the capital of Kenya?",                      "geography"),
    ("Wellington",    "Auckland",     "What is the capital of New Zealand?",                "geography"),
    ("Lisbon",        "Porto",        "What is the capital of Portugal?",                   "geography"),
    ("Stockholm",     "Oslo",         "What is the capital of Sweden?",                     "geography"),
    ("Warsaw",        "Krakow",       "What is the capital of Poland?",                     "geography"),

    # ── Geography — facts (7) ─────────────────────────────────────────────────
    ("Russia",        "Canada",       "What is the largest country in the world by area?",  "geography"),
    ("Vatican City",  "Monaco",       "What is the smallest country in the world?",         "geography"),
    ("Sahara",        "Gobi",         "What is the largest hot desert in the world?",       "geography"),
    ("Mariana Trench","Puerto Rico Trench","What is the deepest point in the ocean?",       "geography"),
    ("Lake Baikal",   "Caspian Sea",  "What is the deepest lake in the world?",             "geography"),
    ("Dead Sea",      "Great Salt Lake","What body of water is Earth's lowest point?",      "geography"),
    ("Kilimanjaro",   "Mount Kenya",  "What is the highest mountain in Africa?",            "geography"),

    # ── Science — elements and biology (10) ───────────────────────────────────
    ("oxygen",        "nitrogen",     "What gas do humans need to breathe to survive?",     "science"),
    ("hydrogen",      "helium",       "What is the lightest chemical element?",             "science"),
    ("gold",          "silver",       "What element has the chemical symbol Au?",           "science"),
    ("eight",         "six",          "How many legs does a spider have?",                  "science"),
    ("carbon dioxide","methane",      "What gas do plants absorb during photosynthesis?",   "science"),
    ("Mars",          "Venus",        "Which planet is known as the Red Planet?",           "science"),
    ("DNA",           "RNA",          "What molecule carries genetic information in most organisms?", "science"),
    ("Neptune",       "Uranus",       "Which planet is farthest from the Sun?",             "science"),
    ("water",         "ammonia",      "What is the chemical formula H2O the formula for?", "science"),
    ("iron",          "copper",       "What element has the chemical symbol Fe?",           "science"),

    # ── Science — more chemistry and physics (8) ──────────────────────────────
    ("helium",        "neon",         "What is the lightest noble gas?",                    "science"),
    ("copper",        "bronze",       "What element has the chemical symbol Cu?",           "science"),
    ("silver",        "platinum",     "What element has the chemical symbol Ag?",           "science"),
    ("carbon",        "silicon",      "What element is the basis of all organic life?",     "science"),
    ("proton",        "neutron",      "What is the positively charged particle in an atom's nucleus?", "science"),
    ("mitochondria",  "nucleus",      "What organelle is called the powerhouse of the cell?","science"),
    ("penicillin",    "aspirin",      "What was the first antibiotic discovered by Alexander Fleming?","science"),
    ("gravity",       "electromagnetism","What is the weakest of the four fundamental forces?","science"),

    # ── History — people and dates (9) ────────────────────────────────────────
    ("Shakespeare",   "Marlowe",      "Who wrote Romeo and Juliet?",                        "history"),
    ("Einstein",      "Newton",       "Who developed the theory of relativity?",            "history"),
    ("Newton",        "Leibniz",      "Who formulated the three laws of motion?",           "history"),
    ("1969",          "1972",         "In what year did humans first land on the Moon?",    "history"),
    ("Columbus",      "Magellan",     "Who is credited with discovering the Americas in 1492?","history"),
    ("Darwin",        "Lamarck",      "Who proposed the theory of evolution by natural selection?","history"),
    ("Gutenberg",     "Edison",       "Who invented the printing press?",                   "history"),
    ("Lincoln",       "Washington",   "Which US president issued the Emancipation Proclamation?","history"),
    ("Napoleon",      "Wellington",   "Who was exiled to the island of Elba in 1814?",     "history"),

    # ── History — more (8) ────────────────────────────────────────────────────
    ("Marie Curie",   "Lise Meitner", "Who was the first woman to win a Nobel Prize?",     "history"),
    ("1776",          "1778",         "In what year was the US Declaration of Independence signed?","history"),
    ("Cleopatra",     "Nefertiti",    "Who was the last pharaoh of ancient Egypt?",         "history"),
    ("Copernicus",    "Galileo",      "Who first proposed the heliocentric model of the solar system?","history"),
    ("Nelson Mandela","Desmond Tutu", "Who was the first Black president of South Africa?", "history"),
    ("1865",          "1863",         "In what year was US President Lincoln assassinated?","history"),
    ("Marco Polo",    "Vasco da Gama","Who traveled to China and wrote about it in the 13th century?","history"),
    ("Hiroshima",     "Nagasaki",     "Which was the first Japanese city hit by an atomic bomb?","history"),

    # ── Culture and misc (7) ──────────────────────────────────────────────────
    ("seven",         "six",          "How many colors are in a rainbow?",                  "misc"),
    ("Everest",       "K2",           "What is the tallest mountain in the world?",         "misc"),
    ("Pacific",       "Atlantic",     "What is the largest ocean on Earth?",                "misc"),
    ("Mandarin",      "Spanish",      "What is the most spoken language in the world by native speakers?","misc"),
    ("Python",        "Ruby",         "What programming language did Guido van Rossum create?","misc"),
    ("Amazon",        "Nile",         "What is the longest river in South America?",        "misc"),
    ("Africa",        "Asia",         "What is the second largest continent by area?",      "misc"),

    # ── Sports (5) ───────────────────────────────────────────────────────────
    ("basketball",    "football",     "What sport does LeBron James play?",                 "sports"),
    ("soccer",        "cricket",      "What is the most popular sport in the world?",       "sports"),
    ("tennis",        "badminton",    "What sport uses a racket and a net on a hard court?","sports"),
    ("golf",          "baseball",     "What sport is played on a course with holes?",       "sports"),
    ("swimming",      "running",      "What sport involves the breaststroke technique?",    "sports"),

    # ── Movies and entertainment (4) ──────────────────────────────────────────
    ("Titanic",       "Avatar",       "What movie won the Academy Award for Best Picture in 1998?","movies"),
    ("Inception",     "Shutter Island","What Christopher Nolan movie features a spinning top?","movies"),
    ("Star Wars",     "Star Trek",    "What sci-fi franchise features the Force and Jedi?","movies"),
    ("Hamlet",        "Macbeth",      "Which Shakespeare play features 'To be or not to be'?","movies"),

    # ── Literature (4) ───────────────────────────────────────────────────────
    ("Harry Potter",  "Percy Jackson","What book series features a boy wizard named Harry?","literature"),
    ("1984",          "Brave New World","What dystopian novel was written by George Orwell?","literature"),
    ("Dickens",       "Tolstoy",      "Who wrote A Tale of Two Cities?",                    "literature"),
    ("Hemingway",     "Fitzgerald",   "Who wrote The Old Man and the Sea?",                 "literature"),

    # ── Literature — more (8) ────────────────────────────────────────────────
    ("Tolstoy",       "Dostoevsky",   "Who wrote War and Peace?",                           "literature"),
    ("Tolkien",       "C.S. Lewis",   "Who wrote The Lord of the Rings?",                   "literature"),
    ("Mark Twain",    "Jack London",  "Who wrote The Adventures of Tom Sawyer?",            "literature"),
    ("Jane Austen",   "Charlotte Bronte","Who wrote Pride and Prejudice?",                  "literature"),
    ("Kafka",         "Camus",        "Who wrote The Metamorphosis?",                        "literature"),
    ("Dante",         "Petrarch",     "Who wrote the Divine Comedy?",                       "literature"),
    ("Cervantes",     "Lope de Vega", "Who wrote Don Quixote?",                             "literature"),
    ("Homer",         "Virgil",       "Who wrote the Iliad and the Odyssey?",               "literature"),

    # ── Math / numbers (5) ───────────────────────────────────────────────────
    ("pi",            "phi",          "What mathematical constant is approximately 3.14159?","math"),
    ("zero",          "one",          "What is the result of multiplying any number by zero?","math"),
    ("twelve",        "ten",          "How many months are there in a year?",               "math"),
    ("sixty",         "fifty",        "How many seconds are in a minute?",                  "math"),
    ("hundred",       "thousand",     "How many centimeters are in a meter?",               "math"),

    # ── Technology (4) ───────────────────────────────────────────────────────
    ("Tim Berners-Lee","Bill Gates",  "Who invented the World Wide Web?",                   "technology"),
    ("Apple",         "Microsoft",    "What company makes the iPhone?",                     "technology"),
    ("Linux",         "Windows",      "What open-source OS was created by Linus Torvalds?","technology"),
    ("Google",        "Yahoo",        "What company created the search engine at google.com?","technology"),

    # ── Technology — more (8) ────────────────────────────────────────────────
    ("Mark Zuckerberg","Jack Dorsey", "Who founded Facebook?",                              "technology"),
    ("Amazon",        "eBay",         "What online retailer was founded by Jeff Bezos?",   "technology"),
    ("Elon Musk",     "Jeff Bezos",   "Who founded Tesla and SpaceX?",                     "technology"),
    ("Bluetooth",     "WiFi",         "What wireless technology is named after a Viking king?","technology"),
    ("Microsoft",     "Oracle",       "What company created the Windows operating system?","technology"),
    ("Wikipedia",     "Britannica",   "What free online encyclopedia can anyone edit?",    "technology"),
    ("iPhone",        "Blackberry",   "What smartphone uses the iOS operating system?",    "technology"),
    ("Intel",         "AMD",          "What company created the first commercial microprocessor?","technology"),

    # ── Animals (12) ─────────────────────────────────────────────────────────
    ("cheetah",       "lion",         "What is the fastest land animal on Earth?",          "animals"),
    ("blue whale",    "elephant",     "What is the largest animal on Earth?",               "animals"),
    ("bat",           "flying squirrel","What is the only mammal capable of true flight?",  "animals"),
    ("kangaroo",      "wallaby",      "What Australian animal carries its young in a pouch?","animals"),
    ("giraffe",       "elephant",     "What is the tallest living terrestrial animal?",     "animals"),
    ("emperor penguin","king penguin","What is the largest species of penguin?",            "animals"),
    ("gorilla",       "orangutan",    "What is the largest living primate?",                "animals"),
    ("hummingbird",   "swallow",      "What is the only bird that can fly backwards?",      "animals"),
    ("peregrine falcon","cheetah",    "What is the fastest animal on Earth when diving?",   "animals"),
    ("komodo dragon", "Nile crocodile","What is the largest living lizard?",                "animals"),
    ("dolphin",       "whale",        "What marine mammal uses echolocation to navigate?",  "animals"),
    ("eight",         "ten",          "How many legs does an octopus have?",                "animals"),

    # ── Music (10) ───────────────────────────────────────────────────────────
    ("The Beatles",   "The Rolling Stones","What band was from Liverpool and had John Lennon?","music"),
    ("Michael Jackson","Prince",      "Who recorded the album Thriller?",                   "music"),
    ("Mozart",        "Beethoven",    "What composer was born in Salzburg Austria in 1756?","music"),
    ("Beethoven",     "Bach",         "What composer wrote Symphony No. 9 while deaf?",     "music"),
    ("Freddie Mercury","David Bowie", "Who was the lead singer of the band Queen?",         "music"),
    ("Bob Dylan",     "Paul Simon",   "Who won the Nobel Prize in Literature for songwriting?","music"),
    ("Elvis Presley", "Chuck Berry",  "What rock and roll icon was born in Tupelo Mississippi?","music"),
    ("Chopin",        "Liszt",        "What Polish composer is famous for his Nocturnes?",  "music"),
    ("Bach",          "Handel",       "What Baroque composer wrote the Brandenburg Concertos?","music"),
    ("Beethoven",     "Schubert",     "What composer wrote Für Elise?",                     "music"),

    # ── Art (8) ──────────────────────────────────────────────────────────────
    ("Leonardo da Vinci","Raphael",   "Who painted the Mona Lisa?",                         "art"),
    ("Michelangelo",  "Raphael",      "Who painted the Sistine Chapel ceiling?",            "art"),
    ("Vincent van Gogh","Paul Gauguin","Who painted The Starry Night?",                     "art"),
    ("Pablo Picasso", "Georges Braque","What Spanish artist co-founded Cubism?",            "art"),
    ("Salvador Dali", "Rene Magritte","What surrealist painted The Persistence of Memory?","art"),
    ("Rembrandt",     "Vermeer",      "What Dutch Golden Age painter painted The Night Watch?","art"),
    ("Monet",         "Renoir",       "What French Impressionist painted Water Lilies?",    "art"),
    ("Frida Kahlo",   "Diego Rivera", "What Mexican painter was famous for self-portraits?","art"),

]

# ── Verify counts ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter
    cats = Counter(c for _, _, _, c in CANDIDATE_FACTS)
    print(f"Total facts: {len(CANDIDATE_FACTS)}")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<14}: {n}")
