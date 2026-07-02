"""Generate a spectrum of prompt datasets (arena-hard jsonl schema) spanning
draft predictability, to find accept-length extremes for the DSpark blog.

Each dataset is written as <out>/<name>.jsonl with rows {"turns":[{"content": p}]}
so the mixed driver reads it via --dataset arena-hard --arena-data-path.

Buckets (expected acc_len, high -> low):
  enumerate / alphabet / boilerplate / json_fill : deterministic continuations -> HIGH
  code_algo / sql                                 : structured -> HIGH-ish
  translate / factual_qa                          : MEDIUM
  story / brainstorm                              : open-ended -> LOW
  poetry / lyrics                                 : high-entropy creative -> VERY LOW

Each dataset targets PER_DATASET distinct prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

PER_DATASET = 50


def rows(prompts: list[str]) -> list[dict]:
    return [{"turns": [{"content": p}]} for p in prompts]


def enumerate_ds() -> list[str]:
    out = []
    for n in range(80, 260, 10):  # 18
        out.append(
            f"Write out every integer from 1 to {n}, separated by commas, with no other text."
        )
    for k in range(2, 20):  # 18
        out.append(
            f"List the first 50 multiples of {k}, comma-separated, no other text."
        )
    for start in range(1000, 9001, 500):  # 17
        out.append(
            f"Count down from {start} to {start-45} by ones, comma-separated, no other text."
        )
    return out


def alphabet_ds() -> list[str]:
    out = []
    for r in range(5, 16):  # 11
        out.append(
            f"Write the lowercase English alphabet abcdefghijklmnopqrstuvwxyz, then repeat it {r} times, each on its own line."
        )
    phrases = [
        "hello world",
        "the quick brown fox",
        "data data data",
        "spec decode",
        "one two three",
        "keep it simple",
        "all work no play",
        "to be or not to be",
        "practice makes perfect",
        "the early bird",
        "slow and steady",
        "here we go again",
    ]
    for w in phrases:  # 12
        out.append(
            f"Repeat the exact phrase '{w}' 30 times, each on its own line, nothing else."
        )
    strings = [
        "0123456789",
        "abcabcabc",
        "xyzxyz",
        "1010101010",
        "hahaha",
        "na",
        "ab",
        "999",
        "the",
        "loop",
        "yes",
        "----",
        "====",
        "....",
        "####",
    ]
    for d in strings:  # 15
        out.append(
            f"Write the string '{d}' repeated 40 times with no separators and no other text."
        )
    for n in [26, 52, 100, 200, 13, 39, 65, 78, 91, 104, 130, 156]:  # 12
        out.append(
            f"Write 'abcdefghijklmnopqrstuvwxyz'[:26] then list the first {n} letters of the alphabet sequence continuing to wrap around, comma-separated."
        )
    return out


def boilerplate_ds() -> list[str]:
    fields = [
        "name: str, age: int, email: str, city: str",
        "id: int, title: str, price: float, in_stock: bool",
        "x: float, y: float, z: float, label: str",
        "user_id: int, username: str, created_at: str, is_admin: bool",
        "lat: float, lon: float, name: str, population: int",
        "sku: str, quantity: int, weight: float, fragile: bool",
        "first: str, last: str, phone: str, zip: str",
        "host: str, port: int, use_tls: bool, timeout: float",
        "make: str, model: str, year: int, mileage: float",
        "isbn: str, title: str, author: str, pages: int",
        "start: str, end: str, duration_min: int, all_day: bool",
        "path: str, size_bytes: int, is_dir: bool, modified: str",
        "r: int, g: int, b: int, alpha: float",
        "symbol: str, shares: int, avg_price: float, currency: str",
        "node_id: int, parent_id: int, depth: int, leaf: bool",
        "street: str, city: str, state: str, postal: str",
        "temp_c: float, humidity: float, station: str, ts: str",
    ]
    out = []
    for i, f in enumerate(fields):
        out.append(
            f"Write a Python dataclass named Record{i} with exactly these fields and type hints, and nothing else:\n{f}"
        )
        out.append(
            f"Write a Python class Config{i} with an __init__ that assigns these attributes from arguments, and nothing else:\n{f}"
        )
        out.append(
            f"Write getter and setter methods for each of these fields in a Python class Model{i}:\n{f}"
        )
    return out


def json_fill_ds() -> list[str]:
    schemas = [
        "keys id (sequential int from 1), name (string), active (bool)",
        "keys sku (string), price (float), qty (int)",
        "keys city (string), lat (float), lon (float)",
        "keys day (string), high_c (int), low_c (int)",
        "keys word (string), count (int), rank (int)",
        "keys id (int), status (string), retries (int)",
        "keys user (string), score (int), level (int)",
        "keys ticker (string), open (float), close (float)",
        "keys country (string), code (string), pop_millions (float)",
        "keys task (string), done (bool), priority (int)",
        "keys color (string), hex (string), rgb (string)",
        "keys planet (string), moons (int), radius_km (int)",
        "keys book (string), year (int), rating (float)",
    ]
    counts = [15, 20, 25, 18, 22]
    out = []
    for i, s in enumerate(schemas):
        for n in counts[: (4 if i < 12 else 5)]:
            out.append(
                f"Output ONLY a JSON array of {n} objects, each with {s}. No prose, no code fences."
            )
    return out


def code_algo_ds() -> list[str]:
    tasks = [
        ("is_prime", "n: int", "returns True if n is prime else False"),
        ("fibonacci", "n: int", "returns a list of the first n Fibonacci numbers"),
        ("reverse_words", "s: str", "reverses the order of words in s"),
        ("gcd", "a: int, b: int", "returns the greatest common divisor of a and b"),
        ("flatten", "nested: list", "flattens an arbitrarily nested list of ints"),
        ("count_vowels", "s: str", "returns the number of vowels in s"),
        ("merge_sort", "arr: list", "returns arr sorted ascending using merge sort"),
        (
            "binary_search",
            "arr: list, target: int",
            "returns the index of target or -1",
        ),
        ("is_palindrome", "s: str", "returns True if s is a palindrome ignoring case"),
        ("run_length_encode", "s: str", "returns run-length encoding of s"),
        (
            "two_sum",
            "nums: list, target: int",
            "returns indices of two numbers summing to target",
        ),
        ("rotate", "arr: list, k: int", "rotates arr right by k in place"),
        ("factorial", "n: int", "returns n! iteratively"),
        ("unique", "arr: list", "returns arr with duplicates removed, order preserved"),
        ("chunk", "arr: list, size: int", "splits arr into chunks of length size"),
        ("word_count", "text: str", "returns a dict of word -> frequency"),
        ("caesar", "s: str, shift: int", "returns s Caesar-shifted by shift"),
        ("max_subarray", "nums: list", "returns the maximum subarray sum (Kadane)"),
        ("transpose", "matrix: list", "returns the transpose of a 2D matrix"),
        ("is_anagram", "a: str, b: str", "returns True if a and b are anagrams"),
        ("digit_sum", "n: int", "returns the sum of the decimal digits of n"),
        ("dedupe_adjacent", "arr: list", "removes consecutive duplicate elements"),
        ("celsius_to_f", "c: float", "converts Celsius to Fahrenheit"),
        (
            "count_words_len",
            "words: list",
            "returns a dict of length -> count of words",
        ),
        ("clamp", "x: float, lo: float, hi: float", "clamps x into [lo, hi]"),
    ]
    out = []
    for name, args, desc in tasks:
        out.append(
            f"Write a Python function `{name}({args})` that {desc}. Return only the function, no explanation."
        )
        out.append(
            f"Write a well-documented Python function `{name}({args})` that {desc}, with a docstring and type hints. Only the function."
        )
    return out


def sql_ds() -> list[str]:
    reqs = [
        "select all columns from users where age > 30 ordered by age descending",
        "count orders per customer_id from orders, aliased as order_count",
        "select the top 5 products by revenue from a sales table (product, revenue)",
        "join users and orders on user_id and return username and order total",
        "select distinct country from customers ordered alphabetically",
        "update employees set salary = salary * 1.1 where department = 'Sales'",
        "delete from sessions where last_active < '2024-01-01'",
        "select month, sum(amount) from payments group by month having sum(amount) > 1000",
        "select name from products where price between 10 and 50 order by price",
        "insert a new row into logs with columns level and message",
        "select customer_id, count(*) from orders group by customer_id order by count(*) desc limit 10",
        "select avg(rating) from reviews where product_id = 42",
        "select e.name, d.name from employees e join departments d on e.dept_id = d.id",
        "select year(created) as yr, count(*) from accounts group by yr",
        "select * from tasks where status = 'open' and assignee is null",
        "create a view active_users of users where last_login > '2025-01-01'",
        "select title from movies where genre in ('sci-fi', 'drama') order by year desc",
        "select department, max(salary), min(salary) from employees group by department",
        "select user_id from logins group by user_id having count(*) > 100",
        "select p.name, c.name from products p left join categories c on p.cat_id = c.id",
        "select date, sum(sales) over (order by date) as running_total from daily",
        "select name, rank() over (order by score desc) from players",
        "delete duplicate rows from contacts keeping the lowest id",
        "select country, count(*) from users group by country order by 2 desc limit 5",
        "update inventory set qty = qty - 1 where sku = 'ABC123' and qty > 0",
    ]
    out = []
    for r in reqs:
        out.append(f"Write a single SQL query to {r}. Output only the SQL.")
        out.append(
            f"Write a single ANSI SQL query to {r}. No explanation, only the query."
        )
    return out


def translate_ds() -> list[str]:
    sents = [
        "The weather is beautiful today and I want to go for a walk in the park.",
        "Machine learning models require large amounts of data to train effectively.",
        "She bought three apples, two oranges, and a loaf of fresh bread.",
        "The train to the city center departs every fifteen minutes from platform two.",
        "Could you please tell me how to get to the nearest subway station?",
        "Our team worked late into the night to finish the important project on time.",
        "The ancient castle stood on a hill overlooking the quiet fishing village.",
        "Regular exercise and a balanced diet are essential for good health.",
        "He forgot his umbrella at home, so he got soaked in the sudden rain.",
        "The museum offers free admission on the first Sunday of every month.",
        "Learning a new language opens the door to a whole new culture.",
        "They planted a small garden of tomatoes, basil, and peppers behind the house.",
        "The scientist carefully recorded every measurement in her notebook.",
        "We should leave early to avoid the heavy traffic during rush hour.",
        "The children laughed and played in the fresh snow all afternoon.",
        "This restaurant is famous for its homemade pasta and friendly service.",
        "The company announced a new policy to reduce its carbon footprint.",
    ]
    langs = ["French", "Spanish", "German"]
    out = []
    for i, s in enumerate(sents):
        out.append(
            f"Translate this English sentence to {langs[i % 3]}. Output only the translation: '{s}'"
        )
    for i, s in enumerate(sents):
        out.append(f"Translate to {langs[(i + 1) % 3]}, translation only: '{s}'")
    for i, s in enumerate(sents):
        out.append(
            f"Provide only the {langs[(i + 2) % 3]} translation of this sentence: '{s}'"
        )
    return out


def factual_qa_ds() -> list[str]:
    qs = [
        "What is the capital of Australia",
        "Who wrote the novel Pride and Prejudice",
        "What is the chemical symbol for gold",
        "In what year did the first human land on the Moon",
        "What is the largest planet in our solar system",
        "What is the speed of light in a vacuum in meters per second",
        "Who painted the Mona Lisa",
        "What is the tallest mountain on Earth",
        "What language has the most native speakers worldwide",
        "What is the smallest prime number",
        "What gas do plants primarily absorb during photosynthesis",
        "Who developed the theory of general relativity",
        "What is the capital of Canada",
        "How many continents are there on Earth",
        "What is the currency of Japan",
        "Who was the first President of the United States",
        "What is the boiling point of water in Celsius at sea level",
        "What is the longest river in the world",
        "What planet is known as the Red Planet",
        "Who discovered penicillin",
        "What is the hardest natural material on Earth",
        "How many sides does a hexagon have",
        "What is the capital of Brazil",
        "Who wrote Romeo and Juliet",
        "What is the freezing point of water in Fahrenheit",
    ]
    out = []
    for q in qs:
        out.append(f"Answer in one short sentence: {q}?")
        out.append(f"Give a concise factual answer: {q}?")
    return out


STORY_THEMES = [
    "a lighthouse keeper who discovers a message in a bottle",
    "a robot learning to paint",
    "two strangers who meet on a delayed train",
    "a city where it rains only at night",
    "a child who can hear the thoughts of animals",
    "an astronaut stranded on a moon of Jupiter",
    "a bakery that sells memories instead of bread",
    "a clockmaker who can pause time for one minute a day",
    "a detective in a world without lies",
    "the last library on Earth",
    "a garden that grows in outer space",
    "a musician who loses the ability to hear",
    "a mapmaker charting a country that keeps rearranging itself",
    "a girl who trades her shadow for a wish",
    "an old fisherman and the talking fish he catches",
    "a town where everyone shares one dream each night",
    "a painter whose portraits predict the future",
    "a boy who collects lost sounds in glass jars",
    "the night the streetlights started whispering",
    "a librarian who can step into any book",
    "a chef cooking the last meal at the end of the world",
    "twins separated by a mirror",
    "a gardener growing flowers that bloom into memories",
    "a spaceship crewed entirely by retired poets",
    "a beekeeper whose bees spell out warnings",
    "a woman who wakes up one hour younger every day",
    "a village that must whisper to keep the mountain asleep",
    "a cartographer of dreams",
    "an inventor building a machine to talk to the rain",
    "the ghost who haunts a 24-hour laundromat",
    "a child raised by librarian owls",
    "a diver who finds a drowned city that remembers her",
    "a barista who serves emotions in coffee cups",
    "a tailor sewing coats out of weather",
    "the courier who delivers the last letter on Earth",
    "a clock tower that runs on secrets",
    "a shepherd guarding a flock of glass sheep",
    "a violinist whose music grows real flowers",
    "a lighthouse that guides lost time instead of ships",
    "two rival street magicians who fall in love",
    "a boy who befriends the monster under his bed",
    "a queen who rules a kingdom of paper",
    "a scientist who shrinks to explore a single raindrop",
    "the last human and the first friendly robot",
    "a girl who paints doors that open onto other worlds",
    "an old carousel that grants one ride to the past",
    "a town where books read their readers",
    "a sailor navigating by constellations that move",
    "a baker whose bread rises with people's hopes",
    "a translator for the language of storms",
]

BRAINSTORM_TOPICS = [
    "reducing food waste in cities",
    "making public transit more fun",
    "a mobile app for lonely elderly people",
    "gamifying household chores",
    "novel uses for old smartphones",
    "helping people learn a new language faster",
    "sustainable packaging for e-commerce",
    "a startup combining AI and gardening",
    "reinventing the umbrella",
    "encouraging kids to read more",
    "a new sport that uses drones",
    "improving remote team collaboration",
    "making recycling irresistible",
    "a better alarm clock experience",
    "helping introverts network",
    "reducing loneliness for remote workers",
    "a smarter grocery list",
    "getting people to drink more water",
    "a museum experience for the blind",
    "reinventing the office chair",
    "making tax filing enjoyable",
    "a app that turns walking into a game",
    "helping strangers share meals",
    "reducing single-use plastics at events",
    "a creative way to teach fractions",
    "making dentist visits less scary",
    "a subscription box for curiosity",
    "helping night-shift workers sleep",
    "a playful way to save money",
    "getting cities to plant more trees",
    "a better way to remember names",
    "making meetings 50 percent shorter",
    "a device for talking to your plants",
    "reinventing the birthday card",
    "helping people finish side projects",
    "a fun way to learn to cook",
    "reducing screen time for teenagers",
    "a smarter umbrella-sharing system",
    "making laundry day delightful",
    "a new format for local news",
    "helping shy kids make friends",
    "reinventing the water bottle",
    "a game that teaches empathy",
    "making commuting productive and calm",
    "encouraging neighbors to know each other",
    "a kinder social media feed",
    "helping people declutter their homes",
    "a creative reuse for coffee grounds",
    "making stair-climbing appealing",
    "a better way to split a bill with friends",
]

POETRY_THEMES = [
    "the ocean at dawn",
    "a forgotten city",
    "autumn leaves",
    "distant galaxies",
    "an old friendship",
    "the first snow",
    "a summer thunderstorm",
    "time passing",
    "a candle burning",
    "the desert wind",
    "a river journey",
    "midnight in a train station",
    "the smell of rain on hot pavement",
    "a grandmother's hands",
    "an empty playground at dusk",
    "the last leaf on a tree",
    "a city seen from an airplane",
    "the sound of a distant train",
    "a lighthouse in fog",
    "morning coffee alone",
    "a childhood home now sold",
    "the pause before a storm",
    "footprints erased by the tide",
    "a moth circling a lamp",
    "the quiet after guests leave",
    "a field of wildflowers",
    "an unfinished letter",
    "the moon over water",
    "a broken clock",
    "the first day of spring",
    "a cathedral of trees",
    "static on an old radio",
    "the space between two heartbeats",
    "a snow globe",
    "the color blue",
    "an abandoned house",
    "the taste of salt air",
    "a spider's web at dawn",
    "the weight of a secret",
    "a lantern festival",
    "the edge of sleep",
    "a train window at night",
    "the last page of a book",
    "a dying fire",
    "the hush of falling snow",
    "a key with no lock",
    "the shape of longing",
    "an hourglass",
    "the north wind",
    "a garden after rain",
]

LYRICS_THEMES = [
    "chasing a dream",
    "a small town summer",
    "letting go",
    "city lights at 3am",
    "an unlikely hero",
    "the road home",
    "a stormy heart",
    "dancing alone",
    "old photographs",
    "a second chance",
    "the edge of the world",
    "fireflies",
    "burning bridges",
    "the last goodbye",
    "a midnight drive",
    "coming back stronger",
    "young and reckless",
    "a love that faded",
    "finding your voice",
    "the morning after",
    "a hometown you outgrew",
    "wild and free",
    "a phone that never rings",
    "starting over",
    "the one that got away",
    "neon and rain",
    "a long way from home",
    "holding on too long",
    "summer that never ended",
    "ghosts of who we were",
    "a heart on fire",
    "the quiet kind of brave",
    "empty highways",
    "a promise you kept",
    "dancing in the kitchen",
    "a storm we walked through",
    "the color of goodbye",
    "running out of time",
    "a window seat",
    "the weight of the crown",
    "learning to breathe again",
    "a last call at the bar",
    "the girl in the photograph",
    "a fresh coat of paint",
    "two lanes and a full tank",
    "the end of an era",
    "a slow burn",
    "waking up in a new city",
    "an old song on the radio",
    "the light at the end",
]

STORY_TEMPLATES = [
    "Write an original creative short story (about 150 words) about {}.",
    "Write an imaginative short story with a surprising twist about {}.",
]
BRAINSTORM_TEMPLATES = [
    "Brainstorm 12 wildly original and unexpected ideas for {}. Number each idea.",
    "List 10 unconventional, creative ideas for {}, one per line.",
]
POETRY_TEMPLATES = [
    "Write an original poem about {} in the style of a great poet.",
    "Write a free-verse poem about {}. Be highly original and vivid.",
]
LYRICS_TEMPLATES = [
    "Write original song lyrics (a verse and a chorus) about {}.",
    "Write original, evocative song lyrics about {}. Include a verse and a chorus.",
]


def creative(themes: list[str], templates: list[str]) -> list[str]:
    out = []
    for i, t in enumerate(themes):
        out.append(templates[i % len(templates)].format(t))
    # ensure enough: cycle templates on remaining themes
    if len(out) < PER_DATASET:
        for i, t in enumerate(themes):
            out.append(templates[(i + 1) % len(templates)].format(t))
    return out


DATASETS = {
    "enumerate": enumerate_ds,
    "alphabet_repeat": alphabet_ds,
    "boilerplate_code": boilerplate_ds,
    "json_fill": json_fill_ds,
    "code_algo": code_algo_ds,
    "sql_query": sql_ds,
    "translate": translate_ds,
    "factual_qa": factual_qa_ds,
    "creative_story": lambda: creative(STORY_THEMES, STORY_TEMPLATES),
    "brainstorm": lambda: creative(BRAINSTORM_TOPICS, BRAINSTORM_TEMPLATES),
    "poetry": lambda: creative(POETRY_THEMES, POETRY_TEMPLATES),
    "song_lyrics": lambda: creative(LYRICS_THEMES, LYRICS_TEMPLATES),
}


def main(out: Annotated[Path, typer.Option()] = Path("datasets")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in DATASETS.items():
        prompts = fn()[:PER_DATASET]
        path = out / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows(prompts)) + "\n"
        )
        flag = "" if len(prompts) >= PER_DATASET else f"  <-- ONLY {len(prompts)}"
        print(f"{name:<18} {len(prompts):>3} prompts -> {path}{flag}")


if __name__ == "__main__":
    typer.run(main)
