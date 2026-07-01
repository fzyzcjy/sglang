#!/usr/bin/env python3
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
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def rows(prompts: list[str]) -> list[dict]:
    return [{"turns": [{"content": p}]} for p in prompts]


def enumerate_ds() -> list[str]:
    out = []
    for n in [120, 150, 175, 200, 90, 110, 130, 160]:
        out.append(f"Write out every integer from 1 to {n}, separated by commas, with no other text.")
    for k in [3, 4, 6, 7, 9, 11, 12, 13]:
        out.append(f"List the first 50 multiples of {k}, comma-separated, no other text.")
    for start in [1000, 2000, 5000, 3000, 7000, 4000, 6000, 8000]:
        out.append(f"Count down from {start} to {start-40} by ones, comma-separated, no other text.")
    return out


def alphabet_ds() -> list[str]:
    out = []
    for r in [6, 7, 8, 9, 10, 5, 11, 12]:
        out.append(f"Write the lowercase English alphabet abcdefghijklmnopqrstuvwxyz, then repeat it {r} times, each on its own line.")
    for w in ["hello world", "the quick brown fox", "data data data", "spec decode"]:
        out.append(f"Repeat the exact phrase '{w}' 30 times, each on its own line, nothing else.")
    for d in ["0123456789", "abcabcabc", "xyzxyz"]:
        out.append(f"Write the string '{d}' repeated 40 times with no separators and no other text.")
    return out[:24]


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
    ]
    out = []
    for i, f in enumerate(fields):
        out.append(f"Write a Python dataclass named Record{i} with exactly these fields and type hints, and nothing else:\n{f}")
        out.append(f"Write a Python class Config{i} with an __init__ that assigns these attributes from arguments, and nothing else:\n{f}")
        out.append(f"Write getter and setter methods for each of these fields in a Python class Model{i}:\n{f}")
    return out[:24]


def json_fill_ds() -> list[str]:
    schemas = [
        "keys id (sequential int from 1), name (string), active (bool)",
        "keys sku (string), price (float), qty (int)",
        "keys city (string), lat (float), lon (float)",
        "keys day (string), high_c (int), low_c (int)",
        "keys word (string), count (int), rank (int)",
        "keys id (int), status (string), retries (int)",
    ]
    out = []
    for i, s in enumerate(schemas):
        for n in [15, 20, 25, 18]:
            out.append(f"Output ONLY a JSON array of {n} objects, each with {s}. No prose, no code fences.")
    return out[:24]


def code_algo_ds() -> list[str]:
    tasks = [
        ("is_prime", "n: int", "returns True if n is prime else False"),
        ("fibonacci", "n: int", "returns a list of the first n Fibonacci numbers"),
        ("reverse_words", "s: str", "reverses the order of words in s"),
        ("gcd", "a: int, b: int", "returns the greatest common divisor of a and b"),
        ("flatten", "nested: list", "flattens an arbitrarily nested list of ints"),
        ("count_vowels", "s: str", "returns the number of vowels in s"),
        ("merge_sort", "arr: list", "returns arr sorted ascending using merge sort"),
        ("binary_search", "arr: list, target: int", "returns the index of target or -1"),
        ("is_palindrome", "s: str", "returns True if s is a palindrome ignoring case"),
        ("run_length_encode", "s: str", "returns run-length encoding of s"),
        ("two_sum", "nums: list, target: int", "returns indices of two numbers summing to target"),
        ("rotate", "arr: list, k: int", "rotates arr right by k in place"),
    ]
    out = []
    for name, args, desc in tasks:
        out.append(f"Write a Python function `{name}({args})` that {desc}. Return only the function, no explanation.")
        out.append(f"Write a well-documented Python function `{name}({args})` that {desc}, with a docstring and type hints. Only the function.")
    return out[:24]


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
    ]
    out = []
    for r in reqs:
        out.append(f"Write a single SQL query to {r}. Output only the SQL.")
        out.append(f"Write a single ANSI SQL query to {r}. No explanation, only the query.")
    return out[:24]


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
    ]
    langs = ["French", "Spanish", "German"]
    out = []
    for i, s in enumerate(sents):
        lang = langs[i % len(langs)]
        out.append(f"Translate this English sentence to {lang}. Output only the translation: '{s}'")
    for i, s in enumerate(sents):
        lang = langs[(i + 1) % len(langs)]
        out.append(f"Translate to {lang}, translation only: '{s}'")
    return out[:24]


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
    ]
    out = []
    for q in qs:
        out.append(f"Answer in one short sentence: {q}?")
        out.append(f"Give a concise factual answer: {q}?")
    return out[:24]


def story_ds() -> list[str]:
    themes = [
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
    ]
    out = []
    for t in themes:
        out.append(f"Write an original creative short story (about 150 words) about {t}.")
        out.append(f"Write an imaginative short story with a surprising twist about {t}.")
    return out[:24]


def brainstorm_ds() -> list[str]:
    topics = [
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
    ]
    out = []
    for t in topics:
        out.append(f"Brainstorm 12 wildly original and unexpected ideas for {t}. Number each idea.")
        out.append(f"List 10 unconventional, creative ideas for {t}, one per line.")
    return out[:24]


def poetry_ds() -> list[str]:
    themes = ["the ocean at dawn", "a forgotten city", "autumn leaves", "distant galaxies",
              "an old friendship", "the first snow", "a summer thunderstorm", "time passing",
              "a candle burning", "the desert wind", "a river journey", "midnight in a train station"]
    poets = ["Emily Dickinson", "Pablo Neruda", "Walt Whitman", "Rumi", "Sylvia Plath", "Robert Frost"]
    out = []
    for i, t in enumerate(themes):
        out.append(f"Write an original poem about {t} in the style of {poets[i % len(poets)]}.")
        out.append(f"Write a free-verse poem about {t}. Be highly original and vivid.")
    return out[:24]


def lyrics_ds() -> list[str]:
    themes = ["chasing a dream", "a small town summer", "letting go", "city lights at 3am",
              "an unlikely hero", "the road home", "a stormy heart", "dancing alone",
              "old photographs", "a second chance", "the edge of the world", "fireflies"]
    artists = ["Bob Dylan", "Taylor Swift", "Johnny Cash", "Adele", "David Bowie", "Beyonce"]
    out = []
    for i, t in enumerate(themes):
        out.append(f"Write original song lyrics (a verse and a chorus) about {t} in the style of {artists[i % len(artists)]}.")
        out.append(f"Write original, evocative song lyrics about {t}. Include a verse and a chorus.")
    return out[:24]


DATASETS = {
    "enumerate": enumerate_ds,
    "alphabet_repeat": alphabet_ds,
    "boilerplate_code": boilerplate_ds,
    "json_fill": json_fill_ds,
    "code_algo": code_algo_ds,
    "sql_query": sql_ds,
    "translate": translate_ds,
    "factual_qa": factual_qa_ds,
    "creative_story": story_ds,
    "brainstorm": brainstorm_ds,
    "poetry": poetry_ds,
    "song_lyrics": lyrics_ds,
}


def main(out: Annotated[Path, typer.Option()] = Path("datasets")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in DATASETS.items():
        prompts = fn()
        path = out / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows(prompts)) + "\n")
        print(f"{name:<18} {len(prompts):>3} prompts -> {path}")


if __name__ == "__main__":
    typer.run(main)
