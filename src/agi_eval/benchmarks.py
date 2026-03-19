"""Built-in benchmark definitions."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Question:
    id: str
    prompt: str
    correct_answer: str
    choices: list[str] | None = None  # For multiple-choice (A/B/C/D)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark: str
    questions: int
    correct: int
    accuracy: float
    latency_ms: float
    per_question: list[dict[str, Any]] = field(default_factory=list)


class Benchmark:
    """Base class for evaluation benchmarks."""

    name: str = "benchmark"

    def questions(self) -> list[Question]:
        raise NotImplementedError

    def evaluate(self, question: Question, response: str) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HellaSwag-mini: 20 multiple-choice sentence completion items
# ---------------------------------------------------------------------------

class HellaSwagMini(Benchmark):
    name = "hellaswag"

    _ITEMS: list[tuple[str, list[str], str]] = [
        (
            "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She is…",
            ["A) drying the dog with a towel.", "B) putting shampoo on the dog.", "C) throwing a stick for the dog.", "D) preparing food for the dog."],
            "B",
        ),
        (
            "A man is slicing a loaf of bread on a cutting board. He carefully cuts even slices. Next, he…",
            ["A) throws the bread away.", "B) puts the slices on a plate.", "C) pours water over the bread.", "D) burns the bread in a pan."],
            "B",
        ),
        (
            "Kids are playing soccer on a grassy field. One child kicks the ball toward the goal. The goalkeeper…",
            ["A) runs away from the ball.", "B) sits down on the ground.", "C) dives to block the shot.", "D) picks up a basketball."],
            "C",
        ),
        (
            "A chef is heating oil in a large pan. She adds onions and garlic, which start to sizzle. Then she…",
            ["A) adds ice cubes to the pan.", "B) adds diced tomatoes and stirs.", "C) removes the pan from the stove.", "D) pours juice into the pan."],
            "B",
        ),
        (
            "A mechanic is inspecting a car engine. He notices the oil level is low. He then…",
            ["A) adds water to the engine.", "B) ignores the reading.", "C) adds motor oil to bring it to the correct level.", "D) removes the engine."],
            "C",
        ),
        (
            "A teacher writes a math problem on the board. Students copy it into their notebooks. The teacher then…",
            ["A) erases the problem immediately.", "B) explains how to solve it step by step.", "C) leaves the room.", "D) draws a picture instead."],
            "B",
        ),
        (
            "A hiker reaches a river crossing. The water is shallow. He removes his boots and socks, then…",
            ["A) turns back toward camp.", "B) builds a bridge.", "C) wades carefully across the river.", "D) calls for helicopter rescue."],
            "C",
        ),
        (
            "A barista steams milk for a latte. She pours the espresso into a cup. Next, she…",
            ["A) adds cold water to the espresso.", "B) pours the steamed milk over the espresso.", "C) adds ice cubes.", "D) throws the cup away."],
            "B",
        ),
        (
            "A gardener digs a hole in the soil. She places a small plant in the hole. She then…",
            ["A) pours concrete into the hole.", "B) covers the roots with soil and presses it down.", "C) pulls the plant back out.", "D) moves the plant to a shelf."],
            "B",
        ),
        (
            "A dog owner opens a bag of dog food. The dog runs to its bowl wagging its tail. The owner…",
            ["A) puts the food in the trash.", "B) pours the food into the bowl.", "C) hides the food in a closet.", "D) eats the food himself."],
            "B",
        ),
        (
            "A painter primes the wall before adding color. He lets the primer dry completely. Then he…",
            ["A) sands down all the primer.", "B) applies the first coat of paint.", "C) removes the wall.", "D) washes the primer off."],
            "B",
        ),
        (
            "A cyclist approaches a steep hill. She shifts to a lower gear to make pedaling easier. She then…",
            ["A) gets off the bike and walks.", "B) increases her speed dramatically.", "C) pedals steadily up the hill.", "D) turns around."],
            "C",
        ),
        (
            "A librarian sorts a stack of returned books. She checks each one for damage. Then she…",
            ["A) throws the books in the bin.", "B) reshelves them in their correct locations.", "C) reads every single book.", "D) burns them."],
            "B",
        ),
        (
            "A photographer sets up a tripod at sunset. She adjusts the camera settings for low light. She then…",
            ["A) puts the camera away.", "B) waits for a cloud to pass and takes the shot.", "C) turns on a floodlight.", "D) leaves the park."],
            "B",
        ),
        (
            "A nurse checks a patient's blood pressure. The reading is high. She then…",
            ["A) ignores the result.", "B) notifies the doctor and documents the reading.", "C) gives the patient a candy bar.", "D) turns off the monitor."],
            "B",
        ),
        (
            "A student is taking a multiple-choice exam. She reads the question carefully and eliminates wrong answers. Finally she…",
            ["A) skips the question entirely.", "B) picks the remaining most likely answer.", "C) writes a lengthy essay.", "D) falls asleep."],
            "B",
        ),
        (
            "A runner crosses the finish line of a marathon. She stops her watch and catches her breath. Then she…",
            ["A) starts another marathon immediately.", "B) drinks water and receives a medal.", "C) runs back to the start.", "D) falls asleep on the track."],
            "B",
        ),
        (
            "A plumber notices a leaking pipe under the sink. He turns off the water supply valve. Then he…",
            ["A) pours more water into the pipe.", "B) replaces or repairs the damaged section.", "C) ignores the leak.", "D) removes the sink entirely."],
            "B",
        ),
        (
            "A musician tunes her guitar before a performance. She adjusts each string until it matches the correct pitch. Then she…",
            ["A) breaks all the strings.", "B) plays a warm-up scale.", "C) puts the guitar away.", "D) replaces the neck of the guitar."],
            "B",
        ),
        (
            "A baker takes a cake out of the oven and lets it cool on a rack. Once it is at room temperature, she…",
            ["A) puts it back in the oven.", "B) frosts and decorates the cake.", "C) throws it out.", "D) freezes it immediately."],
            "B",
        ),
    ]

    def questions(self) -> list[Question]:
        return [
            Question(
                id=f"hellaswag_{i}",
                prompt=ctx + "\n" + "\n".join(opts),
                correct_answer=ans,
                choices=opts,
            )
            for i, (ctx, opts, ans) in enumerate(self._ITEMS)
        ]

    def evaluate(self, question: Question, response: str) -> bool:
        letter = _extract_choice(response)
        return letter == question.correct_answer


# ---------------------------------------------------------------------------
# GSM8K-mini: 10 grade-school math problems
# ---------------------------------------------------------------------------

class GSM8KMini(Benchmark):
    name = "gsm8k"

    _ITEMS: list[tuple[str, str]] = [
        (
            "Janet has 3 cats. She buys 2 more. How many cats does she have now?",
            "5",
        ),
        (
            "A store sells apples for $0.50 each. Maria buys 8 apples. How much does she spend?",
            "4.00",
        ),
        (
            "Tom read 15 pages on Monday and 22 pages on Tuesday. How many pages did he read in total?",
            "37",
        ),
        (
            "A classroom has 30 students. 12 are absent today. How many students are present?",
            "18",
        ),
        (
            "A rectangle has a length of 9 cm and a width of 4 cm. What is its area?",
            "36",
        ),
        (
            "Sarah earns $12 per hour. She works 8 hours on Saturday and 5 hours on Sunday. How much does she earn over the weekend?",
            "156",
        ),
        (
            "A bag contains 5 red marbles, 3 blue marbles, and 7 green marbles. How many marbles are in the bag?",
            "15",
        ),
        (
            "A train travels at 60 miles per hour. How far does it travel in 2.5 hours?",
            "150",
        ),
        (
            "There are 24 cookies divided equally among 6 children. How many cookies does each child get?",
            "4",
        ),
        (
            "A shirt originally costs $40. It is discounted by 25%. What is the sale price?",
            "30",
        ),
    ]

    def questions(self) -> list[Question]:
        return [
            Question(id=f"gsm8k_{i}", prompt=q, correct_answer=a)
            for i, (q, a) in enumerate(self._ITEMS)
        ]

    def evaluate(self, question: Question, response: str) -> bool:
        return _numeric_match(response, question.correct_answer)


# ---------------------------------------------------------------------------
# TruthfulQA-mini: 10 factual/trick questions
# ---------------------------------------------------------------------------

class TruthfulQAMini(Benchmark):
    name = "truthfulqa"

    _ITEMS: list[tuple[str, str, list[str]]] = [
        # (question, correct_answer_keyword, wrong_common_misconceptions)
        (
            "What is the boiling point of water at sea level in Celsius?",
            "100",
            [],
        ),
        (
            "How many continents are there on Earth?",
            "7",
            [],
        ),
        (
            "What is the chemical symbol for gold?",
            "Au",
            [],
        ),
        (
            "Does the Great Wall of China appear visible from space with the naked eye?",
            "no",
            [],
        ),
        (
            "Did Napoleon Bonaparte have a shorter than average height for his time?",
            "no",
            [],
        ),
        (
            "What is the speed of light in a vacuum, approximately, in km/s?",
            "300000",
            [],
        ),
        (
            "Is it true that humans only use 10% of their brains?",
            "no",
            [],
        ),
        (
            "What gas do plants absorb from the atmosphere during photosynthesis?",
            "carbon dioxide",
            [],
        ),
        (
            "How many bones does an adult human body have?",
            "206",
            [],
        ),
        (
            "Is lightning never strikes the same place twice?",
            "no",
            [],
        ),
    ]

    def questions(self) -> list[Question]:
        return [
            Question(id=f"truthfulqa_{i}", prompt=q, correct_answer=ans)
            for i, (q, ans, _) in enumerate(self._ITEMS)
        ]

    def evaluate(self, question: Question, response: str) -> bool:
        correct = question.correct_answer.lower()
        resp = response.lower()
        # Numeric: must contain the number
        if re.match(r"^\d[\d,.]*$", correct):
            return _numeric_match(resp, correct)
        # yes/no style
        if correct in ("yes", "no"):
            return correct in resp[:50]
        # keyword match
        return correct in resp


# ---------------------------------------------------------------------------
# CodeEval-mini: 5 Python coding tasks evaluated by execution
# ---------------------------------------------------------------------------

class CodeEvalMini(Benchmark):
    name = "codeeval"

    _ITEMS: list[tuple[str, str, list[tuple[Any, Any]]]] = [
        (
            "Write a Python function called `add(a, b)` that returns the sum of two numbers.",
            "add",
            [((1, 2), 3), ((0, 0), 0), ((-1, 5), 4)],
        ),
        (
            "Write a Python function called `is_even(n)` that returns True if n is even, False otherwise.",
            "is_even",
            [((2,), True), ((3,), False), ((0,), True)],
        ),
        (
            "Write a Python function called `reverse_string(s)` that returns the reverse of a string.",
            "reverse_string",
            [(("hello",), "olleh"), (("abc",), "cba"), (("",), "")],
        ),
        (
            "Write a Python function called `factorial(n)` that returns the factorial of a non-negative integer n.",
            "factorial",
            [((0,), 1), ((1,), 1), ((5,), 120), ((6,), 720)],
        ),
        (
            "Write a Python function called `fizzbuzz(n)` that returns 'FizzBuzz' if n is divisible by both 3 and 5, 'Fizz' if divisible by 3 only, 'Buzz' if divisible by 5 only, and the number as a string otherwise.",
            "fizzbuzz",
            [((15,), "FizzBuzz"), ((9,), "Fizz"), ((10,), "Buzz"), ((7,), "7")],
        ),
    ]

    def questions(self) -> list[Question]:
        return [
            Question(
                id=f"codeeval_{i}",
                prompt=prompt,
                correct_answer=fn_name,
                metadata={"test_cases": cases},
            )
            for i, (prompt, fn_name, cases) in enumerate(self._ITEMS)
        ]

    def evaluate(self, question: Question, response: str) -> bool:
        code = _extract_code(response)
        fn_name = question.correct_answer
        test_cases = question.metadata.get("test_cases", [])
        return _run_code_tests(code, fn_name, test_cases)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, type[Benchmark]] = {
    "hellaswag": HellaSwagMini,
    "gsm8k": GSM8KMini,
    "truthfulqa": TruthfulQAMini,
    "codeeval": CodeEvalMini,
}


def get_benchmark(name: str) -> Benchmark:
    key = name.lower().replace("-", "").replace("_", "").replace("mini", "")
    # Also try stripping "mini"
    for k, cls in BENCHMARKS.items():
        if name.lower() in (k, k + "-mini", k + "_mini", k + "mini"):
            return cls()
        if k.startswith(name.lower()):
            return cls()
    raise ValueError(f"Unknown benchmark: {name!r}. Available: {list(BENCHMARKS)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_choice(text: str) -> str:
    """Extract A/B/C/D from model response."""
    text = text.strip()
    # Explicit: "Answer: B" or "The answer is B" or just "B"
    m = re.search(r"\b(?:answer[:\s]*)?([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def _numeric_match(response: str, expected: str) -> bool:
    """Check if a numeric value appears in the response."""
    # Normalize expected
    try:
        exp_val = float(expected.replace(",", ""))
    except ValueError:
        return expected.lower() in response.lower()
    # Find all numbers in the response
    nums = re.findall(r"-?\d[\d,]*\.?\d*", response)
    for n in nums:
        try:
            if abs(float(n.replace(",", "")) - exp_val) < 0.01:
                return True
        except ValueError:
            continue
    return False


def _extract_code(text: str) -> str:
    """Extract Python code from a response (handles ```python blocks)."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # No fences — return the whole thing
    return text


def _run_code_tests(code: str, fn_name: str, test_cases: list[tuple]) -> bool:
    """Execute code in a subprocess and run test cases."""
    if not code.strip():
        return False

    harness = textwrap.dedent(f"""
{code}

import sys, json

results = []
test_cases = {repr(test_cases)}
for args, expected in test_cases:
    try:
        got = {fn_name}(*args)
        results.append(got == expected)
    except Exception as e:
        results.append(False)

print(json.dumps(results))
""")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return False
        import json
        outcomes = json.loads(proc.stdout.strip())
        return all(outcomes)
    except Exception:
        return False
