"""Tests for benchmark question loading and evaluation logic."""

import pytest
from agi_eval.benchmarks import (
    HellaSwagMini,
    GSM8KMini,
    TruthfulQAMini,
    CodeEvalMini,
    get_benchmark,
    BENCHMARKS,
)


class TestHellaSwag:
    def setup_method(self):
        self.bench = HellaSwagMini()

    def test_loads_20_questions(self):
        qs = self.bench.questions()
        assert len(qs) == 20

    def test_each_question_has_4_choices(self):
        for q in self.bench.questions():
            assert q.choices is not None
            assert len(q.choices) == 4

    def test_correct_letter_evaluates_true(self):
        q = self.bench.questions()[0]
        assert self.bench.evaluate(q, q.correct_answer)

    def test_wrong_letter_evaluates_false(self):
        q = self.bench.questions()[0]
        wrong = "A" if q.correct_answer != "A" else "C"
        assert not self.bench.evaluate(q, wrong)

    def test_answer_in_sentence_evaluates_correctly(self):
        q = self.bench.questions()[0]
        text = f"The answer is {q.correct_answer}."
        assert self.bench.evaluate(q, text)


class TestGSM8K:
    def setup_method(self):
        self.bench = GSM8KMini()

    def test_loads_10_questions(self):
        assert len(self.bench.questions()) == 10

    def test_numeric_answer_correct(self):
        # Q0: 3+2=5
        q = self.bench.questions()[0]
        assert self.bench.evaluate(q, "The answer is 5.")

    def test_wrong_number_fails(self):
        q = self.bench.questions()[0]
        assert not self.bench.evaluate(q, "The answer is 99.")

    def test_decimal_match(self):
        # Q1: 8 * $0.50 = $4.00
        q = self.bench.questions()[1]
        assert self.bench.evaluate(q, "She spends $4.00 in total.")

    def test_all_have_numeric_answers(self):
        for q in self.bench.questions():
            assert q.correct_answer.replace(".", "").isdigit() or q.correct_answer.replace(".", "").replace(",", "").isdigit()


class TestTruthfulQA:
    def setup_method(self):
        self.bench = TruthfulQAMini()

    def test_loads_10_questions(self):
        assert len(self.bench.questions()) == 10

    def test_boiling_point_correct(self):
        q = self.bench.questions()[0]  # "100"
        assert self.bench.evaluate(q, "Water boils at 100 degrees Celsius.")

    def test_boiling_point_wrong(self):
        q = self.bench.questions()[0]
        assert not self.bench.evaluate(q, "Water boils at 212 degrees.")

    def test_no_answer(self):
        # "Does the Great Wall appear visible from space?" → "no"
        q = self.bench.questions()[3]
        assert self.bench.evaluate(q, "No, it is not visible from space.")

    def test_yes_answer_for_no_question_fails(self):
        q = self.bench.questions()[3]
        assert not self.bench.evaluate(q, "Yes, it is clearly visible.")

    def test_chemical_symbol_au(self):
        q = self.bench.questions()[2]  # correct_answer = "Au"
        assert self.bench.evaluate(q, "The chemical symbol for gold is Au.")


class TestCodeEval:
    def setup_method(self):
        self.bench = CodeEvalMini()

    def test_loads_5_questions(self):
        assert len(self.bench.questions()) == 5

    def test_add_function_correct(self):
        q = self.bench.questions()[0]
        code = "def add(a, b):\n    return a + b"
        assert self.bench.evaluate(q, f"```python\n{code}\n```")

    def test_add_function_wrong(self):
        q = self.bench.questions()[0]
        code = "def add(a, b):\n    return a - b"
        assert not self.bench.evaluate(q, f"```python\n{code}\n```")

    def test_is_even_correct(self):
        q = self.bench.questions()[1]
        code = "def is_even(n):\n    return n % 2 == 0"
        assert self.bench.evaluate(q, f"```python\n{code}\n```")

    def test_factorial_correct(self):
        q = self.bench.questions()[3]
        code = (
            "def factorial(n):\n"
            "    if n == 0: return 1\n"
            "    return n * factorial(n - 1)"
        )
        assert self.bench.evaluate(q, f"```python\n{code}\n```")

    def test_fizzbuzz_correct(self):
        q = self.bench.questions()[4]
        code = (
            "def fizzbuzz(n):\n"
            "    if n % 15 == 0: return 'FizzBuzz'\n"
            "    if n % 3 == 0: return 'Fizz'\n"
            "    if n % 5 == 0: return 'Buzz'\n"
            "    return str(n)"
        )
        assert self.bench.evaluate(q, f"```python\n{code}\n```")

    def test_broken_code_fails(self):
        q = self.bench.questions()[0]
        assert not self.bench.evaluate(q, "```python\nthis is not python\n```")

    def test_empty_response_fails(self):
        q = self.bench.questions()[0]
        assert not self.bench.evaluate(q, "")


class TestRegistry:
    def test_all_benchmarks_in_registry(self):
        assert "hellaswag" in BENCHMARKS
        assert "gsm8k" in BENCHMARKS
        assert "truthfulqa" in BENCHMARKS
        assert "codeeval" in BENCHMARKS

    def test_get_benchmark_by_name(self):
        b = get_benchmark("hellaswag")
        assert isinstance(b, HellaSwagMini)

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError):
            get_benchmark("nonexistent_benchmark_xyz")
