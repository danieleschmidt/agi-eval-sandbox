"""Test data factories for generating test objects."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import factory
from factory import fuzzy


class ModelFactory(factory.Factory):
    """Factory for creating test model objects."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    name = factory.Sequence(lambda n: f"test-model-{n}")
    provider = fuzzy.FuzzyChoice(["openai", "anthropic", "google", "local"])
    version = factory.Faker("word")
    metadata = factory.LazyFunction(dict)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())


class BenchmarkFactory(factory.Factory):
    """Factory for creating test benchmark objects."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    name = fuzzy.FuzzyChoice(["mmlu", "humaneval", "truthfulqa", "hellaswag"])
    version = factory.Faker("numerify", text="#.#.#")
    type = fuzzy.FuzzyChoice(["multiple_choice", "code_generation", "text_generation"])
    config = factory.LazyFunction(dict)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())


class QuestionFactory(factory.Factory):
    """Factory for creating test question objects."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    category = fuzzy.FuzzyChoice(["mathematics", "science", "history", "literature"])
    question = factory.Faker("sentence", nb_words=10)
    choices = factory.LazyFunction(lambda: [factory.Faker("word").generate() for _ in range(4)])
    correct_answer = fuzzy.FuzzyInteger(0, 3)
    difficulty = fuzzy.FuzzyChoice(["easy", "medium", "hard"])


class EvaluationFactory(factory.Factory):
    """Factory for creating test evaluation objects."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    model_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    benchmark_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    status = fuzzy.FuzzyChoice(["pending", "running", "completed", "failed"])
    config = factory.LazyFunction(lambda: {
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_p": 1.0
    })
    started_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    completed_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())


class ResultFactory(factory.Factory):
    """Factory for creating test result objects."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    evaluation_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    metrics = factory.LazyFunction(lambda: {
        "accuracy": factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate(),
        "total_questions": factory.Faker("pyint", min_value=10, max_value=1000).generate(),
        "execution_time": factory.Faker("pyfloat", min_value=1.0, max_value=300.0).generate()
    })
    raw_data_path = factory.Faker("file_path", extension="json")
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())


class MockModelResponseFactory(factory.Factory):
    """Factory for creating mock model API responses."""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: f"chatcmpl-{uuid.uuid4().hex[:10]}")
    object = "chat.completion"
    created = factory.LazyFunction(lambda: int(datetime.now().timestamp()))
    model = factory.Faker("word")
    choices = factory.LazyFunction(lambda: [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": factory.Faker("sentence").generate()
        },
        "finish_reason": "stop"
    }])
    usage = factory.LazyFunction(lambda: {
        "prompt_tokens": factory.Faker("pyint", min_value=10, max_value=1000).generate(),
        "completion_tokens": factory.Faker("pyint", min_value=10, max_value=500).generate(),
        "total_tokens": factory.Faker("pyint", min_value=20, max_value=1500).generate()
    })


def create_sample_dataset(size: int = 10, benchmark_type: str = "multiple_choice") -> List[Dict[str, Any]]:
    """Create a sample dataset for testing."""
    questions = []
    for i in range(size):
        question = QuestionFactory.create()
        question["benchmark_type"] = benchmark_type
        questions.append(question)
    return questions


def create_evaluation_with_results(
    num_questions: int = 5,
    accuracy: float = 0.8
) -> Dict[str, Any]:
    """Create a complete evaluation with questions and results."""
    model = ModelFactory.create()
    benchmark = BenchmarkFactory.create()
    evaluation = EvaluationFactory.create(
        model_id=model["id"],
        benchmark_id=benchmark["id"],
        status="completed"
    )
    
    questions = create_sample_dataset(num_questions)
    correct_answers = int(num_questions * accuracy)
    
    # Create results with specified accuracy
    results = ResultFactory.create(
        evaluation_id=evaluation["id"],
        metrics={
            "accuracy": accuracy,
            "total_questions": num_questions,
            "correct_answers": correct_answers,
            "execution_time": factory.Faker("pyfloat", min_value=10.0, max_value=60.0).generate()
        }
    )
    
    return {
        "model": model,
        "benchmark": benchmark,
        "evaluation": evaluation,
        "questions": questions,
        "results": results
    }