"""
Comprehensive unit tests for benchmark functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch

from agi_eval_sandbox.core.benchmarks import (
    Benchmark, Question, QuestionType, Score, 
    TruthfulQABenchmark, MMLUBenchmark, HumanEvalBenchmark, CustomBenchmark
)
from agi_eval_sandbox.core.exceptions import ValidationError


@pytest.mark.unit
class TestQuestion:
    """Test Question dataclass functionality."""
    
    def test_question_creation(self):
        """Test basic Question creation."""
        question = Question(
            id="test-q1",
            prompt="What is the capital of France?",
            correct_answer="Paris",
            question_type=QuestionType.SHORT_ANSWER,
            choices=["London", "Berlin", "Paris", "Madrid"],
            category="geography",
            difficulty="easy",
            metadata={"source": "test"}
        )
        
        assert question.id == "test-q1"
        assert question.prompt == "What is the capital of France?"
        assert question.correct_answer == "Paris"
        assert question.question_type == QuestionType.SHORT_ANSWER
        assert question.category == "geography"
        assert question.difficulty == "easy"
        assert len(question.choices) == 4
        assert question.metadata["source"] == "test"
    
    def test_question_defaults(self):
        """Test Question with default values."""
        question = Question(
            id="minimal-q1",
            prompt="Minimal question",
            correct_answer="Answer"
        )
        
        assert question.id == "minimal-q1"
        assert question.prompt == "Minimal question"
        assert question.correct_answer == "Answer"
        assert question.question_type == QuestionType.GENERATION
        assert question.choices is None
        assert question.category is None
        assert question.difficulty is None
        assert question.metadata is None
    
    def test_question_from_dict(self):
        """Test Question.from_dict class method."""
        data = {
            "id": "dict-q1",
            "prompt": "Test prompt from dict",
            "correct_answer": "Dict answer",
            "question_type": "multiple_choice",
            "choices": ["A", "B", "C", "D"],
            "category": "test_category",
            "difficulty": "medium",
            "metadata": {"test": True}
        }
        
        question = Question.from_dict(data)
        
        assert question.id == "dict-q1"
        assert question.prompt == "Test prompt from dict"
        assert question.correct_answer == "Dict answer"
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert question.choices == ["A", "B", "C", "D"]
        assert question.category == "test_category"
        assert question.difficulty == "medium"
        assert question.metadata["test"] is True
    
    def test_question_from_dict_minimal(self):
        """Test Question.from_dict with minimal data."""
        data = {
            "id": "minimal-dict",
            "prompt": "Minimal",
            "correct_answer": "Answer"
        }
        
        question = Question.from_dict(data)
        
        assert question.id == "minimal-dict"
        assert question.question_type == QuestionType.GENERATION
        assert question.metadata == {}
    
    def test_question_type_enum(self):
        """Test QuestionType enum values."""
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.TRUE_FALSE.value == "true_false"
        assert QuestionType.SHORT_ANSWER.value == "short_answer"
        assert QuestionType.CODING.value == "coding"
        assert QuestionType.GENERATION.value == "generation"
        
        # Test creation from string
        assert QuestionType("multiple_choice") == QuestionType.MULTIPLE_CHOICE
        assert QuestionType("generation") == QuestionType.GENERATION


@pytest.mark.unit
class TestScore:
    """Test Score dataclass functionality."""
    
    def test_score_creation(self):
        """Test basic Score creation."""
        score = Score(
            value=0.85,
            max_value=1.0,
            passed=True,
            explanation="Good answer",
            metadata={"confidence": 0.9}
        )
        
        assert score.value == 0.85
        assert score.max_value == 1.0
        assert score.passed is True
        assert score.explanation == "Good answer"
        assert score.metadata["confidence"] == 0.9
    
    def test_score_defaults(self):
        """Test Score with default values."""
        score = Score(value=0.5)
        
        assert score.value == 0.5
        assert score.max_value == 1.0
        assert score.passed is False
        assert score.explanation is None
        assert score.metadata is None
    
    def test_score_percentage_property(self):
        """Test Score.percentage property."""
        # Normal case
        score = Score(value=0.75, max_value=1.0)
        assert score.percentage == 75.0
        
        # Different max value
        score = Score(value=3.0, max_value=4.0)
        assert score.percentage == 75.0
        
        # Edge case: max_value is 0
        score = Score(value=1.0, max_value=0.0)
        assert score.percentage == 0.0
        
        # Negative values
        score = Score(value=-0.5, max_value=1.0)
        assert score.percentage == -50.0
    
    def test_score_boolean_conversion(self):
        """Test Score boolean interpretation."""
        passed_score = Score(value=0.8, passed=True)
        failed_score = Score(value=0.3, passed=False)
        
        assert passed_score.passed is True
        assert failed_score.passed is False


@pytest.mark.unit
class TestBenchmarkBase:
    """Test base Benchmark class functionality."""
    
    def test_benchmark_abstract_class(self):
        """Test that Benchmark is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Benchmark("test", "1.0")
    
    def test_benchmark_subclass_implementation(self):
        """Test custom benchmark implementation."""
        class TestBenchmark(Benchmark):
            def load_questions(self):
                return [
                    Question(
                        id="test-1",
                        prompt="Test question 1",
                        correct_answer="Answer 1"
                    ),
                    Question(
                        id="test-2", 
                        prompt="Test question 2",
                        correct_answer="Answer 2"
                    )
                ]
            
            def evaluate_response(self, question, response):
                # Simple exact match evaluation
                passed = response.strip().lower() == question.correct_answer.lower()
                return Score(
                    value=1.0 if passed else 0.0,
                    passed=passed,
                    explanation=f"Expected '{question.correct_answer}', got '{response}'"
                )
        
        benchmark = TestBenchmark("test_benchmark", "1.0")
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.version == "1.0"
        
        questions = benchmark.get_questions()
        assert len(questions) == 2
        assert questions[0].id == "test-1"
        
        # Test evaluation
        question = questions[0]
        
        # Correct answer
        score = benchmark.evaluate_response(question, "Answer 1")
        assert score.passed is True
        assert score.value == 1.0
        
        # Wrong answer
        score = benchmark.evaluate_response(question, "Wrong Answer")
        assert score.passed is False
        assert score.value == 0.0
    
    def test_benchmark_get_sample_questions(self):
        """Test get_sample_questions functionality."""
        class SampleTestBenchmark(Benchmark):
            def load_questions(self):
                return [
                    Question(id=f"q{i}", prompt=f"Question {i}", correct_answer=f"Answer {i}")
                    for i in range(10)
                ]
            
            def evaluate_response(self, question, response):
                return Score(value=0.5, passed=True)
        
        benchmark = SampleTestBenchmark("sample_test")
        
        # Get sample smaller than total
        sample = benchmark.get_sample_questions(n=5, seed=42)
        assert len(sample) == 5
        assert all(isinstance(q, Question) for q in sample)
        
        # Get sample larger than total
        large_sample = benchmark.get_sample_questions(n=15, seed=42)
        assert len(large_sample) == 10  # Should return all questions
        
        # Test reproducibility with seed
        sample1 = benchmark.get_sample_questions(n=5, seed=42)
        sample2 = benchmark.get_sample_questions(n=5, seed=42)
        assert [q.id for q in sample1] == [q.id for q in sample2]
        
        # Test different seed gives different result
        sample3 = benchmark.get_sample_questions(n=5, seed=123)
        assert [q.id for q in sample1] != [q.id for q in sample3] or len(sample1) < 2  # Could be same if very few questions


@pytest.mark.unit
class TestTruthfulQABenchmark:
    """Test TruthfulQA benchmark implementation."""
    
    def test_truthfulqa_creation(self):
        """Test TruthfulQA benchmark creation."""
        benchmark = TruthfulQABenchmark()
        
        assert benchmark.name == "truthfulqa"
        assert benchmark.version == "1.0"
    
    def test_truthfulqa_questions_loading(self):
        """Test TruthfulQA questions loading."""
        benchmark = TruthfulQABenchmark()
        
        questions = benchmark.get_questions()
        
        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)
        assert all(q.id for q in questions)  # All questions should have IDs
        assert all(q.prompt for q in questions)  # All questions should have prompts
        assert all(q.correct_answer for q in questions)  # All should have answers
    
    def test_truthfulqa_evaluation(self):
        """Test TruthfulQA response evaluation."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.get_questions()
        
        if questions:
            question = questions[0]
            
            # Test with correct answer (if we know it)
            score = benchmark.evaluate_response(question, question.correct_answer)
            assert isinstance(score, Score)
            assert 0.0 <= score.value <= 1.0
            
            # Test with obviously wrong answer
            wrong_score = benchmark.evaluate_response(question, "This is definitely wrong")
            assert isinstance(wrong_score, Score)
            assert 0.0 <= wrong_score.value <= 1.0
    
    def test_truthfulqa_question_categories(self):
        """Test TruthfulQA question categories."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.get_questions()
        
        # Should have some categorized questions
        categorized = [q for q in questions if q.category]
        assert len(categorized) > 0
        
        # Check that categories are reasonable strings
        categories = set(q.category for q in categorized if q.category)
        assert all(isinstance(cat, str) and len(cat) > 0 for cat in categories)


@pytest.mark.unit
class TestMMLUBenchmark:
    """Test MMLU benchmark implementation."""
    
    def test_mmlu_creation(self):
        """Test MMLU benchmark creation."""
        benchmark = MMLUBenchmark()
        
        assert benchmark.name == "mmlu"
        assert benchmark.version == "1.0"
    
    def test_mmlu_questions_loading(self):
        """Test MMLU questions loading."""
        benchmark = MMLUBenchmark()
        
        questions = benchmark.get_questions()
        
        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)
        
        # MMLU questions should be multiple choice
        mc_questions = [q for q in questions if q.question_type == QuestionType.MULTIPLE_CHOICE]
        assert len(mc_questions) > 0
        
        # Should have choices
        questions_with_choices = [q for q in questions if q.choices and len(q.choices) > 1]
        assert len(questions_with_choices) > 0
    
    def test_mmlu_subjects(self):
        """Test MMLU subject categories."""
        benchmark = MMLUBenchmark()
        questions = benchmark.get_questions()
        
        # Should have questions from multiple subjects
        subjects = set(q.category for q in questions if q.category)
        assert len(subjects) > 1  # Should have multiple subjects
        
        # Common MMLU subjects should be present
        subject_names = [s.lower() for s in subjects]
        expected_subjects = ["math", "science", "history", "philosophy"]
        found_subjects = [subj for subj in expected_subjects if any(exp in subject for subject in subject_names for exp in [subj])]
        assert len(found_subjects) > 0  # Should find at least some expected subjects
    
    def test_mmlu_evaluation(self):
        """Test MMLU response evaluation."""
        benchmark = MMLUBenchmark()
        questions = benchmark.get_questions()
        
        # Find a multiple choice question
        mc_question = next((q for q in questions if q.question_type == QuestionType.MULTIPLE_CHOICE and q.choices), None)
        
        if mc_question:
            # Test with correct answer
            correct_score = benchmark.evaluate_response(mc_question, mc_question.correct_answer)
            assert isinstance(correct_score, Score)
            assert correct_score.value >= 0.0
            
            # Test with one of the choices (but not necessarily correct)
            if mc_question.choices:
                choice_score = benchmark.evaluate_response(mc_question, mc_question.choices[0])
                assert isinstance(choice_score, Score)
                assert 0.0 <= choice_score.value <= 1.0


@pytest.mark.unit
class TestHumanEvalBenchmark:
    """Test HumanEval benchmark implementation."""
    
    def test_humaneval_creation(self):
        """Test HumanEval benchmark creation."""
        benchmark = HumanEvalBenchmark()
        
        assert benchmark.name == "humaneval"
        assert benchmark.version == "1.0"
    
    def test_humaneval_questions_loading(self):
        """Test HumanEval questions loading."""
        benchmark = HumanEvalBenchmark()
        
        questions = benchmark.get_questions()
        
        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)
        
        # HumanEval questions should be coding type
        coding_questions = [q for q in questions if q.question_type == QuestionType.CODING]
        assert len(coding_questions) > 0
    
    def test_humaneval_coding_structure(self):
        """Test HumanEval coding question structure."""
        benchmark = HumanEvalBenchmark()
        questions = benchmark.get_questions()
        
        coding_questions = [q for q in questions if q.question_type == QuestionType.CODING]
        
        if coding_questions:
            question = coding_questions[0]
            
            # Should have function signature or similar in prompt
            assert "def " in question.prompt or "function" in question.prompt.lower()
            
            # Should have some kind of expected solution
            assert question.correct_answer
            assert len(question.correct_answer) > 0
    
    def test_humaneval_evaluation(self):
        """Test HumanEval response evaluation."""
        benchmark = HumanEvalBenchmark()
        questions = benchmark.get_questions()
        
        coding_questions = [q for q in questions if q.question_type == QuestionType.CODING]
        
        if coding_questions:
            question = coding_questions[0]
            
            # Test evaluation (this is a basic test, real evaluation would be more complex)
            score = benchmark.evaluate_response(question, "def solution(): pass")
            assert isinstance(score, Score)
            assert 0.0 <= score.value <= 1.0
            
            # Test with obviously incomplete code
            bad_score = benchmark.evaluate_response(question, "this is not code")
            assert isinstance(bad_score, Score)
            assert 0.0 <= bad_score.value <= 1.0


@pytest.mark.unit
class TestCustomBenchmark:
    """Test CustomBenchmark implementation."""
    
    def test_custom_benchmark_creation(self):
        """Test custom benchmark creation with questions."""
        questions = [
            Question(
                id="custom-1",
                prompt="What is 2+2?",
                correct_answer="4",
                question_type=QuestionType.SHORT_ANSWER,
                category="math"
            ),
            Question(
                id="custom-2",
                prompt="What color is the sky?",
                correct_answer="blue",
                question_type=QuestionType.SHORT_ANSWER,
                category="general"
            )
        ]
        
        benchmark = CustomBenchmark("my-custom-benchmark", questions)
        
        assert benchmark.name == "my-custom-benchmark"
        assert benchmark.version == "1.0"
        
        loaded_questions = benchmark.get_questions()
        assert len(loaded_questions) == 2
        assert loaded_questions[0].id == "custom-1"
        assert loaded_questions[1].id == "custom-2"
    
    def test_custom_benchmark_empty_questions(self):
        """Test custom benchmark with empty questions list."""
        benchmark = CustomBenchmark("empty-benchmark", [])
        
        questions = benchmark.get_questions()
        assert len(questions) == 0
    
    def test_custom_benchmark_evaluation(self):
        """Test custom benchmark evaluation."""
        questions = [
            Question(
                id="eval-test",
                prompt="Simple test question",
                correct_answer="correct",
                question_type=QuestionType.SHORT_ANSWER
            )
        ]
        
        benchmark = CustomBenchmark("eval-test-benchmark", questions)
        question = questions[0]
        
        # Test exact match (should pass)
        correct_score = benchmark.evaluate_response(question, "correct")
        assert correct_score.passed is True
        assert correct_score.value == 1.0
        
        # Test case-insensitive match
        case_score = benchmark.evaluate_response(question, "CORRECT")
        assert case_score.passed is True
        assert case_score.value == 1.0
        
        # Test with extra whitespace
        space_score = benchmark.evaluate_response(question, "  correct  ")
        assert space_score.passed is True
        assert space_score.value == 1.0
        
        # Test incorrect answer
        wrong_score = benchmark.evaluate_response(question, "incorrect")
        assert wrong_score.passed is False
        assert wrong_score.value == 0.0
    
    def test_custom_benchmark_partial_scoring(self):
        """Test custom benchmark with different question types."""
        questions = [
            Question(
                id="mc-test",
                prompt="Multiple choice test",
                correct_answer="B",
                question_type=QuestionType.MULTIPLE_CHOICE,
                choices=["A", "B", "C", "D"]
            ),
            Question(
                id="tf-test",
                prompt="True/false test",
                correct_answer="True",
                question_type=QuestionType.TRUE_FALSE,
                choices=["True", "False"]
            )
        ]
        
        benchmark = CustomBenchmark("mixed-benchmark", questions)
        
        # Test multiple choice
        mc_question = questions[0]
        mc_score = benchmark.evaluate_response(mc_question, "B")
        assert mc_score.passed is True
        
        # Test true/false
        tf_question = questions[1]
        tf_score = benchmark.evaluate_response(tf_question, "True")
        assert tf_score.passed is True
        
        tf_wrong_score = benchmark.evaluate_response(tf_question, "False")
        assert tf_wrong_score.passed is False
    
    def test_custom_benchmark_with_version(self):
        """Test custom benchmark with custom version."""
        questions = [
            Question(id="v-test", prompt="Version test", correct_answer="v1")
        ]
        
        benchmark = CustomBenchmark("version-test", questions, version="2.1")
        
        assert benchmark.name == "version-test"
        assert benchmark.version == "2.1"


@pytest.mark.unit
class TestBenchmarkIntegration:
    """Test benchmark integration scenarios."""
    
    def test_benchmark_consistency(self):
        """Test that benchmarks behave consistently."""
        benchmarks = [
            TruthfulQABenchmark(),
            MMLUBenchmark(),
            HumanEvalBenchmark()
        ]
        
        for benchmark in benchmarks:
            # All should have names and versions
            assert benchmark.name
            assert benchmark.version
            
            # All should load questions
            questions = benchmark.get_questions()
            assert len(questions) > 0
            
            # All questions should have required fields
            for question in questions[:5]:  # Test first 5 questions
                assert question.id
                assert question.prompt
                assert question.correct_answer
                assert isinstance(question.question_type, QuestionType)
            
            # All should be able to evaluate responses
            if questions:
                question = questions[0]
                score = benchmark.evaluate_response(question, "test response")
                assert isinstance(score, Score)
                assert 0.0 <= score.value <= score.max_value
    
    def test_benchmark_sampling_consistency(self):
        """Test that sampling works consistently across benchmarks."""
        benchmarks = [
            TruthfulQABenchmark(),
            CustomBenchmark("test", [
                Question(id=f"q{i}", prompt=f"Question {i}", correct_answer=f"Answer {i}")
                for i in range(20)
            ])
        ]
        
        for benchmark in benchmarks:
            questions = benchmark.get_questions()
            if len(questions) >= 5:
                # Sample should be reproducible with seed
                sample1 = benchmark.get_sample_questions(n=5, seed=42)
                sample2 = benchmark.get_sample_questions(n=5, seed=42)
                assert [q.id for q in sample1] == [q.id for q in sample2]
                
                # Sample should be subset of all questions
                all_ids = set(q.id for q in questions)
                sample_ids = set(q.id for q in sample1)
                assert sample_ids.issubset(all_ids)
    
    def test_benchmark_score_ranges(self):
        """Test that all benchmarks return scores in valid ranges."""
        benchmarks = [
            TruthfulQABenchmark(),
            MMLUBenchmark(),
            HumanEvalBenchmark(),
            CustomBenchmark("range-test", [
                Question(id="rt1", prompt="Range test", correct_answer="test")
            ])
        ]
        
        test_responses = [
            "",  # Empty response
            "test",  # Simple response
            "A very long response that might contain multiple sentences and various characters !@#$%^&*()",
            "12345",  # Numeric
            "True",  # Boolean-like
            "False"
        ]
        
        for benchmark in benchmarks:
            questions = benchmark.get_questions()
            if questions:
                question = questions[0]
                
                for response in test_responses:
                    score = benchmark.evaluate_response(question, response)
                    
                    # Score should be in valid range
                    assert isinstance(score, Score)
                    assert 0.0 <= score.value <= score.max_value
                    assert isinstance(score.passed, bool)
                    
                    # Explanation should be string or None
                    if score.explanation is not None:
                        assert isinstance(score.explanation, str)


@pytest.mark.unit
class TestBenchmarkEdgeCases:
    """Test edge cases and error conditions in benchmarks."""
    
    def test_custom_benchmark_invalid_questions(self):
        """Test custom benchmark with invalid question data."""
        # This should work - CustomBenchmark accepts any Question objects
        invalid_questions = [
            Question(id="", prompt="Empty ID", correct_answer="test"),  # Empty ID
            Question(id="valid", prompt="", correct_answer="test"),  # Empty prompt
            Question(id="valid2", prompt="Valid", correct_answer=""),  # Empty answer
        ]
        
        benchmark = CustomBenchmark("invalid-test", invalid_questions)
        questions = benchmark.get_questions()
        
        # Should still create benchmark and return questions
        assert len(questions) == 3
    
    def test_benchmark_evaluation_edge_cases(self):
        """Test benchmark evaluation with edge case responses."""
        question = Question(
            id="edge-test",
            prompt="Edge case test",
            correct_answer="normal answer"
        )
        
        benchmark = CustomBenchmark("edge-benchmark", [question])
        
        edge_cases = [
            None,  # None response - should be handled gracefully
            "",    # Empty string
            "   ",  # Whitespace only
            "\n\t\r",  # Various whitespace characters
            "normal answer",  # Exact match
            "NORMAL ANSWER",  # Case different
            "  normal answer  ",  # With whitespace
        ]
        
        for response in edge_cases:
            try:
                score = benchmark.evaluate_response(question, response)
                assert isinstance(score, Score)
                assert 0.0 <= score.value <= score.max_value
            except Exception as e:
                # Some edge cases might raise exceptions - that's acceptable
                # as long as they're handled gracefully
                assert isinstance(e, (TypeError, AttributeError, ValidationError))
    
    def test_question_from_dict_missing_fields(self):
        """Test Question.from_dict with missing required fields."""
        # Missing ID
        with pytest.raises(KeyError):
            Question.from_dict({
                "prompt": "Test",
                "correct_answer": "Answer"
            })
        
        # Missing prompt
        with pytest.raises(KeyError):
            Question.from_dict({
                "id": "test",
                "correct_answer": "Answer"
            })
        
        # Missing correct_answer
        with pytest.raises(KeyError):
            Question.from_dict({
                "id": "test",
                "prompt": "Test"
            })
    
    def test_question_from_dict_invalid_question_type(self):
        """Test Question.from_dict with invalid question type."""
        with pytest.raises(ValueError):
            Question.from_dict({
                "id": "test",
                "prompt": "Test",
                "correct_answer": "Answer",
                "question_type": "invalid_type"
            })
    
    def test_benchmark_lazy_loading(self):
        """Test that benchmark questions are loaded lazily."""
        class LazyTestBenchmark(Benchmark):
            def __init__(self):
                super().__init__("lazy-test")
                self.load_called = False
            
            def load_questions(self):
                self.load_called = True
                return [Question(id="lazy", prompt="Lazy test", correct_answer="lazy")]
            
            def evaluate_response(self, question, response):
                return Score(value=1.0, passed=True)
        
        benchmark = LazyTestBenchmark()
        
        # Questions shouldn't be loaded yet
        assert not benchmark.load_called
        assert benchmark._questions is None
        
        # First call should trigger loading
        questions = benchmark.get_questions()
        assert benchmark.load_called
        assert len(questions) == 1
        
        # Reset load flag
        benchmark.load_called = False
        
        # Second call should not trigger loading again
        questions2 = benchmark.get_questions()
        assert not benchmark.load_called  # Should not be called again
        assert questions == questions2