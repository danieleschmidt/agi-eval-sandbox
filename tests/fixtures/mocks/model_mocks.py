"""Mock model providers for testing."""

import asyncio
import random
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock

class MockModelProvider:
    """Mock model provider for testing without API calls."""
    
    def __init__(
        self,
        name: str = "mock-model",
        provider: str = "mock",
        latency_ms: int = 100,
        error_rate: float = 0.0,
        responses: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.provider = provider
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.responses = responses or {}
        self.call_count = 0
        self.call_history = []
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        self.call_count += 1
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})
        
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Simulate errors
        if random.random() < self.error_rate:
            raise Exception(f"Mock API error from {self.name}")
        
        # Return predefined response or generate mock response
        if prompt in self.responses:
            return self.responses[prompt]
        
        # Generate contextually appropriate mock responses
        if "multiple choice" in prompt.lower():
            return random.choice(["A", "B", "C", "D"])
        elif "true or false" in prompt.lower():
            return random.choice(["True", "False"])
        elif "code" in prompt.lower():
            return "def solution():\n    return 42"
        else:
            return f"Mock response from {self.name}"
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate batch responses synchronously."""
        return [
            asyncio.run(self.generate(prompt, **kwargs)) 
            for prompt in prompts
        ]
    
    def reset(self):
        """Reset call history and counters."""
        self.call_count = 0
        self.call_history.clear()

class MockEvaluationResult:
    """Mock evaluation result for testing."""
    
    def __init__(
        self,
        score: float = 0.8,
        passed: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.score = score
        self.passed = passed
        self.metadata = metadata or {}
        self.timestamp = "2024-01-01T00:00:00Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "score": self.score,
            "passed": self.passed,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class MockBenchmark:
    """Mock benchmark for testing."""
    
    def __init__(
        self,
        name: str = "mock-benchmark",
        questions: Optional[List[Dict]] = None,
        expected_score: float = 0.8
    ):
        self.name = name
        self.questions = questions or self._generate_mock_questions()
        self.expected_score = expected_score
    
    def _generate_mock_questions(self) -> List[Dict]:
        """Generate sample questions."""
        return [
            {
                "id": f"q{i}",
                "question": f"Mock question {i}",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "category": "general"
            }
            for i in range(5)
        ]
    
    async def evaluate(self, model: MockModelProvider) -> MockEvaluationResult:
        """Run mock evaluation."""
        # Simulate evaluation time
        await asyncio.sleep(0.1)
        
        # Generate responses for all questions
        responses = []
        for question in self.questions:
            response = await model.generate(question["question"])
            responses.append(response)
        
        # Calculate mock score
        correct = sum(1 for r in responses if r == "A")  # Simplified scoring
        score = correct / len(responses)
        
        return MockEvaluationResult(
            score=score,
            passed=score >= 0.5,
            metadata={
                "total_questions": len(self.questions),
                "correct_answers": correct,
                "responses": responses
            }
        )