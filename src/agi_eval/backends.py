"""LLM backends: MockBackend for testing, OpenAIBackend for real runs."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResponse:
    text: str
    latency_ms: float
    model: str
    tokens_used: int = 0


class Backend(ABC):
    """Abstract backend for completions."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Return a completion for the given prompt."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# MockBackend — deterministic answers for unit tests
# ---------------------------------------------------------------------------

class MockBackend(Backend):
    """
    Deterministic mock backend for testing.

    Pass ``answers`` to override per-question, or use a single callable.
    If nothing is provided, returns a sensible default per question type.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default_answer: str = "B",
        model: str = "mock-model",
        latency_ms: float = 10.0,
    ):
        self._answers = answers or {}
        self._default = default_answer
        self._model = model
        self._latency_ms = latency_ms

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Allow callers to supply a question_id hint via kwargs
        question_id = kwargs.get("question_id", "")
        answer = self._answers.get(question_id, self._default)
        return CompletionResponse(
            text=answer,
            latency_ms=self._latency_ms,
            model=self._model,
        )


# ---------------------------------------------------------------------------
# OpenAIBackend — any OpenAI-compatible REST endpoint
# ---------------------------------------------------------------------------

class OpenAIBackend(Backend):
    """
    Backend for any OpenAI-compatible API.

    Args:
        model:       Model identifier (e.g. "gpt-4o", "mistral-7b").
        api_key:     API key.
        base_url:    Override base URL for local/custom endpoints
                     (default: https://api.openai.com/v1).
        temperature: Sampling temperature (default 0.0 for determinism).
        max_tokens:  Max completion tokens.
        timeout:     HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        import urllib.request, urllib.error, json as _json

        payload = _json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"API error {exc.code}: {body}") from exc
        latency_ms = (time.monotonic() - t0) * 1000

        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return CompletionResponse(
            text=text,
            latency_ms=latency_ms,
            model=self._model,
            tokens_used=tokens,
        )
