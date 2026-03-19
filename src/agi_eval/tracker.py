"""ResultTracker — JSON-based run history."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResultTracker:
    """
    Persists evaluation run history to a JSON file.

    Each record stores: timestamp, model, benchmark, accuracy, latency, metadata.
    """

    def __init__(self, path: str | Path = "results.json"):
        self.path = Path(path)

    def record(
        self,
        model: str,
        benchmark: str,
        accuracy: float,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a result record and persist to disk."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "benchmark": benchmark,
            "accuracy": round(accuracy, 4),
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata or {},
        }
        history = self.load()
        history.append(entry)
        self.path.write_text(json.dumps(history, indent=2))
        return entry

    def load(self) -> list[dict[str, Any]]:
        """Load existing history or return empty list."""
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary grouped by model+benchmark (latest run per pair)."""
        history = self.load()
        seen: dict[str, dict[str, Any]] = {}
        for entry in history:
            key = f"{entry['model']}|{entry['benchmark']}"
            seen[key] = entry  # last write wins
        return list(seen.values())

    def clear(self) -> None:
        """Delete all stored results."""
        if self.path.exists():
            self.path.unlink()
