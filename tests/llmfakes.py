"""Scripted stand-ins for a model provider, so no test touches a network."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tailor.llm import Message, Usage

Handler = Callable[[list[Message], dict[str, Any]], dict[str, Any]]


class ScriptedLLM:
    """Answers each call from a handler, or from a list of answers in order; records the calls."""

    provider = "openai"
    seed: int | None = None
    supports_seed = True

    def __init__(self, answers: list[dict[str, Any]] | Handler) -> None:
        self._answers = answers
        self.usage = Usage()
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    def complete_json(
        self, messages: list[Message], schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        self.calls.append((messages, schema))
        if callable(self._answers):
            answer = self._answers(messages, schema)
        else:
            if len(self.calls) > len(self._answers):
                raise AssertionError(f"unexpected model call number {len(self.calls)}")
            answer = self._answers[len(self.calls) - 1]
        self.usage.add(10, 5)
        return answer
