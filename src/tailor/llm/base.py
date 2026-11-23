"""The model interface: one call that returns JSON matching a schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, TypedDict

SEED_PROVIDERS: Final = ("openai",)
"""Providers whose API takes a seed. The Anthropic Messages API has none."""


class LLMError(Exception):
    """A model call failed, or the model's answer could not be used."""


class ConfigError(LLMError):
    """The model is not configured, or the configuration is wrong."""


class Message(TypedDict):
    """One chat message. ``system`` messages are folded into the provider's system field."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True)
class Usage:
    """What the calls made through one client have cost, in calls and tokens."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    replayed_calls: int = 0

    def add(self, input_tokens: int, output_tokens: int, *, replayed: bool = False) -> None:
        """Count one call."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if replayed:
            self.replayed_calls += 1

    def to_dict(self) -> dict[str, int]:
        """The usage as plain JSON values."""
        return {
            "calls": self.calls,
            "replayed_calls": self.replayed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class LLM(Protocol):
    """What the pipeline needs from a model provider."""

    usage: Usage
    provider: str
    seed: int | None
    supports_seed: bool

    def complete_json(
        self, messages: list[Message], schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        """Return the model's answer as an object that follows ``schema``.

        Raises:
            LLMError: The call failed, the model refused, or the answer is not a JSON object.
        """
        ...
