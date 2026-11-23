"""Any OpenAI-compatible endpoint through the official SDK, with a JSON-schema response format.

This is also how a local Ollama is reached: ``base_url`` is ``http://localhost:11434/v1``. Only
``response_format`` carries the schema, plus a ``seed`` when one is set; no other sampling
parameters and no token limit are sent (Ollama and OpenAI disagree on the limit's name, and a limit
would cut JSON off mid-object).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Final

import openai

from tailor.llm.base import LLMError, Message, Usage

SCHEMA_NAME: Final = "response"
_THINK: Final = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
TIMEOUT_SECONDS: Final = 900.0


def build_request(
    messages: list[Message], schema: dict[str, Any], *, model: str, seed: int | None = None
) -> dict[str, Any]:
    """The keyword arguments of ``client.chat.completions.create`` for this call."""
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "schema": schema, "strict": True},
        },
    }
    if seed is not None:
        request["seed"] = seed
    return request


def parse_response(response: Any) -> tuple[dict[str, Any], int, int]:
    """Pick the JSON object and the token counts out of a chat completion."""
    if not response.choices:
        raise LLMError("the endpoint returned no choices")
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise LLMError(f"the model refused the request: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise LLMError("the answer was cut off at the token limit; it is not complete JSON")
    content = _THINK.sub("", choice.message.content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise LLMError(f"the answer is not valid JSON ({error}): {content[:200]!r}") from error
    if not isinstance(data, dict):
        raise LLMError("the answer is JSON but not an object")
    usage = response.usage
    return data, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)


class OpenAICompatibleLLM:
    """The OpenAI-compatible provider."""

    provider = "openai"
    supports_seed = True

    def __init__(
        self, base_url: str | None = None, client: Any | None = None, seed: int | None = None
    ) -> None:
        """Create the provider; the SDK client is built on first use unless one is given."""
        self.seed = seed
        self._base_url = base_url
        self._client = client
        self.usage = Usage()

    def _api(self) -> Any:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                if not self._base_url:
                    raise LLMError("OPENAI_API_KEY is not set (and no base_url is configured)")
                key = "none"  # a local server ignores the key, but the SDK insists on one
            self._client = openai.OpenAI(
                base_url=self._base_url, api_key=key, timeout=TIMEOUT_SECONDS
            )
        return self._client

    def complete_json(
        self, messages: list[Message], schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        """Call the endpoint with ``schema`` as a strict JSON-schema response format."""
        try:
            response = self._api().chat.completions.create(
                **build_request(messages, schema, model=model, seed=self.seed)
            )
        except openai.APIError as error:
            raise LLMError(f"the endpoint call failed: {error}") from error
        data, tokens_in, tokens_out = parse_response(response)
        self.usage.add(tokens_in, tokens_out)
        return data
