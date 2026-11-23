"""Anthropic through the official SDK: structured output through one tool.

The tool's input schema is the target schema, so the model's tool call is the answer. Which
``tool_choice`` is allowed depends on the model: the Claude Opus 5.5 and Sonnet 5.5 generations
(and the Fable and Mythos models) answer a forced ``tool_choice`` with a 400. For those the
request uses ``auto`` with a strict tool and an instruction to call it; for the others the call is
forced. A model not on the list that still rejects a forced choice is retried once with ``auto``.
No sampling parameters are sent: several models reject non-default ones.
"""

from __future__ import annotations

from typing import Any, Final

import anthropic

from tailor.llm.base import LLMError, Message, Usage

TOOL_NAME: Final = "respond"
MAX_TOKENS: Final = 16000
_NO_FORCED_TOOL_CHOICE: Final = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-fable-5-2",
    "claude-mythos-5-1",
    "claude-mythos-5-2",
)
_CALL_INSTRUCTION: Final = f"Answer by calling the {TOOL_NAME} tool exactly once."


def forces_tool(model: str) -> bool:
    """Whether a request to ``model`` may force the tool call."""
    return not model.startswith(_NO_FORCED_TOOL_CHOICE)


def build_request(
    messages: list[Message], schema: dict[str, Any], *, model: str, forced: bool | None = None
) -> dict[str, Any]:
    """The keyword arguments of ``client.messages.create`` for this call."""
    forced = forces_tool(model) if forced is None else forced
    system = [m["content"] for m in messages if m["role"] == "system"]
    tool: dict[str, Any] = {
        "name": TOOL_NAME,
        "description": "Return the answer as this tool's input.",
        "input_schema": schema,
    }
    if forced:
        tool_choice: dict[str, str] = {"type": "tool", "name": TOOL_NAME}
    else:
        tool["strict"] = True
        tool_choice = {"type": "auto"}
        system.append(_CALL_INSTRUCTION)
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ],
        "tools": [tool],
        "tool_choice": tool_choice,
    }
    if system:
        request["system"] = "\n\n".join(system)
    return request


def parse_response(response: Any) -> tuple[dict[str, Any], int, int]:
    """Pick the tool input and the token counts out of a Messages response."""
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        raise LLMError("the model refused the request (stop_reason refusal)")
    if stop == "max_tokens":
        raise LLMError("the answer was cut off at max_tokens; it is not complete JSON")
    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            if not isinstance(block.input, dict):
                break
            return dict(block.input), response.usage.input_tokens, response.usage.output_tokens
    raise LLMError(f"the model did not call the {TOOL_NAME} tool (stop_reason {stop})")


class AnthropicLLM:
    """The Anthropic provider."""

    provider = "anthropic"
    supports_seed = False

    def __init__(self, client: Any | None = None, seed: int | None = None) -> None:
        """Create the provider; the SDK client is built on first use unless one is given.

        ``seed`` is kept so a run can say what it was asked for, but the Messages API has no seed
        parameter, so it is never sent.
        """
        self.seed = seed
        self._client = client
        self.usage = Usage()

    def _api(self) -> Any:
        if self._client is None:
            client = anthropic.Anthropic()
            if client.api_key is None and client.auth_token is None:
                raise LLMError(
                    "no Anthropic credentials: set ANTHROPIC_API_KEY (or use --provider openai "
                    "for a local model)"
                )
            self._client = client
        return self._client

    def complete_json(
        self, messages: list[Message], schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        """Call the model with a single tool whose input schema is ``schema``."""
        forced = forces_tool(model)
        for _attempt in range(2):
            request = build_request(messages, schema, model=model, forced=forced)
            try:
                response = self._api().messages.create(**request)
            except anthropic.BadRequestError as error:
                if forced and "tool_choice" in str(error):
                    forced = False
                    continue
                raise LLMError(f"Anthropic rejected the request: {error}") from error
            except anthropic.APIError as error:
                raise LLMError(f"Anthropic call failed: {error}") from error
            data, tokens_in, tokens_out = parse_response(response)
            self.usage.add(tokens_in, tokens_out)
            return data
        raise LLMError("Anthropic rejected the request")
