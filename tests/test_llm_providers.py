"""Request shapes and answer parsing of the two providers, against stored examples."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from tailor.llm import LLMError, Message
from tailor.llm.anthropic_provider import AnthropicLLM, forces_tool
from tailor.llm.anthropic_provider import build_request as anthropic_request
from tailor.llm.openai_provider import OpenAICompatibleLLM
from tailor.llm.openai_provider import build_request as openai_request

DATA = Path(__file__).parent / "data" / "llm"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}
MESSAGES: list[Message] = [
    {"role": "system", "content": "You extract things."},
    {"role": "user", "content": "Posting text"},
]


def stored(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_a_model_that_allows_it_gets_a_forced_single_tool() -> None:
    request = anthropic_request(MESSAGES, SCHEMA, model="claude-haiku-5-5")
    assert request == stored("anthropic_forced.json")


def test_models_that_reject_forced_tool_choice_get_auto_with_a_strict_tool() -> None:
    request = anthropic_request(MESSAGES, SCHEMA, model="claude-opus-5-5")
    assert request == stored("anthropic_auto.json")
    assert not forces_tool("claude-sonnet-5-5")
    assert forces_tool("claude-haiku-5-5")


def test_no_sampling_parameter_is_sent_to_either_provider() -> None:
    for request in (
        anthropic_request(MESSAGES, SCHEMA, model="claude-opus-5-5"),
        openai_request(MESSAGES, SCHEMA, model="qwen3:8b"),
    ):
        assert not {"temperature", "top_p", "top_k", "max_completion_tokens"} & set(request)


def test_the_openai_request_is_a_strict_json_schema_response_format() -> None:
    assert openai_request(MESSAGES, SCHEMA, model="qwen3:8b") == stored("openai_schema.json")


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        outcome = self.outcomes[len(self.requests) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def tool_response(data: Any, stop: str = "tool_use") -> Any:
    block = SimpleNamespace(type="tool_use", name="respond", input=data)
    usage = SimpleNamespace(input_tokens=12, output_tokens=7)
    return SimpleNamespace(content=[block], stop_reason=stop, usage=usage)


def bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx2.Request("POST", "https://api.example.com/v1/messages")
    response = httpx2.Response(400, request=request)
    return anthropic.BadRequestError(message, response=response, body=None)


def anthropic_llm(*outcomes: Any) -> tuple[AnthropicLLM, FakeMessages]:
    fake = FakeMessages(list(outcomes))
    return AnthropicLLM(client=SimpleNamespace(messages=fake)), fake


def test_the_tool_input_is_the_answer_and_usage_is_counted() -> None:
    llm, fake = anthropic_llm(tool_response({"title": "x"}))
    assert llm.complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5") == {"title": "x"}
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 12, 7)
    assert fake.requests[0]["tool_choice"] == {"type": "tool", "name": "respond"}


def test_a_forced_choice_the_model_rejects_is_retried_once_with_auto() -> None:
    llm, fake = anthropic_llm(
        bad_request("tool_choice: type 'tool' is not supported for this model"),
        tool_response({"title": "x"}),
    )
    assert llm.complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5") == {"title": "x"}
    assert [r["tool_choice"]["type"] for r in fake.requests] == ["tool", "auto"]


def test_another_bad_request_is_an_error_not_a_retry() -> None:
    llm, fake = anthropic_llm(bad_request("max_tokens too large"))
    with pytest.raises(LLMError, match="max_tokens too large"):
        llm.complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5")
    assert len(fake.requests) == 1


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_a_refusal_or_a_cut_off_answer_is_an_error(stop: str) -> None:
    llm, _ = anthropic_llm(tool_response({"title": "x"}, stop=stop))
    with pytest.raises(LLMError, match=stop):
        llm.complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5")


def test_an_answer_without_the_tool_call_is_an_error() -> None:
    text = SimpleNamespace(type="text", text="hello")
    response = SimpleNamespace(
        content=[text],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    llm, _ = anthropic_llm(response)
    with pytest.raises(LLMError, match="did not call"):
        llm.complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5")


class FakeCompletions:
    def __init__(
        self, content: str | None, finish: str = "stop", refusal: str | None = None
    ) -> None:
        message = SimpleNamespace(content=content, refusal=refusal)
        choice = SimpleNamespace(message=message, finish_reason=finish)
        usage = SimpleNamespace(prompt_tokens=20, completion_tokens=9)
        self.response = SimpleNamespace(choices=[choice], usage=usage)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return self.response


def openai_llm(fake: FakeCompletions) -> OpenAICompatibleLLM:
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return OpenAICompatibleLLM(base_url="http://localhost:11434/v1", client=client)


def test_the_message_content_is_parsed_and_usage_counted() -> None:
    llm = openai_llm(FakeCompletions('{"title": "x"}'))
    assert llm.complete_json(MESSAGES, SCHEMA, model="m") == {"title": "x"}
    assert (llm.usage.input_tokens, llm.usage.output_tokens) == (20, 9)


def test_a_think_block_before_the_json_is_ignored() -> None:
    llm = openai_llm(FakeCompletions('<think>hmm\n{"a": 1}</think>\n{"title": "x"}'))
    assert llm.complete_json(MESSAGES, SCHEMA, model="m") == {"title": "x"}


@pytest.mark.parametrize(
    ("fake", "match"),
    [
        (FakeCompletions("not json"), "not valid JSON"),
        (FakeCompletions("[1]"), "not an object"),
        (FakeCompletions('{"a":', finish="length"), "cut off"),
        (FakeCompletions(None, refusal="no"), "refused"),
    ],
)
def test_unusable_answers_are_errors(fake: FakeCompletions, match: str) -> None:
    with pytest.raises(LLMError, match=match):
        openai_llm(fake).complete_json(MESSAGES, SCHEMA, model="m")


def test_without_an_api_key_the_error_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM().complete_json(MESSAGES, SCHEMA, model="claude-haiku-5-5")
