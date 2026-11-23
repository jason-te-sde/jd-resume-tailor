"""Record and replay: the same request replays, a different one names its hash."""

import json
from pathlib import Path
from typing import Any

import pytest
from llmfakes import ScriptedLLM

from tailor.llm import CassetteError, CassetteLLM, Message, normalize_request, request_hash

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "string"}}}
MESSAGES: list[Message] = [{"role": "user", "content": "hello"}]


def test_a_recorded_answer_replays_without_the_provider(tmp_path: Path) -> None:
    inner = ScriptedLLM([{"a": "one"}])
    recorder = CassetteLLM("record", tmp_path, provider="openai", inner=inner)
    assert recorder.complete_json(MESSAGES, SCHEMA, model="m") == {"a": "one"}
    assert len(list(tmp_path.glob("*.json"))) == 1

    player = CassetteLLM("replay", tmp_path, provider="openai", inner=None)
    assert player.complete_json(MESSAGES, SCHEMA, model="m") == {"a": "one"}
    assert (player.usage.calls, player.usage.replayed_calls) == (1, 1)
    assert (player.usage.input_tokens, player.usage.output_tokens) == (10, 5)
    assert len(inner.calls) == 1


def test_the_cassette_is_named_by_the_hash_of_the_normalized_request(tmp_path: Path) -> None:
    CassetteLLM(
        "record", tmp_path, provider="openai", inner=ScriptedLLM([{"a": "x"}])
    ).complete_json(MESSAGES, SCHEMA, model="m")
    digest = request_hash(normalize_request("openai", "m", MESSAGES, SCHEMA))
    saved = json.loads((tmp_path / f"{digest}.json").read_text(encoding="utf-8"))
    assert saved["response"] == {"a": "x"}
    assert saved["request"]["messages"] == MESSAGES


def test_replay_with_a_missing_cassette_names_the_hash(tmp_path: Path) -> None:
    player = CassetteLLM("replay", tmp_path, provider="openai", inner=None)
    digest = request_hash(normalize_request("openai", "m", MESSAGES, SCHEMA))
    with pytest.raises(CassetteError, match=digest):
        player.complete_json(MESSAGES, SCHEMA, model="m")


@pytest.mark.parametrize(
    ("provider", "model", "text"),
    [("anthropic", "m", "hello"), ("openai", "other", "hello"), ("openai", "m", "hello!")],
)
def test_any_change_to_provider_model_or_text_is_a_different_request(
    provider: str, model: str, text: str
) -> None:
    base = request_hash(normalize_request("openai", "m", MESSAGES, SCHEMA))
    changed = request_hash(
        normalize_request(provider, model, [{"role": "user", "content": text}], SCHEMA)
    )
    assert base != changed


def test_line_endings_and_edge_whitespace_do_not_change_the_hash() -> None:
    a = normalize_request("openai", "m", [{"role": "user", "content": "a\nb"}], SCHEMA)
    b = normalize_request("openai", "m", [{"role": "user", "content": "a\r\nb\n"}], SCHEMA)
    assert request_hash(a) == request_hash(b)


def test_recording_again_overwrites_and_a_broken_cassette_is_an_error(tmp_path: Path) -> None:
    inner = ScriptedLLM([{"a": "1"}, {"a": "2"}])
    recorder = CassetteLLM("record", tmp_path, provider="openai", inner=inner)
    recorder.complete_json(MESSAGES, SCHEMA, model="m")
    recorder.complete_json(MESSAGES, SCHEMA, model="m")
    player = CassetteLLM("replay", tmp_path, provider="openai", inner=None)
    assert player.complete_json(MESSAGES, SCHEMA, model="m") == {"a": "2"}
    (path,) = tmp_path.glob("*.json")
    path.write_text("{", encoding="utf-8")
    with pytest.raises(CassetteError, match="unreadable"):
        player.complete_json(MESSAGES, SCHEMA, model="m")
