"""The seed: sent where the provider supports one, hashed into the cassette, never a default."""

from pathlib import Path
from typing import Any

import pytest
from llmfakes import ScriptedLLM
from test_llm_providers import FakeCompletions, openai_llm

from tailor.llm import CassetteLLM, LLMConfig, Message, build_llm, load_config, normalize_request
from tailor.llm import request_hash as hash_of
from tailor.llm.anthropic_provider import AnthropicLLM
from tailor.llm.openai_provider import OpenAICompatibleLLM
from tailor.llm.openai_provider import build_request as openai_request

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "string"}}}
MESSAGES: list[Message] = [{"role": "user", "content": "hello"}]


def test_the_openai_request_carries_the_seed_only_when_there_is_one() -> None:
    assert "seed" not in openai_request(MESSAGES, SCHEMA, model="m")
    assert openai_request(MESSAGES, SCHEMA, model="m", seed=7)["seed"] == 7
    assert openai_request(MESSAGES, SCHEMA, model="m", seed=0)["seed"] == 0


def test_the_openai_provider_sends_its_seed_with_every_call() -> None:
    fake = FakeCompletions('{"a": "x"}')
    llm = openai_llm(fake)
    llm.seed = 11
    llm.complete_json(MESSAGES, SCHEMA, model="m")
    assert fake.requests[0]["seed"] == 11


def test_which_providers_apply_a_seed() -> None:
    assert OpenAICompatibleLLM.supports_seed is True
    assert AnthropicLLM.supports_seed is False


def test_the_seed_changes_the_cassette_hash_but_none_keeps_it() -> None:
    plain = hash_of(normalize_request("openai", "m", MESSAGES, SCHEMA))
    assert plain == hash_of(normalize_request("openai", "m", MESSAGES, SCHEMA, seed=None))
    one = hash_of(normalize_request("openai", "m", MESSAGES, SCHEMA, seed=1))
    two = hash_of(normalize_request("openai", "m", MESSAGES, SCHEMA, seed=2))
    assert len({plain, one, two}) == 3


def test_two_seeds_record_and_replay_two_answers_for_one_prompt(tmp_path: Path) -> None:
    for seed, answer in ((1, "one"), (2, "two")):
        CassetteLLM(
            "record", tmp_path, provider="openai", inner=ScriptedLLM([{"a": answer}]), seed=seed
        ).complete_json(MESSAGES, SCHEMA, model="m")
    assert len(list(tmp_path.glob("*.json"))) == 2
    for seed, answer in ((1, "one"), (2, "two")):
        player = CassetteLLM("replay", tmp_path, provider="openai", inner=None, seed=seed)
        assert player.complete_json(MESSAGES, SCHEMA, model="m") == {"a": answer}


def test_the_config_reads_the_seed_from_the_environment_and_flags() -> None:
    env = {"TAILOR_MODEL": "m", "TAILOR_PROVIDER": "openai"}
    assert load_config(None, env=env).seed is None
    assert load_config(None, env={**env, "TAILOR_SEED": "5"}).seed == 5
    assert load_config(None, env={**env, "TAILOR_SEED": "5"}, seed=9).seed == 9


def test_a_seed_that_is_not_an_integer_is_a_config_error() -> None:
    from tailor.llm import ConfigError

    with pytest.raises(ConfigError, match="TAILOR_SEED"):
        load_config(
            None, env={"TAILOR_MODEL": "m", "TAILOR_PROVIDER": "openai", "TAILOR_SEED": "x"}
        )


def test_build_llm_hands_the_seed_to_the_provider_and_the_cassette(tmp_path: Path) -> None:
    config = LLMConfig("openai", "m", seed=3)
    llm = build_llm(config)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.seed == 3
    recorder = build_llm(
        LLMConfig("openai", "m", seed=3, cassette_mode="record", cassette_dir=tmp_path)
    )
    assert isinstance(recorder, CassetteLLM)
    assert recorder.seed == 3
    assert recorder.supports_seed is True
    replay = build_llm(
        LLMConfig("anthropic", "m", seed=3, cassette_mode="replay", cassette_dir=tmp_path)
    )
    assert isinstance(replay, CassetteLLM)
    assert replay.seed == 3
