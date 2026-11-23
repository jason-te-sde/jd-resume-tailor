"""Where the model configuration comes from, and the error when there is none."""

from pathlib import Path

import pytest

from tailor.llm import CassetteLLM, ConfigError, LLMConfig, build_llm, load_config
from tailor.llm.config import seed_warning
from tailor.llm.openai_provider import OpenAICompatibleLLM


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tailor.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_with_no_model_anywhere_the_error_says_how_to_set_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no tailor.toml here
    with pytest.raises(ConfigError, match=r"TAILOR_MODEL[\s\S]*tailor\.toml"):
        load_config(None, env={})


def test_a_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.toml", env={})


def test_the_file_supplies_everything(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[llm]\nprovider = "openai"\nmodel = "qwen3:8b"\nbase_url = "http://localhost:11434/v1"\n'
        '[cassettes]\nmode = "replay"\ndir = "tapes"\n',
    )
    config = load_config(path, env={})
    assert config == LLMConfig(
        "openai", "qwen3:8b", "http://localhost:11434/v1", "replay", Path("tapes")
    )


def test_environment_beats_the_file_and_flags_beat_the_environment(tmp_path: Path) -> None:
    path = write(tmp_path, '[llm]\nprovider = "openai"\nmodel = "from-file"\n')
    env = {"TAILOR_MODEL": "from-env", "TAILOR_PROVIDER": "anthropic", "TAILOR_CASSETTES": "record"}
    config = load_config(path, env=env)
    assert (config.model, config.provider, config.cassette_mode) == (
        "from-env",
        "anthropic",
        "record",
    )
    flagged = load_config(path, env=env, model="from-flag", provider="openai")
    assert (flagged.model, flagged.provider) == ("from-flag", "openai")


def test_an_unknown_provider_or_mode_is_an_error() -> None:
    with pytest.raises(ConfigError, match="provider must be"):
        load_config(None, env={"TAILOR_MODEL": "m", "TAILOR_PROVIDER": "acme"})
    with pytest.raises(ConfigError, match="TAILOR_CASSETTES"):
        load_config(
            None, env={"TAILOR_MODEL": "m", "TAILOR_PROVIDER": "openai", "TAILOR_CASSETTES": "x"}
        )


def test_replay_builds_no_provider_and_record_wraps_one(tmp_path: Path) -> None:
    replay = build_llm(LLMConfig("anthropic", "m", cassette_mode="replay", cassette_dir=tmp_path))
    assert isinstance(replay, CassetteLLM)
    record = build_llm(LLMConfig("openai", "m", cassette_mode="record", cassette_dir=tmp_path))
    assert isinstance(record, CassetteLLM)
    assert isinstance(build_llm(LLMConfig("openai", "m")), OpenAICompatibleLLM)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "examples" / "demo").mkdir(parents=True)
    (repo / "private").mkdir()
    return repo


def recording(directory: Path) -> LLMConfig:
    return LLMConfig("openai", "m", cassette_mode="record", cassette_dir=directory)


def test_recording_inside_the_repository_is_refused_for_a_bank_outside_examples(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    with pytest.raises(ConfigError, match="examples"):
        build_llm(recording(repo / "cassettes"), bank=repo / "private" / "bank.json")
    with pytest.raises(ConfigError, match="examples"):
        build_llm(recording(repo / "cassettes"))


def test_recording_inside_the_repository_is_allowed_for_a_bank_under_examples(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    llm = build_llm(
        recording(repo / "examples" / "demo" / "cassettes"),
        bank=repo / "examples" / "demo" / "bank.json",
    )
    assert isinstance(llm, CassetteLLM)


def test_recording_outside_any_repository_is_allowed_for_any_bank(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    llm = build_llm(recording(scratch / "tapes"), bank=repo / "private" / "bank.json")
    assert isinstance(llm, CassetteLLM)


def test_a_bank_that_only_looks_like_it_is_under_examples_is_refused(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sneaky = repo / "private" / ".." / ".." / "elsewhere" / "examples" / "bank.json"
    with pytest.raises(ConfigError, match="examples"):
        build_llm(recording(repo / "cassettes"), bank=sneaky)


def test_replaying_inside_the_repository_is_not_restricted(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = LLMConfig("openai", "m", cassette_mode="replay", cassette_dir=repo / "cassettes")
    assert isinstance(build_llm(config, bank=repo / "private" / "bank.json"), CassetteLLM)


def test_a_seed_given_to_a_provider_without_one_has_a_warning() -> None:
    warning = seed_warning(LLMConfig("anthropic", "m", seed=3))
    assert warning is not None
    assert "anthropic" in warning
    assert "seed 3" in warning


def test_no_warning_when_the_provider_takes_the_seed_or_none_is_given() -> None:
    assert seed_warning(LLMConfig("openai", "m", seed=3)) is None
    assert seed_warning(LLMConfig("anthropic", "m")) is None
