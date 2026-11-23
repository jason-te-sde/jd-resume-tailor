"""Where the model comes from: ``tailor.toml``, environment variables, then command-line flags.

``tailor.toml`` (in the working directory, or the file named by ``--config``)::

    [llm]
    provider = "openai"                     # or "anthropic"
    model = "qwen3:8b"
    base_url = "http://localhost:11434/v1"  # openai provider only

    [cassettes]
    mode = "off"                            # off | record | replay
    dir = "cassettes"

Environment variables override the file: ``TAILOR_PROVIDER``, ``TAILOR_MODEL``,
``TAILOR_BASE_URL``, ``TAILOR_SEED``, ``TAILOR_CASSETTES`` and ``TAILOR_CASSETTE_DIR``. There is
no default model, and no seed unless one is given.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tailor.llm.base import LLM, SEED_PROVIDERS, ConfigError
from tailor.llm.cassette import CassetteLLM

PROVIDERS: Final = ("anthropic", "openai")
MODES: Final = ("off", "record", "replay")
DEFAULT_CASSETTE_DIR: Final = "cassettes"

_HOW_TO_SET: Final = (
    "no model is configured. Set one with --model (and --provider), the TAILOR_MODEL and "
    "TAILOR_PROVIDER environment variables, or a tailor.toml:\n\n"
    '    [llm]\n    provider = "anthropic"   # or "openai" for any OpenAI-compatible endpoint\n'
    '    model = "<a model id from your provider>"\n\n'
    'For a local Ollama use provider = "openai", base_url = "http://localhost:11434/v1" '
    "and the name of a model from `ollama list`."
)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """A resolved model configuration."""

    provider: str
    model: str
    base_url: str | None = None
    cassette_mode: str = "off"
    cassette_dir: Path = Path(DEFAULT_CASSETTE_DIR)
    seed: int | None = None


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"tailor.toml: [{name}] must be a table")
    return value


def load_config(
    path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> LLMConfig:
    """Resolve the configuration: flags over environment over file.

    Raises:
        ConfigError: No model is configured, or a value is not valid.
    """
    env = os.environ if env is None else env
    data: Mapping[str, Any] = {}
    file = path if path is not None else Path("tailor.toml")
    if path is not None or file.is_file():
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except OSError as error:
            raise ConfigError(f"cannot read {file}: {error.strerror}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{file} is not valid TOML: {error}") from error
    llm = _table(data, "llm")
    cassettes = _table(data, "cassettes")

    chosen_model = model or env.get("TAILOR_MODEL") or llm.get("model")
    if not chosen_model:
        raise ConfigError(_HOW_TO_SET)
    chosen_provider = provider or env.get("TAILOR_PROVIDER") or llm.get("provider")
    if chosen_provider not in PROVIDERS:
        raise ConfigError(
            f"provider must be one of {', '.join(PROVIDERS)}, not {chosen_provider!r}; "
            "set it with --provider, TAILOR_PROVIDER or [llm] provider in tailor.toml"
        )
    mode = env.get("TAILOR_CASSETTES") or cassettes.get("mode") or "off"
    if mode not in MODES:
        raise ConfigError(f"TAILOR_CASSETTES must be one of {', '.join(MODES)}, not {mode!r}")
    directory = env.get("TAILOR_CASSETTE_DIR") or cassettes.get("dir") or DEFAULT_CASSETTE_DIR
    if seed is None and env.get("TAILOR_SEED"):
        try:
            seed = int(env["TAILOR_SEED"])
        except ValueError:
            raise ConfigError(
                f"TAILOR_SEED must be an integer, not {env['TAILOR_SEED']!r}"
            ) from None
    base_url = env.get("TAILOR_BASE_URL") or llm.get("base_url") or None
    return LLMConfig(
        provider=str(chosen_provider),
        model=str(chosen_model),
        base_url=str(base_url) if base_url else None,
        cassette_mode=str(mode),
        cassette_dir=Path(str(directory)),
        seed=seed,
    )


def _repository_of(path: Path) -> Path | None:
    """The nearest ancestor of ``path`` (itself included) that holds a ``.git``, if any."""
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def check_recording(directory: Path, bank: Path | None) -> None:
    """Refuse to record cassettes into a repository for a bank that is not an example.

    A cassette holds the whole request, and so the bank's facts. One written inside a repository
    is one ``git add`` from being published, so recording there is allowed only for a bank under
    that repository's ``examples/`` directory (the invented bank the repository ships).

    Raises:
        ConfigError: ``directory`` is inside a repository and ``bank`` is not under its
            ``examples/``, or is not known.
    """
    repository = _repository_of(directory)
    if repository is None:
        return
    examples = (repository / "examples").resolve()
    if bank is not None and bank.resolve().is_relative_to(examples):
        return
    raise ConfigError(
        f"refusing to record cassettes into {directory}, inside the repository {repository}: "
        "a cassette contains the bank's text, and this bank is not under the repository's "
        "examples/ directory. Record into a directory outside the repository "
        "(TAILOR_CASSETTE_DIR)."
    )


def seed_warning(config: LLMConfig) -> str | None:
    """What to tell the user when a seed was asked of a provider that takes none, else ``None``."""
    if config.seed is None or config.provider in SEED_PROVIDERS:
        return None
    return (
        f"warning: the {config.provider} provider takes no seed, so seed {config.seed} "
        "(--seed or TAILOR_SEED) is not sent and the run is not seeded"
    )


def build_llm(config: LLMConfig, *, bank: Path | None = None) -> LLM:
    """Build the provider for ``config``, wrapped for recording or replay when asked.

    Args:
        config: The resolved configuration.
        bank: The bank the run reads; recording inside a repository is allowed only for a bank
            under its ``examples/`` (see :func:`check_recording`).

    Raises:
        ConfigError: Recording was asked for where a real bank must not be recorded.
    """
    if config.cassette_mode == "record":
        check_recording(config.cassette_dir, bank)
    if config.cassette_mode == "replay":
        return CassetteLLM(
            "replay", config.cassette_dir, provider=config.provider, inner=None, seed=config.seed
        )
    inner: LLM
    if config.provider == "anthropic":
        from tailor.llm.anthropic_provider import AnthropicLLM

        inner = AnthropicLLM(seed=config.seed)
    else:
        from tailor.llm.openai_provider import OpenAICompatibleLLM

        inner = OpenAICompatibleLLM(base_url=config.base_url, seed=config.seed)
    if config.cassette_mode == "record":
        return CassetteLLM(
            "record", config.cassette_dir, provider=config.provider, inner=inner, seed=config.seed
        )
    return inner
