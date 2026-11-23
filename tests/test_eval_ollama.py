"""What a local Ollama reports about itself and its model, read for the record of a run."""

from typing import Any

import httpx
import pytest

from tailor.evals.pipeline import ollama_model_digest, ollama_version

DIGEST = "500a1f067a9f" + "0" * 52


class Reply:
    def __init__(self, body: Any) -> None:
        self.body = body

    def json(self) -> Any:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, Any]) -> list[str]:
    """Answer httpx.get from ``routes`` (path to JSON body); a missing path is a refused call."""
    asked: list[str] = []

    def get(url: str, **_: Any) -> Reply:
        asked.append(url)
        for path, body in routes.items():
            if url.endswith(path):
                return Reply(body)
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", get)
    return asked


def test_the_version_is_read_from_the_server_root_not_the_openai_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = serve(monkeypatch, {"/api/version": {"version": "0.23.4"}})
    assert ollama_version("http://localhost:11434/v1") == "0.23.4"
    assert asked == ["http://localhost:11434/api/version"]


def test_no_version_when_there_is_no_base_url_or_the_server_is_down_or_not_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(monkeypatch, {})
    assert ollama_version(None) is None
    assert ollama_version("http://localhost:11434/v1") is None
    serve(monkeypatch, {"/api/version": ValueError("not JSON")})
    assert ollama_version("http://localhost:11434/v1") is None
    serve(monkeypatch, {"/api/version": ["not", "an", "object"]})
    assert ollama_version("http://localhost:11434/v1") is None


def test_the_digest_of_the_model_is_read_from_the_local_model_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = {
        "models": [
            {"name": "qwen3:4b", "model": "qwen3:4b", "digest": "f" * 64},
            {"name": "qwen3:8b", "model": "qwen3:8b", "digest": DIGEST},
        ]
    }
    asked = serve(monkeypatch, {"/api/tags": models})
    assert ollama_model_digest("http://localhost:11434/v1", "qwen3:8b") == DIGEST
    assert asked == ["http://localhost:11434/api/tags"]
    assert ollama_model_digest("http://localhost:11434/v1", "llama3:8b") is None


def test_no_digest_when_there_is_no_base_url_or_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(monkeypatch, {})
    assert ollama_model_digest(None, "qwen3:8b") is None
    assert ollama_model_digest("http://localhost:11434/v1", "qwen3:8b") is None
    serve(monkeypatch, {"/api/tags": {"models": "not a list"}})
    assert ollama_model_digest("http://localhost:11434/v1", "qwen3:8b") is None
