"""Record and replay model calls.

A cassette is one JSON file named by the hash of the normalized request: provider kind, model,
messages and schema. The endpoint and the key are not part of it, so a run recorded against a
local server replays anywhere. ``record`` always calls the model and writes the cassette; ``replay``
never calls it and fails, naming the hash, when the cassette is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tailor.llm.base import LLM, SEED_PROVIDERS, LLMError, Message, Usage


class CassetteError(LLMError):
    """A replay found no cassette for a request, or the cassette is unreadable."""


def normalize_request(
    provider: str,
    model: str,
    messages: list[Message],
    schema: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    """The request as it is hashed: line endings unified, trailing whitespace dropped.

    The seed is part of it only when there is one, so a request made without a seed keeps the hash
    it always had.
    """
    request: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "messages": [
            {"role": m["role"], "content": m["content"].replace("\r\n", "\n").strip()}
            for m in messages
        ],
        "schema": schema,
    }
    if seed is not None:
        request["seed"] = seed
    return request


def request_hash(request: dict[str, Any]) -> str:
    """The SHA-256 of the request's canonical JSON."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CassetteLLM:
    """Wraps a provider (``record``) or stands in for it (``replay``)."""

    def __init__(
        self,
        mode: str,
        directory: Path,
        *,
        provider: str,
        inner: LLM | None,
        seed: int | None = None,
    ) -> None:
        """Wrap ``inner`` (record) or stand in for it (replay).

        ``seed`` identifies the sample: it is hashed into every cassette, and in record mode it is
        the seed the wrapped provider sends.
        """
        if mode not in ("record", "replay"):
            raise ValueError(f"cassette mode must be record or replay, not {mode!r}")
        if mode == "record" and inner is None:
            raise ValueError("recording needs a provider to call")
        self.mode = mode
        self.directory = directory
        self.provider = provider
        self.seed = seed
        self.supports_seed = provider in SEED_PROVIDERS
        self._inner = inner
        self.usage = Usage()

    def complete_json(
        self, messages: list[Message], schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        """Replay the recorded answer, or call the provider and record it."""
        request = normalize_request(self.provider, model, messages, schema, self.seed)
        digest = request_hash(request)
        path = self.directory / f"{digest}.json"
        if self.mode == "replay":
            if not path.is_file():
                raise CassetteError(
                    f"no cassette for request {digest} in {self.directory}; "
                    "record it with TAILOR_CASSETTES=record"
                )
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                response = saved["response"]
                usage = saved["usage"]
                self.usage.add(usage["input_tokens"], usage["output_tokens"], replayed=True)
            except (OSError, ValueError, KeyError) as error:
                raise CassetteError(f"cassette {path} is unreadable: {error}") from error
            if not isinstance(response, dict):
                raise CassetteError(f"cassette {path}: the response is not an object")
            return response
        assert self._inner is not None
        before = (self._inner.usage.input_tokens, self._inner.usage.output_tokens)
        response = self._inner.complete_json(messages, schema, model=model)
        tokens_in = self._inner.usage.input_tokens - before[0]
        tokens_out = self._inner.usage.output_tokens - before[1]
        self.usage.add(tokens_in, tokens_out)
        self.directory.mkdir(parents=True, exist_ok=True)
        document = {
            "request": request,
            "response": response,
            "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        }
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return response
