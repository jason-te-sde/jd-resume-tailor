"""The technology vocabulary: a curated list of names a bullet might claim, shipped as data.

The list lives in ``technologies.txt`` next to this module so that anyone can read it, correct it
and extend it without touching code; its format is described at the top of that file. This module
parses it and finds the names in a text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Final

_BEFORE: Final = r"(?<![A-Za-z0-9_])"
_AFTER: Final = r"(?![A-Za-z0-9_]|\.[A-Za-z0-9])"


@dataclass(frozen=True, slots=True)
class Mention:
    """One technology named in a text.

    Attributes:
        canonical: The canonical name of the technology (``PostgreSQL`` for ``Postgres``).
        start: Index of the first character of the mention.
        end: Index after the last character.
        case_sensitive: Whether the vocabulary matches this name only in its exact case (``Go``),
            which makes a mention of it weaker evidence than a plain word match.
    """

    canonical: str
    start: int
    end: int
    case_sensitive: bool = False


class Vocabulary:
    """A set of technologies, each with a canonical name, aliases and a case rule."""

    def __init__(self, entries: list[tuple[str, tuple[str, ...], bool]]) -> None:
        """Build a vocabulary from ``(canonical, aliases, case_sensitive)`` entries."""
        self._insensitive: dict[str, str] = {}
        self._sensitive: dict[str, str] = {}
        for canonical, aliases, case_sensitive in entries:
            for name in (canonical, *aliases):
                if case_sensitive:
                    self._sensitive[name] = canonical
                else:
                    self._insensitive[name.lower()] = canonical
        self.canonical_names: frozenset[str] = frozenset(entry[0] for entry in entries)
        self._insensitive_pattern = self._pattern(self._insensitive, re.IGNORECASE)
        self._sensitive_pattern = self._pattern(self._sensitive, 0)

    @staticmethod
    def _pattern(names: dict[str, str], flags: int) -> re.Pattern[str] | None:
        if not names:
            return None
        alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        return re.compile(f"{_BEFORE}(?:{alternatives}){_AFTER}", flags)

    def canonical(self, name: str) -> str | None:
        """The canonical name of ``name`` if it is exactly a name in the vocabulary."""
        return self._sensitive.get(name) or self._insensitive.get(name.lower())

    def find(self, text: str) -> list[Mention]:
        """Return the technologies named in ``text``, in order; the longest name wins an overlap."""
        found: list[Mention] = []
        for pattern, names, fold in (
            (self._insensitive_pattern, self._insensitive, True),
            (self._sensitive_pattern, self._sensitive, False),
        ):
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                key = match.group().lower() if fold else match.group()
                found.append(
                    Mention(names[key], match.start(), match.end(), case_sensitive=not fold)
                )
        found.sort(key=lambda mention: (mention.start, -(mention.end - mention.start)))
        chosen: list[Mention] = []
        for mention in found:
            if not chosen or mention.start >= chosen[-1].end:
                chosen.append(mention)
        return chosen


def parse_vocabulary(source: str) -> Vocabulary:
    """Parse the text of a vocabulary file (see ``technologies.txt`` for the format)."""
    entries: list[tuple[str, tuple[str, ...], bool]] = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        case_sensitive = line.startswith("=")
        canonical, *aliases = [part.strip() for part in line.lstrip("=").split("|")]
        entries.append((canonical, tuple(alias for alias in aliases if alias), case_sensitive))
    return Vocabulary(entries)


@lru_cache(maxsize=1)
def load_vocabulary() -> Vocabulary:
    """Load the vocabulary that ships with the package."""
    return parse_vocabulary(files("tailor.verify").joinpath("technologies.txt").read_text("utf-8"))
