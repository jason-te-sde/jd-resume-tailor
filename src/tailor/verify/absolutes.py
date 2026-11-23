"""Absolute words: claims with no exceptions, which a source rarely makes and a rewrite often adds.

``guaranteed``, ``never``, ``always``, ``exactly-once``, ``100%``, ``completely``, ``every``: each
says "without exception", and a measurement of N runs cannot say that. The check flags an absolute
in a bullet that none of the cited sources contains, with three refinements so that faithful
rewrites pass:

* ``zero`` is a number as much as an absolute: ``zero violations`` is fine when a source says
  ``0 violations`` or ``no violations``, and ``zero-copy`` and ``zero-shot`` are technical terms;
* ``all`` and ``every`` are fine when a source quantifies the same thing (``every number in the
  README`` supports ``every number``), fine before a count (``all 5 views``) and fine as a
  frequency (``every 5 minutes``, ``every night``);
* every other absolute is fine when a source uses the same word.

This is a word-list heuristic. It does not know ``bulletproof`` from ``robust`` unless the list
does, and the list is short on purpose: each word added is another way for a faithful bullet to be
flagged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tailor.verify.text import content_tokens

AbsoluteKind = Literal["plain", "zero", "quantifier", "claim"]


@dataclass(frozen=True, slots=True)
class AbsoluteMention:
    """An absolute word found in a text: its span, the text, and its kind."""

    start: int
    end: int
    text: str
    kind: AbsoluteKind


_PLAIN: Final = re.compile(
    r"\bguarantee(?:d|s|ing)?\b|\bnever\b|\balways\b|\bexactly[- ]once\b"
    r"|(?<![\w.])100(?:\.0+)?\s?(?:%|percent)|\bhundred percent\b"
    r"|\bcompletely\b|\bfully\b|\bentire(?:ly)?\b|\bperfect(?:ly)?\b|\bflawless(?:ly)?\b"
    r"|\bbulletproof\b|\beliminat(?:e|es|ed|ing)\b|\bunlimited\b|\bwithout exception\b"
    r"|\bimpossible\b",
    re.IGNORECASE,
)
_ZERO: Final = re.compile(
    r"\bzero\b(?![- ](?:copy|shot|day|trust|knowledge|config)\b)", re.IGNORECASE
)
_QUANTIFIER: Final = re.compile(r"\b(?:all|every)\b(?![- ]in[- ]one)", re.IGNORECASE)
_ANY_QUANTIFIER: Final = re.compile(r"\b(?:all|every|each|entire)\b", re.IGNORECASE)
_NOT_A_UNIVERSAL: Final = re.compile(
    r"\s+(?:\d|one\b|two\b|three\b|four\b|five\b|six\b|seven\b|eight\b|nine\b|ten\b|other\b|few\b"
    r"|night\b|morning\b|day\b|week\b|hour\b|minute\b|month\b|year\b|weekend\b|sprint\b"
    r"|quarter\b)",
    re.IGNORECASE,
)
_NONE_WORDS: Final = frozenset({"no", "zero", "0", "without", "none", "never"})
_NOUN_REACH: Final = 2
_SOURCE_REACH: Final = 3


def find_absolutes(text: str) -> list[AbsoluteMention]:
    """Return the absolute words of ``text``, in order, except those that are plainly not one."""
    found: list[AbsoluteMention] = []
    for match in _PLAIN.finditer(text):
        found.append(AbsoluteMention(match.start(), match.end(), match.group(), "plain"))
    for match in _ZERO.finditer(text):
        found.append(AbsoluteMention(match.start(), match.end(), match.group(), "zero"))
    for match in _QUANTIFIER.finditer(text):
        if not _NOT_A_UNIVERSAL.match(text, match.end()):
            found.append(AbsoluteMention(match.start(), match.end(), match.group(), "quantifier"))
    return sorted(found, key=lambda mention: mention.start)


def _following_stems(text: str, end: int, count: int) -> set[str]:
    return {token.stem for token in content_tokens(text[end : end + 80])[:count]}


def _zero_is_supported(noun_stems: set[str], sources: Sequence[str]) -> bool:
    """Whether a source says "no X" or "0 X" for an X the bullet gives after ``zero``."""
    for source in sources:
        tokens = content_tokens(source)
        for index, token in enumerate(tokens):
            if token.word in _NONE_WORDS:
                nearby = {t.stem for t in tokens[index + 1 : index + 1 + _SOURCE_REACH]}
                if nearby & noun_stems:
                    return True
    return False


def _quantifier_is_supported(noun_stems: set[str], sources: Sequence[str]) -> bool:
    """Whether a source quantifies (all, every, each, entire) something the bullet does."""
    for source in sources:
        for match in _ANY_QUANTIFIER.finditer(source):
            if _following_stems(source, match.end(), _SOURCE_REACH) & noun_stems:
                return True
    return False


def unsupported_absolutes(bullet: str, sources: Sequence[str]) -> list[AbsoluteMention]:
    """Return the absolutes of ``bullet`` that none of ``sources`` backs."""
    unsupported: list[AbsoluteMention] = []
    for mention in find_absolutes(bullet):
        if mention.kind == "plain":
            pattern = _PLAIN
            supported = any(_same_word(pattern, mention.text, source) for source in sources)
        elif mention.kind == "zero":
            noun_stems = _following_stems(bullet, mention.end, _NOUN_REACH)
            supported = _zero_is_supported(noun_stems, sources)
        else:
            noun_stems = _following_stems(bullet, mention.end, _NOUN_REACH)
            supported = _quantifier_is_supported(noun_stems, sources)
        if not supported:
            unsupported.append(mention)
    return unsupported


def _same_word(pattern: re.Pattern[str], word: str, source: str) -> bool:
    """Whether ``source`` has an absolute of the same family as ``word``."""
    family = _family(word)
    return any(_family(match.group()) == family for match in pattern.finditer(source))


def _family(word: str) -> str:
    lowered = word.lower()
    if lowered.startswith("guarantee"):
        return "guarantee"
    if lowered.startswith("eliminat"):
        return "eliminate"
    if lowered.startswith("exactly"):
        return "exactly-once"
    if lowered.startswith("100") or lowered.startswith("hundred"):
        return "100%"
    if lowered.startswith(("entire", "perfect", "flawless")):
        return lowered.removesuffix("ly")
    return lowered


# ---- claims about where and by whom ----------------------------------------------------------

_PRODUCTION: Final = re.compile(r"(?<![\w-])production\b", re.IGNORECASE)
_LEAD_VERB: Final = (
    r"(?:led|leading|lead|managed|managing|headed|directed|spearheaded|oversaw|supervised)"
)
_LEAD_CLAIM: Final = re.compile(
    rf"\b{_LEAD_VERB}\s+(?:(?:the|a|an|my|our|\d+(?:-\w+)?|\w+-person)\s+)+(?:\w+\s+){{0,2}}?"
    r"(?:team|teams|engineers|developers|squad|group|org|organization)\b",
    re.IGNORECASE,
)
_LEAD_SOURCE: Final = re.compile(rf"\b{_LEAD_VERB}\b", re.IGNORECASE)
_NEGATIONS: Final = frozenset(
    {"not", "no", "never", "without", "nor", "neither", "isn't", "wasn't"}
)
_NEGATION_REACH: Final = 3


def _production_is_stated(source: str) -> bool:
    """Whether ``source`` says something ran in production, and not that it did not."""
    for match in _PRODUCTION.finditer(source):
        words = re.findall(r"[\w']+", source[: match.start()].lower())[-_NEGATION_REACH:]
        if not _NEGATIONS & set(words):
            return True
    return False


def unsupported_claims(bullet: str, sources: Sequence[str]) -> list[AbsoluteMention]:
    """Return the ``production`` and ``led the team`` claims of ``bullet`` that no source makes.

    ``production`` is stated by a source that mentions it without a negation just before
    (``not production traffic`` states the opposite). A leadership claim is stated by a source
    that uses any verb of leading or managing.
    """
    found: list[AbsoluteMention] = []
    if not any(_production_is_stated(source) for source in sources):
        found += [
            AbsoluteMention(m.start(), m.end(), m.group(), "claim")
            for m in _PRODUCTION.finditer(bullet)
        ]
    if not any(_LEAD_SOURCE.search(source) for source in sources):
        found += [
            AbsoluteMention(m.start(), m.end(), m.group(), "claim")
            for m in _LEAD_CLAIM.finditer(bullet)
        ]
    return sorted(found, key=lambda mention: mention.start)
