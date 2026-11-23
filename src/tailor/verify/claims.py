"""Turn a DO NOT CLAIM item into key phrases, and find those phrases in a bullet.

An entry's DO NOT CLAIM list says what its bullets must not be stretched into, in the author's own
words: ``'exactly-once' processing (the consumer is at-least-once with deduplication)``. The part
in parentheses explains what is true instead and shares words with the real facts, so it is not
searched for. What is searched for is the claim itself:

* a quoted term is the key phrase of its claim (``exactly-once``);
* otherwise the claim is the key phrase, minus leading determiners (``any cost savings`` is
  ``cost savings``), and ``X or Y`` and ``X, Y`` are two phrases;
* a phrase is its content stems (stop words dropped, inflection and synonyms folded).

A bullet matches a phrase when it has the phrase's stems close together, all of them for a phrase
of up to five stems and four fifths of them for a longer one, and no negation just before.
``not production`` is a bullet saying what it is not, not a claim.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tailor.verify.text import STOP_WORDS, Token, content_tokens, word_set

_PARENTHETICAL: Final = re.compile(r"\([^()]*\)")
_QUOTED: Final = re.compile(r"(?<!\w)(['\"])(?P<text>.+?)\1(?!\w)")
_OR: Final = re.compile(r",\s+(?:or\s+)?|\s+or\s+", re.IGNORECASE)
_DETERMINER: Final = re.compile(r"^(?:any|an?|the|all|some|that)\s+", re.IGNORECASE)

_NEGATORS: Final = word_set(
    "not no never without non nor neither isn wasn doesn didn aren weren cannot"
)
_NEGATION_REACH: Final = 4
_WINDOW_SLACK: Final = 4
_ALL_STEMS_UP_TO: Final = 5
_LONG_PHRASE_SHARE: Final = 0.8


@dataclass(frozen=True, slots=True)
class KeyPhrase:
    """One thing a DO NOT CLAIM item says not to claim.

    Attributes:
        text: The phrase as written in the item, cleaned up.
        stems: Its content stems, in order.
        claim: The whole DO NOT CLAIM item it came from, explanation included.
    """

    text: str
    stems: tuple[str, ...]
    claim: str


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    """Where a phrase was found in a text: character span and whether a negation precedes it."""

    start: int
    end: int
    negated: bool


def _strip_parentheticals(text: str) -> str:
    while True:
        stripped = _PARENTHETICAL.sub(" ", text)
        if stripped == text:
            return " ".join(text.split())
        text = stripped


def _stems(text: str) -> tuple[str, ...]:
    return tuple(
        token.stem for token in content_tokens(text) if len(token.word) > 1 or token.word.isdigit()
    )


def key_phrases(claim: str) -> list[KeyPhrase]:
    """Return the key phrases of one DO NOT CLAIM item; none if nothing is left of it."""
    head = _strip_parentheticals(claim)
    phrases: list[KeyPhrase] = []
    for alternative in _OR.split(head):
        quoted = [match["text"].strip() for match in _QUOTED.finditer(alternative)]
        texts = quoted or [_DETERMINER.sub("", alternative.strip()).strip(" ,.:;")]
        for text in texts:
            stems = _stems(text)
            if stems:
                phrases.append(KeyPhrase(text, stems, claim))
    return phrases


def find_phrase(tokens: Sequence[Token], phrase: KeyPhrase) -> list[PhraseMatch]:
    """Find ``phrase`` in tokenised text: its stems close together, in any order.

    Args:
        tokens: The tokens of the text (see :func:`tailor.verify.text.tokenize`).
        phrase: The key phrase.

    Returns:
        The matches, in order; each is the earliest window that holds enough of the stems.
    """
    wanted = set(phrase.stems)
    needed = len(wanted)
    if needed > _ALL_STEMS_UP_TO:
        needed = math.ceil(_LONG_PHRASE_SHARE * needed)
    content = [(index, token) for index, token in enumerate(tokens) if token.word not in STOP_WORDS]
    window = len(phrase.stems) + _WINDOW_SLACK
    matches: list[PhraseMatch] = []
    covered_until = -1
    for position, (index, token) in enumerate(content):
        if token.stem not in wanted or position <= covered_until:
            continue
        inside = content[position : position + window]
        present = {t.stem for _, t in inside if t.stem in wanted}
        if len(present) < needed:
            continue
        last = max(offset for offset, (_, t) in enumerate(inside) if t.stem in wanted)
        end_token = inside[last][1]
        before = [t.word for t in tokens[max(0, index - _NEGATION_REACH) : index]]
        matches.append(
            PhraseMatch(token.start, end_token.end, any(word in _NEGATORS for word in before))
        )
        covered_until = position + last
    return matches
