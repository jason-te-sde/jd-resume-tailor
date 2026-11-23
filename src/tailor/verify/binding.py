"""Bind a number of a bullet to the claim it supports.

A bullet's number is supported by a fact only if the fact is *about* the same thing. Pooling the
numbers of every cited fact lets ``Cut p99 latency from 412 ms to 11 minutes`` pass, because
``11 minutes`` is the deploy time of another fact of the same entry, in the right unit family.

**The heuristic.** Take the words around the number in the bullet: up to eight content words
before it and four after it, inside its sentence. The window stops at a connector such as ``by``,
``through`` or ``thanks to`` (the words after it say how the number was reached), and at an
``and`` that starts another claim, or at a comma, unless that would leave it empty. Numbers, units,
generic verbs (``cut``, ``raised``, ``built``...) and the first word of the sentence do not count.
What is left names the metric: ``p99 latency``, ``deploy time``. The number is supported by a fact
only if the fact shares at least one of those words, compared as stems (``deploys`` and ``deploy``
meet; ``reduced`` and ``cut`` are generic). A number whose window has no metric word, as in
``Reached 41,000.``, binds to any fact.

It is a word-overlap heuristic: a bullet that names the metric in words the fact does not use
(``response time`` for ``latency``) is not bound and its number is reported as fabricated. The
verifier evaluation measures how often that happens.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

from tailor.verify.quantities import Quantity
from tailor.verify.text import STOP_WORDS, Token, stem, tokenize, word_set

_BEFORE: Final = 8
_AFTER: Final = 4
_BREAK: Final = re.compile(r"[.;:](?=\s|$)")
# What follows these words is how the number was reached or what to make of it, not what it
# measures: "18,000 writes/s thanks to group commit", "41,000 checks/s as a latency-bounded rate".
# A comma may also just continue the phrase ("the API, from 412 ms to 138 ms"), so the window keeps
# what precedes it when nothing but filler follows.
_SOFT: Final = re.compile(
    r"\b(?:by|through|thanks to|via|because(?: of)?|due to|so that|enabling|allowing|as|thereby)\b"
    r"|,",
    re.IGNORECASE,
)
# These start another claim: "cut latency to 138 ms and raised throughput to 9,200 events/s".
_HARD: Final = re.compile(r"\b(?:and|while|but|plus)\b", re.IGNORECASE)
_FIRST_WORD: Final = re.compile(r"\s*[A-Za-z]+")
_UNITS: Final = word_set(
    """
    ms msec msecs s sec secs min mins h hr hrs kb mb gb tb kib mib gib tib b x pct percent fps
    req reqs second seconds minute minutes hour hours day days week weeks month months year years
    per k m
    """
)
_GENERIC_WORDS: Final = word_set(
    """
    cut cuts cutting reduce reduced reducing lower lowered lowering decrease decreased increase
    increased raise raised raising improve improved improving boost boosted grew grow rose rise
    build built building wrote write written added add adding moved move made make ran run
    achieved achieve delivered deliver reached reach got get took take taking brought bring ended
    end saved save saving dropped drop fell fall helped help showed show ensured ensure
    handled handle sustained sustain processed process replaced replace using used use
    over across within after before during while per also now instead down up only just total
    overall new
    """
)
_GENERIC_STEMS: Final = frozenset(stem(word) for word in _GENERIC_WORDS)


def _is_content(token: Token) -> bool:
    word = token.word
    return (
        word not in STOP_WORDS
        and word not in _UNITS
        and not word[0].isdigit()
        and token.stem not in _GENERIC_STEMS
    )


def claim_stems(text: str, quantity: Quantity) -> frozenset[str]:
    """The stems of the words that say what ``quantity`` measures, in its bullet ``text``."""
    before = text[: quantity.start]
    if ends := list(_BREAK.finditer(before)):
        before = before[ends[-1].end() :]
    # A bullet starts with its verb (``Held``, ``Reached``): it says what was done, not to what.
    first = _FIRST_WORD.match(before)
    before = before[first.end() :] if first else before
    after = text[quantity.end :]
    if end := _BREAK.search(after):
        after = after[: end.start()]
    for stop in (_SOFT, _HARD):
        if found := stop.search(after):
            after = after[: found.start()]
    if hard := list(_HARD.finditer(before)):
        before = before[hard[-1].end() :]
    soft = list(_SOFT.finditer(before))
    # "Reduced latency by 41%" has nothing between "by" and the number: keep what precedes it.
    if soft and any(_is_content(t) for t in tokenize(before[soft[-1].end() :])):
        before = before[soft[-1].end() :]
    near_before = [t for t in tokenize(before) if _is_content(t)][-_BEFORE:]
    near_after = [t for t in tokenize(after) if _is_content(t)][:_AFTER]
    return frozenset(token.stem for token in (*near_before, *near_after))


@lru_cache(maxsize=1024)
def fact_stems(fact: str) -> frozenset[str]:
    """The stems of every content word of a fact."""
    return frozenset(token.stem for token in tokenize(fact) if _is_content(token))


def is_bound(claim: frozenset[str], fact: str) -> bool:
    """Whether ``fact`` is about the claim: it shares a metric word, or the claim has none."""
    return not claim or bool(claim & fact_stems(fact))
