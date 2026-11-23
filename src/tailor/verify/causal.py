"""Causal links: "X by Y", "X so that Y", "X, enabling Y": a bullet saying why something happened.

A source that reports a measurement rarely says what caused it; a rewrite that adds "by batching
writes" or "so that reads stay fast" has claimed a cause nobody measured. The check finds causal
connectors in a bullet and asks whether some single sentence of a source holds both sides of the
link: the *mechanism* (what is said to cause) and the *result* (what is said to follow).

* Mechanism-after connectors: ``by X``, ``with X``, ``via X``, ``thanks to X``, ``due to X``,
  ``because (of) X``, ``driven by X``. The result is what stands before the connector.
* Result-after connectors: ``so that X``, ``in order to X``, ``resulting in X``, ``, which let X``
  and effect participles (``, enabling X``, ``, spreading X``, ``, reducing X``). The mechanism is
  what stands before the connector.

A side is "held" by a sentence when enough of its content stems are in the sentence. A link where
the mechanism's stems are in no source, or the two sides are in different sentences, is flagged.
``by`` before a number or after ``led`` or ``followed``, and ``with`` before a number, are not
causal links.

This is a heuristic over word overlap. It misses a link phrased with a word it does not know, and
it flags a faithful paraphrase whose words differ too much from the source's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tailor.verify.text import content_tokens, sentences, word_set

Direction = Literal["mechanism_after", "result_after"]

_MECHANISM_SHARE: Final = 0.6
_RESULT_SHARE_AFTER_MECHANISM: Final = 0.5
_RESULT_SHARE_EFFECT: Final = 0.75

_NOT_CAUSAL_BEFORE_BY: Final = word_set(
    "led owned followed written maintained reviewed used hosted approved supported backed "
    "authored created run tested verified checked measured validated confirmed proven shown "
    "evaluated"
)
_EFFECT_VERBS: Final = (
    "enabling|allowing|ensuring|letting|making|keeping|preventing|avoiding|eliminating|reducing|"
    "lowering|cutting|improving|boosting|increasing|saving|speeding|accelerating|minimizing|"
    "maximizing|spreading|distributing|balancing|scaling|protecting|thereby|thus"
)
_NUMBER_OR_MONEY: Final = r"(?!\s*[\d$])"

_MECHANISM_CONNECTORS: Final = re.compile(
    rf"\b(?P<by>by){_NUMBER_OR_MONEY}\s+[A-Za-z]"
    rf"|\b(?P<with>with){_NUMBER_OR_MONEY}\s+(?!no\b|zero\b)[A-Za-z]"
    r"|\b(?P<other>via|through|thanks to|due to|owing to|because of|because|driven by|"
    r"powered by)\b",
    re.IGNORECASE,
)
_RESULT_CONNECTORS: Final = re.compile(
    r"\b(?P<phrase>so that|so as to|in order to|resulting in|resulted in|leading to|led to)\b"
    rf"|,\s*(?P<effect>{_EFFECT_VERBS})\b"
    r"|,\s*(?P<which>which\s+(?:let|allowed|enabled|made|reduced|cut|improved|increased|"
    r"prevented|ensured))\b",
    re.IGNORECASE,
)
# A decimal point and a thousands comma sit inside a number; only a stop followed by a space (or the
# end) or a comma not followed by a digit ends a clause.
_CLAUSE_END: Final = re.compile(
    r";|[.:](?=\s|$)|,(?!\d)|\s+(?:which|while|so that)\b|\)", re.IGNORECASE
)
_SENTENCE_START: Final = re.compile(r";\s*|[.:]\s+")


@dataclass(frozen=True, slots=True)
class CausalLink:
    """A causal connector of a bullet with the two sides of the link.

    Attributes:
        start: Index where the flagged span starts: the connector, or for an effect participle
            the verb after its comma.
        end: Index after the last character of the clause that the connector introduces.
        text: ``bullet[start:end]``.
        mechanism: The clause said to cause.
        result: The clause said to follow.
        direction: Which side the connector introduces.
    """

    start: int
    end: int
    text: str
    mechanism: str
    result: str
    direction: Direction


def _clause_end(text: str, start: int) -> int:
    match = _CLAUSE_END.search(text, start)
    return match.start() if match else len(text)


def _sentence_start(text: str, index: int) -> int:
    starts = [m.end() for m in _SENTENCE_START.finditer(text[:index])]
    return starts[-1] if starts else 0


def find_links(bullet: str) -> list[CausalLink]:
    """Return the causal links of ``bullet``, outer ones only, in order."""
    links: list[CausalLink] = []
    for match in _MECHANISM_CONNECTORS.finditer(bullet):
        start = match.start()
        if match.group("by"):
            previous = re.findall(r"[A-Za-z]+", bullet[:start])[-2:]
            if any(word.lower() in _NOT_CAUSAL_BEFORE_BY for word in previous):
                continue
        end = _clause_end(bullet, match.end())
        before = bullet[_sentence_start(bullet, start) : start]
        result = before if before.strip() else bullet[end : _clause_end(bullet, end + 1)]
        links.append(
            CausalLink(start, end, bullet[start:end], bullet[start:end], result, "mechanism_after")
        )
    for match in _RESULT_CONNECTORS.finditer(bullet):
        start = match.start("phrase") if match.group("phrase") else match.start() + 1
        if match.group("effect") or match.group("which"):
            start = match.start("effect") if match.group("effect") else match.start("which")
        end = _clause_end(bullet, match.end())
        mechanism = bullet[_sentence_start(bullet, start) : start]
        links.append(
            CausalLink(start, end, bullet[start:end], mechanism, bullet[start:end], "result_after")
        )
    links.sort(key=lambda link: link.start)
    outer: list[CausalLink] = []
    for link in links:
        if outer and link.start < outer[-1].end:
            continue
        outer.append(link)
    return outer


def _stems(text: str) -> set[str]:
    return {
        token.stem for token in content_tokens(text) if len(token.word) > 1 or token.word.isdigit()
    }


def _share(wanted: set[str], have: set[str]) -> float:
    return len(wanted & have) / len(wanted) if wanted else 1.0


def is_linked(link: CausalLink, sources: Sequence[str]) -> bool:
    """Whether one sentence of a source holds both sides of ``link``."""
    mechanism = _stems(link.mechanism)
    result = _stems(link.result)
    # The connector word itself is not part of either side.
    if link.direction == "mechanism_after":
        mechanism_share, result_share = _MECHANISM_SHARE, _RESULT_SHARE_AFTER_MECHANISM
    else:
        mechanism_share, result_share = _RESULT_SHARE_AFTER_MECHANISM, _RESULT_SHARE_EFFECT
        if link.text.lower().startswith(("so that", "so as to", "in order to")):
            result_share = _MECHANISM_SHARE
    if not mechanism or not result:
        return True
    for source in sources:
        for sentence in sentences(source):
            have = _stems(sentence)
            if _share(mechanism, have) >= mechanism_share and _share(result, have) >= result_share:
                return True
    return False
