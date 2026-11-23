"""Read the structure that bank authors put into an entry's free-text ``scope``.

A scope holds provenance and limits. Two conventions in it carry meaning for the verifier:

* Labelled parts. A label is an all-caps word or words followed by a colon (``EVIDENCE:``,
  ``ORIGIN:``, ``DO NOT CLAIM:``, ``REVIEWED 2026-03-02:``). A part runs from its label to the next
  label or the end of the scope. Text before the first label is an unlabelled part. A label has at
  least four letters, follows whitespace or the start, and is followed by whitespace or the end,
  so ``FFT:`` and ``(README: x)`` stay prose.
* ``Same project as <id>``: this entry describes the same project as the entry with that id.

Everything here works on plain text; normalize LaTeX with :func:`tailor.bank.latex.normalize_latex`
first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

DO_NOT_CLAIM: Final = "DO NOT CLAIM"

_LABEL: Final = re.compile(r"(?<!\S)(?P<label>[A-Z][A-Z0-9-]*(?:\s+[A-Z0-9][A-Z0-9-]*)*):(?=\s|$)")
_MIN_LABEL_LETTERS: Final = 4

_SAME_PROJECT: Final = re.compile(r"Same project as (?P<id>[A-Za-z0-9][A-Za-z0-9_-]*)")

_OPENERS: Final = "(["
_CLOSERS: Final = ")]"
_SENTENCE_ENDS: Final = ".!?"
# Words whose final period does not end a sentence. Lower case, without the period.
_ABBREVIATIONS: Final = frozenset({"e.g", "i.e", "vs", "approx", "cf", "incl", "est"})
_LAST_WORD: Final = re.compile(r"[A-Za-z.]+$")
_TRAILING: Final = re.compile(r"[\s.;!?]+$")


@dataclass(frozen=True, slots=True)
class ScopePart:
    """One labelled (or, before the first label, unlabelled) part of a scope."""

    label: str | None
    text: str


def split_parts(scope: str) -> list[ScopePart]:
    """Split ``scope`` into its parts, in order, each with its text stripped."""
    labels = [
        match
        for match in _LABEL.finditer(scope)
        if sum(char.isalpha() for char in match["label"]) >= _MIN_LABEL_LETTERS
    ]
    parts: list[ScopePart] = []
    head = scope[: labels[0].start()] if labels else scope
    if head.strip():
        parts.append(ScopePart(None, head.strip()))
    for index, match in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(scope)
        parts.append(ScopePart(" ".join(match["label"].split()), scope[match.end() : end].strip()))
    return parts


def _ends_sentence(text: str, index: int) -> bool:
    """Whether the sentence-ending character at ``index`` really ends a sentence."""
    following = text[index + 1 :]
    if following and not following[0].isspace():
        return False  # "3.5K", "v1.2", "example.com"
    next_char = following.lstrip()[:1]
    if next_char and not (next_char.isupper() or next_char in "\"'"):
        return False
    if text[index] == ".":
        last_word = _LAST_WORD.search(text[:index])
        if last_word and last_word.group().lower() in _ABBREVIATIONS:
            return False
    return True


def _split(text: str, *, track_brackets: bool) -> tuple[list[str], bool]:
    """Cut at ';' and sentence ends; report whether the brackets were balanced."""
    pieces: list[str] = []
    start = 0
    depth = 0
    balanced = True
    for index, char in enumerate(text):
        if track_brackets:
            if char in _OPENERS:
                depth += 1
            elif char in _CLOSERS:
                if depth == 0:
                    balanced = False
                else:
                    depth -= 1
            if depth > 0:
                continue
        if char == ";" or (char in _SENTENCE_ENDS and _ends_sentence(text, index)):
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return pieces, balanced and depth == 0


def split_claims(text: str) -> list[str]:
    """Split a list of claims at semicolons and sentence ends, keeping parentheticals whole.

    ``a (b; c); d. E`` gives ``["a (b; c)", "d", "E"]``: a bracketed explanation belongs to the
    claim it explains. If the brackets do not balance, an unclosed one would swallow every later
    claim, so the text is split at every boundary regardless of brackets. Pieces lose their
    surrounding whitespace and trailing punctuation; empty ones are dropped.
    """
    pieces, balanced = _split(text, track_brackets=True)
    if not balanced:
        pieces, _ = _split(text, track_brackets=False)
    claims = (_TRAILING.sub("", piece.strip()) for piece in pieces)
    return [claim for claim in claims if claim]


def do_not_claim_claims(scope: str) -> list[str]:
    """Return the claims of every ``DO NOT CLAIM:`` part of ``scope``, in order."""
    return [
        claim
        for part in split_parts(scope)
        if part.label == DO_NOT_CLAIM
        for claim in split_claims(part.text)
    ]


def strip_do_not_claim(scope: str) -> str:
    """Return ``scope`` without its ``DO NOT CLAIM:`` parts.

    What an entry must not claim cannot also be evidence for a claim, so provenance text is built
    from what remains. Joining the kept parts could in theory splice a new label together from the
    ends of two of them, so the removal repeats until none is left; each round shortens the text.
    """
    while True:
        parts = split_parts(scope)
        if all(part.label != DO_NOT_CLAIM for part in parts):
            return _join(parts)
        scope = _join(part for part in parts if part.label != DO_NOT_CLAIM)


def _join(parts: Iterable[ScopePart]) -> str:
    return " ".join(
        f"{part.label}: {part.text}".strip() if part.label else part.text for part in parts
    )


def same_project_refs(scope: str) -> list[str]:
    """Return the ids that ``scope`` says it is the same project as, in order, without repeats."""
    refs = (match["id"].rstrip("-_") for match in _SAME_PROJECT.finditer(scope))
    return list(dict.fromkeys(ref for ref in refs if ref))
