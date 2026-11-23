"""Words, stems and sentences: the text handling behind the claim checks.

The do-not-claim and causal checks compare what a bullet says with what a source says, and the two
rarely use the same words (``caching`` and ``cache``, ``reduced`` and ``cut``). So they compare
stems. The stemmer is Martin Porter's 1980 algorithm, written out here (it is short, and a
dependency for it would be larger than the code), with three additions that suit resume prose:

* a table of irregular words (``built`` is ``build``, ``ownership`` and ``owned`` are ``own``)
  and of words Porter wrongly merges (``production`` is not ``product``);
* a final ``y`` to ``i`` so that ``deploy`` and ``deployment`` meet;
* a few synonym groups on stems, so that ``reduced``, ``lowered`` and ``cut`` are one word.

Tokens carry their position so a finding can point at the words of the bullet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

# A letter-digit identifier such as p99 or k6 is one token; a number may carry separators.
_TOKEN: Final = re.compile(r"[A-Za-z]+\d+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z]+|\$(?=\d)")


def word_set(text: str) -> frozenset[str]:
    """The whitespace-separated words of ``text`` as a set, for writing word lists readably."""
    return frozenset(text.split())


STOP_WORDS: Final = word_set(
    """
    a an the of in on at to for with by from and or as is are was were be been being it its that
    this these those into over across per via using use used than then so such also do does did has
    have had will would can could may might should about up out off each their there which who whom
    whose what when while where how both any some they them we our us you your he she him her
    """
)
# "each", "any" and "some" are determiners, not claims; the escalation check reads the original
# text for the quantifiers that matter.

_WORD_STEMS: Final = {
    "production": "production",
    "productions": "production",
    "add": "ad",
    "adds": "ad",
    "added": "ad",
    "adding": "ad",
    "built": "build",
    "wrote": "write",
    "written": "write",
    "ran": "run",
    "found": "find",
    "kept": "keep",
    "led": "lead",
    "own": "own",
    "owns": "own",
    "owned": "own",
    "owning": "own",
    "owner": "own",
    "owners": "own",
    "ownership": "own",
}

_SYNONYM_GROUPS: Final = (
    ("cut", "reduc", "reduct", "lower", "decreas", "shrink", "shrank", "slash", "trim"),
    ("build", "implement", "develop", "creat", "construct", "author"),
    ("increas", "rais", "boost", "improv", "grew", "grow", "gain"),
    ("handl", "sustain", "process"),
    ("replac", "swap", "substitut"),
)
_SYNONYMS: Final = {word: group[0] for group in _SYNONYM_GROUPS for word in group}


@dataclass(frozen=True, slots=True)
class Token:
    """A word or number of a text.

    Attributes:
        word: Lower case; thousands separators removed from numbers.
        stem: The comparison form of ``word`` (see :func:`stem`); numbers are their own stem.
        start: Index of its first character in the text.
        end: Index after its last character.
    """

    word: str
    stem: str
    start: int
    end: int


# ---- Porter stemmer --------------------------------------------------------------------------


def _is_consonant(word: str, index: int) -> bool:
    char = word[index]
    if char in "aeiou":
        return False
    if char == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True


def _measure(stem_: str) -> int:
    """Porter's m: the number of vowel-consonant sequences in ``stem_``."""
    count = 0
    index = 0
    length = len(stem_)
    while index < length and _is_consonant(stem_, index):
        index += 1
    while index < length:
        while index < length and not _is_consonant(stem_, index):
            index += 1
        if index >= length:
            break
        count += 1
        while index < length and _is_consonant(stem_, index):
            index += 1
    return count


def _has_vowel(stem_: str) -> bool:
    return any(not _is_consonant(stem_, i) for i in range(len(stem_)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """Whether ``word`` ends consonant-vowel-consonant, the last not w, x or y."""
    if len(word) < 3:
        return False
    return (
        _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


def _replace_suffix(word: str, suffix: str, replacement: str, minimum: int) -> str | None:
    """Replace ``suffix`` when the remaining stem has measure above ``minimum``."""
    if not word.endswith(suffix):
        return None
    base = word[: len(word) - len(suffix)]
    return base + replacement if _measure(base) > minimum else word


_STEP2: Final = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)
_STEP3: Final = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)
_STEP4: Final = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
            word = word[: -len(suffix)]
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step_table(word: str, table: tuple[tuple[str, str], ...]) -> str:
    for suffix, replacement in table:
        if word.endswith(suffix):
            return _replace_suffix(word, suffix, replacement, 0) or word
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if not word.endswith(suffix):
            continue
        base = word[: -len(suffix)]
        if suffix == "ion" and not base.endswith(("s", "t")):
            continue
        return base if _measure(base) > 1 else word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        base = word[:-1]
        measure = _measure(base)
        if measure > 1 or (measure == 1 and not _cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


@lru_cache(maxsize=8192)
def stem(word: str) -> str:
    """Return the comparison form of ``word``: Porter stem, irregulars and synonyms applied."""
    word = word.lower()
    if word in _WORD_STEMS:
        return _WORD_STEMS[word]
    if len(word) <= 2 or not word.isalpha():
        return word
    word = _step1c(_step1b(_step1a(word)))
    word = _step_table(word, _STEP2)
    word = _step_table(word, _STEP3)
    word = _step5(_step4(word))
    if word.endswith("y") and len(word) > 2:
        word = word[:-1] + "i"
    return _SYNONYMS.get(word, word)


# ---- tokens and sentences --------------------------------------------------------------------


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into words and numbers, hyphenated words into their parts."""
    tokens: list[Token] = []
    for match in _TOKEN.finditer(text):
        word = match.group().lower().replace(",", "")
        word = "dollar" if word == "$" else word
        tokens.append(Token(word=word, stem=stem(word), start=match.start(), end=match.end()))
    return tokens


def content_tokens(text: str) -> list[Token]:
    """The tokens of ``text`` that carry content: stop words are dropped, negation is kept."""
    return [token for token in tokenize(text) if token.word not in STOP_WORDS]


_SENTENCE_BREAK: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\s*;\s*")


def sentences(text: str) -> list[str]:
    """Split ``text`` at semicolons and at sentence ends (a period before a capital letter).

    A decimal point or an abbreviation such as ``e.g.`` followed by a lower-case word does not end
    a sentence. Pieces lose surrounding whitespace; empty ones are dropped.
    """
    return [piece.strip() for piece in _SENTENCE_BREAK.split(text) if piece.strip()]
