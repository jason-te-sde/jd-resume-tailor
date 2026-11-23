"""Find the quantities in a piece of text: what number, in what unit, written how.

The verifier compares what a bullet claims with what its sources say, and for numbers that takes
more than a regular expression for digits. ``3.7K+``, ``3,781`` and ``3781`` are one number written
three ways, ``138 ms`` and ``0.138 s`` are one duration, ``0.875`` and ``87.5%`` are one share, and
``9,200 events/s`` is a throughput that a bare ``9,200`` is not. :func:`extract_quantities` reads
all of that and keeps, for each quantity, the value, the unit family, the text it came from and how
precisely it was written, which is what the rounding rules need.

Recognised forms:

* integers and decimals, with thousands separators (``3,781``, ``0.875``);
* magnitude suffixes and words: ``K``, ``M``, ``B``, ``thousand``, ``million``, ``billion``;
* percentages (``74%``, ``74 percent``) and multipliers (``2.5x``);
* durations (``ms``, ``s``, ``sec``, ``min``, ``h``, days, weeks and so on, also hyphenated as in
  ``10-minute``), normalised to seconds;
* throughputs (``events/s``, ``req/s``, ``jobs per second``, ``jobs/min``, ``fps``), normalised to
  per second;
* sizes (``KB``, ``MB``, ``GB``, ``TB``, normalised to bytes) and dollar amounts;
* a trailing ``+`` (``3.7K+``), which says "at least" and is kept apart from the value;
* ranges (``1.2-2.5x``, ``2,844 to 4,250 req/s``) and arrow pairs (``398ms -> 34ms``,
  ``from 412 to 138 ms``): each end is its own quantity and an end with no unit takes the unit of
  the other.

Not quantities: numbers inside identifiers (``p99``, ``k6``, ``v1.2``, ``x86``, ``F1``) and
ordinals (``99th``). Number words (``one``, ``two``) are not read, and ``over 4,800`` carries no
qualifier because ``over`` more often means ``across`` than ``more than``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Final, Literal

Family = Literal["count", "percent", "multiplier", "time", "rate", "money", "size"]

_SECOND: Final = Fraction(1)
_TIME_UNITS: Final[dict[str, tuple[str, Fraction]]] = {
    "ms": ("ms", Fraction(1, 1000)),
    "msec": ("ms", Fraction(1, 1000)),
    "msecs": ("ms", Fraction(1, 1000)),
    "millisecond": ("ms", Fraction(1, 1000)),
    "milliseconds": ("ms", Fraction(1, 1000)),
    "s": ("s", _SECOND),
    "sec": ("s", _SECOND),
    "secs": ("s", _SECOND),
    "second": ("s", _SECOND),
    "seconds": ("s", _SECOND),
    "min": ("min", Fraction(60)),
    "mins": ("min", Fraction(60)),
    "minute": ("min", Fraction(60)),
    "minutes": ("min", Fraction(60)),
    "h": ("h", Fraction(3600)),
    "hr": ("h", Fraction(3600)),
    "hrs": ("h", Fraction(3600)),
    "hour": ("h", Fraction(3600)),
    "hours": ("h", Fraction(3600)),
    "day": ("d", Fraction(86_400)),
    "days": ("d", Fraction(86_400)),
    "week": ("wk", Fraction(604_800)),
    "weeks": ("wk", Fraction(604_800)),
    "wk": ("wk", Fraction(604_800)),
    "wks": ("wk", Fraction(604_800)),
    "month": ("mo", Fraction(2_592_000)),
    "months": ("mo", Fraction(2_592_000)),
    "year": ("yr", Fraction(31_536_000)),
    "years": ("yr", Fraction(31_536_000)),
    "yr": ("yr", Fraction(31_536_000)),
    "yrs": ("yr", Fraction(31_536_000)),
}
_PERIODS: Final = {
    word: unit
    for word, unit in _TIME_UNITS.items()
    if word not in {"ms", "msec", "msecs", "millisecond", "milliseconds"}
}
_SIZE_UNITS: Final = {
    "kb": ("KB", Fraction(10**3)),
    "mb": ("MB", Fraction(10**6)),
    "gb": ("GB", Fraction(10**9)),
    "tb": ("TB", Fraction(10**12)),
    "kib": ("KiB", Fraction(2**10)),
    "mib": ("MiB", Fraction(2**20)),
    "gib": ("GiB", Fraction(2**30)),
    "tib": ("TiB", Fraction(2**40)),
    "byte": ("B", Fraction(1)),
    "bytes": ("B", Fraction(1)),
}
_MAGNITUDE_LETTERS: Final = {"K": 10**3, "k": 10**3, "M": 10**6, "B": 10**9}
_MAGNITUDE_WORDS: Final = {"thousand": 10**3, "million": 10**6, "billion": 10**9}

_NUMBER: Final = re.compile(r"(?<![\w.])(?<![A-Za-z][-/])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_SIZE: Final = re.compile(rf"\s?(?P<unit>{'|'.join(_SIZE_UNITS)})(?![A-Za-z0-9])", re.IGNORECASE)
_WORD_MAGNITUDE: Final = re.compile(rf"\s+(?P<word>{'|'.join(_MAGNITUDE_WORDS)})\b")
_LETTER_MAGNITUDE: Final = re.compile(r"(?P<letter>[KkMB])(?![A-Za-z0-9])")
_PERCENT: Final = re.compile(r"(?:\s?%|\s+(?:percent|pct)\b)")
_MULTIPLIER: Final = re.compile(r"[x\u00d7](?![A-Za-z0-9])")
_TIME_UNIT_NAMES: Final = "|".join(sorted(_TIME_UNITS, key=len, reverse=True))
# "312 simulated ms" and "6 analyst-hours" are durations with a word in front of the unit.
_TIME: Final = re.compile(
    r"(?:\s+simulated|\s+[A-Za-z]+(?=-(?:hours?|days?|weeks?|months?)\b))?"
    rf"[-\s]?(?P<unit>{_TIME_UNIT_NAMES})(?![A-Za-z0-9])"
)
_RATE: Final = re.compile(
    r"(?:\s*[A-Za-z][A-Za-z-]{0,24})?"
    rf"(?:\s*/\s*|\s+per\s+)(?P<period>{'|'.join(sorted(_PERIODS, key=len, reverse=True))})"
    r"(?![A-Za-z0-9])"
)
_FRAME_RATE: Final = re.compile(r"\s?fps(?![A-Za-z0-9])", re.IGNORECASE)
_ORDINAL: Final = re.compile(r"(?:st|nd|rd|th)(?![A-Za-z0-9])")

_QUALIFIER: Final = re.compile(
    r"(?:\b(?P<word>up to|at most|at least|more than|less than|about|around|approximately|"
    r"roughly|nearly|almost|exceeding|above|under|below|within)\s+|(?P<mark>[~\u2248<>])\s*)$",
    re.IGNORECASE,
)
_BASELINE: Final = re.compile(
    r"\b(?P<word>from|against|vs\.?|versus|compared (?:to|with)|baseline of)\s+"
    r"(?:(?:an?|the)\s+)?(?:(?:~|about|around|approximately)\s*)?$",
    re.IGNORECASE,
)
_LINK: Final = re.compile(r"\s*(?P<link>->|\u2192|=>|-|\u2013|\u2014|\bto\b)\s*", re.IGNORECASE)
_ARROWS: Final = ("->", "\u2192", "=>")

_LOWER_BOUND_QUALIFIERS: Final = frozenset({"at least", "more than", "exceeding", "above", ">"})
_MAXIMUM_QUALIFIERS: Final = frozenset({"up to", "at most"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Quantity:
    """A number as a text states it.

    Attributes:
        value: The value in the base unit of its family, exactly: seconds for ``time``, per second
            for ``rate``, bytes for ``size``, dollars for ``money``, the written number for
            ``count``, ``percent`` (74 for ``74%``) and ``multiplier``.
        family: What kind of number it is. Quantities compare only within a family.
        unit: The unit as a short canonical label (``ms``, ``%``, ``/s``, ``KB``); empty for a
            count.
        text: The characters it came from, unit and ``+`` included, qualifier excluded.
        start: Index of the first character of ``text`` in the input.
        end: Index after the last character of ``text``.
        plus: Whether it ends in ``+``: a claim of "more than this".
        qualifier: The phrase just before it that bounds or softens it (``up to``, ``at least``,
            ``about``, ``~``, ``within``...), lower case; ``None`` when there is none.
        baseline: Whether it is the starting point of a comparison (``from 412 ms to``,
            ``against 0.71``), where rounding in the usual flattering direction flips.
        precision_fine: The place value of the last digit written, in ``value``'s base unit,
            counting every written zero as a digit (``8,000`` is written to the unit).
        precision_coarse: The same, reading trailing zeros of a whole number as padding
            (``8,000`` is written to the thousand, ``70`` to the ten).
        factor: How many base units one written unit is (60 for ``min``), kept so a unit read from
            a neighbour can be applied to this number.
    """

    value: Fraction
    family: Family
    unit: str
    text: str
    start: int
    end: int
    plus: bool = False
    qualifier: str | None = None
    baseline: bool = False
    precision_fine: Fraction = Fraction(1)
    precision_coarse: Fraction = Fraction(1)
    factor: Fraction = Fraction(1)

    @property
    def is_lower_bound(self) -> bool:
        """Whether the text says "at least this much": a trailing ``+`` or ``at least``-like."""
        return self.plus or self.qualifier in _LOWER_BOUND_QUALIFIERS

    @property
    def is_maximum(self) -> bool:
        """Whether the text says "up to this much"."""
        return self.qualifier in _MAXIMUM_QUALIFIERS


@dataclass(frozen=True, slots=True)
class _Unit:
    family: Family
    label: str
    factor: Fraction
    end: int


def _precision(literal: str, scale: Fraction) -> tuple[Fraction, Fraction]:
    """The fine and coarse place value of the last digit of ``literal``, times ``scale``."""
    whole, _, decimals = literal.partition(".")
    if decimals:
        place = Fraction(1, 10 ** len(decimals)) * scale
        return place, place
    stripped = whole.rstrip("0")
    padding = len(whole) - len(stripped) if stripped else 0
    return scale, Fraction(10) ** padding * scale


def _read_unit(text: str, index: int) -> _Unit | None:
    """The unit that starts at ``index`` (just after the number and its suffix), if any."""
    tail = text[index:]
    if match := _PERCENT.match(tail):
        return _Unit("percent", "%", Fraction(1), index + match.end())
    if match := _MULTIPLIER.match(tail):
        return _Unit("multiplier", "x", Fraction(1), index + match.end())
    if match := _TIME.match(tail):
        label, seconds = _TIME_UNITS[match["unit"].lower()]
        return _Unit("time", label, seconds, index + match.end())
    if match := _RATE.match(tail):
        label, seconds = _PERIODS[match["period"].lower()]
        return _Unit("rate", f"/{label}", 1 / seconds, index + match.end())
    if match := _FRAME_RATE.match(tail):
        return _Unit("rate", "/s", Fraction(1), index + match.end())
    return None


def _qualifier(text: str, start: int) -> str | None:
    match = _QUALIFIER.search(text[:start])
    if not match:
        return None
    return (match["word"] or match["mark"]).lower()


def _baseline_prefix(text: str, start: int) -> str | None:
    match = _BASELINE.search(text[:start])
    return match["word"].lower() if match else None


def _read_quantity(text: str, number: re.Match[str]) -> tuple[Quantity, bool] | None:
    """Read the quantity whose digits are ``number``; also say whether ``from`` precedes it.

    Returns ``None`` when the digits are part of an identifier or an ordinal (``4vCPU``, ``99th``).
    """
    literal = number.group().replace(",", "")
    start = number.start()
    index = number.end()
    money = start > 0 and text[start - 1] == "$"
    if money:
        start -= 1

    scale = Fraction(1)
    plus = False
    unit: _Unit | None = None
    if size := _SIZE.match(text, index):
        label, factor = _SIZE_UNITS[size["unit"].lower()]
        unit = _Unit("size", label, factor, size.end())
        index = size.end()
    else:
        if words := _WORD_MAGNITUDE.match(text, index):
            scale = Fraction(_MAGNITUDE_WORDS[words["word"]])
            index = words.end()
        elif letter := _LETTER_MAGNITUDE.match(text, index):
            scale = Fraction(_MAGNITUDE_LETTERS[letter["letter"]])
            index = letter.end()
        if text.startswith("+", index):
            plus = True
            index += 1
        if not money and (unit := _read_unit(text, index)):
            index = unit.end
    if text.startswith("+", index):
        plus = True
        index += 1
    if (
        unit is None
        and not money
        and (text[index : index + 1].isalpha() or _ORDINAL.match(text, index))
    ):
        return None

    family: Family
    if money:
        family, label, factor = "money", "$", Fraction(1)
    elif unit:
        family, label, factor = unit.family, unit.label, unit.factor
    else:
        family, label, factor = "count", "", Fraction(1)
    fine, coarse = _precision(literal, scale * factor)
    prefix = _baseline_prefix(text, start)
    quantity = Quantity(
        value=Fraction(literal) * scale * factor,
        family=family,
        unit=label,
        text=text[start:index],
        start=start,
        end=index,
        plus=plus,
        qualifier=_qualifier(text, start),
        baseline=prefix is not None and prefix != "from",
        precision_fine=fine,
        precision_coarse=coarse,
        factor=factor,
    )
    return quantity, prefix == "from"


def _scan(text: str) -> list[tuple[Quantity, bool]]:
    """Every quantity in ``text``, each with whether the word ``from`` stands right before it."""
    return [read for number in _NUMBER.finditer(text) if (read := _read_quantity(text, number))]


def _link_pairs(text: str, scanned: list[tuple[Quantity, bool]]) -> list[Quantity]:
    """Resolve ranges and arrow pairs: share a missing unit and mark the starting point."""
    result = [quantity for quantity, _ in scanned]
    for index in range(len(scanned) - 1):
        first, after_from = scanned[index]
        second = result[index + 1]
        link = _LINK.fullmatch(text[first.end : second.start])
        if not link:
            continue
        is_arrow = link["link"] in _ARROWS
        if is_arrow or (after_from and link["link"].lower() == "to"):
            first = replace(first, baseline=True)
        if first.family == "count" and not first.plus and second.family != "count":
            first = replace(
                first,
                value=first.value * second.factor,
                family=second.family,
                unit=second.unit,
                precision_fine=first.precision_fine * second.factor,
                precision_coarse=first.precision_coarse * second.factor,
                factor=second.factor,
            )
        result[index] = first
    return result


def extract_quantities(text: str) -> list[Quantity]:
    """Return the quantities in ``text``, in reading order.

    Args:
        text: Plain text; LaTeX escapes must already be resolved (see
            :func:`tailor.bank.latex.normalize_latex`).

    Returns:
        One :class:`Quantity` per number, with ranges and arrow pairs as two quantities.
    """
    return _link_pairs(text, _scan(text))
