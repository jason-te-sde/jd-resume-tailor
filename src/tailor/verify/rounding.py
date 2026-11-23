"""Decide whether a number in a bullet is the number in its source, or a flattering version of it.

A resume bullet may restate a number: ``3,781`` as ``3.7K+``, ``0.875`` as ``87%``. It may not
improve it: ``0.875`` as ``88%`` is a bigger number than anyone measured. The difference is the
direction of the rounding, and which direction flatters depends on what the number measures.

**The rule (default policy, ``conservative``).** A number that differs from its source must be that
source rounded to a coarser place, and it must have moved in the *unflattering* direction:

* *higher is better* (throughputs, shares, accuracy, counts of things built): round down, so
  ``0.875`` may be written ``87%`` or ``0.87`` but not ``88%`` or ``0.88``, and ``3,781`` may be
  ``3.7K`` but not ``3.8K``;
* *lower is better* (latency, duration, size, error rate, deviation): round **up**, so ``29.5 s``
  may be written ``30 s`` but not ``29 s``, and ``138 ms`` may be ``140 ms`` but not ``130 ms``.
  Rounding down is what flatters a latency, so it is the direction that is refused.

The starting point of a comparison (``from 412 ms to 138 ms``) flatters the other way: a bigger
"before" makes the improvement look bigger, so for a baseline the directions swap.

**Plus.** ``3.7K+`` claims "at least 3,700". It is honest only when the source is strictly above
it (and not more than ten times above, which would make it a different number): ``3,781`` may be
``3.7K+``; ``3.8K+`` and ``3,781+`` claim more than the source says, and so does ``8K+`` where the
source says exactly ``8,000``. ``at least`` works the same way. These rules do not relax.

**Up to.** A source that says ``up to 5 hours`` is a maximum; stating it as ``5 hours`` drops the
qualifier and is drift.

``--rounding nearest`` relaxes the direction rule to ordinary rounding: the claimed number may be
the nearest rounding of the source, in either direction, as long as it keeps at least two
significant figures (``88%`` for ``0.875``, ``3.1x`` for ``3.08x``, but not ``4K`` for ``3,781``).

**Polarity** is guessed from the unit family (durations and sizes: lower is better; everything else:
higher) and the words near the number. The guess is a heuristic, which is why the conservative
policy is the default. One case is decided by the shape, not by the metric: a percentage that is the
size of a reduction (``reduced p99 latency by 41%``, ``a 41% reduction``, ``41% fewer errors``).
Latency is lower-is-better, but "41%" here is how much was saved, so higher flatters: 40.6% may be
written 40%, not 41%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Final, Literal

from tailor.verify.quantities import Quantity

Polarity = Literal["higher", "lower"]
Rounding = Literal["conservative", "nearest"]
Verdict = Literal["exact", "rounded", "drift", "mismatch"]

_LOWER_IS_BETTER_FAMILIES: Final = frozenset({"time", "size"})
_LOWER_CUES: Final = re.compile(
    r"\b(?:latenc\w*|p\d{2}(?:\.\d+)?|response|duration|downtime|staleness|stale|delay\w*|lag|"
    r"errors?|failures?|loss|losses|mape|wmape|deviation|overhead|memory|cpu|costs?|pages|"
    r"incidents|violations|bugs|defects|regressions|footprint|bundle|startup|runtime|"
    r"within|under|below|fewer|cuts?|cutting|reduc\w*|lower\w*|decreas\w*|drop\w*|fell|fall\w*|"
    r"shrank|shrink\w*|slash\w*|trim\w*)\b",
    re.IGNORECASE,
)
_HIGHER_CUES: Final = re.compile(
    r"\b(?:throughput|accuracy|recall|precision|f1|coverage|uptime|rose|rise|rises|raised|"
    r"increas\w*|improv\w*|boost\w*|grew|grow\w*|gain\w*|faster|speed\w*|completion|"
    r"conversion|score\w*|reach\w*)\b",
    re.IGNORECASE,
)
# A percentage that says how big an improvement is: "reduced latency by 41%", "cut 41%", "a 41%
# reduction", "41% fewer", "41% lower latency". Here the figure is the size of a gain, so a higher
# one flatters, whatever the metric it is a reduction of.
_REDUCTION_VERBS: Final = (
    r"(?:reduc\w*|cut\w*|lower\w*|decreas\w*|shr[ia]nk\w*|slash\w*|trim\w*|shav\w*|dropp?\w*|"
    r"fell|fall\w*|declin\w*|sav\w*|eliminat\w*)"
)
_REDUCTION_BEFORE: Final = re.compile(
    rf"\b{_REDUCTION_VERBS}\b(?:[^.;:,]{{0,90}}?\bby)?\s+(?:about |around |roughly |nearly |almost "
    r"|approximately |up to |~)?$",
    re.IGNORECASE,
)
_REDUCTION_AFTER: Final = re.compile(
    r"\s+(?:fewer|less|lower|smaller|shorter|cheaper|leaner|lighter|reduction|decrease|drop|"
    r"cut|saving|savings|decline)\b",
    re.IGNORECASE,
)
_CLAUSE_END: Final = re.compile(r"[.;:](?=\s|$)")
_AFTER_END: Final = re.compile(r"[.;:,](?=\s|$)")
_CUE_WINDOW_BEFORE: Final = 60
_CUE_WINDOW_AFTER: Final = 30
_REDUCTION_REACH: Final = 110
_PLUS_REACH: Final = 10
_MIN_SIGNIFICANT_FIGURES: Final = 2


@dataclass(frozen=True, slots=True)
class Comparison:
    """How a claimed quantity relates to one source quantity.

    Attributes:
        verdict: ``exact`` (the same quantity, however written), ``rounded`` (a permitted
            rounding), ``drift`` (a rounding or a qualifier change that flatters), or
            ``mismatch`` (a different quantity).
        reason: A sentence saying why, for the report.
    """

    verdict: Verdict
    reason: str


def polarity(text: str, quantity: Quantity) -> Polarity:
    """Guess whether a higher or a lower value of ``quantity`` is the better one.

    Durations and sizes default to lower, everything else to higher. A cue word in the same clause
    overrides the default: the cue nearest to the number wins. The exception is a percentage that
    is the size of a reduction (``reduced latency by 41%``, ``a 41% reduction``, ``41% fewer``):
    whatever it is a reduction of, a bigger one is the better one, so it is ``higher``. Heuristic.
    """
    before = text[max(0, quantity.start - _CUE_WINDOW_BEFORE) : quantity.start]
    after = text[quantity.end : quantity.end + _CUE_WINDOW_AFTER]
    if ends := list(_CLAUSE_END.finditer(before)):
        before = before[ends[-1].end() :]
    if end := _AFTER_END.search(after):
        after = after[: end.start()]
    if quantity.family == "percent" and not quantity.baseline:
        tail = text[quantity.end : quantity.end + _CUE_WINDOW_AFTER]
        lead = text[max(0, quantity.start - _REDUCTION_REACH) : quantity.start]
        if ends := list(_CLAUSE_END.finditer(lead)):
            lead = lead[ends[-1].end() :]
        if _REDUCTION_BEFORE.search(lead) or _REDUCTION_AFTER.match(tail):
            return "higher"
    nearest: tuple[int, Polarity] | None = None
    kinds: tuple[tuple[re.Pattern[str], Polarity], ...] = (
        (_LOWER_CUES, "lower"),
        (_HIGHER_CUES, "higher"),
    )
    for cues, kind in kinds:
        for match in cues.finditer(before):
            distance = len(before) - match.end()
            if nearest is None or distance < nearest[0]:
                nearest = (distance, kind)
        for match in cues.finditer(after):
            if nearest is None or match.start() < nearest[0]:
                nearest = (match.start(), kind)
    if nearest:
        return nearest[1]
    return "lower" if quantity.family in _LOWER_IS_BETTER_FAMILIES else "higher"


def _as_percent(quantity: Quantity) -> Quantity | None:
    """A decimal fraction such as ``0.875`` read as the percentage it stands for."""
    if quantity.family == "count" and 0 <= quantity.value <= 1 and quantity.precision_fine < 1:
        return replace(
            quantity,
            family="percent",
            unit="%",
            value=quantity.value * 100,
            precision_fine=quantity.precision_fine * 100,
            precision_coarse=quantity.precision_coarse * 100,
        )
    return None


def _align(claimed: Quantity, source: Quantity) -> tuple[Quantity, Quantity] | None:
    """Put both quantities on one scale, or return ``None`` if they are of different kinds."""
    if claimed.family == source.family:
        return claimed, source
    if claimed.family == "percent" and (as_percent := _as_percent(source)):
        return claimed, as_percent
    if source.family == "percent" and (as_percent := _as_percent(claimed)):
        return as_percent, source
    return None


def _significant_figures(quantity: Quantity) -> int:
    steps = abs(quantity.value) / quantity.precision_coarse
    return len(str(int(steps))) if steps.denominator == 1 else _MIN_SIGNIFICANT_FIGURES + 1


def _flattering(claimed: Quantity, source: Quantity, pol: Polarity) -> bool:
    effective: Polarity = pol
    if claimed.baseline:
        effective = "lower" if pol == "higher" else "higher"
    if effective == "higher":
        return claimed.value > source.value
    return claimed.value < source.value


def _is_rounding(claimed: Quantity, source: Quantity) -> bool:
    """Whether ``claimed`` is ``source`` cut to a coarser place, one unit of that place at most."""
    place = claimed.precision_coarse
    return place > source.precision_fine and abs(claimed.value - source.value) < place


def _compare_lower_bound(claimed: Quantity, source: Quantity) -> Comparison:
    label = claimed.qualifier if claimed.qualifier and not claimed.plus else "+"
    if claimed.value < source.value:
        if source.value < _PLUS_REACH * claimed.value:
            return Comparison(
                "rounded", f"{claimed.text!r} is a lower bound below the source's {source.text!r}"
            )
        return Comparison("mismatch", f"{claimed.text!r} is far below the source's {source.text!r}")
    if _is_rounding(claimed, source):
        return Comparison(
            "drift",
            f"{claimed.text!r} claims more than the source's {source.text!r}: a lower bound "
            f"({label}) may not reach the source's value",
        )
    return Comparison("mismatch", f"{claimed.text!r} is above the source's {source.text!r}")


def compare(
    claimed: Quantity,
    source: Quantity,
    pol: Polarity,
    *,
    rounding: Rounding = "conservative",
) -> Comparison:
    """Compare a quantity of a bullet with a quantity of its source.

    Args:
        claimed: A quantity read from the bullet.
        source: A quantity read from a cited fact.
        pol: Whether a higher or a lower ``claimed`` value is the better one (see
            :func:`polarity`).
        rounding: ``conservative`` (default) or ``nearest``; see the module docstring.

    Returns:
        The verdict and the reason for it.
    """
    pair = _align(claimed, source)
    if pair is None:
        return Comparison(
            "mismatch", f"{claimed.text!r} is a {claimed.family}, not a {source.family}"
        )
    claimed, source = pair

    if claimed.value == source.value:
        if claimed.is_lower_bound and not source.is_lower_bound:
            return Comparison(
                "drift",
                f"{claimed.text!r} says more than the source's exact {source.text!r}: a lower "
                "bound may not reach the source's value",
            )
        if source.is_maximum and not claimed.is_maximum:
            return Comparison(
                "drift",
                f"the source says {source.qualifier!r} {source.text!r}; the bullet states it "
                "without the qualifier",
            )
        return Comparison("exact", f"{claimed.text!r} equals the source's {source.text!r}")

    if claimed.is_lower_bound:
        return _compare_lower_bound(claimed, source)

    if not _is_rounding(claimed, source):
        return Comparison(
            "mismatch", f"{claimed.text!r} is not {source.text!r} or a rounding of it"
        )

    direction = "up" if claimed.value > source.value else "down"
    flattering = _flattering(claimed, source, pol)
    better = "higher is better" if pol == "higher" else "lower is better"
    if claimed.baseline:
        better += ", for the starting point of a comparison the reverse"
    detail = f"{claimed.text!r} is {source.text!r} rounded {direction}; where {better}"
    if not flattering:
        return Comparison("rounded", f"{detail}, that is the unflattering direction")
    if (
        rounding == "nearest"
        and abs(claimed.value - source.value) <= claimed.precision_coarse / 2
        and _significant_figures(claimed) >= _MIN_SIGNIFICANT_FIGURES
    ):
        return Comparison("rounded", f"{detail}, but it is the nearest rounding at two figures")
    return Comparison("drift", f"{detail}, that is the flattering direction")


__all__ = ["Comparison", "Polarity", "Rounding", "Verdict", "compare", "polarity"]
