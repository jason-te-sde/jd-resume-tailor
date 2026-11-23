"""What a resume bullet may cite: facts, and the numbers recorded with them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tailor.bank.latex import normalize_latex
from tailor.bank.scope import strip_do_not_claim

FactKind = Literal["bullet", "scope", "explicit"]


@dataclass(frozen=True, slots=True)
class Number:
    """A quantity recorded with a fact.

    Attributes:
        value: The numeric value, as a JSON number.
        unit: What it counts or measures (``ms``, ``%``, ``x``, ``req/s``); empty for a bare count.
        as_measured: The number as it was written when measured (``0.875``, ``3,781``), which keeps
            the precision and formatting that ``value`` loses. ``None`` when not recorded.
    """

    value: int | float
    unit: str
    as_measured: str | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    """One thing a bullet may cite, identified so a rewritten bullet can name its sources.

    Attributes:
        id: Unique across the bank. ``<entry id>#<n>`` for the n-th bullet (counting from 1) of an
            entry without facts and ``<entry id>#scope`` for its scope text; otherwise the id the
            bank author chose.
        entry_id: The entry the fact belongs to.
        text: Plain text, with LaTeX escapes resolved.
        kind: Where the fact comes from: a ``bullet``, the ``scope`` text, or an ``explicit`` fact.
        numbers: The quantities recorded with it. Empty for derived facts, whose numbers are in
            their text.
    """

    id: str
    entry_id: str
    text: str
    kind: FactKind
    numbers: tuple[Number, ...] = ()


def bullet_fact_id(entry_id: str, number: int) -> str:
    """The id of the ``number``-th bullet (counting from 1) of an entry without facts."""
    return f"{entry_id}#{number}"


def scope_fact_id(entry_id: str) -> str:
    """The id of the scope fact of an entry without facts."""
    return f"{entry_id}#scope"


def scope_provenance(scope: str) -> str:
    """The scope as plain text, without its DO NOT CLAIM parts; empty if nothing is left.

    What an entry must not claim cannot also be evidence for a claim, so those parts are not part
    of the text a bullet may cite. They stay available as the entry's list of limits.
    """
    return strip_do_not_claim(normalize_latex(scope))


def derive_facts(entry_id: str, bullets: Sequence[str], scope: str) -> list[Fact]:
    """The facts of an entry that gives none: one per bullet, then its scope text if any."""
    facts = [
        Fact(
            id=bullet_fact_id(entry_id, number),
            entry_id=entry_id,
            text=normalize_latex(bullet),
            kind="bullet",
        )
        for number, bullet in enumerate(bullets, start=1)
    ]
    provenance = scope_provenance(scope)
    if provenance:
        facts.append(
            Fact(id=scope_fact_id(entry_id), entry_id=entry_id, text=provenance, kind="scope")
        )
    return facts
