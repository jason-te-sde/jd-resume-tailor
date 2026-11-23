"""Verify bullets against a fact bank: run every check and build the report.

``verify(bank, bullets)`` is the entry point. Each bullet names the entry it is about and,
optionally, the ids of the facts it cites; without them every fact of the entry is a source. The
checks run on the text of the cited facts, with the entry's technology list, title and company as
extra evidence for names and its DO NOT CLAIM items as limits.

One rule combines the checks. A causal link whose span overlaps a fabricated name or number is not
reported separately: ``by moving to Kubernetes`` is one problem, the unbacked name, and a second
finding for the same words would only make the report longer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from tailor.bank import Bank, Entry, Fact, WorkEntry
from tailor.verify.checks import (
    check_do_not_claim,
    check_entities,
    check_escalation_absolute,
    check_escalation_causal,
    check_numbers,
    check_unsupported_claims,
)
from tailor.verify.findings import Finding
from tailor.verify.report import BulletResult, Report
from tailor.verify.rounding import Rounding

_SPAN_SUPERSEDES_CAUSAL: Final = frozenset(
    {"fabricated_number", "number_drift", "fabricated_entity"}
)
_BULLET_FIELDS: Final = frozenset({"text", "entry", "sources", "id"})


class VerifyError(Exception):
    """The input of a verification is wrong (an unknown entry or fact, a malformed file)."""


@dataclass(frozen=True, slots=True)
class Bullet:
    """A bullet to verify.

    Attributes:
        text: The bullet as it would appear on the resume, plain text.
        entry: The id of the bank entry it is about.
        sources: The ids of the facts it cites; ``None`` means every fact of the entry.
        id: An optional label that the report repeats, so a caller can match results to bullets.
    """

    text: str
    entry: str
    sources: tuple[str, ...] | None = field(default=None)
    id: str | None = None


def verify_text(
    text: str,
    sources: Sequence[str],
    *,
    claims: Sequence[str] = (),
    context: Sequence[str] = (),
    rounding: Rounding = "conservative",
    scope: Sequence[str] = (),
) -> list[Finding]:
    """Run every check on one bullet and return the findings in the order of the text.

    Args:
        text: The bullet.
        sources: The text of the facts it cites that state what was achieved.
        claims: The entry's DO NOT CLAIM items.
        context: Further evidence for names: the entry's tech stack, title and company.
        rounding: The rounding policy of the number check.
        scope: The text of the scope facts it cites. A scope records provenance and test
            conditions, not achievements, so its numbers cannot back a number of the bullet. It
            still backs names, absolutes, causes and the claims of ``claims``.
    """
    background = [*sources, *scope]
    limits = check_do_not_claim(text, claims, background)
    unsupported = [
        f
        for f in check_unsupported_claims(text, background, claims)
        if not any(f.start < other.end and other.start < f.end for other in limits)
    ]
    findings = [
        *check_numbers(text, sources, rounding=rounding),
        *check_entities(text, background, context=context),
        *limits,
        *unsupported,
        *check_escalation_absolute(text, background),
        *check_escalation_causal(text, background),
    ]
    explained = [f for f in findings if f.kind in _SPAN_SUPERSEDES_CAUSAL]
    kept = [
        f
        for f in findings
        if f.kind != "escalation_causal"
        or not any(f.start < other.end and other.start < f.end for other in explained)
    ]
    return sorted(kept, key=lambda finding: (finding.start, finding.end, finding.kind))


def entry_context(entry: Entry) -> list[str]:
    """The text that backs the names in a bullet about ``entry``: tech list, title, company."""
    if isinstance(entry, WorkEntry):
        return [entry.tech, entry.title, entry.company]
    return [entry.tech_stack, entry.title]


def _source_text(fact: Fact) -> str:
    """The text a bullet may draw on: the fact, and the numbers recorded with it as measured."""
    if not fact.numbers:
        return fact.text
    recorded = "; ".join(f"{n.as_measured or n.value} {n.unit}".strip() for n in fact.numbers)
    return f"{fact.text}; {recorded}"


def _cited(bank: Bank, number: int, bullet: Bullet) -> list[Fact]:
    try:
        facts = bank.facts_for(bullet.entry)
    except KeyError:
        raise VerifyError(f"bullet {number}: unknown entry {bullet.entry!r}") from None
    if bullet.sources is None:
        return facts
    by_id = {fact.id: fact for fact in facts}
    cited: list[Fact] = []
    for fact_id in bullet.sources:
        if fact_id in by_id:
            cited.append(by_id[fact_id])
            continue
        try:
            owner = bank.fact(fact_id).entry_id
        except KeyError:
            raise VerifyError(f"bullet {number}: unknown fact id {fact_id!r}") from None
        raise VerifyError(f"bullet {number}: fact {fact_id} belongs to {owner}, not {bullet.entry}")
    return cited


def verify(
    bank: Bank,
    bullets: Sequence[Bullet],
    *,
    rounding: Rounding = "conservative",
) -> Report:
    """Verify every bullet against the facts it cites.

    Args:
        bank: The fact bank.
        bullets: The bullets to verify, in order.
        rounding: ``conservative`` (default) or ``nearest``; see :mod:`tailor.verify.rounding`.

    Returns:
        A :class:`Report` with one result per bullet, in the same order.

    Raises:
        VerifyError: A bullet names an entry or a fact that the bank does not have, or cites a
            fact of another entry.
    """
    results: list[BulletResult] = []
    for index, bullet in enumerate(bullets):
        cited = _cited(bank, index + 1, bullet)
        findings = verify_text(
            bullet.text,
            [_source_text(fact) for fact in cited if fact.kind != "scope"],
            claims=bank.do_not_claim(bullet.entry),
            context=entry_context(bank.entry(bullet.entry)),
            rounding=rounding,
            scope=[_source_text(fact) for fact in cited if fact.kind == "scope"],
        )
        results.append(
            BulletResult(
                index=index,
                id=bullet.id,
                entry=bullet.entry,
                text=bullet.text,
                sources=tuple(fact.id for fact in cited),
                findings=tuple(findings),
            )
        )
    return Report(rounding=rounding, results=tuple(results))


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise VerifyError(f"{where}: must be a string")
    return value


def _bullet(raw: Any, where: str) -> Bullet:
    if not isinstance(raw, dict):
        raise VerifyError(f"{where}: must be an object")
    for key in raw:
        if key not in _BULLET_FIELDS:
            raise VerifyError(f'{where}: unknown field "{key}"')
    for key in ("text", "entry"):
        if key not in raw:
            raise VerifyError(f'{where}: missing "{key}"')
    sources: tuple[str, ...] | None = None
    if "sources" in raw:
        listed = raw["sources"]
        if not isinstance(listed, list):
            raise VerifyError(f"{where}.sources: must be an array of fact ids")
        sources = tuple(_string(item, f"{where}.sources[{i}]") for i, item in enumerate(listed))
    label = _string(raw["id"], f"{where}.id") if "id" in raw else None
    return Bullet(
        _string(raw["text"], f"{where}.text"),
        _string(raw["entry"], f"{where}.entry"),
        sources,
        label,
    )


def load_bullets(path: Path) -> list[Bullet]:
    """Read bullets from a JSON file: an array of objects, or an object with a ``bullets`` array.

    Each bullet is ``{"text": ..., "entry": ..., "sources": [fact ids]?, "id": ...?}``. Unknown
    fields are errors, so a misspelled ``sources`` cannot silently widen what a bullet may cite.

    Raises:
        VerifyError: The file cannot be read, is not JSON, or is not shaped like that.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerifyError(f"cannot read {path}: {error.strerror}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise VerifyError(f"{path} is not valid JSON: {error}") from error
    if isinstance(document, dict) and isinstance(document.get("bullets"), list):
        document = document["bullets"]
    if not isinstance(document, list):
        raise VerifyError(
            f'{path}: expected an array of bullets or an object with a "bullets" array'
        )
    return [_bullet(item, f"$[{index}]") for index, item in enumerate(document)]
