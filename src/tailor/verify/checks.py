"""The checks: one function per kind of claim a bullet must not make beyond its sources.

Each check takes the text of a bullet and the text of the facts it cites, and returns a list of
:class:`~tailor.verify.findings.Finding`, each with the offending span of the bullet and the
evidence. An empty list means the check found nothing to object to. The checks do not know about
banks or entries: :func:`tailor.verify.verifier.verify` supplies the facts, the entry's technology
list and its DO NOT CLAIM list.

The number and entity checks are deterministic rules over extracted quantities and names. The
do-not-claim check is a phrase match. The escalation checks are **heuristics**: they read English
with word lists and stems, and miss what they have no word for; their precision and recall are
measured separately in the verifier evaluation (see ``eval/verifier/README.md``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from tailor.verify.absolutes import unsupported_absolutes, unsupported_claims
from tailor.verify.binding import claim_stems, is_bound
from tailor.verify.causal import find_links, is_linked
from tailor.verify.claims import find_phrase, key_phrases
from tailor.verify.entities import KnownNames, extract_entities
from tailor.verify.findings import Finding, FindingKind
from tailor.verify.quantities import Quantity, extract_quantities
from tailor.verify.rounding import Comparison, Rounding, Verdict, compare, polarity
from tailor.verify.text import tokenize

_VERDICT_RANK: Final[dict[Verdict, int]] = {"exact": 0, "rounded": 1, "drift": 2, "mismatch": 3}
_EVIDENCE_VALUES: Final = 6


def _closest(claimed: Quantity, candidates: list[Quantity]) -> list[Quantity]:
    """The candidates of the claimed quantity's family, nearest in value first."""
    same = [c for c in candidates if c.family == claimed.family]
    return sorted(same, key=lambda c: abs(c.value - claimed.value))


def _number_evidence(claimed: Quantity, source_quantities: list[Quantity]) -> str:
    same = _closest(claimed, source_quantities)
    if same:
        shown = ", ".join(dict.fromkeys(q.text for q in same[:_EVIDENCE_VALUES]))
        text = (
            f"no {claimed.family} in the sources equals {claimed.text!r}; the sources have {shown}"
        )
    else:
        text = f"the sources have no {claimed.family} quantity, so {claimed.text!r} has no source"
    elsewhere = [
        q for q in source_quantities if q.family != claimed.family and q.value == claimed.value
    ]
    if elsewhere:
        text += f" (they have {elsewhere[0].text!r}, which is a {elsewhere[0].family})"
    return text


def check_numbers(
    bullet: str,
    sources: Sequence[str],
    *,
    rounding: Rounding = "conservative",
) -> list[Finding]:
    """Check every quantity of ``bullet`` against the quantities of ``sources``.

    A quantity passes when some quantity of the same unit family in the sources is the same number
    (however written), a permitted rounding of it, or the same share written as a fraction or a
    percentage; see :mod:`tailor.verify.rounding` for the rounding rules. The source must also be
    about the same thing: it has to share a word with the words around the number in the bullet
    (see :mod:`tailor.verify.binding`), so a number cannot be borrowed from an unrelated fact. One
    that has no such
    source is a ``fabricated_number``. One that matches a source only by rounding the flattering
    way, by a ``+`` that reaches the source's value, or by dropping the source's ``up to``, is
    ``number_drift``.

    Args:
        bullet: The text of the bullet.
        sources: The text of the facts the bullet cites.
        rounding: ``conservative`` (the default) or ``nearest``.

    Returns:
        One finding per offending quantity, in the order they appear in the bullet.
    """
    per_fact = [(text, extract_quantities(text)) for text in sources]
    source_quantities = [q for _, quantities in per_fact for q in quantities]
    findings: list[Finding] = []
    for claimed in extract_quantities(bullet):
        pol = polarity(bullet, claimed)
        words = claim_stems(bullet, claimed)
        best: tuple[Quantity, Comparison] | None = None
        misplaced: Quantity | None = None
        for fact, quantities in per_fact:
            bound = is_bound(words, fact)
            for candidate in quantities:
                result = compare(claimed, candidate, pol, rounding=rounding)
                if not bound:
                    if result.verdict in {"exact", "rounded"} and misplaced is None:
                        misplaced = candidate
                    continue
                if best is None or _VERDICT_RANK[result.verdict] < _VERDICT_RANK[best[1].verdict]:
                    best = (candidate, result)
        if best is not None and best[1].verdict in {"exact", "rounded"}:
            continue
        if best is not None and best[1].verdict == "drift":
            findings.append(
                Finding(
                    "number_drift",
                    claimed.start,
                    claimed.end,
                    claimed.text,
                    f"{best[1].reason} (source: {best[0].text!r})",
                )
            )
            continue
        if misplaced is not None:
            evidence = (
                f"{claimed.text!r} appears in the sources, but in a fact that shares none of the "
                f"words around it in the bullet ({', '.join(sorted(words))}): it supports a "
                "different claim"
            )
        else:
            evidence = _number_evidence(claimed, source_quantities)
        findings.append(
            Finding("fabricated_number", claimed.start, claimed.end, claimed.text, evidence)
        )
    return findings


_PRODUCTION_LIMIT: Final = re.compile(
    r"production|deploy|real (?:users|support team)", re.IGNORECASE
)
_ENTITY_LABEL: Final = {
    "technology": "technology",
    "identifier": "name",
    "name": "name",
}


def check_entities(
    bullet: str,
    sources: Sequence[str],
    *,
    context: Sequence[str] = (),
) -> list[Finding]:
    """Check that every technology and proper noun of ``bullet`` is backed by its sources.

    Names are found by :func:`tailor.verify.entities.extract_entities`: the curated technology
    vocabulary, CamelCase and ALLCAPS tokens, capitalised multi-word names and capitalised words
    inside a sentence. A name is backed when it (or a vocabulary alias of it) appears in the
    sources or in ``context``, which holds the entry's technology list and its own names (title,
    company). One that is not is a ``fabricated_entity``.

    Args:
        bullet: The text of the bullet.
        sources: The text of the facts the bullet cites.
        context: Further text that backs names: the entry's tech stack, title and company.

    Returns:
        One finding per unbacked name, in the order they appear in the bullet.
    """
    known = KnownNames.of([*sources, *context])
    return [
        Finding(
            "fabricated_entity",
            entity.start,
            entity.end,
            entity.text,
            f"the {_ENTITY_LABEL[entity.kind]} {entity.text!r} appears in none of the cited facts "
            "and not in the entry's tech stack",
        )
        for entity in extract_entities(bullet)
        if not known.supports(entity)
    ]


def check_do_not_claim(
    bullet: str,
    claims: Sequence[str],
    sources: Sequence[str],
) -> list[Finding]:
    """Check ``bullet`` against what its entry must not claim.

    Each DO NOT CLAIM item is reduced to key phrases (see :mod:`tailor.verify.claims`): the quoted
    term of the item, or the item itself without its explanation in parentheses. A bullet that has
    the stems of a phrase close together, with no negation just before, is a ``do_not_claim``
    finding, unless the sources contain the phrase themselves (then the fact says it, not the
    rewrite).

    Args:
        bullet: The text of the bullet.
        claims: The entry's DO NOT CLAIM items, one string each (``Bank.do_not_claim``).
        sources: The text of the facts the bullet cites.

    Returns:
        One finding per phrase found in the bullet, in the order of the claims.
    """
    bullet_tokens = tokenize(bullet)
    source_tokens = [tokenize(text) for text in sources]
    findings: list[Finding] = []
    for claim in claims:
        for phrase in key_phrases(claim):
            hits = [m for m in find_phrase(bullet_tokens, phrase) if not m.negated]
            if not hits:
                continue
            if any(not m.negated for tokens in source_tokens for m in find_phrase(tokens, phrase)):
                continue
            hit = hits[0]
            findings.append(
                Finding(
                    "do_not_claim",
                    hit.start,
                    hit.end,
                    bullet[hit.start : hit.end],
                    f"the entry lists {phrase.text!r} under DO NOT CLAIM: {claim!r}",
                )
            )
    return findings


def check_escalation_absolute(bullet: str, sources: Sequence[str]) -> list[Finding]:
    """Flag absolute words (``guaranteed``, ``never``, ``exactly-once``, ``100%``, ``every``...).

    An absolute in the bullet that none of the cited sources contains claims more than the facts
    say. ``zero`` is fine where a source says ``0`` or ``no`` of the same thing, ``all`` and
    ``every`` where a source quantifies the same thing; see :mod:`tailor.verify.absolutes`.

    This check is a **heuristic**: a word list, not an understanding of the sentence.

    Args:
        bullet: The text of the bullet.
        sources: The text of the facts the bullet cites.

    Returns:
        One finding per unsupported absolute, in order.
    """
    return [
        Finding(
            "escalation_absolute",
            mention.start,
            mention.end,
            mention.text,
            f"{mention.text!r} is an absolute; none of the cited sources says it",
        )
        for mention in unsupported_absolutes(bullet, sources)
    ]


def check_escalation_causal(bullet: str, sources: Sequence[str]) -> list[Finding]:
    """Flag a cause or purpose that the sources do not link to the result.

    For each causal connector of the bullet (``by X``, ``with X``, ``so that X``, ``thanks to X``,
    ``, enabling X``...) the mechanism and the result must both be held by one sentence of a
    source; otherwise it is an ``escalation_causal`` finding. See :mod:`tailor.verify.causal`.

    This check is a **heuristic** over word overlap: it misses a link phrased with an unknown
    connector and flags a paraphrase that shares too few words with the source.

    Args:
        bullet: The text of the bullet.
        sources: The text of the facts the bullet cites.

    Returns:
        One finding per unsupported link, in order.
    """
    return [
        Finding(
            "escalation_causal",
            link.start,
            link.end,
            link.text,
            f"the cited sources do not link {link.text!r} to the result in one statement",
        )
        for link in find_links(bullet)
        if not is_linked(link, sources)
    ]


def check_unsupported_claims(
    bullet: str, sources: Sequence[str], claims: Sequence[str] = ()
) -> list[Finding]:
    """Flag a claim about how or by whom the work was done that no source makes.

    ``in production`` and ``led the team`` are claims of their own, not numbers or technologies,
    and a rewrite adds them readily. One that no cited source states (a source that says
    ``not production`` does not) is a ``do_not_claim`` finding when the entry's DO NOT CLAIM list
    rules out production, deployment or real users, and an ``escalation_absolute`` finding
    otherwise. See
    :mod:`tailor.verify.absolutes`.

    Args:
        bullet: The text of the bullet.
        sources: The text of the facts the bullet cites, scope text included.
        claims: The entry's DO NOT CLAIM items.

    Returns:
        One finding per unsupported claim, in order.
    """
    limited = any(_PRODUCTION_LIMIT.search(claim) for claim in claims)
    findings: list[Finding] = []
    for mention in unsupported_claims(bullet, sources):
        kind: FindingKind = (
            "do_not_claim"
            if limited and mention.text.lower() == "production"
            else "escalation_absolute"
        )
        findings.append(
            Finding(
                kind,
                mention.start,
                mention.end,
                mention.text,
                f"{mention.text!r} claims more than the cited sources say: none states it",
            )
        )
    return findings
