"""Rewrite the chosen bullets for the posting, and let the verifier decide what survives.

One model call per entry returns ``{bullets: [{text, sources: [fact ids]}]}``, bullet k being the
rewrite of the k-th chosen bullet. The prompt states the rules, the verifier enforces them: every
rewritten bullet is verified against the facts it cites. A bullet that cites nothing, or a fact the
entry does not have, fails, and so does one the verifier passes whose text carries a fact id copied
from the prompt. The bullets that fail get one retry in which the model sees what was wrong;
any that still fail are replaced by the bank's own bullet, with the reason recorded.
``adaptable: false`` entries are copied verbatim without a call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from tailor.bank import Bank, Entry, WorkEntry, normalize_latex
from tailor.bank.facts import Fact, bullet_fact_id
from tailor.jd.extract import JobDescription
from tailor.llm import LLM, Message
from tailor.verify import Bullet, Rounding, verify

Outcome = Literal["first_pass", "after_retry", "fallback", "verbatim"]
COPIED_CITATION: Final = "copied_citation"
"""The kind of the problem of a bullet whose text carries a fact id copied from the prompt."""
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "bullets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["bullets"],
    "additionalProperties": False,
}
SYSTEM: Final = """\
You tailor resume bullets to a job posting without changing what is true. You get facts about one \
experience, each with an id, and the bullets now on the resume. Rewrite each bullet for the \
posting and cite the facts it rests on.

Rules. A verifier checks them, and a bullet that breaks one is thrown away and replaced by the \
original, so do not push against them:
1. No new numbers. Every number must come from a fact you cite; do not add, combine or compute.
2. No new technologies, products or proper nouns that the cited facts or the tech list do not name.
3. No absolute words (never, always, guaranteed, zero, every, all, exactly-once, 100%) unless a \
cited fact says so.
4. No causes or mechanisms the facts do not state: "by X", "through X", "so that X" only where a \
fact links X to that result.
5. Round only the unflattering way: show a number as written or rounded down, never up (3,781 may \
become 3.7K+, never 3.8K+).
6. Use the posting's words only where they describe this work accurately.
Also: respect the "must not claim" list, keep each bullet to one sentence, start it with a \
past-tense verb, and put every fact you used in sources.
Return exactly as many bullets as you are asked for, in the order given."""


_LABEL: Final = re.compile(r"\b(?:sources?|facts?|cites?|cited)\b\s*:?\s*\[?\s*$", re.IGNORECASE)
_OPEN: Final = re.compile(r"[\[(][^\[\]()]*$")
_CLOSE: Final = re.compile(r"^[^\[\]()]*[\])]")


def copied_citations(text: str, fact_ids: Collection[str]) -> tuple[str, ...]:
    """The fact ids that ``text`` carries as a citation, in order of appearance.

    The prompt lists each fact as ``[id] text`` and asks for the ids in ``sources``; a model
    sometimes copies them into the bullet instead, as ``[id]``, ``(id)``, ``(sources: [id])``,
    ``(fact: id)`` or ``(Cites id)``. A fact id in the text is such a citation when it sits in
    square brackets or parentheses, when a label (``source``, ``fact``, ``cites``) comes right
    before it, or when it is an ``entry#n`` id, which is never prose. An explicit id that reads as
    a word (``f-latency``) is not a citation on its own.
    """
    found: dict[int, str] = {}
    for fact_id in fact_ids:
        pattern = re.compile(rf"(?<![\w#-]){re.escape(fact_id)}(?![\w#-])", re.IGNORECASE)
        for match in pattern.finditer(text):
            before, after = text[: match.start()], text[match.end() :]
            if (
                "#" in fact_id
                or (_OPEN.search(before) and _CLOSE.search(after))
                or _LABEL.search(before)
            ):
                found.setdefault(match.start(), fact_id)
    return tuple(dict.fromkeys(found[start] for start in sorted(found)))


@dataclass(frozen=True, slots=True)
class Attempt:
    """One model answer for one bullet and what was wrong with it (empty: nothing)."""

    text: str
    sources: tuple[str, ...]
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalBullet:
    """The bullet that goes on the resume, how it got there, and the evidence.

    Attributes:
        text: Plain text of the final bullet.
        sources: Fact ids it cites.
        outcome: ``first_pass``, ``after_retry``, ``fallback`` (the bank's bullet), or ``verbatim``.
        verdict: ``ok`` when the verifier passed it, ``bank_original`` for a bullet taken from the
            bank unchanged.
        original: The bank bullet this one stands for, as plain text.
        reason: Why the bank bullet was used; empty otherwise.
        attempts: What the model returned, first answer first.
    """

    text: str
    sources: tuple[str, ...]
    outcome: Outcome
    verdict: Literal["ok", "bank_original"]
    original: str
    reason: str = ""
    attempts: tuple[Attempt, ...] = ()


@dataclass(slots=True)
class EntryResult:
    """The rewritten bullets of one entry, and the calls it took."""

    entry: Entry
    bullets: list[FinalBullet] = field(default_factory=list)
    calls: int = 0


def _fact_line(fact: Fact) -> str:
    line = f"[{fact.id}] {fact.text}"
    if fact.numbers:
        recorded = "; ".join(f"{n.as_measured or n.value} {n.unit}".strip() for n in fact.numbers)
        line += f" (numbers: {recorded})"
    return line


def _posting(jd: JobDescription) -> str:
    def block(name: str, items: tuple[str, ...]) -> str:
        return f"{name}:\n" + "\n".join(f"- {i}" for i in items) if items else ""

    parts = [
        f"Posting: {jd.title} at {jd.company}",
        block("Responsibilities", jd.responsibilities),
        block("Requirements", jd.requirements),
        block("Keywords", jd.keywords),
    ]
    return "\n".join(p for p in parts if p)


def build_messages(
    bank: Bank, jd: JobDescription, entry: Entry, originals: list[str]
) -> list[Message]:
    """The rewriting prompt for one entry."""
    stack = entry.tech if isinstance(entry, WorkEntry) else entry.tech_stack
    where = f" at {entry.company}" if isinstance(entry, WorkEntry) else ""
    claims = bank.do_not_claim(entry.id)
    lines = [
        _posting(jd),
        "",
        f"Experience: {normalize_latex(entry.title)}{where}",
        f"Tech list: {normalize_latex(stack)}",
        "Facts you may cite:",
        *(_fact_line(f) for f in bank.facts_for(entry.id)),
        "Must not claim:",
        *([f"- {c}" for c in claims] or ["- (nothing listed)"]),
        "",
        f"Bullets to rewrite (return exactly {len(originals)}; bullet k is the rewrite of k):",
        *(f"{n}. {text}" for n, text in enumerate(originals, start=1)),
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def retry_messages(
    first: list[Message], answer: dict[str, Any], failing: list[tuple[int, str, tuple[str, ...]]]
) -> list[Message]:
    """The follow-up prompt: the first exchange, then what the verifier found.

    ``failing`` is ``(bullet number, its text, its problems)`` for each bullet to redo.
    """
    report = "\n".join(
        f"Bullet {number} was rejected. You wrote: {text!r}\n"
        + "\n".join(f"  - {problem}" for problem in problems)
        for number, text, problems in failing
    )
    follow_up = (
        f"The verifier rejected these bullets:\n{report}\n\n"
        f"Rewrite only these {len(failing)} bullets, in the order listed, so that none of the "
        "problems remains. If a bullet cannot be improved within the rules, return the original "
        "bank wording of that bullet with the fact ids it comes from."
    )
    return [
        *first,
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True)},
        {"role": "user", "content": follow_up},
    ]


def _bank_sources(bank: Bank, entry: Entry, index: int) -> tuple[str, ...]:
    if entry.facts is None:
        return (bullet_fact_id(entry.id, index + 1),)
    return tuple(f.id for f in bank.facts_for(entry.id))


def _answers(raw: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    items = raw.get("bullets", [])
    out: list[tuple[str, tuple[str, ...]]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            out.append(("", ()))
            continue
        sources = item.get("sources", [])
        out.append(
            (
                str(item.get("text", "")).strip(),
                tuple(s for s in sources if isinstance(s, str))
                if isinstance(sources, list)
                else (),
            )
        )
    return out


def check_bullets(
    bank: Bank,
    entry: Entry,
    answers: list[tuple[str, tuple[str, ...]]],
    *,
    rounding: Rounding,
) -> list[tuple[str, ...]]:
    """What is wrong with each answer, as sentences; an empty tuple means the bullet is ok.

    A bullet that cites nothing, or a fact the entry does not have, is not verified. One that is
    verified fails on the verifier's findings; only when there are none is its text checked for a
    fact id copied from the prompt (:func:`copied_citations`). That order keeps the problems of a
    bullet the verifier rejects, and so the retry prompt, what they were before the check existed.
    """
    own = {f.id for f in bank.facts_for(entry.id)}
    problems: list[tuple[str, ...]] = [() for _ in answers]
    to_verify: list[tuple[int, Bullet]] = []
    for position, (text, sources) in enumerate(answers):
        found: list[str] = []
        if not text:
            found.append("the bullet is empty or missing")
        if not sources:
            found.append("the bullet cites no fact")
        for fact_id in sources:
            if fact_id not in own:
                found.append(f"it cites {fact_id!r}, which is not a fact of this experience")
        if found:
            problems[position] = tuple(found)
        else:
            to_verify.append((position, Bullet(text, entry.id, sources)))
    if to_verify:
        report = verify(bank, [b for _, b in to_verify], rounding=rounding)
        for (position, bullet), result in zip(to_verify, report.results, strict=True):
            found = [f"{f.kind}: {f.text!r} - {f.evidence}" for f in result.findings]
            cited = () if found else copied_citations(bullet.text, own)
            if cited:
                found.append(
                    f"{COPIED_CITATION}: its text carries the fact ids {', '.join(cited)}, copied "
                    "from the prompt; fact ids go in sources, never in the bullet"
                )
            problems[position] = tuple(found)
    return problems


def rewrite_entry(
    llm: LLM,
    bank: Bank,
    jd: JobDescription,
    entry: Entry,
    chosen: tuple[int, ...],
    *,
    model: str,
    rounding: Rounding = "conservative",
) -> EntryResult:
    """Rewrite the ``chosen`` bullets (0-based indexes) of ``entry`` and verify the result."""
    result = EntryResult(entry)
    originals = [normalize_latex(entry.bullets[i]) for i in chosen]
    if not entry.adaptable:
        result.bullets = [
            FinalBullet(
                text,
                _bank_sources(bank, entry, i),
                "verbatim",
                "bank_original",
                text,
                "the entry is marked adaptable: false",
            )
            for i, text in zip(chosen, originals, strict=True)
        ]
        return result

    first_messages = build_messages(bank, jd, entry, originals)
    raw = llm.complete_json(first_messages, SCHEMA, model=model)
    result.calls = 1
    answers = _answers(raw)[: len(originals)]
    answers += [("", ())] * (len(originals) - len(answers))
    problems = check_bullets(bank, entry, answers, rounding=rounding)
    attempts: list[list[Attempt]] = [
        [Attempt(text, sources, found)]
        for (text, sources), found in zip(answers, problems, strict=True)
    ]
    final: list[tuple[str, tuple[str, ...]] | None] = [
        None if found else answer for answer, found in zip(answers, problems, strict=True)
    ]
    outcomes: list[Outcome] = ["first_pass"] * len(originals)

    failing = [k for k, ans in enumerate(final) if ans is None]
    if failing:
        follow_up = retry_messages(
            first_messages,
            raw,
            [(k + 1, answers[k][0], problems[k]) for k in failing],
        )
        retried = _answers(llm.complete_json(follow_up, SCHEMA, model=model))
        result.calls = 2
        retried += [("", ())] * (len(failing) - len(retried))
        retried = retried[: len(failing)]
        again = check_bullets(bank, entry, retried, rounding=rounding)
        for k, answer, found in zip(failing, retried, again, strict=True):
            attempts[k].append(Attempt(answer[0], answer[1], found))
            if not found:
                final[k] = answer
                outcomes[k] = "after_retry"
            else:
                outcomes[k] = "fallback"

    for k, (index, original) in enumerate(zip(chosen, originals, strict=True)):
        accepted = final[k]
        tried = tuple(attempts[k])
        if accepted is None:
            reason = "; ".join(tried[-1].problems)
            first = "; ".join(tried[0].problems)
            if len(tried) > 1 and first != reason:
                reason = f"first attempt: {first}. second attempt: {reason}"
            result.bullets.append(
                FinalBullet(
                    original,
                    _bank_sources(bank, entry, index),
                    "fallback",
                    "bank_original",
                    original,
                    reason,
                    tried,
                )
            )
        else:
            result.bullets.append(
                FinalBullet(accepted[0], accepted[1], outcomes[k], "ok", original, "", tried)
            )
    return result
