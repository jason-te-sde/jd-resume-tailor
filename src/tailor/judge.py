"""An LLM judge: does a bullet say more than the facts it cites support?

It answers the same six questions as the verifier, one per violation class, with a short reason,
through the same provider layer as the rest of the tool. It sees what a person labelling the bullet
would see (the bullet, the facts it cites, their scope notes, the entry's DO NOT CLAIM list and its
tech list) and never the verifier's findings, so its verdicts and the verifier's are independent
opinions that can be compared. It is a second reader to be measured, not an authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from tailor.bank import Bank, WorkEntry
from tailor.bank.facts import scope_provenance
from tailor.llm import LLM, LLMError, Message
from tailor.verify import KINDS

CLASSES: Final = KINDS
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        **{name: {"type": "boolean"} for name in CLASSES},
        "reason": {"type": "string"},
    },
    "required": [*CLASSES, "reason"],
    "additionalProperties": False,
}
SYSTEM: Final = """\
You check one resume bullet against the facts it was written from. The facts are the only \
evidence. Decide, for each class below, whether the bullet commits that violation, and set its \
field to true only if it clearly does:
- fabricated_number: the bullet states a number that no fact states (or that cannot be read off \
a fact).
- number_drift: a number the facts do state, but written more favourably than the fact (rounded \
up when higher is better, down when lower is better, a bound such as "8K+" or "up to" made from \
an exact value).
- fabricated_entity: a technology, product or proper noun that no fact or the tech list names.
- do_not_claim: the bullet claims something on the "must not claim" list, or says it was used in \
production or that the author led or managed a team when no fact says so.
- escalation_absolute: an absolute word (never, always, guaranteed, zero, every, all, \
exactly-once, 100%) that no fact uses.
- escalation_causal: a cause or mechanism ("by X", "through X", "so that X") that no fact ties to \
that result.
A faithful bullet may reorder, shorten or reword the facts; that is not a violation. Scope notes \
describe provenance and test conditions: they are context, not achievements, and numbers in them \
are not results. Leave every field false when the bullet is faithful. Give a one-sentence reason \
that names the offending words, or says the bullet is faithful."""


@dataclass(frozen=True, slots=True)
class JudgeInput:
    """What the judge is shown about one bullet.

    Attributes:
        bullet: The bullet to check, plain text.
        facts: The text of the facts it cites that state what was achieved.
        scope: The text of the scope notes among the cited facts.
        claims: The entry's DO NOT CLAIM items.
        tech: The entry's tech list.
    """

    bullet: str
    facts: tuple[str, ...]
    scope: tuple[str, ...]
    claims: tuple[str, ...]
    tech: str


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """The judge's answer: the classes it flagged (in the verifier's order) and why."""

    classes: tuple[str, ...]
    reason: str

    @property
    def flagged(self) -> bool:
        """Whether the judge found any violation."""
        return bool(self.classes)


def build_messages(item: JudgeInput) -> list[Message]:
    """The judge's prompt for one bullet."""
    lines = [
        "Facts:",
        *(f"- {fact}" for fact in item.facts),
        "Scope notes (context only):",
        *([f"- {note}" for note in item.scope] or ["- (none)"]),
        "Must not claim:",
        *([f"- {claim}" for claim in item.claims] or ["- (nothing listed)"]),
        f"Tech list: {item.tech}",
        "",
        f"Bullet: {item.bullet}",
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_verdict(raw: dict[str, Any]) -> JudgeVerdict:
    """Read the judge's answer.

    Raises:
        LLMError: A class is missing or is not a boolean, or the reason is not a string.
    """
    flagged: list[str] = []
    for name in CLASSES:
        value = raw.get(name)
        if not isinstance(value, bool):
            raise LLMError(f"the judge's answer has no boolean {name!r}")
        if value:
            flagged.append(name)
    reason = raw.get("reason")
    if not isinstance(reason, str):
        raise LLMError("the judge's answer has no reason")
    return JudgeVerdict(tuple(flagged), reason.strip())


def judge_bullet(llm: LLM, item: JudgeInput, *, model: str) -> JudgeVerdict:
    """Ask the model whether ``item.bullet`` is supported by its facts.

    Raises:
        LLMError: The call failed or the answer is not a verdict.
    """
    return parse_verdict(llm.complete_json(build_messages(item), SCHEMA, model=model))


def input_for(bank: Bank, bullet: str, entry_id: str, sources: Sequence[str]) -> JudgeInput:
    """The judge's input for a bullet of ``entry_id`` that cites the facts ``sources``.

    The scope note of the entry is always shown (without its DO NOT CLAIM parts, which are listed
    as limits), whether or not the bullet cites it, as a person labelling the bullet would see it.

    Raises:
        KeyError: The entry or a fact id is not in the bank.
    """
    entry = bank.entry(entry_id)
    cited = [bank.fact(fact_id) for fact_id in sources]
    provenance = scope_provenance(entry.scope)
    tech = entry.tech if isinstance(entry, WorkEntry) else entry.tech_stack
    return JudgeInput(
        bullet=bullet,
        facts=tuple(f.text for f in cited if f.kind != "scope"),
        scope=(provenance,) if provenance else (),
        claims=tuple(bank.do_not_claim(entry_id)),
        tech=tech,
    )
