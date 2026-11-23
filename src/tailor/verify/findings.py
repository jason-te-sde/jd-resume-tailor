"""What a check reports: a kind of violation, where in the bullet, and the evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

FindingKind = Literal[
    "fabricated_number",
    "number_drift",
    "fabricated_entity",
    "do_not_claim",
    "escalation_absolute",
    "escalation_causal",
]

KINDS: Final[tuple[FindingKind, ...]] = get_args(FindingKind)


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with a bullet.

    Attributes:
        kind: The class of violation.
        start: Index of the first character of the offending span in the bullet.
        end: Index after the last character of the span.
        text: The offending span, as written in the bullet.
        evidence: What the sources say (or fail to say) about it, in a sentence.
    """

    kind: FindingKind
    start: int
    end: int
    text: str
    evidence: str
