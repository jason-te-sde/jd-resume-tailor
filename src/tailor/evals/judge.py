"""Measure the LLM judge on labelled rows, next to the verifier on the same rows.

The judge is run on every row of a dataset (the holdout split, the blind set) and its flagged
classes are scored against the row labels exactly as the verifier's findings are, with the same
per-class precision and recall. A row the judge fails on (a call error, an answer that is not a
verdict) is recorded with its error and left out of both scores, so the two are always compared on
the same rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from tailor.bank import Bank, WorkEntry
from tailor.bank.facts import scope_provenance
from tailor.evals.verifier import Row, evaluate, score
from tailor.judge import JudgeInput, judge_bullet
from tailor.llm import LLM, LLMError


def row_input(bank: Bank, row: Row) -> JudgeInput:
    """What the judge is shown for a dataset row: never its labels."""
    entry = bank.entry(row.entry)
    scope_texts = {f.text for f in bank.facts_for(row.entry) if f.kind == "scope"}
    provenance = scope_provenance(entry.scope)
    return JudgeInput(
        bullet=row.bullet,
        facts=tuple(s for s in row.sources if s not in scope_texts),
        scope=(provenance,) if provenance else (),
        claims=tuple(bank.do_not_claim(row.entry)),
        tech=entry.tech if isinstance(entry, WorkEntry) else entry.tech_stack,
    )


def judge_rows(
    llm: LLM,
    bank: Bank,
    rows: Sequence[Row],
    *,
    model: str,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the judge on every row.

    Returns:
        One record per row, in order: ``{id, labels, judge, reason}``, where ``judge`` is the list
        of classes it flagged, or ``None`` with an ``error`` when the call failed.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"id": row.id, "labels": list(row.labels)}
        try:
            verdict = judge_bullet(llm, row_input(bank, row), model=model)
        except LLMError as error:
            record |= {"judge": None, "reason": None, "error": str(error)}
        else:
            record |= {"judge": list(verdict.classes), "reason": verdict.reason}
        records.append(record)
        if on_row is not None:
            on_row(record)
    return records


def score_judged(
    bank: Bank, rows: Sequence[Row], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Score the judge and the verifier on the rows the judge answered.

    Returns:
        ``{rows, errors, judge, verifier}``; the two scores are as ``Scores.to_dict`` writes them.
    """
    by_id = {record["id"]: record for record in records}
    answered = [row for row in rows if by_id[row.id]["judge"] is not None]
    judge = score((row.labels, tuple(by_id[row.id]["judge"])) for row in answered)
    return {
        "rows": len(answered),
        "errors": len(rows) - len(answered),
        "judge": judge.to_dict(),
        "verifier": evaluate(answered, bank).to_dict(),
    }
