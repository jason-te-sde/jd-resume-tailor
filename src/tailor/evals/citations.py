"""Fact ids the model copied from the prompt into its bullets, counted over the evaluation's runs.

The prompt lists each fact as ``[id] text`` and asks for the ids in ``sources``; a model sometimes
writes them into the bullet instead. The gate refuses such a bullet
(:func:`tailor.rewrite.copied_citations`); this counts them in the recorded runs, in the first
rewrites and in the final bullets that passed, by the shape they took.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from tailor.bank import Bank
from tailor.rewrite import copied_citations
from tailor.verify import KINDS

SHAPES: Final = (
    "sources_label",
    "fact_label",
    "cites_label",
    "bracketed_id",
    "parenthesized_id",
    "bare_id",
)
_PASSED: Final = ("first_pass", "after_retry")
_NUMBER_SPAN: Final = re.compile(r"^fabricated_number: ('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\") - ")
_FORMS: Final = (
    ("sources_label", re.compile(r"\(\s*sources?\s*:", re.IGNORECASE)),
    ("fact_label", re.compile(r"\(\s*facts?\s*:", re.IGNORECASE)),
    ("cites_label", re.compile(r"\(\s*cites?\b", re.IGNORECASE)),
    ("bracketed_id", re.compile(r"\[[^\]]+\]")),
    ("parenthesized_id", re.compile(r"\([\w-]+#\w+\)")),
)


def shape_of(text: str) -> str:
    """The shape of the copied citation in ``text``: the first of :data:`SHAPES` that fits."""
    return next((name for name, form in _FORMS if form.search(text)), "bare_id")


def _only_id_digits(problems: Sequence[str], ids: Sequence[str]) -> bool:
    """Whether every ``fabricated_number`` span of the problems is the digit of a copied id."""
    digits = {fact_id.split("#", 1)[1] for fact_id in ids if "#" in fact_id}
    spans = [ast.literal_eval(m.group(1)) for p in problems if (m := _NUMBER_SPAN.match(p))]
    return bool(spans) and all(span in digits for span in spans)


def _tally(texts: list[str]) -> dict[str, Any]:
    shapes = [shape_of(text) for text in texts]
    return {"bullets": len(texts), "by_shape": {name: shapes.count(name) for name in SHAPES}}


def count_citations(document: Mapping[str, Any], bank: Bank) -> dict[str, Any]:
    """The bullets of the completed runs whose text carries a fact id of their entry.

    Returns:
        ``first_rewrites``: the first rewrites that carry one, by shape; how many of them the
        verifier passed; how many it flagged ``fabricated_number`` only for the digit of a copied
        ``entry#n`` id (every ``fabricated_number`` span is such a digit); and how many of those
        have no other finding. ``final_bullets``: the bullets that passed and stayed, by shape.
    """
    first: list[str] = []
    passed_first = only_digits = flagged_only_digits = 0
    final: list[str] = []
    for run in document["runs"]:
        if run["status"] != "ok":
            continue
        for bullet in run["bullets"]:
            ids = [fact.id for fact in bank.facts_for(bullet["entry"])]
            text = bullet["first_text"]
            copied = copied_citations(text, ids) if text else ()
            if copied:
                first.append(text)
                kinds = [k for k in bullet["first_kinds"] if k in KINDS]
                passed_first += not kinds
                if "fabricated_number" in kinds and _only_id_digits(
                    bullet["first_problems"], copied
                ):
                    only_digits += 1
                    flagged_only_digits += kinds == ["fabricated_number"]
            if bullet["outcome"] in _PASSED and copied_citations(bullet["final_text"], ids):
                final.append(bullet["final_text"])
    return {
        "first_rewrites": {
            **_tally(first),
            "passed_by_the_verifier": passed_first,
            "fabricated_number_only_id_digits": only_digits,
            "flagged_only_for_id_digits": flagged_only_digits,
        },
        "final_bullets": _tally(final),
    }
