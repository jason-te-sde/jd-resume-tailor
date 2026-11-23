"""Fact ids the model copied into its bullets, counted by shape over the runs."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.bank import load_bank
from tailor.evals.citations import SHAPES, count_citations, shape_of


@pytest.fixture
def parsed(tmp_path: Path) -> Any:
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    return load_bank(path)


def test_each_shape_of_a_copied_citation_is_named() -> None:
    assert shape_of("Cut latency (sources: [acme-swe#1]).") == "sources_label"
    assert shape_of("Cut latency (Sources: [acme-swe#1])") == "sources_label"
    assert shape_of("Cut latency (fact: acme-swe#1).") == "fact_label"
    assert shape_of("Cut latency. (Cites acme-swe#1)") == "cites_label"
    assert shape_of("Cut latency. [acme-swe#1]") == "bracketed_id"
    assert shape_of("Cut latency ([acme-swe#1]).") == "bracketed_id"
    assert shape_of("Cut latency (acme-swe#1).") == "parenthesized_id"
    assert shape_of("Cut latency, acme-swe#1.") == "bare_id"
    assert set(SHAPES) == {
        "sources_label", "fact_label", "cites_label", "bracketed_id", "parenthesized_id", "bare_id"
    }  # fmt: skip


def bullet(outcome: str, first: str, final: str, kinds: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "entry": "acme-swe",
        "outcome": outcome,
        "first_text": first,
        "first_kinds": list(kinds),
        "first_problems": [f"{kind}: 'x' - none" for kind in kinds],
        "final_text": final,
    }


def test_copied_citations_are_counted_in_first_rewrites_and_in_passed_final_bullets(
    parsed: Any,
) -> None:
    clean = "Cut p99 latency from 412 ms to 138 ms."
    number = ("fabricated_number",)
    document = {
        "runs": [
            {
                "status": "ok",
                "bullets": [
                    bullet("first_pass", clean + " [acme-swe#1]", clean + " [acme-swe#1]"),
                    bullet(
                        "fallback", clean + " (fact: acme-swe#1)", clean, ("fabricated_number",)
                    ),
                    bullet("after_retry", clean + " (acme-swe#1)", clean, number),
                    bullet("first_pass", clean, clean),
                ],
            },
            {"status": "failed", "bullets": []},
        ]
    }
    counted = count_citations(document, parsed)
    assert counted["first_rewrites"]["bullets"] == 3
    assert counted["first_rewrites"]["passed_by_the_verifier"] == 1
    assert counted["first_rewrites"]["by_shape"]["fact_label"] == 1
    assert counted["first_rewrites"]["by_shape"]["sources_label"] == 0
    assert counted["final_bullets"]["bullets"] == 1
    assert counted["final_bullets"]["by_shape"]["bracketed_id"] == 1


def flagged(text: str, problems: list[str]) -> dict[str, Any]:
    kinds = list(dict.fromkeys(p.split(":", 1)[0] for p in problems))
    return {
        "entry": "acme-swe",
        "outcome": "fallback",
        "first_text": text,
        "first_kinds": kinds,
        "first_problems": problems,
        "final_text": "Cut p99 latency from 412 ms to 138 ms.",
    }


def test_a_number_flag_that_is_only_the_digit_of_a_copied_id_is_counted(parsed: Any) -> None:
    clean = "Cut p99 latency from 412 ms to 138 ms."
    digit = "fabricated_number: '1' - the sources have no count quantity"
    other = "fabricated_number: '99 ms' - no source"
    cause = "escalation_causal: 'by caching' - none"
    document = {
        "runs": [
            {
                "status": "ok",
                "bullets": [
                    flagged(clean + " [acme-swe#1]", [digit]),  # the id's digit and nothing else
                    flagged(clean + " [acme-swe#1]", [digit, cause]),  # and another finding
                    flagged(clean + " to 99 ms [acme-swe#1]", [other, digit]),  # a real number too
                    flagged(clean + " to 99 ms", [other]),  # no copied id
                ],
            }
        ]
    }
    counted = count_citations(document, parsed)["first_rewrites"]
    assert counted["fabricated_number_only_id_digits"] == 2
    assert counted["flagged_only_for_id_digits"] == 1
