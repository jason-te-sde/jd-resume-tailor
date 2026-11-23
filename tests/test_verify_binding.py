"""A number is bound to the claim it supports: it needs a fact about the same thing."""

import pytest

from tailor.verify.binding import claim_stems, fact_stems, is_bound
from tailor.verify.checks import check_numbers
from tailor.verify.quantities import extract_quantities
from tailor.verify.text import stem

LATENCY = "Cut p99 latency of the tracking API from 412 ms to 138 ms with a batched query."
DEPLOY = (
    "Moved 14 services to one shared pipeline, cutting median deploy time from 38 to 11 minutes."
)
PAGES = "Added 23 alerts; pages per on-call week fell from 31 to 9 over two quarters."


def kinds(bullet: str, *sources: str) -> list[str]:
    return [finding.kind for finding in check_numbers(bullet, sources)]


def words(text: str, number: str) -> frozenset[str]:
    quantity = next(q for q in extract_quantities(text) if q.text.startswith(number))
    return claim_stems(text, quantity)


def test_a_number_borrowed_from_a_fact_about_something_else_is_fabricated() -> None:
    bullet = "Cut p99 latency of the tracking API from 412 ms to 11 minutes."
    found = check_numbers(bullet, [LATENCY, DEPLOY])
    assert [f.kind for f in found] == ["fabricated_number"]
    assert found[0].text == "11 minutes"
    assert "different claim" in found[0].evidence


def test_the_same_number_in_the_fact_it_belongs_to_passes() -> None:
    bullet = "Cut median deploy time from 38 to 11 minutes and p99 latency from 412 ms to 138 ms."
    assert kinds(bullet, LATENCY, DEPLOY) == []


def test_a_count_borrowed_from_another_fact_is_fabricated() -> None:
    assert kinds("Cut pages per on-call week from 31 to 14.", PAGES, DEPLOY) == [
        "fabricated_number"
    ]


def test_a_number_with_no_metric_word_binds_to_any_fact() -> None:
    assert kinds("Reached 138 ms.", LATENCY, DEPLOY) == []


def test_the_leading_verb_is_not_a_metric_word() -> None:
    assert words("Held 138 ms.", "138") == frozenset()
    assert words("Cut p99 latency to 138 ms.", "138") == {"p99", "latenc"}


def test_what_follows_a_connector_says_how_not_what() -> None:
    bullet = "Reached 18,000 writes/s thanks to group commit across writers."
    assert words(bullet, "18,000") == frozenset()
    source = "Raised write throughput from 7,200 to 18,000 writes/s by enlarging the write buffer."
    assert kinds(bullet, source) != ["fabricated_number"]


def test_a_connector_right_before_the_number_does_not_hide_the_metric() -> None:
    assert words("Reduced p99 latency by 41%.", "41") == {"p99", "latenc"}


def test_numbers_and_units_are_not_metric_words() -> None:
    assert words("Cut p99 latency from 412 ms to 138 ms.", "138") == {"p99", "latenc"}


@pytest.mark.parametrize(
    ("claim", "fact", "bound"),
    [
        (frozenset({stem("deploy")}), DEPLOY, True),
        (frozenset({stem("latency")}), DEPLOY, False),
        (frozenset(), DEPLOY, True),
    ],
)
def test_is_bound(claim: frozenset[str], fact: str, bound: bool) -> None:
    assert is_bound(claim, fact) is bound


def test_fact_stems_ignore_numbers_units_and_generic_verbs() -> None:
    stems = fact_stems(DEPLOY)
    assert "deploi" in stems or "deploy" in stems
    assert "38" not in stems
    assert "minut" not in stems
    assert "cut" not in stems
