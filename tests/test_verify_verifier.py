"""verify(): bullets against a bank, with cited facts, entry limits and the report it builds."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.bank import Bank, load_bank, parse_bank
from tailor.verify.rounding import Rounding
from tailor.verify.verifier import Bullet, VerifyError, load_bullets, verify, verify_text


@pytest.fixture
def the_bank() -> Bank:
    return parse_bank(bank(WORK, PROJECT))


def kinds(report_result: Any) -> list[str]:
    return [finding.kind for finding in report_result.findings]


def test_a_faithful_bullet_is_ok(the_bank: Bank) -> None:
    report = verify(the_bank, [Bullet("Reduced p99 latency from 412 ms to 138 ms.", "acme-swe")])
    assert report.ok
    assert report.results[0].findings == ()


def test_without_sources_every_fact_of_the_entry_is_cited(the_bank: Bank) -> None:
    report = verify(the_bank, [Bullet("Wrote 23 alerts.", "acme-swe")])
    assert report.results[0].sources == ("acme-swe#1", "acme-swe#2", "acme-swe#scope")


def test_only_the_cited_facts_count_as_sources(the_bank: Bank) -> None:
    bullet = Bullet("Cut p99 latency to 138 ms.", "acme-swe", sources=("acme-swe#2",))
    (result,) = verify(the_bank, [bullet]).results
    assert kinds(result) == ["fabricated_number"]
    assert result.sources == ("acme-swe#2",)


def test_a_bullet_may_cite_the_scope_fact(the_bank: Bank) -> None:
    bullet = Bullet("Ran a load test.", "acme-swe", sources=("acme-swe#scope",))
    assert verify(the_bank, [bullet]).ok


def test_the_entry_tech_stack_backs_a_technology(the_bank: Bank) -> None:
    ok = Bullet("Tuned PostgreSQL queries.", "acme-swe", sources=("acme-swe#1",))
    bad = Bullet("Tuned MongoDB queries.", "acme-swe", sources=("acme-swe#1",))
    results = verify(the_bank, [ok, bad]).results
    assert kinds(results[0]) == []
    assert kinds(results[1]) == ["fabricated_entity"]


def test_the_entry_do_not_claim_list_is_applied(the_bank: Bank) -> None:
    (result,) = verify(the_bank, [Bullet("Delivered cost savings.", "acme-swe")]).results
    assert kinds(result) == ["do_not_claim"]
    (result,) = verify(the_bank, [Bullet("Measured a production p99.", "acme-swe")]).results
    assert kinds(result) == ["do_not_claim"]


def test_a_numbers_drift_and_an_absolute_are_both_reported_in_text_order(the_bank: Bank) -> None:
    bullet = Bullet("Always cut p99 latency from 412 ms to 130 ms.", "acme-swe")
    (result,) = verify(the_bank, [bullet]).results
    assert kinds(result) == ["escalation_absolute", "number_drift"]
    assert [f.start for f in result.findings] == sorted(f.start for f in result.findings)


def test_a_causal_link_naming_an_unbacked_entity_is_reported_once(the_bank: Bank) -> None:
    bullet = Bullet("Cut p99 latency by moving the reads to Kubernetes.", "acme-swe")
    (result,) = verify(the_bank, [bullet]).results
    assert kinds(result) == ["fabricated_entity"]


def test_a_causal_link_with_a_fabricated_number_is_reported_once(the_bank: Bank) -> None:
    bullet = Bullet("Cut p99 latency by 90 ms thanks to 7 caches.", "acme-swe")
    (result,) = verify(the_bank, [bullet]).results
    assert "escalation_causal" not in kinds(result)
    assert "fabricated_number" in kinds(result)


def test_rounding_nearest_is_passed_through() -> None:
    project = {**PROJECT, "facts": [{"id": "recall", "text": "Reached a recall of 0.875"}]}
    the_bank = parse_bank(bank(project))
    bullet = Bullet("Reached a recall of 88%.", "queue-lab")
    assert [f.kind for f in verify(the_bank, [bullet]).results[0].findings] == ["number_drift"]
    assert not verify(the_bank, [bullet], rounding="conservative").ok
    assert verify(the_bank, [bullet], rounding="nearest").ok
    assert verify(the_bank, [bullet], rounding="nearest").rounding == "nearest"


def test_structured_numbers_of_an_explicit_fact_are_sources() -> None:
    project = {
        **PROJECT,
        "facts": [
            {
                "id": "recall",
                "text": "Reached a recall on the rare class",
                "numbers": [{"value": 0.875, "unit": "", "as_measured": "0.875"}],
            }
        ],
    }
    the_bank = parse_bank(bank(project))
    ok = verify(the_bank, [Bullet("Reached a recall of 0.875.", "queue-lab")])
    drift = verify(the_bank, [Bullet("Reached a recall of 88%.", "queue-lab")])
    assert ok.ok
    assert [f.kind for f in drift.results[0].findings] == ["number_drift"]


def test_an_unknown_entry_is_an_error(the_bank: Bank) -> None:
    with pytest.raises(VerifyError, match=r"bullet 1: .*no-such-entry"):
        verify(the_bank, [Bullet("x", "no-such-entry")])


def test_an_unknown_fact_id_is_an_error(the_bank: Bank) -> None:
    with pytest.raises(VerifyError, match=r"bullet 2: .*acme-swe#9"):
        verify(
            the_bank,
            [Bullet("x", "acme-swe"), Bullet("x", "acme-swe", sources=("acme-swe#9",))],
        )


def test_a_fact_of_another_entry_is_an_error(the_bank: Bank) -> None:
    with pytest.raises(VerifyError, match="belongs to queue-lab, not acme-swe"):
        verify(the_bank, [Bullet("x", "acme-swe", sources=("queue-lab#1",))])


def test_the_report_counts_and_orders_its_results(the_bank: Bank) -> None:
    bullets = [
        Bullet("Wrote 23 alerts.", "acme-swe", id="a"),
        Bullet("Wrote 24 alerts.", "acme-swe", id="b"),
    ]
    report = verify(the_bank, bullets)
    assert [r.id for r in report.results] == ["a", "b"]
    assert [r.index for r in report.results] == [0, 1]
    assert not report.ok
    assert report.violation_count == 1
    assert report.finding_counts == {"fabricated_number": 1}


def test_verify_text_works_without_a_bank() -> None:
    findings = verify_text(
        "Always cut latency to 90 ms.",
        ["Cut latency to 138 ms."],
        claims=[],
        context=[],
    )
    assert [f.kind for f in findings] == ["escalation_absolute", "fabricated_number"]


def test_every_bullet_of_the_example_bank_verifies_against_its_own_facts() -> None:
    example = load_bank(Path(__file__).resolve().parents[1] / "examples/alex-rivera/bank.json")
    bullets = [
        Bullet(bullet, entry.id)
        for entry in example.entries
        for bullet in (fact.text for fact in example.facts_for(entry.id) if fact.kind != "scope")
    ]
    roundings: tuple[Rounding, ...] = ("conservative", "nearest")
    for rounding in roundings:
        report = verify(example, bullets, rounding=rounding)
        assert report.ok, [
            (r.entry, r.text, [(f.kind, f.text) for f in r.findings])
            for r in report.results
            if not r.ok
        ]


# ---- loading bullets -------------------------------------------------------------------------


def write(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "bullets.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_bullets_load_from_a_list_or_an_object_with_bullets(tmp_path: Path) -> None:
    item = {"text": "t", "entry": "e", "sources": ["e#1"], "id": "x"}
    expected = [Bullet("t", "e", sources=("e#1",), id="x")]
    assert load_bullets(write(tmp_path, [item])) == expected
    assert load_bullets(write(tmp_path, {"bullets": [item]})) == expected
    assert load_bullets(write(tmp_path, [{"text": "t", "entry": "e"}])) == [Bullet("t", "e")]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"nope": []}, "an array of bullets"),
        ([1], r"\[0\]: must be an object"),
        ([{"entry": "e"}], r"\[0\]: missing \"text\""),
        ([{"text": "t"}], r"\[0\]: missing \"entry\""),
        ([{"text": 1, "entry": "e"}], r"\[0\]\.text: must be a string"),
        ([{"text": "t", "entry": "e", "sources": "e#1"}], r"\[0\]\.sources: must be an array"),
        ([{"text": "t", "entry": "e", "sources": [1]}], r"\[0\]\.sources\[0\]: must be a string"),
        ([{"text": "t", "entry": "e", "extra": 1}], r"\[0\]: unknown field \"extra\""),
    ],
)
def test_malformed_bullets_are_reported_with_their_path(
    tmp_path: Path, document: object, message: str
) -> None:
    with pytest.raises(VerifyError, match=message):
        load_bullets(write(tmp_path, document))


def test_invalid_json_and_a_missing_file_are_errors(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(VerifyError, match="not valid JSON"):
        load_bullets(broken)
    with pytest.raises(VerifyError, match="cannot read"):
        load_bullets(tmp_path / "missing.json")


def scoped_bank() -> Bank:
    work = {
        **WORK,
        "bullets": ["Cut p99 latency from 412 ms to 138 ms.", "Wrote 23 alerts."],
        "scope": (
            "EVIDENCE: a k6 run in staging at 200 req/s for 10 min, not production traffic. "
            "DO NOT CLAIM: cost savings."
        ),
    }
    return parse_bank(bank(work, PROJECT))


def test_a_number_of_the_scope_is_not_an_achievement(the_bank: Bank) -> None:
    bullet = Bullet("Handled 200 req/s in a staging run.", "acme-swe")
    assert kinds(verify(scoped_bank(), [bullet]).results[0]) == ["fabricated_number"]


def test_the_scope_still_backs_names_and_absolutes() -> None:
    bullet = Bullet("Ran a k6 test in staging.", "acme-swe")
    assert verify(scoped_bank(), [bullet]).ok


def test_verify_text_takes_the_scope_apart_from_the_sources() -> None:
    found = verify_text("Handled 200 req/s.", ["Wrote 23 alerts."], scope=["A run at 200 req/s."])
    assert [f.kind for f in found] == ["fabricated_number"]
    assert verify_text("Handled 200 req/s.", ["A run at 200 req/s."]) == []


def test_production_is_flagged_when_no_fact_says_it_even_if_the_scope_denies_it() -> None:
    bullet = Bullet("Cut p99 latency from 412 ms to 138 ms in production.", "acme-swe")
    result = verify(scoped_bank(), [bullet]).results[0]
    assert [f.text for f in result.findings if f.kind != "fabricated_number"] == ["production"]
