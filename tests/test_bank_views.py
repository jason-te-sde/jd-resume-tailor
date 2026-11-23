from typing import Any

import pytest
from bankdata import NO_CODE, PROJECT, WORK, bank

from tailor.bank import Bank, Fact, Number, ProjectEntry, WorkEntry, parse_bank
from tailor.bank.validate import validate


def entry(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**base, **changes}


def make(*entries: dict[str, Any]) -> Bank:
    return parse_bank(bank(*entries))


# facts_for


def test_an_entry_without_facts_gets_a_fact_per_bullet_and_one_for_its_scope() -> None:
    work = entry(
        WORK, bullets=[r"Cut errors by 38\%.", "Wrote 23 alerts."], bullet_capabilities=["a", "b"]
    )
    facts = make(work).facts_for("acme-swe")
    assert [f.id for f in facts] == ["acme-swe#1", "acme-swe#2", "acme-swe#scope"]
    assert [f.kind for f in facts] == ["bullet", "bullet", "scope"]
    assert [f.entry_id for f in facts] == ["acme-swe"] * 3
    assert facts[0].text == "Cut errors by 38%."
    assert facts[1].text == "Wrote 23 alerts."
    assert all(f.numbers == () for f in facts)


def test_bullet_fact_numbers_start_at_one() -> None:
    ids = [f.id for f in make(PROJECT).facts_for("queue-lab")]
    assert ids[0] == "queue-lab#1"
    assert "queue-lab#0" not in ids


def test_the_scope_fact_leaves_out_what_the_entry_must_not_claim() -> None:
    (scope_fact,) = [f for f in make(WORK).facts_for("acme-swe") if f.kind == "scope"]
    assert scope_fact.id == "acme-swe#scope"
    assert scope_fact.text == "EVIDENCE: load test."
    assert "production p99" not in scope_fact.text
    assert "DO NOT CLAIM" not in scope_fact.text


def test_the_scope_fact_is_plain_text() -> None:
    work = entry(WORK, scope=r"EVIDENCE: cut errors by 38\% (run 3).")
    (scope_fact,) = [f for f in make(work).facts_for("acme-swe") if f.kind == "scope"]
    assert scope_fact.text == "EVIDENCE: cut errors by 38% (run 3)."


@pytest.mark.parametrize(
    "scope", ["", "   ", "DO NOT CLAIM: users; scale."], ids=["empty", "blank", "only limits"]
)
def test_no_scope_fact_when_nothing_is_left_of_the_scope(scope: str) -> None:
    facts = make(entry(WORK, scope=scope)).facts_for("acme-swe")
    assert [f.kind for f in facts] == ["bullet", "bullet"]


def test_no_scope_fact_when_the_scope_is_absent() -> None:
    facts = make(NO_CODE).facts_for("private-model")
    assert [f.id for f in facts] == ["private-model#1"]


def test_explicit_facts_replace_bullets_and_scope() -> None:
    project = entry(
        PROJECT,
        facts=[
            {
                "id": "q-throughput",
                "text": r"9,200 jobs/s at 99\% success",
                "numbers": [{"value": 9200, "unit": "jobs/s", "as_measured": "9,200"}],
            }
        ],
    )
    facts = make(project).facts_for("queue-lab")
    assert facts == [
        Fact(
            id="q-throughput",
            entry_id="queue-lab",
            text="9,200 jobs/s at 99% success",
            kind="explicit",
            numbers=(Number(9200, "jobs/s", "9,200"),),
        )
    ]


def test_facts_for_returns_a_fresh_list() -> None:
    loaded = make(WORK)
    loaded.facts_for("acme-swe").clear()
    assert len(loaded.facts_for("acme-swe")) == 3


def test_facts_for_an_unknown_entry_names_it() -> None:
    with pytest.raises(KeyError, match="no-such-entry"):
        make().facts_for("no-such-entry")


def test_fact_ids_are_unique_across_the_bank() -> None:
    loaded = make(WORK, PROJECT, NO_CODE)
    ids = [f.id for e in loaded.entries for f in loaded.facts_for(e.id)]
    assert len(ids) == len(set(ids)) == 6  # 2 + scope, 1 + scope, 1


def test_a_fact_can_be_found_by_id() -> None:
    loaded = make(WORK, entry(PROJECT, facts=[{"id": "q1", "text": "Ran on 3 nodes"}]))
    assert loaded.fact("acme-swe#2").text == "Wrote 23 alerts."
    assert loaded.fact("acme-swe#scope").kind == "scope"
    assert loaded.fact("q1").entry_id == "queue-lab"
    with pytest.raises(KeyError, match="nope#1"):
        loaded.fact("nope#1")


def test_entry_lookup() -> None:
    loaded = make()
    assert isinstance(loaded.entry("acme-swe"), WorkEntry)
    assert isinstance(loaded.entry("queue-lab"), ProjectEntry)
    with pytest.raises(KeyError, match="ghost"):
        loaded.entry("ghost")


# Fact ids must not collide.


def paths(document: dict[str, Any]) -> list[str]:
    return [str(p) for p in validate(document)]


def test_explicit_fact_ids_must_be_unique_within_an_entry() -> None:
    project = entry(PROJECT, facts=[{"id": "f", "text": "a"}, {"id": "f", "text": "b"}])
    assert paths(bank(project)) == [
        "$.experiences[0].facts[1].id: duplicate fact id 'f',"
        " first used at $.experiences[0].facts[0].id"
    ]


def test_explicit_fact_ids_must_be_unique_across_entries() -> None:
    first = entry(WORK, facts=[{"id": "shared", "text": "a"}])
    second = entry(PROJECT, facts=[{"id": "shared", "text": "b"}])
    assert paths(bank(first, second)) == [
        "$.experiences[1].facts[0].id: duplicate fact id 'shared',"
        " first used at $.experiences[0].facts[0].id"
    ]


def test_an_explicit_fact_id_may_not_take_the_id_an_entry_derives_for_a_bullet() -> None:
    clash = entry(PROJECT, facts=[{"id": "acme-swe#1", "text": "x"}])
    (problem,) = paths(bank(WORK, clash))
    assert problem.startswith("$.experiences[1].facts[0].id: duplicate fact id 'acme-swe#1'")


def test_an_explicit_fact_id_may_not_take_the_scope_id_either() -> None:
    clash = entry(PROJECT, facts=[{"id": "acme-swe#scope", "text": "x"}])
    (problem,) = paths(bank(WORK, clash))
    assert "duplicate fact id 'acme-swe#scope'" in problem


def test_an_entry_with_facts_derives_nothing_so_it_cannot_clash() -> None:
    first = entry(WORK, facts=[{"id": "w1", "text": "a"}])
    second = entry(PROJECT, facts=[{"id": "acme-swe#1", "text": "x"}])
    assert paths(bank(first, second)) == []


def test_a_clash_with_a_later_derived_id_is_reported_at_the_derived_entry() -> None:
    early = entry(PROJECT, facts=[{"id": "acme-swe#2", "text": "x"}])
    (problem,) = paths(bank(early, WORK))
    assert problem.startswith("$.experiences[1].id: fact id 'acme-swe#2' derived for bullet 2")
    assert "$.experiences[0].facts[0].id" in problem


def test_duplicate_entry_ids_are_not_also_reported_as_fact_clashes() -> None:
    duplicate = entry(PROJECT, id="acme-swe")
    (problem,) = paths(bank(WORK, duplicate))
    assert "duplicate id 'acme-swe'" in problem


# do_not_claim


def test_do_not_claim_splits_the_part_into_claims() -> None:
    assert make(WORK).do_not_claim("acme-swe") == ["production p99", "cost savings"]


def test_do_not_claim_collects_every_part_in_order() -> None:
    scope = "DO NOT CLAIM: a; b. EVIDENCE: x. DO NOT CLAIM: c."
    assert make(entry(WORK, scope=scope)).do_not_claim("acme-swe") == ["a", "b", "c"]


def test_do_not_claim_is_plain_text() -> None:
    scope = r"DO NOT CLAIM: a 38\% cut; R\&D ownership"
    assert make(entry(WORK, scope=scope)).do_not_claim("acme-swe") == [
        "a 38% cut",
        "R&D ownership",
    ]


def test_do_not_claim_keeps_a_parenthetical_with_its_claim() -> None:
    scope = "DO NOT CLAIM: 'transactions/s' (3.5K is HTTP requests; writes are 150/s); sharding"
    assert make(entry(WORK, scope=scope)).do_not_claim("acme-swe") == [
        "'transactions/s' (3.5K is HTTP requests; writes are 150/s)",
        "sharding",
    ]


@pytest.mark.parametrize("scope", ["", "EVIDENCE: run 3.", "do not claim: lower case"])
def test_do_not_claim_is_empty_without_the_label(scope: str) -> None:
    assert make(entry(WORK, scope=scope)).do_not_claim("acme-swe") == []


def test_do_not_claim_applies_to_entries_that_have_explicit_facts() -> None:
    project = entry(PROJECT, facts=[{"id": "f", "text": "x"}])
    assert make(project).do_not_claim("queue-lab") == ["exactly-once delivery"]


def test_do_not_claim_for_an_unknown_entry_names_it() -> None:
    with pytest.raises(KeyError, match="ghost"):
        make().do_not_claim("ghost")


# same_project_groups


def test_a_pair_is_one_group_even_if_only_one_side_says_so() -> None:
    loaded = make(WORK, entry(PROJECT, scope="Same project as acme-swe: the other side."))
    assert loaded.same_project_groups() == [frozenset({"acme-swe", "queue-lab"})]


def test_both_sides_saying_so_is_still_one_group() -> None:
    loaded = make(
        entry(WORK, scope="Same project as queue-lab."),
        entry(PROJECT, scope="Same project as acme-swe."),
    )
    assert loaded.same_project_groups() == [frozenset({"acme-swe", "queue-lab"})]


def test_references_chain_into_one_group() -> None:
    a, b, c = (entry(NO_CODE, id=name) for name in ("a", "b", "c"))
    b["scope"] = "Same project as a."
    c["scope"] = "Same project as b."
    assert make(a, b, c).same_project_groups() == [frozenset({"a", "b", "c"})]


def test_several_entries_pointing_at_one_make_one_group() -> None:
    a, b, c = (entry(NO_CODE, id=name) for name in ("a", "b", "c"))
    b["scope"] = "Same project as a."
    c["scope"] = "Same project as a."
    assert make(a, b, c).same_project_groups() == [frozenset({"a", "b", "c"})]


def test_independent_groups_come_out_sorted_by_their_smallest_id() -> None:
    names = ["a", "b", "x", "y", "solo"]
    a, b, x, y, solo = (entry(NO_CODE, id=name) for name in names)
    y["scope"] = "Same project as x."
    b["scope"] = "Same project as a."
    assert make(solo, y, b, x, a).same_project_groups() == [
        frozenset({"a", "b"}),
        frozenset({"x", "y"}),
    ]


def test_no_references_no_groups() -> None:
    assert make(WORK, PROJECT, NO_CODE).same_project_groups() == []


def test_a_reference_written_with_latex_escapes_still_groups() -> None:
    target = entry(NO_CODE, id="snake_case")
    pointer = entry(PROJECT, scope=r"Same project as snake\_case.")
    assert make(target, pointer).same_project_groups() == [frozenset({"snake_case", "queue-lab"})]


# candidates


def test_candidates_are_every_work_entry_and_the_projects_with_code() -> None:
    loaded = make(PROJECT, NO_CODE, WORK, entry(NO_CODE, id="second-private"))
    assert [e.id for e in loaded.candidates()] == ["queue-lab", "acme-swe"]


def test_a_work_entry_is_always_a_candidate() -> None:
    assert [e.id for e in make(WORK).candidates()] == ["acme-swe"]


def test_a_bank_without_code_has_no_project_candidates() -> None:
    assert make(NO_CODE).candidates() == []


def test_candidates_ignore_adaptable() -> None:
    frozen = entry(PROJECT, adaptable=False)
    assert [e.id for e in make(frozen).candidates()] == ["queue-lab"]


# warnings


def warnings_of(*entries: dict[str, Any]) -> list[str]:
    return [str(w) for w in make(*entries).warnings()]


def test_a_complete_bank_has_no_warnings() -> None:
    assert warnings_of(WORK, PROJECT) == []


def test_a_project_without_description_is_warned_about() -> None:
    project = {k: v for k, v in PROJECT.items() if k != "description"}
    assert warnings_of(project) == [
        "$.experiences[0].description: project 'queue-lab' has no description"
    ]


def test_a_blank_description_counts_as_none() -> None:
    assert warnings_of(entry(PROJECT, description="  ")) == [
        "$.experiences[0].description: project 'queue-lab' has no description"
    ]


def test_a_work_entry_without_description_is_not_warned_about() -> None:
    work = {k: v for k, v in WORK.items() if k != "description"}
    assert warnings_of(work) == []


def test_an_entry_without_scope_is_warned_about() -> None:
    work = {k: v for k, v in WORK.items() if k != "scope"}
    (warning,) = warnings_of(work)
    assert warning.startswith("$.experiences[0].scope: 'acme-swe' has no scope")


def test_has_code_without_a_link_is_warned_about() -> None:
    project = {k: v for k, v in PROJECT.items() if k != "github"}
    assert warnings_of(project) == [
        "$.experiences[0].github: project 'queue-lab' has has_code set but no github link"
    ]


def test_a_link_without_has_code_is_warned_about() -> None:
    assert warnings_of(entry(PROJECT, has_code=False)) == [
        "$.experiences[0].has_code: project 'queue-lab' has a github link but has_code is false"
    ]


def test_an_empty_do_not_claim_part_is_warned_about() -> None:
    (warning,) = warnings_of(entry(WORK, scope="EVIDENCE: x. DO NOT CLAIM: EVIDENCE: y."))
    assert warning == "$.experiences[0].scope: 'acme-swe' has a DO NOT CLAIM part with no claims"


def test_warnings_point_at_the_right_entry() -> None:
    bare = {k: v for k, v in PROJECT.items() if k != "description"}
    (warning,) = warnings_of(WORK, entry(bare, id="second"))
    assert warning.startswith("$.experiences[1].description:")
