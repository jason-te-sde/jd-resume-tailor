"""Selection: the deterministic choice from scores, and the one scoring call."""

import copy
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK
from llmfakes import ScriptedLLM

from tailor.bank import Bank, load_bank, parse_bank
from tailor.jd.extract import JobDescription
from tailor.llm import LLMError
from tailor.select import Score, bullets_to_keep, choose, normalize_scores, select

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alex-rivera" / "bank.json"
JD = JobDescription("Backend Engineer", "Northwind", ("a", "b", "c", "d"), (), (), ())


def work(entry_id: str, bullets: int = 4) -> dict[str, Any]:
    entry = copy.deepcopy(WORK)
    entry.update(id=entry_id, bullets=[f"Bullet {n}." for n in range(1, bullets + 1)])
    entry["bullet_capabilities"] = ["skill"] * bullets
    entry["scope"] = ""
    return entry


def project(entry_id: str, scope: str = "") -> dict[str, Any]:
    entry = copy.deepcopy(PROJECT)
    entry.update(id=entry_id, bullets=["One.", "Two.", "Three.", "Four."], scope=scope)
    entry["bullet_capabilities"] = ["skill"] * 4
    return entry


def bank_of(*entries: dict[str, Any]) -> Bank:
    return parse_bank({"experiences": list(entries)})


def score(
    entry: str, value: int, covers: set[int] | None = None, order: tuple[int, ...] = ()
) -> Score:
    return Score(entry, value, frozenset(covers or ()), "because", order)


def picked(bank: Bank, scores: list[Score]) -> list[str]:
    return [p.entry.id for p in choose(bank, bank.candidates(), scores)]


def test_at_most_three_entries_of_each_kind_work_first() -> None:
    bank = bank_of(*(work(f"w{n}") for n in range(5)), *(project(f"p{n}") for n in range(5)))
    scores = [score(e.id, 9, {1}) for e in bank.entries]
    ids = picked(bank, scores)
    assert ids == ["w0", "w1", "w2", "p0", "p1", "p2"]


def test_two_are_taken_whatever_their_score_but_a_third_needs_a_real_score() -> None:
    bank = bank_of(work("w0"), work("w1"), work("w2"), work("w3"))
    weak = [score(e.id, 0) for e in bank.entries]
    assert picked(bank, weak) == ["w0", "w1"]
    middling = [score("w0", 3), score("w1", 2), score("w2", 3), score("w3", 1)]
    assert picked(bank, middling) == ["w0", "w2"]
    strong = [score("w0", 5), score("w1", 4), score("w2", 4), score("w3", 1)]
    assert picked(bank, strong) == ["w0", "w1", "w2"]


def test_entries_that_cover_different_responsibilities_beat_a_higher_score_that_repeats() -> None:
    bank = bank_of(work("w0"), work("w1"), work("w2"))
    scores = [score("w0", 9, {1, 2}), score("w1", 8, {1, 2}), score("w2", 5, {3})]
    assert picked(bank, scores) == ["w0", "w2", "w1"]


def test_ties_are_broken_by_entry_id_not_by_input_order() -> None:
    bank = bank_of(work("b"), work("a"), work("c"))
    scores = [score("b", 7, {1}), score("a", 7, {1}), score("c", 7, {1})]
    assert picked(bank, scores) == ["a", "b", "c"]


def test_an_entry_of_the_same_project_as_a_chosen_one_is_skipped() -> None:
    bank = bank_of(
        project("p0"),
        project("p1", scope="Same project as p0: the other side."),
        project("p2"),
        project("p3"),
    )
    scores = [score("p0", 9, {1}), score("p1", 9, {1}), score("p2", 5, {2}), score("p3", 5, {3})]
    assert picked(bank, scores) == ["p0", "p2", "p3"]


def test_the_same_project_rule_holds_whichever_of_the_two_scores_higher() -> None:
    bank = bank_of(
        project("p0"),
        project("p1", scope="Same project as p0: the other side."),
        project("p2"),
    )
    scores = [score("p0", 4, {1}), score("p1", 9, {1}), score("p2", 1)]
    assert picked(bank, scores) == ["p1", "p2"]


def test_projects_without_code_are_never_candidates() -> None:
    no_code = project("p0")
    no_code["has_code"] = False
    bank = bank_of(work("w0"), work("w1"), no_code, project("p1"), project("p2"))
    assert picked(bank, [score(e.id, 8) for e in bank.candidates()]) == ["w0", "w1", "p1", "p2"]


def test_bullets_kept_grow_with_the_score_and_never_exceed_what_the_entry_has() -> None:
    entry = bank_of(work("w0")).entries[0]
    assert [len(bullets_to_keep(entry, score("w0", v))) for v in (0, 4, 5, 7, 8, 10)] == [
        2, 2, 3, 3, 4, 4,
    ]  # fmt: skip
    short = bank_of(work("w1", bullets=1)).entries[0]
    assert bullets_to_keep(short, score("w1", 10)) == (0,)


def test_bullets_come_in_the_stated_priority_then_bank_order() -> None:
    entry = bank_of(work("w0")).entries[0]
    assert bullets_to_keep(entry, score("w0", 10, order=(3, 1))) == (2, 0, 1, 3)
    assert bullets_to_keep(entry, score("w0", 5, order=(4,))) == (3, 0, 1)
    assert bullets_to_keep(entry, score("w0", 5)) == (0, 1, 2)


def test_the_model_answer_is_cleaned_before_anything_depends_on_it() -> None:
    bank = bank_of(work("w0"), work("w1"), work("w2"))
    raw = {
        "scores": [
            {"entry": "w0", "score": 99, "covers": [0, 2, 9, 2], "reason": " r ",
             "bullet_priority": [2, 2, 7, 0, 1]},
            {"entry": "w0", "score": 1, "covers": [], "reason": "duplicate", "bullet_priority": []},
            {"entry": "ghost", "score": 5, "covers": [1], "reason": "", "bullet_priority": []},
            {"entry": "w1", "score": -4, "covers": [], "reason": "", "bullet_priority": []},
        ]
    }  # fmt: skip
    first, second, third = normalize_scores(bank.candidates(), raw, 4)
    assert (first.score, first.covers, first.reason, first.bullet_priority) == (
        10, frozenset({2}), "r", (2, 1),
    )  # fmt: skip
    assert second.score == 0
    assert (third.score, third.reason) == (0, "not scored by the model")


def test_one_call_scores_every_candidate_and_only_candidates() -> None:
    bank = load_bank(EXAMPLE)
    candidates = bank.candidates()
    answer = {
        "scores": [
            {"entry": e.id, "score": 5, "covers": [1], "reason": "ok", "bullet_priority": [1]}
            for e in candidates
        ]
    }
    llm = ScriptedLLM([answer])
    selection = select(llm, bank, JD, model="m")
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][-1]["content"]
    assert all(f"id: {e.id}\n" in prompt for e in candidates)
    assert "id: shelf-forecast\n" not in prompt  # has_code is false
    assert len(selection.scores) == len(candidates)
    ids = [p.entry.id for p in selection.picks]
    assert ids[:3] == ["lantern-swe", "marlow-intern", "tidewell-coop"]
    assert not {"gridlock", "gridlock-bench"} <= set(ids)


def test_covers_that_is_null_or_not_a_list_is_an_llm_error() -> None:
    bank = bank_of(work("w1"))
    candidates = bank.candidates()
    for covers in (None, 3, "1", {"a": 1}):
        raw = {"scores": [{"entry": "w1", "score": 5, "covers": covers, "bullet_priority": [1]}]}
        with pytest.raises(LLMError, match="covers"):
            normalize_scores(candidates, raw, 4)


def test_bullet_priority_that_is_null_or_not_a_list_is_an_llm_error() -> None:
    candidates = bank_of(work("w1")).candidates()
    for priority in (None, 2, "1"):
        raw = {"scores": [{"entry": "w1", "score": 5, "covers": [1], "bullet_priority": priority}]}
        with pytest.raises(LLMError, match="bullet_priority"):
            normalize_scores(candidates, raw, 4)


def test_scores_that_is_not_a_list_is_an_llm_error() -> None:
    candidates = bank_of(work("w1")).candidates()
    for scores in (None, {"entry": "w1"}, "w1"):
        with pytest.raises(LLMError, match="scores"):
            normalize_scores(candidates, {"scores": scores}, 4)


def test_a_missing_covers_or_priority_is_an_empty_list_not_an_error() -> None:
    candidates = bank_of(work("w1")).candidates()
    (score_,) = normalize_scores(candidates, {"scores": [{"entry": "w1", "score": 5}]}, 4)
    assert score_.covers == frozenset()
    assert score_.bullet_priority == ()
