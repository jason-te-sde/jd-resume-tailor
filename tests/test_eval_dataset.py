"""The committed dataset, split and results: the properties the evaluation promises."""

import json
from collections import Counter
from pathlib import Path

import pytest

from tailor.bank import Bank, load_bank
from tailor.evals.residual import judgement_call_ids
from tailor.evals.verifier import (
    CLASSES,
    DEFAULT_TOLERANCE,
    Row,
    build_split,
    check_gate,
    evaluate,
    load_rows,
    load_split,
    validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "verifier"


@pytest.fixture(scope="module")
def rows() -> list[Row]:
    return load_rows(DATA / "labeled.jsonl")


@pytest.fixture(scope="module")
def example() -> Bank:
    return load_bank(ROOT / "examples" / "alex-rivera" / "bank.json")


def test_the_dataset_has_at_least_240_rows(rows: list[Row]) -> None:
    assert len(rows) >= 240


def test_every_violation_class_has_at_least_20_rows(rows: list[Row]) -> None:
    counts = Counter(label for row in rows for label in row.labels)
    for name in CLASSES:
        assert counts[name] >= 20, (name, counts[name])


def test_there_are_at_least_70_ok_rows(rows: list[Row]) -> None:
    assert sum(row.labels == ("ok",) for row in rows) >= 70


def test_the_overclaim_pattern_rows_are_still_the_bulk_of_the_violations(
    rows: list[Row],
) -> None:
    violations = [row for row in rows if row.labels != ("ok",)]
    pattern = [row for row in violations if row.id.startswith("pattern-")]
    assert len(pattern) == 89
    assert all(row.id.startswith(("pattern-", "synth-")) for row in rows)


def test_the_ok_rows_are_rewrites_not_copies_of_a_bank_bullet(
    rows: list[Row], example: Bank
) -> None:
    bank_bullets = {fact.text for entry in example.entries for fact in example.facts_for(entry.id)}
    copies = [row.id for row in rows if row.labels == ("ok",) and row.bullet in bank_bullets]
    assert copies == []


def test_every_source_is_a_fact_of_the_rows_entry(rows: list[Row], example: Bank) -> None:
    assert validate_rows(rows, example) == []


def test_no_bullet_repeats(rows: list[Row]) -> None:
    bullets = Counter((row.entry, row.bullet) for row in rows)
    assert [key for key, count in bullets.items() if count > 1] == []


def test_split_json_lists_every_row_once(rows: list[Row]) -> None:
    sides = load_split(DATA / "split.json", rows)
    assert len(sides["dev"]) + len(sides["holdout"]) == len(rows)
    recorded = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    assert recorded == build_split(rows, recorded["seed"], previous=recorded)


def test_the_holdout_is_the_84_rows_drawn_at_the_start(rows: list[Row]) -> None:
    assert len(load_split(DATA / "split.json", rows)["holdout"]) == 84


def test_about_thirty_percent_of_the_dataset_is_held_out(rows: list[Row]) -> None:
    holdout = len(load_split(DATA / "split.json", rows)["holdout"])
    assert 0.2 < holdout / len(rows) < 0.4


def test_every_class_has_holdout_rows(rows: list[Row]) -> None:
    holdout = load_split(DATA / "split.json", rows)["holdout"]
    counts = Counter(label for row in holdout for label in row.labels)
    for name in (*CLASSES, "ok"):
        assert counts[name] >= 3, (name, counts[name])


def test_the_committed_results_pass_the_gate(rows: list[Row], example: Bank) -> None:
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    holdout = load_split(DATA / "split.json", rows)["holdout"]
    fresh = evaluate(holdout, example)
    assert check_gate(fresh, results["holdout"], DEFAULT_TOLERANCE) == []
    assert results["dataset"]["rows"] == len(rows)
    assert results["dataset"]["holdout"] == len(holdout)


def test_the_blind_rows_are_facts_of_the_bank_and_their_rescore_is_the_committed_one(
    example: Bank,
) -> None:
    blind = DATA / "blind"
    blind_rows = load_rows(blind / "rows.jsonl")
    assert len(blind_rows) == 86
    assert validate_rows(blind_rows, example) == []
    recorded = json.loads((blind / "results.json").read_text(encoding="utf-8"))
    assert recorded["rescored"]["scores"] == evaluate(blind_rows, example).to_dict()


def test_the_recorded_counts_of_the_dataset_history_follow_from_the_dataset() -> None:
    history = json.loads((DATA / "history.json").read_text(encoding="utf-8"))["dataset"]
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))["dataset"]
    before = history["rows_before_the_review_fixes"]
    assert history["rows_added_after_the_review"] == results["rows"] - before
    assert history["dev_rows_before_the_review_fixes"] == before - results["holdout"]


def test_the_blind_sets_judgement_calls_are_counted_from_its_file() -> None:
    blind = DATA / "blind"
    calls = judgement_call_ids((blind / "judgement-calls.md").read_text(encoding="utf-8"))
    kept = {row.id for row in load_rows(blind / "rows.jsonl")}
    recorded = json.loads((blind / "results.json").read_text(encoding="utf-8"))
    assert recorded["judgement_calls"] == {
        "listed": len(calls),
        "rows_in_the_set": sum(call in kept for call in calls),
    }
