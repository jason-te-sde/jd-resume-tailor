"""The verifier evaluation: rows, the fixed split, the metrics and the regression gate."""

import json
from pathlib import Path

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.bank import Bank, parse_bank
from tailor.evals.verifier import (
    CLASSES,
    EvalError,
    Row,
    Scores,
    assign_split,
    build_split,
    check_gate,
    evaluate,
    load_rows,
    load_split,
    render_table,
    score,
    validate_rows,
)

SEED = "test-seed"


def row(
    row_id: str,
    bullet: str,
    labels: list[str],
    entry: str = "acme-swe",
    sources: list[str] | None = None,
) -> Row:
    return Row(
        id=row_id,
        entry=entry,
        sources=tuple(
            sources if sources is not None else ["Cut p99 latency from 412 ms to 138 ms."]
        ),
        bullet=bullet,
        labels=tuple(labels),
    )


@pytest.fixture
def the_bank() -> Bank:
    return parse_bank(bank(WORK, PROJECT))


# ---- loading and validating rows -------------------------------------------------------------


def write_rows(tmp_path: Path, rows: list[object]) -> Path:
    path = tmp_path / "labeled.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


GOOD = {
    "id": "a",
    "entry": "acme-swe",
    "sources": ["s"],
    "bullet": "b",
    "labels": ["ok"],
}


def test_rows_load_from_jsonl(tmp_path: Path) -> None:
    rows = load_rows(write_rows(tmp_path, [GOOD, {**GOOD, "id": "b", "labels": ["number_drift"]}]))
    assert [r.id for r in rows] == ["a", "b"]
    assert rows[1].labels == ("number_drift",)
    assert rows[0].sources == ("s",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"labels": ["nonsense"]}, "unknown label"),
        ({"labels": []}, "at least one label"),
        ({"labels": ["ok", "number_drift"]}, "ok cannot be combined"),
        ({"labels": ["number_drift", "number_drift"]}, "repeats a label"),
        ({"sources": []}, "at least one source"),
        ({"bullet": ""}, "bullet must be a non-empty string"),
        ({"id": 3}, "id must be a non-empty string"),
        ({"extra": 1}, "unknown field"),
    ],
)
def test_invalid_rows_are_rejected_with_their_line(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    with pytest.raises(EvalError, match=rf"line 2: .*{message}"):
        load_rows(write_rows(tmp_path, [GOOD, {**GOOD, "id": "b", **change}]))


def test_a_missing_field_and_bad_json_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match=r"line 1: missing field"):
        load_rows(write_rows(tmp_path, [{"id": "a"}]))
    path = tmp_path / "x.jsonl"
    path.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(EvalError, match="line 1: not valid JSON"):
        load_rows(path)
    with pytest.raises(EvalError, match="cannot read"):
        load_rows(tmp_path / "none.jsonl")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="line 2: duplicate id"):
        load_rows(write_rows(tmp_path, [GOOD, GOOD]))


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(GOOD) + "\n\n", encoding="utf-8")
    assert len(load_rows(path)) == 1


def test_sources_must_be_facts_of_the_rows_entry(the_bank: Bank) -> None:
    good = row("a", "b", ["ok"], sources=["Wrote 23 alerts."])
    unknown_entry = row("b", "b", ["ok"], entry="nowhere")
    foreign_text = row("c", "b", ["ok"], sources=["Processed 9,200 jobs/s in a 3-node test."])
    problems = validate_rows([good, unknown_entry, foreign_text], the_bank)
    assert problems == [
        "b: unknown entry 'nowhere'",
        "c: source 1 is not a fact of acme-swe: 'Processed 9,200 jobs/s in a 3-node test.'",
    ]


# ---- the split -------------------------------------------------------------------------------


def test_the_split_is_a_function_of_seed_and_id_only() -> None:
    assert assign_split("row-1", SEED) == assign_split("row-1", SEED)
    ids = [f"row-{n}" for n in range(200)]
    forward = {i: assign_split(i, SEED) for i in ids}
    backward = {i: assign_split(i, SEED) for i in reversed(ids)}
    assert forward == backward


def test_about_thirty_percent_of_rows_are_held_out() -> None:
    ids = [f"row-{n}" for n in range(2000)]
    share = sum(assign_split(i, SEED) == "holdout" for i in ids) / len(ids)
    assert 0.27 < share < 0.33


def test_a_different_seed_gives_a_different_split() -> None:
    ids = [f"row-{n}" for n in range(200)]
    assert [assign_split(i, SEED) for i in ids] != [assign_split(i, "other") for i in ids]


def test_adding_rows_does_not_move_the_others() -> None:
    few = [f"row-{n}" for n in range(50)]
    many = [*few, *(f"extra-{n}" for n in range(50))]
    assert [assign_split(i, SEED) for i in few] == [assign_split(i, SEED) for i in many[:50]]


def test_build_split_lists_each_row_in_exactly_one_side() -> None:
    rows = [row(f"r{n}", "b", ["ok"]) for n in range(100)]
    split = build_split(rows, SEED)
    assert split["seed"] == SEED
    assert set(split["dev"]) | set(split["holdout"]) == {r.id for r in rows}
    assert not set(split["dev"]) & set(split["holdout"])
    assert split["dev"] == sorted(split["dev"])
    assert split["holdout_fraction"] == 0.3


def test_a_recorded_split_keeps_its_sides_and_sends_new_rows_to_dev() -> None:
    old = [row(f"r{n}", "b", ["ok"]) for n in range(100)]
    first = build_split(old, SEED)
    new = [*old, *(row(f"n{n}", "b", ["ok"]) for n in range(200))]
    second = build_split(new, SEED, previous=first)
    assert second["holdout"] == first["holdout"]
    assert {i for i in second["dev"] if i.startswith("n")} == {f"n{n}" for n in range(200)}


def test_a_recorded_split_drops_rows_that_are_gone() -> None:
    old = [row(f"r{n}", "b", ["ok"]) for n in range(100)]
    first = build_split(old, SEED)
    second = build_split(old[:50], SEED, previous=first)
    assert set(second["dev"]) | set(second["holdout"]) == {f"r{n}" for n in range(50)}


def test_load_split_refuses_a_row_on_both_sides_or_on_none(tmp_path: Path) -> None:
    rows = [row("a", "b", ["ok"]), row("b", "b", ["ok"]), row("c", "b", ["ok"])]
    path = tmp_path / "split.json"
    for dev, holdout in ((["a", "b"], []), (["a", "b", "c"], ["c"]), (["a", "b"], ["c", "x"])):
        path.write_text(json.dumps({"dev": dev, "holdout": holdout}), encoding="utf-8")
        with pytest.raises(EvalError, match="does not match"):
            load_split(path, rows)
    path.write_text(json.dumps({"dev": ["a", "b"], "holdout": ["c"]}), encoding="utf-8")
    sides = load_split(path, rows)
    assert [r.id for r in sides["holdout"]] == ["c"]


# ---- scoring ---------------------------------------------------------------------------------


def test_per_class_precision_and_recall_count_multi_label_rows_in_each_class() -> None:
    truth_and_prediction = [
        (("fabricated_number",), ("fabricated_number",)),
        (("fabricated_number",), ()),
        (("number_drift", "escalation_absolute"), ("escalation_absolute",)),
        (("ok",), ("number_drift",)),
        (("ok",), ()),
    ]
    scores = score(truth_and_prediction)
    number = scores.classes["fabricated_number"]
    assert (number.support, number.predicted, number.true_positives) == (2, 1, 1)
    assert number.precision == 1.0
    assert number.recall == 0.5
    drift = scores.classes["number_drift"]
    assert (drift.support, drift.predicted, drift.precision, drift.recall) == (1, 1, 0.0, 0.0)
    absolute = scores.classes["escalation_absolute"]
    assert (absolute.precision, absolute.recall) == (1.0, 1.0)


def test_caught_violation_recall_counts_any_finding_on_a_violation_row() -> None:
    scores = score(
        [
            (("fabricated_number",), ("number_drift",)),
            (("do_not_claim",), ()),
            (("ok",), ("do_not_claim",)),
            (("ok",), ()),
        ]
    )
    assert scores.violation_rows == 2
    assert scores.caught_violation_recall == 0.5
    assert scores.false_alarm_rate == 0.5
    assert scores.ok.precision == 0.5
    assert scores.ok.recall == 0.5


def test_an_empty_denominator_gives_none_not_a_number() -> None:
    scores = score([(("ok",), ())])
    assert scores.classes["number_drift"].precision is None
    assert scores.classes["number_drift"].recall is None
    assert scores.caught_violation_recall is None


def test_every_class_is_scored_even_with_no_rows() -> None:
    assert list(score([]).classes) == list(CLASSES)


def test_evaluate_runs_the_verifier_and_scores_the_split(the_bank: Bank) -> None:
    rows = [
        row("a", "Cut p99 latency from 412 ms to 138 ms.", ["ok"]),
        row("b", "Cut p99 latency to 90 ms.", ["fabricated_number"]),
        row("c", "Delivered cost savings.", ["do_not_claim"], sources=["Wrote 23 alerts."]),
    ]
    scores = evaluate(rows, the_bank)
    assert scores.classes["fabricated_number"].recall == 1.0
    assert scores.classes["do_not_claim"].recall == 1.0
    assert scores.caught_violation_recall == 1.0
    assert scores.false_alarm_rate == 0.0


def test_evaluate_keeps_the_numbers_of_a_scope_fact_out_of_the_sources() -> None:
    work = {**WORK, "scope": "EVIDENCE: a load test at 200 req/s. DO NOT CLAIM: cost savings."}
    scoped = parse_bank(bank(work, PROJECT))
    scope_text = "EVIDENCE: a load test at 200 req/s."
    sources = ["Wrote 23 alerts.", scope_text]
    rows = [row("a", "Handled 200 req/s.", ["fabricated_number"], sources=sources)]
    assert evaluate(rows, scoped).classes["fabricated_number"].recall == 1.0


def test_evaluate_uses_the_conservative_policy(the_bank: Bank) -> None:
    rows = [row("a", "Cut p99 latency from 412 ms to 130 ms.", ["number_drift"])]
    assert evaluate(rows, the_bank).classes["number_drift"].recall == 1.0


def test_the_table_has_a_line_per_class_and_the_overall_numbers() -> None:
    scores = score([(("fabricated_number",), ("fabricated_number",)), (("ok",), ())])
    table = render_table("holdout", scores)
    lines = table.splitlines()
    assert lines[0].startswith("holdout (2 rows)")
    assert any(line.startswith("fabricated_number") and "1.00" in line for line in lines)
    assert any(line.startswith("ok") for line in lines)
    assert any("caught-violation recall" in line and "1.00" in line for line in lines)
    assert any("false-alarm rate" in line for line in lines)
    assert "n/a" in table


# ---- the gate --------------------------------------------------------------------------------


def scores_with(recall_hits: int, of: int = 10) -> Scores:
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("fabricated_number",), ("fabricated_number",) if n < recall_hits else ())
        for n in range(of)
    ]
    return score([*pairs, (("ok",), ())])


def test_the_gate_passes_when_nothing_dropped() -> None:
    assert check_gate(scores_with(9), scores_with(9).to_dict(), tolerance=0.02) == []


def test_the_gate_passes_when_the_numbers_improved() -> None:
    assert check_gate(scores_with(10), scores_with(8).to_dict(), tolerance=0.02) == []


def test_the_gate_fails_on_a_drop_larger_than_the_tolerance() -> None:
    failures = check_gate(scores_with(8), scores_with(9).to_dict(), tolerance=0.02)
    assert any("fabricated_number recall" in f and "0.80" in f and "0.90" in f for f in failures)


def test_the_gate_tolerates_a_drop_within_the_tolerance() -> None:
    committed = scores_with(90, of=100).to_dict()
    assert check_gate(scores_with(89, of=100), committed, tolerance=0.02) == []
    assert check_gate(scores_with(87, of=100), committed, tolerance=0.02) != []


def test_the_gate_fails_when_false_alarms_rise() -> None:
    clean = score([(("ok",), ()) for _ in range(10)])
    noisy = score([(("ok",), ("number_drift",) if n < 3 else ()) for n in range(10)])
    failures = check_gate(noisy, clean.to_dict(), tolerance=0.02)
    assert any("false-alarm rate" in f for f in failures)


def test_the_gate_skips_metrics_the_baseline_has_no_value_for() -> None:
    assert check_gate(scores_with(3), score([]).to_dict(), tolerance=0.02) == []


def test_the_gate_fails_on_a_metric_that_became_undefined() -> None:
    failures = check_gate(score([(("ok",), ())]), scores_with(9).to_dict(), tolerance=0.02)
    assert any("fabricated_number recall" in f and "undefined" in f for f in failures)


def test_scores_round_trip_through_a_dict() -> None:
    scores = scores_with(7)
    document = scores.to_dict()
    assert document["classes"]["fabricated_number"] == {
        "support": 10,
        "predicted": 7,
        "precision": 1.0,
        "recall": 0.7,
    }
    assert document["rows"] == 11
    assert json.loads(json.dumps(document)) == document


def test_the_written_scores_carry_the_counts_behind_the_overall_rates() -> None:
    document = score(
        [
            (("fabricated_number",), ("number_drift",)),
            (("do_not_claim",), ()),
            (("escalation_causal",), ("escalation_causal",)),
            (("ok",), ("do_not_claim",)),
            (("ok",), ()),
            (("ok",), ()),
        ]
    ).to_dict()
    assert (document["caught"], document["violation_rows"]) == (2, 3)
    assert (document["false_alarms"], document["ok"]["support"]) == (1, 3)
    assert document["caught_percent"] == 200 / 3  # exact: 66.67, where the rate says 0.6667
    assert document["false_alarm_percent"] == 100 / 3
