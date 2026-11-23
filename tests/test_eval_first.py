"""The labelled check of first rewrites: drawing the blind sample of what the gate saw."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.bank import load_bank
from tailor.evals.first import draw_first_sample, first_frame, identical_rows, score_first
from tailor.evals.residual import ResidualError, judgement_call_ids, wilson


def bullet(
    outcome: str,
    first: str,
    *,
    final: str | None = None,
    kinds: tuple[str, ...] = (),
    structural: bool = False,
    entry: str = "acme-swe",
    source: str = "acme-swe#1",
) -> dict[str, Any]:
    return {
        "entry": entry,
        "outcome": outcome,
        "first_text": first,
        "first_sources": [source],
        "first_kinds": list(kinds),
        "first_structural": structural,
        "first_problems": [],
        "final_text": final if final is not None else first,
        "final_sources": [source],
        "original": "orig",
    }


def document(count: int = 10) -> dict[str, Any]:
    bullets = [
        bullet("first_pass", f"Cut p99 latency number {i} from 412 ms to 138 ms.")
        for i in range(count)
    ]
    bullets.append(
        bullet(
            "fallback",
            "Cut p99 latency by 90% from 412 ms.",
            final="Cut p99 latency from 412 ms to 138 ms.",
            kinds=("fabricated_number",),
        )
    )
    bullets.append(
        bullet(
            "after_retry",
            "Processed 9,500 jobs/s.",
            final="Processed 9,200 jobs/s.",
            kinds=("fabricated_number",),
            entry="queue-lab",
            source="queue-lab#1",
        )
    )
    bullets.append(bullet("after_retry", "", final="Wrote 23 alerts.", structural=True))
    bullets.append(bullet("verbatim", "Verbatim bullet.", source="acme-swe#2"))
    run = {"jd": "a", "seed": 1, "status": "ok", "bullets": bullets}
    again = {"jd": "a", "seed": 2, "status": "ok", "bullets": bullets[:2]}
    failed = {"jd": "b", "seed": 1, "status": "failed", "bullets": []}
    return {"runs": [run, failed, again]}


@pytest.fixture
def parsed(tmp_path: Path) -> Any:
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    return load_bank(path)


def test_the_frame_is_every_first_rewrite_that_reached_the_verifier_one_per_run() -> None:
    frame = first_frame(document())
    # run 0: ten passed, one fallback, one retried; the structural and the verbatim bullet are
    # out. Run 2 repeats two bullets, which count again (the run is the unit); run 1 failed.
    assert frame == [(0, i) for i in range(12)] + [(2, 0), (2, 1)]


def test_a_row_shows_the_first_rewrite_and_only_what_a_labeller_needs(parsed: Any) -> None:
    rows, mapping = draw_first_sample(document(), parsed, seed=7, size=14)
    for row in rows:
        assert set(row) == {"id", "bullet", "facts", "scope", "do_not_claim", "tech_stack"}
        assert row["facts"]
    shown = {row["bullet"] for row in rows}
    assert "Processed 9,500 jobs/s." in shown  # the first rewrite, not the final bullet
    assert "Processed 9,200 jobs/s." not in shown
    text = json.dumps(rows)
    for leak in ("outcome", "first_pass", "fallback", "fabricated", "kinds", "seed", '"a"'):
        assert leak not in text
    retried = next(i for i, info in mapping["rows"].items() if info["entry"] == "queue-lab")
    assert next(r for r in rows if r["id"] == retried)["tech_stack"] == "Go, SQLite"


def test_the_draw_is_seeded_and_the_ids_are_opaque(parsed: Any) -> None:
    rows, mapping = draw_first_sample(document(), parsed, seed=7, size=5)
    again, _ = draw_first_sample(document(), parsed, seed=7, size=5)
    other, _ = draw_first_sample(document(), parsed, seed=8, size=5)
    assert rows == again
    assert [r["bullet"] for r in rows] != [r["bullet"] for r in other]
    assert [r["id"] for r in rows] == [f"f-{n:03d}" for n in range(1, 6)]
    assert sorted(mapping["rows"]) == [r["id"] for r in rows]
    positions = [(info["seed"], info["index"]) for info in mapping["rows"].values()]
    assert len(set(positions)) == 5


def test_the_mapping_says_where_each_row_came_from_and_how_it_was_drawn(parsed: Any) -> None:
    rows, mapping = draw_first_sample(document(), parsed, seed=7, size=5)
    assert {k: v for k, v in mapping.items() if k != "rows"} == {
        "seed": 7,
        "size": 5,
        "frame": 14,
        "rewritten": 15,
        "structural": 1,
        "sampled": 5,
    }
    doc = document()
    for row in rows:
        info = mapping["rows"][row["id"]]
        run = next(r for r in doc["runs"] if (r["jd"], r["seed"]) == (info["jd"], info["seed"]))
        assert run["bullets"][info["index"]]["first_text"] == row["bullet"]
        assert run["bullets"][info["index"]]["entry"] == info["entry"]


def test_a_draw_larger_than_the_frame_takes_all_of_it(parsed: Any) -> None:
    rows, mapping = draw_first_sample(document(), parsed, seed=7, size=500)
    assert len(rows) == mapping["sampled"] == mapping["frame"] == 14


def labelled(
    mapping: dict[str, Any], by_position: dict[tuple[int, int], list[str]]
) -> dict[str, list[str]]:
    """Labels for every sampled row: ``by_position`` keyed by (seed, bullet index), else ok."""
    return {
        row_id: by_position.get((info["seed"], info["index"]), ["ok"])
        for row_id, info in mapping["rows"].items()
    }


def scored(parsed: Any, by_position: dict[tuple[int, int], list[str]]) -> dict[str, Any]:
    doc = document()
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=500)
    return score_first(doc, mapping, labelled(mapping, by_position))


# In run (a, 1), bullet 10 (fabricated_number) and bullet 11 (fabricated_number) were flagged.
LABELS = {
    (1, 10): ["fabricated_number"],  # flagged, and the class is right
    (1, 11): ["do_not_claim"],  # flagged, a violation of another class
    (1, 3): ["escalation_causal"],  # passed by the verifier, labelled a violation
}


def test_the_violation_rate_of_first_rewrites_is_the_labelled_share_with_an_interval(
    parsed: Any,
) -> None:
    result = scored(parsed, LABELS)
    rate = result["violation_rate"]
    assert (rate["violating"], rate["rows"]) == (3, 14)
    assert rate["rate"] == round(3 / 14, 4)
    assert rate["interval"] == [round(v, 4) for v in wilson(3, 14)]
    assert result["labelled"]["by_class"]["do_not_claim"] == 1


def test_the_gate_off_flag_is_scored_as_a_yes_or_no_against_the_labels(parsed: Any) -> None:
    flag = scored(parsed, LABELS)["gate_off_flag"]
    flag = {k: v for k, v in flag.items() if not k.endswith("percent")}
    assert flag == {
        "flagged": 2,
        "flagged_violating": 2,
        "precision": 1.0,
        "violating": 3,
        "recall": round(2 / 3, 4),
        "unflagged": 12,
        "unflagged_violating": 1,
        "ok": 11,
        "false_alarms": 0,
        "false_alarm_rate": 0.0,
    }


def test_the_verifier_is_scored_per_class_and_on_whether_its_class_is_right(
    parsed: Any,
) -> None:
    verifier = scored(parsed, LABELS)["verifier"]
    number = verifier["classes"]["fabricated_number"]
    assert (number["support"], number["predicted"]) == (1, 2)
    assert (number["precision"], number["recall"]) == (0.5, 1.0)
    assert verifier["classes"]["do_not_claim"]["recall"] == 0.0
    assert verifier["caught_violation_recall"] == round(2 / 3, 4)
    # one flagged row has a labelled class among its findings, of 2 flagged and 3 violating
    right = {k: v for k, v in verifier["class_right"].items() if not k.endswith("percent")}
    assert right == {"rows": 1, "precision": 0.5, "recall": round(1 / 3, 4)}


def test_labels_must_cover_exactly_the_sampled_rows(parsed: Any) -> None:
    doc = document()
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=3)
    labels = labelled(mapping, {})
    with pytest.raises(ResidualError, match="no label"):
        score_first(doc, mapping, dict(list(labels.items())[1:]))
    with pytest.raises(ResidualError, match="not in the sample"):
        score_first(doc, mapping, {**labels, "f-999": ["ok"]})


def test_a_mapping_that_does_not_point_at_its_bullet_is_refused(parsed: Any) -> None:
    doc = document()
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=3)
    first = sorted(mapping["rows"])[0]
    mapping["rows"][first]["entry"] = (
        "queue-lab" if mapping["rows"][first]["entry"] != "queue-lab" else "acme-swe"
    )
    with pytest.raises(ResidualError, match=first):
        score_first(doc, mapping, labelled(mapping, {}))


def test_the_rates_of_first_rewrites_come_with_exact_percents(parsed: Any) -> None:
    result = scored(parsed, LABELS)
    low, high = wilson(3, 14)
    assert result["confidence_percent"] == 95
    rate = result["violation_rate"]
    assert rate["percent"] == 300 / 14
    assert rate["interval_percent"] == [100 * low, 100 * high]
    flag = result["gate_off_flag"]
    assert (flag["precision_percent"], flag["recall_percent"]) == (100.0, 200 / 3)
    assert flag["false_alarm_percent"] == 0.0
    assert result["verifier"]["class_right"]["precision_percent"] == 50.0


def test_a_first_rewrite_refused_for_a_copied_citation_is_in_the_frame_and_not_a_flag(
    parsed: Any,
) -> None:
    doc = document()
    doc["runs"][0]["bullets"][0]["first_kinds"] = ["copied_citation"]
    assert (0, 0) in first_frame(doc)
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=500)
    result = score_first(doc, mapping, labelled(mapping, LABELS))
    assert result["gate_off_flag"]["flagged"] == 2


# ---- rows the passed-bullet sample also holds ------------------------------------------------


def row(row_id: str, bullet: str, facts: tuple[str, ...] = ("fact",)) -> dict[str, Any]:
    return {"id": row_id, "bullet": bullet, "facts": list(facts), "scope": [], "do_not_claim": []}


def test_rows_identical_in_bullet_and_facts_to_a_passed_sample_row_are_found() -> None:
    first = [row("f-001", "A"), row("f-002", "B"), row("f-003", "C", ("other",))]
    passed = [row("s-001", "B"), row("s-002", "C"), row("s-003", "B")]
    assert identical_rows(first, passed) == {"f-002": ["s-001", "s-003"]}


def test_the_overlap_is_scored_with_the_label_agreement_and_the_rest_apart(parsed: Any) -> None:
    doc = document()
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=500)
    labels = labelled(mapping, LABELS)
    ids = sorted(mapping["rows"])
    by_position = {(info["seed"], info["index"]): i for i, info in mapping["rows"].items()}
    retried, passed_ok = by_position[(1, 11)], by_position[(1, 0)]  # do_not_claim; ok
    repeats = {retried: ["s-009"], passed_ok: ["s-001"]}
    passed_labels = {"s-009": ["do_not_claim"], "s-001": ["ok"]}
    result = score_first(doc, mapping, labels, repeats=repeats, passed_labels=passed_labels)
    overlap = result["overlap_with_passed_sample"]
    assert {k: overlap[k] for k in ("rows", "same_labels", "ok", "violating")} == {
        "rows": 2,
        "same_labels": 2,
        "ok": 1,
        "violating": 1,
    }
    assert overlap["pairs"] == {retried: ["s-009"], passed_ok: ["s-001"]}
    rest = result["without_overlap"]
    assert (rest["rows"], rest["violating"]) == (len(ids) - 2, 2)
    assert rest["percent"] == 200 / (len(ids) - 2)
    assert (rest["gate_off_flag"]["flagged"], rest["gate_off_flag"]["flagged_violating"]) == (1, 1)
    changed = score_first(
        doc, mapping, labels, repeats=repeats, passed_labels={**passed_labels, "s-001": ["x"]}
    )
    assert changed["overlap_with_passed_sample"]["same_labels"] == 1


def test_judgement_calls_are_counted_and_must_name_sampled_rows(parsed: Any) -> None:
    doc = document()
    _, mapping = draw_first_sample(doc, parsed, seed=7, size=500)
    labels = labelled(mapping, LABELS)
    text = "# Judgement calls\n\n- f-001 (ok): a reason.\n- f-004 (ok): Same call as f-001.\n"
    calls = judgement_call_ids(text)
    assert calls == ["f-001", "f-004"]
    assert score_first(doc, mapping, labels, calls=calls)["judgement_calls"] == 2
    with pytest.raises(ResidualError, match="f-999"):
        score_first(doc, mapping, labels, calls=["f-999"])
