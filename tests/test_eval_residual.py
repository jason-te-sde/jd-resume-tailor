"""The residual check: judging the final bullets, drawing the blind sample, scoring its labels."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM

from tailor.bank import load_bank
from tailor.evals.residual import (
    ResidualError,
    draw_sample,
    frame_of,
    judge_runs,
    judged_positions,
    judged_summary,
    judgement_call_ids,
    score_residual,
    wilson,
)
from tailor.llm import Message


def test_the_wilson_interval_matches_the_known_values() -> None:
    low, high = wilson(0, 60)
    assert low == 0.0
    assert high == pytest.approx(0.0602, abs=0.001)
    low, high = wilson(6, 60)
    assert low == pytest.approx(0.0466, abs=0.001)
    assert high == pytest.approx(0.2015, abs=0.001)
    assert wilson(60, 60)[1] == 1.0


def final(
    outcome: str, text: str, entry: str = "acme-swe", source: str = "acme-swe#1"
) -> dict[str, Any]:
    return {
        "entry": entry,
        "outcome": outcome,
        "first_text": text,
        "first_sources": [source],
        "first_kinds": [],
        "first_structural": False,
        "first_problems": [],
        "final_text": text,
        "final_sources": [source],
        "original": "orig",
    }


def document(count: int = 12) -> dict[str, Any]:
    bullets = [
        final("first_pass", f"Cut p99 latency number {i} from 412 ms to 138 ms.")
        for i in range(count)
    ]
    bullets.append(final("fallback", "Wrote 23 alerts.", source="acme-swe#2"))
    bullets.append(final("verbatim", "Verbatim bullet.", source="acme-swe#2"))
    bullets.append(final("after_retry", "Processed 9,200 jobs/s.", "queue-lab", "queue-lab#1"))
    run = {"jd": "a", "seed": 1, "status": "ok", "bullets": bullets}
    failed = {"jd": "b", "seed": 1, "status": "failed", "bullets": []}
    return {"runs": [run, failed]}


@pytest.fixture
def parsed(tmp_path: Path) -> Any:
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    return load_bank(path)


def flags(flagged: set[str]) -> ScriptedLLM:
    def handler(messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        bullet = messages[-1]["content"].rsplit("Bullet: ", 1)[1]
        answer: dict[str, Any] = {name: False for name in schema["properties"]}
        answer["reason"] = "r"
        if bullet in flagged:
            answer["fabricated_entity"] = True
        return answer

    return ScriptedLLM(handler)


def test_the_frame_is_the_rewritten_bullets_the_verifier_passed_in_completed_runs() -> None:
    frame = frame_of(document())
    outcomes = {document()["runs"][0]["bullets"][i]["outcome"] for _, i in frame}
    assert outcomes == {"first_pass", "after_retry"}
    assert len(frame) == 13
    assert all(run == 0 for run, _ in frame)


def test_every_final_bullet_of_a_completed_run_is_judged_once_per_distinct_text(
    parsed: Any,
) -> None:
    doc = document(2)
    doc["runs"].append({**doc["runs"][0], "seed": 2})  # the same bullets again
    llm = flags(set())
    judged = judge_runs(llm, parsed, doc, model="m")
    assert len(judged) == 10  # two completed runs of five bullets; the failed run has none
    assert len(llm.calls) == 5  # one call for each distinct (entry, text, sources)
    assert {j["outcome"] for j in judged} == {"first_pass", "fallback", "verbatim", "after_retry"}
    assert all(j["judge"] == [] for j in judged)


def sample_of(parsed: Any, flagged: set[str], seed: int = 7, size: int = 5) -> tuple[Any, Any]:
    doc = document()
    judged = judge_runs(flags(flagged), parsed, doc, model="m")
    return draw_sample(doc, judged, parsed, seed=seed, size=size)


FLAGGED = {"Cut p99 latency number 3 from 412 ms to 138 ms.", "Processed 9,200 jobs/s."}


def test_the_sample_holds_every_flagged_bullet_and_the_random_draw_of_the_rest(
    parsed: Any,
) -> None:
    rows, mapping = sample_of(parsed, FLAGGED)
    strata = mapping["strata"]
    assert (strata["flagged"], strata["unflagged"], strata["sampled"]) == (2, 11, 5)
    assert len(rows) == 7
    assert mapping["frame"] == 13
    by_stratum: dict[str, int] = {}
    for info in mapping["rows"].values():
        by_stratum[info["stratum"]] = by_stratum.get(info["stratum"], 0) + 1
    assert by_stratum == {"flagged": 2, "random": 5}
    flagged_ids = {i for i, info in mapping["rows"].items() if info["stratum"] == "flagged"}
    flagged_text = {r["bullet"] for r in rows if r["id"] in flagged_ids}
    assert flagged_text == FLAGGED


def test_a_row_of_the_sample_shows_only_what_a_labeller_needs(parsed: Any) -> None:
    rows, _ = sample_of(parsed, FLAGGED)
    for row in rows:
        assert set(row) == {"id", "bullet", "facts", "scope", "do_not_claim", "tech_stack"}
        assert row["facts"]
    text = json.dumps(rows)
    for leak in ("outcome", "first_pass", "fallback", "judge", "flagged", "finding", "seed"):
        assert leak not in text


def test_the_sample_is_the_same_for_the_same_seed_and_different_for_another(
    parsed: Any,
) -> None:
    first, _ = sample_of(parsed, FLAGGED, seed=7)
    again, _ = sample_of(parsed, FLAGGED, seed=7)
    other, _ = sample_of(parsed, FLAGGED, seed=8)
    assert first == again
    assert [r["bullet"] for r in first] != [r["bullet"] for r in other]


def test_ids_are_opaque_and_the_order_is_shuffled(parsed: Any) -> None:
    rows, mapping = sample_of(parsed, FLAGGED)
    assert [r["id"] for r in rows] == sorted(mapping["rows"])
    positions = [mapping["rows"][r["id"]]["index"] for r in rows]
    assert positions != sorted(positions)


def test_a_draw_larger_than_the_unflagged_frame_takes_all_of_it(parsed: Any) -> None:
    rows, mapping = sample_of(parsed, FLAGGED, size=500)
    assert mapping["strata"]["sampled"] == 11
    assert len(rows) == 13


def labels_for(mapping: dict[str, Any], violating: set[str]) -> dict[str, list[str]]:
    return {i: (["fabricated_entity"] if i in violating else ["ok"]) for i in mapping["rows"]}


def test_the_residual_is_the_flagged_confirmed_plus_the_random_rate_extrapolated(
    parsed: Any,
) -> None:
    _, mapping = sample_of(parsed, FLAGGED, size=5)
    ids = sorted(mapping["rows"])
    random_ids = [i for i in ids if mapping["rows"][i]["stratum"] == "random"]
    flagged_ids = [i for i in ids if mapping["rows"][i]["stratum"] == "flagged"]
    labels = labels_for(mapping, {random_ids[0], flagged_ids[0]})
    result = score_residual(mapping, labels)
    drawn = {k: v for k, v in result["random"].items() if not k.endswith("percent")}
    assert drawn == {
        "sampled": 5,
        "violations": 1,
        "rate": 0.2,
        "interval": [pytest.approx(0.0362, abs=0.001), pytest.approx(0.6245, abs=0.001)],
    }
    assert result["census"]["confirmed"] == 1
    assert result["census"]["flagged"] == 2
    # 1 confirmed + 11 unflagged bullets at the sampled rate of 0.2, out of a frame of 13
    assert result["residual"]["estimate"] == pytest.approx(1 + 11 * 0.2)
    assert result["residual"]["rate"] == pytest.approx((1 + 11 * 0.2) / 13, abs=1e-4)
    assert result["residual"]["rate_interval"][0] < result["residual"]["rate"]
    assert result["residual"]["rate"] < result["residual"]["rate_interval"][1]
    assert result["judge_on_pipeline"]["precision"] == 0.5


def test_a_sample_that_was_drawn_in_full_has_no_extrapolation_interval(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED, size=500)
    ids = sorted(mapping["rows"])
    result = score_residual(mapping, labels_for(mapping, {ids[0], ids[1]}))
    low, high = result["residual"]["rate_interval"]
    assert low == high == result["residual"]["rate"]


def test_labels_must_cover_exactly_the_sampled_rows_with_known_labels(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED)
    labels = labels_for(mapping, set())
    missing = dict(labels)
    missing.pop(sorted(labels)[0])
    with pytest.raises(ResidualError, match="no label"):
        score_residual(mapping, missing)
    with pytest.raises(ResidualError, match="not in the sample"):
        score_residual(mapping, {**labels, "s-999": ["ok"]})
    with pytest.raises(ResidualError, match="unknown label"):
        score_residual(mapping, {**labels, sorted(labels)[0]: ["bogus"]})
    with pytest.raises(ResidualError, match="ok cannot"):
        score_residual(mapping, {**labels, sorted(labels)[0]: ["ok", "do_not_claim"]})


def test_the_violations_in_the_sample_are_counted_per_class(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED)
    labels = labels_for(mapping, set())
    first, second = sorted(labels)[:2]
    labels[first] = ["number_drift", "escalation_causal"]
    labels[second] = ["number_drift"]
    result = score_residual(mapping, labels)
    assert result["labelled"]["by_class"]["number_drift"] == 2
    assert result["labelled"]["by_class"]["escalation_causal"] == 1
    assert result["labelled"]["rows"] == 7
    assert result["labelled"]["violating"] == 2


def test_the_summary_of_judged_bullets_counts_flags_and_errors() -> None:
    records: list[dict[str, Any]] = [
        {"outcome": "first_pass", "judge": []},
        {"outcome": "first_pass", "judge": ["do_not_claim", "number_drift"]},
        {"outcome": "fallback", "judge": ["do_not_claim"]},
        {"outcome": "after_retry", "judge": None, "error": "e"},
    ]
    summary = judged_summary(records)
    assert (summary["bullets"], summary["flagged"], summary["errors"]) == (4, 2, 1)
    assert summary["flagged_by_class"]["do_not_claim"] == 2
    assert summary["flagged_by_class"]["number_drift"] == 1
    assert summary["flagged_by_class"]["fabricated_number"] == 0
    assert summary["flagged_by_outcome"] == {"first_pass": 1, "after_retry": 0, "fallback": 1}


def test_the_judged_positions_are_every_final_bullet_of_the_completed_runs(parsed: Any) -> None:
    doc = document(2)
    judged = judge_runs(flags(set()), parsed, doc, model="m")
    assert judged_positions(doc) == [
        {k: j[k] for k in ("jd", "seed", "index", "entry", "outcome")} for j in judged
    ]
    assert len(judged_positions(doc)) == 5


def test_a_sample_with_no_random_draw_cannot_be_extrapolated(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED, size=0)
    assert mapping["strata"]["unflagged"] > 0
    with pytest.raises(ResidualError, match="no random draw"):
        score_residual(mapping, labels_for(mapping, set()))


def test_the_residual_gives_exact_percents_and_the_faithful_share(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED, size=5)
    ids = sorted(mapping["rows"])
    random_ids = [i for i in ids if mapping["rows"][i]["stratum"] == "random"]
    flagged_ids = [i for i in ids if mapping["rows"][i]["stratum"] == "flagged"]
    result = score_residual(mapping, labels_for(mapping, {random_ids[0], flagged_ids[0]}))
    low, high = wilson(1, 5)
    assert result["confidence_percent"] == 95
    assert result["random"]["percent"] == 20.0
    assert result["random"]["interval_percent"] == [100 * low, 100 * high]
    residual = result["residual"]
    assert residual["extrapolated"] == pytest.approx(11 * 0.2)
    estimate = 1 + 11 * 0.2
    assert residual["percent"] == pytest.approx(100 * estimate / 13)
    assert residual["interval_percent"] == [
        pytest.approx(100 * (1 + 11 * low) / 13),
        pytest.approx(100 * (1 + 11 * high) / 13),
    ]
    assert residual["percent_of_final_bullets"] == pytest.approx(100 * estimate / 15)
    assert residual["faithful_percent"] == pytest.approx(100 - 100 * estimate / 13)
    assert residual["faithful_interval_percent"] == [
        pytest.approx(100 - 100 * (1 + 11 * high) / 13),
        pytest.approx(100 - 100 * (1 + 11 * low) / 13),
    ]
    judge = result["judge_on_pipeline"]
    assert judge["precision_percent"] == 50.0
    assert judge["recall_estimate_percent"] == pytest.approx(100 / estimate)


def test_the_residual_counts_the_judgement_calls_of_its_labelling(parsed: Any) -> None:
    _, mapping = sample_of(parsed, FLAGGED)
    labels = labels_for(mapping, set())
    first = sorted(mapping["rows"])[0]
    calls = judgement_call_ids(f"# Calls\n\n- {first} (ok): reason.\nprose - s-999 (x)\n")
    assert calls == [first]
    assert score_residual(mapping, labels, calls=calls)["judgement_calls"] == 1
    with pytest.raises(ResidualError, match="twice"):
        score_residual(mapping, labels, calls=[first, first])
    with pytest.raises(ResidualError, match="s-999"):
        score_residual(mapping, labels, calls=["s-999"])
