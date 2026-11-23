"""Every scored file of the evaluation can be recomputed from its committed inputs (``--check``)."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_cli_eval_pipeline import (  # noqa: F401 (repo is a fixture)
    git,
    judge_data,
    judge_model,
    make_results,
    patch_factory,
    repo,
)

from tailor.cli import main

PIPELINE = Path("eval") / "pipeline"


def edit(path: Path, change: Any) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def edit_jsonl(path: Path, change: Any) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def label_all(sample: Path, labels: Path, label: str = "ok") -> None:
    rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
    labels.write_text(
        "".join(json.dumps({"id": r["id"], "labels": [label]}) + "\n" for r in rows),
        encoding="utf-8",
    )


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def calls(sample: Path, target: Path, count: int = 1) -> None:
    """A judgement-calls file that lists the first ``count`` rows of the sample."""
    rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
    listed = "".join(f"- {r['id']} (ok): a borderline reading.\n" for r in rows[:count])
    target.write_text("# Judgement calls\n\n" + listed, encoding="utf-8")


@pytest.fixture
def scored(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:  # noqa: F811
    """A repository holding results, judged runs, both samples and the residual score."""
    make_results(repo, monkeypatch, tmp_path)
    patch_factory(monkeypatch, lambda: judge_model("Wrote 23 alerts"))
    assert main(["eval", "judge-runs", "--bank", "bank.json"]) == 0
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "judged")
    assert main(["eval", "sample", "--bank", "bank.json", "--seed", "5", "--size", "2"]) == 0
    label_all(repo / PIPELINE / "blind-sample.jsonl", repo / PIPELINE / "blind-labels.jsonl")
    calls(repo / PIPELINE / "blind-sample.jsonl", repo / PIPELINE / "blind-judgement-calls.md")
    assert main(["eval", "residual"]) == 0
    assert main(["eval", "first-sample", "--bank", "bank.json", "--seed", "5"]) == 0
    return repo


def test_the_residual_check_recomputes_the_score_from_the_map_and_labels(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "residual", "--check"]) == 0
    edit(scored / PIPELINE / "residual.json", lambda d: d["residual"].update(rate=0.5))
    assert main(["eval", "residual", "--check"]) == 1
    assert "residual.rate" in capsys.readouterr().err


def test_the_residual_check_does_not_write(scored: Path) -> None:
    target = scored / PIPELINE / "residual.json"
    edit(target, lambda d: d["residual"].update(rate=0.5))
    before = target.read_text(encoding="utf-8")
    main(["eval", "residual", "--check"])
    assert target.read_text(encoding="utf-8") == before


def test_the_sample_check_redraws_the_sample_from_the_seed_in_its_map(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "sample", "--bank", "bank.json", "--check"]) == 0
    edit_jsonl(scored / PIPELINE / "blind-sample.jsonl", lambda rows: rows[0].update(bullet="x"))
    assert main(["eval", "sample", "--bank", "bank.json", "--check"]) == 1
    assert "bullet" in capsys.readouterr().err


def test_the_sample_check_fails_when_the_map_is_edited(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edit(scored / PIPELINE / "blind-sample-map.json", lambda d: d.update(frame=99))
    assert main(["eval", "sample", "--bank", "bank.json", "--check"]) == 1
    assert "frame" in capsys.readouterr().err


def test_the_first_sample_check_redraws_the_first_rewrites(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "first-sample", "--bank", "bank.json", "--check"]) == 0
    edit_jsonl(scored / PIPELINE / "first-sample.jsonl", lambda rows: rows.pop())
    assert main(["eval", "first-sample", "--bank", "bank.json", "--check"]) == 1
    assert "first-sample" in capsys.readouterr().err


def test_a_seed_given_with_check_is_refused_since_the_map_holds_it(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "sample", "--bank", "bank.json", "--check", "--seed", "6"]) == 2
    assert "--seed" in capsys.readouterr().err
    assert main(["eval", "first-sample", "--bank", "bank.json", "--check", "--size", "1"]) == 2


def test_a_draw_without_a_seed_is_refused(scored: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "first-sample", "--bank", "bank.json"]) == 2
    assert "--seed" in capsys.readouterr().err


def test_the_judge_runs_check_recomputes_the_summary_and_the_bullets_judged(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "judge-runs", "--check"]) == 0
    edit(scored / PIPELINE / "judge-runs.json", lambda d: d["summary"].update(flagged=7))
    assert main(["eval", "judge-runs", "--check"]) == 1
    assert "summary.flagged" in capsys.readouterr().err


def test_the_judge_runs_check_fails_when_a_judged_bullet_is_not_a_bullet_of_the_runs(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edit(scored / PIPELINE / "judge-runs.json", lambda d: d["bullets"][0].update(index=40))
    assert main(["eval", "judge-runs", "--check"]) == 1
    assert "index" in capsys.readouterr().err


def first_scored(root: Path) -> None:
    label_all(root / PIPELINE / "first-sample.jsonl", root / PIPELINE / "first-labels.jsonl")
    calls(root / PIPELINE / "first-sample.jsonl", root / PIPELINE / "first-judgement-calls.md")
    assert main(["eval", "first-rewrites"]) == 0


def test_the_residual_records_how_many_judgement_calls_its_labelling_listed(scored: Path) -> None:
    residual = json.loads((scored / PIPELINE / "residual.json").read_text(encoding="utf-8"))
    assert residual["judgement_calls"] == 1


def test_first_rewrites_records_the_rows_the_passed_sample_also_holds(scored: Path) -> None:
    first_scored(scored)
    document = json.loads((scored / PIPELINE / "first-rewrites.json").read_text(encoding="utf-8"))
    first = rows_of(scored / PIPELINE / "first-sample.jsonl")
    passed = rows_of(scored / PIPELINE / "blind-sample.jsonl")
    repeated = {
        f["id"]
        for f in first
        for p in passed
        if (f["bullet"], f["facts"]) == (p["bullet"], p["facts"])
    }
    assert repeated  # a first-pass bullet is its own first rewrite
    overlap = document["overlap_with_passed_sample"]
    assert (overlap["rows"], overlap["same_labels"]) == (len(repeated), len(repeated))
    assert document["without_overlap"]["rows"] == len(first) - len(repeated)
    assert document["judgement_calls"] == 1


def test_first_rewrites_scores_the_labels_and_the_check_holds_them(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_scored(scored)
    document = json.loads((scored / PIPELINE / "first-rewrites.json").read_text(encoding="utf-8"))
    assert document["violation_rate"]["violating"] == 0
    assert "first rewrites" in capsys.readouterr().out
    assert main(["eval", "first-rewrites", "--check"]) == 0
    edit(scored / PIPELINE / "first-rewrites.json", lambda d: d["gate_off_flag"].update(flagged=9))
    assert main(["eval", "first-rewrites", "--check"]) == 1
    assert "gate_off_flag.flagged" in capsys.readouterr().err


def test_the_first_rewrites_check_passes_before_a_sample_is_drawn_and_says_so(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("first-sample.jsonl", "first-sample-map.json"):
        (scored / PIPELINE / name).unlink()
    assert main(["eval", "first-rewrites", "--check"]) == 0
    assert "no first-rewrite sample" in capsys.readouterr().out


def test_the_first_rewrites_check_fails_when_a_drawn_sample_has_no_labels(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "first-rewrites", "--check"]) == 2
    assert "first-labels.jsonl" in capsys.readouterr().err


def test_the_first_rewrites_check_fails_when_only_one_of_labels_and_score_exists(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_scored(scored)
    (scored / PIPELINE / "first-rewrites.json").unlink()
    assert main(["eval", "first-rewrites", "--check"]) == 2
    first_scored(scored)
    (scored / PIPELINE / "first-labels.jsonl").unlink()
    assert main(["eval", "first-rewrites", "--check"]) == 2


def test_the_judge_check_rescores_the_recorded_verdicts_with_no_model(
    repo: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge_data(repo)
    patch_factory(monkeypatch, lambda: judge_model("99 ms"))
    assert main(["eval", "judge", "--bank", "bank.json"]) == 0
    patch_factory(monkeypatch, lambda: pytest.fail("the check called a model"))
    assert main(["eval", "judge", "--bank", "bank.json", "--check"]) == 0
    edit(
        repo / PIPELINE / "judge-accuracy.json",
        lambda d: d["datasets"]["blind"]["judge"].update(false_alarm_rate=0.9),
    )
    assert main(["eval", "judge", "--bank", "bank.json", "--check"]) == 1
    assert "blind.judge.false_alarm_rate" in capsys.readouterr().err


def test_a_label_file_that_labels_a_row_twice_is_refused(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = scored / PIPELINE / "blind-labels.jsonl"
    first = labels.read_text(encoding="utf-8").splitlines()[0]
    relabel = json.dumps({**json.loads(first), "labels": ["do_not_claim"]})
    labels.write_text(labels.read_text(encoding="utf-8") + relabel + "\n", encoding="utf-8")
    assert main(["eval", "residual"]) == 2
    assert "labelled twice" in capsys.readouterr().err


# ---- --rescore: write what --check computes, from the same inputs ---------------------------


def test_rescore_rewrites_the_pipeline_scores_and_halves_and_nothing_else(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "pipeline", "--halves", "1"]) == 0
    results = scored / PIPELINE / "results.json"
    before = json.loads(results.read_text(encoding="utf-8"))
    edit(results, lambda d: d["scores"]["gate_off"].update(rate=0.0))
    edit(scored / PIPELINE / "halves.json", lambda d: d["first"]["gate_off"].update(rate=0.0))
    assert main(["eval", "pipeline", "--check"]) == 1
    assert main(["eval", "pipeline", "--rescore"]) == 0
    assert main(["eval", "pipeline", "--check"]) == 0
    after = json.loads(results.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "scores"} == {
        k: v for k, v in before.items() if k != "scores"
    }
    assert after["scores"] == before["scores"]


def test_rescore_rewrites_the_judge_runs_summary_from_the_records(scored: Path) -> None:
    edit(scored / PIPELINE / "judge-runs.json", lambda d: d["summary"].update(flagged=7))
    assert main(["eval", "judge-runs", "--rescore"]) == 0
    assert main(["eval", "judge-runs", "--check"]) == 0


def test_rescore_rewrites_the_judge_scores_from_the_recorded_verdicts(
    repo: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge_data(repo)
    patch_factory(monkeypatch, lambda: judge_model("99 ms"))
    assert main(["eval", "judge", "--bank", "bank.json"]) == 0
    patch_factory(monkeypatch, lambda: pytest.fail("the rescore called a model"))
    target = repo / PIPELINE / "judge-accuracy.json"
    edit(target, lambda d: d["datasets"]["blind"]["judge"].update(false_alarm_rate=0.9))
    records = json.loads(target.read_text(encoding="utf-8"))["datasets"]["blind"]["records"]
    assert main(["eval", "judge", "--bank", "bank.json", "--rescore"]) == 0
    assert main(["eval", "judge", "--bank", "bank.json", "--check"]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["datasets"]["blind"]["records"] == records


def test_check_and_rescore_cannot_be_given_together(scored: Path) -> None:
    assert main(["eval", "pipeline", "--check", "--rescore"]) == 2


def test_the_citation_counts_are_written_and_checked_next_to_the_results(
    scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "pipeline", "--bank", "bank.json", "--citations"]) == 0
    target = scored / PIPELINE / "citations.json"
    assert json.loads(target.read_text(encoding="utf-8"))["final_bullets"]["bullets"] == 0
    assert main(["eval", "pipeline", "--bank", "bank.json", "--check"]) == 0
    edit(target, lambda d: d["final_bullets"].update(bullets=4))
    assert main(["eval", "pipeline", "--bank", "bank.json", "--check"]) == 1
    assert "citations.final_bullets.bullets" in capsys.readouterr().err
    assert main(["eval", "pipeline", "--bank", "bank.json", "--rescore"]) == 0
    assert main(["eval", "pipeline", "--bank", "bank.json", "--check"]) == 0
