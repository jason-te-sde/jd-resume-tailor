"""``tailor eval verifier``: writing, the dev-only mode, the gate and the split."""

import json
from pathlib import Path

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.cli import main
from tailor.evals.verifier import assign_split


def setup(tmp_path: Path) -> tuple[Path, Path]:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    source = "Cut p99 latency from 412 ms to 138 ms."
    rows = []
    for n in range(40):
        bad = n % 2 == 1
        rows.append(
            {
                "id": f"r{n}",
                "entry": "acme-swe",
                "sources": [source],
                "bullet": "Cut p99 latency to 90 ms." if bad else "Cut p99 latency to 138 ms.",
                "labels": ["fabricated_number"] if bad else ["ok"],
            }
        )
    (data / "labeled.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    return bank_path, data


def run(
    capsys: pytest.CaptureFixture[str], bank_path: Path, data: Path, *extra: str
) -> tuple[int, str, str]:
    code = main(["eval", "verifier", "--bank", str(bank_path), "--data", str(data), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_split_is_written_then_a_full_run_writes_results(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    assert run(capsys, bank_path, data, "--write-split")[0] == 0
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    assert split["holdout"] == sorted(
        f"r{n}" for n in range(40) if assign_split(f"r{n}", split["seed"]) == "holdout"
    )
    code, out, _ = run(capsys, bank_path, data)
    assert code == 0
    assert "dev (" in out
    assert "holdout (" in out
    results = json.loads((data / "results.json").read_text(encoding="utf-8"))
    assert results["dataset"]["rows"] == 40
    assert results["dataset"]["violation_rows"] == 20
    assert results["dataset"]["labels"] == {
        "ok": 20,
        "fabricated_number": 20,
        "number_drift": 0,
        "fabricated_entity": 0,
        "do_not_claim": 0,
        "escalation_absolute": 0,
        "escalation_causal": 0,
    }
    assert results["holdout"]["classes"]["fabricated_number"]["recall"] in (1.0, None)


def test_the_dev_only_run_prints_dev_and_writes_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    run(capsys, bank_path, data, "--write-split")
    code, out, _ = run(capsys, bank_path, data, "--split", "dev")
    assert code == 0
    assert "dev (" in out
    assert "holdout (" not in out
    assert not (data / "results.json").exists()


def test_check_passes_against_its_own_results_and_fails_when_they_were_better(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    run(capsys, bank_path, data, "--write-split")
    run(capsys, bank_path, data)
    assert run(capsys, bank_path, data, "--check")[0] == 0
    results = json.loads((data / "results.json").read_text(encoding="utf-8"))
    results["holdout"]["classes"]["number_drift"] = {
        "support": 5,
        "predicted": 5,
        "precision": 1.0,
        "recall": 1.0,
    }
    (data / "results.json").write_text(json.dumps(results), encoding="utf-8")
    code, _, err = run(capsys, bank_path, data, "--check")
    assert code == 1
    assert "number_drift recall" in err


def test_a_stale_split_is_refused(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bank_path, data = setup(tmp_path)
    run(capsys, bank_path, data, "--write-split")
    with (data / "labeled.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "new",
                    "entry": "acme-swe",
                    "sources": ["Wrote 23 alerts."],
                    "bullet": "Wrote 23 alerts.",
                    "labels": ["ok"],
                }
            )
            + "\n"
        )
    code, _, err = run(capsys, bank_path, data)
    assert code == 2
    assert "--write-split" in err


def test_a_source_that_is_not_a_fact_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    (data / "labeled.jsonl").write_text(
        json.dumps(
            {"id": "x", "entry": "acme-swe", "sources": ["nope"], "bullet": "b", "labels": ["ok"]}
        )
        + "\n",
        encoding="utf-8",
    )
    code, _, err = run(capsys, bank_path, data)
    assert code == 2
    assert "not a fact" in err


def test_a_missing_dataset_exits_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bank_path, _ = setup(tmp_path)
    code, _, err = run(capsys, bank_path, tmp_path / "empty")
    assert code == 2
    assert "cannot read" in err


def test_rewriting_the_split_keeps_every_side_and_puts_new_rows_in_dev(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    run(capsys, bank_path, data, "--write-split")
    before = json.loads((data / "split.json").read_text(encoding="utf-8"))
    with (data / "labeled.jsonl").open("a", encoding="utf-8") as handle:
        for n in range(60):
            line = {
                "id": f"new{n}",
                "entry": "acme-swe",
                "sources": ["Wrote 23 alerts."],
                "bullet": f"Wrote 23 alerts, take {n}.",
                "labels": ["ok"],
            }
            handle.write(json.dumps(line) + "\n")
    assert run(capsys, bank_path, data, "--write-split")[0] == 0
    after = json.loads((data / "split.json").read_text(encoding="utf-8"))
    assert after["holdout"] == before["holdout"]
    assert set(after["dev"]) == set(before["dev"]) | {f"new{n}" for n in range(60)}
    assert run(capsys, bank_path, data, "--split", "dev")[0] == 0


def test_score_only_scores_every_row_without_a_split_and_writes_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    (data / "labeled.jsonl").rename(data / "rows.jsonl")
    code, out, _ = run(capsys, bank_path, data, "--score-only")
    assert code == 0
    assert "all rows (40 rows)" in out
    assert "dev (" not in out
    assert sorted(path.name for path in data.iterdir()) == ["rows.jsonl"]


def test_score_only_cannot_be_combined_with_the_gate_or_the_split(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, data = setup(tmp_path)
    for flag in ("--check", "--write-split"):
        code, _, err = run(capsys, bank_path, data, "--score-only", flag)
        assert code == 2
        assert "--score-only" in err
