"""The committed measurement is internally consistent: every scored file is what its inputs give."""

import json
from pathlib import Path

import pytest

from tailor.bank import load_bank
from tailor.cli import main
from tailor.evals.pipeline import check_document, halves_document, replay_subset
from tailor.llm import LLMConfig

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "eval" / "pipeline"


def results() -> dict:  # type: ignore[type-arg]
    return json.loads((PIPELINE / "results.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_the_committed_scores_are_what_the_committed_runs_give() -> None:
    assert check_document(results()) == []


def test_the_committed_halves_are_what_the_committed_runs_give() -> None:
    halves = json.loads((PIPELINE / "halves.json").read_text(encoding="utf-8"))
    assert halves == halves_document(results(), halves["split_after"])


def test_the_results_are_from_a_clean_tree_and_carry_no_machine_path() -> None:
    text = (PIPELINE / "results.json").read_text(encoding="utf-8")
    assert results()["meta"]["dirty"] is False
    assert "/Users/" not in text
    assert "/private/" not in text


def test_a_committed_run_replays_from_the_committed_cassettes() -> None:
    document = results()
    problems = replay_subset(
        document,
        bank=load_bank(ROOT / "examples" / "alex-rivera" / "bank.json"),
        bank_path=ROOT / "examples" / "alex-rivera" / "bank.json",
        jds_dir=PIPELINE / "jds",
        cassette_dir=PIPELINE / "cassettes",
        jd="ios-engineer",
        seed=1,
        config=LLMConfig(document["meta"]["provider"], document["meta"]["model"]),
    )
    assert problems == []


@pytest.mark.parametrize(
    "command",
    [
        ["judge", "--check"],
        ["judge-runs", "--check"],
        ["sample", "--check"],
        ["first-sample", "--check"],
        ["residual", "--check"],
        ["first-rewrites", "--check"],
    ],
    ids=lambda command: command[0],
)
def test_every_scored_file_is_what_its_committed_inputs_give(
    command: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(ROOT)
    assert main(["eval", *command]) == 0, capsys.readouterr().err
