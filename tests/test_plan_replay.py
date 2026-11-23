"""The recorded local-model run of ``tailor plan``, replayed end to end with no network.

The cassettes in examples/alex-rivera/cassettes were recorded against a local model on the example
bank and examples/alex-rivera/jds/backend-engineer.txt. Replaying them must give the committed plan,
and every bullet the plan keeps must pass the verifier when it is checked again from scratch.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tailor.bank import load_bank
from tailor.cli import main
from tailor.verify import Bullet, verify

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = Path("examples/alex-rivera")


@pytest.fixture
def replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TAILOR_CASSETTES", "replay")
    monkeypatch.setenv("TAILOR_CASSETTE_DIR", str(EXAMPLE / "cassettes"))
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")
    monkeypatch.setenv("TAILOR_MODEL", "qwen3:8b")
    out = tmp_path / "plan.json"
    code = main(
        [
            "plan",
            "--jd", str(EXAMPLE / "jds" / "backend-engineer.txt"),
            "--bank", str(EXAMPLE / "bank.json"),
            "--profile", str(EXAMPLE / "profile.json"),
            "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().err
    document: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    document["markdown"] = out.with_suffix(".md").read_text(encoding="utf-8")
    return document


def test_replay_gives_the_committed_plan(replayed: dict[str, Any]) -> None:
    committed = json.loads(
        (EXAMPLE / "plans" / "backend-engineer.plan.json").read_text(encoding="utf-8")
    )
    markdown = (EXAMPLE / "plans" / "backend-engineer.plan.md").read_text(encoding="utf-8")
    assert replayed.pop("markdown") == markdown
    for plan in (replayed, committed):
        plan["usage"].pop("replayed_calls")
    assert replayed == committed


def test_the_plan_reports_how_each_bullet_came_out(replayed: dict[str, Any]) -> None:
    summary = replayed["summary"]
    assert summary["bullets"] == sum(len(e["bullets"]) for e in replayed["entries"])
    assert summary["bullets"] == (
        summary["first_pass"] + summary["after_retry"] + summary["fallback"] + summary["verbatim"]
    )
    assert (
        replayed["usage"]["replayed_calls"]
        == replayed["usage"]["calls"]
        >= 2 + len(replayed["entries"])
    )
    assert 2 <= len([e for e in replayed["entries"] if e["type"] == "work"]) <= 3
    assert 2 <= len([e for e in replayed["entries"] if e["type"] == "project"]) <= 3


def test_every_kept_bullet_passes_the_verifier_again(replayed: dict[str, Any]) -> None:
    bank = load_bank(EXAMPLE / "bank.json")
    rewritten = [
        Bullet(b["text"], e["id"], tuple(b["sources"]))
        for e in replayed["entries"]
        for b in e["bullets"]
        if b["outcome"] in ("first_pass", "after_retry")
    ]
    assert rewritten
    assert verify(bank, rewritten).ok


def test_a_fallback_is_the_banks_own_bullet_with_a_reason(replayed: dict[str, Any]) -> None:
    fallbacks = [b for e in replayed["entries"] for b in e["bullets"] if b["outcome"] == "fallback"]
    assert fallbacks
    for bullet in fallbacks:
        assert bullet["text"] == bullet["original"]
        assert bullet["verdict"] == "bank_original"
        assert bullet["reason"]
        assert len(bullet["attempts"]) == 2


def test_the_selection_never_lists_both_halves_of_one_project(replayed: dict[str, Any]) -> None:
    picked = set(replayed["selection"]["picked"])
    assert not {"gridlock", "gridlock-bench"} <= picked
