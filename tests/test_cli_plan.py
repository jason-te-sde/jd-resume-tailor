"""``tailor plan``: what it writes, its exit codes, and its errors."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM

from tailor import cli
from tailor.cli import main
from tailor.llm import LLM, Message

EXTRACTION = {
    "title": "Backend Engineer",
    "company": "Northwind",
    "responsibilities": ["Reduce API latency", "Run job queues"],
    "requirements": ["Go"],
    "nice_to_have": [],
    "keywords": ["Go", "PostgreSQL"],
}


def scripted(rewrites: dict[str, list[tuple[str, list[str]]]]) -> ScriptedLLM:
    """A model that extracts, scores both entries, and rewrites each from ``rewrites``."""

    def handler(messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema["properties"]
        if "scores" in properties:
            return {
                "scores": [
                    {"entry": "acme-swe", "score": 9, "covers": [1], "reason": "latency",
                     "bullet_priority": [1, 2]},
                    {"entry": "queue-lab", "score": 8, "covers": [2], "reason": "queues",
                     "bullet_priority": [1]},
                ]
            }  # fmt: skip
        if "bullets" in properties:
            user = messages[-1]["content"]
            entry = "acme-swe" if "Facts you may cite:\n[acme-swe" in user else "queue-lab"
            return {"bullets": [{"text": t, "sources": s} for t, s in rewrites[entry]]}
        return EXTRACTION

    return ScriptedLLM(handler)


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    jd = tmp_path / "jd.txt"
    jd.write_text("Backend Engineer at Northwind. Reduce API latency.", encoding="utf-8")
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text('{"name": "Test Person"}', encoding="utf-8")
    return {"jd": jd, "bank": bank_path, "profile": profile, "out": tmp_path / "out" / "plan.json"}


def plan_args(files: dict[str, Path], *extra: str) -> list[str]:
    return [
        "plan", "--jd", str(files["jd"]), "--bank", str(files["bank"]),
        "--profile", str(files["profile"]), "--out", str(files["out"]), *extra,
    ]  # fmt: skip


def patch_llm(monkeypatch: pytest.MonkeyPatch, llm: LLM) -> None:
    monkeypatch.setattr(cli, "build_llm", lambda config, **kwargs: llm)
    monkeypatch.setenv("TAILOR_MODEL", "test-model")
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")


def test_a_plan_and_a_provenance_report_are_written_and_the_exit_is_zero(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [("Cut p99 latency from 412 ms to 138 ms in a load test.", ["acme-swe#1"])]
    bad = "Cut p99 latency to 99 ms."
    llm = scripted(
        {
            "acme-swe": [*good, ("Wrote 23 alerts.", ["acme-swe#2"])],
            "queue-lab": [(bad, ["queue-lab#1"])],
        }
    )
    patch_llm(monkeypatch, llm)
    assert main(plan_args(files)) == 0
    plan = json.loads(files["out"].read_text(encoding="utf-8"))
    assert plan["job"]["title"] == "Backend Engineer"
    assert plan["model"] == {"provider": "openai", "model": "test-model"}
    assert plan["selection"]["picked"] == ["acme-swe", "queue-lab"]
    assert plan["profile"] == "profile.json"
    work, project = plan["entries"]
    assert [b["outcome"] for b in work["bullets"]] == ["first_pass", "first_pass"]
    assert work["bullets"][0]["sources"] == ["acme-swe#1"]
    assert work["bullets"][0]["verdict"] == "ok"
    (fallback,) = project["bullets"]
    assert fallback["outcome"] == "fallback"
    assert fallback["text"] == "Processed 9,200 jobs/s in a 3-node test."
    assert "fabricated_number" in fallback["reason"]
    assert plan["summary"] == {
        "bullets": 3, "first_pass": 2, "after_retry": 0, "fallback": 1, "verbatim": 0,
    }  # fmt: skip
    assert plan["usage"]["calls"] == 5
    report = files["out"].with_suffix(".md").read_text(encoding="utf-8")
    assert "2 passed the verifier first time" in report
    assert "1 fell back" in report
    assert "why the bank wording was used" in report
    assert "wrote" in capsys.readouterr().out


def test_no_model_configured_is_an_error_that_says_how_to_set_one(
    files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TAILOR_MODEL", raising=False)
    monkeypatch.delenv("TAILOR_PROVIDER", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(plan_args(files)) == 2
    assert "TAILOR_MODEL" in capsys.readouterr().err
    assert not files["out"].exists()


def test_an_unreadable_job_description_is_an_error(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_llm(monkeypatch, scripted({}))
    files["jd"].unlink()
    assert main(plan_args(files)) == 2
    assert "cannot read" in capsys.readouterr().err


def test_replay_with_no_cassette_fails_and_names_the_hash(
    files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TAILOR_MODEL", "m")
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")
    monkeypatch.setenv("TAILOR_CASSETTES", "replay")
    monkeypatch.setenv("TAILOR_CASSETTE_DIR", str(tmp_path / "empty"))
    assert main(plan_args(files)) == 2
    assert "no cassette for request" in capsys.readouterr().err


def test_the_anthropic_provider_without_a_key_ends_in_a_one_line_error(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TAILOR_CASSETTES", raising=False)
    monkeypatch.setenv("TAILOR_MODEL", "claude-haiku-5-5")
    assert main(plan_args(files, "--provider", "anthropic")) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "ANTHROPIC_API_KEY" in err
    assert len(err.strip().splitlines()) == 1
    assert not files["out"].exists()


def test_an_out_path_ending_in_md_does_not_let_the_report_overwrite_the_plan(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = [("Cut p99 latency from 412 ms to 138 ms in a load test.", ["acme-swe#1"])]
    llm = scripted({"acme-swe": good, "queue-lab": [("Wrote 23 alerts.", ["queue-lab#1"])]})
    patch_llm(monkeypatch, llm)
    out = files["out"].with_name("plan.md")
    args = plan_args(files)
    args[args.index(str(files["out"]))] = str(out)
    assert main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["job"]["title"] == "Backend Engineer"
    report = out.with_name("plan.report.md")
    assert "passed the verifier" in report.read_text(encoding="utf-8")


def test_the_seed_flag_reaches_the_model_configuration(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int | None] = []
    llm = scripted({"acme-swe": [("Cut p99 latency.", ["acme-swe#1"])] * 2, "queue-lab": []})

    def build(config: Any, **kwargs: Any) -> LLM:
        seen.append(config.seed)
        return llm

    monkeypatch.setattr(cli, "build_llm", build)
    monkeypatch.setenv("TAILOR_MODEL", "test-model")
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")
    main(plan_args(files, "--seed", "42"))
    assert seen == [42]


def test_a_seed_for_a_provider_without_one_is_warned_about_on_stderr(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted({"acme-swe": [("Cut p99 latency.", ["acme-swe#1"])] * 2, "queue-lab": []})
    patch_llm(monkeypatch, llm)
    main(plan_args(files, "--provider", "anthropic", "--seed", "42"))
    assert "anthropic provider takes no seed, so seed 42" in capsys.readouterr().err
    main(plan_args(files, "--seed", "42"))
    assert "takes no seed" not in capsys.readouterr().err
