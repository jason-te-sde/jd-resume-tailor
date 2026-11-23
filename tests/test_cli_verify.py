"""``tailor verify``: exit codes, output formats and error messages."""

import json
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank

from tailor.cli import main

GOOD = {"text": "Reduced p99 latency from 412 ms to 138 ms.", "entry": "acme-swe"}
BAD = {"text": "Always cut p99 latency to 90 ms.", "entry": "acme-swe"}


def files(tmp_path: Path, bullets: Any, document: Any = None) -> tuple[Path, Path]:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(document or bank(WORK, PROJECT)), encoding="utf-8")
    bullets_path = tmp_path / "bullets.json"
    bullets_path.write_text(json.dumps(bullets), encoding="utf-8")
    return bank_path, bullets_path


def run(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, bullets: Any, *extra: str
) -> tuple[int, str, str]:
    bank_path, bullets_path = files(tmp_path, bullets)
    code = main(["verify", "--bank", str(bank_path), "--bullets", str(bullets_path), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_all_bullets_ok_exits_zero(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out, err = run(capsys, tmp_path, [GOOD])
    assert code == 0, err
    assert err == ""
    assert "all bullets ok" in out


def test_a_violation_exits_one_and_the_report_says_which(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = run(capsys, tmp_path, [GOOD, BAD])
    assert code == 1
    assert "escalation_absolute" in out
    assert "fabricated_number" in out
    assert "1 ok, 1 with violations" in out


def test_json_format_is_valid_json_with_the_same_verdict(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = run(capsys, tmp_path, [GOOD, BAD], "--format", "json")
    assert code == 1
    document = json.loads(out)
    assert document["ok"] is False
    assert [b["status"] for b in document["bullets"]] == ["ok", "violation"]


def test_markdown_is_the_default_and_md_is_accepted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    default = run(capsys, tmp_path, [GOOD])[1]
    explicit = run(capsys, tmp_path, [GOOD], "--format", "md")[1]
    assert default == explicit
    assert default.startswith("# Verification report")


def test_rounding_nearest_relaxes_the_number_rule(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    project = {**PROJECT, "facts": [{"id": "recall", "text": "Reached a recall of 0.875"}]}
    bullets = [{"text": "Reached a recall of 88%.", "entry": "queue-lab"}]
    bank_path, bullets_path = files(tmp_path, bullets, bank(project))
    args = ["verify", "--bank", str(bank_path), "--bullets", str(bullets_path)]
    assert main(args) == 1
    capsys.readouterr()
    assert main([*args, "--rounding", "conservative"]) == 1
    capsys.readouterr()
    assert main([*args, "--rounding", "nearest"]) == 0


def test_bullets_may_cite_specific_facts(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bullets = [
        {"text": "Cut p99 latency to 138 ms.", "entry": "acme-swe", "sources": ["acme-swe#2"]}
    ]
    code, out, _ = run(capsys, tmp_path, bullets)
    assert code == 1
    assert "fabricated_number" in out


def test_an_unknown_entry_exits_two_with_a_message(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, err = run(capsys, tmp_path, [{"text": "x", "entry": "nope"}])
    assert code == 2
    assert out == ""
    assert "bullet 1" in err
    assert "nope" in err


def test_a_malformed_bullets_file_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, _, err = run(capsys, tmp_path, [{"entry": "acme-swe"}])
    assert code == 2
    assert 'missing "text"' in err


def test_a_missing_bullets_file_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path, _ = files(tmp_path, [])
    code = main(["verify", "--bank", str(bank_path), "--bullets", str(tmp_path / "none.json")])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_an_invalid_bank_exits_two_and_names_the_problems(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"experiences": [{"id": "x"}]}), encoding="utf-8")
    bullets_path = tmp_path / "bullets.json"
    bullets_path.write_text("[]", encoding="utf-8")
    code = main(["verify", "--bank", str(bank_path), "--bullets", str(bullets_path)])
    assert code == 2
    assert "invalid fact bank" in capsys.readouterr().err


def test_a_missing_bank_exits_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    _, bullets_path = files(tmp_path, [])
    code = main(["verify", "--bank", str(tmp_path / "none.json"), "--bullets", str(bullets_path)])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_missing_arguments_and_bad_choices_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify"]) == 2
    assert main(["verify", "--bank", "b", "--bullets", "x", "--rounding", "loose"]) == 2
    assert main(["verify", "--bank", "b", "--bullets", "x", "--format", "html"]) == 2
    capsys.readouterr()


def test_the_help_documents_the_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--rounding" in out
    assert "Exit status" in out


def test_an_empty_list_of_bullets_is_ok(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out, _ = run(capsys, tmp_path, [])
    assert code == 0
    assert "0 bullets" in out
