import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from bankdata import NO_CODE, PROJECT, WORK, bank

import tailor
from tailor.cli import main


def write_bank(tmp_path: Path, document: Any, name: str = "bank.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def summary_lines(out: str) -> dict[str, str]:
    """Map each label of a summary line to the rest of the line."""
    lines: dict[str, str] = {}
    for line in out.splitlines():
        for label in ("entries", "candidates", "facts", "do not claim", "same project", "warnings"):
            if line.startswith(label):
                lines[label] = line[len(label) :].strip()
    return lines


def test_a_valid_bank_exits_zero_and_prints_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_bank(tmp_path, bank(WORK, PROJECT, NO_CODE))
    code, out, err = run(capsys, "bank", "check", str(path))
    assert code == 0, err
    assert err == ""
    assert out.splitlines()[0] == f"ok: {path}"
    lines = summary_lines(out)
    assert lines["entries"] == "3 (work: 1, project: 2)"
    assert lines["candidates"] == "2 (work: 1, project with code: 1)"
    # WORK: 2 bullets + scope; PROJECT: 1 bullet + scope; NO_CODE: 1 bullet, no scope.
    assert lines["facts"] == "6 (4 from bullets, 2 from scope, 0 explicit)"
    assert lines["do not claim"] == "3 claims in 2 entries"
    assert lines["same project"] == "none"
    assert lines["warnings"] == "2"


def test_explicit_facts_are_counted_separately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = {**PROJECT, "facts": [{"id": "f1", "text": "a"}, {"id": "f2", "text": "b"}]}
    code, out, _ = run(capsys, "bank", "check", str(write_bank(tmp_path, bank(project))))
    assert code == 0
    assert summary_lines(out)["facts"] == "2 (0 from bullets, 0 from scope, 2 explicit)"


def test_same_project_groups_are_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sibling = {**NO_CODE, "id": "queue-lab-bench", "scope": "Same project as queue-lab."}
    path = write_bank(tmp_path, bank(PROJECT, sibling))
    code, out, _ = run(capsys, "bank", "check", str(path))
    assert code == 0
    assert summary_lines(out)["same project"] == "1 group"
    assert "  queue-lab, queue-lab-bench" in out.splitlines()


def test_warnings_are_listed_with_their_paths_and_do_not_fail_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bare = {k: v for k, v in PROJECT.items() if k != "description"}
    code, out, _ = run(capsys, "bank", "check", str(write_bank(tmp_path, bank(WORK, bare))))
    assert code == 0
    assert summary_lines(out)["warnings"] == "1"
    assert "  $.experiences[1].description: project 'queue-lab' has no description" in out


def test_a_bank_without_warnings_says_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(capsys, "bank", "check", str(write_bank(tmp_path, bank(WORK, PROJECT))))
    assert code == 0
    assert summary_lines(out)["warnings"] == "none"


def test_an_invalid_bank_exits_one_and_lists_every_problem_with_its_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = bank(WORK, PROJECT)
    del document["experiences"][0]["id"]
    document["experiences"][1]["bullet_capabilities"] = ["a", "b"]
    path = write_bank(tmp_path, document)
    code, out, err = run(capsys, "bank", "check", str(path))
    assert code == 1
    assert out == "", "nothing but the problems on a failure"
    assert f"invalid fact bank in {path}: 2 problems" in err
    assert "$.experiences[0].id: required property is missing" in err
    assert "$.experiences[1].bullet_capabilities: has 2 items but there are 1 bullets" in err


def test_invalid_json_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bank.json"
    path.write_text('{"experiences": [', encoding="utf-8")
    code, out, err = run(capsys, "bank", "check", str(path))
    assert code == 1
    assert out == ""
    assert "not valid JSON" in err


def test_an_unreadable_file_exits_two_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(capsys, "bank", "check", str(tmp_path / "missing.json"))
    assert code == 2
    assert out == ""
    assert "missing.json" in err
    assert "No such file" in err


def test_a_directory_is_an_unreadable_file_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(capsys, "bank", "check", str(tmp_path))
    assert code == 2
    assert str(tmp_path) in err


@pytest.mark.parametrize("argv", [[], ["bank"], ["bank", "check"], ["nonsense"]])
def test_a_usage_error_exits_two(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, *argv)
    assert code == 2
    assert out == ""
    assert "usage:" in err


def test_version_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--version")
    assert code == 0
    assert out.strip() == f"tailor {tailor.__version__}"


def test_help_documents_the_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "bank", "check", "--help")
    assert code == 0
    assert "exit status" in out.lower()


def test_the_module_entry_point_sets_the_exit_status(tmp_path: Path) -> None:
    good = write_bank(tmp_path, bank(WORK, PROJECT), "good.json")
    bad = write_bank(tmp_path, {"experiences": []}, "bad.json")
    ok = subprocess.run(
        [sys.executable, "-m", "tailor", "bank", "check", str(good)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.startswith("ok: ")
    failed = subprocess.run(
        [sys.executable, "-m", "tailor", "bank", "check", str(bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert "$.experiences" in failed.stderr
