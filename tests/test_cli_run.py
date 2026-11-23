"""``tailor run`` and ``tailor render``: the files they write, their exit codes and their errors."""

import json
from pathlib import Path
from typing import Any

import pytest

from tailor.cli import main
from tailor.render.layout import parse_bbox, pdf_bbox
from tailor.render.model import load_plan, load_profile
from tailor.render.tex import render_tex

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = Path("examples/alex-rivera")
PLAN = EXAMPLE / "plans" / "backend-engineer.plan.json"
PROFILE = EXAMPLE / "profile.json"


@pytest.fixture(autouse=True)
def replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TAILOR_CASSETTES", "replay")
    monkeypatch.setenv("TAILOR_CASSETTE_DIR", str(EXAMPLE / "cassettes"))
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")
    monkeypatch.setenv("TAILOR_MODEL", "qwen3:8b")


def run_args(out: Path) -> list[str]:
    return [
        "run",
        "--jd", str(EXAMPLE / "jds" / "backend-engineer.txt"),
        "--bank", str(EXAMPLE / "bank.json"),
        "--profile", str(PROFILE),
        "--out", str(out),
    ]  # fmt: skip


def page_count(pdf: Path) -> int:
    xml = pdf_bbox(pdf)
    assert xml is not None
    return len(parse_bbox(xml))


@pytest.mark.requires_tex
def test_run_replays_the_recorded_plan_into_a_one_page_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(run_args(tmp_path))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "plan.json",
        "provenance.md",
        "resume.pdf",
        "resume.tex",
    ]
    assert page_count(tmp_path / "resume.pdf") == 1
    assert "resume.pdf (1 page" in captured.out
    report = (tmp_path / "provenance.md").read_text(encoding="utf-8")
    assert report.startswith("# Provenance report")
    assert "## Layout" in report
    assert "- pages: 1" in report


@pytest.mark.requires_tex
def test_the_pdf_text_holds_the_bullets_of_the_plan(tmp_path: Path) -> None:
    assert main(run_args(tmp_path)) == 0
    xml = pdf_bbox(tmp_path / "resume.pdf")
    assert xml is not None
    words = {w.text for page in parse_bbox(xml) for w in page.words}
    document = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    first = document["entries"][0]["bullets"][0]["text"].split()
    assert set(first) <= words | {"->"}


def test_run_stops_before_the_model_when_there_is_no_tex_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("TAILOR_CASSETTE_DIR", str(tmp_path / "no-cassettes"))
    code = main(run_args(tmp_path / "out"))
    err = capsys.readouterr().err
    assert code == 2
    assert "no TeX engine found" in err
    assert "pdflatex" in err
    assert not (tmp_path / "out").exists()


def test_run_refuses_a_bad_profile_before_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "profile.json"
    bad.write_text("{}", encoding="utf-8")
    args = run_args(tmp_path / "out")
    args[args.index("--profile") + 1] = str(bad)
    code = main(args)
    assert code == 2
    assert "'name' is required" in capsys.readouterr().err


@pytest.mark.requires_tex
def test_render_makes_the_same_pdf_from_a_plan_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TAILOR_MODEL")
    monkeypatch.delenv("TAILOR_PROVIDER")
    code = main(["render", "--plan", str(PLAN), "--profile", str(PROFILE), "--out", str(tmp_path)])
    assert code == 0
    assert page_count(tmp_path / "resume.pdf") == 1
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8")) == json.loads(
        PLAN.read_text(encoding="utf-8")
    )


def test_render_of_a_plan_without_heading_data_exits_2_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = json.loads(PLAN.read_text(encoding="utf-8"))
    for entry in old["entries"]:
        for key in ("company", "location", "date", "tech", "link"):
            del entry[key]
    plan = tmp_path / "old.json"
    plan.write_text(json.dumps(old), encoding="utf-8")
    code = main(["render", "--plan", str(plan), "--profile", str(PROFILE), "--out", str(tmp_path)])
    assert code == 2
    assert "run `tailor plan` again" in capsys.readouterr().err


def test_render_of_a_missing_plan_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "render",
            "--plan",
            str(tmp_path / "x.json"),
            "--profile",
            str(PROFILE),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.requires_tex
def test_a_plan_that_cannot_be_fitted_exits_1_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document: dict[str, Any] = json.loads(PLAN.read_text(encoding="utf-8"))
    document["entries"] = document["entries"][:1]
    for bullet in document["entries"][0]["bullets"][:2]:
        bullet["text"] = ("word " * 900).strip()
    document["entries"][0]["bullets"] = document["entries"][0]["bullets"][:2]
    plan = tmp_path / "huge.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    code = main(
        ["render", "--plan", str(plan), "--profile", str(PROFILE), "--out", str(tmp_path / "o")]
    )
    assert code == 1
    assert "nothing can be dropped" in capsys.readouterr().err


def test_the_committed_example_tex_is_what_the_renderer_makes_of_the_committed_plan() -> None:
    # The example run dropped nothing, so the committed .tex is the plain rendering of the plan.
    _, plan = load_plan(PLAN)
    tex = render_tex(plan, load_profile(PROFILE))
    assert (EXAMPLE / "out" / "resume.tex").read_text(encoding="utf-8") == tex
    assert "Nothing was dropped" in (EXAMPLE / "out" / "provenance.md").read_text(encoding="utf-8")
