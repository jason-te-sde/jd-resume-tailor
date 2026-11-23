"""The layout checks: on bounding boxes built by hand, and on small PDFs when TeX is installed."""

import shutil
from pathlib import Path

import pytest
from renderdata import LINE, entry, profile

from tailor.render.compile import (
    INSTALL_HINT,
    NoTeXEngineError,
    compile_tex,
    extract_errors,
    find_engine,
)
from tailor.render.layout import (
    BOTTOM_MARGIN_PT,
    bullet_layout,
    check_layout,
    find_unescaped,
    overfull_count,
    page_fill,
    pdf_bbox,
    repeated_openers,
)
from tailor.render.model import ResumePlan
from tailor.render.tex import render_tex


def bbox(*lines: tuple[float, float, str], height: float = 792.0) -> str:
    """``pdftotext -bbox`` output for lines given as ``(x, baseline y, text)``; a word per token."""
    words = []
    for x, y, text in lines:
        for n, token in enumerate(text.split()):
            left = x + 40 * n
            words.append(
                f'<word xMin="{left}" yMin="{y - 10}" xMax="{left + 30}" yMax="{y}">{token}</word>'
            )
    body = "\n".join(words)
    return f'<doc><page width="612.0" height="{height}">\n{body}\n</page></doc>'


BULLET = "\u2022"


class TestBulletLayout:
    def test_two_bullets_one_line_apart_are_clean(self) -> None:
        xml = bbox((70, 100, f"{BULLET} first bullet"), (70, 112, f"{BULLET} second bullet"))
        assert bullet_layout(xml) == ([], [])

    def test_an_empty_row_between_two_bullets_is_a_gap(self) -> None:
        xml = bbox((70, 100, f"{BULLET} first bullet"), (70, 126, f"{BULLET} second bullet"))
        assert bullet_layout(xml) == (["\u2022 first bullet"], [])

    def test_a_line_at_the_text_indent_below_a_bullet_is_a_wrap(self) -> None:
        xml = bbox(
            (70, 100, f"{BULLET} first bullet"),
            (80, 112, "runs on"),
            (70, 124, f"{BULLET} second bullet"),
        )
        assert bullet_layout(xml) == ([], ["first bullet"])

    def test_a_heading_after_the_last_bullet_is_neither(self) -> None:
        xml = bbox((70, 100, f"{BULLET} only bullet"), (30, 112, "Next Heading"))
        assert bullet_layout(xml) == ([], [])

    def test_a_far_away_line_after_a_bullet_is_not_a_wrap(self) -> None:
        xml = bbox((70, 100, f"{BULLET} only bullet"), (80, 160, "something else"))
        assert bullet_layout(xml) == ([], [])


class TestPageFill:
    def test_fill_is_the_lowest_line_over_the_usable_height(self) -> None:
        usable = 792.0 - BOTTOM_MARGIN_PT
        fill = page_fill(bbox((70, 100, "top"), (70, usable / 2, "middle")))
        assert fill == pytest.approx(0.5)

    def test_no_text_means_no_fill(self) -> None:
        assert page_fill('<doc><page width="612.0" height="792.0">\n</page></doc>') is None


class TestUnescaped:
    def test_flags_each_special_character_without_a_backslash(self) -> None:
        tex = "\\begin{document}\nSales & R&D at 100% of cost_a, #x and $5\n"
        assert {u.char for u in find_unescaped(tex)} == {"&", "%", "_", "#", "$"}

    def test_lets_escaped_characters_through(self) -> None:
        tex = "\\begin{document}\nSales \\& R\\&D at 100\\% of cost\\_a, \\#1 and \\$5\n"
        assert find_unescaped(tex) == []

    def test_lets_math_comments_macros_tables_and_urls_through(self) -> None:
        tex = (
            "\\newcommand{\\a}[1]{#1 & _}\n\\begin{document}\n% 50% off & more\n"
            "$a_b$ and $\\rightarrow$\n\\href{https://example.com/a_b}{link}\n"
            "\\begin{tabular}{ll} a & b \\\\ \\end{tabular}\n"
        )
        assert find_unescaped(tex) == []

    def test_an_escaped_hash_does_not_hide_a_raw_ampersand_on_the_same_line(self) -> None:
        tex = "\\begin{document}\nthe \\#1 endpoint & more\n"
        assert [u.char for u in find_unescaped(tex)] == ["&"]

    def test_reports_the_line_number_in_the_whole_file(self) -> None:
        tex = "line 1\n\\begin{document}\nfine\nbad & line\n"
        assert [u.line for u in find_unescaped(tex)] == [4]


def test_overfull_boxes_are_counted_from_the_log() -> None:
    log = "Overfull \\hbox (3.0pt too wide) in paragraph\nUnderfull \\hbox\nOverfull \\hbox (1pt)\n"
    assert overfull_count(log) == 2


class TestRepeatedOpeners:
    def test_two_bullets_of_one_entry_with_the_same_first_word(self) -> None:
        plan = ResumePlan(entries=(entry("a", "work", 5, ["Built a", "built b", "Ran c"]),))
        found = repeated_openers(plan)
        assert [(r.word, r.count) for r in found] == [("Built", 2)]

    def test_the_same_word_in_two_entries_is_fine(self) -> None:
        plan = ResumePlan(
            entries=(entry("a", "work", 5, ["Built a"]), entry("b", "project", 5, ["Built b"]))
        )
        assert repeated_openers(plan) == []


class TestExtractErrors:
    def test_takes_each_bang_line_with_its_context(self) -> None:
        log = (
            "x\n! Undefined control sequence.\nl.12 \\foo\n   bar\n"
            "! Missing $ inserted.\nl.30 a_b\n"
        )
        errors = extract_errors(log)
        assert len(errors) == 2
        assert errors[0].startswith("! Undefined control sequence.")
        assert "l.12" in errors[0]

    def test_a_log_with_no_bang_line_falls_back_to_markers(self) -> None:
        assert extract_errors("ok\nEmergency stop at line 3\n") == ["Emergency stop at line 3"]

    def test_an_empty_log_still_says_something(self) -> None:
        assert extract_errors("") == ["LaTeX failed without an error message; see the log."]


def test_no_engine_gives_the_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(NoTeXEngineError) as error:
        find_engine()
    assert str(error.value) == INSTALL_HINT
    assert "pdflatex" in INSTALL_HINT


def test_pdftotext_missing_skips_its_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert pdf_bbox(tmp_path / "x.pdf") is None


# Everything below compiles real documents.


def _compile(tmp_path: Path, plan: ResumePlan) -> tuple[str, Path, str, int | None]:
    tex = render_tex(plan, profile())
    path = tmp_path / "resume.tex"
    path.write_text(tex, encoding="utf-8")
    result = compile_tex(path)
    assert result.ok, result.errors
    assert result.pdf is not None
    return tex, result.pdf, result.log, result.pages


SHORT = ["Cut p99 latency from 412 ms to 138 ms.", "Built a Kafka consumer for status events."]


@pytest.mark.requires_tex
def test_a_short_plan_is_one_page_with_no_wraps_and_no_gaps(tmp_path: Path) -> None:
    plan = ResumePlan(entries=(entry("a", "work", 5, SHORT), entry("p", "project", 4, SHORT)))
    tex, pdf, log, pages = _compile(tmp_path, plan)
    report = check_layout(tex, pdf, log, plan, pages=pages)
    assert report.pages == 1
    assert report.one_page
    assert report.gaps == ()
    assert report.wrapped == ()
    assert report.overfull == 0
    assert report.ats_ok is True
    assert report.unescaped == ()
    assert report.skipped == ()
    assert report.fill is not None
    assert 0.2 < report.fill < 0.8


@pytest.mark.requires_tex
def test_a_long_bullet_is_reported_as_wrapped(tmp_path: Path) -> None:
    long = LINE + "the hot path of the service so that every request finished well inside budget"
    plan = ResumePlan(entries=(entry("a", "work", 5, [long, SHORT[0]]),))
    tex, pdf, log, pages = _compile(tmp_path, plan)
    report = check_layout(tex, pdf, log, plan, pages=pages)
    assert report.wrapped is not None
    assert len(report.wrapped) == 1
    assert report.wrapped[0].startswith("Reduced p99 latency")


@pytest.mark.requires_tex
def test_more_than_a_page_is_reported_as_two_pages(tmp_path: Path) -> None:
    many = ResumePlan(
        entries=tuple(entry(f"e{n}", "work", 5, [LINE + "x", LINE + "y"]) for n in range(14))
    )
    tex, pdf, log, pages = _compile(tmp_path, many)
    assert pages == 2
    assert check_layout(tex, pdf, log, many, pages=None).pages == 2


@pytest.mark.requires_tex
def test_a_latex_error_comes_back_as_errors_not_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "bad.tex"
    path.write_text("\\documentclass{article}\n\\begin{document}\n\\nosuchmacro\n\\end{document}\n")
    result = compile_tex(path)
    assert not result.ok
    assert result.pdf is None
    assert any("Undefined control sequence" in e for e in result.errors)


@pytest.mark.requires_tex
@pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex not installed")
def test_a_unicode_error_retries_with_xelatex(tmp_path: Path) -> None:
    text = "Shipped to \U0001f680 and \u6771\u4eac"
    plan = ResumePlan(entries=(entry("a", "work", 5, [text, SHORT[0]]),))
    tex = render_tex(plan, profile())
    path = tmp_path / "resume.tex"
    path.write_text(tex, encoding="utf-8")
    first = compile_tex(path, engine="pdflatex")
    assert not first.ok
    result = compile_tex(path)
    assert result.retried
    assert result.engine == "xelatex"


@pytest.mark.requires_tex
def test_the_aux_files_are_cleaned_up(tmp_path: Path) -> None:
    plan = ResumePlan(entries=(entry("a", "work", 5, SHORT),))
    _compile(tmp_path, plan)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["resume.pdf", "resume.tex"]


@pytest.mark.requires_tex
def test_the_golden_document_compiles_to_one_page_with_clean_checks(tmp_path: Path) -> None:
    golden = Path(__file__).parent / "data" / "render" / "golden.tex"
    path = tmp_path / "resume.tex"
    path.write_text(golden.read_text(encoding="utf-8"), encoding="utf-8")
    result = compile_tex(path)
    assert result.ok, result.errors
    assert result.pages == 1
    assert overfull_count(result.log) == 0
