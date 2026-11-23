"""Fitting to one page: the order of the drops, on a scripted builder and on real TeX."""

from dataclasses import replace
from pathlib import Path

import pytest
from renderdata import entry, long_plan, profile

from tailor.render.compile import compile_tex
from tailor.render.fit import (
    MIN_BULLETS,
    Builder,
    Built,
    FitError,
    fit_to_one_page,
    next_drop,
    tex_builder,
)
from tailor.render.layout import LayoutReport
from tailor.render.model import ResumePlan
from tailor.render.report import layout_markdown
from tailor.render.tex import render_tex


def report(pages: int, wrapped: int = 0) -> LayoutReport:
    return LayoutReport(
        pages=pages,
        overfull=0,
        gaps=(),
        wrapped=tuple("x" for _ in range(wrapped)),
        fill=0.9,
        words=100,
        ats_ok=True,
        repeated_openers=(),
        unescaped=(),
    )


def scripted_builder(max_bullets: int, *, wrap_over: int | None = None) -> Builder:
    """A builder whose PDF has two pages while the plan holds more than ``max_bullets`` bullets."""

    def build(plan: ResumePlan) -> Built:
        bullets = sum(len(e.bullets) for e in plan.entries)
        wrapped = 1 if wrap_over is not None and bullets > wrap_over else 0
        pages = 1 if bullets <= max_bullets else 2
        return Built("tex", Path("resume.pdf"), report(pages, wrapped), "pdflatex")

    return build


# Scores: A 9, B 5, C 2 (work); P 7, Q 3, R 1 (projects). Four bullets each.
ORDER = [
    ("R", 4), ("R", 3),
    ("C", 4), ("C", 3),
    ("Q", 4), ("Q", 3),
    ("B", 4), ("B", 3),
    ("P", 4), ("P", 3),
    ("A", 4), ("A", 3),
]  # fmt: skip


def drops_until_nothing_is_left() -> list[tuple[str, int, str]]:
    plan = long_plan()
    out: list[tuple[str, int, str]] = []
    while step := next_drop(plan, "test"):
        plan, drop = step
        out.append((drop.entry_id, drop.position, drop.kind))
    return out


def test_bullets_go_first_lowest_scoring_entry_first_last_bullet_first() -> None:
    sequence = drops_until_nothing_is_left()
    assert [(i, p) for i, p, kind in sequence if kind == "bullet"] == ORDER


def test_no_entry_goes_below_the_minimum_bullets_until_its_bullets_are_the_last_thing_left() -> (
    None
):
    plan = long_plan()
    for _ in ORDER:
        step = next_drop(plan, "test")
        assert step is not None
        plan = step[0]
        assert all(len(e.bullets) >= MIN_BULLETS for e in plan.entries)


def test_entries_go_after_every_bullet_lowest_score_first_and_the_last_work_entry_stays() -> None:
    sequence = drops_until_nothing_is_left()
    entries = [(i, p) for i, p, kind in sequence if kind == "entry"]
    assert entries == [("R", 0), ("C", 0), ("Q", 0), ("B", 0), ("P", 0)]
    assert [kind for *_, kind in sequence] == ["bullet"] * len(ORDER) + ["entry"] * 5


def test_equal_scores_drop_the_later_entry_first() -> None:
    plan = ResumePlan(
        entries=(
            entry("first", "work", 5, ["a", "b", "c"]),
            entry("second", "work", 5, ["a", "b", "c"]),
        )
    )
    step = next_drop(plan, "test")
    assert step is not None
    assert step[1].entry_id == "second"


def test_a_drop_records_the_bullet_its_position_and_the_reason() -> None:
    step = next_drop(long_plan(), "the resume was 2 pages")
    assert step is not None
    new_plan, drop = step
    removed = long_plan().entries[-1].bullets[-1].text
    assert (drop.kind, drop.entry_id, drop.position, drop.text, drop.reason) == (
        "bullet", "R", 4, removed, "the resume was 2 pages",
    )  # fmt: skip
    assert len(new_plan.entries[-1].bullets) == 3


def test_a_plan_that_already_fits_drops_nothing() -> None:
    result = fit_to_one_page(long_plan(), scripted_builder(100))
    assert result.drops == ()
    assert result.plan == long_plan()


def test_drops_stop_at_the_first_plan_that_fits() -> None:
    # 24 bullets; room for 20 means four drops: R4, R3, C4, C3.
    result = fit_to_one_page(long_plan(), scripted_builder(20))
    assert [(d.entry_id, d.position) for d in result.drops] == ORDER[:4]
    assert sum(len(e.bullets) for e in result.plan.entries) == 20
    assert {d.reason for d in result.drops} == {"the resume was 2 pages"}


def test_a_wrapped_bullet_only_counts_when_one_line_bullets_are_asked_for() -> None:
    build = scripted_builder(100, wrap_over=22)
    assert fit_to_one_page(long_plan(), build).drops == ()
    result = fit_to_one_page(long_plan(), build, one_line=True)
    assert len(result.drops) == 2
    assert result.drops[0].reason == "1 bullet ran onto a second line"


def test_when_nothing_can_be_dropped_it_says_so() -> None:
    with pytest.raises(FitError, match="nothing can be dropped"):
        fit_to_one_page(long_plan(), scripted_builder(1))


def test_the_pdf_page_count_unknown_is_an_error_not_a_guess() -> None:
    def build(plan: ResumePlan) -> Built:
        return Built("tex", Path("x.pdf"), replace(report(1), pages=None), "pdflatex")

    with pytest.raises(Exception, match="how many pages"):
        fit_to_one_page(long_plan(), build)


def test_the_layout_section_lists_every_drop_in_order() -> None:
    result = fit_to_one_page(long_plan(), scripted_builder(21))
    text = layout_markdown(result)
    assert "3 dropped to fit one page, in this order (nothing was reworded):" in text
    assert text.index("1. Bullet 4 of Title of R") < text.index("2. Bullet 3 of Title of R")
    assert text.index("2. Bullet 3 of Title of R") < text.index("3. Bullet 4 of Title of C")
    assert "- pages: 1" in text


def test_the_layout_section_says_when_nothing_was_dropped() -> None:
    text = layout_markdown(fit_to_one_page(long_plan(), scripted_builder(100)))
    assert "Nothing was dropped" in text


@pytest.mark.requires_tex
def test_an_overlong_plan_is_fitted_by_dropping_in_the_recorded_order(tmp_path: Path) -> None:
    result = fit_to_one_page(long_plan(), tex_builder(profile(), tmp_path))
    assert result.built.report.pages == 1
    assert 1 <= len(result.drops) <= len(ORDER)
    assert [(d.entry_id, d.position) for d in result.drops] == ORDER[: len(result.drops)]
    # Minimal: one drop fewer would still be over a page.
    plan = long_plan()
    for _ in result.drops[:-1]:
        step = next_drop(plan, "x")
        assert step is not None
        plan = step[0]
    path = tmp_path / "check" / "resume.tex"
    path.parent.mkdir()
    path.write_text(render_tex(plan, profile()), encoding="utf-8")
    assert compile_tex(path).pages == 2


@pytest.mark.requires_tex
def test_no_surviving_bullet_was_reworded(tmp_path: Path) -> None:
    original = {b.text for e in long_plan().entries for b in e.bullets}
    result = fit_to_one_page(long_plan(), tex_builder(profile(), tmp_path))
    assert {b.text for e in result.plan.entries for b in e.bullets} <= original
