"""Fit a plan to one page by dropping, in a fixed order, and say what was dropped.

Dropping is the only thing done: no bullet is shortened or reworded, because a reworded bullet
would not have passed the verifier. The order is deterministic, so the same plan and the same TeX
give the same resume:

1. The last bullet of an entry (the selection ranked bullets, most useful first) that still has
   more than ``MIN_BULLETS``; among those entries, the one with the lowest coverage score, and on
   a tie the one later in the plan.
2. When no entry can lose a bullet, a whole entry: the lowest coverage score, later in the plan
   on a tie. The last work entry is never dropped, and a resume without one does not render.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

from tailor.render.compile import compile_tex
from tailor.render.layout import LayoutReport, check_layout
from tailor.render.model import PlanEntry, Profile, RenderError, ResumePlan
from tailor.render.tex import render_tex

MIN_BULLETS: Final = 2
TEX_NAME: Final = "resume.tex"


class FitError(RenderError):
    """The resume does not fit one page even after every allowed drop."""


@dataclass(frozen=True, slots=True)
class Drop:
    """One thing taken off the resume, and why.

    Attributes:
        kind: ``bullet`` or ``entry``.
        entry_id: The entry the bullet belonged to, or the entry dropped.
        entry_title: Its title.
        position: For a bullet, its 1-based place in the entry before the drop; 0 for an entry.
        text: The bullet's text; empty for an entry.
        reason: What was wrong with the resume that forced this drop.
    """

    kind: Literal["bullet", "entry"]
    entry_id: str
    entry_title: str
    position: int
    text: str
    reason: str


@dataclass(frozen=True, slots=True)
class Built:
    """A document built and checked: the source, the PDF, the checks and the engine."""

    tex: str
    pdf: Path
    report: LayoutReport
    engine: str
    retried: bool = False


@dataclass(frozen=True, slots=True)
class FitResult:
    """The resume that fits, the plan it was built from and every drop on the way."""

    plan: ResumePlan
    built: Built
    drops: tuple[Drop, ...]


Builder = Callable[[ResumePlan], Built]


def next_drop(plan: ResumePlan, reason: str) -> tuple[ResumePlan, Drop] | None:
    """Take the next bullet or entry off ``plan`` and return the new plan and what was dropped.

    ``None`` when nothing may be dropped any more.
    """
    shrinkable = [i for i, e in enumerate(plan.entries) if len(e.bullets) > MIN_BULLETS]
    if shrinkable:
        index = min(shrinkable, key=lambda i: (plan.entries[i].score, -i))
        entry = plan.entries[index]
        gone = entry.bullets[-1]
        entries = list(plan.entries)
        entries[index] = replace(entry, bullets=entry.bullets[:-1])
        drop = Drop("bullet", entry.id, entry.title, len(entry.bullets), gone.text, reason)
        return replace(plan, entries=tuple(entries)), drop
    work = sum(e.type == "work" for e in plan.entries)
    removable = [i for i, e in enumerate(plan.entries) if e.type == "project" or work > 1]
    if not removable:
        return None
    index = min(removable, key=lambda i: (plan.entries[i].score, -i))
    gone_entry: PlanEntry = plan.entries[index]
    entries = [e for i, e in enumerate(plan.entries) if i != index]
    drop = Drop("entry", gone_entry.id, gone_entry.title, 0, "", reason)
    return replace(plan, entries=tuple(entries)), drop


def _problem(report: LayoutReport, *, one_line: bool) -> str | None:
    if report.pages is None:
        raise RenderError("cannot tell how many pages the PDF has")
    if report.pages > 1:
        return f"the resume was {report.pages} pages"
    if one_line and report.wrapped:
        count = len(report.wrapped)
        return f"{count} bullet{'s' if count != 1 else ''} ran onto a second line"
    return None


def fit_to_one_page(plan: ResumePlan, build: Builder, *, one_line: bool = False) -> FitResult:
    """Build ``plan`` and drop until the PDF is one page (and, with ``one_line``, no bullet wraps).

    Each drop is followed by a rebuild, so every drop is justified by the page the previous build
    produced.

    Raises:
        FitError: when the limit is reached with nothing left that may be dropped.
    """
    drops: list[Drop] = []
    while True:
        built = build(plan)
        problem = _problem(built.report, one_line=one_line)
        if problem is None:
            return FitResult(plan, built, tuple(drops))
        step = next_drop(plan, problem)
        if step is None:
            raise FitError(
                f"{problem}, and nothing can be dropped: every entry is at its minimum "
                f"of {MIN_BULLETS} bullets and only the last work entry is left"
            )
        plan, drop = step
        drops.append(drop)


def tex_builder(profile: Profile, out_dir: Path, *, engine: str | None = None) -> Builder:
    """A builder that writes ``resume.tex`` to ``out_dir``, compiles it and checks the layout."""

    def build(plan: ResumePlan) -> Built:
        out_dir.mkdir(parents=True, exist_ok=True)
        tex = render_tex(plan, profile)
        path = out_dir / TEX_NAME
        path.write_text(tex, encoding="utf-8")
        result = compile_tex(path, engine=engine)
        if not result.ok or result.pdf is None:
            raise RenderError("LaTeX failed:\n  " + "\n  ".join(result.errors[:5]))
        report = check_layout(tex, result.pdf, result.log, plan, pages=result.pages)
        return Built(tex, result.pdf, report, result.engine, result.retried)

    return build
