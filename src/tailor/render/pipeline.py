"""Render a plan into a directory: ``resume.tex``, ``resume.pdf`` and the provenance report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from tailor.render.fit import FitResult, fit_to_one_page, tex_builder
from tailor.render.model import ResumePlan, load_profile
from tailor.render.report import provenance_markdown

PROVENANCE_NAME: Final = "provenance.md"


def render_plan(
    document: dict[str, Any],
    plan: ResumePlan,
    profile_path: Path,
    out_dir: Path,
    *,
    engine: str | None = None,
    one_line: bool = False,
) -> FitResult:
    """Fit ``plan`` to one page and write the resume and its provenance report to ``out_dir``.

    ``document`` is the plan as ``tailor plan`` wrote it; the report is built from it plus the
    drops. Raises ``RenderError`` (or a subclass) when the profile is unreadable, no TeX engine is
    installed, LaTeX fails, or the plan cannot be made to fit.
    """
    profile = load_profile(profile_path)
    result = fit_to_one_page(plan, tex_builder(profile, out_dir, engine=engine), one_line=one_line)
    (out_dir / PROVENANCE_NAME).write_text(provenance_markdown(document, result), encoding="utf-8")
    return result
