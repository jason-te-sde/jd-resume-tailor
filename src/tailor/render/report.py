"""The layout part of the provenance report: how the resume was built and what was dropped."""

from __future__ import annotations

from typing import Any

from tailor.plan import markdown_from_dict
from tailor.render.fit import Drop, FitResult
from tailor.render.layout import LayoutReport


def _percent(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.0%}"


def _drop_line(number: int, drop: Drop) -> str:
    if drop.kind == "bullet":
        return (
            f"{number}. Bullet {drop.position} of {drop.entry_title} (`{drop.entry_id}`), "
            f"because {drop.reason}: {drop.text}"
        )
    return f"{number}. The entry {drop.entry_title} (`{drop.entry_id}`), because {drop.reason}."


def _checks(report: LayoutReport) -> list[str]:
    def count(label: str, found: tuple[str, ...] | None) -> str:
        return f"- {label}: " + ("not checked" if found is None else str(len(found)))

    lines = [
        f"- pages: {report.pages}",
        f"- overfull boxes: {report.overfull}",
        count("bullet gaps", report.gaps),
        count("wrapped bullets", report.wrapped),
        f"- page fill: {_percent(report.fill)}",
        "- text extractable for ATS: "
        + (
            "not checked"
            if report.ats_ok is None
            else f"{'yes' if report.ats_ok else 'NO'} ({report.words} words)"
        ),
        f"- repeated opening words: {len(report.repeated_openers)}",
        f"- unescaped characters: {len(report.unescaped)}",
    ]
    lines += [f"- skipped: {note}" for note in report.skipped]
    lines += [
        f"- repeated opener in {r.entry}: {r.word} x{r.count}" for r in report.repeated_openers
    ]
    return lines


def layout_markdown(result: FitResult) -> str:
    """The "Layout" section: the engine, every drop with its reason, and the final checks."""
    built = result.built
    kept = sum(len(e.bullets) for e in result.plan.entries)
    lines = [
        "## Layout",
        "",
        f"Compiled with {built.engine}"
        + (" after a font or Unicode error with the first engine" if built.retried else "")
        + f". The resume has {len(result.plan.entries)} entries and {kept} bullets.",
        "",
    ]
    if result.drops:
        lines += [
            f"{len(result.drops)} dropped to fit one page, in this order (nothing was reworded):",
            "",
            *(_drop_line(n, d) for n, d in enumerate(result.drops, start=1)),
        ]
    else:
        lines.append("Nothing was dropped: the plan fit one page as it was.")
    lines += ["", "Checks on the final PDF:", "", *_checks(built.report)]
    return "\n".join(lines) + "\n"


def provenance_markdown(document: dict[str, Any], result: FitResult) -> str:
    """The plan's provenance report followed by the layout section."""
    return markdown_from_dict(document) + "\n" + layout_markdown(result)
