"""The verification report and its two renderings: JSON for programs, Markdown for people."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from tailor.verify.findings import KINDS, Finding
from tailor.verify.rounding import Rounding


@dataclass(frozen=True, slots=True)
class BulletResult:
    """The verdict on one bullet.

    Attributes:
        index: Position of the bullet in the input, from 0.
        id: The label the caller gave the bullet, if any.
        entry: The bank entry the bullet is about.
        text: The bullet.
        sources: Ids of the facts it was checked against.
        findings: What is wrong with it, in the order of the text; empty when it is ok.
    """

    index: int
    id: str | None
    entry: str
    text: str
    sources: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        """Whether the bullet has no findings."""
        return not self.findings


@dataclass(frozen=True, slots=True)
class Report:
    """The verdicts on a list of bullets, and the policy they were reached under."""

    rounding: Rounding
    results: tuple[BulletResult, ...]

    @property
    def ok(self) -> bool:
        """Whether every bullet is ok."""
        return all(result.ok for result in self.results)

    @property
    def violation_count(self) -> int:
        """How many bullets have at least one finding."""
        return sum(not result.ok for result in self.results)

    @property
    def finding_counts(self) -> dict[str, int]:
        """How many findings of each kind there are, in the order of the kinds."""
        counts = Counter(f.kind for result in self.results for f in result.findings)
        return {kind: counts[kind] for kind in KINDS if counts[kind]}


def _finding_dict(finding: Finding) -> dict[str, Any]:
    return {
        "kind": finding.kind,
        "span": [finding.start, finding.end],
        "text": finding.text,
        "evidence": finding.evidence,
    }


def render_json(report: Report) -> str:
    """Render the report as JSON text ending in a newline."""
    document = {
        "ok": report.ok,
        "rounding": report.rounding,
        "summary": {
            "bullets": len(report.results),
            "ok": len(report.results) - report.violation_count,
            "violations": report.violation_count,
            "findings": report.finding_counts,
        },
        "bullets": [
            {
                "index": result.index,
                "id": result.id,
                "entry": result.entry,
                "text": result.text,
                "sources": list(result.sources),
                "status": "ok" if result.ok else "violation",
                "findings": [_finding_dict(f) for f in result.findings],
            }
            for result in report.results
        ],
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def render_markdown(report: Report) -> str:
    """Render the report as Markdown: a summary, then each bullet with its findings."""
    total = len(report.results)
    noun = "bullet" if total == 1 else "bullets"
    if report.ok:
        verdict = f"{total} {noun}: all bullets ok"
    else:
        verdict = f"{total} {noun}: {total - report.violation_count} ok, "
        verdict += f"{report.violation_count} with violations"
    lines = ["# Verification report", "", f"{verdict} (rounding: {report.rounding})."]
    for result in report.results:
        count = len(result.findings)
        status = "ok" if result.ok else f"{count} violation{'s' if count != 1 else ''}"
        label = f" ({result.id})" if result.id else ""
        lines += ["", f"## {result.index + 1}. {status}{label}", "", f"> {result.text}", ""]
        lines.append(f"Entry `{result.entry}`; sources: {', '.join(result.sources) or 'none'}.")
        if result.findings:
            lines.append("")
            lines += [
                f"- **{f.kind}** `{f.text}` (characters {f.start}-{f.end}): {f.evidence}"
                for f in result.findings
            ]
    return "\n".join(lines) + "\n"
