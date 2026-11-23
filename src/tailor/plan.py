"""Build the plan: the job's structure, the entries chosen, and every final bullet with its proof.

``build_plan`` runs extraction, selection and rewriting. ``render_json`` and ``render_markdown`` are
the two outputs of ``tailor plan``: the plan a renderer reads, and a provenance report for a person.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

from tailor.bank import Bank, Entry, ProjectEntry, WorkEntry, normalize_latex
from tailor.jd.extract import JobDescription, extract_job
from tailor.jd.sources import JobText
from tailor.llm import LLM, LLMConfig
from tailor.rewrite import EntryResult, FinalBullet, rewrite_entry
from tailor.select import Selection, coverage_targets, select
from tailor.verify import Rounding

PLAN_VERSION: Final = 1
_OUTCOMES: Final = ("first_pass", "after_retry", "fallback", "verbatim")


@dataclass(frozen=True, slots=True)
class Plan:
    """Everything ``tailor plan`` decided, and how."""

    job: JobDescription
    job_source: str
    config: LLMConfig
    rounding: Rounding
    selection: Selection
    entries: tuple[EntryResult, ...]
    usage: dict[str, int]
    profile: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        """How many final bullets came out each way."""
        bullets = [b for e in self.entries for b in e.bullets]
        counts = {name: sum(b.outcome == name for b in bullets) for name in _OUTCOMES}
        return {"bullets": len(bullets), **counts}


def build_plan(
    llm: LLM,
    config: LLMConfig,
    bank: Bank,
    job: JobText,
    *,
    rounding: Rounding = "conservative",
    profile: str | None = None,
) -> Plan:
    """Extract the job's structure, select entries and bullets, rewrite and verify them."""
    description = extract_job(
        llm, job.text, model=config.model, title_hint=job.title, company_hint=job.company
    )
    selection = select(llm, bank, description, model=config.model)
    entries = tuple(
        rewrite_entry(
            llm, bank, description, pick.entry, pick.bullets, model=config.model, rounding=rounding
        )
        for pick in selection.picks
    )
    return Plan(
        job=description,
        job_source=job.source,
        config=config,
        rounding=rounding,
        selection=selection,
        entries=entries,
        usage=llm.usage.to_dict(),
        profile=profile,
    )


def _bullet_dict(bullet: FinalBullet) -> dict[str, Any]:
    return {
        "text": bullet.text,
        "sources": list(bullet.sources),
        "verdict": bullet.verdict,
        "outcome": bullet.outcome,
        "original": bullet.original,
        "reason": bullet.reason,
        "attempts": [
            {"text": a.text, "sources": list(a.sources), "problems": list(a.problems)}
            for a in bullet.attempts
        ],
    }


def _plain(text: str) -> str:
    # A date range is written with "--" in the bank; keep it as the dash it stands for.
    return normalize_latex(text.replace("--", "\u2013"))


def _heading(entry: Entry) -> dict[str, str]:
    """What a renderer prints in the heading of an entry, as plain text (it does the escaping)."""
    if isinstance(entry, WorkEntry):
        company, tech, link = entry.company, entry.tech, ""
    else:
        assert isinstance(entry, ProjectEntry)
        company, tech, link = "", entry.tech_stack, entry.github
    return {
        "company": _plain(company),
        "location": _plain(entry.location),
        "date": _plain(entry.date),
        "tech": _plain(tech),
        "link": link,
    }


def to_dict(plan: Plan) -> dict[str, Any]:
    """The plan as plain JSON values."""
    scores = {s.entry: s for s in plan.selection.scores}
    return {
        "version": PLAN_VERSION,
        "job": {"source": plan.job_source, **plan.job.to_dict()},
        "profile": plan.profile,
        "model": {"provider": plan.config.provider, "model": plan.config.model},
        "rounding": plan.rounding,
        "selection": {
            "targets": list(coverage_targets(plan.job)),
            "scores": [
                {
                    "entry": s.entry,
                    "score": s.score,
                    "covers": sorted(s.covers),
                    "reason": s.reason,
                    "bullet_priority": list(s.bullet_priority),
                }
                for s in plan.selection.scores
            ],
            "picked": [e.entry.id for e in plan.entries],
        },
        "entries": [
            {
                "id": e.entry.id,
                "type": e.entry.type,
                "title": normalize_latex(e.entry.title),
                **_heading(e.entry),
                "adaptable": e.entry.adaptable,
                "score": scores[e.entry.id].score,
                "calls": e.calls,
                "bullets": [_bullet_dict(b) for b in e.bullets],
            }
            for e in plan.entries
        ],
        "summary": plan.counts,
        "usage": plan.usage,
    }


def render_json(plan: Plan) -> str:
    """The plan as JSON text ending in a newline."""
    return json.dumps(to_dict(plan), indent=2, ensure_ascii=False) + "\n"


def render_markdown(plan: Plan) -> str:
    """The provenance report of a plan."""
    return markdown_from_dict(to_dict(plan))


def markdown_from_dict(document: dict[str, Any]) -> str:
    """The provenance report: for each bullet, where it came from and what the verifier said.

    Works on the plan as ``to_dict`` writes it, so a plan read back from ``plan.json`` gives the
    same report as the plan that was just built.
    """
    counts = document["summary"]
    usage = document["usage"]
    model = document["model"]
    job = document["job"]
    scores = {s["entry"]: s for s in document["selection"]["scores"]}
    lines = [
        "# Provenance report",
        "",
        f"Job: {job['title'] or 'untitled'} at {job['company'] or 'unknown company'} "
        f"({job['source']}).",
        f"Model: {model['model']} ({model['provider']}); rounding: {document['rounding']}; "
        f"{usage['calls']} calls, {usage['input_tokens']} input and "
        f"{usage['output_tokens']} output tokens.",
        "",
        f"{counts['bullets']} bullets: {counts['first_pass']} passed the verifier first time, "
        f"{counts['after_retry']} after one retry, {counts['fallback']} fell back to the bank's "
        f"wording, {counts['verbatim']} copied verbatim (adaptable: false).",
    ]
    for entry in document["entries"]:
        score = scores[entry["id"]]
        lines += [
            "",
            f"## {entry['title']} (`{entry['id']}`)",
            "",
            f"Coverage score {score['score']}/10: {score['reason'] or 'no reason given'}",
        ]
        for number, bullet in enumerate(entry["bullets"], start=1):
            lines += ["", f"{number}. {bullet['text']}", ""]
            lines.append(f"   - outcome: {bullet['outcome']}; verdict: {bullet['verdict']}")
            lines.append(f"   - sources: {', '.join(bullet['sources']) or 'none'}")
            if bullet["outcome"] != "verbatim" and bullet["text"] != bullet["original"]:
                lines.append(f"   - bank wording: {bullet['original']}")
            if bullet["reason"]:
                lines.append(f"   - why the bank wording was used: {bullet['reason']}")
            elif len(bullet["attempts"]) > 1:
                lines.append(
                    f"   - first attempt rejected: {'; '.join(bullet['attempts'][0]['problems'])}"
                )
    return "\n".join(lines) + "\n"
