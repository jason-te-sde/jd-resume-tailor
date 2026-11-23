"""The two inputs of a render, read from JSON and checked: the plan and the profile."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PLAN_VERSION = 1


class RenderError(Exception):
    """The document cannot be built: LaTeX failed or the plan cannot be made to fit."""


class InputError(RenderError):
    """The plan or the profile cannot be read, or is not what a render needs."""


@dataclass(frozen=True, slots=True)
class PlanBullet:
    """A final bullet: plain text, as the verifier saw it."""

    text: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanEntry:
    """One entry of the resume. Strings are plain text; the renderer escapes them.

    Attributes:
        id: The bank's id for the entry.
        type: ``work`` or ``project``.
        title: The role (work) or the project's name.
        company: The employer; empty for a project.
        location: Where the work was; empty when not given.
        date: The dates as printed.
        tech: The technologies line of the entry.
        link: The project's repository URL; empty when not given.
        score: The coverage score that the selection gave the entry (0 to 10).
        bullets: The bullets in the order the selection ranked them, most useful first.
    """

    id: str
    type: Literal["work", "project"]
    title: str
    company: str
    location: str
    date: str
    tech: str
    link: str
    score: int
    bullets: tuple[PlanBullet, ...]


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """The entries that go on the resume, in order, and the posting's keywords."""

    entries: tuple[PlanEntry, ...]
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EducationItem:
    """One school."""

    school: str
    degree: str
    location: str = ""
    date: str = ""
    gpa: str = ""
    coursework: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Profile:
    """The person: contact details, education and the skills they may list."""

    name: str
    contacts: tuple[tuple[str, str], ...] = ()
    education: tuple[EducationItem, ...] = ()
    skills: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())


_CONTACT_FIELDS = ("phone", "email", "linkedin", "github", "website", "location")


def _fail(where: Path | str, message: str) -> InputError:
    return InputError(f"{where}: {message}")


def _text(raw: dict[str, Any], key: str, where: str, *, required: bool = False) -> str:
    value = raw.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise _fail(where, f"{key!r} must be a string")
    if required and not value.strip():
        raise _fail(where, f"{key!r} is required")
    return value


def _strings(raw: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _fail(where, f"{key!r} must be a list of strings")
    return tuple(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"cannot read {path}: {error.strerror}") from error
    except json.JSONDecodeError as error:
        raise InputError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise _fail(path, "expected a JSON object")
    return document


def plan_from_dict(document: dict[str, Any], where: Path | str = "plan") -> ResumePlan:
    """Read the plan ``tailor plan`` writes; raise ``RenderError`` saying what is wrong."""
    if document.get("version") != PLAN_VERSION:
        raise _fail(where, f"plan version {document.get('version')!r} is not {PLAN_VERSION}")
    entries: list[PlanEntry] = []
    for index, raw in enumerate(document.get("entries", [])):
        label = f"{where}: entries[{index}]"
        if not isinstance(raw, dict):
            raise _fail(label, "expected an object")
        if raw.get("type") not in ("work", "project"):
            raise _fail(label, "'type' must be 'work' or 'project'")
        if "company" not in raw or "date" not in raw:
            raise _fail(label, "no heading data: run `tailor plan` again with this version")
        bullets = tuple(
            PlanBullet(
                _text(b, "text", f"{label}.bullets", required=True), _strings(b, "sources", label)
            )
            for b in raw.get("bullets", [])
        )
        score = raw.get("score", 0)
        if not isinstance(score, int):
            raise _fail(label, "'score' must be an integer")
        entries.append(
            PlanEntry(
                id=_text(raw, "id", label, required=True),
                type=raw["type"],
                title=_text(raw, "title", label, required=True),
                company=_text(raw, "company", label),
                location=_text(raw, "location", label),
                date=_text(raw, "date", label),
                tech=_text(raw, "tech", label),
                link=_text(raw, "link", label),
                score=score,
                bullets=bullets,
            )
        )
    job = document.get("job", {})
    keywords = tuple(k for k in job.get("keywords", []) if isinstance(k, str))
    return ResumePlan(entries=tuple(entries), keywords=keywords)


def load_plan(path: Path) -> tuple[dict[str, Any], ResumePlan]:
    """Read ``plan.json``: the document as written (kept for the report) and the plan in it."""
    document = _read_json(path)
    return document, plan_from_dict(document, path)


def load_profile(path: Path) -> Profile:
    """Read a profile JSON file; raise ``RenderError`` saying what is wrong."""
    raw = _read_json(path)
    where = str(path)
    name = _text(raw, "name", where, required=True)
    contacts = tuple(
        (key, value) for key in _CONTACT_FIELDS if (value := _text(raw, key, where).strip())
    )
    education: list[EducationItem] = []
    for index, item in enumerate(raw.get("education", [])):
        label = f"{where}: education[{index}]"
        if not isinstance(item, dict):
            raise _fail(label, "expected an object")
        education.append(
            EducationItem(
                school=_text(item, "school", label, required=True),
                degree=_text(item, "degree", label, required=True),
                location=_text(item, "location", label),
                date=_text(item, "date", label),
                gpa=_text(item, "gpa", label),
                coursework=_strings(item, "coursework", label),
            )
        )
    skills: list[tuple[str, tuple[str, ...]]] = []
    bank = raw.get("skills_bank", {})
    if not isinstance(bank, dict):
        raise _fail(where, "'skills_bank' must be an object of lists")
    for category, items in bank.items():
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise _fail(where, f"skills_bank.{category} must be a list of strings")
        if items:
            skills.append((category, tuple(items)))
    return Profile(name, contacts, tuple(education), tuple(skills))
