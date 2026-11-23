"""Fill the LaTeX template from a plan and a profile.

The template (``template.tex``) holds only layout. Every string that reaches the document passes
through ``tailor.render.escape`` here, once, as it is placed; nothing else in this module writes
user text.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from typing import Final

from tailor.render.escape import escape_latex, escape_url, is_link
from tailor.render.model import EducationItem, PlanEntry, Profile, ResumePlan

_PLACEHOLDER: Final = re.compile(r"@@([A-Z]+)@@")
_REGION: Final = re.compile(r"%%BEGIN (\w+)\n(.*?)%%END \1\n", re.DOTALL)
_CATEGORY_LABELS: Final = {
    "programming_languages": "Languages",
    "frameworks_libraries": "Frameworks and Libraries",
    "devops_cloud": "DevOps and Cloud",
    "databases": "Databases and Messaging",
    "developer_tools": "Developer Tools",
}
_LINK_CONTACTS: Final = ("linkedin", "github", "website")


@cache
def template_text() -> str:
    """The template that ships with the package."""
    return resources.files("tailor.render").joinpath("template.tex").read_text(encoding="utf-8")


def _link(url: str) -> str:
    """A link whose text is the URL without its scheme, or the plain text if it is no link."""
    shown = re.sub(r"^https?://(www\.)?", "", url.strip()).rstrip("/")
    if not is_link(url):
        return escape_latex(url)
    return rf"\href{{{escape_url(url.strip())}}}{{{escape_latex(shown)}}}"


def _contacts(profile: Profile) -> str:
    contacts = dict(profile.contacts)
    parts: list[str] = []
    for key in ("location", "phone", "email"):
        if key in contacts:
            value = contacts[key]
            if key == "email":
                parts.append(rf"\href{{{escape_url('mailto:' + value)}}}{{{escape_latex(value)}}}")
            else:
                parts.append(escape_latex(value))
    parts += [_link(contacts[key]) for key in _LINK_CONTACTS if key in contacts]
    return r" $|$ ".join(parts)


def _education(item: EducationItem) -> str:
    degree = item.degree + (f", GPA: {item.gpa}" if item.gpa else "")
    return (
        rf"    \resumeSubheading{{{escape_latex(item.school)}}}{{{escape_latex(item.location)}}}"
        rf"{{{escape_latex(degree)}}}{{{escape_latex(item.date)}}}"
    )


def _bullets(entry: PlanEntry) -> str:
    items = [rf"        \resumeItem{{{escape_latex(b.text)}}}" for b in entry.bullets]
    return "\n".join([r"      \resumeItemListStart", *items, r"      \resumeItemListEnd"])


def _work(entry: PlanEntry) -> str:
    heading = (
        rf"    \resumeSubheading{{{escape_latex(entry.company)}}}{{{escape_latex(entry.location)}}}"
        rf"{{{escape_latex(entry.title)}}}{{{escape_latex(entry.date)}}}"
    )
    return f"{heading}\n{_bullets(entry)}"


def project_name(title: str) -> str:
    """The short name of a project: what comes before the first colon of its title."""
    head, _, _ = title.partition(": ")
    return head.strip() or title


def _project(entry: PlanEntry) -> str:
    left = rf"\textbf{{{escape_latex(project_name(entry.title))}}}"
    if entry.tech:
        left += rf" $|$ \emph{{{escape_latex(entry.tech)}}}"
    if entry.link and is_link(entry.link):
        left += rf" \codelink{{{escape_url(entry.link)}}}"
    heading = rf"    \resumeProjectHeading{{{left}}}{{{escape_latex(entry.date)}}}"
    return f"{heading}\n{_bullets(entry)}"


def _wanted(item: str, keywords: tuple[str, ...]) -> bool:
    base = re.sub(r"\s*\(.*?\)", "", item).strip().lower()
    if not base:
        return False
    # "+" and "#" belong to a name (C++, C#), so "C" is not found inside "C++".
    pattern = re.compile(rf"(?<![\w+#]){re.escape(base)}(?![\w+#])")
    return any(pattern.search(keyword.lower()) for keyword in keywords)


def category_label(key: str) -> str:
    """The heading of a skills category: a known name, else the key spelled as words."""
    return _CATEGORY_LABELS.get(key) or key.replace("_", " ").strip().capitalize()


def ordered_skills(items: tuple[str, ...], keywords: tuple[str, ...]) -> list[str]:
    """The skills of one category, the ones the posting names first, each group in input order."""
    wanted = [item for item in items if _wanted(item, keywords)]
    return wanted + [item for item in items if item not in wanted]


def _skills(profile: Profile, keywords: tuple[str, ...]) -> str:
    rows = [
        rf"     \textbf{{{escape_latex(category_label(key))}}}"
        rf"{{: {escape_latex(', '.join(ordered_skills(items, keywords)))}}}"
        for key, items in profile.skills
    ]
    return " \\\\\n".join(rows)


def render_tex(plan: ResumePlan, profile: Profile) -> str:
    """Return the resume as a ``.tex`` document.

    The header, education and skills come from the profile; the work entries and the projects come
    from the plan, in the plan's order.
    """
    work = [e for e in plan.entries if e.type == "work"]
    projects = [e for e in plan.entries if e.type == "project"]
    values = {
        "NAME": escape_latex(profile.name),
        "CONTACT": _contacts(profile),
        "EDUCATION": "\n".join(_education(item) for item in profile.education),
        "EXPERIENCE": "\n".join(_work(e) for e in work),
        "PROJECTS": "\n".join(_project(e) for e in projects),
        "SKILLS": _skills(profile, plan.keywords),
    }

    def keep_region(match: re.Match[str]) -> str:
        name = match.group(1).upper()
        return match.group(2) if values.get(name) else ""

    text = _REGION.sub(keep_region, template_text())
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], text)
