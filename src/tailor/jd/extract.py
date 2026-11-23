"""Extract the structure of a job description with one model call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from tailor.llm import LLM, LLMError, Message

_LIST: Final = {"type": "array", "items": {"type": "string"}}
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "company": {"type": "string"},
        "responsibilities": _LIST,
        "requirements": _LIST,
        "nice_to_have": _LIST,
        "keywords": _LIST,
    },
    "required": [
        "title",
        "company",
        "responsibilities",
        "requirements",
        "nice_to_have",
        "keywords",
    ],
    "additionalProperties": False,
}
SYSTEM: Final = """\
You read a job posting and extract its structure. Rules:
- Copy or lightly shorten the posting's own wording; add nothing it does not say.
- responsibilities: what the person will do, one short item each, in the posting's order.
- requirements: what the posting says is required; nice_to_have: what it calls preferred or a plus.
- keywords: the technologies, methods and domain terms the posting names, each as written.
- title and company: as written in the posting; an empty string when it does not say.
- Leave out benefits, salary, location, and equal-opportunity text."""


@dataclass(frozen=True, slots=True)
class JobDescription:
    """The structure of a posting."""

    title: str
    company: str
    responsibilities: tuple[str, ...]
    requirements: tuple[str, ...]
    nice_to_have: tuple[str, ...]
    keywords: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """The description as plain JSON values."""
        return {
            "title": self.title,
            "company": self.company,
            "responsibilities": list(self.responsibilities),
            "requirements": list(self.requirements),
            "nice_to_have": list(self.nice_to_have),
            "keywords": list(self.keywords),
        }


def _strings(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise LLMError(f"the extraction's {key} is not a list")
    seen: dict[str, None] = {}
    for item in value:
        if isinstance(item, str) and item.strip():
            seen.setdefault(item.strip())
    return tuple(seen)


def extract_job(
    llm: LLM, text: str, *, model: str, title_hint: str = "", company_hint: str = ""
) -> JobDescription:
    """Extract ``{title, company, responsibilities, requirements, nice_to_have, keywords}``."""
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Job posting:\n\n{text}"},
    ]
    data = llm.complete_json(messages, SCHEMA, model=model)
    title = str(data.get("title") or "").strip() or title_hint
    company = str(data.get("company") or "").strip() or company_hint
    return JobDescription(
        title=title,
        company=company,
        responsibilities=_strings(data, "responsibilities"),
        requirements=_strings(data, "requirements"),
        nice_to_have=_strings(data, "nice_to_have"),
        keywords=_strings(data, "keywords"),
    )
