"""Check a parsed bank against the format and report every problem with its JSON path.

Two layers. The JSON Schema shipped with the package (``bank.schema.json``) checks shape and types.
The rules a schema cannot express, which compare one part of the bank with another, are checked
here: unique ids, one capability label per bullet, and ``Same project as`` pointing at an entry
that exists. Both layers run on the raw parsed data and tolerate what the other found wrong, so one
call reports everything it can instead of one problem at a time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from tailor.bank.facts import bullet_fact_id, scope_fact_id, scope_provenance
from tailor.bank.latex import normalize_latex
from tailor.bank.scope import same_project_refs

SCHEMA_RESOURCE: Final = "bank.schema.json"
_MAX_MESSAGE = 200

PathPart = str | int


def format_path(path: Sequence[PathPart]) -> str:
    """Render a path into the data as JSONPath: ``$.experiences[3].bullets[1]``."""
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part.isidentifier():
            rendered += f".{part}"
        else:
            rendered += "[" + repr(part) + "]"
    return rendered


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong (or, for warnings, doubtful) at a place in the bank."""

    path: tuple[PathPart, ...]
    message: str

    def __str__(self) -> str:
        """Render as ``<JSONPath>: <message>``."""
        return f"{format_path(self.path)}: {self.message}"


class BankError(Exception):
    """The input is not a valid bank. ``problems`` lists every problem found, with its path."""

    def __init__(self, problems: Sequence[Problem], source: str | None = None) -> None:
        """Keep the problems, and put each one on its own line of the message."""
        self.problems = tuple(problems)
        self.source = source
        noun = "problem" if len(self.problems) == 1 else "problems"
        where = f" in {source}" if source else ""
        lines = "\n".join(f"  {problem}" for problem in self.problems)
        super().__init__(f"invalid fact bank{where}: {len(self.problems)} {noun}\n{lines}")


@cache
def _validator() -> Draft202012Validator:
    text = resources.files("tailor.bank").joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")
    schema = json.loads(text)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _clip(message: str) -> str:
    return message if len(message) <= _MAX_MESSAGE else message[: _MAX_MESSAGE - 3] + "..."


def _is_entry_path(path: Sequence[PathPart]) -> bool:
    return len(path) == 2 and path[0] == "experiences"


def _problems_from(error: ValidationError) -> list[Problem]:
    path = tuple(error.absolute_path)
    instance: Any = error.instance
    schema: Mapping[str, Any] = error.schema if isinstance(error.schema, Mapping) else {}
    if error.validator == "required" and isinstance(instance, dict):
        required: Any = error.validator_value
        return [
            Problem((*path, name), "required property is missing")
            for name in required
            if name not in instance
        ]
    if error.validator == "additionalProperties" and isinstance(instance, dict):
        allowed = set(schema.get("properties", {}))
        kind = instance.get("type")
        message = "unknown field: not part of the bank format"
        if _is_entry_path(path) and isinstance(kind, str):
            message += f" for a {kind} entry"
        return [Problem((*path, name), message) for name in sorted(set(instance) - allowed)]
    if error.validator == "pattern" and "x-error" in schema:
        return [Problem(path, schema["x-error"])]
    return [Problem(path, _clip(error.message))]


def schema_problems(data: object) -> list[Problem]:
    """Check ``data`` against the JSON Schema only."""
    problems: list[Problem] = []
    for error in _validator().iter_errors(data):
        problems.extend(_problems_from(error))
    return problems


def _entries(data: object) -> list[tuple[int, dict[str, Any]]]:
    """The entries that are objects, with their positions; empty if the bank has no list of them."""
    if not isinstance(data, dict) or not isinstance(data.get("experiences"), list):
        return []
    return [(i, entry) for i, entry in enumerate(data["experiences"]) if isinstance(entry, dict)]


def _id_of(entry: dict[str, Any]) -> str | None:
    value = entry.get("id")
    return value if isinstance(value, str) else None


def semantic_problems(data: object) -> list[Problem]:
    """Check the rules that relate one part of ``data`` to another."""
    entries = _entries(data)
    problems: list[Problem] = []
    first_seen: dict[str, int] = {}
    for index, entry in entries:
        entry_id = _id_of(entry)
        if entry_id is None:
            continue
        if entry_id in first_seen:
            earlier = format_path(("experiences", first_seen[entry_id], "id"))
            problems.append(
                Problem(
                    ("experiences", index, "id"),
                    f"duplicate id {entry_id!r}, first used at {earlier}",
                )
            )
        else:
            first_seen[entry_id] = index

    for index, entry in entries:
        bullets, capabilities = entry.get("bullets"), entry.get("bullet_capabilities")
        if (
            isinstance(bullets, list)
            and isinstance(capabilities, list)
            and len(bullets) != len(capabilities)
        ):
            problems.append(
                Problem(
                    ("experiences", index, "bullet_capabilities"),
                    f"has {len(capabilities)} items but there are {len(bullets)} bullets;"
                    " it needs one label per bullet",
                )
            )
        scope = entry.get("scope")
        if isinstance(scope, str):
            for ref in same_project_refs(normalize_latex(scope)):
                path = ("experiences", index, "scope")
                if ref == _id_of(entry):
                    problems.append(
                        Problem(path, f"'Same project as {ref}' names the entry itself")
                    )
                elif ref not in first_seen:
                    problems.append(
                        Problem(path, f"'Same project as {ref}' names an id that no entry has")
                    )
    problems.extend(_fact_id_problems(entries))
    return problems


def _fact_id_problems(entries: list[tuple[int, dict[str, Any]]]) -> list[Problem]:
    """Report a fact id used twice, naming the place where it was first used.

    Explicit ids are checked against each other and against the ids an entry without facts
    derives for its bullets and scope. Two derived ids can only collide when two entries share an
    id, which is reported on its own, so that pair is left out.
    """
    first_use: dict[str, tuple[str, bool]] = {}  # id -> (path, explicit)
    problems: list[Problem] = []

    def claim(
        fact_id: str, path: tuple[PathPart, ...], *, explicit: bool, derived_for: str = ""
    ) -> None:
        if fact_id not in first_use:
            first_use[fact_id] = (format_path(path), explicit)
            return
        where, earlier_was_explicit = first_use[fact_id]
        if not (explicit or earlier_was_explicit):
            return
        if explicit:
            message = f"duplicate fact id {fact_id!r}, first used at {where}"
        else:
            message = f"fact id {fact_id!r} derived for {derived_for} is already used at {where}"
        problems.append(Problem(path, message))

    for index, entry in entries:
        entry_id = _id_of(entry)
        if entry_id is None:
            continue
        facts = entry.get("facts")
        if isinstance(facts, list):
            for number, fact in enumerate(facts):
                if isinstance(fact, dict) and isinstance(fact.get("id"), str):
                    path = ("experiences", index, "facts", number, "id")
                    claim(fact["id"], path, explicit=True)
            continue
        entry_path = ("experiences", index, "id")
        bullets, scope = entry.get("bullets"), entry.get("scope", "")
        if isinstance(bullets, list):
            for number in range(1, len(bullets) + 1):
                derived_for = f"bullet {number}"
                claim(
                    bullet_fact_id(entry_id, number),
                    entry_path,
                    explicit=False,
                    derived_for=derived_for,
                )
        if isinstance(scope, str) and scope_provenance(scope):
            claim(scope_fact_id(entry_id), entry_path, explicit=False, derived_for="the scope")
    return problems


def _order(problem: Problem) -> tuple[tuple[int, int, str], ...]:
    """Sort key that follows the document: positions numerically, names alphabetically."""
    return tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in problem.path)


def validate(data: object) -> list[Problem]:
    """Return every problem in ``data``, in document order; empty when it is a valid bank."""
    # jsonschema reports each missing required property as its own error, and each error names
    # all of them, so the same problem can come out of several errors.
    unique = dict.fromkeys([*schema_problems(data), *semantic_problems(data)])
    return sorted(unique, key=_order)
