"""Turn a bank file or parsed JSON into a validated :class:`Bank`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

from tailor.bank.facts import Fact, Number
from tailor.bank.latex import normalize_latex
from tailor.bank.model import Bank, Entry, ProjectEntry, WorkEntry
from tailor.bank.validate import BankError, Problem, validate


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            owner = dict(pairs).get("id")
            where = f" in the object with id {owner!r}" if isinstance(owner, str) else ""
            raise _DuplicateKeyError(
                f"duplicate key {key!r}{where}; JSON lets the last one win, which would silently"
                " drop the first"
            )
        seen.add(key)
    return dict(pairs)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a JSON number")


def _numbers(raw: list[dict[str, Any]]) -> tuple[Number, ...]:
    return tuple(Number(n["value"], n["unit"], n.get("as_measured")) for n in raw)


def _facts(entry_id: str, raw: list[dict[str, Any]]) -> tuple[Fact, ...]:
    return tuple(
        Fact(
            id=f["id"],
            entry_id=entry_id,
            text=normalize_latex(f["text"]),
            kind="explicit",
            numbers=_numbers(f.get("numbers", [])),
        )
        for f in raw
    )


def _entry(raw: dict[str, Any]) -> Entry:
    common: dict[str, Any] = {
        "id": raw["id"],
        "title": raw["title"],
        "bullets": tuple(raw["bullets"]),
        "bullet_capabilities": tuple(raw.get("bullet_capabilities", ())),
        "tags": tuple(raw.get("tags", ())),
        "adaptable": raw.get("adaptable", True),
        "date": raw.get("date", ""),
        "location": raw.get("location", ""),
        "description": raw.get("description", ""),
        "scope": raw.get("scope", ""),
        "facts": _facts(raw["id"], raw["facts"]) if "facts" in raw else None,
    }
    if raw["type"] == "work":
        return WorkEntry(**common, company=raw["company"], tech=raw.get("tech", ""))
    return ProjectEntry(
        **common,
        has_code=raw["has_code"],
        tech_stack=raw.get("tech_stack", ""),
        github=raw.get("github", ""),
    )


def parse_bank(data: object, *, source: str | None = None) -> Bank:
    """Validate parsed JSON and return it as a :class:`Bank`.

    Args:
        data: The parsed JSON document.
        source: A name for the document (a path), used in the error message.

    Raises:
        BankError: ``data`` is not a valid bank. The error lists every problem with its JSON path.
    """
    problems = validate(data)
    if problems:
        raise BankError(problems, source)
    assert isinstance(data, dict)  # validate() has checked it
    return Bank(entries=tuple(_entry(raw) for raw in data["experiences"]))


def load_bank(path: str | Path) -> Bank:
    """Read, validate and return the bank in the JSON file at ``path``.

    Raises:
        BankError: The file is not valid JSON (or has a duplicate key) or not a valid bank.
        OSError: The file cannot be read.
    """
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise BankError([Problem((), f"not valid UTF-8: {error.reason}")], source) from error
    try:
        data = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        message = f"not valid JSON: {error.msg} (line {error.lineno}, column {error.colno})"
        raise BankError([Problem((), message)], source) from error
    except ValueError as error:
        raise BankError([Problem((), str(error))], source) from error
    return parse_bank(data, source=source)
