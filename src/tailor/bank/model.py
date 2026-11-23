"""The fact bank as typed, immutable values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from tailor.bank.facts import Fact, derive_facts
from tailor.bank.latex import normalize_latex
from tailor.bank.scope import (
    DO_NOT_CLAIM,
    do_not_claim_claims,
    same_project_refs,
    split_claims,
    split_parts,
)
from tailor.bank.validate import Problem


@dataclass(frozen=True, slots=True, kw_only=True)
class _EntryBase:
    id: str
    title: str
    bullets: tuple[str, ...]
    bullet_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    adaptable: bool = True
    date: str = ""
    location: str = ""
    description: str = ""
    scope: str = ""
    facts: tuple[Fact, ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkEntry(_EntryBase):
    """A job or internship.

    Strings are as written in the bank, LaTeX escapes included, so a renderer can use them as
    they are. Optional strings are empty when absent; ``facts`` is ``None`` when absent.
    """

    type: ClassVar[Literal["work"]] = "work"
    company: str
    tech: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectEntry(_EntryBase):
    """A project. Only a project with ``has_code`` set is a candidate for a resume."""

    type: ClassVar[Literal["project"]] = "project"
    has_code: bool
    tech_stack: str = ""
    github: str = ""


Entry = WorkEntry | ProjectEntry


@dataclass(frozen=True, slots=True)
class Bank:
    """A validated fact bank, and the views of it that the verifier reads.

    Build one with :func:`tailor.bank.load.load_bank` or :func:`tailor.bank.load.parse_bank`.
    """

    entries: tuple[Entry, ...]

    def entry(self, entry_id: str) -> Entry:
        """Return the entry with this id; raise ``KeyError`` naming the id if there is none."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(f"no entry with id {entry_id!r}")

    def facts_for(self, entry_id: str) -> list[Fact]:
        """Return what a bullet about this entry may cite.

        An entry with ``facts`` gets exactly those. Otherwise each bullet is a fact
        ``<id>#<n>`` (n from 1) and the scope text, without its DO NOT CLAIM parts, is the fact
        ``<id>#scope``; a scope with nothing left gives none. Text is plain, LaTeX resolved.
        """
        entry = self.entry(entry_id)
        if entry.facts is not None:
            return list(entry.facts)
        return derive_facts(entry.id, entry.bullets, entry.scope)

    def fact(self, fact_id: str) -> Fact:
        """Return the fact with this id; raise ``KeyError`` naming the id if there is none."""
        for entry in self.entries:
            for fact in self.facts_for(entry.id):
                if fact.id == fact_id:
                    return fact
        raise KeyError(f"no fact with id {fact_id!r}")

    def do_not_claim(self, entry_id: str) -> list[str]:
        """Return what this entry must not claim, one plain-text claim per item.

        The claims come from every ``DO NOT CLAIM:`` part of its scope, split at semicolons and
        sentence ends with parentheticals kept whole. Empty when the scope has no such part.
        """
        return do_not_claim_claims(normalize_latex(self.entry(entry_id).scope))

    def same_project_groups(self) -> list[frozenset[str]]:
        """Return the sets of entry ids that describe one project, smallest id first.

        ``Same project as <id>`` links two entries whichever of them says it, and links chain:
        if B says it of A and C says it of B, the group is {A, B, C}. Entries linked to nothing
        are not listed.
        """
        parent = {entry.id: entry.id for entry in self.entries}

        def find(entry_id: str) -> str:
            while parent[entry_id] != entry_id:
                parent[entry_id] = parent[parent[entry_id]]
                entry_id = parent[entry_id]
            return entry_id

        for entry in self.entries:
            for ref in same_project_refs(normalize_latex(entry.scope)):
                if ref in parent:
                    parent[find(ref)] = find(entry.id)
        members: dict[str, set[str]] = {}
        for entry_id in parent:
            members.setdefault(find(entry_id), set()).add(entry_id)
        groups = [frozenset(group) for group in members.values() if len(group) > 1]
        return sorted(groups, key=min)

    def candidates(self) -> list[Entry]:
        """Return the entries a resume may draw on: every work entry, and projects with code."""
        return [entry for entry in self.entries if isinstance(entry, WorkEntry) or entry.has_code]

    def warnings(self) -> list[Problem]:
        """Return things that are legal but probably a mistake, in document order."""
        found: list[Problem] = []
        for index, entry in enumerate(self.entries):

            def warn(field: str, message: str, index: int = index) -> None:
                found.append(Problem(("experiences", index, field), message))

            if isinstance(entry, ProjectEntry):
                if not entry.description.strip():
                    warn("description", f"project {entry.id!r} has no description")
                if entry.has_code and not entry.github:
                    warn("github", f"project {entry.id!r} has has_code set but no github link")
                if entry.github and not entry.has_code:
                    warn(
                        "has_code",
                        f"project {entry.id!r} has a github link but has_code is false",
                    )
            if not entry.scope.strip():
                warn(
                    "scope",
                    f"{entry.id!r} has no scope: no provenance to cite and no DO NOT CLAIM limits",
                )
            elif any(
                part.label == DO_NOT_CLAIM and not split_claims(part.text)
                for part in split_parts(normalize_latex(entry.scope))
            ):
                warn("scope", f"{entry.id!r} has a DO NOT CLAIM part with no claims")
        return found
