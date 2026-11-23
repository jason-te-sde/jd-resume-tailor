"""Check that every number in a measured README section is a value of the files that section names.

A measured section opens with a header that names it and its sources, and closes with an end
marker::

    <!-- measured:begin results eval/pipeline/results.json#scores eval/pipeline/residual.json -->
    ...
    <!-- measured:end -->

A source is a JSON file relative to the repository root, optionally narrowed to one subtree by a
dotted path after ``#`` (a list index is a number, and ``*`` stands for every key or item at that
level). The header may run over several lines.

**What is checked.** Inside a section, a number is any figure in the running text or tables: not
one inside a code span, a fenced block, a link target, a URL or an HTML comment, not a list marker,
and not part of an identifier (``qwen3``, ``8b``, ``v0.1.0``, a commit hash). Each must be a
numeric value of the section's sources (an integer or a float; never a boolean, and never digits
inside a string), either exactly or rounded half up to the places the figure shows (never to
zero). A figure followed by ``%`` must be a value under a key that says ``percent``, and any other
figure a value under a key that does not. Nothing else is derived: a percentage is checked against
a percent the scorer wrote from the counts, never made from a rounded rate, so a figure rounded
twice cannot pass. A date (``2026-03``, ``2026-11-21``) must be the date, or
the start of the date, of a date or timestamp string in the sources.

Usage: ``python scripts/readme_numbers.py [--root DIR] [README ...]`` checks the given READMEs, or
by default the top-level one and every ``eval/**/README.md``, each of which must have a measured
section. ``--mutations`` instead changes every checked figure of those READMEs by one in its last
digit, one at a time, and reports how many changes the check rejects.

Exit codes: 0 all numbers found, 1 some are not, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

BEGIN = re.compile(r"<!--\s*measured:begin\b(.*?)(-->|$)")
END = re.compile(r"<!--\s*measured:end\s*-->")
_DATE = re.compile(r"(?<![\w.#/-])\d{4}-\d{2}(?:-\d{2})?(?![\w-])")
_NUMBER = re.compile(r"(?<![\w.#/-])\d[\d,]*(?:\.\d+)?(?![\w])")
_CODE_SPAN = re.compile(r"`[^`]*`")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_COMMENT = re.compile(r"<!--.*?-->")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T.*)?$|^\d{4}-\d{2}$")


class CheckError(Exception):
    """The check could not run: no measured section, a bad header, an unreadable source."""


@dataclass(frozen=True)
class Section:
    """One measured section: its name, its sources, and its lines as ``(line number, text)``."""

    name: str
    sources: tuple[str, ...]
    lines: tuple[tuple[int, str], ...] = field(repr=False)


# ---- reading a README ------------------------------------------------------------------------


def sections(text: str) -> list[Section]:
    """The measured sections of ``text``.

    Raises:
        CheckError: A header is not closed by ``-->``, or a section is never closed.
    """
    found: list[Section] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        opening = BEGIN.search(lines[index])
        if not opening:
            index += 1
            continue
        header, closed = opening.group(1), opening.group(2) == "-->"
        start = index + 1
        while not closed:
            index += 1
            if index >= len(lines):
                raise CheckError(f"a measured:begin header at line {start} is never closed by -->")
            head, sep, _ = lines[index].partition("-->")
            header += " " + head
            closed = bool(sep)
        name, *sources = header.split()
        body: list[tuple[int, str]] = []
        index += 1
        while index < len(lines) and not END.search(lines[index]):
            body.append((index + 1, lines[index]))
            index += 1
        if index >= len(lines):
            raise CheckError(f"the measured section {name!r} is never closed")
        found.append(Section(name, tuple(sources), tuple(body)))
        index += 1
    return found


def numbers_in(section: Section) -> list[tuple[int, str]]:
    """The checked figures of a section, as ``(line number, figure as written)``, in order."""
    found: list[tuple[int, str]] = []
    fenced = False
    for number, line in section.lines:
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for _, token in _tokens(line):
            found.append((number, token))
    return found


def _tokens(line: str) -> list[tuple[int, str]]:
    """The figures of one line with their positions: dates first, then other numbers."""
    for pattern in (_COMMENT, _CODE_SPAN, _LINK_TARGET, _URL):
        line = pattern.sub(lambda m: " " * len(m.group(0)), line)
    marker = _LIST_MARKER.match(line)
    if marker:
        line = " " * marker.end() + line[marker.end() :]
    found = [(m.start(), m.group(0)) for m in _DATE.finditer(line)]
    line = _DATE.sub(lambda m: " " * len(m.group(0)), line)
    for match in _NUMBER.finditer(line):
        token = match.group(0).rstrip(",")
        percent = line[match.start() + len(token) : match.start() + len(token) + 1] == "%"
        found.append((match.start(), token + "%" if percent else token))
    return sorted(found)


# ---- the sources -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Values:
    """What a section's sources hold.

    Attributes:
        numbers: The numbers whose key does not say ``percent``: shown without a percent sign.
        percents: The numbers under a key that says ``percent``: shown with one.
        dates: The date and timestamp strings.
    """

    numbers: frozenset[Decimal]
    percents: frozenset[Decimal]
    dates: frozenset[str]

    def of(self, token: str) -> frozenset[Decimal]:
        """The numbers a figure may be: the percents when it is shown with ``%``."""
        return self.percents if token.endswith("%") else self.numbers


def _leaves(node: Any, key: str) -> Iterator[tuple[Any, str]]:
    """Each leaf under ``node`` with the last object key on its path (``key`` at the top)."""
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _leaves(value, name)
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value, key)
    else:
        yield node, key


Loaded = dict[Path, Any]
"""JSON documents already read during one run of the check, by path."""


def _load(path: Path, loaded: Loaded) -> Any:
    if path in loaded:
        return loaded[path]
    try:
        loaded[path] = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CheckError(f"cannot read {path}: {error.strerror}") from error
    except json.JSONDecodeError as error:
        raise CheckError(f"{path} is not valid JSON: {error}") from error
    return loaded[path]


def _subtrees(document: Any, source: str, dotted: str) -> list[tuple[Any, str]]:
    """The nodes a dotted path names, each with the last object key on its path.

    ``*`` stands for every key of an object or item of a list; a node it reaches that does not go
    on to the rest of the path is skipped. A path that names nothing at all is an error.
    """
    nodes: list[tuple[Any, str]] = [(document, "")]
    for key in dotted.split(".") if dotted else []:
        found: list[tuple[Any, str]] = []
        for node, last in nodes:
            if key == "*" and isinstance(node, dict):
                found.extend((value, name) for name, value in node.items())
            elif key == "*" and isinstance(node, list):
                found.extend((value, last) for value in node)
            elif isinstance(node, dict) and key in node:
                found.append((node[key], key))
            elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
                found.append((node[int(key)], last))
        if not found:
            raise CheckError(f"{source}: there is no {dotted} in it")
        nodes = found
    return nodes


def source_values(root: Path, sources: Sequence[str], loaded: Loaded | None = None) -> Values:
    """The numeric values and date strings under each source.

    Raises:
        CheckError: A source file cannot be read, or its path is not in it.
    """
    numbers: set[Decimal] = set()
    percents: set[Decimal] = set()
    dates: set[str] = set()
    cache = {} if loaded is None else loaded
    for source in sources:
        name, _, dotted = source.partition("#")
        for node, top in _subtrees(_load((root / name).resolve(), cache), source, dotted):
            for leaf, key in _leaves(node, top):
                if isinstance(leaf, bool):
                    continue
                if isinstance(leaf, int | float):
                    kind = percents if "percent" in key else numbers
                    kind.add(Decimal(repr(leaf)).normalize())
                elif isinstance(leaf, str) and _ISO_DATE.match(leaf):
                    dates.add(leaf)
    return Values(frozenset(numbers), frozenset(percents), frozenset(dates))


def _places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def _is_date_of(token: str, date: str) -> bool:
    """Whether a date as written is ``date``, or its month or day, from a date or timestamp."""
    return date.startswith(token) and (len(date) == len(token) or date[len(token)] in "-T")


def accepts(token: str, values: Values) -> bool:
    """Whether a figure as written is a value of the sources, exactly or rounded half up."""
    if _DATE.fullmatch(token):
        return any(_is_date_of(token, date) for date in values.dates)
    written = _decimal(token)
    if written.normalize() in values.of(token):
        return True
    return _rounded_from(written, values.of(token))


def _decimal(token: str) -> Decimal:
    return Decimal(token.rstrip("%").replace(",", ""))


def _rounded_from(written: Decimal, numbers: frozenset[Decimal]) -> bool:
    """Whether a nonzero figure is a value with more places, rounded half up to the ones shown."""
    if written == 0:
        return False
    shown = _places(written)
    step = Decimal(1).scaleb(-shown)
    return any(
        _places(value) > shown and value.quantize(step, rounding=ROUND_HALF_UP) == written
        for value in numbers
    )


def is_exact_value(token: str, values: Values) -> bool:
    """Whether a figure as written is exactly a value (or a date) of the sources."""
    if _DATE.fullmatch(token):
        return any(_is_date_of(token, date) for date in values.dates)
    return _decimal(token).normalize() in values.of(token)


# ---- the check -------------------------------------------------------------------------------


def check_text(text: str, name: str, root: Path, loaded: Loaded | None = None) -> list[str]:
    """One message per figure of a measured section that is not a value of its sources.

    Raises:
        CheckError: The text has no measured section, a section names no source, a header or a
            section is not closed, or a source cannot be read.
    """
    found = sections(text)
    if not found:
        raise CheckError(f"{name} has no measured section (<!-- measured:begin NAME SOURCE -->)")
    problems: list[str] = []
    cache = {} if loaded is None else loaded
    for section in found:
        if not section.sources:
            raise CheckError(f"{name}: the measured section {section.name!r} names no source")
        values = source_values(root, section.sources, cache)
        shown = ", ".join(section.sources)
        problems.extend(
            f"{name}:{number}: {token} (section {section.name}) is not a value of {shown}"
            for number, token in numbers_in(section)
            if not accepts(token, values)
        )
    return problems


def check(readme: Path, root: Path) -> list[str]:
    """:func:`check_text` on a README file, named by its path relative to ``root``."""
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError as error:
        raise CheckError(f"cannot read {readme}: {error.strerror}") from error
    resolved = readme.resolve()
    try:
        name = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        name = readme.name
    return check_text(text, name, root)


def default_readmes(root: Path) -> list[Path]:
    """The top-level README and every README under ``eval/``."""
    return [root / "README.md", *sorted((root / "eval").glob("**/README.md"))]


# ---- mutations -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mutation:
    """One figure changed by one in its last digit, and what the check made of it."""

    readme: str
    line: int
    section: str
    written: str
    changed: str
    rejected: bool
    exact: bool
    rounded: bool

    @property
    def is_source_value(self) -> bool:
        """Whether the changed figure is itself a value of the sources, as written or rounded."""
        return self.exact or self.rounded

    def describe(self) -> str:
        """The mutation as ``file:line: written -> changed``."""
        return f"{self.readme}:{self.line}: {self.written} -> {self.changed} ({self.section})"


@dataclass(frozen=True)
class MutationReport:
    """Every mutation of the checked figures."""

    mutations: tuple[Mutation, ...]

    @property
    def total(self) -> int:
        """How many mutations were made."""
        return len(self.mutations)

    @property
    def killed(self) -> int:
        """How many the check rejected."""
        return sum(m.rejected for m in self.mutations)

    @property
    def survivors(self) -> list[Mutation]:
        """The mutations the check accepted."""
        return [m for m in self.mutations if not m.rejected]


def _group(digits: str) -> str:
    whole, dot, fraction = digits.partition(".")
    return f"{int(whole):,}" + dot + fraction


def changed(token: str) -> list[str]:
    """The figure with its last digit one up and one down (when that is not negative)."""
    if _DATE.fullmatch(token):
        head, _, last = token.rpartition("-")
        return [f"{head}-{n:02d}" for n in (int(last) + 1, int(last) - 1) if 1 <= n <= 28]
    suffix = "%" if token.endswith("%") else ""
    plain = token.rstrip("%").replace(",", "")
    places = len(plain.partition(".")[2])
    units = int(plain.replace(".", ""))
    out: list[str] = []
    for n in (units + 1, units - 1):
        if n < 0:
            continue
        digits = str(n).rjust(places + 1, "0")
        text = f"{digits[:-places]}.{digits[-places:]}" if places else digits
        out.append((_group(text) if "," in token else text) + suffix)
    return out


def mutations(root: Path, readmes: Sequence[Path] | None = None) -> MutationReport:
    """Change each checked figure of the READMEs by one in its last digit, one at a time."""
    made: list[Mutation] = []
    loaded: Loaded = {}
    for readme in readmes if readmes is not None else default_readmes(root):
        text = readme.read_text(encoding="utf-8")
        name = readme.resolve().relative_to(root.resolve()).as_posix()
        lines = text.splitlines(keepends=True)
        for section in sections(text):
            values = source_values(root, section.sources, loaded)
            positions = _positions(section)
            for number, column, token in positions:
                for other in changed(token):
                    line = lines[number - 1]
                    lines[number - 1] = line[:column] + other + line[column + len(token) :]
                    problems = check_text("".join(lines), name, root, loaded)
                    lines[number - 1] = line
                    rejected = any(p.startswith(f"{name}:{number}: {other} ") for p in problems)
                    exact = is_exact_value(other, values)
                    made.append(
                        Mutation(
                            name,
                            number,
                            section.name,
                            token,
                            other,
                            rejected,
                            exact,
                            not exact and accepts(other, values),
                        )
                    )
    return MutationReport(tuple(made))


def _positions(section: Section) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    fenced = False
    for number, line in section.lines:
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            found.extend((number, column, token) for column, token in _tokens(line))
    return found


# ---- the command line ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check (or the mutations); findings print as ``file:line: number``."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None, allow_abbrev=False
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument("--mutations", action="store_true", help="report the mutation kill rate")
    parser.add_argument("readmes", nargs="*", type=Path, help="default: every README checked")
    args = parser.parse_args(argv)
    readmes = args.readmes or default_readmes(args.root)
    try:
        if args.mutations:
            report = mutations(args.root, readmes)
            print(
                f"{report.killed} of {report.total} changed figures rejected "
                f"({report.killed * 100 / report.total:.1f}%)"
                if report.total
                else "no figures"
            )
            for survivor in report.survivors:
                if survivor.exact:
                    kind = "a value of its sources"
                elif survivor.rounded:
                    kind = "another value of its sources, rounded"
                else:
                    kind = "NOT a value of its sources"
                print(f"  accepted: {survivor.describe()}: {kind}")
            return 0
        problems = [problem for readme in readmes for problem in check(readme, args.root)]
    except CheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        shown = ", ".join(str(r) for r in readmes)
        print(f"ok: every number in the measured sections of {shown} is a value of its sources")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
