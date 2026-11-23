"""Checks on a compiled resume: what LaTeX does not warn about but a reader sees.

Most checks read the PDF through ``pdftotext -bbox`` (poppler), which is optional: without it
those checks are skipped and the report says so. The checks that read the ``.tex`` or the log
need nothing.

* **Page count**, from the LaTeX log, and **overfull boxes**, from the same log.
* **Bullet gaps**: a bullet that fills its line exactly can push the trailing space of the item
  onto a line of its own, leaving an empty row before the next bullet. LaTeX reports nothing for
  it, so it is found from the geometry: two bullets of one list further apart than a line.
* **Wrapped bullets**: a bullet followed by a line at the text indent instead of a new bullet.
* **Page fill**: the lowest line of text over the usable height of the page.
* **ATS text**: the PDF must give back enough words, or an applicant tracking system reads nothing.
* **Repeated openers**: bullets of one entry that start with the same word.
* **Unescaped characters** in the generated ``.tex``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from html import unescape
from itertools import pairwise
from pathlib import Path
from typing import Final

from tailor.render.model import ResumePlan

BOTTOM_MARGIN_PT: Final = 36.0  # the template's bottom margin: half an inch, in PDF points
MIN_ATS_WORDS: Final = 40
MIN_ATS_CHARS: Final = 200
_BULLET: Final = "\u2022"
_WORD: Final = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)">([^<]*)</word>'
)
_PAGE: Final = re.compile(r'<page width="([\d.]+)" height="([\d.]+)"')
_OVERFULL: Final = re.compile(r"Overfull \\hbox")
_SPECIAL: Final = ("&", "%", "#", "$", "_")


@dataclass(frozen=True, slots=True)
class Unescaped:
    """A special character in the document body that LaTeX would read as a command."""

    line: int
    char: str


@dataclass(frozen=True, slots=True)
class RepeatedOpener:
    """Bullets of one entry that start with the same word."""

    entry: str
    word: str
    count: int


@dataclass(frozen=True, slots=True)
class LayoutReport:
    """Everything the checks found. ``None`` means the check could not run."""

    pages: int | None
    overfull: int
    gaps: tuple[str, ...] | None
    wrapped: tuple[str, ...] | None
    fill: float | None
    words: int | None
    ats_ok: bool | None
    repeated_openers: tuple[RepeatedOpener, ...]
    unescaped: tuple[Unescaped, ...]
    skipped: tuple[str, ...] = field(default=())

    @property
    def one_page(self) -> bool:
        """Whether the document is known to be a single page."""
        return self.pages == 1


def overfull_count(log: str) -> int:
    r"""How many ``Overfull \hbox`` warnings the log holds: lines that run past the margin."""
    return len(_OVERFULL.findall(log))


def find_unescaped(tex: str) -> list[Unescaped]:
    r"""Find special characters in the body that no backslash protects.

    Skips comments, macro definitions (``#1``), table rows (``&`` is the cell separator), math
    ``$...$`` on one line, and the argument of ``\href``, where ``_`` is legal.
    """
    body = tex.split(r"\begin{document}", 1)[-1]
    offset = tex.count("\n") - body.count("\n") + 1
    found: list[Unescaped] = []
    for number, raw in enumerate(body.splitlines(), start=offset):
        line = raw.strip()
        if line.startswith("%"):
            continue
        # Drop what is escaped, what is math and the url of a link; what is left must be clean.
        line = re.sub(r"\\[&%$#_{}\\]", "", line)
        line = re.sub(r"\$[^$]*\$", "", line)
        line = re.sub(r"\\href\{[^{}]*\}", r"\\href{}", line)
        if re.search(r"#\d", line) or "tabular" in line:
            continue
        found += [Unescaped(number, char) for char in _SPECIAL if char in line]
    return found


@dataclass(frozen=True, slots=True)
class _Word:
    x: float
    y_max: float
    height: float
    text: str


@dataclass(frozen=True, slots=True)
class _Page:
    width: float
    height: float
    words: tuple[_Word, ...]


def parse_bbox(xml: str) -> list[_Page]:
    """Read the output of ``pdftotext -bbox`` into pages of words with their geometry."""
    pages: list[_Page] = []
    for chunk in xml.split("<page")[1:]:
        size = _PAGE.match("<page" + chunk)
        words = tuple(
            _Word(float(x), float(y1), float(y1) - float(y0), unescape(text))
            for x, y0, y1, text in _WORD.findall(chunk)
        )
        pages.append(
            _Page(float(size.group(1)), float(size.group(2)), words) if size else _Page(0, 0, words)
        )
    return pages


def _lines(page: _Page) -> list[tuple[float, list[tuple[float, str]], float]]:
    """Cluster words into lines by baseline; bold and regular glyphs differ slightly in yMax."""
    clusters: list[tuple[float, list[tuple[float, str]], list[float]]] = []
    for word in sorted(page.words, key=lambda w: (w.y_max, w.x)):
        if clusters and word.y_max - clusters[-1][0] < 4.5:  # a bullet sits ~3.4pt above the base
            clusters[-1][1].append((word.x, word.text))
            clusters[-1][2].append(word.height)
        else:
            clusters.append((word.y_max, [(word.x, word.text)], [word.height]))
    return [(y, sorted(words), max(heights)) for y, words, heights in clusters]


def bullet_layout(xml: str) -> tuple[list[str], list[str]]:
    """Find bullets followed by an empty row and bullets that run onto a second line.

    Returns ``(gaps, wrapped)``: the start of the text of each such bullet.

    A line is a bullet when its first word is the bullet glyph. A bullet is followed by a gap when
    the next line is a bullet at the same indent and further than 1.7 line heights away; it wraps
    when the next line sits at a deeper indent than the glyph and is no further than that.
    """
    gaps: list[str] = []
    wrapped: list[str] = []
    for page in parse_bbox(xml):
        lines = _lines(page)
        bullet_x: float | None = None
        for (y0, words0, height), (y1, words1, _) in pairwise(lines):
            if words0[0][1] == _BULLET:
                bullet_x = words0[0][0]
            if bullet_x is None or words0[0][0] < bullet_x - 1:
                bullet_x = None
                continue
            next_is_bullet = words1[0][1] == _BULLET and abs(words1[0][0] - bullet_x) < 2
            near = (y1 - y0) < 1.7 * height
            if next_is_bullet and not near:
                gaps.append(" ".join(w for _, w in words0)[:90])
            if (
                words0[0][1] == _BULLET
                and not next_is_bullet
                and words1[0][0] > bullet_x + 2
                and near
            ):
                wrapped.append(" ".join(w for _, w in words0[1:])[:90])
    return gaps, wrapped


def page_fill(xml: str) -> float | None:
    """The bottom of the lowest text on the first page, as a share of the usable height."""
    pages = parse_bbox(xml)
    if not pages or not pages[0].words or pages[0].height <= BOTTOM_MARGIN_PT:
        return None
    page = pages[0]
    return max(w.y_max for w in page.words) / (page.height - BOTTOM_MARGIN_PT)


def repeated_openers(plan: ResumePlan) -> list[RepeatedOpener]:
    """Bullets of one entry whose first word is the same (two entries that both start ``Built``)."""
    found: list[RepeatedOpener] = []
    for entry in plan.entries:
        openers = Counter(
            b.text.split()[0].lower().strip(",.;:") for b in entry.bullets if b.text.split()
        )
        found += [
            RepeatedOpener(entry.title, word.capitalize(), count)
            for word, count in sorted(openers.items())
            if count > 1
        ]
    return found


def pdf_bbox(pdf: Path) -> str | None:
    """Run ``pdftotext -bbox`` on a PDF; ``None`` when poppler is not installed or fails."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        done = subprocess.run(
            ["pdftotext", "-bbox", str(pdf), "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return done.stdout if done.returncode == 0 else None


def check_layout(
    tex: str, pdf: Path, log: str, plan: ResumePlan, *, pages: int | None
) -> LayoutReport:
    """Run every check on a compiled document and return what they found.

    ``pages`` is the count LaTeX reported; when it is ``None`` the count is taken from the PDF if
    pdftotext is available.
    """
    skipped: list[str] = []
    xml = pdf_bbox(pdf)
    gaps = wrapped = fill = words = ats_ok = None
    if xml is None:
        skipped.append("pdftotext not found: bullet gaps, wrapped bullets, page fill and ATS text")
    else:
        found_gaps, found_wrapped = bullet_layout(xml)
        gaps, wrapped = tuple(found_gaps), tuple(found_wrapped)
        fill = page_fill(xml)
        if pages is None:
            pages = len(parse_bbox(xml))
        text_words = [w.text for page in parse_bbox(xml) for w in page.words]
        words = len(text_words)
        ats_ok = words >= MIN_ATS_WORDS and len("".join(text_words)) >= MIN_ATS_CHARS
    return LayoutReport(
        pages=pages,
        overfull=overfull_count(log),
        gaps=gaps,
        wrapped=wrapped,
        fill=fill,
        words=words,
        ats_ok=ats_ok,
        repeated_openers=tuple(repeated_openers(plan)),
        unescaped=tuple(find_unescaped(tex)),
        skipped=tuple(skipped),
    )
