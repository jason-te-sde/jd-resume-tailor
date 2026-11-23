r"""Turn user text into LaTeX that always compiles.

Everything a person wrote (a bullet, a name, a skill) reaches the document through
:func:`escape_latex` or :func:`escape_url`, and nowhere else: the template holds no text of its
own to escape and the renderer never builds LaTeX out of unescaped input. The property tests
compile what these functions return.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import quote

_SPECIAL: Final = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    # Typographic characters that plain pdfLaTeX would otherwise stop on or draw badly.
    chr(0x2018): "`",
    chr(0x2019): "'",
    chr(0x201C): "``",
    chr(0x201D): "''",
    chr(0x2013): "--",
    chr(0x2014): "---",
    chr(0x2026): r"\ldots{}",
    chr(0x2022): r"\textbullet{}",
}
_ARROW: Final = "->"
_ARROW_TEX: Final = r"$\rightarrow$"
_MATH: Final = {
    chr(0xD7): r"$\times$",
    chr(0x2248): r"$\approx$",
    chr(0x2264): r"$\leq$",
    chr(0x2265): r"$\geq$",
}
_SPACE: Final = re.compile(r"\s+")
# What is left to drop once whitespace is gone: C0 and C1 controls, zero-width and bidi marks,
# the word joiner and the byte-order mark. Built with chr() so no invisible character is written
# into this file.
_INVISIBLE_RANGES: Final = (
    (0x00, 0x1F),
    (0x7F, 0x9F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2060),
    (0xFEFF, 0xFEFF),
)
_INVISIBLE: Final = re.compile(
    "[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _INVISIBLE_RANGES) + "]"
)
_URL_SAFE: Final = "-._:/?=;,+@!*'()[]%#"
_SCHEMES: Final = ("http://", "https://", "mailto:")


def escape_latex(text: str) -> str:
    r"""Return ``text`` as LaTeX that draws the same characters and cannot break a document.

    Every LaTeX special character is escaped or replaced by its text command, whitespace of any
    kind (a newline, a blank line) becomes one space so the result is safe inside the argument of
    a macro, and invisible control characters are dropped. ``->`` becomes an arrow; ``--`` is left
    alone and prints as an en dash, which is how a date range is written. Characters outside ASCII
    that have no entry here are passed through: pdfLaTeX can draw the Latin ones, and the compiler
    retries with XeLaTeX for the rest.
    """
    cleaned = _INVISIBLE.sub("", _SPACE.sub(" ", text)).strip()
    out: list[str] = []
    index = 0
    while index < len(cleaned):
        if cleaned.startswith(_ARROW, index):
            out.append(_ARROW_TEX)
            index += len(_ARROW)
            continue
        char = cleaned[index]
        out.append(_SPECIAL.get(char) or _MATH.get(char) or char)
        index += 1
    return "".join(out)


def is_link(url: str) -> bool:
    """Whether ``url`` is one :func:`escape_url` will turn into a link (http, https, mailto)."""
    return url.lower().startswith(_SCHEMES)


def escape_url(url: str) -> str:
    r"""Return ``url`` as the argument of ``\href``: percent-encoded where TeX would choke.

    ``%`` and ``#`` stay as the URL's own characters but are written with a backslash, which is
    what ``\href`` expects inside a macro argument. Everything else outside a conservative set of
    URL characters is percent-encoded, so no brace, backslash, dollar sign or space survives.
    """
    # quote() never encodes "~", but an active tilde in a macro argument is not what \href wants.
    encoded = quote(_SPACE.sub("", url), safe=_URL_SAFE).replace("~", "%7E")
    return encoded.replace("%", r"\%").replace("#", r"\#")
