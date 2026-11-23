r"""Turn the LaTeX-ready strings of a fact bank into plain text.

Bank strings are written to be pasted into a LaTeX template, so they carry escapes such as ``\%``
and ``\textbf{...}``. The verifier compares what a bullet says with what the bank says, and that
comparison has to see ``38%``, not ``38\%``. Everything that reads bank text for meaning goes
through :func:`normalize_latex`; the original strings stay untouched for rendering.
"""

from __future__ import annotations

import re
from typing import Final

# What each "$\name$" math symbol becomes. Anything not listed is left as written.
_SYMBOLS: Final = {
    "rightarrow": "->",
    "to": "->",
    "leftarrow": "<-",
    "times": "x",
    "approx": "~",
    "sim": "~",
    "leq": "<=",
    "geq": ">=",
}

_WRAPPERS: Final = ("textbf", "textit", "emph", "texttt", "textsc", "textrm", "textsf", "underline")

_TOKEN: Final = re.compile(
    "|".join(
        [
            r"(?P<linebreak>\\\\)",
            r"\\(?P<escaped>[%&_#${}])",
            r"(?P<tilde>\\textasciitilde(?:\{\})?)",
            r"(?P<caret>\\textasciicircum(?:\{\})?)",
            rf"\$\\(?P<symbol>{'|'.join(_SYMBOLS)})\$",
            r"\$\^\{(?P<power>[^{}]*)\}\$",
            rf"\\(?:{'|'.join(_WRAPPERS)})\{{(?P<wrapped>[^{{}}]*)\}}",
            r"(?P<dash>-{2,})",
            r"(?P<quote>``|'')",
        ]
    )
)


def _replace(match: re.Match[str]) -> str:
    kind = match.lastgroup
    if kind == "linebreak":
        return " "
    if kind == "escaped":
        return match.group("escaped")
    if kind == "tilde":
        return "~"
    if kind == "caret":
        return "^"
    if kind == "symbol":
        return _SYMBOLS[match.group("symbol")]
    if kind == "power":
        return "^" + match.group("power")
    if kind == "wrapped":
        return match.group("wrapped")
    if kind == "dash":
        return "-"
    if kind == "quote":
        return '"'
    raise AssertionError(f"token without a handler: {match.group(0)!r}")


def _pass(text: str) -> str:
    return " ".join(_TOKEN.sub(_replace, text).split())


def normalize_latex(text: str) -> str:
    r"""Return ``text`` as plain text: LaTeX escapes resolved, whitespace collapsed.

    Handled: escaped specials (``\%`` ``\&`` ``\_`` ``\#`` ``\$`` ``\{`` ``\}``),
    ``\textasciitilde`` and ``\textasciicircum``, the math symbols in ``_SYMBOLS``, ``$^{n}$``
    superscripts, font commands such as ``\textbf{...}`` (unwrapped, nesting included), ``--`` and
    ``---`` dashes (a single hyphen, which is how a range reads in plain text), LaTeX quotes and
    ``\`` line breaks. A raw ``~`` is left alone: in a bank it is as likely to mean "about" as
    LaTeX's non-breaking space. Unknown commands are kept as written.

    The rewrite repeats until nothing changes, so the result is a fixed point: normalizing it
    again returns it unchanged. That holds even when one rewrite exposes another, as in
    ``\textbf{\%}``. No rewrite makes the text longer and each token rewrite makes it shorter,
    so the loop always ends.
    """
    current = _pass(text)
    while True:
        following = _pass(current)
        if following == current:
            return current
        current = following
