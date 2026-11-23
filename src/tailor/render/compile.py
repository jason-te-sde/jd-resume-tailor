"""Compile a ``.tex`` file with pdfLaTeX, retrying with XeLaTeX on a font or Unicode error."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tailor.render.model import InputError

ENGINES: Final = ("pdflatex", "xelatex")
TIMEOUT_SECONDS: Final = 120
_AUX_SUFFIXES: Final = (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz")
_PAGES: Final = re.compile(r"Output written on .*?\((\d+) pages?\b")

INSTALL_HINT: Final = (
    "no TeX engine found (looked for pdflatex and xelatex on PATH). Install a TeX distribution, "
    "for example on Debian or Ubuntu: sudo apt-get install texlive-latex-base "
    "texlive-latex-extra texlive-fonts-recommended; on macOS: brew install --cask basictex."
)


class NoTeXEngineError(InputError):
    """Neither pdflatex nor xelatex is installed: a problem with the machine, not the plan."""


@dataclass(frozen=True, slots=True)
class CompileResult:
    """The outcome of compiling a document.

    Attributes:
        ok: Whether a PDF was produced.
        pdf: Its path; ``None`` when ``ok`` is false.
        engine: The engine that produced the result (the fallback one if it was retried).
        errors: What LaTeX complained about, one string per error; empty on success.
        log: The log of the last run (read before the auxiliary files are removed).
        pages: The page count LaTeX reports; ``None`` when it did not say.
        retried: Whether the first engine failed on a font or Unicode error and another ran.
    """

    ok: bool
    pdf: Path | None
    engine: str
    errors: tuple[str, ...]
    log: str
    pages: int | None
    retried: bool = False


def find_engine(preferred: str | None = None) -> str:
    """Return the engine to use: ``preferred`` if given, else pdflatex, else xelatex.

    Raises:
        NoTeXEngineError: when no engine is installed (or ``preferred`` is not).
    """
    candidates = (preferred,) if preferred else ENGINES
    for engine in candidates:
        if engine and shutil.which(engine):
            return engine
    raise NoTeXEngineError(INSTALL_HINT if preferred is None else f"{preferred} not found on PATH")


def extract_errors(log: str) -> list[str]:
    """Pull the useful lines out of a LaTeX log: each ``!`` error with the context after it."""
    lines = log.splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("!"):
            continue
        message = line.strip()
        for follow in lines[index + 1 : index + 4]:
            context = follow.strip()
            if context:
                message += "\n    " + context
            if context.startswith("l."):
                break
        errors.append(message)
    if errors:
        return errors
    markers = ("Fatal error", "Emergency stop", "Misplaced", "Undefined", "LaTeX Error")
    errors = [line.strip() for line in lines if any(marker in line for marker in markers)]
    return errors or ["LaTeX failed without an error message; see the log."]


def _is_font_trouble(errors: list[str]) -> bool:
    return any("font" in error.lower() or "unicode" in error.lower() for error in errors)


def _run(tex: Path, engine: str) -> CompileResult:
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", tex.name]
    pdf = tex.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    try:
        done = subprocess.run(
            command,
            cwd=tex.parent,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(False, None, engine, (f"{engine} timed out",), "", None)
    log_path = tex.with_suffix(".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    for suffix in _AUX_SUFFIXES:
        tex.with_suffix(suffix).unlink(missing_ok=True)
    if done.returncode == 0 and pdf.exists():
        match = _PAGES.search(log)
        return CompileResult(True, pdf, engine, (), log, int(match.group(1)) if match else None)
    return CompileResult(False, None, engine, tuple(extract_errors(log or done.stdout)), log, None)


def compile_tex(tex: Path, *, engine: str | None = None) -> CompileResult:
    """Compile ``tex`` next to itself and return the result; the PDF replaces any older one.

    The first engine is ``engine`` or the first one installed (pdflatex before xelatex). If it
    fails with an error about a font or a Unicode character and the other engine is installed, the
    document is compiled again with that one and the result says so.

    Raises:
        NoTeXEngineError: when no engine is installed.
    """
    first = find_engine(engine)
    result = _run(tex, first)
    if result.ok or engine is not None or not _is_font_trouble(list(result.errors)):
        return result
    other = next((e for e in ENGINES if e != first and shutil.which(e)), None)
    if other is None:
        return result
    retried = _run(tex, other)
    return CompileResult(
        retried.ok, retried.pdf, other, retried.errors, retried.log, retried.pages, retried=True
    )
