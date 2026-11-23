"""Escaping: what each special character becomes, and the property that nothing breaks."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tailor.render.escape import escape_latex, escape_url, is_link


@pytest.mark.parametrize(
    ("raw", "tex"),
    [
        ("plain text", "plain text"),
        ("38% faster", r"38\% faster"),
        ("R&D", r"R\&D"),
        ("costs $5", r"costs \$5"),
        ("item #3", r"item \#3"),
        ("snake_case", r"snake\_case"),
        ("{braces}", r"\{braces\}"),
        ("a~b", r"a\textasciitilde{}b"),
        ("2^20", r"2\textasciicircum{}20"),
        ("a<b>c", r"a\textless{}b\textgreater{}c"),
        ("a|b", r"a\textbar{}b"),
        (r"C:\dir", r"C:\textbackslash{}dir"),
        ("queue -> worker", r"queue $\rightarrow$ worker"),
        ("Sep. 2019 -- May 2023", "Sep. 2019 -- May 2023"),
        ("line one\nline two", "line one line two"),
        ("a\n\nb", "a b"),
        ("  padded \t text  ", "padded text"),
        ("", ""),
        ("zero\u200bwidth", "zerowidth"),
        ("ctrl\x00char\x1b", "ctrlchar"),
        ("\u201cquoted\u201d \u2014 and \u2026", "``quoted'' --- and \\ldots{}"),
        ("2\u00d7 faster", r"2$\times$ faster"),
    ],
)
def test_escape_latex(raw: str, tex: str) -> None:
    assert escape_latex(raw) == tex


def test_a_text_that_looks_like_a_placeholder_is_just_text() -> None:
    assert escape_latex("@@NAME@@") == "@@NAME@@"


def test_non_latin_text_passes_through_for_the_engine_retry() -> None:
    assert escape_latex("caf\u00e9 \u20ac") == "caf\u00e9 \u20ac"


@pytest.mark.parametrize(
    ("url", "tex"),
    [
        ("https://example.com/a", "https://example.com/a"),
        ("https://example.com/a%20b", r"https://example.com/a\%20b"),
        ("https://example.com/#top", r"https://example.com/\#top"),
        ("https://example.com/?a=1&b=2", r"https://example.com/?a=1\%26b=2"),
        ("https://example.com/a b", "https://example.com/ab"),
        ("https://example.com/~sam", r"https://example.com/\%7Esam"),
        ("https://example.com/{x}$y\\", r"https://example.com/\%7Bx\%7D\%24y\%5C"),
    ],
)
def test_escape_url(url: str, tex: str) -> None:
    assert escape_url(url) == tex


def test_only_web_and_mail_addresses_are_links() -> None:
    assert is_link("https://example.com")
    assert is_link("HTTP://example.com")
    assert is_link("mailto:sam@example.com")
    assert not is_link("javascript:alert(1)")
    assert not is_link("example.com")


# What an escaped string may contain: command names, escaped specials and the arrow. With these
# removed, no character that LaTeX reads as syntax may be left.
_ALLOWED = re.compile(
    r"\\textbackslash\{\}|\\textasciitilde\{\}|\\textasciicircum\{\}|\\textless\{\}"
    r"|\\textgreater\{\}|\\textbar\{\}|\\ldots\{\}|\\textbullet\{\}"
    r"|\$\\rightarrow\$|\$\\times\$|\$\\approx\$|\$\\leq\$|\$\\geq\$|\\[&%$#_{}]"
)
_SYNTAX = re.compile(r"[\\{}&%$#_^~<>|]")
_TEXT = st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=60) | st.text(
    alphabet=r"\{}&%$#_^~<>| -\n\t.[]*", max_size=60
)


@given(_TEXT)
def test_escaped_text_holds_no_syntax_character(text: str) -> None:
    left = _ALLOWED.sub("", escape_latex(text))
    assert not _SYNTAX.search(left), left


@given(_TEXT)
def test_escaped_text_is_one_line_with_no_blank_line(text: str) -> None:
    escaped = escape_latex(text)
    assert "\n" not in escaped
    assert escaped == escaped.strip()


@given(_TEXT)
def test_escape_url_leaves_nothing_tex_reads_as_syntax(url: str) -> None:
    left = re.sub(r"\\[%#]", "", escape_url(url))
    assert not re.search(r"[\\{}$&^~|<>\s]", left), left


_DOC = (
    "\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
    "\\usepackage[hidelinks]{hyperref}\n\\newcommand{\\row}[1]{\\item\\relax #1}\n"
    "\\begin{document}\n\\begin{itemize}\n%s\n\\end{itemize}\n\\end{document}\n"
)


@pytest.mark.requires_tex
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.lists(
        st.text(alphabet=st.characters(min_codepoint=9, max_codepoint=126), max_size=50),
        min_size=1,
        max_size=8,
    )
)
def test_escaped_text_always_compiles(tmp_path: Path, rows: list[str]) -> None:
    body = "\n".join(
        f"\\row{{{escape_latex(text)}}}\n\\href{{{escape_url('https://example.com/' + text)}}}"
        f"{{{escape_latex(text)}}}"
        for text in rows
    )
    tex = tmp_path / "doc.tex"
    tex.write_text(_DOC % body, encoding="utf-8")
    done = subprocess.run(
        [
            shutil.which("pdflatex") or "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex.name,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, (rows, done.stdout[-800:])
