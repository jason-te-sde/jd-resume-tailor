import pytest
from hypothesis import given
from hypothesis import strategies as st

from tailor.bank.latex import normalize_latex


@pytest.mark.parametrize(
    ("raw", "plain"),
    [
        (r"cut errors by 38\%", "cut errors by 38%"),
        (r"Search \& ranking", "Search & ranking"),
        (r"\textasciitilde{}4 weeks", "~4 weeks"),
        (r"\textasciitilde4 weeks", "~4 weeks"),
        (r"queue $\rightarrow$ worker", "queue -> worker"),
        (r"queue $\to$ worker", "queue -> worker"),
        (r"worker $\leftarrow$ queue", "worker <- queue"),
        (r"\textbf{3.2x} faster", "3.2x faster"),
        (r"an \textit{idempotent} \emph{consumer} in \texttt{Go}", "an idempotent consumer in Go"),
        (r"snake\_case and item \#3", "snake_case and item #3"),
        (r"costs \$5 \{a\}", "costs $5 {a}"),
        (r"2$\times$ faster, $\approx$10 ms, $\sim$5 ms", "2x faster, ~10 ms, ~5 ms"),
        (r"p99 $\leq$ 50 ms and $\geq$ 99 ok", "p99 <= 50 ms and >= 99 ok"),
        (r"10$^{6}$ rows and 2$^{12}$ keys", "10^6 rows and 2^12 keys"),
        (r"2\textasciicircum{}20 bytes", "2^20 bytes"),
        (r"1.2--2.5x and a --- b", "1.2-2.5x and a - b"),
        (r"``quoted'' text", '"quoted" text'),
        (r"line one\\line two", "line one line two"),
        ("many   spaces\n and\tlines ", "many spaces and lines"),
    ],
)
def test_latex_constructs_become_plain_text(raw: str, plain: str) -> None:
    assert normalize_latex(raw) == plain


def test_one_bullet_with_every_escape_the_bank_uses() -> None:
    raw = (
        r"Cut p99 by 70\% \& saved \textbf{\textasciitilde{}3 hrs} a week "
        r"$\rightarrow$ 10$^{6}$ rows."
    )
    assert normalize_latex(raw) == "Cut p99 by 70% & saved ~3 hrs a week -> 10^6 rows."


@pytest.mark.parametrize(
    "text",
    [
        "p99 < 5 ms, 3.5K req/s, 12% of 1,200 users",
        "email me at a@example.com; see https://example.com/a_b?x=1&y=2",
        "tilde ~ and 100% and AT&T and $5 and 2^3 and a single - hyphen",
        r"unknown \foo{bar} commands stay",
    ],
)
def test_plain_text_is_left_alone(text: str) -> None:
    assert normalize_latex(text) == text


def test_nested_commands_unwrap_from_the_inside_out() -> None:
    assert normalize_latex(r"\textbf{a \textit{b \texttt{c}}} d") == "a b c d"


def test_an_escape_inside_a_command_is_still_an_escape() -> None:
    assert normalize_latex(r"\textbf{50\%}") == "50%"


def test_a_double_backslash_is_not_the_start_of_an_escape() -> None:
    # Without this, a first pass would leave "\%" behind and a second pass would change it again.
    assert normalize_latex(r"a\\%b") == "a %b"


def test_an_unclosed_command_is_kept_as_written() -> None:
    assert normalize_latex(r"\textbf{unclosed") == r"\textbf{unclosed"


LATEX_FRAGMENTS = [
    "\\", "\\\\", "%", "&", "$", "{", "}", "_", "#", "~", "^", "-", "--", "---", "`", "``", "'",
    "''", " ", "\n", "\t", "a", "7", ".", "\\%", "\\&", "\\$", "\\{", "\\}", "\\_", "\\#",
    "\\textbf{", "\\textit{", "\\emph{", "\\texttt{", "\\textasciitilde{}", "\\textasciicircum{}",
    "\\rightarrow", "\\to", "\\times", "$\\rightarrow$", "$\\times$", "$^{", "}$", "$\\leq$",
]  # fmt: skip


@given(st.text())
def test_normalizing_twice_equals_normalizing_once(text: str) -> None:
    once = normalize_latex(text)
    assert normalize_latex(once) == once


@given(st.lists(st.sampled_from(LATEX_FRAGMENTS), max_size=24).map("".join))
def test_normalizing_twice_equals_normalizing_once_on_latex_shaped_text(text: str) -> None:
    once = normalize_latex(text)
    assert normalize_latex(once) == once, f"{text!r} -> {once!r} -> {normalize_latex(once)!r}"


@given(st.text(alphabet=st.characters(blacklist_characters="\\$-`'", blacklist_categories=["Cs"])))
def test_text_without_latex_syntax_only_loses_extra_whitespace(text: str) -> None:
    assert normalize_latex(text) == " ".join(text.split())
