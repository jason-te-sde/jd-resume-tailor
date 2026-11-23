"""Reduce HTML to the plain text a person would read."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Final

_SKIP: Final = frozenset(
    {"script", "style", "noscript", "template", "svg", "head", "nav", "footer"}
)
_BLOCK: Final = frozenset(
    {
        "p",
        "div",
        "br",
        "section",
        "article",
        "ul",
        "ol",
        "table",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "main",
        "blockquote",
    }
)


class _Reducer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    """Return the text of ``markup``: blocks on their own lines, list items as ``- `` lines."""
    reducer = _Reducer()
    reducer.feed(markup)
    reducer.close()
    text = "".join(reducer.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def unescape_html(markup: str) -> str:
    """Undo one level of entity escaping (Greenhouse sends its HTML escaped)."""
    return html.unescape(markup)
