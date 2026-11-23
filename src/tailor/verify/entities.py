"""Find the names in a bullet that must be backed by its sources: technologies and proper nouns.

Four kinds of name are looked for:

* ``technology``: a name in the curated vocabulary (``technologies.txt``), in any case unless the
  vocabulary says otherwise;
* ``identifier``: a token with a capital inside it or written in capitals (``FastAPI``, ``gRPC``,
  ``HIPAA``), unless it is a common acronym or unit (``API``, ``CI``, ``MB``);
* ``name``: two or more capitalised words in a row (``Acme Logistics``);
* ``name``: a single capitalised word that is not the first of its sentence (``Contoso``).

The first word of a sentence is capitalised whatever it is, so it never counts by itself.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tailor.verify.text import word_set
from tailor.verify.vocabulary import Mention, Vocabulary, load_vocabulary

EntityKind = Literal["technology", "identifier", "name"]


_WORD: Final = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_SENTENCE_END: Final = re.compile(r"[.!?;:]\s*$")

# Acronyms and units that name a concept, not a tool or a standard someone could claim to have used.
GENERIC_ACRONYMS: Final = word_set(
    """
    API APIS CI CD CRUD ETL ELT ML AI UI UX QA KPI KPIS SLA SLAS SLO SLI OKR CPU CPUS GPU GPUS RAM
    SSD JSON CSV XML HTML CSS URL URLS HTTP HTTPS TCP UDP DNS TLS SSL SSH IP VM VMS OS IO LLM LLMS
    NLP PR PRS MVP RPC CLI SDK IDE WAL LSM MAPE WMAPE SKU SKUS TF IDF PDF OLTP OLAP ACID LRU TTL
    QPS RPS TPS DAG DAGS ORM MVC B2B SEO NOSQL SAAS PAAS IAAS UTC USD KB MB GB TB KIB MIB GIB TIB
    HZ FPS RPC UUID ID IDS
    """
)
_NOT_NAMES: Final = word_set(
    """
    January February March April May June July August September October November December Monday
    Tuesday Wednesday Thursday Friday Saturday Sunday I
    """
)


@dataclass(frozen=True, slots=True)
class Entity:
    """A name found in a text.

    Attributes:
        text: The name as written.
        start: Index of its first character.
        end: Index after its last character.
        kind: ``technology``, ``identifier`` or ``name``.
        mention: For a technology, the vocabulary match (canonical name and case rule).
    """

    text: str
    start: int
    end: int
    kind: EntityKind
    mention: Mention | None = None


def _is_sentence_start(text: str, index: int) -> bool:
    return not text[:index].strip() or bool(_SENTENCE_END.search(text[:index]))


def _is_identifier(word: str) -> bool:
    return len(word) >= 2 and any(char.isupper() for char in word[1:])


def _is_capitalised(word: str) -> bool:
    return len(word) >= 2 and word[0].isupper() and word[1:].islower()


def extract_entities(text: str, vocabulary: Vocabulary | None = None) -> list[Entity]:
    """Return the names in ``text`` that a source would have to back, in reading order."""
    vocab = vocabulary or load_vocabulary()
    entities = [
        Entity(text[m.start : m.end], m.start, m.end, "technology", m) for m in vocab.find(text)
    ]
    covered = [(entity.start, entity.end) for entity in entities]

    def is_covered(start: int, end: int) -> bool:
        return any(start < c_end and c_start < end for c_start, c_end in covered)

    run: list[re.Match[str]] = []

    def flush() -> None:
        # A run of capitalised words is one name; at a sentence start its first word is just the
        # sentence's first word.
        words = run[1:] if run and _is_sentence_start(text, run[0].start()) else list(run)
        words = [w for w in words if w.group() not in _NOT_NAMES]
        if words:
            entities.append(
                Entity(
                    text[words[0].start() : words[-1].end()],
                    words[0].start(),
                    words[-1].end(),
                    "name",
                )
            )
        run.clear()

    for match in _WORD.finditer(text):
        word = match.group()
        if is_covered(match.start(), match.end()):
            flush()
            continue
        if _is_identifier(word) and word.upper() not in GENERIC_ACRONYMS:
            flush()
            entities.append(Entity(word, match.start(), match.end(), "identifier"))
            continue
        gap_is_space = bool(run) and text[run[-1].end() : match.start()].isspace()
        if _is_capitalised(word) and (not run or gap_is_space):
            run.append(match)
            continue
        flush()
        if _is_capitalised(word):
            run.append(match)
    flush()
    return sorted(entities, key=lambda entity: entity.start)


def _loose(word: str) -> str:
    word = word.lower()
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


@dataclass(frozen=True, slots=True)
class KnownNames:
    """The names a set of texts supports: the words they use and the technologies they name."""

    words: frozenset[str]
    technologies: frozenset[str]

    @classmethod
    def of(cls, texts: Sequence[str], vocabulary: Vocabulary | None = None) -> KnownNames:
        """Collect the words and technologies named anywhere in ``texts``."""
        vocab = vocabulary or load_vocabulary()
        joined = "\n".join(texts)
        return cls(
            frozenset(_loose(word) for word in _WORD.findall(joined)),
            frozenset(mention.canonical for mention in vocab.find(joined)),
        )

    def supports(self, entity: Entity) -> bool:
        """Whether the texts back this name: by canonical technology, or by every word of it."""
        if entity.mention and entity.mention.canonical in self.technologies:
            return True
        if entity.mention and entity.mention.case_sensitive:
            return False
        return all(_loose(word) in self.words for word in _WORD.findall(entity.text))
