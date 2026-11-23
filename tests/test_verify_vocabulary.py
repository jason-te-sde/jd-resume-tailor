"""The technology vocabulary: what it finds in text, and what it must leave alone."""

import pytest

from tailor.verify.vocabulary import Vocabulary, load_vocabulary, parse_vocabulary

VOCAB = load_vocabulary()


def names(text: str) -> list[str]:
    return [mention.canonical for mention in VOCAB.find(text)]


def test_the_shipped_vocabulary_is_loaded_and_sizeable() -> None:
    assert len(VOCAB.canonical_names) >= 300
    assert {"Redis", "PostgreSQL", "Kubernetes", "PyTorch"} <= VOCAB.canonical_names


def test_a_mention_points_at_the_words_in_the_text() -> None:
    text = "Cached reads in Redis and wrote to Postgres."
    found = VOCAB.find(text)
    assert [(m.canonical, text[m.start : m.end]) for m in found] == [
        ("Redis", "Redis"),
        ("PostgreSQL", "Postgres"),
    ]


def test_aliases_resolve_to_one_canonical_name() -> None:
    assert names("Postgres and PostgreSQL and postgres") == ["PostgreSQL"] * 3
    assert names("k8s") == ["Kubernetes"]
    assert names("scikit-learn and sklearn") == ["scikit-learn"] * 2
    assert names("Apache Kafka") == ["Kafka"]


def test_matching_ignores_case_except_where_the_vocabulary_says_not_to() -> None:
    assert names("we run kubernetes and REDIS") == ["Kubernetes", "Redis"]
    assert names("Wrote it in Go.") == ["Go"]
    assert names("Let it go and spark joy; the rest of it") == []
    assert names("Apache Spark and PySpark") == ["Spark", "Spark"]


def test_the_longest_name_wins() -> None:
    assert names("Added Redis Cluster support") == ["Redis Cluster"]
    assert names("Used Redis and Redis Streams") == ["Redis", "Redis Streams"]
    assert names("AWS Lambda functions") == ["AWS Lambda"]


@pytest.mark.parametrize(
    "text",
    ["Redis-backed limits", "Redis, Kafka", "(Redis)", "in Redis.", "Redis/Postgres"],
)
def test_punctuation_next_to_a_name_does_not_hide_it(text: str) -> None:
    assert "Redis" in names(text)


def test_a_name_inside_a_longer_word_is_not_a_mention() -> None:
    assert names("Redisson and Gopher and Pythonic") == []
    assert names("a gRPC call") == ["gRPC"]
    assert names("a C++ and C# codebase") == ["C++", "C#"]
    assert names("on .NET and ASP.NET") == [".NET", "ASP.NET"]


def test_names_with_symbols_and_spaces_match() -> None:
    assert names("Weights & Biases and Hugging Face") == ["Weights & Biases", "Hugging Face"]
    assert names("Node.js and Next.js") == ["Node.js", "Next.js"]


def test_parsing_reads_canonical_aliases_and_the_case_flag() -> None:
    vocab = parse_vocabulary("# comment\n\nFoo | Bar | baz qux\n=Case | Sens\n")
    assert isinstance(vocab, Vocabulary)
    assert vocab.canonical_names == {"Foo", "Case"}
    assert [m.canonical for m in vocab.find("bar BAZ QUX foo")] == ["Foo", "Foo", "Foo"]
    assert [m.canonical for m in vocab.find("case Case sens SENS Sens")] == ["Case", "Case"]


def test_canonical_of_a_name_is_none_when_unknown() -> None:
    assert VOCAB.canonical("postgres") == "PostgreSQL"
    assert VOCAB.canonical("Bananas") is None


def test_every_line_of_the_shipped_file_parses_to_distinct_names() -> None:
    seen: dict[str, str] = {}
    from importlib.resources import files

    for raw in files("tailor.verify").joinpath("technologies.txt").read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        canonical, *aliases = [part.strip() for part in line.lstrip("=").split("|")]
        for name in (canonical, *aliases):
            key = name.lower() if not line.startswith("=") else name
            assert seen.setdefault(key, canonical) == canonical, f"{name!r} is on two lines"
