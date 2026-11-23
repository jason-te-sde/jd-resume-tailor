"""Tokens, stems and sentences: what the claim checks compare."""

import pytest

from tailor.verify.text import (
    content_tokens,
    sentences,
    stem,
    tokenize,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("cache", "caching"),
        ("cached", "caches"),
        ("replace", "replacing"),
        ("batch", "batched"),
        ("batches", "batching"),
        ("query", "queries"),
        ("commit", "committing"),
        ("commit", "commits"),
        ("add", "adding"),
        ("add", "added"),
        ("run", "running"),
        ("deduplicate", "deduplicates"),
        ("deploy", "deployment"),
        ("build", "built"),
        ("write", "wrote"),
        ("own", "ownership"),
        ("own", "owned"),
        ("latency", "latencies"),
    ],
)
def test_inflected_forms_share_a_stem(left: str, right: str) -> None:
    assert stem(left) == stem(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("production", "product"),
        ("real", "reel"),
        ("user", "used"),
        ("write", "read"),
        ("load", "loss"),
    ],
)
def test_unrelated_words_do_not_share_a_stem(left: str, right: str) -> None:
    assert stem(left) != stem(right)


def test_synonyms_share_a_stem() -> None:
    assert stem("reduced") == stem("cut")
    assert stem("lowered") == stem("cutting")
    assert stem("implemented") == stem("built")


def test_tokens_carry_their_position_in_the_text() -> None:
    text = "Cut p99 latency by 3,781 events/s"
    tokens = tokenize(text)
    assert [t.word for t in tokens] == ["cut", "p99", "latency", "by", "3781", "events", "s"]
    for token in tokens:
        assert text[token.start : token.end].lower().replace(",", "") == token.word


def test_hyphenated_words_split_and_keep_their_parts() -> None:
    words = [t.word for t in tokenize("group-committing read-through exactly-once")]
    assert words == ["group", "committing", "read", "through", "exactly", "once"]


def test_a_decimal_is_one_token() -> None:
    assert [t.word for t in tokenize("p99 of 2.1 ms")] == ["p99", "of", "2.1", "ms"]


def test_content_tokens_drop_stop_words_but_keep_negation() -> None:
    words = [t.word for t in content_tokens("a team of the six did not own it")]
    assert words == ["team", "six", "not", "own"]


def test_sentences_split_at_semicolons_and_sentence_ends() -> None:
    text = "ORIGIN: my notes. EVIDENCE: bullet 1 is a run; bullet 2 is a test, not traffic. Done"
    assert sentences(text) == [
        "ORIGIN: my notes.",
        "EVIDENCE: bullet 1 is a run",
        "bullet 2 is a test, not traffic.",
        "Done",
    ]


def test_a_decimal_point_or_an_abbreviation_does_not_end_a_sentence() -> None:
    text = "Median of 3.5 minutes, e.g. over 5 runs. Next one"
    assert sentences(text) == ["Median of 3.5 minutes, e.g. over 5 runs.", "Next one"]


def test_a_dollar_sign_before_a_number_is_the_word_dollar() -> None:
    words = [t.word for t in tokenize("saved $50 a year")]
    assert words == ["saved", "dollar", "50", "a", "year"]
