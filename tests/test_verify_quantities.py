"""Quantity extraction: examples for each supported form, then properties over the formats."""

from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tailor.verify.quantities import Quantity, extract_quantities


def one(text: str) -> Quantity:
    found = extract_quantities(text)
    assert len(found) == 1, [(q.text, q.family, q.value) for q in found]
    return found[0]


def values(text: str) -> list[Fraction]:
    return [q.value for q in extract_quantities(text)]


def families(text: str) -> list[str]:
    return [q.family for q in extract_quantities(text)]


# ---- forms -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12 tests", "12"),
        ("3,781 events", "3781"),
        ("1,234,567 rows", "1234567"),
        ("0.875 recall", "0.875"),
        ("about 29.5 units", "29.5"),
    ],
)
def test_plain_integers_decimals_and_thousands_separators(text: str, value: str) -> None:
    quantity = one(text)
    assert quantity.value == Fraction(value)
    assert quantity.family == "count"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("3.7K rows", "3700"),
        ("3.7k rows", "3700"),
        ("2M rows", "2000000"),
        ("1.5B rows", "1500000000"),
        ("2 million rows", "2000000"),
        ("48 million rows", "48000000"),
        ("5 thousand users", "5000"),
        ("1.2 billion rows", "1200000000"),
    ],
)
def test_magnitude_suffixes_and_words(text: str, value: str) -> None:
    quantity = one(text)
    assert quantity.value == Fraction(value)
    assert quantity.family == "count"


def test_a_percent_keeps_its_written_value() -> None:
    for text in ("74%", "74 %", "74 percent"):
        quantity = one(text)
        assert (quantity.value, quantity.family, quantity.unit) == (Fraction(74), "percent", "%")
    assert one("87.5% of a core").value == Fraction("87.5")


def test_multipliers() -> None:
    assert (one("2.5x faster").value, one("2.5x faster").family) == (Fraction("2.5"), "multiplier")
    assert one("8x the configured rate").value == Fraction(8)
    assert one("3\u00d7 faster").family == "multiplier"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("138 ms", "0.138"),
        ("138ms", "0.138"),
        ("2.1 ms", "0.0021"),
        ("29.5 s", "29.5"),
        ("29s", "29"),
        ("5 sec", "5"),
        ("5 seconds", "5"),
        ("11 min", "660"),
        ("38 minutes", "2280"),
        ("6 h", "21600"),
        ("6 hours", "21600"),
        ("3 weeks", "1814400"),
    ],
)
def test_time_units_are_normalised_to_seconds(text: str, seconds: str) -> None:
    quantity = one(text)
    assert quantity.family == "time"
    assert quantity.value == Fraction(seconds)


def test_a_hyphenated_unit_is_a_unit() -> None:
    assert one("a 10-minute test").value == Fraction(600)
    assert one("a 3-week A/B test").family == "time"
    assert one("5-minute runs").family == "time"


def test_a_word_that_merely_starts_with_a_unit_is_not_a_unit() -> None:
    assert one("4,800 sessions").family == "count"
    assert one("12 msgs").family == "count"
    assert one("6 hourly jobs").family == "count"


@pytest.mark.parametrize(
    "text",
    [
        "9,200 events/s",
        "9,200 events/sec",
        "9,200 events per second",
        "9,200 req/s",
        "9,200 ops/s",
        "9,200 jobs/sec",
        "9,200 checks/s",
        "9.2K events/s",
        "9.2K/s",
    ],
)
def test_throughput_units(text: str) -> None:
    quantity = one(text)
    assert quantity.family == "rate"
    assert quantity.value == Fraction(9200)


def test_throughput_per_minute_is_normalised_to_per_second() -> None:
    assert one("300 jobs/min").value == Fraction(5)
    assert one("18,000 jobs per hour").value == Fraction(5)


def test_a_frame_rate_is_a_throughput() -> None:
    assert (one("58 fps").family, one("58 fps").value) == ("rate", Fraction(58))


def test_a_duration_per_period_is_a_duration_not_a_throughput() -> None:
    quantity = one("saving 6 hours per week")
    assert quantity.family == "time"
    assert quantity.value == Fraction(21600)


def test_sizes() -> None:
    assert one("612 KB").family == "size"
    assert one("310 GB").value == Fraction(310) * 10**9
    assert one("a 310 MB image").value == Fraction(310) * 10**6
    assert one("2.4 GB").value == Fraction("2.4e9")


def test_dollar_amounts() -> None:
    quantity = one("saving $1.2M a year")
    assert (quantity.family, quantity.value) == ("money", Fraction(1_200_000))
    assert one("$40K+").plus is True
    assert one("$5").value == Fraction(5)


def test_a_trailing_plus_is_kept_apart_from_the_value() -> None:
    for text in ("3.7K+ events", "3.7K+", "50+ tests"):
        quantity = one(text)
        assert quantity.plus is True
    assert one("3.7K+ events").value == Fraction(3700)
    assert one("3.7K events").plus is False
    assert one("40%+").plus is True


def test_a_range_gives_both_ends_with_the_shared_unit() -> None:
    found = extract_quantities("a 1.2-2.5x speed-up")
    assert [(q.value, q.family) for q in found] == [
        (Fraction("1.2"), "multiplier"),
        (Fraction("2.5"), "multiplier"),
    ]
    found = extract_quantities("between 2,844 to 4,250 req/s")
    assert [(q.value, q.family) for q in found] == [
        (Fraction(2844), "rate"),
        (Fraction(4250), "rate"),
    ]
    found = extract_quantities("50 to 80 ms")
    assert [q.value for q in found] == [Fraction("0.05"), Fraction("0.08")]


def test_an_arrow_pair_gives_both_ends_and_marks_the_first_as_the_baseline() -> None:
    found = extract_quantities("398ms -> 34ms")
    assert [q.value for q in found] == [Fraction("0.398"), Fraction("0.034")]
    assert [q.baseline for q in found] == [True, False]
    found = extract_quantities("38 -> 11 minutes")
    assert [(q.value, q.family) for q in found] == [
        (Fraction(2280), "time"),
        (Fraction(660), "time"),
    ]
    found = extract_quantities("from 412 to 138 ms")
    assert [(q.value, q.family) for q in found] == [
        (Fraction("0.412"), "time"),
        (Fraction("0.138"), "time"),
    ]
    assert found[0].baseline


def test_a_comparison_baseline_is_marked() -> None:
    found = extract_quantities("reaching 0.87 against 0.71 for a baseline")
    assert [q.baseline for q in found] == [False, True]
    found = extract_quantities("cut it from 27.4% to 19.8%")
    assert [q.baseline for q in found] == [True, False]
    assert not any(q.baseline for q in extract_quantities("sustained 9,200 events/s"))


def test_a_bare_range_is_not_a_baseline() -> None:
    assert not any(q.baseline for q in extract_quantities("2,844 to 4,250 req/s"))
    assert not any(q.baseline for q in extract_quantities("1.2-2.5x"))


@pytest.mark.parametrize(
    "text",
    [
        "p99 latency",
        "p99.9 latency",
        "the 99th percentile",
        "macro-F1 and k6 and S3",
        "version v1.2",
        "an x86 host",
        "HTTP/2 only",
    ],
)
def test_identifiers_and_ordinals_with_digits_are_not_quantities(text: str) -> None:
    assert extract_quantities(text) == []


def test_numbers_beside_identifiers_still_count() -> None:
    assert values("Redis 7.2 and 7.4") == [Fraction("7.2"), Fraction("7.4")]
    assert values("p99 of 2.1 ms at 20,000 checks/s") == [Fraction("0.0021"), Fraction(20000)]


def test_qualifiers_before_a_number_are_recorded() -> None:
    assert one("removing up to 5 hours").qualifier == "up to"
    assert one("at least 5 hours").qualifier == "at least"
    assert one("about 6 weeks").qualifier == "about"
    assert one("~6 weeks").qualifier == "~"
    assert one("within 2%").qualifier == "within"
    assert one("5 hours").qualifier is None


def test_over_is_not_a_qualifier_because_it_often_means_across() -> None:
    assert one("an A/B test over 4,800 sessions").qualifier is None


def test_precision_is_the_unit_of_the_last_written_digit() -> None:
    assert one("29.5 s").precision_fine == Fraction("0.1")
    assert one("0.875").precision_fine == Fraction("0.001")
    assert one("87.5%").precision_fine == Fraction("0.1")
    assert one("3.7K").precision_fine == Fraction(100)
    assert one("8K").precision_fine == Fraction(1000)
    assert one("3,781").precision_fine == Fraction(1)
    assert one("2.5 min").precision_fine == Fraction(6)


def test_coarse_precision_reads_trailing_zeros_of_a_whole_number_as_padding() -> None:
    assert one("8,000").precision_fine == Fraction(1)
    assert one("8,000").precision_coarse == Fraction(1000)
    assert one("70").precision_coarse == Fraction(10)
    assert one("8.0").precision_coarse == Fraction("0.1")
    assert one("80K").precision_coarse == Fraction(10000)
    assert one("0").precision_coarse == Fraction(1)


def test_the_text_and_span_point_back_into_the_input() -> None:
    sentence = "Cut p99 from 412 ms to 138 ms and handled 9,200 events/s, up 3.7K+ in 2.5x runs."
    for quantity in extract_quantities(sentence):
        assert sentence[quantity.start : quantity.end] == quantity.text
        assert quantity.text.strip() == quantity.text
    assert any(q.text == "9,200 events/s" for q in extract_quantities(sentence))
    assert any(q.text == "3.7K+" for q in extract_quantities(sentence))


def test_text_with_no_numbers_has_no_quantities() -> None:
    assert extract_quantities("Built a rate limiter in Rust and Redis.") == []
    assert extract_quantities("") == []


# ---- properties ------------------------------------------------------------------------------

FILLER = [
    "handled",
    "services",
    "across",
    "the",
    "requests",
    "latency",
    "with",
    "tests",
    "and",
    "reached",
    "on",
    "a",
]
words = st.lists(st.sampled_from(FILLER), min_size=0, max_size=4).map(" ".join)

whole = st.integers(min_value=0, max_value=10**9)
decimals = st.builds(
    lambda n, places: Decimal(n).scaleb(-places),
    st.integers(min_value=0, max_value=10**7),
    st.integers(min_value=0, max_value=3),
)
magnitudes = st.sampled_from([("K", 10**3), ("k", 10**3), ("M", 10**6), ("B", 10**9)])
TIME = [
    ("ms", Fraction(1, 1000)),
    ("s", Fraction(1)),
    ("sec", Fraction(1)),
    ("seconds", Fraction(1)),
    ("min", Fraction(60)),
    ("minutes", Fraction(60)),
    ("h", Fraction(3600)),
    ("hours", Fraction(3600)),
]
RATE = [
    ("events/s", Fraction(1)),
    ("req/s", Fraction(1)),
    ("ops/s", Fraction(1)),
    ("jobs/sec", Fraction(1)),
    ("checks per second", Fraction(1)),
    ("/s", Fraction(1)),
    ("jobs/min", Fraction(1, 60)),
]


def around(core: str, before: str, after: str) -> str:
    return " ".join(part for part in (before, core, after) if part)


@given(whole, st.booleans())
def test_a_whole_number_round_trips_with_or_without_separators(n: int, grouped: bool) -> None:
    text = f"{n:,}" if grouped else str(n)
    assert [q.value for q in extract_quantities(text)] == [Fraction(n)]


@given(decimals, st.booleans())
def test_a_decimal_round_trips(value: Decimal, grouped: bool) -> None:
    text = f"{value:,f}" if grouped else f"{value:f}"
    assert [q.value for q in extract_quantities(text)] == [Fraction(value)]


@given(decimals, magnitudes)
def test_a_magnitude_suffix_round_trips(value: Decimal, suffix: tuple[str, int]) -> None:
    letter, factor = suffix
    found = extract_quantities(f"{value:f}{letter}")
    assert [q.value for q in found] == [Fraction(value) * factor]


@given(decimals)
def test_a_percent_round_trips(value: Decimal) -> None:
    found = extract_quantities(f"{value:f}%")
    assert [(q.value, q.family) for q in found] == [(Fraction(value), "percent")]


@given(decimals)
def test_a_multiplier_round_trips(value: Decimal) -> None:
    found = extract_quantities(f"{value:f}x")
    assert [(q.value, q.family) for q in found] == [(Fraction(value), "multiplier")]


@given(decimals, st.sampled_from(TIME), st.booleans())
def test_a_duration_round_trips(value: Decimal, unit: tuple[str, Fraction], spaced: bool) -> None:
    label, seconds = unit
    found = extract_quantities(f"{value:f}{' ' if spaced else ''}{label}")
    assert [(q.value, q.family) for q in found] == [(Fraction(value) * seconds, "time")]


@given(decimals, st.sampled_from(RATE))
def test_a_throughput_round_trips(value: Decimal, unit: tuple[str, Fraction]) -> None:
    label, per_second = unit
    sep = "" if label.startswith("/") else " "
    found = extract_quantities(f"{value:f}{sep}{label}")
    assert [(q.value, q.family) for q in found] == [(Fraction(value) * per_second, "rate")]


@given(decimals, magnitudes)
def test_a_dollar_amount_round_trips(value: Decimal, suffix: tuple[str, int]) -> None:
    letter, factor = suffix
    found = extract_quantities(f"${value:f}{letter}")
    assert [(q.value, q.family) for q in found] == [(Fraction(value) * factor, "money")]


@given(decimals, st.sampled_from(["", "K", "%", "x", " ms", " events/s"]))
def test_a_trailing_plus_never_changes_the_value(value: Decimal, unit: str) -> None:
    plain = extract_quantities(f"{value:f}{unit}")
    plus = extract_quantities(f"{value:f}{unit}+")
    assert [q.value for q in plus] == [q.value for q in plain]
    assert all(q.plus for q in plus)


UNITS = st.sampled_from(["x", "%", " ms", " req/s", ""])


@given(whole, whole, UNITS, st.sampled_from(["-", " to "]))
def test_a_range_round_trips_both_ends(low: int, high: int, unit: str, joiner: str) -> None:
    found = extract_quantities(f"{low:,}{joiner}{high:,}{unit}")
    alone = extract_quantities(f"{low:,}{unit}")
    other = extract_quantities(f"{high:,}{unit}")
    assert [q.value for q in found] == [alone[0].value, other[0].value]
    assert found[0].family == found[1].family == other[0].family


@given(decimals, decimals, st.sampled_from(TIME))
def test_an_arrow_pair_round_trips(
    first: Decimal, second: Decimal, unit: tuple[str, Fraction]
) -> None:
    label, seconds = unit
    found = extract_quantities(f"{first:f}{label} -> {second:f}{label}")
    assert [q.value for q in found] == [Fraction(first) * seconds, Fraction(second) * seconds]


@given(
    st.one_of(
        decimals.map(lambda v: f"{v:f}"),
        decimals.map(lambda v: f"{v:f}%"),
        decimals.map(lambda v: f"{v:f}x"),
        decimals.map(lambda v: f"{v:f} ms"),
        decimals.map(lambda v: f"{v:f} req/s"),
        decimals.map(lambda v: f"${v:f}M"),
    ),
    words,
    words,
)
def test_surrounding_words_never_change_what_is_read(core: str, before: str, after: str) -> None:
    alone = [(q.value, q.family, q.plus) for q in extract_quantities(core)]
    inside = [(q.value, q.family, q.plus) for q in extract_quantities(around(core, before, after))]
    assert inside == alone


def test_a_person_unit_compound_is_a_duration() -> None:
    assert one("saving 6 analyst-hours a week").value == Fraction(21600)
    assert one("saving 6 analyst-hours a week").family == "time"
    assert one("about 2 person-days").family == "time"


def test_the_word_simulated_may_sit_between_a_number_and_its_unit() -> None:
    assert one("a median of 312 simulated ms").value == Fraction(312, 1000)
    assert one("312 simulated ms").text == "312 simulated ms"
