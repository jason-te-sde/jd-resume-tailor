"""The rounding rules: which way a number may move, by unit polarity, under both policies."""

import pytest

from tailor.verify.quantities import extract_quantities
from tailor.verify.rounding import Comparison, Polarity, Rounding, compare, polarity


def cmp(
    claimed: str,
    source: str,
    *,
    pol: Polarity = "higher",
    rounding: Rounding = "conservative",
) -> Comparison:
    """Compare the first quantity of ``claimed`` with the first of ``source``."""
    return compare(
        extract_quantities(claimed)[0], extract_quantities(source)[0], pol, rounding=rounding
    )


def verdict(
    claimed: str, source: str, *, pol: Polarity = "higher", rounding: Rounding = "conservative"
) -> str:
    return cmp(claimed, source, pol=pol, rounding=rounding).verdict


# ---- equality and conversions ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("claimed", "source"),
    [
        ("3781", "3,781"),
        ("3.7K", "3,700"),
        ("8,000", "8K"),
        ("138 ms", "0.138 s"),
        ("1.5 min", "90 s"),
        ("20K checks/s", "20,000 checks per second"),
        ("87.5%", "0.875"),
        ("0.875", "87.5%"),
        ("87%", "0.87"),
        ("2.5x", "2.5\u00d7"),
    ],
)
def test_the_same_quantity_written_differently_is_exact(claimed: str, source: str) -> None:
    assert verdict(claimed, source) == "exact"


@pytest.mark.parametrize(
    ("claimed", "source"),
    [
        ("5 ms", "138 ms"),
        ("9,200", "9,200 events/s"),
        ("9,200 events/s", "9,200"),
        ("50%", "50"),
        ("3x", "3"),
        ("2 weeks", "2 services"),
        ("1 ms", "1 s"),
        ("4", "0.4"),
    ],
)
def test_other_families_and_other_values_do_not_match(claimed: str, source: str) -> None:
    assert verdict(claimed, source) == "mismatch"


# ---- rounding direction: higher is better ----------------------------------------------------


def test_rounding_down_a_higher_is_better_value_is_fine() -> None:
    assert verdict("87%", "0.875") == "rounded"
    assert verdict("0.87", "0.875") == "rounded"
    assert verdict("3.7K", "3,781") == "rounded"
    assert verdict("3.0x", "3.08x") == "rounded"


def test_rounding_up_a_higher_is_better_value_is_drift() -> None:
    assert verdict("88%", "0.875") == "drift"
    assert verdict("0.88", "0.875") == "drift"
    assert verdict("3.8K", "3,781") == "drift"
    assert verdict("3.1x", "3.08x") == "drift"


def test_drift_says_which_way_it_moved_and_what_the_source_says() -> None:
    comparison = cmp("88%", "0.875")
    assert "up" in comparison.reason
    assert "higher is better" in comparison.reason


# ---- rounding direction: lower is better -----------------------------------------------------


def test_rounding_up_a_lower_is_better_value_is_fine() -> None:
    assert verdict("30 s", "29.5 s", pol="lower") == "rounded"
    assert verdict("140 ms", "138 ms", pol="lower") == "rounded"
    assert verdict("0.5 s", "0.46 s", pol="lower") == "rounded"


def test_rounding_down_a_lower_is_better_value_is_drift() -> None:
    assert verdict("29 s", "29.5 s", pol="lower") == "drift"
    assert verdict("130 ms", "138 ms", pol="lower") == "drift"
    assert verdict("0.4 s", "0.46 s", pol="lower") == "drift"


def test_the_same_pair_flips_with_the_polarity() -> None:
    assert verdict("29", "29.5", pol="higher") == "rounded"
    assert verdict("29", "29.5", pol="lower") == "drift"
    assert verdict("30", "29.5", pol="higher") == "drift"
    assert verdict("30", "29.5", pol="lower") == "rounded"


def test_a_baseline_flips_the_flattering_direction() -> None:
    # In "from 412 ms to 138 ms" a smaller starting point makes the improvement look smaller.
    baseline = extract_quantities("from 410 ms to 138 ms")[0]
    source = extract_quantities("412 ms")[0]
    assert baseline.baseline
    assert compare(baseline, source, "lower", rounding="conservative").verdict == "rounded"
    bigger = extract_quantities("from 420 ms to 138 ms")[0]
    assert compare(bigger, source, "lower", rounding="conservative").verdict == "drift"


# ---- what counts as a rounding at all --------------------------------------------------------


def test_a_different_number_that_is_not_a_rounding_of_the_source_is_a_mismatch() -> None:
    assert verdict("135 ms", "138 ms", pol="lower") == "mismatch"
    assert verdict("86%", "0.875") == "mismatch"
    assert verdict("0.9", "0.875") == "drift"


def test_a_value_may_not_be_more_precise_than_its_source() -> None:
    assert verdict("29.52 s", "29.5 s", pol="lower") == "mismatch"


def test_coarse_roundings_follow_the_same_direction_rule() -> None:
    assert verdict("70 tests", "76 tests") == "rounded"
    assert verdict("80 tests", "76 tests") == "drift"
    assert verdict("20 services", "14 services") == "drift"


# ---- the nearest policy ----------------------------------------------------------------------


def test_nearest_accepts_ordinary_rounding_at_two_significant_figures() -> None:
    assert verdict("88%", "0.875", rounding="nearest") == "rounded"
    assert verdict("0.88", "0.875", rounding="nearest") == "rounded"
    assert verdict("3.1x", "3.08x", rounding="nearest") == "rounded"
    assert verdict("3.8K", "3,781", rounding="nearest") == "rounded"
    assert verdict("29 s", "29.5 s", pol="lower", rounding="nearest") == "rounded"
    assert verdict("30 s", "29.5 s", pol="lower", rounding="nearest") == "rounded"


def test_nearest_still_refuses_what_is_not_the_nearest_rounding() -> None:
    assert verdict("88%", "0.871", rounding="nearest") == "drift"
    assert verdict("89%", "0.875", rounding="nearest") == "mismatch"
    assert verdict("3.2x", "3.08x", rounding="nearest") == "mismatch"


def test_nearest_does_not_accept_a_single_significant_figure() -> None:
    assert verdict("4K", "3,781", rounding="nearest") == "drift"
    assert verdict("90%", "0.875", rounding="nearest") == "drift"


def test_nearest_never_relaxes_the_plus_rule() -> None:
    assert verdict("3.8K+", "3,781", rounding="nearest") == "drift"
    assert verdict("8K+", "8,000", rounding="nearest") == "drift"


# ---- plus ------------------------------------------------------------------------------------


def test_a_plus_below_the_source_is_fine() -> None:
    assert verdict("3.7K+", "3,781") == "rounded"
    assert verdict("70+ tests", "76 tests") == "rounded"
    assert verdict("1K+", "3,781") == "rounded"


def test_a_plus_may_not_reach_the_source() -> None:
    assert verdict("3,781+", "3,781") == "drift"
    assert verdict("3.8K+", "3,781") == "drift"
    assert verdict("8K+", "8,000") == "drift"
    assert "exact" in cmp("8K+", "8,000").reason


def test_a_plus_may_equal_a_source_that_is_itself_a_lower_bound() -> None:
    assert verdict("8K+", "8,000+") == "exact"
    assert verdict("5K+", "8,000+") == "rounded"


def test_a_plus_far_below_the_source_is_a_different_number() -> None:
    assert verdict("100+", "38,000") == "mismatch"


def test_at_least_works_like_a_plus() -> None:
    assert verdict("at least 8,000", "8,000") == "drift"
    assert verdict("at least 7,000", "8,000") == "rounded"


# ---- up to -----------------------------------------------------------------------------------


def test_dropping_up_to_is_drift() -> None:
    assert verdict("5 hours", "up to 5 hours") == "drift"
    assert verdict("up to 5 hours", "up to 5 hours") == "exact"
    assert verdict("up to 5 hours", "5 hours") == "exact"


# ---- polarity --------------------------------------------------------------------------------


def polarity_of(text: str, number_text: str) -> str:
    quantity = next(q for q in extract_quantities(text) if q.text == number_text)
    return polarity(text, quantity)


@pytest.mark.parametrize(
    ("text", "number", "expected"),
    [
        ("Cut p99 latency to 138 ms", "138 ms", "lower"),
        ("Sustained 9,200 events/s", "9,200 events/s", "higher"),
        ("Form completion rose to 74%", "74%", "higher"),
        ("Kept the rate within 2% of the limit", "2%", "lower"),
        ("Pages fell from 31 to 9 over two quarters", "9", "lower"),
        ("Cut the error rate to 1%", "1%", "lower"),
        ("Reached macro-F1 of 0.87", "0.87", "higher"),
        ("Bundle shrank to 238 KB", "238 KB", "lower"),
        ("Wrote 76 tests", "76", "higher"),
        ("Cut MAPE from 27.4% to 19.8%", "19.8%", "lower"),
        ("Raised accuracy from 71% to 87%", "87%", "higher"),
    ],
)
def test_polarity_follows_the_unit_and_the_words_around_the_number(
    text: str, number: str, expected: str
) -> None:
    assert polarity_of(text, number) == expected


def test_durations_and_sizes_default_to_lower_and_everything_else_to_higher() -> None:
    assert polarity_of("Ran for 5 s", "5 s") == "lower"
    assert polarity_of("An image of 310 MB", "310 MB") == "lower"
    assert polarity_of("Reached 3x", "3x") == "higher"
    assert polarity_of("Served 41,000 requests", "41,000") == "higher"


def test_a_comparison_after_a_comma_does_not_set_the_polarity_of_the_number_before_it() -> None:
    text = "Brought compaction down to 29s from 91 s, a 3.08x speed-up."
    assert polarity_of(text, "29s") == "lower"
