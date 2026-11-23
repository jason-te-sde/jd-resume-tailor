"""The number check: what it lets through, what it calls fabricated, and what it calls drift."""

from collections.abc import Sequence

import pytest

from tailor.verify.checks import check_numbers
from tailor.verify.findings import Finding
from tailor.verify.rounding import Rounding

LATENCY = "Cut p99 latency of the tracking API from 412 ms to 138 ms with a batched query."
RECALL = "Reached a recall of 0.875 on the rarest of 12 categories."
COMPACTION = "Compaction of the 310 GB table takes 29.5 s, down from 41 s."
THROUGHPUT = "Handled 41,000 checks/s on one 4-vCPU VM, with a p99 of 2.1 ms at 20,000 checks/s."
EVENTS = "Sustained 9,200 events/s on a 6-partition topic; 3,781 distinct carriers."


def kinds(bullet: str, sources: Sequence[str], rounding: Rounding = "conservative") -> list[str]:
    found = check_numbers(bullet, sources, rounding=rounding)
    return [finding.kind for finding in found]


def only(bullet: str, sources: Sequence[str], rounding: Rounding = "conservative") -> Finding:
    found = check_numbers(bullet, sources, rounding=rounding)
    assert len(found) == 1, found
    return found[0]


def test_numbers_that_match_the_sources_pass() -> None:
    bullet = "Reduced p99 latency from 412 ms to 138 ms using a batched query."
    assert check_numbers(bullet, [LATENCY]) == []


def test_a_bullet_without_numbers_has_nothing_to_check() -> None:
    assert check_numbers("Built a rate limiter in Rust.", [LATENCY]) == []


def test_a_number_the_sources_do_not_contain_is_fabricated() -> None:
    finding = only("Cut p99 latency of the tracking API to 90 ms.", [LATENCY])
    assert finding.kind == "fabricated_number"
    assert finding.text == "90 ms"
    assert "time" in finding.evidence


def test_the_span_of_a_finding_points_at_the_number() -> None:
    bullet = "Handled 50,000 checks/s on one VM."
    finding = only(bullet, [THROUGHPUT])
    assert bullet[finding.start : finding.end] == finding.text == "50,000 checks/s"


def test_a_number_in_another_unit_family_is_not_a_match() -> None:
    finding = only("Processed 9,200 events from the topic.", [EVENTS])
    assert finding.kind == "fabricated_number"
    assert "events/s" in finding.evidence or "rate" in finding.evidence


def test_a_number_may_match_any_of_the_cited_sources() -> None:
    assert check_numbers("Cut latency to 138 ms and reached 0.875.", [LATENCY, RECALL]) == []


def test_rounding_up_a_higher_is_better_number_is_drift() -> None:
    finding = only("Reached 88% recall on the rarest category.", [RECALL])
    assert finding.kind == "number_drift"
    assert finding.text == "88%"
    assert "0.875" in finding.evidence


def test_rounding_down_is_fine_for_the_same_number() -> None:
    assert check_numbers("Reached 87% recall on the rarest category.", [RECALL]) == []
    assert check_numbers("Reached 0.87 recall on the rarest category.", [RECALL]) == []


def test_the_percentage_and_the_fraction_are_the_same_number() -> None:
    assert check_numbers("Reached 87.5% recall on the rarest category.", [RECALL]) == []


def test_rounding_down_a_lower_is_better_number_is_drift() -> None:
    drifted = "Cut compaction time from 41 s to 29s."
    finding = only(drifted, [COMPACTION])
    assert (finding.kind, finding.text) == ("number_drift", "29s")
    assert "lower is better" in finding.evidence


def test_rounding_up_a_lower_is_better_number_is_fine() -> None:
    assert check_numbers("Cut compaction time from 41 s to 30 s.", [COMPACTION]) == []


def test_the_same_drift_is_accepted_under_nearest() -> None:
    assert kinds("Reached 88% recall on the rarest category.", [RECALL], "nearest") == []
    assert kinds("Cut compaction time from 41 s to 29s.", [COMPACTION], "nearest") == []


def test_nearest_still_flags_a_number_that_is_not_a_rounding() -> None:
    assert kinds("Reached 90% recall on the rarest category.", [RECALL], "nearest") == [
        "number_drift"
    ]
    assert kinds("Reached 70% recall on the rarest category.", [RECALL], "nearest") == [
        "fabricated_number"
    ]


def test_a_plus_may_not_reach_or_pass_the_source() -> None:
    sources = ["Served 3,781 distinct carriers."]
    assert check_numbers("Served 3.7K+ carriers.", sources) == []
    assert kinds("Served 3.8K+ carriers.", sources) == ["number_drift"]
    assert kinds("Served 3,781+ carriers.", sources) == ["number_drift"]


def test_a_plus_on_an_exact_source_is_drift() -> None:
    sources = ["Loaded exactly 8,000 documents."]
    finding = only("Loaded 8K+ documents.", sources)
    assert finding.kind == "number_drift"
    assert "exact" in finding.evidence


def test_up_to_a_rounded_up_best_case_is_drift() -> None:
    sources = ["The best case was a 3.08x speed-up."]
    assert kinds("Delivered up to 3.1x speed-up.", sources) == ["number_drift"]
    assert kinds("Delivered up to 3.1x speed-up.", sources, "nearest") == []


def test_dropping_a_sources_up_to_is_drift() -> None:
    sources = ["Removed up to 5 hours of staleness."]
    assert kinds("Removed 5 hours of staleness.", sources) == ["number_drift"]
    assert check_numbers("Removed up to 5 hours of staleness.", sources) == []


def test_both_ends_of_a_range_are_checked() -> None:
    sources = ["Speed-ups of 1.2 to 3.08x across the suite."]
    assert check_numbers("A 1.2-3.08x speed-up.", sources) == []
    finding = only("A 1.2-3.1x speed-up.", sources)
    assert (finding.kind, finding.text) == ("number_drift", "3.1x")


def test_the_starting_point_of_a_comparison_may_round_in_its_own_direction() -> None:
    sources = ["p99 went from 412 ms to 138 ms."]
    assert check_numbers("p99 went from 410 ms to 138 ms.", sources) == []
    assert kinds("p99 went from 420 ms to 138 ms.", sources) == ["number_drift"]


def test_an_arrow_pair_is_read_like_from_to() -> None:
    sources = ["Cut deploy time from 38 -> 11 minutes."]
    assert check_numbers("Cut deploy time from 38 to 11 minutes.", sources) == []
    assert kinds("Cut deploy time from 38 to 9 minutes.", sources) == ["fabricated_number"]


def test_every_bad_number_gets_its_own_finding() -> None:
    found = check_numbers("Cut latency from 500 ms to 90 ms.", [LATENCY])
    assert [f.text for f in found] == ["500 ms", "90 ms"]


def test_the_closest_source_value_is_named_in_the_evidence() -> None:
    finding = only("Reached 88% recall on the rarest category.", [RECALL, LATENCY])
    assert "0.875" in finding.evidence
    assert "412" not in finding.evidence


def test_numbers_inside_identifiers_are_not_checked() -> None:
    assert check_numbers("Reported p99.9 and p50 latency.", [LATENCY]) == []


# ---- the size of a reduction: higher flatters ------------------------------------------------

REDUCTION = (
    "Reduced latency, load time, error rate, memory use, cost and build time of the nightly job "
    "by 40.6% over 30 runs."
)
FEWER = "Ended with 28.3% fewer failed runs over 8 weeks."


def test_a_percent_reduction_rounded_up_is_drift() -> None:
    finding = only("Reduced p99 latency by 41% over 30 runs.", [REDUCTION])
    assert finding.kind == "number_drift"
    assert finding.text == "41%"


def test_a_percent_reduction_rounded_down_passes() -> None:
    assert check_numbers("Reduced p99 latency by 40% over 30 runs.", [REDUCTION]) == []


def test_a_percent_reduction_written_exactly_passes() -> None:
    assert check_numbers("Cut load time by 40.6% over 30 runs.", [REDUCTION]) == []


@pytest.mark.parametrize(
    "bullet",
    [
        "Cut error rate by 41% over 30 runs.",
        "Lowered memory use by 41% over 30 runs.",
        "Slashed build time by 41% over 30 runs.",
        "Trimmed cost by 41% over 30 runs.",
        "A 41% reduction in latency over 30 runs.",
        "A 41% decrease in latency over 30 runs.",
        "Latency 41% lower over 30 runs.",
        "41% less latency over 30 runs.",
    ],
)
def test_every_phrasing_of_a_reduction_has_the_same_polarity(bullet: str) -> None:
    assert kinds(bullet, [REDUCTION]) == ["number_drift"]


def test_n_percent_fewer_rounded_up_is_drift_and_rounded_down_passes() -> None:
    assert kinds("Had 29% fewer failed runs over 8 weeks.", [FEWER]) == ["number_drift"]
    assert kinds("Had 28% fewer failed runs over 8 weeks.", [FEWER]) == []
    assert kinds("Had 29% less failed work over 8 weeks.", [FEWER]) == ["number_drift"]


def test_a_reduction_between_two_endpoints_keeps_the_endpoint_rule() -> None:
    source = "Cut weighted MAPE from 27.4% to 19.8% against a last-week baseline."
    assert check_numbers("Cut weighted MAPE from 27.4% to 20% against a baseline.", [source]) == []
    assert kinds("Cut weighted MAPE from 27.4% to 19% against a baseline.", [source]) == [
        "number_drift"
    ]
