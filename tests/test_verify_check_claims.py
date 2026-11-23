"""The do-not-claim check: phrase overlap with what an entry must not claim."""

from tailor.verify.checks import check_do_not_claim
from tailor.verify.claims import key_phrases

OWNERSHIP = "ownership of the tracking platform (a team of 6 owned it; the author owned the API)"
EXACTLY_ONCE = "'exactly-once' processing (the consumer is at-least-once with deduplication)"
PRODUCTION = "138 ms as a production p99 (it is a staging load test)"
COSTS = "any cost savings (no cost data was collected)"
USERS = "production use or real users (it has only run on test machines)"
ZERO = "'zero downtime' deploys (the pipeline does rolling deploys)"
FOLLOWER = "strict global limits while Redis is down (the fallback limits are per process)"

SOURCES = [
    "Built an idempotent Kafka consumer that deduplicates by event id; a staging load test "
    "of the tracking API, not production traffic."
]


def flagged(bullet: str, claims: list[str], sources: list[str] | None = None) -> list[str]:
    found = check_do_not_claim(bullet, claims, SOURCES if sources is None else sources)
    assert all(f.kind == "do_not_claim" for f in found)
    return [f.text for f in found]


# ---- key phrases -----------------------------------------------------------------------------


def test_the_explanation_in_parentheses_is_not_part_of_the_claim() -> None:
    phrases = key_phrases(OWNERSHIP)
    assert [p.text for p in phrases] == ["ownership of the tracking platform"]


def test_a_quoted_term_is_the_key_phrase_of_its_claim() -> None:
    assert [p.text for p in key_phrases(EXACTLY_ONCE)] == ["exactly-once"]
    assert [p.text for p in key_phrases(ZERO)] == ["zero downtime"]


def test_alternatives_joined_by_or_are_separate_phrases() -> None:
    assert [p.text for p in key_phrases(USERS)] == ["production use", "real users"]
    assert [p.text for p in key_phrases("revenue or waste reduction")] == [
        "revenue",
        "waste reduction",
    ]


def test_leading_determiners_are_dropped() -> None:
    assert [p.text for p in key_phrases(COSTS)] == ["cost savings"]


def test_a_claim_of_only_stop_words_has_no_phrase() -> None:
    assert key_phrases("it is a (nothing)") == []


# ---- matching --------------------------------------------------------------------------------


def test_a_bullet_that_makes_the_claim_is_flagged() -> None:
    assert flagged("Built an exactly-once consumer for carrier events.", [EXACTLY_ONCE]) == [
        "exactly-once"
    ]
    assert flagged("Owned the tracking platform for the company.", [OWNERSHIP]) == [
        "Owned the tracking platform"
    ]
    assert flagged("Delivered cost savings across the fleet.", [COSTS]) == ["cost savings"]


def test_the_claim_is_found_through_inflection_and_hyphenation() -> None:
    assert flagged("Shipped zero-downtime deploys.", [ZERO]) == ["zero-downtime"]
    assert flagged("Exactly once delivery of events.", [EXACTLY_ONCE]) == ["Exactly once"]
    assert flagged("Owning the tracking platform end to end.", [OWNERSHIP])


def test_a_faithful_bullet_that_shares_only_some_words_passes() -> None:
    assert flagged("Built the API of the tracking platform.", [OWNERSHIP]) == []
    assert flagged("Reduced cloud costs of the batch jobs.", [COSTS]) == []
    assert flagged("Measured p99 of 138 ms in a staging load test.", [PRODUCTION]) == []


def test_a_number_in_the_claim_must_be_in_the_bullet_too() -> None:
    assert flagged("Reached a p99 of 138 ms in production.", [PRODUCTION]) == [
        "p99 of 138 ms in production"
    ]
    assert flagged("Reached a p99 of 140 ms in production.", [PRODUCTION]) == []


def test_alternatives_each_match_on_their_own() -> None:
    assert flagged("Served real users from day one.", [USERS]) == ["real users"]
    assert flagged("Ran in production for a year.", [USERS]) == ["production"]


def test_a_negated_mention_is_not_a_claim() -> None:
    assert flagged("Staging load test, not production.", [USERS]) == []
    assert flagged("Without any cost savings claimed.", [COSTS]) == []
    assert flagged("Never claimed exactly-once delivery.", [EXACTLY_ONCE]) == []


def test_a_phrase_the_sources_themselves_contain_is_not_a_violation() -> None:
    sources = ["Cut cost by moving to spot instances; cost savings were 20%."]
    assert flagged("Delivered cost savings of 20%.", [COSTS], sources) == []


def test_only_the_claims_of_the_entry_are_checked() -> None:
    assert flagged("Built an exactly-once consumer.", [COSTS]) == []


def test_the_evidence_names_the_claim() -> None:
    (finding,) = check_do_not_claim("Built an exactly-once consumer.", [EXACTLY_ONCE], SOURCES)
    assert "DO NOT CLAIM" in finding.evidence
    assert "exactly-once" in finding.evidence


def test_the_span_covers_the_matched_words() -> None:
    bullet = "We shipped real users into the product."
    (finding,) = check_do_not_claim(bullet, [USERS], [])
    assert bullet[finding.start : finding.end] == finding.text == "real users"


def test_a_long_claim_needs_most_of_its_words() -> None:
    claim = "accuracy on SKUs outside the 1,850 modelled (one dataset)"
    assert flagged("Improved forecast accuracy across 1,850 SKUs.", [claim], []) == []
    assert flagged("Claimed accuracy on SKUs outside the 1,850 modelled.", [claim], [])


def test_no_claims_means_no_findings() -> None:
    assert check_do_not_claim("Built an exactly-once consumer.", [], SOURCES) == []


def test_a_comma_list_is_a_list_of_alternatives() -> None:
    claim = "a production model, a deployment or real users (it runs on one laptop)"
    assert [p.text for p in key_phrases(claim)] == ["production model", "deployment", "real users"]
    assert flagged("Deployed the agent to staff.", [claim], []) == ["Deployed"]


def test_a_reduction_is_a_cut() -> None:
    assert flagged("Reduced food waste.", ["revenue or waste reduction (never measured)"], [])


def test_a_dollar_amount_is_a_dollar_saving() -> None:
    claim = "any dollar savings (no cost data was collected)"
    assert flagged("Saved the company over $50K a year.", [claim], [])
