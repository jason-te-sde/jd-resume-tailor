"""The escalation checks: absolute words and causal links that the sources do not make."""

from tailor.verify.checks import (
    check_escalation_absolute,
    check_escalation_causal,
    check_unsupported_claims,
)

# ---- absolutes -------------------------------------------------------------------------------

RUNS = "Killed Redis 50 times during a load test; no loss of checks across the runs."
SIM = "Ran 10,000 seeds with 0 safety violations across 5 invariants."
README = "Published the raw CSVs so that every number in the README can be regenerated."


def absolutes(bullet: str, sources: list[str]) -> list[str]:
    found = check_escalation_absolute(bullet, sources)
    assert all(f.kind == "escalation_absolute" for f in found)
    return [f.text for f in found]


def test_an_absolute_word_the_sources_lack_is_flagged() -> None:
    assert absolutes("Guaranteed delivery of every carrier event.", [RUNS]) == [
        "Guaranteed",
        "every",
    ]
    assert absolutes("Never dropped a check.", [RUNS]) == ["Never"]
    assert absolutes("Always returned to Redis-backed limits.", [RUNS]) == ["Always"]


def test_the_word_family_is_matched_not_just_the_exact_word() -> None:
    assert absolutes("The limiter guarantees atomic updates.", [RUNS]) == ["guarantees"]
    assert absolutes("Completely eliminated stale reads.", [RUNS]) == ["Completely", "eliminated"]
    assert absolutes("A flawless and bulletproof rollout.", [RUNS]) == ["flawless", "bulletproof"]
    assert absolutes("Fully automated, with perfect recall.", [RUNS]) == ["Fully", "perfect"]


def test_exactly_once_and_one_hundred_percent_are_absolutes() -> None:
    assert absolutes("Built an exactly-once consumer.", [RUNS]) == ["exactly-once"]
    assert absolutes("Processed events Exactly Once.", [RUNS]) == ["Exactly Once"]
    assert absolutes("Reached 100% uptime.", [RUNS]) == ["100%"]
    assert absolutes("Reached 100 percent uptime.", [RUNS]) == ["100 percent"]


def test_an_absolute_the_sources_use_themselves_is_not_flagged() -> None:
    assert (
        absolutes("The limiter guarantees atomic updates.", ["It guarantees atomic updates."]) == []
    )
    assert absolutes("Reached 100% uptime.", ["Uptime was 100%."]) == []


def test_zero_is_fine_when_the_sources_say_none_of_the_same_thing() -> None:
    assert absolutes("Zero safety violations across 10,000 seeds.", [SIM]) == []
    assert absolutes("Zero loss of checks across 50 kills.", [RUNS]) == []
    assert absolutes("Zero data loss of checks.", [RUNS]) == []


def test_zero_is_flagged_when_nothing_in_the_sources_says_none_of_it() -> None:
    assert absolutes("Zero downtime deploys.", [SIM]) == ["Zero"]
    assert absolutes("Zero-downtime deploys.", [SIM]) == ["Zero"]


def test_technical_terms_that_start_with_zero_are_not_absolutes() -> None:
    assert absolutes("Added a zero-copy path and zero-shot prompts.", [SIM]) == []


def test_every_and_all_are_fine_when_the_sources_quantify_the_same_thing() -> None:
    assert absolutes("Made every number in the README reproducible.", [README]) == []
    assert absolutes("Made every deploy reproducible.", [README]) == ["every"]
    assert absolutes("Handled all requests.", [README]) == ["all"]


def test_all_before_a_count_and_every_before_a_period_are_not_absolutes() -> None:
    assert absolutes("Ran Lighthouse on all 5 main views.", [SIM]) == []
    assert absolutes("Ran the export every 5 minutes and every night.", [SIM]) == []
    assert absolutes("Checked every other build.", [SIM]) == []


def test_a_word_inside_another_word_is_not_an_absolute() -> None:
    assert absolutes("Overall latency fell; the callback is tall.", [SIM]) == []
    assert absolutes("Alwaysfree and neverland are names.", [SIM]) == []


def test_absolutes_point_at_their_span_and_say_why() -> None:
    bullet = "We always cut latency."
    (finding,) = check_escalation_absolute(bullet, [SIM])
    assert bullet[finding.start : finding.end] == "always"
    assert "always" in finding.evidence
    assert "sources" in finding.evidence


def test_a_bullet_with_no_absolutes_is_clean() -> None:
    assert check_escalation_absolute("Cut latency in a staging test.", [SIM]) == []


# ---- causal links ----------------------------------------------------------------------------

LATENCY = (
    "Cut p99 latency of the shipment-tracking API from 412 ms to 138 ms by replacing a "
    "per-request fan-out of 7 queries with one batched query and adding a Redis read-through "
    "cache."
)
BENCH = (
    "Reached 18,000 writes/s in a single-threaded benchmark; "
    "most of the gain came from a larger memtable."
)
REPLICAS = (
    "Added 2 read replicas behind a router. "
    "Reporting p95 fell from 2.4 s to 0.9 s in a staging run."
)
LUA = (
    "Updated token buckets with one Lua script, so a check and its update are a single atomic step."
)


def causal(bullet: str, sources: list[str]) -> list[str]:
    found = check_escalation_causal(bullet, sources)
    assert all(f.kind == "escalation_causal" for f in found)
    return [f.text for f in found]


def test_a_mechanism_the_sources_link_to_the_result_passes() -> None:
    bullet = "Cut p99 latency from 412 ms to 138 ms by batching queries and adding a Redis cache."
    assert causal(bullet, [LATENCY]) == []


def test_a_mechanism_the_sources_never_mention_is_flagged() -> None:
    bullet = "Doubled write throughput by group-committing writers."
    assert causal(bullet, [BENCH]) == ["by group-committing writers"]


def test_a_mechanism_and_a_result_that_the_sources_keep_apart_are_flagged() -> None:
    bullet = "Cut reporting p95 from 2.4 s to 0.9 s by adding read replicas."
    assert causal(bullet, [REPLICAS]) == ["by adding read replicas"]


def test_an_effect_clause_the_sources_do_not_support_is_flagged() -> None:
    bullet = "Added 2 read replicas, spreading read load across them."
    assert causal(bullet, [REPLICAS]) == ["spreading read load across them"]


def test_so_that_passes_when_the_source_makes_the_same_link() -> None:
    bullet = "Updated token buckets with a Lua script so that the check and update are atomic."
    assert causal(bullet, [LUA]) == []


def test_so_that_is_flagged_when_it_adds_a_purpose() -> None:
    bullet = "Updated token buckets with a Lua script so that clients never see stale limits."
    assert causal(bullet, [LUA]) == ["so that clients never see stale limits"]


def test_thanks_to_because_of_and_due_to_are_causal_links() -> None:
    assert causal("Faster reads thanks to caching.", [BENCH]) == ["thanks to caching"]
    assert causal("Faster reads because of caching.", [BENCH]) == ["because of caching"]
    assert causal("Faster reads due to caching.", [BENCH]) == ["due to caching"]


def test_a_which_clause_that_claims_an_effect_is_flagged() -> None:
    bullet = "Added a larger memtable, which let writers batch more."
    assert causal(bullet, [BENCH]) == ["which let writers batch more"]


def test_with_naming_a_mechanism_is_checked() -> None:
    assert causal("Improved throughput with connection pooling.", [BENCH]) == [
        "with connection pooling"
    ]
    assert causal("Improved throughput with a larger memtable.", [BENCH]) == []


def test_by_before_a_number_or_after_led_is_not_a_mechanism() -> None:
    assert causal("Cut deploy time by 27 minutes.", [BENCH]) == []
    assert causal("Removed 5 hours of staleness, up by 3x.", [BENCH]) == []
    assert causal("A project led by the platform team.", [BENCH]) == []
    assert causal("Followed by a review.", [BENCH]) == []


def test_with_before_a_number_is_not_a_mechanism() -> None:
    assert causal("Wrote tests with 81% coverage.", [BENCH]) == []
    assert causal("Ran 10,000 seeds with 0 violations.", [BENCH]) == []


def test_the_whole_clause_of_the_source_may_be_reworded() -> None:
    bullet = (
        "Reduced p99 latency from 412 ms to 138 ms by replacing 7 queries with one batched query."
    )
    assert causal(bullet, [LATENCY]) == []


def test_nested_links_are_checked_once() -> None:
    bullet = "Cut latency by moving the cache with a Redis cluster."
    assert len(causal(bullet, [BENCH])) == 1


def test_causal_findings_point_at_their_span_and_say_why() -> None:
    bullet = "Doubled write throughput by group-committing writers."
    (finding,) = check_escalation_causal(bullet, [BENCH])
    assert bullet[finding.start : finding.end] == "by group-committing writers"
    assert "sources" in finding.evidence


def test_a_bullet_without_a_causal_link_is_clean() -> None:
    assert check_escalation_causal("Reached 18,000 writes/s.", [BENCH]) == []


def test_a_decimal_point_or_a_thousands_comma_does_not_end_a_clause() -> None:
    source = "Handled 41,000 checks/s on one 4-vCPU VM, with a p99 of 2.1 ms at 20,000 checks/s."
    assert (
        causal("Handled 41,000 checks/s with a p99 of 2.1 ms at 20,000 checks/s.", [source]) == []
    )
    bullet = "Reached 1,200 ms by group-committing 3.5 K writers."
    assert causal(bullet, [source]) == ["by group-committing 3.5 K writers"]


def test_tested_by_and_verified_by_name_a_method_not_a_cause() -> None:
    source = "Killed Redis 50 times during a load test."
    assert causal("Limits keep working, tested by killing Redis 50 times.", [source]) == []
    assert causal("Verified it by killing Redis 50 times.", [source]) == []


def test_through_names_a_mechanism() -> None:
    assert causal("Reached 18,000 writes/s through group commit.", [BENCH]) == [
        "through group commit"
    ]


# ---- claims about where and by whom ----------------------------------------------------------

LOAD = "Cut p99 latency from 412 ms to 138 ms in a staging load test, not production traffic."
TEAM = "Led a 4-person effort to replace the nightly cron job with an event-driven job."


def claims(
    bullet: str, sources: list[str], limits: list[str] | None = None
) -> list[tuple[str, str]]:
    return [(f.kind, f.text) for f in check_unsupported_claims(bullet, sources, limits or [])]


def test_production_is_flagged_when_no_source_says_it() -> None:
    assert claims("Served 9,200 events/s in production.", [TEAM]) == [
        ("escalation_absolute", "production")
    ]


def test_a_source_that_says_not_production_does_not_state_it() -> None:
    assert claims("Cut p99 latency to 138 ms in production.", [LOAD]) == [
        ("escalation_absolute", "production")
    ]


def test_a_source_that_says_production_states_it() -> None:
    source = "Ran the consumer in production for 3 months."
    assert claims("Ran the consumer in production.", [source]) == []


def test_pre_production_and_non_production_are_not_production() -> None:
    assert claims("Tuned a pre-production cluster.", [TEAM]) == []


def test_production_is_a_do_not_claim_when_the_entry_rules_out_production_or_deployment() -> None:
    for limit in ("a production p99 (staging only)", "deployment or use by a real team"):
        assert claims("Ran in production.", [TEAM], [limit]) == [("do_not_claim", "production")]


def test_leading_a_team_is_flagged_when_no_source_leads_anything() -> None:
    source = "Built 12 dbt models and an Airflow DAG."
    assert claims("Led the team that built 12 dbt models.", [source]) == [
        ("escalation_absolute", "Led the team")
    ]
    managed = claims("Managed a team of 3 engineers.", [source])
    assert [kind for kind, _ in managed] == ["escalation_absolute"]


def test_a_source_that_leads_something_backs_a_leadership_claim() -> None:
    assert claims("Led the team that replaced the cron job.", [TEAM]) == []
