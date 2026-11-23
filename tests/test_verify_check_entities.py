"""The entity check: technologies and proper nouns must come from the facts or the tech stack."""

from tailor.verify.checks import check_entities

SOURCE = (
    "Built a rate limiter in Rust that keeps token buckets in Redis and updates them with one "
    "Lua script; checked it with a load test on a Chrome profile."
)
CONTEXT = ["Rust, Redis, Lua, Tokio, Docker, GitHub Actions", "Gridlock: a distributed limiter"]


def names(
    bullet: str, sources: list[str] | None = None, context: list[str] | None = None
) -> list[str]:
    found = check_entities(
        bullet,
        [SOURCE] if sources is None else sources,
        context=CONTEXT if context is None else context,
    )
    assert all(f.kind == "fabricated_entity" for f in found)
    return [f.text for f in found]


def test_technologies_named_in_the_sources_pass() -> None:
    assert names("Wrote a limiter in Rust with a Lua script on Redis.") == []


def test_a_technology_the_sources_never_mention_is_fabricated() -> None:
    assert names("Wrote a limiter in Rust, backed by Kubernetes.") == ["Kubernetes"]


def test_the_entry_tech_stack_counts_as_a_source() -> None:
    assert names("Shipped it in Docker with GitHub Actions.") == []
    assert names("Shipped it in Docker with GitHub Actions.", context=[]) == [
        "Docker",
        "GitHub Actions",
    ]


def test_an_alias_is_the_same_technology() -> None:
    sources = ["Stored rows in PostgreSQL and ran jobs on k8s."]
    assert names("Stored rows in Postgres and ran jobs on Kubernetes.", sources, []) == []


def test_matching_ignores_case_for_plain_names() -> None:
    assert names("Cached reads in redis.") == []
    assert names("Deployed on kubernetes.") == ["kubernetes"]


def test_a_longer_name_needs_its_own_evidence() -> None:
    assert names("Added Redis Cluster support.") == ["Redis Cluster"]
    assert names("Added Redis Streams.", ["Used Redis Streams for jobs."], []) == []


def test_a_shorter_name_is_supported_by_a_longer_one_in_the_sources() -> None:
    assert names("Cached reads in Redis.", ["Consumed Redis Streams."], []) == []


def test_a_case_sensitive_name_is_not_supported_by_the_plain_word() -> None:
    assert names("Wrote the service in Go.", ["Let the requests go to the replica."], []) == ["Go"]


def test_camel_case_and_all_caps_identifiers_are_entities() -> None:
    assert names("Streamed to ClickPipe, HIPAA audited.", [SOURCE]) == [
        "ClickPipe",
        "HIPAA",
    ]
    assert names("Streamed to ClickPipe.", ["Streamed to ClickPipe."], []) == []


def test_generic_acronyms_and_units_are_not_entities() -> None:
    bullet = "Built an HTTP API with CRUD endpoints on a 4-core CPU and a 310 MB image in CI."
    assert names(bullet, ["x"], []) == []


def test_a_capitalised_multi_word_name_must_appear_in_the_sources() -> None:
    assert names("Piloted it with Acme Logistics in two regions.") == ["Acme Logistics"]
    sources = ["A pilot with Acme Logistics ran in two regions."]
    assert names("Piloted it with Acme Logistics in two regions.", sources, []) == []


def test_a_single_capitalised_word_mid_sentence_is_a_proper_noun() -> None:
    assert names("Integrated the limiter with Contoso for billing.") == ["Contoso"]


def test_the_first_word_of_a_sentence_is_not_a_proper_noun() -> None:
    assert names("Built it. Deployed it. Measured it.") == []


def test_the_entry_title_and_company_are_known_names() -> None:
    assert names(
        "Shipped Gridlock to Lantern Freight Systems.", [SOURCE], ["Lantern Freight Systems"]
    ) == ["Gridlock"]


def test_months_and_the_pronoun_i_are_not_proper_nouns() -> None:
    assert names("Shipped in March and again in October.", ["x"], []) == []


def test_each_finding_points_at_its_span() -> None:
    bullet = "Backed by Kubernetes today."
    (finding,) = check_entities(bullet, [SOURCE], context=CONTEXT)
    assert bullet[finding.start : finding.end] == "Kubernetes"
    assert "Kubernetes" in finding.evidence
    assert "tech stack" in finding.evidence


def test_a_bullet_with_no_entities_has_nothing_to_check() -> None:
    assert names("cut the latency by batching reads") == []
