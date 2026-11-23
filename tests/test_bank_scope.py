import pytest
from hypothesis import given
from hypothesis import strategies as st

from tailor.bank.scope import (
    do_not_claim_claims,
    same_project_refs,
    split_claims,
    split_parts,
    strip_do_not_claim,
)

# Parts: labels and where they end.


def test_a_scope_without_labels_is_one_unlabelled_part() -> None:
    parts = split_parts("Built as a coursework project. Ran on one laptop.")
    assert [(p.label, p.text) for p in parts] == [
        (None, "Built as a coursework project. Ran on one laptop.")
    ]


def test_each_label_starts_a_part_that_runs_to_the_next_label() -> None:
    scope = "Intro text. ORIGIN: a hackathon. EVIDENCE: perf/run-3.csv. DO NOT CLAIM: users."
    assert [(p.label, p.text) for p in split_parts(scope)] == [
        (None, "Intro text."),
        ("ORIGIN", "a hackathon."),
        ("EVIDENCE", "perf/run-3.csv."),
        ("DO NOT CLAIM", "users."),
    ]


def test_multi_word_and_dated_labels() -> None:
    scope = "HIS WORK: the API. NOT HIS: the UI. REVIEWED 2026-03-02: rewrote two bullets."
    assert [p.label for p in split_parts(scope)] == ["HIS WORK", "NOT HIS", "REVIEWED 2026-03-02"]


@pytest.mark.parametrize(
    "scope",
    [
        "The FFT: a transform of the signal, not a label.",
        "Seen in (README: the linked file) only.",
        "NOTE:no space after the colon",
        "an ACRONYM-free sentence with an ID: 42",
    ],
)
def test_text_that_only_looks_like_a_label_is_not_one(scope: str) -> None:
    assert [p.label for p in split_parts(scope)] == [None]


def test_a_label_needs_four_letters() -> None:
    # "FFT" is three letters and a common acronym; "DATA" is four and is taken as a label.
    assert [p.label for p in split_parts("x FFT: y")] == [None]
    assert [p.label for p in split_parts("x DATA: y")] == [None, "DATA"]


def test_labels_are_case_sensitive() -> None:
    assert [p.label for p in split_parts("Do not claim: anything")] == [None]


# Claims: where a DO NOT CLAIM part splits.


def test_claims_split_at_semicolons() -> None:
    assert split_claims("real users; 'exactly-once' delivery; production use") == [
        "real users",
        "'exactly-once' delivery",
        "production use",
    ]


def test_claims_split_at_sentence_ends() -> None:
    assert split_claims("Real users. Production use! A SLA?") == [
        "Real users",
        "Production use",
        "A SLA",
    ]


def test_a_parenthetical_stays_with_its_claim_even_with_a_semicolon_inside() -> None:
    text = "'transactions/s' (3.5K is HTTP requests; writes are 150/s); sharding"
    assert split_claims(text) == [
        "'transactions/s' (3.5K is HTTP requests; writes are 150/s)",
        "sharding",
    ]


def test_a_sentence_end_inside_brackets_does_not_split() -> None:
    assert split_claims("autoscaling (not built. Never planned) [see notes. Later]; HA") == [
        "autoscaling (not built. Never planned) [see notes. Later]",
        "HA",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p99 of 3.5K req/s at v1.2.3", ["p99 of 3.5K req/s at v1.2.3"]),
        ("Kafka, e.g. MSK or similar; Redis", ["Kafka, e.g. MSK or similar", "Redis"]),
        ("a vs. B comparison", ["a vs. B comparison"]),
        ("about approx. 5 nodes. Second claim", ["about approx. 5 nodes", "Second claim"]),
        ("see https://example.com/a.b. Next claim", ["see https://example.com/a.b", "Next claim"]),
    ],
)
def test_periods_that_are_not_sentence_ends(text: str, expected: list[str]) -> None:
    assert split_claims(text) == expected


def test_a_sentence_may_start_with_a_quote() -> None:
    assert split_claims("No users. 'Scale' is not claimed.") == [
        "No users",
        "'Scale' is not claimed",
    ]


def test_unbalanced_brackets_fall_back_to_splitting_everywhere() -> None:
    # An unclosed "(" would otherwise swallow every later claim into one.
    assert split_claims("first (never closed; second; third") == [
        "first (never closed",
        "second",
        "third",
    ]
    assert split_claims("first) closed twice; second") == ["first) closed twice", "second"]


def test_empty_pieces_and_trailing_punctuation_are_dropped() -> None:
    assert split_claims("; ;  a ;; b. ; ") == ["a", "b"]
    assert split_claims("") == []
    assert split_claims("   ") == []


@given(st.text())
def test_claims_are_never_empty_and_never_padded(text: str) -> None:
    for claim in split_claims(text):
        assert claim, f"empty claim from {text!r}"
        assert claim == claim.strip(), f"padded claim {claim!r} from {text!r}"


# The DO NOT CLAIM parts of a scope.


def test_claims_of_a_scope_with_one_part() -> None:
    scope = "EVIDENCE: run 3. DO NOT CLAIM: real users; 'exactly-once'."
    assert do_not_claim_claims(scope) == ["real users", "'exactly-once'"]


def test_claims_of_a_scope_with_several_parts_are_joined_in_order() -> None:
    scope = "DO NOT CLAIM: a; b. EVIDENCE: x. DO NOT CLAIM: c. ORIGIN: y."
    assert do_not_claim_claims(scope) == ["a", "b", "c"]


def test_a_part_ends_at_the_next_label_not_at_a_sentence() -> None:
    scope = "DO NOT CLAIM: a. Also b. EVIDENCE: not a claim."
    assert do_not_claim_claims(scope) == ["a", "Also b"]


def test_a_scope_without_the_label_has_no_claims() -> None:
    assert do_not_claim_claims("Nothing to disclaim here.") == []
    assert do_not_claim_claims("") == []


def test_an_empty_part_has_no_claims() -> None:
    assert do_not_claim_claims("DO NOT CLAIM: EVIDENCE: x") == []


def test_strip_removes_only_the_do_not_claim_parts() -> None:
    scope = "Intro. EVIDENCE: run 3. DO NOT CLAIM: users; scale. ORIGIN: a hackathon."
    assert strip_do_not_claim(scope) == "Intro. EVIDENCE: run 3. ORIGIN: a hackathon."


def test_strip_of_a_scope_that_is_all_claims_is_empty() -> None:
    assert strip_do_not_claim("DO NOT CLAIM: users") == ""


def test_strip_leaves_a_scope_without_the_label_as_it_was() -> None:
    assert strip_do_not_claim("EVIDENCE: run 3.") == "EVIDENCE: run 3."


def test_a_label_may_wrap_across_a_line_break() -> None:
    # Found by a property run: "DO NOT\nCLAIM:" once parsed as a label named CLAIM, and joining the
    # kept parts with spaces then spelled a real DO NOT CLAIM label out of the leftovers.
    assert [p.label for p in split_parts("DO NOT\nCLAIM: x")] == ["DO NOT CLAIM"]
    assert strip_do_not_claim("DO NOT\nCLAIM: x") == ""


@given(st.text())
def test_stripped_scope_has_no_do_not_claim_part_left(text: str) -> None:
    assert [p for p in split_parts(strip_do_not_claim(text)) if p.label == "DO NOT CLAIM"] == []


# Same-project references.


def test_same_project_reference() -> None:
    assert same_project_refs("Same project as beacon-queue: the benchmark side.") == [
        "beacon-queue"
    ]


def test_several_references_in_order_without_repeats() -> None:
    scope = "Same project as a-1 (x). Later: Same project as b_2. Same project as a-1."
    assert same_project_refs(scope) == ["a-1", "b_2"]


@pytest.mark.parametrize(
    "scope",
    ["", "the same project as another", "Same project: beacon-queue", "Same project as"],
)
def test_text_that_is_not_the_phrase_has_no_references(scope: str) -> None:
    assert same_project_refs(scope) == []


def test_punctuation_after_the_id_is_not_part_of_it() -> None:
    assert same_project_refs("Same project as alpha.") == ["alpha"]
    assert same_project_refs("Same project as alpha, and more") == ["alpha"]
    assert same_project_refs("Same project as alpha-") == ["alpha"]
