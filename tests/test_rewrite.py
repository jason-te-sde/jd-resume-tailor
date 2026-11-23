"""Rewriting with citations, and the gate: verify, retry once, then fall back."""

import copy
from typing import Any

from bankdata import WORK, bank
from llmfakes import ScriptedLLM

from tailor.bank import Bank, parse_bank
from tailor.jd.extract import JobDescription
from tailor.rewrite import check_bullets, copied_citations, rewrite_entry

JD = JobDescription("Backend Engineer", "Northwind", ("Reduce API latency",), (), (), ("Go",))
GOOD = "Cut p99 latency from 412 ms to 138 ms."


def answer(*bullets: tuple[str, list[str]]) -> dict[str, Any]:
    return {"bullets": [{"text": t, "sources": s} for t, s in bullets]}


def one_entry(**changes: Any) -> Bank:
    entry = copy.deepcopy(WORK)
    entry["bullets"] = [GOOD]
    entry.update(changes)
    entry["bullet_capabilities"] = ["skill"] * len(entry["bullets"])
    return parse_bank(bank(entry))


def run(llm: ScriptedLLM, source: Bank, chosen: tuple[int, ...] = (0,)) -> Any:
    entry = source.entries[0]
    return rewrite_entry(llm, source, JD, entry, chosen, model="m")


def test_a_rewrite_the_verifier_accepts_is_kept_after_one_call() -> None:
    text = "Reduced p99 latency from 412 ms to 138 ms for the tracking API."
    llm = ScriptedLLM([answer((text, ["acme-swe#1"]))])
    result = run(llm, one_entry())
    (bullet,) = result.bullets
    assert (bullet.text, bullet.outcome, bullet.verdict) == (text, "first_pass", "ok")
    assert bullet.sources == ("acme-swe#1",)
    assert bullet.original == GOOD
    assert (result.calls, len(llm.calls)) == (1, 1)


def test_a_number_fabricated_twice_ends_with_the_bank_bullet_and_the_reason() -> None:
    bad = "Cut p99 latency from 412 ms to 99 ms."
    llm = ScriptedLLM([answer((bad, ["acme-swe#1"])), answer((bad, ["acme-swe#1"]))])
    result = run(llm, one_entry())
    (bullet,) = result.bullets
    assert (bullet.text, bullet.outcome, bullet.verdict) == (GOOD, "fallback", "bank_original")
    assert bullet.sources == ("acme-swe#1",)
    assert "fabricated_number" in bullet.reason
    assert "99" in bullet.reason
    assert [a.text for a in bullet.attempts] == [bad, bad]
    assert result.calls == 2


def test_a_bullet_fixed_on_the_retry_keeps_the_rewrite_and_the_model_saw_the_findings() -> None:
    bad = "Cut p99 latency from 412 ms to 99 ms."
    fixed = "Cut p99 latency from 412 ms to 138 ms in a load test."
    llm = ScriptedLLM([answer((bad, ["acme-swe#1"])), answer((fixed, ["acme-swe#1"]))])
    result = run(llm, one_entry())
    (bullet,) = result.bullets
    assert (bullet.text, bullet.outcome, bullet.verdict) == (fixed, "after_retry", "ok")
    assert bullet.reason == ""
    retry_messages = llm.calls[1][0]
    assert retry_messages[-1]["role"] == "user"
    assert "fabricated_number" in retry_messages[-1]["content"]
    assert bad in retry_messages[-1]["content"]
    assert retry_messages[-2]["role"] == "assistant"


def test_only_the_failing_bullets_are_retried() -> None:
    source = one_entry(bullets=[GOOD, "Wrote 23 alerts."])
    bad = "Wrote 40 alerts."
    llm = ScriptedLLM(
        [
            answer((GOOD, ["acme-swe#1"]), (bad, ["acme-swe#2"])),
            answer(("Wrote 23 alerts for on-call.", ["acme-swe#2"])),
        ]
    )
    result = run(llm, source, (0, 1))
    assert [b.outcome for b in result.bullets] == ["first_pass", "after_retry"]
    assert "Rewrite only these 1 bullets" in llm.calls[1][0][-1]["content"]


def test_a_bullet_that_cites_nothing_fails_even_if_its_text_is_true() -> None:
    llm = ScriptedLLM([answer((GOOD, [])), answer((GOOD, []))])
    (bullet,) = run(llm, one_entry()).bullets
    assert bullet.outcome == "fallback"
    assert "cites no fact" in bullet.reason


def test_a_bullet_that_cites_an_unknown_fact_fails() -> None:
    llm = ScriptedLLM([answer((GOOD, ["acme-swe#9"])), answer((GOOD, ["acme-swe#1"]))])
    (bullet,) = run(llm, one_entry()).bullets
    assert bullet.outcome == "after_retry"
    assert "acme-swe#9" in bullet.attempts[0].problems[0]


def test_a_fact_of_another_entry_is_not_citable() -> None:
    other = copy.deepcopy(WORK)
    other.update(id="other", bullets=["Wrote 23 alerts."], bullet_capabilities=["alerts"])
    source = parse_bank(bank(WORK, other))
    llm = ScriptedLLM([answer((GOOD, ["other#1"])), answer((GOOD, ["other#1"]))])
    (bullet,) = rewrite_entry(llm, source, JD, source.entries[0], (0,), model="m").bullets
    assert bullet.outcome == "fallback"
    assert "not a fact of this experience" in bullet.reason


def test_a_missing_answer_counts_as_failing() -> None:
    source = one_entry(bullets=[GOOD, "Wrote 23 alerts."])
    llm = ScriptedLLM([answer((GOOD, ["acme-swe#1"])), {"bullets": []}])
    result = run(llm, source, (0, 1))
    assert [b.outcome for b in result.bullets] == ["first_pass", "fallback"]
    assert result.bullets[1].text == "Wrote 23 alerts."


def test_an_entry_that_is_not_adaptable_is_copied_verbatim_without_a_call() -> None:
    llm = ScriptedLLM([])
    result = run(llm, one_entry(adaptable=False))
    (bullet,) = result.bullets
    assert (bullet.text, bullet.outcome, bullet.verdict) == (GOOD, "verbatim", "bank_original")
    assert result.calls == 0
    assert llm.calls == []


def test_the_prompt_states_the_rules_and_lists_facts_and_limits() -> None:
    llm = ScriptedLLM([answer((GOOD, ["acme-swe#1"]))])
    run(llm, one_entry())
    system, user = (m["content"] for m in llm.calls[0][0])
    for rule in ("No new numbers", "No new technologies", "No absolute words", "No causes",
                 "Round only the unflattering way", "posting's words"):  # fmt: skip
        assert rule in system
    assert f"[acme-swe#1] {GOOD}" in user
    assert "- production p99" in user
    assert "1. " + GOOD in user


def test_bank_bullets_with_latex_escapes_are_plain_text_in_the_result() -> None:
    source = one_entry(bullets=["Raised completion from 61\\% to 74\\%."], adaptable=False)
    (bullet,) = run(ScriptedLLM([]), source).bullets
    assert bullet.text == "Raised completion from 61% to 74%."


def test_explicit_facts_are_the_sources_of_a_fallback_bullet() -> None:
    source = one_entry(
        facts=[{"id": "f-latency", "text": "p99 went from 412 ms to 138 ms"}],
    )
    bad = "Cut p99 latency from 412 ms to 99 ms."
    llm = ScriptedLLM([answer((bad, ["f-latency"])), answer((bad, ["f-latency"]))])
    (bullet,) = run(llm, source).bullets
    assert (bullet.outcome, bullet.sources) == ("fallback", ("f-latency",))


def test_the_retry_prompt_does_not_depend_on_the_key_order_of_the_first_answer() -> None:
    bad = "Cut p99 latency from 412 ms to 99 ms."
    keys_a = {"bullets": [{"text": bad, "sources": ["acme-swe#1"]}]}
    keys_b = {"bullets": [{"sources": ["acme-swe#1"], "text": bad}]}
    prompts = []
    for first in (keys_a, keys_b):
        llm = ScriptedLLM([first, answer((GOOD, ["acme-swe#1"]))])
        run(llm, one_entry())
        prompts.append(llm.calls[1][0])
    assert prompts[0] == prompts[1]


def test_an_empty_answer_is_rejected_even_when_it_cites_a_fact() -> None:
    from tailor.rewrite import check_bullets

    source = one_entry()
    problems = check_bullets(
        source, source.entries[0], [("", ("acme-swe#1",))], rounding="conservative"
    )
    assert problems == [("the bullet is empty or missing",)]


def test_a_blank_rewrite_never_reaches_the_resume() -> None:
    llm = ScriptedLLM([answer(("   ", ["acme-swe#1"])), answer(("", ["acme-swe#1"]))])
    (bullet,) = run(llm, one_entry()).bullets
    assert (bullet.text, bullet.outcome) == (GOOD, "fallback")
    assert "empty" in bullet.reason


# ---- citations copied into the text ----------------------------------------------------------

COUNTED = "Wrote 2 alerts."


def counted_entry() -> Bank:
    """An entry whose fact ids pass the verifier inside a bullet: "alerts#2" has the fact's count
    and the fact's word, so only the citation check can refuse it."""
    return one_entry(id="alerts", bullets=[GOOD, COUNTED])


def test_the_forms_of_a_citation_copied_into_the_text_are_found() -> None:
    ids = ("acme-swe#1", "acme-swe#2", "f-latency")
    for text, cited in [
        ("Merged 2 tables (sources: [f-latency]).", ("f-latency",)),
        ("Merged 2 tables (Sources: [acme-swe#2], [f-latency])", ("acme-swe#2", "f-latency")),
        ("Merged 2 tables (fact: acme-swe#2).", ("acme-swe#2",)),
        ("Merged 2 tables. [acme-swe#2]", ("acme-swe#2",)),
        ("Merged 2 tables ([acme-swe#2]).", ("acme-swe#2",)),
        ("Merged 2 tables (acme-swe#2).", ("acme-swe#2",)),
        ("Merged 2 tables. (Cites acme-swe#2)", ("acme-swe#2",)),
        ("Merged 2 tables (f-latency).", ("f-latency",)),
        ("Merged 2 tables, acme-swe#2.", ("acme-swe#2",)),
    ]:
        assert copied_citations(text, ids) == cited, text


def test_prose_that_looks_like_a_citation_but_names_no_fact_is_not_one() -> None:
    ids = ("acme-swe#1", "acme-swe#2", "f-latency")
    for text in [
        "Merged data from 3 sources: Kafka, PostgreSQL and S3.",
        "Cut p99 latency [in staging] from 412 ms to 138 ms.",
        "Cut f-latency-budget misses by half.",  # an id inside a longer word
        "Wrote the f-latency runbook.",  # an explicit id in prose, with no bracket or label
        "Merged 2 tables (acme-swe#12).",  # not a fact of the entry
    ]:
        assert copied_citations(text, ids) == (), text


def test_a_copied_sources_list_is_refused_and_retried() -> None:
    source = one_entry(facts=[{"id": "f-latency", "text": "p99 went from 412 ms to 138 ms"}])
    cited = "Cut p99 latency from 412 ms to 138 ms (sources: [f-latency])"
    fixed = "Cut p99 latency from 412 ms to 138 ms."
    llm = ScriptedLLM([answer((cited, ["f-latency"])), answer((fixed, ["f-latency"]))])
    (bullet,) = run(llm, source).bullets
    assert (bullet.text, bullet.outcome) == (fixed, "after_retry")
    (problem,) = bullet.attempts[0].problems
    assert "f-latency" in problem
    assert "f-latency" in llm.calls[1][0][-1]["content"]


def test_a_copied_fact_label_is_refused_and_falls_back_when_repeated() -> None:
    cited = "Wrote 2 alerts (fact: alerts#2)."
    llm = ScriptedLLM([answer((cited, ["alerts#2"])), answer((cited, ["alerts#2"]))])
    (bullet,) = run(llm, counted_entry(), (1,)).bullets
    assert (bullet.text, bullet.outcome) == (COUNTED, "fallback")
    assert "alerts#2" in bullet.reason


def test_a_copied_bracketed_fact_id_is_refused() -> None:
    cited = "Wrote 2 alerts. [alerts#2]"
    llm = ScriptedLLM([answer((cited, ["alerts#2"])), answer((COUNTED, ["alerts#2"]))])
    (bullet,) = run(llm, counted_entry(), (1,)).bullets
    assert (bullet.text, bullet.outcome) == (COUNTED, "after_retry")
    assert "alerts#2" in bullet.attempts[0].problems[0]


def test_a_copied_citation_is_its_own_kind_neither_structural_nor_a_finding() -> None:
    from tailor.evals.pipeline import bullet_kinds

    source = counted_entry()
    (problems,) = check_bullets(
        source, source.entries[0], [("Wrote 2 alerts. [alerts#2]", ("alerts#2",))],
        rounding="conservative",
    )  # fmt: skip
    assert problems[0].startswith("copied_citation: ")
    assert bullet_kinds(problems) == (("copied_citation",), False)


def test_a_rewrite_with_a_finding_is_refused_with_the_findings_alone() -> None:
    # The recorded evaluation runs were made before this check: a rewrite the verifier rejects
    # must give the same problems, and so the same retry prompt, as it did then.
    source = counted_entry()
    (problems,) = check_bullets(
        source, source.entries[0], [("Wrote 9 alerts. [alerts#2]", ("alerts#2",))],
        rounding="conservative",
    )  # fmt: skip
    assert problems
    assert all(problem.startswith("fabricated_number: ") for problem in problems)
