"""The LLM judge: its prompt, how its answer is parsed, and the facts it is given."""

import json
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM

from tailor.bank import parse_bank
from tailor.judge import CLASSES, SCHEMA, JudgeInput, build_messages, judge_bullet, parse_verdict
from tailor.llm import LLMError
from tailor.verify import KINDS


def answer(**flags: bool) -> dict[str, Any]:
    document: dict[str, Any] = {name: False for name in CLASSES}
    document.update(flags)
    document["reason"] = "because"
    return document


INPUT = JudgeInput(
    bullet="Cut p99 latency to 90 ms using Kubernetes.",
    facts=("Cut p99 latency from 412 ms to 138 ms.",),
    scope=("EVIDENCE: measured on a staging cluster.",),
    claims=("production traffic",),
    tech="Go, PostgreSQL",
)


def test_the_judge_answers_for_exactly_the_verifier_classes() -> None:
    assert CLASSES == KINDS
    assert set(SCHEMA["properties"]) == {*KINDS, "reason"}
    assert SCHEMA["required"] == [*KINDS, "reason"]
    assert SCHEMA["additionalProperties"] is False


def test_the_prompt_holds_the_bullet_the_facts_the_scope_the_limits_and_the_stack() -> None:
    user = build_messages(INPUT)[-1]["content"]
    for text in (
        INPUT.bullet,
        INPUT.facts[0],
        INPUT.scope[0],
        INPUT.claims[0],
        INPUT.tech,
    ):
        assert text in user


def test_the_system_prompt_defines_every_class() -> None:
    system = build_messages(INPUT)[0]["content"]
    for name in KINDS:
        assert name in system


def test_the_prompt_never_carries_a_verifier_finding() -> None:
    system, user = (m["content"] for m in build_messages(INPUT))
    assert "finding" not in (system + user).lower()


def test_a_verdict_lists_the_classes_the_judge_flagged() -> None:
    verdict = parse_verdict(answer(fabricated_entity=True, number_drift=True))
    assert verdict.classes == ("number_drift", "fabricated_entity")
    assert verdict.flagged
    assert verdict.reason == "because"


def test_a_verdict_with_no_flags_is_ok() -> None:
    verdict = parse_verdict(answer())
    assert verdict.classes == ()
    assert not verdict.flagged


@pytest.mark.parametrize(
    "raw",
    [
        {"reason": "x"},
        {**answer(), "do_not_claim": "yes"},
        {**answer(), "reason": 3},
        {**{k: v for k, v in answer().items() if k != "number_drift"}},
    ],
)
def test_an_answer_that_is_not_a_verdict_is_an_llm_error(raw: dict[str, Any]) -> None:
    with pytest.raises(LLMError):
        parse_verdict(raw)


def test_judge_bullet_makes_one_call_with_the_schema() -> None:
    llm = ScriptedLLM([answer(fabricated_entity=True)])
    verdict = judge_bullet(llm, INPUT, model="m")
    assert verdict.classes == ("fabricated_entity",)
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == SCHEMA


def test_the_input_of_a_bullet_comes_from_the_facts_it_cites() -> None:
    parsed = parse_bank(json.loads(json.dumps(bank(WORK, PROJECT))))
    from tailor.judge import input_for

    entry = parsed.entries[0]
    ids = tuple(f.id for f in parsed.facts_for(entry.id) if f.kind == "bullet")[:1]
    made = input_for(parsed, "text", entry.id, ids)
    assert made.bullet == "text"
    assert made.facts == (parsed.fact(ids[0]).text,)
    assert made.claims == tuple(parsed.do_not_claim(entry.id))
    assert made.scope == tuple(f.text for f in parsed.facts_for(entry.id) if f.kind == "scope")
