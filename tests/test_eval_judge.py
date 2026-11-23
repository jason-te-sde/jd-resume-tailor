"""Measuring the judge: its verdicts on labelled rows next to the verifier's on the same rows."""

import json
from pathlib import Path
from typing import Any

from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM

from tailor.bank import load_bank
from tailor.evals.judge import judge_rows, row_input, score_judged
from tailor.evals.verifier import Row
from tailor.llm import LLMError, Message


def load(tmp_path: Path) -> Any:
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    return load_bank(path)


FACT = "Cut p99 latency from 412 ms to 138 ms."
ROWS = [
    Row("r1", "acme-swe", (FACT,), "Cut p99 latency from 412 ms to 138 ms.", ("ok",)),
    Row("r2", "acme-swe", (FACT,), "Cut p99 latency to 99 ms.", ("fabricated_number",)),
    Row("r3", "acme-swe", (FACT,), "Reduced p99 latency from 412 ms to 138 ms.", ("ok",)),
]


def verdicts(flags: dict[str, list[str]]) -> ScriptedLLM:
    def handler(messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        bullet = messages[-1]["content"].rsplit("Bullet: ", 1)[1]
        answer: dict[str, Any] = {name: False for name in schema["properties"]}
        for name in flags.get(bullet, []):
            answer[name] = True
        answer["reason"] = "r"
        return answer

    return ScriptedLLM(handler)


def test_a_row_is_shown_to_the_judge_as_its_bullet_its_facts_and_the_entry_context(
    tmp_path: Path,
) -> None:
    shown = row_input(load(tmp_path), ROWS[0])
    assert shown.bullet == ROWS[0].bullet
    assert shown.facts == (FACT,)
    assert shown.claims == ("production p99", "cost savings")
    assert shown.tech == "Go, PostgreSQL"
    assert shown.scope == ("EVIDENCE: load test.",)


def test_the_judge_never_sees_the_row_labels(tmp_path: Path) -> None:
    llm = verdicts({})
    judge_rows(llm, load(tmp_path), ROWS, model="m")
    seen = " ".join(m["content"] for messages, _ in llm.calls for m in messages)
    assert "fabricated_number" not in seen.split("Bullet:")[-1]
    assert "labels" not in seen


def test_judge_rows_records_the_verdict_of_each_row(tmp_path: Path) -> None:
    llm = verdicts({ROWS[1].bullet: ["fabricated_number"]})
    records = judge_rows(llm, load(tmp_path), ROWS, model="m")
    assert [r["id"] for r in records] == ["r1", "r2", "r3"]
    assert records[1]["judge"] == ["fabricated_number"]
    assert records[0]["judge"] == []
    assert records[1]["labels"] == ["fabricated_number"]
    assert records[1]["reason"] == "r"


def test_a_judge_error_is_recorded_on_its_row_and_the_run_goes_on(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise LLMError("bad json")
        return {**{name: False for name in schema["properties"]}, "reason": ""}

    records = judge_rows(ScriptedLLM(handler), load(tmp_path), ROWS, model="m")
    assert records[1]["judge"] is None
    assert "bad json" in records[1]["error"]
    assert records[2]["judge"] == []


def test_the_judge_and_the_verifier_are_scored_on_the_same_judged_rows(tmp_path: Path) -> None:
    parsed = load(tmp_path)
    llm = verdicts({ROWS[1].bullet: ["fabricated_number"], ROWS[2].bullet: ["do_not_claim"]})
    records = judge_rows(llm, parsed, ROWS, model="m")
    scored = score_judged(parsed, ROWS, records)
    assert scored["rows"] == 3
    assert scored["errors"] == 0
    judge = scored["judge"]
    assert judge["classes"]["fabricated_number"]["precision"] == 1.0
    assert judge["classes"]["fabricated_number"]["recall"] == 1.0
    assert judge["classes"]["do_not_claim"]["precision"] == 0.0
    assert judge["false_alarm_rate"] == 0.5
    verifier = scored["verifier"]
    assert verifier["classes"]["fabricated_number"]["recall"] == 1.0
    assert verifier["false_alarm_rate"] == 0.0


def test_rows_the_judge_failed_on_are_left_out_of_both_scores(tmp_path: Path) -> None:
    parsed = load(tmp_path)
    records = judge_rows(verdicts({}), parsed, ROWS, model="m")
    records[1] = {**records[1], "judge": None, "error": "x"}
    scored = score_judged(parsed, ROWS, records)
    assert (scored["rows"], scored["errors"]) == (2, 1)
    assert scored["verifier"]["rows"] == 2
    assert scored["judge"]["rows"] == 2
