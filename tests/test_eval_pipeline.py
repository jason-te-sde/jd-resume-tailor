"""The pipeline evaluation: what a run records, how it is scored, when results may be written."""

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM
from test_cli_plan import scripted

from tailor.bank import load_bank
from tailor.evals.pipeline import (
    PipelineError,
    build_document,
    bullet_kinds,
    check_document,
    differences,
    replay_subset,
    run_once,
    score_runs,
    tree_state,
    write_results,
)
from tailor.llm import LLM, CassetteLLM, LLMConfig, build_llm

GOOD = ("Cut p99 latency from 412 ms to 138 ms in a load test.", ["acme-swe#1"])
SECOND = ("Wrote 23 alerts.", ["acme-swe#2"])
INVENTED = ("Cut p99 latency to 99 ms using Kubernetes.", ["queue-lab#1"])
ORIGINAL = ("Processed 9,200 jobs/s in a 3-node test.", ["queue-lab#1"])


def make_run(**changes: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "jd": "a",
        "seed": 1,
        "status": "ok",
        "error": None,
        "bullets_planned": 0,
        "calls": 5,
        "input_tokens": 1000,
        "output_tokens": 100,
        "wall_seconds": 60.0,
        "load_average_1m": 1.5,
        "bullets": [],
    }
    run.update(changes)
    return run


def make_bullet(
    outcome: str, kinds: tuple[str, ...] = (), structural: bool = False
) -> dict[str, Any]:
    return {
        "entry": "acme-swe",
        "outcome": outcome,
        "first_text": None if outcome == "verbatim" else "first",
        "first_kinds": list(kinds),
        "first_structural": structural,
        "first_problems": [f"{k}: x" for k in kinds],
        "final_text": "final",
        "final_sources": ["acme-swe#1"],
        "original": "orig",
    }


def test_the_kinds_of_a_problem_list_are_read_from_their_prefix() -> None:
    problems = (
        "fabricated_number: '99 ms' - no source",
        "escalation_causal: 'by X' - none",
        "fabricated_number: '5' - none",
    )
    assert bullet_kinds(problems) == (("fabricated_number", "escalation_causal"), False)


def test_problems_that_are_not_verifier_findings_are_structural() -> None:
    assert bullet_kinds(("the bullet cites no fact",)) == ((), True)
    assert bullet_kinds(("it cites 'x#9', which is not a fact of this experience",)) == ((), True)
    assert bullet_kinds(()) == ((), False)


def test_the_score_counts_gate_off_violations_per_class_over_rewritten_bullets() -> None:
    bullets = [
        make_bullet("first_pass"),
        make_bullet("after_retry", ("fabricated_entity",)),
        make_bullet("fallback", ("fabricated_number", "escalation_causal")),
        make_bullet("fallback", structural=True),
        make_bullet("verbatim"),
    ]
    scores = score_runs([make_run(bullets=bullets, bullets_planned=5)])
    assert scores["bullets"] == {"planned": 5, "rewritten": 4, "verbatim": 1}
    off = scores["gate_off"]
    assert (off["flagged"], off["rate"]) == (2, 0.5)
    assert off["by_class"]["fabricated_entity"] == {"bullets": 1, "rate": 0.25, "percent": 25.0}
    assert off["by_class"]["escalation_causal"]["bullets"] == 1
    assert off["by_class"]["do_not_claim"] == {"bullets": 0, "rate": 0.0, "percent": 0.0}
    assert off["structural"] == 1
    on = scores["gate_on"]
    assert on["first_pass"] == {"bullets": 1, "rate": 0.25, "percent": 25.0}
    assert on["after_retry"] == {"bullets": 1, "rate": 0.25, "percent": 25.0}
    assert on["fallback"] == {"bullets": 2, "rate": 0.5, "percent": 50.0}


def test_time_and_tokens_per_resume_give_median_and_range_over_ok_runs() -> None:
    runs = [
        make_run(wall_seconds=10.0, input_tokens=100, output_tokens=10, calls=4),
        make_run(wall_seconds=30.0, input_tokens=300, output_tokens=30, calls=6),
        make_run(wall_seconds=20.0, input_tokens=200, output_tokens=20, calls=5),
        make_run(status="failed", error="boom", wall_seconds=999.0),
    ]
    scores = score_runs(runs)
    assert scores["runs"] == {"total": 4, "ok": 3, "failed": 1}
    seconds = scores["per_resume"]["seconds"]
    assert (seconds["median"], seconds["min"], seconds["max"]) == (20.0, 10.0, 30.0)
    assert scores["per_resume"]["input_tokens"]["median"] == 200
    assert scores["per_resume"]["output_tokens"]["max"] == 30
    assert scores["per_resume"]["calls"]["median"] == 5


def test_a_copied_citation_is_counted_apart_from_verifier_flags_and_structural() -> None:
    bullets = [
        make_bullet("after_retry", ("copied_citation",)),
        make_bullet("fallback", ("fabricated_number",)),
        make_bullet("first_pass"),
    ]
    off = score_runs([make_run(bullets=bullets)])["gate_off"]
    assert (off["flagged"], off["copied_citation"], off["structural"]) == (1, 1, 0)


def test_a_percent_is_computed_from_the_counts_not_from_the_rounded_rate() -> None:
    bullets = [make_bullet("fallback", ("do_not_claim",))] * 8 + [make_bullet("first_pass")] * 34
    off = score_runs([make_run(bullets=bullets)])["gate_off"]
    assert off["rate"] == 0.1905  # 8 of 42, rounded to 4 places
    assert off["percent"] == 100 * 8 / 42  # 19.047...: it reads 19.0%, where 19.05 would read 19.1%


def test_the_totals_are_summed_over_every_run() -> None:
    runs = [
        make_run(calls=4, input_tokens=100, output_tokens=10),
        make_run(calls=6, input_tokens=300, output_tokens=30),
        make_run(status="failed", error="boom", calls=1, input_tokens=50, output_tokens=0),
    ]
    assert score_runs(runs)["totals"] == {"calls": 11, "input_tokens": 450, "output_tokens": 40}


def test_the_load_is_summarised_and_the_time_is_split_at_a_load_of_two() -> None:
    runs = [
        make_run(load_average_1m=0.9, wall_seconds=100.0),
        make_run(load_average_1m=1.7, wall_seconds=300.0),
        make_run(load_average_1m=2.0, wall_seconds=500.0),
        make_run(load_average_1m=19.9, wall_seconds=700.0),
        make_run(load_average_1m=3.0, wall_seconds=900.0),
        make_run(status="failed", error="boom", load_average_1m=50.0, wall_seconds=1.0),
    ]
    scores = score_runs(runs)
    assert scores["per_resume"]["load_average_1m"] == {"median": 2.0, "min": 0.9, "max": 19.9}
    assert scores["by_load"] == {
        "threshold": 2.0,
        "below": {"runs": 2, "seconds": {"median": 200.0, "min": 100.0, "max": 300.0}},
        "at_or_above": {"runs": 3, "seconds": {"median": 700.0, "min": 500.0, "max": 900.0}},
    }


def test_the_score_is_reported_per_job_description() -> None:
    runs = [
        make_run(jd="a", bullets=[make_bullet("first_pass")]),
        make_run(jd="b", bullets=[make_bullet("fallback", ("do_not_claim",))]),
    ]
    scores = score_runs(runs)
    assert set(scores["by_jd"]) == {"a", "b"}
    assert scores["by_jd"]["a"]["gate_off"]["rate"] == 0.0
    assert scores["by_jd"]["b"]["gate_off"]["rate"] == 1.0
    assert scores["by_jd"]["b"]["gate_off"]["by_class"]["do_not_claim"]["bullets"] == 1


def test_a_score_with_no_rewritten_bullets_has_no_rate_instead_of_dividing_by_zero() -> None:
    scores = score_runs([make_run(status="failed", error="x")])
    assert scores["gate_off"]["rate"] is None
    assert scores["gate_on"]["fallback"]["rate"] is None
    assert scores["per_resume"]["seconds"]["median"] is None


# ---- runs, results and the replay ----------------------------------------------------------


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    jds = tmp_path / "jds"
    jds.mkdir()
    (jds / "latency.txt").write_text("Backend Engineer at Northwind. Reduce latency.", "utf-8")
    return {
        "bank_path": bank_path,
        "bank": load_bank(bank_path),
        "jds": jds,
        "cassettes": tmp_path / "cassettes",
        "config": LLMConfig("openai", "test-model", cassette_mode="record"),
    }


def recorder(setup: dict[str, Any], llm: ScriptedLLM):  # type: ignore[no-untyped-def]
    def make(config: LLMConfig) -> LLM:
        return CassetteLLM(
            "record", setup["cassettes"], provider="openai", inner=llm, seed=config.seed
        )

    return make


def live_run(setup: dict[str, Any], seed: int = 1) -> dict[str, Any]:
    llm = scripted({"acme-swe": [GOOD, SECOND], "queue-lab": [INVENTED]})
    # The invented rewrite fails twice (the script answers the retry the same way): a fallback.
    return run_once(
        recorder(setup, llm),
        setup["config"],
        setup["bank"],
        setup["jds"] / "latency.txt",
        seed,
    )


def test_a_run_records_the_bullets_calls_tokens_time_and_load(setup: dict[str, Any]) -> None:
    run = live_run(setup)
    assert (run["jd"], run["seed"], run["status"]) == ("latency", 1, "ok")
    assert run["bullets_planned"] == 3
    assert run["calls"] == 5
    assert run["input_tokens"] == 50
    assert run["output_tokens"] == 25
    assert run["wall_seconds"] >= 0
    assert run["load_average_1m"] >= 0
    by_outcome = [b["outcome"] for b in run["bullets"]]
    assert by_outcome == ["first_pass", "first_pass", "fallback"]
    fallback = run["bullets"][2]
    assert fallback["first_text"] == INVENTED[0]
    assert "fabricated_entity" in fallback["first_kinds"]
    assert fallback["final_text"] == ORIGINAL[0]
    assert fallback["original"] == ORIGINAL[0]
    assert run["bullets"][0]["first_kinds"] == []


def test_a_crashing_run_is_recorded_as_failed_not_dropped(setup: dict[str, Any]) -> None:
    def broken(config: LLMConfig) -> LLM:
        class Down(ScriptedLLM):
            def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("the endpoint is gone")

        return Down([])

    run = run_once(broken, setup["config"], setup["bank"], setup["jds"] / "latency.txt", 2)
    assert run["status"] == "failed"
    assert "the endpoint is gone" in run["error"]
    assert run["bullets"] == []
    assert (run["jd"], run["seed"]) == ("latency", 2)


META = {"commit": "abc1234", "dirty": False, "model": "m", "provider": "openai"}


def test_the_document_carries_the_runs_the_meta_and_scores_computed_from_them(
    setup: dict[str, Any],
) -> None:
    runs = [live_run(setup)]
    document = build_document(runs, META, {"seeds": [1]})
    assert document["runs"] == runs
    assert document["meta"] == META
    assert document["scores"] == score_runs(runs)
    assert check_document(document) == []


def test_a_hand_edited_score_is_caught_by_the_check(setup: dict[str, Any]) -> None:
    document = build_document([live_run(setup)], META, {})
    document["scores"]["gate_off"]["rate"] = 0.0
    problems = check_document(document)
    assert problems
    assert "gate_off" in problems[0]


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "T")
    (path / "a.txt").write_text("a", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "init")
    return path


def test_a_clean_tree_reports_the_commit_and_no_changes(repo: Path) -> None:
    state = tree_state(repo)
    assert state.commit == git(repo, "rev-parse", "HEAD").strip()
    assert not state.dirty
    assert state.changed == ()


def test_an_edited_or_an_untracked_file_makes_the_tree_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    (repo / "new.txt").write_text("n", encoding="utf-8")
    state = tree_state(repo)
    assert state.dirty
    assert set(state.changed) == {"a.txt", "new.txt"}


def test_paths_the_run_itself_writes_do_not_count_as_dirt(repo: Path) -> None:
    (repo / "results.json").write_text("{}", encoding="utf-8")
    (repo / "cassettes").mkdir()
    (repo / "cassettes" / "x.json").write_text("{}", encoding="utf-8")
    state = tree_state(repo, ignore=[repo / "results.json", repo / "cassettes"])
    assert not state.dirty


def test_results_are_not_written_from_a_dirty_tree(repo: Path) -> None:
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    target = repo / "out" / "results.json"
    with pytest.raises(PipelineError, match="uncommitted"):
        write_results(target, {"runs": []}, repo)
    assert not target.exists()


def test_results_are_written_from_a_clean_tree_with_the_commit_recorded(repo: Path) -> None:
    target = repo / "results.json"
    state = write_results(target, {"meta": {}, "runs": []}, repo)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["meta"]["commit"] == state.commit
    assert saved["meta"]["dirty"] is False


def replay_args(setup: dict[str, Any]) -> dict[str, Any]:
    return {
        "bank": setup["bank"],
        "bank_path": setup["bank_path"],
        "jds_dir": setup["jds"],
        "cassette_dir": setup["cassettes"],
        "jd": "latency",
        "seed": 1,
        "config": LLMConfig("openai", "test-model"),
    }


def test_replaying_a_recorded_run_gives_the_committed_numbers(setup: dict[str, Any]) -> None:
    document = build_document([live_run(setup)], META, {})
    assert replay_subset(document, **replay_args(setup)) == []


def test_a_replay_that_differs_from_the_committed_run_is_reported(setup: dict[str, Any]) -> None:
    document = build_document([live_run(setup)], META, {})
    changed = copy.deepcopy(document)
    changed["runs"][0]["bullets"][0]["outcome"] = "after_retry"
    problems = replay_subset(changed, **replay_args(setup))
    assert problems
    assert "outcome" in problems[0] or "score" in problems[0]


def test_a_replay_of_a_run_that_is_not_in_the_results_is_an_error(setup: dict[str, Any]) -> None:
    document = build_document([live_run(setup)], META, {})
    args = replay_args(setup)
    args["seed"] = 9
    with pytest.raises(PipelineError, match="no completed run"):
        replay_subset(document, **args)


def test_replay_builds_its_llm_from_the_cassettes_only(setup: dict[str, Any]) -> None:
    live_run(setup)
    llm = build_llm(
        LLMConfig(
            "openai", "test-model", cassette_mode="replay", cassette_dir=setup["cassettes"], seed=1
        )
    )
    assert isinstance(llm, CassetteLLM)


def test_results_that_hold_an_absolute_path_are_not_written(repo: Path) -> None:
    document = {"meta": {}, "runs": [{"error": "cannot read /Users/someone/work/jd.txt"}]}
    with pytest.raises(PipelineError, match="absolute path"):
        write_results(repo / "results.json", document, repo)
    assert not (repo / "results.json").exists()


def test_a_recorded_error_carries_no_machine_path(setup: dict[str, Any]) -> None:
    def broken(config: LLMConfig) -> LLM:
        raise RuntimeError("cannot open /Users/someone/work/x.json now")

    run = run_once(broken, setup["config"], setup["bank"], setup["jds"] / "latency.txt", 1)
    assert "/Users" not in run["error"]
    assert "<path>" in run["error"]


def test_the_halves_score_the_runs_before_and_after_a_point(setup: dict[str, Any]) -> None:
    from tailor.evals.pipeline import halves_document

    runs = [
        make_run(jd="a", bullets=[make_bullet("first_pass")]),
        make_run(jd="b", bullets=[make_bullet("fallback", ("do_not_claim",))]),
        make_run(jd="c", bullets=[make_bullet("fallback", ("do_not_claim",))]),
    ]
    halves = halves_document({"runs": runs}, 1)
    assert halves["split_after"] == 1
    assert halves["first"] == _overall(runs[:1])
    assert halves["second"] == _overall(runs[1:])
    assert halves["first"]["gate_off"]["rate"] == 0.0
    assert halves["second"]["gate_off"]["rate"] == 1.0
    assert "by_jd" not in halves["first"]


def _overall(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = score_runs(runs)
    scores.pop("by_jd")
    return scores


def test_differences_name_the_item_of_a_list_that_differs() -> None:
    committed = {"rows": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]}
    fresh = {"rows": [{"id": "a", "text": "x"}, {"id": "b", "text": "z"}]}
    assert differences("f", committed, fresh) == ["f.rows[1].text: committed 'y', now 'z'"]


def test_differences_report_a_list_of_another_length_once() -> None:
    assert differences("f", {"rows": [1, 2]}, {"rows": [1]}) == ["f.rows has 2 items, now 1"]
