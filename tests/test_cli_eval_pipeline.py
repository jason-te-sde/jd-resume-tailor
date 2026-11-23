"""The command lines of the pipeline evaluation, judge and residual check."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from bankdata import PROJECT, WORK, bank
from llmfakes import ScriptedLLM
from test_cli_plan import scripted

from tailor.cli import main
from tailor.evals import commands
from tailor.evals.verifier import build_split
from tailor.llm import LLM, CassetteLLM, LLMConfig, Message

GOOD = ("Cut p99 latency from 412 ms to 138 ms in a load test.", ["acme-swe#1"])
SECOND = ("Wrote 23 alerts.", ["acme-swe#2"])
INVENTED = ("Cut p99 latency to 99 ms using Kubernetes.", ["queue-lab#1"])


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "repo"
    (path / "eval" / "pipeline" / "jds").mkdir(parents=True)
    (path / "bank.json").write_text(json.dumps(bank(WORK, PROJECT)), encoding="utf-8")
    jds = path / "eval" / "pipeline" / "jds"
    (jds / "latency.txt").write_text("Backend Engineer at Northwind. Reduce latency.", "utf-8")
    (jds / "queues.txt").write_text("Backend Engineer at Northwind. Run job queues.", "utf-8")
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "T")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(path)
    monkeypatch.setenv("TAILOR_MODEL", "test-model")
    monkeypatch.setenv("TAILOR_PROVIDER", "openai")
    monkeypatch.delenv("TAILOR_CASSETTES", raising=False)
    monkeypatch.delenv("TAILOR_SEED", raising=False)
    monkeypatch.setattr(commands, "ollama_version", lambda base_url: "0.0-test")
    monkeypatch.setattr(commands, "ollama_model_digest", lambda base_url, model: "d" * 64)
    return path


def patch_factory(monkeypatch: pytest.MonkeyPatch, llm_for: Any) -> None:
    """Make every run use a fresh scripted model, recorded as a real one would be."""

    def build(config: LLMConfig, **kwargs: Any) -> LLM:
        if config.cassette_mode == "replay":
            from tailor.llm import build_llm as real

            replayer: LLM = real(config, **kwargs)
            return replayer
        inner: LLM = llm_for()
        if config.cassette_mode == "record":
            return CassetteLLM(
                "record", config.cassette_dir, provider="openai", inner=inner, seed=config.seed
            )
        return inner

    monkeypatch.setattr(commands, "build_llm", build)


def good_model() -> ScriptedLLM:
    return scripted({"acme-swe": [GOOD, SECOND], "queue-lab": [INVENTED]})


def live(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "eval", "pipeline", "--bank", "bank.json", "--seeds", "1,2",
            "--cassettes", str(tmp_path / "cassettes"), *extra,
        ]
    )  # fmt: skip


def test_a_live_run_writes_every_run_the_meta_and_the_scores(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0, capsys.readouterr().err
    out = repo / "eval" / "pipeline" / "results.json"
    document = json.loads(out.read_text(encoding="utf-8"))
    assert [(r["jd"], r["seed"]) for r in document["runs"]] == [
        ("latency", 1), ("latency", 2), ("queues", 1), ("queues", 2),
    ]  # fmt: skip
    meta = document["meta"]
    assert meta["model"] == "test-model"
    assert meta["provider"] == "openai"
    assert meta["dirty"] is False
    assert len(meta["commit"]) == 40
    assert meta["ollama_version"] == "0.0-test"
    assert meta["model_digest"] == "d" * 64
    assert meta["seed_applied"] is True
    assert meta["machine"]["machine"]
    assert document["protocol"]["jds"] == 2
    assert document["protocol"]["seeds"] == [1, 2]
    assert document["scores"]["runs"] == {"total": 4, "ok": 4, "failed": 0}
    assert len(list((tmp_path / "cassettes").glob("*.json"))) > 0
    assert not (repo / "eval" / "pipeline" / "results.partial.json").exists()
    assert "gate-off" in capsys.readouterr().out


def test_a_live_run_from_a_dirty_tree_refuses_before_any_call(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "notes.txt").write_text("unsaved", encoding="utf-8")
    calls: list[int] = []

    def counted() -> ScriptedLLM:
        calls.append(1)
        return good_model()

    patch_factory(monkeypatch, counted)
    assert live(tmp_path) == 2
    assert "uncommitted" in capsys.readouterr().err
    assert calls == []
    assert not (repo / "eval" / "pipeline" / "results.json").exists()


def test_a_crashing_run_is_in_the_results_as_failed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Down(ScriptedLLM):
        def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("gone")

    patch_factory(monkeypatch, lambda: Down([]))
    assert live(tmp_path) == 0
    document = json.loads((repo / "eval" / "pipeline" / "results.json").read_text("utf-8"))
    assert document["scores"]["runs"] == {"total": 4, "ok": 0, "failed": 4}
    assert document["runs"][0]["error"].startswith("RuntimeError")


def test_resume_skips_the_runs_the_partial_file_already_holds(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    results = repo / "eval" / "pipeline" / "results.json"
    document = json.loads(results.read_text(encoding="utf-8"))
    partial = {"meta": document["meta"], "runs": document["runs"][:3]}
    results.unlink()
    (repo / "eval" / "pipeline" / "results.partial.json").write_text(json.dumps(partial), "utf-8")
    made: list[int] = []

    def counted() -> ScriptedLLM:
        made.append(1)
        return good_model()

    patch_factory(monkeypatch, counted)
    assert live(tmp_path, "--resume") == 0
    assert len(made) == 1  # only (queues, 2) was left
    again = json.loads(results.read_text(encoding="utf-8"))
    assert len(again["runs"]) == 4


def test_check_passes_on_the_committed_scores_and_fails_on_an_edited_one(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    results = repo / "eval" / "pipeline" / "results.json"
    assert main(["eval", "pipeline", "--check"]) == 0
    document = json.loads(results.read_text(encoding="utf-8"))
    document["scores"]["gate_off"]["rate"] = 0.0
    results.write_text(json.dumps(document), encoding="utf-8")
    assert main(["eval", "pipeline", "--check"]) == 1
    assert "gate_off" in capsys.readouterr().err


def test_replay_reproduces_a_recorded_run_with_no_model(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    monkeypatch.undo()
    monkeypatch.chdir(repo)
    cassettes = str(tmp_path / "cassettes")
    code = main(
        ["eval", "pipeline", "--replay", "--bank", "bank.json", "--cassettes", cassettes,
         "--jd", "latency", "--replay-seed", "2"]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().err
    assert "replay matches: latency with seed 2" in capsys.readouterr().out


def test_seed_is_not_taken_as_an_abbreviation_of_seeds(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    capsys.readouterr()
    code = main(
        ["eval", "pipeline", "--replay", "--bank", "bank.json",
         "--cassettes", str(tmp_path / "cassettes"), "--jd", "latency", "--seed", "2"]
    )  # fmt: skip
    assert code == 2
    assert "--seed" in capsys.readouterr().err


def test_no_option_of_any_command_can_be_abbreviated(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (
        ["eval", "pipeline", "--chec"],
        ["eval", "verifier", "--tol", "0.1"],
        ["eval", "sample", "--see", "1"],
        ["plan", "--jd", "x", "--bank", "b.json", "--out", "o", "--roun", "nearest"],
    ):
        assert main(argv) == 2, argv
        assert "unrecognized arguments" in capsys.readouterr().err, argv


def test_replay_fails_when_the_committed_numbers_are_not_what_the_cassettes_give(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    monkeypatch.undo()
    monkeypatch.chdir(repo)
    results = repo / "eval" / "pipeline" / "results.json"
    document = json.loads(results.read_text(encoding="utf-8"))
    document["runs"][0]["bullets"][0]["first_kinds"] = ["do_not_claim"]
    results.write_text(json.dumps(document), encoding="utf-8")
    code = main(
        ["eval", "pipeline", "--replay", "--bank", "bank.json",
         "--cassettes", str(tmp_path / "cassettes"), "--jd", "latency", "--replay-seed", "1"]
    )  # fmt: skip
    assert code == 1
    assert "first_kinds" in capsys.readouterr().err


# ---- judge, sample, residual ---------------------------------------------------------------


def judge_model(flag: str = "") -> ScriptedLLM:
    def handler(messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        bullet = messages[-1]["content"].rsplit("Bullet: ", 1)[1]
        answer: dict[str, Any] = {name: False for name in schema["properties"]}
        answer["reason"] = "r"
        if flag and flag in bullet:
            answer["fabricated_entity"] = True
        return answer

    return ScriptedLLM(handler)


def test_the_judge_command_reports_each_bullet(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, lambda: judge_model("99 ms"))
    bullets = repo / "bullets.json"
    bullets.write_text(
        json.dumps(
            [
                {"entry": "acme-swe", "sources": ["acme-swe#1"], "text": GOOD[0]},
                {"entry": "acme-swe", "sources": ["acme-swe#1"], "text": "Cut p99 to 99 ms."},
            ]
        ),
        encoding="utf-8",
    )
    code = main(["judge", "--bank", "bank.json", "--bullets", str(bullets)])
    out = capsys.readouterr().out
    assert code == 1
    assert "fabricated_entity" in out
    assert "ok" in out


def test_the_judge_command_warns_that_a_seed_is_not_sent_to_a_provider_without_one(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, judge_model)
    bullets = repo / "bullets.json"
    bullets.write_text(json.dumps([{"entry": "acme-swe", "text": GOOD[0]}]), encoding="utf-8")
    argv = ["judge", "--bank", "bank.json", "--bullets", str(bullets), "--seed", "4"]
    assert main([*argv, "--provider", "anthropic"]) == 0
    assert "takes no seed, so seed 4" in capsys.readouterr().err


def test_the_judge_command_exits_zero_when_nothing_is_flagged(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_factory(monkeypatch, judge_model)
    bullets = repo / "bullets.json"
    bullets.write_text(
        json.dumps([{"entry": "acme-swe", "text": GOOD[0]}]),
        encoding="utf-8",
    )
    assert main(["judge", "--bank", "bank.json", "--bullets", str(bullets)]) == 0


def judge_data(repo: Path) -> None:
    """Commit a three-row verifier dataset, all held out, and the same rows as the blind set."""
    fact = "Cut p99 latency from 412 ms to 138 ms."
    rows = [
        {"id": f"r{i}", "entry": "acme-swe", "sources": [fact], "bullet": text, "labels": labels}
        for i, (text, labels) in enumerate(
            [
                ("Cut p99 latency from 412 ms to 138 ms.", ["ok"]),
                ("Cut p99 latency to 99 ms.", ["fabricated_number"]),
                ("Reduced p99 latency from 412 ms to 138 ms.", ["ok"]),
            ]
        )
    ]
    data = repo / "eval" / "verifier"
    (data / "blind").mkdir(parents=True)
    lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    (data / "labeled.jsonl").write_text(lines, encoding="utf-8")
    (data / "blind" / "rows.jsonl").write_text(lines, encoding="utf-8")
    from tailor.evals.verifier import load_rows

    split = build_split(load_rows(data / "labeled.jsonl"))
    split["holdout"], split["dev"] = sorted(split["holdout"] + split["dev"]), []
    (data / "split.json").write_text(json.dumps(split), encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "data")


def test_eval_judge_writes_the_judge_next_to_the_verifier_on_holdout_and_blind(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    judge_data(repo)
    patch_factory(monkeypatch, lambda: judge_model("99 ms"))
    out = repo / "eval" / "pipeline" / "judge-accuracy.json"
    code = main(["eval", "judge", "--bank", "bank.json", "--out", str(out)])
    assert code == 0, capsys.readouterr().err
    document = json.loads(out.read_text(encoding="utf-8"))
    assert set(document["datasets"]) == {"holdout", "blind"}
    holdout = document["datasets"]["holdout"]
    assert holdout["rows"] == 3
    assert holdout["judge"]["classes"]["fabricated_number"]["support"] == 1
    assert holdout["verifier"]["classes"]["fabricated_number"]["recall"] == 1.0
    assert len(holdout["records"]) == 3
    assert document["meta"]["model"] == "test-model"


def make_results(repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path, "--seeds", "1") == 0
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "results")
    return repo / "eval" / "pipeline" / "results.json"


def test_judge_runs_sample_and_residual_work_end_to_end(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = make_results(repo, monkeypatch, tmp_path)
    patch_factory(monkeypatch, lambda: judge_model("Wrote 23 alerts"))
    judged = repo / "eval" / "pipeline" / "judge-runs.json"
    assert main(["eval", "judge-runs", "--bank", "bank.json", "--results", str(results),
                 "--out", str(judged)]) == 0  # fmt: skip
    document = json.loads(judged.read_text(encoding="utf-8"))
    assert document["bullets"]
    assert document["meta"]["model"] == "test-model"
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "judged")

    sample = repo / "eval" / "pipeline" / "blind-sample.jsonl"
    mapping = repo / "eval" / "pipeline" / "blind-sample-map.json"
    assert main(["eval", "sample", "--bank", "bank.json", "--results", str(results),
                 "--judged", str(judged), "--seed", "5", "--size", "2",
                 "--out", str(sample), "--map", str(mapping)]) == 0  # fmt: skip
    rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
    mapped = json.loads(mapping.read_text(encoding="utf-8"))
    assert {r["id"] for r in rows} == set(mapped["rows"])
    assert mapped["seed"] == 5

    labels = repo / "eval" / "pipeline" / "blind-labels.jsonl"
    labels.write_text(
        "\n".join(json.dumps({"id": r["id"], "labels": ["ok"]}) for r in rows) + "\n",
        encoding="utf-8",
    )
    calls = repo / "eval" / "pipeline" / "blind-judgement-calls.md"
    calls.write_text("# Judgement calls\n", encoding="utf-8")
    residual = repo / "eval" / "pipeline" / "residual.json"
    code = main(["eval", "residual", "--map", str(mapping), "--labels", str(labels),
                 "--calls", str(calls), "--out", str(residual)])  # fmt: skip
    assert code == 0, capsys.readouterr().err
    scored = json.loads(residual.read_text(encoding="utf-8"))
    assert scored["residual"]["estimate"] == 0
    assert "residual" in capsys.readouterr().out


def test_first_sample_writes_the_first_rewrites_for_labelling_and_their_mapping(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = make_results(repo, monkeypatch, tmp_path)
    sample = repo / "eval" / "pipeline" / "first-sample.jsonl"
    mapping = repo / "eval" / "pipeline" / "first-sample-map.json"
    code = main(["eval", "first-sample", "--bank", "bank.json", "--results", str(results),
                 "--seed", "5", "--size", "2"])  # fmt: skip
    assert code == 0, capsys.readouterr().err
    rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
    mapped = json.loads(mapping.read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["f-001", "f-002"]
    assert set(mapped["rows"]) == {"f-001", "f-002"}
    assert (mapped["seed"], mapped["sampled"]) == (5, 2)
    assert "2 rows" in capsys.readouterr().out


def test_residual_rejects_labels_that_do_not_match_the_sample(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps(
            {"seed": 1, "size": 1, "frame": 1, "final_bullets": 1,
             "strata": {"flagged": 0, "unjudged": 0, "unflagged": 1, "sampled": 1},
             "rows": {"s-001": {"jd": "a", "seed": 1, "index": 0, "entry": "e",
                                "stratum": "random"}}}
        ),
        encoding="utf-8",
    )  # fmt: skip
    labels = tmp_path / "labels.jsonl"
    labels.write_text("", encoding="utf-8")
    calls = tmp_path / "calls.md"
    calls.write_text("# Judgement calls\n", encoding="utf-8")
    code = main(["eval", "residual", "--map", str(mapping), "--labels", str(labels),
                 "--calls", str(calls), "--out", str(tmp_path / "r.json")])  # fmt: skip
    assert code == 2
    assert "no label" in capsys.readouterr().err


def test_halves_are_written_and_the_check_holds_them(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_factory(monkeypatch, good_model)
    assert live(tmp_path) == 0
    assert main(["eval", "pipeline", "--halves", "2"]) == 0
    halves = repo / "eval" / "pipeline" / "halves.json"
    document = json.loads(halves.read_text(encoding="utf-8"))
    assert document["split_after"] == 2
    assert document["first"]["runs"]["total"] == 2
    assert main(["eval", "pipeline", "--check"]) == 0
    document["second"]["gate_off"]["rate"] = 0.0
    halves.write_text(json.dumps(document), encoding="utf-8")
    assert main(["eval", "pipeline", "--check"]) == 1
    assert "halves" in capsys.readouterr().err


def test_the_judge_runs_seed_zero_only_for_a_provider_that_takes_one_and_warns_only_if_asked(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = make_results(repo, monkeypatch, tmp_path)
    patch_factory(monkeypatch, judge_model)
    out = tmp_path / "judged.json"
    argv = ["eval", "judge-runs", "--bank", "bank.json", "--results", str(results)]
    assert main([*argv, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["judge_seed"] == 0
    capsys.readouterr()
    assert main([*argv, "--out", str(out), "--provider", "anthropic"]) == 0
    assert "takes no seed" not in capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["judge_seed"] is None
    assert main([*argv, "--out", str(out), "--provider", "anthropic", "--seed", "3"]) == 0
    assert "takes no seed, so seed 3" in capsys.readouterr().err
