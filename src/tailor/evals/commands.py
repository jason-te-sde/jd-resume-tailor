"""The command lines of the pipeline evaluation, the judge and the residual check.

``tailor eval pipeline`` runs the plan live over the job descriptions and seeds (or replays one
recorded run, or re-scores the committed results); ``tailor eval judge`` measures the judge on the
verifier's labelled rows; ``tailor eval judge-runs``, ``sample`` and ``residual`` are the residual
check; ``tailor judge`` judges bullets you give it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tailor.bank import Bank, BankError, load_bank
from tailor.evals.citations import count_citations
from tailor.evals.first import draw_first_sample, identical_rows, score_first
from tailor.evals.judge import judge_rows, score_judged
from tailor.evals.pipeline import (
    PipelineError,
    build_document,
    check_document,
    differences,
    halves_document,
    machine_description,
    ollama_model_digest,
    ollama_version,
    replay_subset,
    run_all,
    score_runs,
    tree_state,
    write_results,
)
from tailor.evals.residual import (
    ResidualError,
    draw_sample,
    judge_runs,
    judged_positions,
    judged_summary,
    judgement_call_ids,
    score_residual,
)
from tailor.evals.verifier import EvalError, Row, load_rows, load_split, validate_rows
from tailor.jd import JDError
from tailor.judge import input_for, judge_bullet
from tailor.llm import LLMConfig, LLMError, build_llm, load_config
from tailor.llm.base import SEED_PROVIDERS
from tailor.llm.config import seed_warning
from tailor.verify.verifier import VerifyError, load_bullets

EXIT_OK: Final = 0
EXIT_FINDINGS: Final = 1
EXIT_USAGE: Final = 2

DEFAULT_BANK: Final = Path("examples/alex-rivera/bank.json")
DEFAULT_DRAW: Final = 60
_DRAW_CHECK: Final = "draw again with the seed and size in the committed map, and compare"
PIPELINE_DIR: Final = Path("eval/pipeline")
_ERRORS: Final = (
    BankError,
    EvalError,
    JDError,
    LLMError,
    PipelineError,
    ResidualError,
    VerifyError,
    OSError,
    json.JSONDecodeError,
)


# ---- parsers ---------------------------------------------------------------------------------


def _check_or_rescore(command: argparse.ArgumentParser, check: str, rescore: str) -> None:
    """``--check`` compares a scored file with what its inputs give; ``--rescore`` writes it."""
    group = command.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help=check)
    group.add_argument("--rescore", action="store_true", help=rescore)


def _model_options(command: argparse.ArgumentParser, *, seed: int | bool | None) -> None:
    """The bank and model options.

    ``seed`` is the seed a provider that takes one gets when none is given (``None``: no seed),
    or ``False`` for no ``--seed`` option at all.
    """
    command.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="the bank JSON file")
    command.add_argument("--model", help="the model id (overrides TAILOR_MODEL and tailor.toml)")
    command.add_argument("--provider", choices=["anthropic", "openai"], help="the provider")
    command.add_argument("--config", type=Path, help="a tailor.toml (default: ./tailor.toml)")
    if seed is not False:
        default = "" if seed is None else f" (default for one that does: {seed})"
        command.add_argument(
            "--seed",
            type=int,
            help=f"the seed sent to a provider that takes one (TAILOR_SEED){default}",
        )
        command.set_defaults(default_seed=seed)


def add_judge_parser(commands: Any) -> None:
    """Register ``tailor judge``."""
    judge = commands.add_parser(
        "judge",
        help="ask a model whether bullets are supported by the facts they cite",
        description=(
            "For each bullet, ask the model (through the same provider layer as `tailor plan`) "
            "whether it commits any violation class against the facts it cites, with a reason. "
            "The judge is a second reader to be measured (`tailor eval judge`), not an authority."
        ),
        epilog=(
            "Same bullets file as `tailor verify`. Exit status: 0 when no bullet is flagged, 1 "
            "when one is, 2 when the command could not run."
        ),
    )
    _model_options(judge, seed=None)
    judge.add_argument("--bullets", type=Path, required=True, help="the bullets to check, as JSON")
    judge.add_argument("--format", choices=["json", "md"], default="md")


def add_eval_parsers(eval_commands: Any) -> None:
    """Register the evaluation subcommands other than ``verifier``."""
    pipeline = eval_commands.add_parser(
        "pipeline",
        help="run the plan live over the job descriptions and seeds, and score it",
        description=(
            "Run `tailor plan` over the bank, every job description in --jds and every seed, "
            "record each model exchange into --cassettes, and write one results file. --replay "
            "rebuilds one committed run from cassettes with no model and checks it against the "
            "committed numbers; --check only re-scores the committed runs."
        ),
        epilog=(
            "Results are written only from a clean working tree. Exit status: 0 on success, 1 "
            "when a replay or check differs from the committed results, 2 when it could not run."
        ),
    )
    _model_options(pipeline, seed=False)
    pipeline.add_argument("--jds", type=Path, default=PIPELINE_DIR / "jds")
    pipeline.add_argument("--seeds", default="1,2,3", help="comma-separated (default: 1,2,3)")
    pipeline.add_argument("--cassettes", type=Path, help="where model exchanges are recorded")
    pipeline.add_argument("--out", type=Path, default=PIPELINE_DIR / "results.json")
    pipeline.add_argument("--resume", action="store_true", help="continue from the partial file")
    pipeline.add_argument("--replay", action="store_true", help="replay one committed run")
    _check_or_rescore(
        pipeline,
        "recompute the scores (and halves.json) from the committed runs and compare",
        "write the scores of results.json (and halves.json) from its runs, nothing else",
    )
    pipeline.add_argument(
        "--halves",
        type=int,
        metavar="N",
        help="write halves.json next to --out: the scores of the first N runs and of the rest",
    )
    pipeline.add_argument(
        "--citations",
        action="store_true",
        help="write citations.json next to --out: the fact ids copied into bullets, by shape",
    )
    pipeline.add_argument("--jd", help="with --replay: the job description (default: first)")
    pipeline.add_argument(
        "--replay-seed", type=int, help="with --replay: the seed (default: first)"
    )

    accuracy = eval_commands.add_parser(
        "judge",
        help="the judge's precision and recall next to the verifier's",
        description=(
            "Run the judge on the verifier's holdout split and on the blind set, and score it "
            "per class next to the verifier on the same rows."
        ),
    )
    _model_options(accuracy, seed=0)
    accuracy.add_argument("--data", type=Path, default=Path("eval/verifier"))
    accuracy.add_argument("--out", type=Path, default=PIPELINE_DIR / "judge-accuracy.json")
    _check_or_rescore(
        accuracy,
        "rescore the committed verdicts against the rows and compare, with no model call",
        "write the scores from the committed verdicts, with no model call",
    )

    runs = eval_commands.add_parser(
        "judge-runs",
        help="judge every final bullet of the pipeline runs",
        description="Run the judge on every final bullet of every completed run in --results.",
    )
    _model_options(runs, seed=0)
    runs.add_argument("--results", type=Path, default=PIPELINE_DIR / "results.json")
    runs.add_argument("--out", type=Path, default=PIPELINE_DIR / "judge-runs.json")
    _check_or_rescore(
        runs,
        "recompute the summary, and the bullets judged, from the committed files",
        "write the summary from the committed verdicts, with no model call",
    )

    sample = eval_commands.add_parser(
        "sample",
        help="draw the blind sample for labelling",
        description=(
            "Draw every judge-flagged bullet and a seeded random sample of the others, shuffle "
            "them under opaque ids, and write what a labeller needs (no outcome, verdict or "
            "finding) and, separately, the id to run mapping."
        ),
    )
    sample.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    sample.add_argument("--results", type=Path, default=PIPELINE_DIR / "results.json")
    sample.add_argument("--judged", type=Path, default=PIPELINE_DIR / "judge-runs.json")
    sample.add_argument("--seed", type=int, help="the seed of the draw (required to draw)")
    sample.add_argument("--size", type=int, help="random draw size (default: 60)")
    sample.add_argument("--out", type=Path, default=PIPELINE_DIR / "blind-sample.jsonl")
    sample.add_argument("--map", type=Path, default=PIPELINE_DIR / "blind-sample-map.json")
    sample.add_argument("--check", action="store_true", help=_DRAW_CHECK)

    first = eval_commands.add_parser(
        "first-sample",
        help="draw the blind sample of first rewrites for labelling",
        description=(
            "Draw a seeded random sample of the first rewrites that reached the verifier, and "
            "write what a labeller needs (no outcome or finding) and, separately, the id to run "
            "mapping."
        ),
    )
    first.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    first.add_argument("--results", type=Path, default=PIPELINE_DIR / "results.json")
    first.add_argument("--seed", type=int, help="the seed of the draw (required to draw)")
    first.add_argument("--size", type=int, help="draw size (default: 60)")
    first.add_argument("--out", type=Path, default=PIPELINE_DIR / "first-sample.jsonl")
    first.add_argument("--map", type=Path, default=PIPELINE_DIR / "first-sample-map.json")
    first.add_argument("--check", action="store_true", help=_DRAW_CHECK)

    residual = eval_commands.add_parser(
        "residual",
        help="the residual violation rate from the blind labels",
        description=(
            "Score the labels of the blind sample: the flagged bullets confirmed by label plus "
            "the random stratum's rate extrapolated over the frame, with an interval."
        ),
    )
    residual.add_argument("--map", type=Path, default=PIPELINE_DIR / "blind-sample-map.json")
    residual.add_argument("--labels", type=Path, default=PIPELINE_DIR / "blind-labels.jsonl")
    residual.add_argument(
        "--calls",
        type=Path,
        default=PIPELINE_DIR / "blind-judgement-calls.md",
        help="the labeller's judgement calls; their number is recorded",
    )
    residual.add_argument("--out", type=Path, default=PIPELINE_DIR / "residual.json")
    residual.add_argument(
        "--check", action="store_true", help="rescore the committed labels and compare"
    )

    scored = eval_commands.add_parser(
        "first-rewrites",
        help="the labelled violation rate of first rewrites, and the verifier against the labels",
        description=(
            "Score the labels of the first-rewrite sample: the share labelled violating, with an "
            "interval, and the verifier's findings on those rewrites (the gate-off flag, and per "
            "class) against the labels."
        ),
    )
    scored.add_argument("--results", type=Path, default=PIPELINE_DIR / "results.json")
    scored.add_argument("--map", type=Path, default=PIPELINE_DIR / "first-sample-map.json")
    scored.add_argument("--labels", type=Path, default=PIPELINE_DIR / "first-labels.jsonl")
    scored.add_argument(
        "--calls",
        type=Path,
        default=PIPELINE_DIR / "first-judgement-calls.md",
        help="the labeller's judgement calls; their number is recorded",
    )
    scored.add_argument("--sample", type=Path, default=PIPELINE_DIR / "first-sample.jsonl")
    scored.add_argument(
        "--passed-sample",
        type=Path,
        default=PIPELINE_DIR / "blind-sample.jsonl",
        help="the sample of passed bullets, whose rows the first-rewrite sample may repeat",
    )
    scored.add_argument("--passed-labels", type=Path, default=PIPELINE_DIR / "blind-labels.jsonl")
    scored.add_argument("--out", type=Path, default=PIPELINE_DIR / "first-rewrites.json")
    scored.add_argument(
        "--check",
        action="store_true",
        help="rescore the committed labels and compare (passes while no label is committed)",
    )


# ---- shared ----------------------------------------------------------------------------------


def _fail(error: BaseException) -> int:
    print(f"error: {error}", file=sys.stderr)
    return EXIT_USAGE


def _meta(config: LLMConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "provider": config.provider,
        "ollama_version": ollama_version(config.base_url),
        "model_digest": ollama_model_digest(config.base_url, config.model),
        "machine": machine_description(),
        "seed_applied": config.provider in SEED_PROVIDERS,
        "temperature": "provider default",
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ignored(config: LLMConfig, *extra: Path) -> list[Path]:
    """Paths a run writes into the repository besides its results: not dirt."""
    paths = list(extra)
    if config.cassette_mode == "record":
        paths.append(config.cassette_dir)
    return paths


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _write_json(path: Path, document: Any) -> None:
    """Write a scored file as every committed one is written."""
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _checked(problems: Sequence[str], what: str) -> int:
    """Report a --check: each difference on stderr, or ``ok: <what>``."""
    for problem in problems:
        print(f"check: {problem}", file=sys.stderr)
    if not problems:
        print(f"ok: {what}")
    return EXIT_FINDINGS if problems else EXIT_OK


# ---- eval pipeline ---------------------------------------------------------------------------


def _pipeline(args: argparse.Namespace) -> int:
    try:
        if args.halves is not None:
            return _pipeline_halves(args)
        if args.citations:
            target = args.out.with_name("citations.json")
            _write_json(target, count_citations(_read_json(args.out), load_bank(args.bank)))
            print(f"wrote {target}")
            return EXIT_OK
        if args.check:
            return _pipeline_check(args)
        if args.rescore:
            return _pipeline_rescore(args)
        if args.replay:
            return _pipeline_replay(args)
        return _pipeline_live(args)
    except _ERRORS as error:
        return _fail(error)


def _pipeline_halves(args: argparse.Namespace) -> int:
    target = args.out.with_name("halves.json")
    target.write_text(
        json.dumps(halves_document(_read_json(args.out), args.halves), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target}")
    return EXIT_OK


def _pipeline_check(args: argparse.Namespace) -> int:
    document = _read_json(args.out)
    problems = check_document(document)
    halves = args.out.with_name("halves.json")
    if halves.exists():
        committed = _read_json(halves)
        problems += differences(
            "halves", committed, halves_document(document, committed["split_after"])
        )
    citations = args.out.with_name("citations.json")
    if citations.exists():
        fresh = count_citations(document, load_bank(args.bank))
        problems += differences("citations", _read_json(citations), fresh)
    for problem in problems:
        print(f"check: {problem}", file=sys.stderr)
    if not problems:
        print(f"ok: the scores in {args.out} are what its runs give")
    return EXIT_FINDINGS if problems else EXIT_OK


def _pipeline_rescore(args: argparse.Namespace) -> int:
    document = _read_json(args.out)
    _write_json(args.out, {**document, "scores": score_runs(document["runs"])})
    written = [str(args.out)]
    halves = args.out.with_name("halves.json")
    if halves.exists():
        _write_json(halves, halves_document(document, _read_json(halves)["split_after"]))
        written.append(str(halves))
    citations = args.out.with_name("citations.json")
    if citations.exists():
        _write_json(citations, count_citations(document, load_bank(args.bank)))
        written.append(str(citations))
    print(f"wrote the scores of {' and '.join(written)}")
    return EXIT_OK


def _pipeline_replay(args: argparse.Namespace) -> int:
    document = _read_json(args.out)
    problems = check_document(document)
    meta, protocol = document["meta"], document["protocol"]
    jd = args.jd or protocol["jd_names"][0]
    seed = args.replay_seed if args.replay_seed is not None else protocol["seeds"][0]
    cassettes = args.cassettes or PIPELINE_DIR / "cassettes"
    problems += replay_subset(
        document,
        bank=load_bank(args.bank),
        bank_path=args.bank,
        jds_dir=args.jds,
        cassette_dir=cassettes,
        jd=jd,
        seed=seed,
        config=LLMConfig(meta["provider"], meta["model"]),
    )
    for problem in problems:
        print(f"replay: {problem}", file=sys.stderr)
    if not problems:
        print(f"replay matches: {jd} with seed {seed} gives the committed run and scores")
    return EXIT_FINDINGS if problems else EXIT_OK


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise PipelineError(f"--seeds must be comma-separated integers, not {text!r}") from None
    if not seeds or len(set(seeds)) != len(seeds):
        raise PipelineError("--seeds needs at least one seed, and no repeats")
    return seeds


def _pipeline_live(args: argparse.Namespace) -> int:
    if args.cassettes is None:
        raise PipelineError("--cassettes is required for a live run: say where to record")
    base = load_config(args.config, provider=args.provider, model=args.model)
    config = dataclasses.replace(base, cassette_mode="record", cassette_dir=args.cassettes)
    bank = load_bank(args.bank)
    seeds = _seeds(args.seeds)
    jd_paths = sorted(args.jds.glob("*.txt"))
    if not jd_paths:
        raise PipelineError(f"no .txt job descriptions in {args.jds}")
    partial = args.out.with_name(f"{args.out.stem}.partial.json")
    ignore = _ignored(config, args.out, partial)
    state = tree_state(Path.cwd(), ignore)
    if state.dirty:
        shown = ", ".join(state.changed[:5])
        raise PipelineError(
            f"the tree has uncommitted changes ({shown}); a live run needs a clean tree so the "
            "recorded commit is the code that ran"
        )
    meta = {**_meta(config), "commit": state.commit, "dirty": False}
    runs: list[dict[str, Any]] = []
    if args.resume and partial.exists():
        saved = _read_json(partial)
        for key in ("commit", "model", "provider"):
            if saved["meta"].get(key) != meta[key]:
                raise PipelineError(f"the partial file was made with another {key}; start over")
        runs = saved["runs"]
        meta = saved["meta"]
    total = len(jd_paths) * len(seeds)

    def progress(record: dict[str, Any]) -> None:
        runs.append(record)
        partial.write_text(json.dumps({"meta": meta, "runs": runs}, indent=1), encoding="utf-8")
        print(
            f"[{len(runs)}/{total}] {record['jd']} seed {record['seed']}: {record['status']}, "
            f"{record['bullets_planned']} bullets, {record['wall_seconds']} s, "
            f"load {record['load_average_1m']}"
            + (f", {record['error']}" if record["error"] else ""),
            file=sys.stderr,
            flush=True,
        )

    done = {(r["jd"], r["seed"]) for r in runs}
    run_all(
        lambda cfg: build_llm(cfg, bank=args.bank),
        config,
        bank,
        jd_paths,
        seeds,
        done=done,
        on_run=progress,
    )
    order = {(p.stem, s): n for n, (p, s) in enumerate((p, s) for p in jd_paths for s in seeds)}
    runs.sort(key=lambda r: order[(r["jd"], r["seed"])])
    protocol = {
        "jds": len(jd_paths),
        "jd_names": [p.stem for p in jd_paths],
        "seeds": seeds,
        "runs": total,
        "model": config.model,
        "provider": config.provider,
        "temperature": "provider default",
        "rounding": "conservative",
    }
    document = build_document(runs, meta, protocol)
    write_results(args.out, document, Path.cwd(), ignore)
    partial.unlink(missing_ok=True)
    scores = document["scores"]
    off, on = scores["gate_off"], scores["gate_on"]
    print(
        f"wrote {args.out}: {scores['runs']['ok']} of {scores['runs']['total']} runs completed, "
        f"{scores['bullets']['rewritten']} rewritten bullets; gate-off violation rate "
        f"{_percent(off['rate'])}; fell back {_percent(on['fallback']['rate'])}"
    )
    return EXIT_OK


# ---- eval judge, judge-runs ------------------------------------------------------------------


def _judge_config(args: argparse.Namespace) -> LLMConfig:
    """The model configuration of a judge command.

    A seed the user gave (``--seed`` or ``TAILOR_SEED``) is warned about when the provider takes
    none. With no seed given, a provider that takes one gets the command's default seed.
    """
    config = load_config(args.config, provider=args.provider, model=args.model, seed=args.seed)
    warning = seed_warning(config)
    if warning:
        print(warning, file=sys.stderr)
    default = getattr(args, "default_seed", None)
    if config.seed is None and default is not None and config.provider in SEED_PROVIDERS:
        config = dataclasses.replace(config, seed=default)
    return config


def _judge_datasets(data: Path, bank: Bank) -> dict[str, list[Row]]:
    """The rows the judge is measured on: the verifier's holdout split and the blind set."""
    rows = load_rows(data / "labeled.jsonl")
    datasets = {
        "holdout": load_split(data / "split.json", rows)["holdout"],
        "blind": load_rows(data / "blind" / "rows.jsonl"),
    }
    for name, chosen in datasets.items():
        problems = validate_rows(chosen, bank)
        if problems:
            raise EvalError(f"the {name} rows do not match the bank:\n  " + "\n  ".join(problems))
    return datasets


def _eval_judge_check(args: argparse.Namespace) -> int:
    bank = load_bank(args.bank)
    document = _read_json(args.out)
    committed = document["datasets"]
    problems: list[str] = []
    for name, rows in _judge_datasets(args.data, bank).items():
        entry = committed[name]
        recorded = [record["id"] for record in entry["records"]]
        if recorded != [row.id for row in rows]:
            problems.append(f"{name}: the recorded verdicts are not for the rows of the dataset")
            continue
        fresh = score_judged(bank, rows, entry["records"])
        if args.rescore:
            committed[name] = {**fresh, "records": entry["records"]}
            continue
        scores = {key: value for key, value in entry.items() if key != "records"}
        problems += differences(name, scores, fresh)
    if args.rescore and not problems:
        _write_json(args.out, document)
        print(f"wrote the scores of {args.out} from its recorded verdicts")
        return EXIT_OK
    return _checked(problems, f"the scores in {args.out} are what its recorded verdicts give")


def _eval_judge(args: argparse.Namespace) -> int:
    try:
        if args.check or args.rescore:
            return _eval_judge_check(args)
        config = _judge_config(args)
        bank = load_bank(args.bank)
        chosen_sets = _judge_datasets(args.data, bank)
        llm = build_llm(config, bank=args.bank)
        datasets: dict[str, Any] = {}
        for name, chosen in chosen_sets.items():
            records = judge_rows(
                llm,
                bank,
                chosen,
                model=config.model,
                on_row=lambda r, name=name: print(  # type: ignore[misc]
                    f"{name} {r['id']}: {r['judge'] if r['judge'] is not None else r['error']}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
            datasets[name] = {**score_judged(bank, chosen, records), "records": records}
            if datasets[name]["rows"] == 0:
                raise PipelineError(
                    f"the judge answered no {name} row (first error: {records[0]['error']})"
                )
        document = {
            "version": 1,
            "meta": {**_meta(config), "judge_seed": config.seed, "usage": llm.usage.to_dict()},
            "datasets": datasets,
        }
        write_results(args.out, document, Path.cwd(), _ignored(config))
    except _ERRORS as error:
        return _fail(error)
    for name, entry in datasets.items():
        print(
            f"{name}: {entry['rows']} rows judged ({entry['errors']} errors); judge caught "
            f"{_percent(entry['judge']['caught_violation_recall'])}, false alarms "
            f"{_percent(entry['judge']['false_alarm_rate'])}; verifier caught "
            f"{_percent(entry['verifier']['caught_violation_recall'])}"
        )
    print(f"wrote {args.out}")
    return EXIT_OK


def _eval_judge_runs_check(args: argparse.Namespace) -> int:
    committed = _read_json(args.out)
    records = committed["bullets"]
    keys = ("jd", "seed", "index", "entry", "outcome")
    shown = [{key: record[key] for key in keys} for record in records]
    problems = differences("bullets", shown, judged_positions(_read_json(args.results)))
    if args.rescore and not problems:
        _write_json(args.out, {**committed, "summary": judged_summary(records)})
        print(f"wrote the summary of {args.out} from its records")
        return EXIT_OK
    problems += differences("summary", committed["summary"], judged_summary(records))
    return _checked(problems, f"{args.out} judged every final bullet and its summary holds")


def _eval_judge_runs(args: argparse.Namespace) -> int:
    try:
        if args.check or args.rescore:
            return _eval_judge_runs_check(args)
        config = _judge_config(args)
        bank = load_bank(args.bank)
        document = _read_json(args.results)
        llm = build_llm(config, bank=args.bank)
        records = judge_runs(
            llm,
            bank,
            document,
            model=config.model,
            on_item=lambda r: print(
                f"{r['jd']}/{r['seed']}/{r['index']}: "
                f"{r['judge'] if r['judge'] is not None else r['error']}",
                file=sys.stderr,
                flush=True,
            ),
        )
        summary = judged_summary(records)
        flagged = summary["flagged"]
        out = {
            "version": 1,
            "meta": {**_meta(config), "judge_seed": config.seed, "usage": llm.usage.to_dict()},
            "summary": summary,
            "bullets": records,
        }
        write_results(args.out, out, Path.cwd(), _ignored(config))
    except _ERRORS as error:
        return _fail(error)
    print(f"wrote {args.out}: {len(records)} bullets judged, {flagged} flagged")
    return EXIT_OK


# ---- eval sample, residual -------------------------------------------------------------------


def _jsonl(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _draw_settings(args: argparse.Namespace) -> tuple[int, int]:
    """The seed and size of a draw: from the options, or with --check from the committed map."""
    if args.check:
        if args.seed is not None or args.size is not None:
            raise PipelineError("--seed and --size are read from the committed map with --check")
        recorded = _read_json(args.map)
        return int(recorded["seed"]), int(recorded["size"])
    if args.seed is None:
        raise PipelineError("--seed is required to draw a sample")
    return args.seed, DEFAULT_DRAW if args.size is None else args.size


def _check_draw(args: argparse.Namespace, rows: list[dict[str, Any]], mapping: Any) -> int:
    problems = differences(str(args.out), _read_jsonl(args.out), rows)
    problems += differences(str(args.map), _read_json(args.map), mapping)
    return _checked(problems, f"{args.out} and {args.map} are the draw their seed gives")


def _eval_sample(args: argparse.Namespace) -> int:
    try:
        seed, size = _draw_settings(args)
        bank = load_bank(args.bank)
        judged = _read_json(args.judged)["bullets"]
        rows, mapping = draw_sample(_read_json(args.results), judged, bank, seed=seed, size=size)
        if args.check:
            return _check_draw(args, rows, mapping)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_jsonl(rows), encoding="utf-8")
        args.map.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    except _ERRORS as error:
        return _fail(error)
    strata = mapping["strata"]
    print(
        f"wrote {args.out} ({len(rows)} rows: {strata['sampled']} drawn at random from "
        f"{strata['unflagged']} unflagged, {strata['flagged']} flagged, {strata['unjudged']} "
        f"unjudged, of a frame of {mapping['frame']}) and {args.map}"
    )
    return EXIT_OK


def _eval_first_sample(args: argparse.Namespace) -> int:
    try:
        seed, size = _draw_settings(args)
        rows, mapping = draw_first_sample(
            _read_json(args.results), load_bank(args.bank), seed=seed, size=size
        )
        if args.check:
            return _check_draw(args, rows, mapping)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_jsonl(rows), encoding="utf-8")
        args.map.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    except _ERRORS as error:
        return _fail(error)
    print(
        f"wrote {args.out} ({len(rows)} rows drawn at random from a frame of {mapping['frame']} "
        f"first rewrites) and {args.map}"
    )
    return EXIT_OK


def _read_labels(path: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_id, given = str(row["id"]), [str(label) for label in row["labels"]]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ResidualError(f"{path}, line {number}: not a label row ({error})") from error
        if row_id in labels:
            raise ResidualError(f"{path}, line {number}: {row_id} is labelled twice")
        labels[row_id] = given
    return labels


def _eval_residual(args: argparse.Namespace) -> int:
    try:
        calls = judgement_call_ids(args.calls.read_text(encoding="utf-8"))
        scored = score_residual(_read_json(args.map), _read_labels(args.labels), calls=calls)
        if args.check:
            problems = differences(args.out.name, _read_json(args.out), scored)
            return _checked(problems, f"{args.out} is what the labels give")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    except _ERRORS as error:
        return _fail(error)
    residual, drawn = scored["residual"], scored["random"]
    low, high = residual["rate_interval"]
    print(
        f"residual: {drawn['violations']} of {drawn['sampled']} drawn bullets violate "
        f"({_percent(drawn['rate'])}); {scored['census']['confirmed']} of "
        f"{scored['census']['flagged']} flagged confirmed; estimated rate "
        f"{_percent(residual['rate'])} (95% interval {_percent(low)} to {_percent(high)}) "
        f"over a frame of {residual['frame']}"
    )
    print(f"wrote {args.out}")
    return EXIT_OK


def _eval_first_rewrites(args: argparse.Namespace) -> int:
    try:
        drawn = args.map.exists() or args.labels.exists() or args.out.exists()
        if args.check and not drawn:
            print(f"ok: no first-rewrite sample ({args.map}), so there is no score to check")
            return EXIT_OK
        if args.check and not args.labels.exists():
            raise ResidualError(f"the sample {args.map} is drawn but {args.labels} is missing")
        if args.check and not args.out.exists():
            raise ResidualError(f"{args.labels} is committed but its score {args.out} is not")
        scored = score_first(
            _read_json(args.results),
            _read_json(args.map),
            _read_labels(args.labels),
            repeats=identical_rows(_read_jsonl(args.sample), _read_jsonl(args.passed_sample)),
            passed_labels=_read_labels(args.passed_labels),
            calls=judgement_call_ids(args.calls.read_text(encoding="utf-8")),
        )
        if args.check:
            problems = differences(args.out.name, _read_json(args.out), scored)
            return _checked(problems, f"{args.out} is what the labels give")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    except _ERRORS as error:
        return _fail(error)
    rate, flag = scored["violation_rate"], scored["gate_off_flag"]
    low, high = rate["interval"]
    print(
        f"first rewrites: {rate['violating']} of {rate['rows']} labelled violating "
        f"({_percent(rate['rate'])}, 95% interval {_percent(low)} to {_percent(high)}); the "
        f"gate-off flag: precision {_percent(flag['precision'])}, recall "
        f"{_percent(flag['recall'])}"
    )
    print(f"wrote {args.out}")
    return EXIT_OK


# ---- tailor judge ----------------------------------------------------------------------------


def run_judge(args: argparse.Namespace) -> int:
    """``tailor judge``: judge each bullet of a file."""
    try:
        config = _judge_config(args)
        bank = load_bank(args.bank)
        bullets = load_bullets(args.bullets)
        llm = build_llm(config, bank=args.bank)
        results: list[dict[str, Any]] = []
        for index, bullet in enumerate(bullets):
            try:
                ids = (
                    bullet.sources
                    if bullet.sources is not None
                    else tuple(f.id for f in bank.facts_for(bullet.entry))
                )
                shown = input_for(bank, bullet.text, bullet.entry, ids)
            except KeyError as error:
                raise VerifyError(f"bullet {index + 1}: unknown entry or fact {error}") from None
            verdict = judge_bullet(llm, shown, model=config.model)
            results.append(
                {
                    "index": index,
                    "id": bullet.id,
                    "text": bullet.text,
                    "verdict": "flagged" if verdict.flagged else "ok",
                    "classes": list(verdict.classes),
                    "reason": verdict.reason,
                }
            )
    except _ERRORS as error:
        return _fail(error)
    if args.format == "json":
        print(json.dumps({"model": config.model, "results": results}, indent=2))
    else:
        for item in results:
            label = ", ".join(item["classes"]) if item["classes"] else "ok"
            print(f"{item['index'] + 1}. {label}: {item['text']}\n   {item['reason']}")
    return EXIT_FINDINGS if any(r["classes"] for r in results) else EXIT_OK


_HANDLERS: Final = {
    "pipeline": _pipeline,
    "judge": _eval_judge,
    "judge-runs": _eval_judge_runs,
    "sample": _eval_sample,
    "first-sample": _eval_first_sample,
    "first-rewrites": _eval_first_rewrites,
    "residual": _eval_residual,
}


def run_eval(args: argparse.Namespace) -> int:
    """Run an evaluation subcommand other than ``verifier``."""
    return _HANDLERS[args.eval_command](args)
