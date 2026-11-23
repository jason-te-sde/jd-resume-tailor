"""The ``tailor`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tailor import __version__
from tailor.bank import Bank, BankError, ProjectEntry, WorkEntry, load_bank
from tailor.evals.commands import add_eval_parsers, add_judge_parser, run_eval, run_judge
from tailor.evals.verifier import (
    DEFAULT_TOLERANCE,
    LABELS,
    EvalError,
    build_split,
    check_gate,
    evaluate,
    load_rows,
    load_split,
    read_split,
    render_table,
    validate_rows,
)
from tailor.jd import JDError, load_job
from tailor.llm import LLMError, build_llm, load_config
from tailor.llm.config import seed_warning
from tailor.plan import Plan, build_plan
from tailor.plan import render_json as render_plan_json
from tailor.plan import render_markdown as render_plan_markdown
from tailor.render.compile import ENGINES, find_engine
from tailor.render.fit import FitResult
from tailor.render.model import InputError, RenderError, load_plan, load_profile
from tailor.render.pipeline import render_plan
from tailor.verify.report import render_json, render_markdown
from tailor.verify.rounding import Rounding
from tailor.verify.verifier import VerifyError, load_bullets, verify

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_VIOLATIONS = 1
EXIT_RENDER = 1
EXIT_USAGE = 2

_CHECK_EPILOG = f"""\
Exit status: {EXIT_OK} when the bank is valid (warnings are printed but do not fail the run),
{EXIT_INVALID} when it is not valid JSON or breaks the bank format, and {EXIT_USAGE} when the
command line is wrong or the file cannot be read.
"""


_VERIFY_EPILOG = f"""\
Each bullet is {{"text": ..., "entry": <entry id>, "sources": [<fact ids>]?, "id": ...?}}; without
"sources" every fact of the entry is a source. The file is an array of bullets or an object with a
"bullets" array.

Exit status: {EXIT_OK} when every bullet is ok, {EXIT_VIOLATIONS} when at least one has a
violation, {EXIT_USAGE} when the command could not run (wrong command line, an unreadable or invalid
bank or bullets file, a bullet that names an entry or fact the bank does not have).
"""


_PLAN_EPILOG = f"""\
The model comes from --model/--provider, the TAILOR_MODEL and TAILOR_PROVIDER environment variables
or tailor.toml; there is no default. TAILOR_CASSETTES=record|replay|off (and TAILOR_CASSETTE_DIR)
records model calls to files or replays them without a network.

Writes the plan to --out and a Markdown provenance report next to it (same name, .md).

Exit status: {EXIT_OK} when the plan was written, {EXIT_USAGE} when it could not be (wrong command
line, an unreadable or invalid file, no model configured, a failed model call).
"""


_RUN_EPILOG = f"""\
Plans as `tailor plan` does (same model options), then writes to --out: resume.tex, resume.pdf,
plan.json and provenance.md (the plan's report plus what was dropped to fit one page). The resume
is fitted by dropping bullets and entries in a fixed order, never by rewording them.

Needs pdflatex or xelatex on PATH; poppler's pdftotext is optional and enables the layout checks
that read the PDF. TAILOR_CASSETTES works as for `tailor plan`.

Exit status: {EXIT_OK} when the PDF was written and is one page, {EXIT_RENDER} when LaTeX failed or
the plan cannot be fitted to one page, {EXIT_USAGE} when the command could not run (wrong command
line, an unreadable or invalid file, no model configured, a failed model call, no TeX engine).
"""

_RENDER_EPILOG = f"""\
Renders the plan that `tailor plan` wrote, with no model call: resume.tex, resume.pdf and
provenance.md go to --out, and a copy of the plan to plan.json there.

Exit status: {EXIT_OK} when the PDF was written and is one page, {EXIT_RENDER} when LaTeX failed or
the plan cannot be fitted to one page, {EXIT_USAGE} when the command could not run (wrong command
line, an unreadable or invalid file, no TeX engine).
"""


def render_summary(path: str, bank: Bank) -> str:
    """Describe a valid bank: its size, what the verifier will have to work with, and doubts."""
    entry_counts = Counter(entry.type for entry in bank.entries)
    candidates = bank.candidates()
    candidate_work = sum(isinstance(entry, WorkEntry) for entry in candidates)
    candidate_projects = sum(isinstance(entry, ProjectEntry) for entry in candidates)
    fact_kinds = Counter(fact.kind for entry in bank.entries for fact in bank.facts_for(entry.id))
    claims = {entry.id: len(bank.do_not_claim(entry.id)) for entry in bank.entries}
    groups = bank.same_project_groups()
    warnings = bank.warnings()

    lines = [
        f"ok: {path}",
        f"{'entries':<14}{sum(entry_counts.values())} "
        f"(work: {entry_counts['work']}, project: {entry_counts['project']})",
        f"{'candidates':<14}{len(candidates)} "
        f"(work: {candidate_work}, project with code: {candidate_projects})",
        f"{'facts':<14}{sum(fact_kinds.values())} ({fact_kinds['bullet']} from bullets, "
        f"{fact_kinds['scope']} from scope, {fact_kinds['explicit']} explicit)",
        f"{'do not claim':<14}{sum(claims.values())} claims in "
        f"{sum(1 for count in claims.values() if count)} entries",
    ]
    if groups:
        noun = "group" if len(groups) == 1 else "groups"
        lines.append(f"{'same project':<14}{len(groups)} {noun}")
        lines.extend(f"  {', '.join(sorted(group))}" for group in groups)
    else:
        lines.append(f"{'same project':<14}none")
    lines.append(f"{'warnings':<14}{len(warnings) or 'none'}")
    lines.extend(f"  {warning}" for warning in warnings)
    return "\n".join(lines)


def _bank_check(path: Path) -> int:
    try:
        bank = load_bank(path)
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_INVALID
    except OSError as error:
        print(f"error: cannot read {path}: {error.strerror}", file=sys.stderr)
        return EXIT_USAGE
    print(render_summary(str(path), bank))
    return EXIT_OK


def _verify(bank_path: Path, bullets_path: Path, rounding: Rounding, output: str) -> int:
    try:
        bank = load_bank(bank_path)
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except OSError as error:
        print(f"error: cannot read {bank_path}: {error.strerror}", file=sys.stderr)
        return EXIT_USAGE
    try:
        report = verify(bank, load_bullets(bullets_path), rounding=rounding)
    except VerifyError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(render_json(report) if output == "json" else render_markdown(report), end="")
    return EXIT_OK if report.ok else EXIT_VIOLATIONS


def _make_plan(args: argparse.Namespace) -> Plan:
    """Run ``tailor plan`` for the arguments both ``plan`` and ``run`` share."""
    config = load_config(args.config, provider=args.provider, model=args.model, seed=args.seed)
    warning = seed_warning(config)
    if warning:
        print(warning, file=sys.stderr)
    bank = load_bank(args.bank)
    job = load_job(args.jd)
    profile = None
    if args.profile is not None:
        json.loads(args.profile.read_text(encoding="utf-8"))
        profile = args.profile.name
    llm = build_llm(config, bank=args.bank)
    return build_plan(llm, config, bank, job, rounding=args.rounding, profile=profile)


def _plan(args: argparse.Namespace) -> int:
    try:
        plan = _make_plan(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_plan_json(plan), encoding="utf-8")
        report = args.out.with_suffix(".md")
        if report == args.out:
            report = args.out.with_suffix(".report.md")
        report.write_text(render_plan_markdown(plan), encoding="utf-8")
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except (LLMError, JDError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    counts = plan.counts
    print(
        f"wrote {args.out} and {report}: {counts['bullets']} bullets "
        f"({counts['first_pass']} first pass, {counts['after_retry']} after retry, "
        f"{counts['fallback']} fell back, {counts['verbatim']} verbatim)"
    )
    return EXIT_OK


def _summarize(result: FitResult, out: Path) -> str:
    bullets = sum(len(e.bullets) for e in result.plan.entries)
    dropped = len(result.drops)
    return (
        f"wrote {out / 'resume.pdf'} ({result.built.report.pages} page, "
        f"{len(result.plan.entries)} entries, {bullets} bullets, {dropped} dropped to fit) "
        f"and {out / 'provenance.md'}"
    )


def _render_failure(error: RenderError) -> int:
    print(f"error: {error}", file=sys.stderr)
    return EXIT_USAGE if isinstance(error, InputError) else EXIT_RENDER


def _run(args: argparse.Namespace) -> int:
    try:
        # Before any model call: a run that cannot render should not spend tokens first.
        load_profile(args.profile)
        find_engine(args.engine)
        plan = _make_plan(args)
        args.out.mkdir(parents=True, exist_ok=True)
        plan_path = args.out / "plan.json"
        plan_path.write_text(render_plan_json(plan), encoding="utf-8")
        document, resume_plan = load_plan(plan_path)
        result = render_plan(
            document, resume_plan, args.profile, args.out,
            engine=args.engine, one_line=args.one_line_bullets,
        )  # fmt: skip
    except RenderError as error:
        return _render_failure(error)
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except (LLMError, JDError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(_summarize(result, args.out))
    return EXIT_OK


def _render(args: argparse.Namespace) -> int:
    try:
        document, plan = load_plan(args.plan)
        load_profile(args.profile)
        find_engine(args.engine)
        args.out.mkdir(parents=True, exist_ok=True)
        copy = args.out / "plan.json"
        if copy.resolve() != args.plan.resolve():
            copy.write_text(args.plan.read_text(encoding="utf-8"), encoding="utf-8")
        result = render_plan(
            document, plan, args.profile, args.out,
            engine=args.engine, one_line=args.one_line_bullets,
        )  # fmt: skip
    except RenderError as error:
        return _render_failure(error)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(_summarize(result, args.out))
    return EXIT_OK


def _score_only(args: argparse.Namespace) -> int:
    data: Path = args.data
    if args.check or args.write_split:
        print(
            "error: --score-only cannot be combined with --check or --write-split", file=sys.stderr
        )
        return EXIT_USAGE
    try:
        bank = load_bank(args.bank)
        path = data / "labeled.jsonl"
        rows = load_rows(path if path.exists() else data / "rows.jsonl")
        problems = validate_rows(rows, bank)
        if problems:
            raise EvalError("the dataset does not match the bank:\n  " + "\n  ".join(problems))
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except EvalError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(render_table("all rows", evaluate(rows, bank)))
    return EXIT_OK


def _eval_verifier(args: argparse.Namespace) -> int:
    if args.score_only:
        return _score_only(args)
    data: Path = args.data
    try:
        bank = load_bank(args.bank)
        rows = load_rows(data / "labeled.jsonl")
        problems = validate_rows(rows, bank)
        if problems:
            raise EvalError("the dataset does not match the bank:\n  " + "\n  ".join(problems))
        if args.write_split:
            recorded = read_split(data / "split.json") if (data / "split.json").exists() else None
            (data / "split.json").write_text(
                json.dumps(build_split(rows, previous=recorded), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {data / 'split.json'}")
            return EXIT_OK
        sides = load_split(data / "split.json", rows)
        scores = {side: evaluate(sides[side], bank) for side in ("dev", "holdout")}
        baseline = None
        if args.check:
            baseline = json.loads((data / "results.json").read_text(encoding="utf-8"))
    except BankError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except (EvalError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    shown = ("dev", "holdout") if args.split == "both" else (args.split,)
    print("\n\n".join(render_table(side, scores[side]) for side in shown))
    if args.check:
        assert baseline is not None
        failures = check_gate(scores["holdout"], baseline["holdout"], args.tolerance)
        for failure in failures:
            print(f"gate: {failure}", file=sys.stderr)
        return EXIT_VIOLATIONS if failures else EXIT_OK
    if args.split == "both":
        document = {
            "dataset": {
                "rows": len(rows),
                "dev": len(sides["dev"]),
                "holdout": len(sides["holdout"]),
                "violation_rows": sum("ok" not in row.labels for row in rows),
                "labels": {name: sum(name in row.labels for row in rows) for name in LABELS},
            },
            "policy": "conservative",
            "dev": scores["dev"].to_dict(),
            "holdout": scores["holdout"].to_dict(),
        }
        (data / "results.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {data / 'results.json'}")
    return EXIT_OK


class _Parser(argparse.ArgumentParser):
    """A parser that takes no abbreviated option: ``--seed`` is never ``--seeds``.

    Every subcommand's parser is made from the class of its parent, so this holds throughout.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="tailor",
        description="Tailor a resume to a job description from a bank of facts.",
    )
    parser.add_argument("--version", action="version", version=f"tailor {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    bank = commands.add_parser("bank", help="work with a fact bank")
    bank_commands = bank.add_subparsers(dest="bank_command", required=True, metavar="subcommand")
    check = bank_commands.add_parser(
        "check",
        help="validate a bank and summarise it",
        description="Validate a fact bank against the bank format and print a summary of it.",
        epilog=_CHECK_EPILOG,
    )
    check.add_argument("path", type=Path, help="the bank JSON file")

    evaluation = commands.add_parser("eval", help="measure the tool against labelled data")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True, metavar="what")
    verifier = eval_commands.add_parser(
        "verifier",
        help="precision and recall of the verifier on the labelled dataset",
        description=(
            "Run the verifier on eval/verifier/labeled.jsonl and score it per class on the dev "
            "and holdout splits. A full run writes results.json; --split dev (for tuning) and "
            "--check (the CI gate) write nothing. The holdout is for the final numbers: tune "
            "on dev."
        ),
    )
    verifier.add_argument("--data", type=Path, default=Path("eval/verifier"), help="dataset dir")
    verifier.add_argument(
        "--bank", type=Path, default=Path("examples/alex-rivera/bank.json"), help="the bank"
    )
    verifier.add_argument("--split", choices=["dev", "holdout", "both"], default="both")
    verifier.add_argument(
        "--check", action="store_true", help="fail if the holdout is worse than results.json"
    )
    verifier.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    verifier.add_argument(
        "--write-split",
        action="store_true",
        help="rewrite split.json: recorded sides are kept, rows not in it go to dev",
    )
    verifier.add_argument(
        "--score-only",
        action="store_true",
        help="score every row of the dataset (labeled.jsonl or rows.jsonl) with no split; "
        "writes nothing and is not part of the gate",
    )

    add_eval_parsers(eval_commands)
    add_judge_parser(commands)

    verify_parser = commands.add_parser(
        "verify",
        help="check bullets against the facts they cite",
        description=(
            "Check each bullet for numbers, technologies and claims that the facts it cites "
            "do not support."
        ),
        epilog=_VERIFY_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify_parser.add_argument("--bank", type=Path, required=True, help="the bank JSON file")
    verify_parser.add_argument(
        "--bullets", type=Path, required=True, help="the bullets to check, as JSON"
    )
    verify_parser.add_argument(
        "--rounding",
        choices=["conservative", "nearest"],
        default="conservative",
        help=(
            "conservative (default): a number may only be rounded the unflattering way; "
            "nearest: ordinary rounding at two significant figures"
        ),
    )
    verify_parser.add_argument(
        "--format", choices=["json", "md"], default="md", help="report format (default: md)"
    )

    plan = commands.add_parser(
        "plan",
        help="choose and rewrite bullets for a job description",
        description=(
            "Extract a job description, choose the entries and bullets that fit it, rewrite them "
            "with the facts they cite, and keep only what the verifier passes."
        ),
        epilog=_PLAN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan.add_argument(
        "--jd", required=True, help="a text file, or a Greenhouse/Lever/Ashby/job URL"
    )
    plan.add_argument("--bank", type=Path, required=True, help="the bank JSON file")
    plan.add_argument("--profile", type=Path, help="the profile JSON file (recorded in the plan)")
    plan.add_argument("--out", type=Path, required=True, help="where to write plan.json")
    plan.add_argument("--model", help="the model id (overrides TAILOR_MODEL and tailor.toml)")
    plan.add_argument("--provider", choices=["anthropic", "openai"], help="the provider")
    plan.add_argument("--config", type=Path, help="a tailor.toml (default: ./tailor.toml if any)")
    plan.add_argument("--rounding", choices=["conservative", "nearest"], default="conservative")
    plan.add_argument(
        "--seed", type=int, help="the seed sent to a provider that takes one (TAILOR_SEED)"
    )

    def render_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--profile", type=Path, required=True, help="the profile JSON file")
        command.add_argument("--out", type=Path, required=True, help="the output directory")
        command.add_argument("--engine", choices=list(ENGINES), help="the TeX engine to use first")
        command.add_argument(
            "--one-line-bullets",
            action="store_true",
            help="also drop until no bullet wraps (needs pdftotext); default: only fit one page",
        )

    run = commands.add_parser(
        "run",
        help="plan, render and fit a one-page resume",
        description="Plan a resume for a job description, render it to LaTeX and PDF, and fit it.",
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("--jd", required=True, help="a text file, or a Greenhouse/Lever/Ashby/job URL")
    run.add_argument("--bank", type=Path, required=True, help="the bank JSON file")
    render_options(run)
    run.add_argument("--model", help="the model id (overrides TAILOR_MODEL and tailor.toml)")
    run.add_argument("--provider", choices=["anthropic", "openai"], help="the provider")
    run.add_argument("--config", type=Path, help="a tailor.toml (default: ./tailor.toml if any)")
    run.add_argument("--rounding", choices=["conservative", "nearest"], default="conservative")
    run.add_argument(
        "--seed", type=int, help="the seed sent to a provider that takes one (TAILOR_SEED)"
    )

    render = commands.add_parser(
        "render",
        help="render an existing plan to a one-page PDF",
        description="Render the plan.json that `tailor plan` wrote, and fit it to one page.",
        epilog=_RENDER_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    render.add_argument("--plan", type=Path, required=True, help="the plan.json to render")
    render_options(render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit status."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:  # argparse exits for --help, --version and usage errors
        return stop.code if isinstance(stop.code, int) else EXIT_USAGE
    if args.command == "judge":
        return run_judge(args)
    if args.command == "eval":
        return _eval_verifier(args) if args.eval_command == "verifier" else run_eval(args)
    if args.command == "plan":
        return _plan(args)
    if args.command == "run":
        return _run(args)
    if args.command == "render":
        return _render(args)
    if args.command == "verify":
        return _verify(args.bank, args.bullets, args.rounding, args.format)
    return _bank_check(args.path)
