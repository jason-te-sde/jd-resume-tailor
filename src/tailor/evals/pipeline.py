"""The pipeline evaluation: run ``tailor plan`` over job descriptions and seeds, and score the runs.

One live run of the plan gives both sides of the gate. The first answer the model gives for each
bullet is checked by the verifier and kept with its findings: that is what the resume would have
held with the gate off. The bullet that ends up on the resume (passed first time, passed after a
retry, or the bank's own wording) is what it holds with the gate on. The model is not run twice.

A run is recorded as a plain dict (see :func:`run_once`); :func:`score_runs` computes every metric
from those dicts and nothing else, so the committed results file can be re-scored, and a replay of
one run from its cassettes can be checked against the committed numbers, with no model.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from tailor.bank import Bank
from tailor.jd import load_job
from tailor.llm import LLM, LLMConfig, build_llm
from tailor.plan import build_plan
from tailor.rewrite import COPIED_CITATION
from tailor.verify import KINDS
from tailor.verify.rounding import Rounding

RESULTS_VERSION: Final = 1
LOAD_SPLIT: Final = 2.0
"""Runs are also timed apart by the 1-minute load average at their start: below this, and not."""
_TIMING: Final = ("wall_seconds", "load_average_1m")
_ABSOLUTE_PATH: Final = re.compile(r"(?<![\w.:/-])/(?:Users|home|private|var|tmp)/")
LLMFactory = Callable[[LLMConfig], LLM]


class PipelineError(Exception):
    """The evaluation cannot run or its results cannot be written."""


# ---- reading a run -----------------------------------------------------------------------------


def bullet_kinds(problems: Iterable[str]) -> tuple[tuple[str, ...], bool]:
    """Split the problems found with a rewrite into kinds and structural ones.

    A verifier finding is written ``<kind>: <text> - <evidence>``, and so is a bullet the verifier
    passed whose text carries a copied fact id (``copied_citation: ...``, checked only after the
    verifier passes). Anything else (an empty bullet, no fact cited, a fact the entry does not
    have) is structural: the verifier never ran on it.

    Returns:
        The distinct kinds in order of appearance (verifier kinds, or ``copied_citation``), and
        whether any problem was structural.
    """
    kinds: dict[str, None] = {}
    structural = False
    for problem in problems:
        kind = problem.split(":", 1)[0]
        if kind in KINDS or kind == COPIED_CITATION:
            kinds.setdefault(kind)
        else:
            structural = True
    return tuple(kinds), structural


def _bullet_records(plan: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in plan.entries:
        for bullet in result.bullets:
            first = bullet.attempts[0] if bullet.attempts else None
            kinds, structural = bullet_kinds(first.problems) if first else ((), False)
            records.append(
                {
                    "entry": result.entry.id,
                    "outcome": bullet.outcome,
                    "first_text": first.text if first else None,
                    "first_sources": list(first.sources) if first else [],
                    "first_kinds": list(kinds),
                    "first_structural": structural,
                    "first_problems": list(first.problems) if first else [],
                    "final_text": bullet.text,
                    "final_sources": list(bullet.sources),
                    "original": bullet.original,
                }
            )
    return records


def _scrub(message: str) -> str:
    """The message without machine paths, so a recorded error can be committed."""
    cleaned = message.replace(str(Path.home()), "~")
    return re.sub(r"(?<![\w.:/-])/(?:Users|home|private|var|tmp)/\S*", "<path>", cleaned)


def run_once(
    make_llm: LLMFactory,
    config: LLMConfig,
    bank: Bank,
    jd_path: Path,
    seed: int,
    rounding: Rounding = "conservative",
) -> dict[str, Any]:
    """Run ``tailor plan`` once and record what it did.

    A run that raises anything is recorded with ``status`` ``failed`` and the error, not dropped:
    a crash is a result, and leaving it out would flatter the rest.

    Returns:
        ``{jd, seed, status, error, bullets_planned, calls, input_tokens, output_tokens,
        wall_seconds, load_average_1m, bullets}``; ``bullets`` is empty for a failed run.
    """
    seeded = dataclasses.replace(config, seed=seed)
    record: dict[str, Any] = {
        "jd": jd_path.stem,
        "seed": seed,
        "status": "ok",
        "error": None,
        "bullets_planned": 0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_seconds": 0.0,
        "load_average_1m": round(os.getloadavg()[0], 2),
        "bullets": [],
    }
    llm: LLM | None = None
    started = time.perf_counter()
    try:
        llm = make_llm(seeded)
        job = load_job(str(jd_path))
        plan = build_plan(llm, seeded, bank, job, rounding=rounding)
        record["bullets"] = _bullet_records(plan)
        record["bullets_planned"] = len(record["bullets"])
    except Exception as error:  # a failed run is data
        record["status"] = "failed"
        record["error"] = _scrub(f"{type(error).__name__}: {error}")
    record["wall_seconds"] = round(time.perf_counter() - started, 2)
    if llm is not None:
        record["calls"] = llm.usage.calls
        record["input_tokens"] = llm.usage.input_tokens
        record["output_tokens"] = llm.usage.output_tokens
    return record


def run_all(
    make_llm: LLMFactory,
    config: LLMConfig,
    bank: Bank,
    jd_paths: Sequence[Path],
    seeds: Sequence[int],
    *,
    done: Collection[tuple[str, int]] = (),
    on_run: Callable[[dict[str, Any]], None] | None = None,
    rounding: Rounding = "conservative",
) -> list[dict[str, Any]]:
    """Run every job description under every seed, skipping the ``(jd, seed)`` pairs in ``done``."""
    runs: list[dict[str, Any]] = []
    for jd_path in jd_paths:
        for seed in seeds:
            if (jd_path.stem, seed) in done:
                continue
            record = run_once(make_llm, config, bank, jd_path, seed, rounding)
            runs.append(record)
            if on_run is not None:
                on_run(record)
    return runs


# ---- scoring ---------------------------------------------------------------------------------


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _share(count: int, total: int) -> dict[str, Any]:
    """A count with its rate (4 places) and its percent, exact from the counts for display."""
    return {
        "bullets": count,
        "rate": _rate(count, total),
        "percent": count * 100 / total if total else None,
    }


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None}
    return {
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values),
    }


def _summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in runs if r["status"] == "ok"]
    bullets = [b for r in ok for b in r["bullets"]]
    rewritten = [b for b in bullets if b["outcome"] != "verbatim"]
    total = len(rewritten)
    flagged = [b for b in rewritten if any(k in KINDS for k in b["first_kinds"])]
    below = [r for r in ok if r["load_average_1m"] < LOAD_SPLIT]
    above = [r for r in ok if r["load_average_1m"] >= LOAD_SPLIT]
    return {
        "runs": {"total": len(runs), "ok": len(ok), "failed": len(runs) - len(ok)},
        "bullets": {
            "planned": len(bullets),
            "rewritten": total,
            "verbatim": len(bullets) - total,
        },
        "gate_off": {
            "flagged": len(flagged),
            "rate": _rate(len(flagged), total),
            "percent": len(flagged) * 100 / total if total else None,
            "by_class": {
                kind: _share(sum(kind in b["first_kinds"] for b in rewritten), total)
                for kind in KINDS
            },
            "structural": sum(b["first_structural"] for b in rewritten),
            "copied_citation": sum(COPIED_CITATION in b["first_kinds"] for b in rewritten),
        },
        "gate_on": {
            outcome: _share(sum(b["outcome"] == outcome for b in rewritten), total)
            for outcome in ("first_pass", "after_retry", "fallback")
        },
        "per_resume": {
            "seconds": _spread([r["wall_seconds"] for r in ok]),
            "input_tokens": _spread([r["input_tokens"] for r in ok]),
            "output_tokens": _spread([r["output_tokens"] for r in ok]),
            "calls": _spread([r["calls"] for r in ok]),
            "load_average_1m": _spread([r["load_average_1m"] for r in ok]),
        },
        "by_load": {
            "threshold": LOAD_SPLIT,
            "below": {"runs": len(below), "seconds": _spread([r["wall_seconds"] for r in below])},
            "at_or_above": {
                "runs": len(above),
                "seconds": _spread([r["wall_seconds"] for r in above]),
            },
        },
        "totals": {
            "calls": sum(r["calls"] for r in runs),
            "input_tokens": sum(r["input_tokens"] for r in runs),
            "output_tokens": sum(r["output_tokens"] for r in runs),
        },
    }


def score_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every metric of a set of runs, overall and per job description.

    Rates are over the bullets the model rewrote: a bullet of an ``adaptable: false`` entry is
    copied without a call and is counted as ``verbatim``, not in any rate. Each rate is rounded to
    4 places and has a ``percent`` next to it, exact from the counts, which is what a rounded
    display figure is taken from. Time, tokens and load are spreads over the runs that completed,
    and time is also given apart for runs that started below a load of :data:`LOAD_SPLIT` and the
    rest. The totals of calls and tokens are over every run, failed ones included, since their
    calls were made. A rate with nothing to divide is ``None``.
    """
    scores = _summary(runs)
    names = sorted({r["jd"] for r in runs})
    scores["by_jd"] = {name: _summary([r for r in runs if r["jd"] == name]) for name in names}
    return scores


# ---- the document ----------------------------------------------------------------------------


def build_document(
    runs: Sequence[dict[str, Any]], meta: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """The results file: the protocol, who and what ran, every run, and the scores."""
    return {
        "version": RESULTS_VERSION,
        "protocol": protocol,
        "meta": meta,
        "runs": list(runs),
        "scores": score_runs(runs),
    }


def halves_document(document: dict[str, Any], split_after: int) -> dict[str, Any]:
    """The scores of the first ``split_after`` runs and of the rest, as the scorer gives them.

    A run of the evaluation was resumed on a machine whose operating system had been updated, so
    the headline rates are also given for the runs before and after the break, to make a shift
    visible. Per-posting scores are left out: a half holds few runs of each posting.
    """
    runs = document["runs"]
    halves: dict[str, Any] = {"split_after": split_after}
    for name, part in (("first", runs[:split_after]), ("second", runs[split_after:])):
        scores = score_runs(part)
        scores.pop("by_jd")
        halves[name] = scores
    return halves


def differences(path: str, committed: Any, fresh: Any) -> list[str]:
    """One message per leaf where ``committed`` and ``fresh`` differ, under the name ``path``."""
    if isinstance(committed, dict) and isinstance(fresh, dict):
        found: list[str] = []
        for key in sorted(set(committed) | set(fresh)):
            if key not in committed or key not in fresh:
                found.append(f"{path}.{key} is missing from one side")
            else:
                found += differences(f"{path}.{key}", committed[key], fresh[key])
        return found
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{path} has {len(committed)} items, now {len(fresh)}"]
        found = []
        for index, (old, new) in enumerate(zip(committed, fresh, strict=True)):
            found += differences(f"{path}[{index}]", old, new)
        return found
    return [] if committed == fresh else [f"{path}: committed {committed!r}, now {fresh!r}"]


def check_document(document: dict[str, Any]) -> list[str]:
    """Recompute the scores from the runs; one message per number that is not the committed one."""
    return differences("scores", document["scores"], score_runs(document["runs"]))


def replay_subset(
    document: dict[str, Any],
    *,
    bank: Bank,
    bank_path: Path,
    jds_dir: Path,
    cassette_dir: Path,
    jd: str,
    seed: int,
    config: LLMConfig,
) -> list[str]:
    """Replay one committed run from its cassettes and compare it with the committed record.

    The plan is rebuilt with no model (every call is answered from ``cassette_dir``), recorded as a
    live run is, and compared with the committed run, field by field, except wall time and load
    (which are measurements of the live run and are carried over). The scorer is then run on both.

    Returns:
        One message per difference; empty when the replay reproduces the committed numbers.

    Raises:
        PipelineError: The results hold no completed run for ``jd`` and ``seed``.
    """
    committed = next((r for r in document["runs"] if r["jd"] == jd and r["seed"] == seed), None)
    if committed is None or committed["status"] != "ok":
        raise PipelineError(f"the results hold no completed run for {jd} with seed {seed}")
    replaying = dataclasses.replace(
        config, cassette_mode="replay", cassette_dir=cassette_dir, seed=seed
    )
    fresh = run_once(
        lambda cfg: build_llm(cfg, bank=bank_path), replaying, bank, jds_dir / f"{jd}.txt", seed
    )
    if fresh["status"] != "ok":
        return [f"the replay failed: {fresh['error']}"]
    for key in _TIMING:
        fresh[key] = committed[key]
    problems = differences(f"run {jd}/{seed}", committed, fresh)
    problems += differences(f"score {jd}/{seed}", score_runs([committed]), score_runs([fresh]))
    return problems


# ---- the tree --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeState:
    """The repository at the moment of a run.

    Attributes:
        commit: ``HEAD``.
        dirty: Whether any tracked file is modified or any file is untracked (outside ``ignore``).
        changed: The paths that make it dirty, relative to the repository root.
    """

    commit: str
    dirty: bool
    changed: tuple[str, ...]


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"git {' '.join(args)} failed in {repo}: {error}") from error
    return done.stdout


def tree_state(repo: Path, ignore: Sequence[Path] = ()) -> TreeState:
    """The commit and whether the working tree has changes, not counting the paths in ``ignore``.

    The evaluation writes cassettes and a results file, which would otherwise make every tree
    dirty as soon as the first run finished.
    """
    root = Path(_git(repo, "rev-parse", "--show-toplevel").strip()).resolve()
    commit = _git(repo, "rev-parse", "HEAD").strip()
    skipped = [Path(p).resolve() for p in ignore]
    changed: list[str] = []
    for entry in _git(repo, "status", "--porcelain", "--untracked-files=all", "-z").split("\0"):
        if len(entry) < 4:
            continue
        relative = entry[3:]
        absolute = (root / relative).resolve()
        if any(absolute == s or s in absolute.parents for s in skipped):
            continue
        changed.append(relative)
    return TreeState(commit, bool(changed), tuple(changed))


def write_results(
    path: Path, document: dict[str, Any], repo: Path, ignore: Sequence[Path] = ()
) -> TreeState:
    """Write the results file, recording the commit, and only from a clean tree.

    Raises:
        PipelineError: The tree has uncommitted changes (outside ``ignore`` and the file itself),
            so the commit would not say what produced the numbers.
    """
    state = tree_state(repo, [path, *ignore])
    if state.dirty:
        shown = ", ".join(state.changed[:5]) + (" ..." if len(state.changed) > 5 else "")
        raise PipelineError(
            f"refusing to write {path}: the tree has uncommitted changes ({shown}); "
            "commit them first, so the recorded commit is the code that ran"
        )
    meta = {**document.get("meta", {}), "commit": state.commit, "dirty": False}
    text = json.dumps({**document, "meta": meta}, indent=2, ensure_ascii=False) + "\n"
    leak = _ABSOLUTE_PATH.search(text) or (str(Path.home()) in text)
    if leak:
        raise PipelineError(
            f"refusing to write {path}: the results hold an absolute path (a home directory or "
            "a temporary location); results are committed, so they carry no machine paths"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return state


# ---- the machine -----------------------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    try:
        done = subprocess.run(
            ["sysctl", "-n", name], check=True, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def machine_description() -> dict[str, Any]:
    """The machine the runs happened on: OS, architecture, CPU and memory where they can be read."""
    memory = _sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "memory_gb": round(int(memory) / 2**30) if memory and memory.isdigit() else None,
        "python": platform.python_version(),
    }


def _ollama(base_url: str | None, path: str) -> Any:
    """The JSON a local Ollama answers at ``path``, or ``None`` when it cannot be read."""
    if not base_url:
        return None
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        return httpx.get(f"{root}{path}", timeout=5.0).json()
    except (httpx.HTTPError, ValueError):
        return None


def ollama_version(base_url: str | None) -> str | None:
    """The version a local Ollama reports, or ``None`` when the endpoint is not one (or is down)."""
    answer = _ollama(base_url, "/api/version")
    version = answer.get("version") if isinstance(answer, dict) else None
    return str(version) if version else None


def ollama_model_digest(base_url: str | None, model: str) -> str | None:
    """The digest of ``model`` in the local Ollama's model list, or ``None`` when it is not there.

    It is the sha256 of the model's manifest; ``ollama list`` shows its first 12 characters as the
    model's id. A tag such as ``qwen3:8b`` can be pulled again and point at other weights, so the
    record of a run keeps the digest.
    """
    answer = _ollama(base_url, "/api/tags")
    models = answer.get("models") if isinstance(answer, dict) else None
    for item in models if isinstance(models, list) else []:
        if isinstance(item, dict) and model in (item.get("name"), item.get("model")):
            digest = item.get("digest")
            return str(digest) if digest else None
    return None
