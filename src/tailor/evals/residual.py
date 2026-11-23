"""The residual check: what the verifier lets through, found by the judge and by a blind labelling.

**The frame** is the set of bullets the model rewrote and the verifier passed (first time or after
a retry) in the completed runs. Those are the only bullets that can carry a violation the gate
missed: a fallback is the bank's own sentence, and a verbatim one was never rewritten.

**Judging.** :func:`judge_runs` runs the LLM judge on every final bullet of every completed run
(each distinct bullet once), including the ones outside the frame.

**The sample.** :func:`draw_sample` takes two strata of the frame. Every bullet the judge flagged
(or could not judge) goes in whole. From the rest, a seeded random draw of a fixed size goes in. The
two are shuffled together under opaque ids and written without the bullet's outcome, the judge's
verdict, or any verifier finding, so the person labelling sees the bullet and its facts only. A
separate mapping keeps what each id was.

**The rate.** :func:`score_residual` reads the labels. The violating bullets among the flagged are
counted as they are. The share of violating bullets in the random draw is extrapolated to the rest
of the frame with a Wilson interval (95%). The residual estimate is the sum, and the rate is that
over the frame; its complement is the share of the frame that is faithful. Each rate is rounded to
4 places and also given as an unrounded percent, which is what a displayed figure is rounded
from. The interval is conservative in one respect (no finite-population correction) and
treats the flagged stratum as exact.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from tailor.bank import Bank
from tailor.evals.verifier import CLASSES, LABELS
from tailor.judge import input_for, judge_bullet
from tailor.llm import LLM, LLMError

_PASSED: Final = ("first_pass", "after_retry")
_Z: Final = 1.96
CONFIDENCE_PERCENT: Final = 95
"""The level of every interval here: a Wilson interval with z = 1.96."""


_CALL: Final = re.compile(r"^- ([\w-]+) \(", re.MULTILINE)


class ResidualError(Exception):
    """The sample cannot be drawn or scored: bad labels, or labels that do not match the sample."""


def wilson(successes: int, trials: int, z: float = _Z) -> tuple[float, float]:
    """The Wilson score interval for a proportion (95% by default)."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, round(centre - margin, 12)), min(1.0, round(centre + margin, 12)))


def frame_of(document: Mapping[str, Any]) -> list[tuple[int, int]]:
    """The ``(run index, bullet index)`` of each bullet in the frame, in order."""
    return [
        (r, b)
        for r, run in enumerate(document["runs"])
        if run["status"] == "ok"
        for b, bullet in enumerate(run["bullets"])
        if bullet["outcome"] in _PASSED
    ]


def judge_runs(
    llm: LLM,
    bank: Bank,
    document: Mapping[str, Any],
    *,
    model: str,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the judge on every final bullet of every completed run.

    A bullet that appears in several runs (same entry, text and sources) is judged once and the
    verdict is reused.

    Returns:
        One record per bullet: ``{jd, seed, index, entry, outcome, judge, reason}``, where
        ``judge`` is the flagged classes, or ``None`` with an ``error`` when the call failed.
    """
    cache: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for run in document["runs"]:
        if run["status"] != "ok":
            continue
        for index, bullet in enumerate(run["bullets"]):
            key = (bullet["entry"], bullet["final_text"], tuple(bullet["final_sources"]))
            if key not in cache:
                shown = input_for(bank, bullet["final_text"], bullet["entry"], key[2])
                try:
                    verdict = judge_bullet(llm, shown, model=model)
                    cache[key] = {"judge": list(verdict.classes), "reason": verdict.reason}
                except LLMError as error:
                    cache[key] = {"judge": None, "reason": None, "error": str(error)}
            record = {
                "jd": run["jd"],
                "seed": run["seed"],
                "index": index,
                "entry": bullet["entry"],
                "outcome": bullet["outcome"],
                **cache[key],
            }
            records.append(record)
            if on_item is not None:
                on_item(record)
    return records


def judged_positions(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where each record of :func:`judge_runs` comes from, in its order: every final bullet."""
    return [
        {
            "jd": run["jd"],
            "seed": run["seed"],
            "index": index,
            "entry": bullet["entry"],
            "outcome": bullet["outcome"],
        }
        for run in document["runs"]
        if run["status"] == "ok"
        for index, bullet in enumerate(run["bullets"])
    ]


def judged_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many bullets were judged, flagged and failed (a call error) in ``records``.

    The flagged ones are also counted by class (a bullet can carry two) and by the outcome of the
    bullet: a passed one is a candidate residual violation, a fallback is the bank's sentence.
    """
    flagged = [r for r in records if r["judge"]]
    return {
        "bullets": len(records),
        "flagged": len(flagged),
        "errors": sum(r["judge"] is None for r in records),
        "flagged_by_class": {c: sum(c in r["judge"] for r in flagged) for c in CLASSES},
        "flagged_by_outcome": {
            outcome: sum(r["outcome"] == outcome for r in flagged)
            for outcome in (*_PASSED, "fallback")
        },
    }


def draw_sample(
    document: Mapping[str, Any],
    judged: Sequence[Mapping[str, Any]],
    bank: Bank,
    *,
    seed: int,
    size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the blind sample.

    Args:
        document: The results.
        judged: The records of :func:`judge_runs` for it.
        bank: The bank the runs used.
        seed: The seed of the draw and of the shuffle; recorded in the mapping.
        size: How many bullets to draw at random from the frame's unflagged bullets (all of them
            when there are fewer).

    Returns:
        The rows to show a labeller (``id``, ``bullet``, ``facts``, ``scope``, ``do_not_claim``,
        ``tech_stack``), and the mapping from each id to its run, bullet and stratum.
    """
    verdicts = {(j["jd"], j["seed"], j["index"]): j for j in judged}
    frame = frame_of(document)
    flagged: list[tuple[int, int]] = []
    unjudged: list[tuple[int, int]] = []
    rest: list[tuple[int, int]] = []
    for r, b in frame:
        run = document["runs"][r]
        verdict = verdicts.get((run["jd"], run["seed"], b))
        if verdict is None or verdict["judge"] is None:
            unjudged.append((r, b))
        elif verdict["judge"]:
            flagged.append((r, b))
        else:
            rest.append((r, b))
    rng = random.Random(seed)
    drawn = rng.sample(rest, min(size, len(rest)))
    chosen = [(item, "flagged") for item in flagged]
    chosen += [(item, "unjudged") for item in unjudged]
    chosen += [(item, "random") for item in drawn]
    rng.shuffle(chosen)
    rows: list[dict[str, Any]] = []
    mapping_rows: dict[str, Any] = {}
    for number, ((r, b), stratum) in enumerate(chosen, start=1):
        run = document["runs"][r]
        bullet = run["bullets"][b]
        shown = input_for(bank, bullet["final_text"], bullet["entry"], bullet["final_sources"])
        row_id = f"s-{number:03d}"
        rows.append(
            {
                "id": row_id,
                "bullet": shown.bullet,
                "facts": list(shown.facts),
                "scope": list(shown.scope),
                "do_not_claim": list(shown.claims),
                "tech_stack": shown.tech,
            }
        )
        mapping_rows[row_id] = {
            "jd": run["jd"],
            "seed": run["seed"],
            "index": b,
            "entry": bullet["entry"],
            "stratum": stratum,
        }
    final_bullets = sum(len(run["bullets"]) for run in document["runs"] if run["status"] == "ok")
    mapping = {
        "seed": seed,
        "size": size,
        "frame": len(frame),
        "final_bullets": final_bullets,
        "strata": {
            "flagged": len(flagged),
            "unjudged": len(unjudged),
            "unflagged": len(rest),
            "sampled": len(drawn),
        },
        "rows": mapping_rows,
    }
    return rows, mapping


def check_labels(mapping: Mapping[str, Any], labels: Mapping[str, Sequence[str]]) -> None:
    """Check that the labels cover exactly the sampled rows, with known labels.

    Raises:
        ResidualError: A sampled row has no label, a label names no sampled row or is unknown.
    """
    for row_id in labels:
        if row_id not in mapping["rows"]:
            raise ResidualError(f"label for {row_id!r}, which is not in the sample")
    for row_id in mapping["rows"]:
        if row_id not in labels:
            raise ResidualError(f"no label for {row_id!r}")
    for row_id, given in labels.items():
        if not given:
            raise ResidualError(f"{row_id}: empty labels")
        for label in given:
            if label not in LABELS:
                raise ResidualError(f"{row_id}: unknown label {label!r}")
        if "ok" in given and len(given) > 1:
            raise ResidualError(f"{row_id}: ok cannot be combined with another label")


def judgement_call_ids(text: str) -> list[str]:
    """The row ids a judgement-calls file lists, one per ``- <id> (<labels>): ...`` line."""
    return _CALL.findall(text)


def check_calls(mapping: Mapping[str, Any], calls: Sequence[str]) -> None:
    """Check that every judgement call names a sampled row, once.

    Raises:
        ResidualError: A call names a row that is not in the sample, or a row twice.
    """
    for row_id in calls:
        if row_id not in mapping["rows"]:
            raise ResidualError(f"a judgement call names {row_id!r}, which is not in the sample")
    if len(set(calls)) != len(calls):
        raise ResidualError("a judgement-calls file names a row twice")


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percent(numerator: float, denominator: float) -> float | None:
    """The share as a percent, unrounded: what a displayed percentage is rounded from."""
    return numerator * 100 / denominator if denominator else None


def score_residual(
    mapping: Mapping[str, Any],
    labels: Mapping[str, Sequence[str]],
    *,
    calls: Sequence[str] = (),
) -> dict[str, Any]:
    """The residual violation rate from the blind labels.

    ``calls`` are the rows the labeller listed as judgement calls; their number is recorded.

    Raises:
        ResidualError: A sampled row has no label, a label names no sampled row or is unknown, or
            a judgement call names no sampled row or one twice.
    """
    check_labels(mapping, labels)
    check_calls(mapping, calls)
    info = mapping["rows"]
    strata = mapping["strata"]
    violating = {i for i, given in labels.items() if "ok" not in given}
    by_class = {c: sum(c in labels[i] for i in labels) for c in CLASSES}
    random_ids = [i for i in info if info[i]["stratum"] == "random"]
    flagged_ids = [i for i in info if info[i]["stratum"] == "flagged"]
    census_ids = [i for i in info if info[i]["stratum"] != "random"]
    drawn = len(random_ids)
    random_violations = sum(i in violating for i in random_ids)
    confirmed = sum(i in violating for i in census_ids)
    confirmed_flagged = sum(i in violating for i in flagged_ids)
    if drawn == 0 and strata["unflagged"]:
        raise ResidualError(
            f"no random draw from the {strata['unflagged']} unflagged bullets: "
            "nothing to extrapolate from"
        )
    low, high = wilson(random_violations, drawn)
    if drawn == strata["unflagged"]:  # the whole stratum was labelled: nothing is extrapolated
        rate_random = random_violations / drawn if drawn else 0.0
        low = high = rate_random
    else:
        rate_random = random_violations / drawn
    rest = strata["unflagged"]
    frame = mapping["frame"]
    extrapolated = rest * rate_random
    estimate = confirmed + extrapolated
    interval = (confirmed + rest * low, confirmed + rest * high)
    total_final = mapping["final_bullets"]
    share = _percent(estimate, frame)
    share_low, share_high = _percent(interval[0], frame), _percent(interval[1], frame)
    return {
        "confidence_percent": CONFIDENCE_PERCENT,
        "labelled": {
            "rows": len(labels),
            "violating": len(violating),
            "by_class": by_class,
        },
        "judgement_calls": len(calls),
        "random": {
            "sampled": drawn,
            "violations": random_violations,
            "rate": round(rate_random, 4),
            "interval": [round(low, 4), round(high, 4)],
            "percent": rate_random * 100,
            "interval_percent": [low * 100, high * 100],
        },
        "census": {
            "flagged": strata["flagged"],
            "unjudged": strata["unjudged"],
            "confirmed": confirmed,
        },
        "residual": {
            "frame": frame,
            "extrapolated": round(extrapolated, 2),
            "estimate": round(estimate, 2),
            "estimate_interval": [round(interval[0], 2), round(interval[1], 2)],
            "rate": _ratio(estimate, frame),
            "rate_interval": [_ratio(interval[0], frame), _ratio(interval[1], frame)],
            "percent": share,
            "interval_percent": [share_low, share_high],
            "faithful_percent": None if share is None else 100 - share,
            "faithful_interval_percent": [
                None if share_high is None else 100 - share_high,
                None if share_low is None else 100 - share_low,
            ],
            "final_bullets": total_final,
            "rate_of_final_bullets": _ratio(estimate, total_final),
            "percent_of_final_bullets": _percent(estimate, total_final),
        },
        "judge_on_pipeline": {
            "flagged": strata["flagged"],
            "confirmed_of_flagged": confirmed_flagged,
            "precision": _ratio(confirmed_flagged, strata["flagged"]),
            "recall_estimate": _ratio(confirmed_flagged, estimate),
            "precision_percent": _percent(confirmed_flagged, strata["flagged"]),
            "recall_estimate_percent": _percent(confirmed_flagged, estimate),
        },
    }
