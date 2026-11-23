"""The labelled check of first rewrites: how often what the gate saw was wrong, by a blind label.

The gate-off rate counts the first rewrites the verifier flagged, so it is only as accurate as the
verifier. This check labels a sample of those first rewrites from the facts, without the
verifier's findings, and measures both the violation rate and the verifier against the labels.

**The frame** is every first rewrite that reached the verifier, in the completed runs: a bullet the
model rewrote, whose first answer cited facts of its entry (a structural answer never reached the
verifier, and a verbatim bullet was never rewritten). The run is the unit: a first rewrite that two
runs share is in the frame once per run.

**The sample.** :func:`draw_first_sample` draws a seeded random sample of the frame and writes it
exactly as the residual sample is written: the bullet, the facts it cites, the entry's scope note,
its DO NOT CLAIM list and its tech list, under opaque ids. A separate mapping keeps what each id
was.

**The scores.** :func:`score_first` reads the labels. The violation rate is the share of labelled
violations in the sample, with a Wilson interval (95%). The verifier's findings on each sampled
first rewrite, as the run recorded them, are scored against the labels in two ways: as the
gate-off flag (any finding, whatever its class, is a refusal), and per class, as the verifier is
scored on its own dataset, together with whether a flagged row's labelled class is among its
findings.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from tailor.bank import Bank
from tailor.evals.residual import (
    CONFIDENCE_PERCENT,
    ResidualError,
    check_calls,
    check_labels,
    wilson,
)
from tailor.evals.verifier import CLASSES, score
from tailor.judge import input_for


def first_frame(document: Mapping[str, Any]) -> list[tuple[int, int]]:
    """The ``(run index, bullet index)`` of each first rewrite that reached the verifier."""
    return [
        (r, b)
        for r, run in enumerate(document["runs"])
        if run["status"] == "ok"
        for b, bullet in enumerate(run["bullets"])
        if bullet["outcome"] != "verbatim" and not bullet["first_structural"]
    ]


def draw_first_sample(
    document: Mapping[str, Any], bank: Bank, *, seed: int, size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the blind sample of first rewrites.

    Args:
        document: The results.
        bank: The bank the runs used.
        seed: The seed of the draw; recorded in the mapping.
        size: How many first rewrites to draw (all of the frame when it is smaller).

    Returns:
        The rows to show a labeller (``id``, ``bullet``, ``facts``, ``scope``, ``do_not_claim``,
        ``tech_stack``), in the order drawn, and the mapping from each id to its run and bullet.
    """
    frame = first_frame(document)
    drawn = random.Random(seed).sample(frame, min(size, len(frame)))
    rows: list[dict[str, Any]] = []
    mapping_rows: dict[str, Any] = {}
    for number, (r, b) in enumerate(drawn, start=1):
        run = document["runs"][r]
        bullet = run["bullets"][b]
        shown = input_for(bank, bullet["first_text"], bullet["entry"], bullet["first_sources"])
        row_id = f"f-{number:03d}"
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
        }
    completed = [run for run in document["runs"] if run["status"] == "ok"]
    rewritten = [b for run in completed for b in run["bullets"] if b["outcome"] != "verbatim"]
    mapping = {
        "seed": seed,
        "size": size,
        "frame": len(frame),
        "rewritten": len(rewritten),
        "structural": sum(b["first_structural"] for b in rewritten),
        "sampled": len(drawn),
        "rows": mapping_rows,
    }
    return rows, mapping


def identical_rows(
    first_rows: Sequence[Mapping[str, Any]], passed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """The first-rewrite rows whose bullet and cited facts are those of a passed-sample row.

    Returns:
        Each such first-rewrite id with the ids of the passed-sample rows it repeats.
    """
    by_content: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for passed in passed_rows:
        by_content.setdefault((passed["bullet"], tuple(passed["facts"])), []).append(passed["id"])
    found: dict[str, list[str]] = {}
    for first in first_rows:
        repeated = by_content.get((first["bullet"], tuple(first["facts"])))
        if repeated:
            found[first["id"]] = repeated
    return found


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percent(numerator: int, denominator: int) -> float | None:
    """The share as a percent, unrounded: what a displayed percentage is rounded from."""
    return numerator * 100 / denominator if denominator else None


def _findings(document: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, list[str]]:
    """The verifier's finding kinds on each sampled first rewrite, as its run recorded them.

    A refusal for a copied fact id is not a verifier finding and is left out.
    """
    runs = {(run["jd"], run["seed"]): run for run in document["runs"] if run["status"] == "ok"}
    found: dict[str, list[str]] = {}
    for row_id, info in mapping["rows"].items():
        run = runs.get((info["jd"], info["seed"]))
        bullets = run["bullets"] if run is not None else []
        index = info["index"]
        if not 0 <= index < len(bullets) or bullets[index]["entry"] != info["entry"]:
            raise ResidualError(
                f"{row_id}: the mapping points at no bullet of {info['entry']} in the results"
            )
        found[row_id] = [k for k in bullets[index]["first_kinds"] if k in CLASSES]
    return found


def _rate(ids: Sequence[str], violating: set[str]) -> dict[str, Any]:
    low, high = wilson(len(violating), len(ids))
    return {
        "violating": len(violating),
        "rows": len(ids),
        "rate": _ratio(len(violating), len(ids)),
        "interval": [round(low, 4), round(high, 4)],
        "percent": _percent(len(violating), len(ids)),
        "interval_percent": [low * 100, high * 100],
    }


def _flag(ids: Sequence[str], violating: set[str], flagged: set[str]) -> dict[str, Any]:
    """The gate-off flag (any finding) as a yes or no against the labels, over ``ids``."""
    flagged, violating = flagged & set(ids), violating & set(ids)
    hits, wrong = len(flagged & violating), len(flagged - violating)
    ok_rows = len(ids) - len(violating)
    return {
        "flagged": len(flagged),
        "flagged_violating": hits,
        "precision": _ratio(hits, len(flagged)),
        "violating": len(violating),
        "recall": _ratio(hits, len(violating)),
        "unflagged": len(ids) - len(flagged),
        "unflagged_violating": len(violating - flagged),
        "ok": ok_rows,
        "false_alarms": wrong,
        "false_alarm_rate": _ratio(wrong, ok_rows),
        "precision_percent": _percent(hits, len(flagged)),
        "recall_percent": _percent(hits, len(violating)),
        "false_alarm_percent": _percent(wrong, ok_rows),
    }


def _overlap(
    repeats: Mapping[str, Sequence[str]],
    labels: Mapping[str, Sequence[str]],
    passed_labels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    same = [
        i
        for i, others in repeats.items()
        if all(set(passed_labels.get(o, ())) == set(labels[i]) for o in others)
    ]
    return {
        "rows": len(repeats),
        "same_labels": len(same),
        "ok": sum("ok" in labels[i] for i in repeats),
        "violating": sum("ok" not in labels[i] for i in repeats),
        "pairs": {i: list(others) for i, others in sorted(repeats.items())},
    }


def score_first(
    document: Mapping[str, Any],
    mapping: Mapping[str, Any],
    labels: Mapping[str, Sequence[str]],
    *,
    repeats: Mapping[str, Sequence[str]] | None = None,
    passed_labels: Mapping[str, Sequence[str]] | None = None,
    calls: Sequence[str] = (),
) -> dict[str, Any]:
    """The labelled violation rate of first rewrites, and the verifier against the labels.

    Args:
        document: The results.
        mapping: The sample's mapping.
        labels: The labels of the sample.
        repeats: The sampled rows that repeat rows of the passed-bullet sample
            (:func:`identical_rows`). A labeller who labelled both could know the verifier
            passed those, so the scores are also given without them, and their labels are
            compared with the ones the same bullets got there (``passed_labels``).
        passed_labels: The labels of the passed-bullet sample.
        calls: The rows listed as judgement calls; their number is recorded.

    Raises:
        ResidualError: The labels do not cover exactly the sampled rows, a label is unknown, a
            row of the mapping points at no bullet of the results, or a judgement call or a
            repeated row names no sampled row.
    """
    check_labels(mapping, labels)
    check_calls(mapping, calls)
    repeated = dict(repeats or {})
    for row_id in repeated:
        if row_id not in mapping["rows"]:
            raise ResidualError(f"{row_id!r} repeats a passed row but is not in the sample")
    kinds = _findings(document, mapping)
    ids = sorted(mapping["rows"])
    violating = {i for i in ids if "ok" not in labels[i]}
    flagged = {i for i in ids if kinds[i]}
    class_right = {i for i in flagged if set(kinds[i]) & set(labels[i])}
    verifier = score((labels[i], kinds[i]) for i in ids).to_dict()
    rest = [i for i in ids if i not in repeated]
    return {
        "confidence_percent": CONFIDENCE_PERCENT,
        "seed": mapping["seed"],
        "frame": mapping["frame"],
        "sampled": len(ids),
        "labelled": {
            "rows": len(ids),
            "violating": len(violating),
            "by_class": {c: sum(c in labels[i] for i in ids) for c in CLASSES},
        },
        "judgement_calls": len(calls),
        "violation_rate": _rate(ids, violating),
        "gate_off_flag": _flag(ids, violating, flagged),
        "verifier": {
            **verifier,
            "class_right": {
                "rows": len(class_right),
                "precision": _ratio(len(class_right), len(flagged)),
                "recall": _ratio(len(class_right), len(violating)),
                "precision_percent": _percent(len(class_right), len(flagged)),
                "recall_percent": _percent(len(class_right), len(violating)),
            },
        },
        "overlap_with_passed_sample": _overlap(repeated, labels, passed_labels or {}),
        "without_overlap": {
            **_rate(rest, violating & set(rest)),
            "gate_off_flag": _flag(rest, violating, flagged),
        },
    }
