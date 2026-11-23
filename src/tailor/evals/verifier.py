"""Evaluate the verifier on the labelled dataset, with a fixed split and a regression gate.

The dataset is ``eval/verifier/labeled.jsonl``: rows of ``{id, entry, sources, bullet, labels}``,
where ``sources`` are the texts of the facts the bullet cites and ``labels`` say what is wrong with
it (``ok`` or any of the violation classes). The evaluation runs the deterministic verifier on
every row and compares its findings with the labels.

**The split.** 30% of the rows were held out once, chosen by a seeded hash of the row id, and the
choice is recorded in ``split.json``: a row's side never changes after that, and nobody picks it.
A row added later goes to *dev*, so the holdout never grows or moves and a rule tuned on new rows
cannot have seen it. The rules of the verifier are tuned on the *dev* rows only. The *holdout* is
evaluated once at the end, and the numbers that are reported come from it.

**The metrics.** Per class: precision (of the rows the verifier put in the class, how many are)
and recall (of the rows that are, how many it put in the class). Rows may carry several labels and
get several findings, and each class is scored on its own. Over all violation classes together:
the *caught-violation recall*, the share of violating rows with at least one finding of any kind,
which is what a gate that rejects on any finding cares about. For the other side: the
*false-alarm rate*, the share of ``ok`` rows with a finding.

**The gate.** :func:`check_gate` compares a fresh run with the committed ``results.json`` and
fails if a holdout metric fell by more than the tolerance (0.02), or the false-alarm rate rose by
more than it. A change that makes the verifier worse therefore fails CI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

from tailor.bank import Bank
from tailor.verify.findings import KINDS
from tailor.verify.rounding import Rounding
from tailor.verify.verifier import entry_context, verify_text

Split = Literal["dev", "holdout"]
CLASSES: Final[tuple[str, ...]] = KINDS
LABELS: Final[tuple[str, ...]] = ("ok", *KINDS)
SPLIT_SEED: Final = "jd-resume-tailor/verifier-eval/v1"
HOLDOUT_FRACTION: Final = 0.30
DEFAULT_TOLERANCE: Final = 0.02
_ROW_FIELDS: Final = frozenset({"id", "entry", "sources", "bullet", "labels"})
_DIGITS: Final = 4
_SPLITS: Final[tuple[str, ...]] = get_args(Split)


class EvalError(Exception):
    """The evaluation cannot run: bad data, a stale split, or an unreadable file."""


@dataclass(frozen=True, slots=True)
class Row:
    """One labelled example.

    Attributes:
        id: Unique and stable; ``split.json`` lists it on one side. Rows adapted from overclaim
            patterns found in real resume bullets have ids that start ``pattern-``, the rest
            ``synth-``.
        entry: The bank entry the bullet is about (its tech stack and DO NOT CLAIM list apply).
        sources: The texts of the facts the bullet cites.
        bullet: The bullet to verify.
        labels: ``("ok",)`` or one or more violation classes.
    """

    id: str
    entry: str
    sources: tuple[str, ...]
    bullet: str
    labels: tuple[str, ...]


# ---- rows ------------------------------------------------------------------------------------


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _row(raw: Any) -> Row:
    if not isinstance(raw, dict):
        raise ValueError("a row must be a JSON object")
    for key in raw:
        if key not in _ROW_FIELDS:
            raise ValueError(f"unknown field {key!r}")
    for key in sorted(_ROW_FIELDS):
        if key not in raw:
            raise ValueError(f"missing field {key!r}")
    sources, labels = raw["sources"], raw["labels"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must have at least one source")
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must have at least one label")
    for label in labels:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
    if len(set(labels)) != len(labels):
        raise ValueError("labels repeats a label")
    if "ok" in labels and len(labels) > 1:
        raise ValueError("ok cannot be combined with another label")
    return Row(
        id=_text(raw["id"], "id"),
        entry=_text(raw["entry"], "entry"),
        sources=tuple(_text(source, "a source") for source in sources),
        bullet=_text(raw["bullet"], "bullet"),
        labels=tuple(labels),
    )


def load_rows(path: Path) -> list[Row]:
    """Read the dataset: one JSON object per line, blank lines ignored.

    Raises:
        EvalError: The file cannot be read, a line is not a valid row, or an id repeats.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvalError(f"cannot read {path}: {error.strerror}") from error
    rows: list[Row] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _row(json.loads(line))
        except json.JSONDecodeError as error:
            raise EvalError(f"{path}, line {number}: not valid JSON: {error}") from error
        except ValueError as error:
            raise EvalError(f"{path}, line {number}: {error}") from error
        if row.id in seen:
            raise EvalError(f"{path}, line {number}: duplicate id {row.id!r}")
        seen.add(row.id)
        rows.append(row)
    return rows


def validate_rows(rows: Sequence[Row], bank: Bank) -> list[str]:
    """Check that every row's entry exists and every source is a fact text of that entry.

    A dataset whose sources drift away from the bank would measure the verifier on facts the bank
    no longer states, so the evaluation refuses to run on one.

    Returns:
        One message per problem; empty when the dataset is consistent with the bank.
    """
    problems: list[str] = []
    for row in rows:
        try:
            facts = {fact.text for fact in bank.facts_for(row.entry)}
        except KeyError:
            problems.append(f"{row.id}: unknown entry {row.entry!r}")
            continue
        problems.extend(
            f"{row.id}: source {number} is not a fact of {row.entry}: {source!r}"
            for number, source in enumerate(row.sources, start=1)
            if source not in facts
        )
    return problems


# ---- the split -------------------------------------------------------------------------------


def assign_split(row_id: str, seed: str = SPLIT_SEED) -> Split:
    """Return the side of the split a row falls on: a function of the seed and the id only."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "holdout" if value < HOLDOUT_FRACTION else "dev"


def build_split(
    rows: Iterable[Row],
    seed: str = SPLIT_SEED,
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the contents of ``split.json`` for ``rows``.

    Args:
        rows: Every row of the dataset.
        seed: The seed of the first draw; used only when there is no ``previous`` split.
        previous: The recorded split, if there is one. A row it lists keeps its side; a row it
            does not list is new and goes to dev, so the holdout never changes.
    """
    sides: dict[str, list[str]] = {"dev": [], "holdout": []}
    held_out = set(previous.get("holdout", ())) if previous is not None else set()
    known = (set(previous.get("dev", ())) | held_out) if previous is not None else set()
    for row in rows:
        if previous is None:
            side = assign_split(row.id, seed)
        elif row.id in known:
            side = "holdout" if row.id in held_out else "dev"
        else:
            side = "dev"
        sides[side].append(row.id)
    return {
        "seed": seed,
        "holdout_fraction": HOLDOUT_FRACTION,
        "method": "the holdout was drawn once: a row was held out when the first 8 bytes of "
        "sha256(seed + ':' + id), read as a fraction of 2^64, were below holdout_fraction. "
        "Membership is recorded below and does not change; a row added later goes to dev.",
        "dev": sorted(sides["dev"]),
        "holdout": sorted(sides["holdout"]),
    }


def read_split(path: Path) -> dict[str, Any]:
    """Read ``split.json`` as it is, without checking it against rows.

    Raises:
        EvalError: The file is missing, unreadable or not a JSON object.
    """
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvalError(f"cannot read {path}: {error.strerror}") from error
    except json.JSONDecodeError as error:
        raise EvalError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(recorded, dict):
        raise EvalError(f"{path} must be a JSON object")
    return recorded


def load_split(path: Path, rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Read ``split.json``, check it against the rows, and return the rows of each side.

    Raises:
        EvalError: The file is missing or unreadable, or it does not list exactly the ids of the
            rows, each on one side (rows were added or removed without rewriting it).
    """
    recorded = read_split(path)
    ids = {row.id for row in rows}
    lists: dict[str, list[str]] = {}
    for side in _SPLITS:
        listed = recorded.get(side)
        if isinstance(listed, list) and all(isinstance(item, str) for item in listed):
            lists[side] = listed
    valid = (
        len(lists) == len(_SPLITS)
        and not set(lists["dev"]) & set(lists["holdout"])
        and set(lists["dev"]) | set(lists["holdout"]) == ids
        and all(listed == sorted(set(listed)) for listed in lists.values())
    )
    if not valid:
        raise EvalError(
            f"{path} does not match the dataset: the rows changed; "
            "rewrite it with `tailor eval verifier --write-split`"
        )
    by_id = {row.id: row for row in rows}
    return {side: [by_id[row_id] for row_id in lists[side]] for side in _SPLITS}


# ---- scores ----------------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, _DIGITS) if denominator else None


def _percent(numerator: int, denominator: int) -> float | None:
    return numerator * 100 / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class ClassScore:
    """Precision and recall of one class.

    Attributes:
        support: Rows labelled with the class.
        predicted: Rows the verifier put in the class.
        true_positives: Rows both.
    """

    support: int
    predicted: int
    true_positives: int

    @property
    def precision(self) -> float | None:
        """Of the rows predicted, the share that are labelled; ``None`` if none were predicted."""
        return _ratio(self.true_positives, self.predicted)

    @property
    def recall(self) -> float | None:
        """Of the rows labelled, the share predicted; ``None`` if none are labelled."""
        return _ratio(self.true_positives, self.support)

    def to_dict(self) -> dict[str, Any]:
        """The score as it is written to ``results.json``."""
        return {
            "support": self.support,
            "predicted": self.predicted,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class Scores:
    """The scores of a set of rows.

    Attributes:
        rows: How many rows were scored.
        classes: The score of each violation class.
        ok: The score of the ``ok`` verdict: precision is the share of rows the verifier passed
            that are labelled ok, recall the share of ok rows it passed.
        violation_rows: Rows labelled with at least one violation.
        caught: Of those, rows with at least one finding of any kind.
        ok_rows: Rows labelled ok.
        false_alarms: Of those, rows with at least one finding.
    """

    rows: int
    classes: dict[str, ClassScore]
    ok: ClassScore
    violation_rows: int
    caught: int
    ok_rows: int
    false_alarms: int

    @property
    def caught_violation_recall(self) -> float | None:
        """The share of violating rows that got any finding."""
        return _ratio(self.caught, self.violation_rows)

    @property
    def false_alarm_rate(self) -> float | None:
        """The share of ok rows that got any finding."""
        return _ratio(self.false_alarms, self.ok_rows)

    def to_dict(self) -> dict[str, Any]:
        """The scores as they are written to ``results.json``.

        The overall rates come with their counts and with an unrounded percent, which is what a
        displayed percentage is rounded from.
        """
        return {
            "rows": self.rows,
            "violation_rows": self.violation_rows,
            "classes": {name: score_.to_dict() for name, score_ in self.classes.items()},
            "ok": self.ok.to_dict(),
            "caught_violation_recall": self.caught_violation_recall,
            "false_alarm_rate": self.false_alarm_rate,
            "caught": self.caught,
            "false_alarms": self.false_alarms,
            "caught_percent": _percent(self.caught, self.violation_rows),
            "false_alarm_percent": _percent(self.false_alarms, self.ok_rows),
        }


def score(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> Scores:
    """Score the verifier's findings against the labels.

    Args:
        pairs: One ``(labels, kinds)`` pair per row: the labels as in a row, and the kinds of the
            findings the verifier made (empty for a bullet it passed).

    Returns:
        The per-class, ok and overall scores.
    """
    support = dict.fromkeys(CLASSES, 0)
    predicted = dict.fromkeys(CLASSES, 0)
    hits = dict.fromkeys(CLASSES, 0)
    ok_support = ok_predicted = ok_hits = 0
    violation_rows = caught = ok_rows = false_alarms = 0
    total = 0
    for labels, kinds in pairs:
        total += 1
        truth = set(labels)
        found = set(kinds)
        for name in CLASSES:
            support[name] += name in truth
            predicted[name] += name in found
            hits[name] += name in truth and name in found
        is_ok = "ok" in truth
        ok_support += is_ok
        ok_predicted += not found
        ok_hits += is_ok and not found
        if is_ok:
            ok_rows += 1
            false_alarms += bool(found)
        else:
            violation_rows += 1
            caught += bool(found)
    return Scores(
        rows=total,
        classes={name: ClassScore(support[name], predicted[name], hits[name]) for name in CLASSES},
        ok=ClassScore(ok_support, ok_predicted, ok_hits),
        violation_rows=violation_rows,
        caught=caught,
        ok_rows=ok_rows,
        false_alarms=false_alarms,
    )


def evaluate(rows: Sequence[Row], bank: Bank, rounding: Rounding = "conservative") -> Scores:
    """Run the verifier on every row and score its findings against the labels."""
    pairs: list[tuple[Sequence[str], Sequence[str]]] = []
    for row in rows:
        scope_texts = {fact.text for fact in bank.facts_for(row.entry) if fact.kind == "scope"}
        findings = verify_text(
            row.bullet,
            [source for source in row.sources if source not in scope_texts],
            claims=bank.do_not_claim(row.entry),
            context=entry_context(bank.entry(row.entry)),
            rounding=rounding,
            scope=[source for source in row.sources if source in scope_texts],
        )
        pairs.append((row.labels, tuple(dict.fromkeys(f.kind for f in findings))))
    return score(pairs)


# ---- output ----------------------------------------------------------------------------------


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_table(name: str, scores: Scores) -> str:
    """Render the scores of one split as a text table."""
    lines = [
        f"{name} ({scores.rows} rows)",
        f"{'class':<22}{'support':>8}{'predicted':>10}{'precision':>11}{'recall':>8}",
    ]
    for label, entry in (*scores.classes.items(), ("ok", scores.ok)):
        lines.append(
            f"{label:<22}{entry.support:>8}{entry.predicted:>10}"
            f"{_cell(entry.precision):>11}{_cell(entry.recall):>8}"
        )
    lines.append(
        f"caught-violation recall {_cell(scores.caught_violation_recall)} "
        f"({scores.caught} of {scores.violation_rows} violating rows have a finding)"
    )
    lines.append(
        f"false-alarm rate {_cell(scores.false_alarm_rate)} "
        f"({scores.false_alarms} of {scores.ok_rows} ok rows have a finding)"
    )
    return "\n".join(lines)


def _metrics(document: dict[str, Any]) -> dict[str, tuple[float | None, bool]]:
    """Every gated metric of a scores dict: name to ``(value, higher_is_better)``."""
    metrics: dict[str, tuple[float | None, bool]] = {}
    for name, entry in (*document["classes"].items(), ("ok", document["ok"])):
        metrics[f"{name} precision"] = (entry["precision"], True)
        metrics[f"{name} recall"] = (entry["recall"], True)
    metrics["caught-violation recall"] = (document["caught_violation_recall"], True)
    metrics["false-alarm rate"] = (document["false_alarm_rate"], False)
    return metrics


def check_gate(fresh: Scores, committed: dict[str, Any], tolerance: float) -> list[str]:
    """Compare fresh holdout scores with the committed ones.

    Args:
        fresh: The scores of a new run on the holdout.
        committed: The ``holdout`` object of the committed ``results.json``.
        tolerance: How far a metric may fall (or the false-alarm rate rise) before it is a failure.

    Returns:
        One message per metric that got worse by more than the tolerance; empty if none did.
        Metrics the committed file has no value for are not gated.
    """
    failures: list[str] = []
    new = _metrics(fresh.to_dict())
    for name, (old_value, higher_is_better) in _metrics(committed).items():
        if old_value is None:
            continue
        new_value = new[name][0]
        if new_value is None:
            failures.append(f"{name} is undefined now (it was {old_value:.2f})")
            continue
        worse_by = (old_value - new_value) if higher_is_better else (new_value - old_value)
        if worse_by > tolerance + 1e-9:
            verb = "fell" if higher_is_better else "rose"
            failures.append(
                f"{name} {verb} from {old_value:.2f} to {new_value:.2f} (tolerance {tolerance:.2f})"
            )
    return failures
