"""Choose which entries and which bullets go on the resume.

The bank has already narrowed the field (``Bank.candidates()``: every work entry, and projects with
code). One model call scores each candidate's coverage of the posting's responsibilities and says
which responsibilities it covers and in what order its bullets matter. Everything after that is
deterministic: clamping, tie-breaks, the choice of entries and the number of bullets kept.

Choice, separately for work entries and projects:

* up to three entries, and at least two when the bank has them;
* each step takes the entry that covers the most responsibilities not yet covered, then the higher
  score, then the smaller entry id; the third entry also needs a score of ``OPTIONAL_MIN_SCORE``;
* an entry of the same project as one already chosen is skipped (``Same project as`` in its scope).

An entry keeps ``2 + (score >= 5) + (score >= 8)`` bullets (fewer if it has fewer), first the ones
the model put first, then the rest in bank order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from tailor.bank import Bank, Entry, ProjectEntry, WorkEntry, normalize_latex
from tailor.jd.extract import JobDescription
from tailor.llm import LLM, LLMError, Message

MAX_PER_KIND: Final = 3
MIN_PER_KIND: Final = 2
OPTIONAL_MIN_SCORE: Final = 4
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "score": {"type": "integer"},
                    "covers": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                    "bullet_priority": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["entry", "score", "covers", "reason", "bullet_priority"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}
SYSTEM: Final = """\
You help choose which experiences go on a one-page resume for a job posting. You get the posting's \
numbered responsibilities and a list of candidate experiences with their numbered bullets. For \
every candidate, return:
- entry: its id, exactly as given;
- score: an integer from 0 (nothing relevant) to 10 (covers most responsibilities directly);
- covers: the numbers of the responsibilities its bullets show evidence for (empty if none);
- reason: one short sentence;
- bullet_priority: its bullet numbers, most useful for this posting first.
Judge only what the bullets say; do not assume anything they do not state."""


@dataclass(frozen=True, slots=True)
class Score:
    """The model's judgement of one candidate, cleaned up."""

    entry: str
    score: int
    covers: frozenset[int]
    reason: str
    bullet_priority: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Pick:
    """A chosen entry and the bullets kept, as 0-based indexes in priority order."""

    entry: Entry
    score: Score
    bullets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Selection:
    """Every candidate's score (bank order) and the picks (work entries first)."""

    scores: tuple[Score, ...]
    picks: tuple[Pick, ...]


def coverage_targets(jd: JobDescription) -> tuple[str, ...]:
    """The numbered list the scores refer to: responsibilities, else requirements."""
    return jd.responsibilities or jd.requirements


def _describe(entry: Entry) -> str:
    where = entry.company if isinstance(entry, WorkEntry) else ""
    stack = entry.tech if isinstance(entry, WorkEntry) else entry.tech_stack
    lines = [
        f"id: {entry.id}",
        f"title: {normalize_latex(entry.title)}" + (f" at {where}" if where else ""),
        f"tech: {normalize_latex(stack)}",
        f"tags: {', '.join(entry.tags)}",
    ]
    lines += [f"bullet {n}: {normalize_latex(b)}" for n, b in enumerate(entry.bullets, start=1)]
    return "\n".join(lines)


def build_messages(jd: JobDescription, candidates: list[Entry]) -> list[Message]:
    """The scoring prompt."""
    targets = "\n".join(f"{n}. {t}" for n, t in enumerate(coverage_targets(jd), start=1))
    body = "\n\n".join(f"Candidate\n{_describe(e)}" for e in candidates)
    user = f"Posting: {jd.title} at {jd.company}\nResponsibilities:\n{targets}\n\n{body}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _number_list(item: dict[str, Any], key: str, entry_id: str) -> list[Any]:
    """``item[key]`` as a list (missing is empty); anything else is the model's mistake."""
    value = item.get(key, [])
    if not isinstance(value, list):
        raise LLMError(
            f"the model's score for {entry_id!r} has {key!r} = {value!r}; it must be a list"
        )
    return value


def normalize_scores(
    candidates: list[Entry], raw: dict[str, Any], target_count: int
) -> list[Score]:
    """One clean score per candidate, in bank order.

    The first answer for an id wins; ids that are not candidates are ignored; a candidate the model
    skipped scores 0. Scores are clamped to 0-10, ``covers`` to valid responsibility numbers, and
    ``bullet_priority`` to valid, distinct bullet numbers.

    Raises:
        LLMError: ``scores``, ``covers`` or ``bullet_priority`` is present but not a list.
    """
    answers: dict[str, dict[str, Any]] = {}
    items = raw.get("scores", [])
    if not isinstance(items, list):
        raise LLMError(f"the model's 'scores' must be a list, not {type(items).__name__}")
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("entry"), str):
            answers.setdefault(item["entry"], item)
    scores: list[Score] = []
    for entry in candidates:
        item = answers.get(entry.id)
        if item is None:
            scores.append(Score(entry.id, 0, frozenset(), "not scored by the model", ()))
            continue
        value = item.get("score")
        score = max(0, min(10, value)) if isinstance(value, int) else 0
        covers = frozenset(
            n
            for n in _number_list(item, "covers", entry.id)
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= target_count
        )
        priority: list[int] = []
        for n in _number_list(item, "bullet_priority", entry.id):
            if isinstance(n, int) and 1 <= n <= len(entry.bullets) and n not in priority:
                priority.append(n)
        scores.append(
            Score(entry.id, score, covers, str(item.get("reason", "")).strip(), tuple(priority))
        )
    return scores


def bullets_to_keep(entry: Entry, score: Score) -> tuple[int, ...]:
    """The 0-based bullet indexes kept for a picked entry, in priority order."""
    keep = min(len(entry.bullets), 2 + (score.score >= 5) + (score.score >= 8))
    order = [n - 1 for n in score.bullet_priority]
    order += [i for i in range(len(entry.bullets)) if i not in order]
    return tuple(order[:keep])


def _choose_kind(
    pool: list[tuple[Entry, Score]], blocked: set[str], groups: list[frozenset[str]]
) -> list[tuple[Entry, Score]]:
    chosen: list[tuple[Entry, Score]] = []
    covered: set[int] = set()
    remaining = list(pool)
    while len(chosen) < MAX_PER_KIND:
        eligible = [
            (e, s)
            for e, s in remaining
            if e.id not in blocked and (len(chosen) < MIN_PER_KIND or s.score >= OPTIONAL_MIN_SCORE)
        ]
        if not eligible:
            break

        seen = frozenset(covered)

        def key(pair: tuple[Entry, Score], seen: frozenset[int] = seen) -> tuple[int, int, str]:
            entry, score = pair
            gain = len(score.covers - seen) if score.score > 0 else 0
            return (-gain, -score.score, entry.id)

        best = min(eligible, key=key)
        chosen.append(best)
        remaining.remove(best)
        if best[1].score > 0:
            covered |= best[1].covers
        for group in groups:
            if best[0].id in group:
                blocked |= group
    return chosen


def choose(bank: Bank, candidates: list[Entry], scores: list[Score]) -> list[Pick]:
    """Pick the entries and bullets from clean scores; no model involved."""
    by_id = {s.entry: s for s in scores}
    groups = bank.same_project_groups()
    blocked: set[str] = set()
    picks: list[Pick] = []
    for kind in (WorkEntry, ProjectEntry):
        pool = [(e, by_id[e.id]) for e in candidates if isinstance(e, kind)]
        for entry, score in _choose_kind(pool, blocked, groups):
            picks.append(Pick(entry, score, bullets_to_keep(entry, score)))
    return picks


def select(llm: LLM, bank: Bank, jd: JobDescription, *, model: str) -> Selection:
    """Score the candidates with one model call, then choose deterministically."""
    candidates = bank.candidates()
    raw = llm.complete_json(build_messages(jd, candidates), SCHEMA, model=model)
    scores = normalize_scores(candidates, raw, len(coverage_targets(jd)))
    return Selection(tuple(scores), tuple(choose(bank, candidates, scores)))
