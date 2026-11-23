"""Scripted plans and profiles for the render tests."""

from tailor.render.model import (
    EducationItem,
    PlanBullet,
    PlanEntry,
    Profile,
    ResumePlan,
)


def entry(
    entry_id: str,
    kind: str,
    score: int,
    bullets: list[str],
    *,
    title: str = "",
    tech: str = "Go, Redis",
    link: str = "",
) -> PlanEntry:
    """A plan entry with a heading made from its id and the given bullets."""
    assert kind in ("work", "project")
    return PlanEntry(
        id=entry_id,
        type="work" if kind == "work" else "project",
        title=title or f"Title of {entry_id}",
        company=f"Company of {entry_id}" if kind == "work" else "",
        location="Remote" if kind == "work" else "",
        date="Jan. 2022 -- Present",
        tech=tech,
        link=link,
        score=score,
        bullets=tuple(PlanBullet(text, (f"{entry_id}#{n}",)) for n, text in enumerate(bullets, 1)),
    )


def profile() -> Profile:
    """A profile with every section filled."""
    return Profile(
        name="Sam Example",
        contacts=(
            ("location", "Springfield, ST"),
            ("phone", "(555) 555-0100"),
            ("email", "sam@example.com"),
            ("linkedin", "https://example.com/in/sam"),
            ("github", "https://example.com/code/sam"),
        ),
        education=(
            EducationItem(
                school="Eastvale State University",
                degree="B.S. in Computer Science",
                location="Eastvale, ST",
                date="Sep. 2019 -- May 2023",
                gpa="3.6/4.0",
            ),
        ),
        skills=(
            ("programming_languages", ("Go", "Rust", "SQL")),
            ("databases", ("PostgreSQL", "Redis", "Kafka")),
        ),
    )


LINE = "Reduced p99 latency from 412 ms to 138 ms by batching seven queries into one and caching "
"""A bullet that is long enough to wrap."""


def long_plan() -> ResumePlan:
    """Three work entries and three projects of four two-line bullets each: more than a page.

    Scores are all different so the drop order is fully determined: R, C, Q, B, P, A.
    """

    def bullets(tag: str) -> list[str]:
        return [
            f"{LINE}the hot path of service {tag}{n} so that every request in the load test "
            f"finished well inside the budget"
            for n in range(1, 5)
        ]

    return ResumePlan(
        entries=(
            entry("A", "work", 9, bullets("a")),
            entry("B", "work", 5, bullets("b")),
            entry("C", "work", 2, bullets("c")),
            entry("P", "project", 7, bullets("p")),
            entry("Q", "project", 3, bullets("q")),
            entry("R", "project", 1, bullets("r")),
        ),
        keywords=("Go",),
    )
