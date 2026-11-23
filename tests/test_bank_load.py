import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from bankdata import NO_CODE, PROJECT, WORK, bank
from jsonschema import Draft202012Validator

from tailor.bank import (
    Bank,
    BankError,
    Fact,
    Number,
    ProjectEntry,
    WorkEntry,
    load_bank,
    parse_bank,
)
from tailor.bank.validate import Problem, format_path, validate

# The format itself.


def test_the_shipped_schema_is_a_valid_draft_2020_12_document() -> None:
    text = resources.files("tailor.bank").joinpath("bank.schema.json").read_text(encoding="utf-8")
    schema = json.loads(text)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)


def test_a_minimal_bank_is_valid() -> None:
    assert validate(bank(NO_CODE)) == []


def test_the_dollar_schema_key_is_allowed_at_the_top() -> None:
    document = bank()
    document["$schema"] = "https://example.com/bank.json"
    assert validate(document) == []


# Loading into typed values.


def test_work_and_project_entries_load_with_every_field() -> None:
    loaded = parse_bank(bank())
    work, project = loaded.entries
    assert isinstance(work, WorkEntry)
    assert isinstance(project, ProjectEntry)
    assert (work.id, work.type, work.company, work.tech) == (
        "acme-swe",
        "work",
        "Acme Corp",
        "Go, PostgreSQL",
    )
    assert work.bullets == tuple(WORK["bullets"])
    assert work.bullet_capabilities == ("performance", "observability")
    assert work.tags == ("backend", "latency")
    assert (work.location, work.date, work.description) == (
        "Remote",
        "Jan. 2024 -- Present",
        "Backend work.",
    )
    assert work.scope == WORK["scope"]
    assert work.adaptable is True
    assert work.facts is None
    assert (project.id, project.type, project.has_code) == ("queue-lab", "project", True)
    assert (project.tech_stack, project.github) == (
        "Go, SQLite",
        "https://example.com/code/queue-lab",
    )


def test_absent_optional_fields_take_their_defaults() -> None:
    (entry,) = parse_bank(bank(NO_CODE)).entries
    assert isinstance(entry, ProjectEntry)
    assert entry.has_code is False
    assert entry.adaptable is True  # an entry opts out of rewriting, not in
    assert entry.bullet_capabilities == ()
    assert entry.tags == ()
    assert (entry.date, entry.location, entry.description, entry.scope) == ("", "", "", "")
    assert (entry.tech_stack, entry.github) == ("", "")
    assert entry.facts is None


def test_explicit_facts_load_with_plain_text_and_numbers() -> None:
    document = bank(PROJECT)
    document["experiences"][0]["facts"] = [
        {
            "id": "q1",
            "text": r"Reached 9,200 jobs/s at 99\% success",
            "numbers": [
                {"value": 9200, "unit": "jobs/s", "as_measured": "9,200"},
                {"value": 99, "unit": "%"},
            ],
        },
        {"id": "q2", "text": "Ran on 3 nodes"},
    ]
    (entry,) = parse_bank(document).entries
    assert entry.facts == (
        Fact(
            id="q1",
            entry_id="queue-lab",
            text="Reached 9,200 jobs/s at 99% success",
            kind="explicit",
            numbers=(Number(9200, "jobs/s", "9,200"), Number(99, "%")),
        ),
        Fact(id="q2", entry_id="queue-lab", text="Ran on 3 nodes", kind="explicit"),
    )


def test_entries_keep_the_order_of_the_file() -> None:
    ids = [e.id for e in parse_bank(bank(PROJECT, WORK, NO_CODE)).entries]
    assert ids == ["queue-lab", "acme-swe", "private-model"]


def test_the_loaded_bank_is_immutable() -> None:
    loaded = parse_bank(bank())
    with pytest.raises(AttributeError):
        loaded.entries = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        loaded.entries[0].title = "x"  # type: ignore[misc]


# Broken banks: each must be rejected, at the right path.


def mutate(document: dict[str, Any], entry: int, **changes: Any) -> dict[str, Any]:
    """Set (or, with the value DELETE, remove) fields of an entry of ``document``."""
    for key, value in changes.items():
        if value is DELETE:
            del document["experiences"][entry][key]
        else:
            document["experiences"][entry][key] = value
    return document


DELETE = object()

BROKEN: list[tuple[str, Any, str, str]] = [
    # (case id, document, path that must be reported, text that must be in its message)
    (
        "missing id",
        mutate(bank(), 1, id=DELETE),
        "$.experiences[1].id",
        "required property is missing",
    ),
    (
        "missing type",
        mutate(bank(), 0, type=DELETE),
        "$.experiences[0].type",
        "required property is missing",
    ),
    ("unknown type", mutate(bank(), 0, type="hobby"), "$.experiences[0].type", "not one of"),
    ("missing title", mutate(bank(), 0, title=DELETE), "$.experiences[0].title", "missing"),
    (
        "work without company",
        mutate(bank(), 0, company=DELETE),
        "$.experiences[0].company",
        "missing",
    ),
    (
        "project without has_code",
        mutate(bank(), 1, has_code=DELETE),
        "$.experiences[1].has_code",
        "missing",
    ),
    ("missing bullets", mutate(bank(), 0, bullets=DELETE), "$.experiences[0].bullets", "missing"),
    (
        "empty bullets",
        mutate(bank(), 0, bullets=[], bullet_capabilities=[]),
        "$.experiences[0].bullets",
        "non-empty",
    ),
    (
        "blank bullet",
        mutate(bank(), 0, bullets=["ok", "  "], bullet_capabilities=["a", "b"]),
        "$.experiences[0].bullets[1]",
        "must not be blank",
    ),
    ("blank title", mutate(bank(), 0, title=""), "$.experiences[0].title", "must not be blank"),
    (
        "bullets not a list",
        mutate(bank(), 0, bullets="one bullet"),
        "$.experiences[0].bullets",
        "not of type 'array'",
    ),
    (
        "bullet not a string",
        mutate(bank(), 0, bullets=["ok", 7], bullet_capabilities=["a", "b"]),
        "$.experiences[0].bullets[1]",
        "not of type 'string'",
    ),
    (
        "has_code not a boolean",
        mutate(bank(), 1, has_code="yes"),
        "$.experiences[1].has_code",
        "not of type 'boolean'",
    ),
    (
        "adaptable not a boolean",
        mutate(bank(), 0, adaptable="true"),
        "$.experiences[0].adaptable",
        "not of type 'boolean'",
    ),
    (
        "tags not a list",
        mutate(bank(), 0, tags={"a": 1}),
        "$.experiences[0].tags",
        "not of type 'array'",
    ),
    (
        "scope not a string",
        mutate(bank(), 0, scope=["x"]),
        "$.experiences[0].scope",
        "not of type 'string'",
    ),
    (
        "bad id characters",
        mutate(bank(), 0, id="has space"),
        "$.experiences[0].id",
        "letters, digits",
    ),
    ("id with a hash", mutate(bank(), 0, id="a#1"), "$.experiences[0].id", "letters, digits"),
    (
        "bad github url",
        mutate(bank(), 1, github="github.com/x/y"),
        "$.experiences[1].github",
        "http(s) URL",
    ),
    (
        "misspelled field",
        mutate(bank(), 0, scpoe="DO NOT CLAIM: x"),
        "$.experiences[0].scpoe",
        "unknown field",
    ),
    (
        "work-only field on a project",
        mutate(bank(), 1, company="x"),
        "$.experiences[1].company",
        "for a project entry",
    ),
    (
        "project-only field on work",
        mutate(bank(), 0, has_code=True),
        "$.experiences[0].has_code",
        "for a work entry",
    ),
    (
        "duplicate id",
        mutate(bank(), 1, id="acme-swe"),
        "$.experiences[1].id",
        "duplicate id 'acme-swe', first used at $.experiences[0].id",
    ),
    (
        "same project as a missing id",
        mutate(bank(), 1, scope="Same project as no-such-entry: x."),
        "$.experiences[1].scope",
        "no-such-entry",
    ),
    (
        "same project as itself",
        mutate(bank(), 1, scope="Same project as queue-lab."),
        "$.experiences[1].scope",
        "names the entry itself",
    ),
    (
        "fewer capabilities than bullets",
        mutate(bank(), 0, bullet_capabilities=["a"]),
        "$.experiences[0].bullet_capabilities",
        "has 1 items but there are 2 bullets",
    ),
    (
        "more capabilities than bullets",
        mutate(bank(), 1, bullet_capabilities=["a", "b", "c"]),
        "$.experiences[1].bullet_capabilities",
        "has 3 items but there are 1 bullets",
    ),
    ("empty facts", mutate(bank(), 0, facts=[]), "$.experiences[0].facts", "non-empty"),
    (
        "fact without text",
        mutate(bank(), 0, facts=[{"id": "f1"}]),
        "$.experiences[0].facts[0].text",
        "missing",
    ),
    (
        "fact id with a space",
        mutate(bank(), 0, facts=[{"id": "f 1", "text": "x"}]),
        "$.experiences[0].facts[0].id",
        "letters, digits",
    ),
    (
        "fact with an unknown field",
        mutate(bank(), 0, facts=[{"id": "f1", "text": "x", "nums": []}]),
        "$.experiences[0].facts[0].nums",
        "unknown field",
    ),
    (
        "number value a string",
        mutate(
            bank(), 0, facts=[{"id": "f1", "text": "x", "numbers": [{"value": "9", "unit": "ms"}]}]
        ),
        "$.experiences[0].facts[0].numbers[0].value",
        "not of type 'number'",
    ),
    (
        "number value a boolean",
        mutate(
            bank(), 0, facts=[{"id": "f1", "text": "x", "numbers": [{"value": True, "unit": "ms"}]}]
        ),
        "$.experiences[0].facts[0].numbers[0].value",
        "not of type 'number'",
    ),
    (
        "number without unit",
        mutate(bank(), 0, facts=[{"id": "f1", "text": "x", "numbers": [{"value": 9}]}]),
        "$.experiences[0].facts[0].numbers[0].unit",
        "missing",
    ),
    (
        "as_measured a number",
        mutate(
            bank(),
            0,
            facts=[
                {"id": "f1", "text": "x", "numbers": [{"value": 9, "unit": "", "as_measured": 9}]}
            ],
        ),
        "$.experiences[0].facts[0].numbers[0].as_measured",
        "not of type 'string'",
    ),
    (
        "entry not an object",
        {"experiences": ["acme-swe"]},
        "$.experiences[0]",
        "not of type 'object'",
    ),
    ("no experiences", {}, "$.experiences", "missing"),
    ("empty experiences", {"experiences": []}, "$.experiences", "non-empty"),
    ("experiences not a list", {"experiences": {"a": 1}}, "$.experiences", "not of type 'array'"),
    ("unknown top-level key", {**bank(), "version": 2}, "$.version", "unknown field"),
    ("top level not an object", [], "$", "not of type 'object'"),
]


@pytest.mark.parametrize(
    ("document", "path", "needle"),
    [pytest.param(d, p, n, id=case) for case, d, p, n in BROKEN],
)
def test_a_broken_bank_is_rejected_at_the_right_path(document: Any, path: str, needle: str) -> None:
    with pytest.raises(BankError) as raised:
        parse_bank(document)
    reported = [str(problem) for problem in raised.value.problems]
    matching = [line for line in reported if line.startswith(path + ":") and needle in line]
    assert matching, f"expected {path!r} with {needle!r} among {reported}"
    assert matching[0] in str(raised.value), "the path must be in the exception message"


def test_every_problem_is_reported_at_once() -> None:
    document = bank()
    mutate(document, 0, title=DELETE, bullet_capabilities=["a"])  # schema + capabilities
    mutate(document, 1, id="acme-swe", scope="Same project as ghost.")  # duplicate + dangling
    document["experiences"].append({"id": "x", "type": "work"})  # three more
    problems = [str(p) for p in validate(document)]
    assert problems == [
        "$.experiences[0].bullet_capabilities: has 1 items but there are 2 bullets;"
        " it needs one label per bullet",
        "$.experiences[0].title: required property is missing",
        "$.experiences[1].id: duplicate id 'acme-swe', first used at $.experiences[0].id",
        "$.experiences[1].scope: 'Same project as ghost' names an id that no entry has",
        "$.experiences[2].bullets: required property is missing",
        "$.experiences[2].company: required property is missing",
        "$.experiences[2].title: required property is missing",
    ], problems


def test_the_error_message_counts_and_lists_the_problems() -> None:
    with pytest.raises(BankError) as raised:
        parse_bank({"experiences": [{"type": "work"}]}, source="my-bank.json")
    text = str(raised.value)
    assert text.startswith("invalid fact bank in my-bank.json: 4 problems\n")
    assert "  $.experiences[0].id: required property is missing" in text


def test_a_single_problem_is_not_pluralised() -> None:
    with pytest.raises(BankError, match=r"1 problem\n"):
        parse_bank(mutate(bank(), 0, bullet_capabilities=["a"]))


def test_same_project_may_point_forward_and_backward() -> None:
    document = bank(WORK, PROJECT)
    mutate(document, 0, scope="Same project as queue-lab.")
    mutate(document, 1, scope="Same project as acme-swe.")
    assert validate(document) == []


def test_latex_in_a_same_project_reference_is_resolved_first() -> None:
    later = {**NO_CODE, "id": "snake_case"}
    document = bank(WORK, later)
    mutate(document, 0, scope=r"Same project as snake\_case.")
    assert validate(document) == []


def test_a_long_offending_value_is_clipped_in_the_message() -> None:
    (problem,) = validate(mutate(bank(), 0, tags="x" * 5000))
    assert len(str(problem)) < 300


# Paths.


@pytest.mark.parametrize(
    ("path", "rendered"),
    [
        ((), "$"),
        (("experiences",), "$.experiences"),
        (("experiences", 3, "bullets", 1), "$.experiences[3].bullets[1]"),
        (("a b",), "$['a b']"),
        (("$schema",), "$['$schema']"),
    ],
)
def test_json_path_rendering(path: tuple[str | int, ...], rendered: str) -> None:
    assert format_path(path) == rendered
    assert str(Problem(path, "msg")) == f"{rendered}: msg"


# Reading files.


def write(tmp_path: Path, text: str | bytes, name: str = "bank.json") -> Path:
    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_load_bank_reads_a_valid_file(tmp_path: Path) -> None:
    path = write(tmp_path, json.dumps(bank()))
    loaded = load_bank(path)
    assert isinstance(loaded, Bank)
    assert [e.id for e in loaded.entries] == ["acme-swe", "queue-lab"]


def test_load_bank_accepts_a_str_path_and_a_byte_order_mark(tmp_path: Path) -> None:
    path = write(tmp_path, b"\xef\xbb\xbf" + json.dumps(bank()).encode("utf-8"))
    assert len(load_bank(str(path)).entries) == 2


def test_load_bank_names_the_file_in_the_error(tmp_path: Path) -> None:
    path = write(tmp_path, json.dumps(mutate(bank(), 0, id=DELETE)))
    with pytest.raises(BankError) as raised:
        load_bank(path)
    assert raised.value.source == str(path)
    assert str(path) in str(raised.value)


def test_a_missing_file_is_an_os_error_not_a_bank_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bank(tmp_path / "nope.json")


def test_invalid_json_is_reported_with_its_position(tmp_path: Path) -> None:
    path = write(tmp_path, '{"experiences": [\n  {"id": }\n]}')
    with pytest.raises(BankError) as raised:
        load_bank(path)
    (problem,) = raised.value.problems
    assert problem.path == ()
    assert "not valid JSON" in problem.message
    assert "line 2" in problem.message


def test_a_duplicate_key_is_an_error_not_a_silent_last_wins(tmp_path: Path) -> None:
    # A repeated "scope" would drop the first one, and with it the DO NOT CLAIM limits.
    text = (
        '{"experiences": [{"id": "a", "type": "project", "title": "t", "has_code": true,'
        ' "bullets": ["b"], "scope": "DO NOT CLAIM: x", "scope": "fine"}]}'
    )
    with pytest.raises(BankError) as raised:
        load_bank(write(tmp_path, text))
    (problem,) = raised.value.problems
    assert "duplicate key 'scope'" in problem.message
    assert "'a'" in problem.message


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(tmp_path: Path, constant: str) -> None:
    text = (
        '{"experiences": [{"id": "a", "type": "project", "title": "t", "has_code": true,'
        ' "bullets": ["b"], "facts": [{"id": "f", "text": "t", "numbers":'
        f' [{{"value": {constant}, "unit": ""}}]}}]}}]}}'
    )
    with pytest.raises(BankError) as raised:
        load_bank(write(tmp_path, text))
    assert constant in raised.value.problems[0].message


def test_a_file_that_is_not_utf8_is_a_bank_error(tmp_path: Path) -> None:
    with pytest.raises(BankError, match="not valid UTF-8"):
        load_bank(write(tmp_path, b'{"experiences": "\xff"}'))
