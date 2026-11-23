"""The invented example bank must stay valid and keep the features it is there to show."""

import json
import re
from pathlib import Path

import pytest

from tailor.bank import Bank, ProjectEntry, WorkEntry, load_bank
from tailor.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alex-rivera"


@pytest.fixture(scope="module")
def example() -> Bank:
    return load_bank(EXAMPLE / "bank.json")


def test_the_example_is_a_valid_bank_with_no_warnings(example: Bank) -> None:
    assert example.warnings() == []


def test_the_example_has_three_work_entries_and_ten_projects(example: Bank) -> None:
    assert sum(isinstance(e, WorkEntry) for e in example.entries) == 3
    assert sum(isinstance(e, ProjectEntry) for e in example.entries) == 10


def test_two_projects_have_no_code_and_everything_else_is_a_candidate(example: Bank) -> None:
    no_code = [e.id for e in example.entries if isinstance(e, ProjectEntry) and not e.has_code]
    assert len(no_code) == 2, no_code
    assert len(example.candidates()) == 13 - 2
    assert not {e.id for e in example.candidates()} & set(no_code)


def test_the_example_has_exactly_one_same_project_pair(example: Bank) -> None:
    groups = example.same_project_groups()
    assert len(groups) == 1, groups
    assert len(groups[0]) == 2, groups[0]


def test_the_example_has_at_least_six_do_not_claim_items_across_entries(example: Bank) -> None:
    per_entry = {e.id: example.do_not_claim(e.id) for e in example.entries}
    assert sum(len(claims) for claims in per_entry.values()) >= 6
    assert sum(1 for claims in per_entry.values() if claims) >= 6, per_entry


def test_one_entry_has_two_do_not_claim_parts(example: Bank) -> None:
    two_parts = [e.id for e in example.entries if e.scope.count("DO NOT CLAIM:") == 2]
    assert two_parts, "the example should show a scope with more than one DO NOT CLAIM part"


def test_one_entry_carries_structured_facts_with_numbers(example: Bank) -> None:
    with_facts = [e for e in example.entries if e.facts is not None]
    assert len(with_facts) == 1
    numbers = [n for fact in with_facts[0].facts or () for n in fact.numbers]
    assert numbers
    assert any(n.as_measured for n in numbers)


def test_every_entry_has_numbers_in_its_bullets_and_provenance_in_its_scope(example: Bank) -> None:
    for entry in example.entries:
        assert any(re.search(r"\d", bullet) for bullet in entry.bullets), entry.id
        assert "EVIDENCE:" in entry.scope or entry.facts is not None, entry.id
        assert len(entry.bullet_capabilities) == len(entry.bullets), entry.id


def test_the_example_uses_latex_escapes_that_facts_resolve(example: Bank) -> None:
    raw = "\n".join(b for e in example.entries for b in e.bullets)
    assert "\\%" in raw
    assert "$\\rightarrow$" in raw
    for entry in example.entries:
        for fact in example.facts_for(entry.id):
            assert "\\" not in fact.text, f"{fact.id}: {fact.text!r}"


def test_every_link_in_the_example_is_on_example_com(example: Bank) -> None:
    links = [e.github for e in example.entries if isinstance(e, ProjectEntry) and e.github]
    assert links
    assert all(re.fullmatch(r"https://example\.com/\S+", link) for link in links), links


def test_the_profile_is_invented_contact_data() -> None:
    profile = json.loads((EXAMPLE / "profile.json").read_text(encoding="utf-8"))
    assert profile["name"] == "Alex Rivera"
    assert profile["email"].endswith("@example.com")
    assert re.fullmatch(r"\(\d{3}\) 555-01\d\d", profile["phone"]), profile["phone"]
    for key in ("linkedin", "github", "website"):
        assert profile[key].startswith("https://example.com/"), key


def test_the_example_readme_says_everything_is_invented() -> None:
    text = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    assert "Everything in this directory is invented" in text


def test_the_readme_command_prints_the_summary_the_readme_describes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["bank", "check", str(EXAMPLE / "bank.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "entries       13 (work: 3, project: 10)" in out
    assert "same project  1 group" in out
    assert "  gridlock, gridlock-bench" in out
    assert "warnings      none" in out


def test_the_readme_shows_the_output_the_command_really_prints(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    readme = (EXAMPLE.parents[1] / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)```", readme, flags=re.DOTALL)
    monkeypatch.chdir(EXAMPLE.parents[1])
    assert main(["bank", "check", "examples/alex-rivera/bank.json"]) == 0
    printed = capsys.readouterr().out
    assert printed in blocks, f"README output blocks are stale; the command prints:\n{printed}"


def test_the_readme_shows_the_error_the_command_really_prints(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readme = (EXAMPLE.parents[1] / "README.md").read_text(encoding="utf-8")
    shown = re.findall(r"```text\n(invalid fact bank.*?)```", readme, flags=re.DOTALL)
    assert len(shown) == 1
    project = {"id": "p", "type": "project", "title": "t", "has_code": False, "bullets": ["one"]}
    work = {"type": "work", "title": "t", "company": "c", "bullets": ["one"]}
    document = {"experiences": [work, {**project, "bullet_capabilities": ["a", "b"]}]}
    (tmp_path / "my-bank.json").write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["bank", "check", "my-bank.json"]) == 1
    assert capsys.readouterr().err == shown[0]
