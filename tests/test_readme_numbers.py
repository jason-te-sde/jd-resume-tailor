"""scripts/readme_numbers.py: a number in a measured section is a value of the section's sources.

A checker that only sees numbers it should accept cannot be told from one that accepts everything,
so each rule has a number it must reject, and the real READMEs are mutated one figure at a time.
"""

import json
from pathlib import Path

import pytest
import readme_numbers

ROOT = Path(__file__).resolve().parents[1]

README = """\
# Title

Prose with 7 things outside any measured section.

<!-- measured:begin demo results.json#scores other.json -->
## Results

The model flagged 5 of 40 bullets (12.5%) in 183 s, 1,204 tokens, qwen3:8b, `v0.1.0`,
[link](https://example.com/a/99). It ran from 2026-03 to 2026-11, seed 3.

1. A list marker is not a number.

```text
a fenced block with 4242 is not checked
```
<!-- measured:end -->

More prose, 31337, outside.
"""

RESULTS = {
    "scores": {
        "flagged": 5,
        "total": 40,
        "percent": 12.5,
        "seconds": 183.4,
        "tokens": 1204,
        "seeds": [1, 2, 3],
        "applied": True,
        "note": "seed 99 and 31337 in a string",
    },
    "outside": {"value": 77},
}
OTHER = {"first_month": "2026-03", "last_month": "2026-11-30T10:00:00Z"}


def repo(tmp_path: Path, readme: str = README, results: object = RESULTS) -> Path:
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps(OTHER), encoding="utf-8")
    return tmp_path


def check(tmp_path: Path, readme: str = README, results: object = RESULTS) -> list[str]:
    root = repo(tmp_path, readme, results)
    return readme_numbers.check(root / "README.md", root)


def test_a_section_whose_numbers_are_all_values_of_its_sources_passes(tmp_path: Path) -> None:
    assert check(tmp_path) == []


def test_a_section_names_its_sources_and_a_json_path_narrows_one() -> None:
    (section,) = readme_numbers.sections(README)
    assert section.name == "demo"
    assert section.sources == ("results.json#scores", "other.json")


def test_a_number_that_is_no_value_of_the_sources_is_reported_with_its_line(
    tmp_path: Path,
) -> None:
    problems = check(tmp_path, README.replace("5 of 40", "6 of 40"))
    assert problems == [
        "README.md:8: 6 (section demo) is not a value of results.json#scores, other.json"
    ]


def test_a_value_outside_the_named_json_path_does_not_count(tmp_path: Path) -> None:
    assert check(tmp_path, README.replace("seed 3", "seed 77"))


def test_digits_inside_a_string_value_do_not_count(tmp_path: Path) -> None:
    assert check(tmp_path, README.replace("seed 3", "seed 99"))


def test_a_boolean_is_not_the_number_one(tmp_path: Path) -> None:
    results = json.loads(json.dumps(RESULTS))
    results["scores"]["seeds"] = [2, 3]
    assert check(tmp_path, README.replace("seed 3", "seed 1"), results)


def test_a_figure_may_be_a_value_rounded_half_up_to_the_places_it_shows(tmp_path: Path) -> None:
    assert check(tmp_path, README.replace("in 183 s", "in 183.4 s")) == []
    assert check(tmp_path, README.replace("in 183 s", "in 184 s"))


def test_nothing_is_rounded_to_zero(tmp_path: Path) -> None:
    results = json.loads(json.dumps(RESULTS))
    results["scores"]["rate"] = 0.2647
    assert check(tmp_path, README.replace("seed 3", "seed 0"), results)
    assert check(tmp_path, README.replace("seed 3", "seed 0.3"), results) == []
    results["scores"]["failed"] = 0
    assert check(tmp_path, README.replace("seed 3", "seed 0"), results) == []


def test_a_percent_is_a_value_too_and_is_not_made_from_a_rate(tmp_path: Path) -> None:
    results = json.loads(json.dumps(RESULTS))
    results["scores"]["rate"] = 0.1905  # 8 of 42, rounded: 19.05 would read 19.1
    results["scores"]["percent"] = 100 * 8 / 42
    readme = README.replace("(12.5%)", "(19.0%)")
    assert check(tmp_path, readme, results) == []
    assert check(tmp_path, readme.replace("19.0%", "19.1%"), results)


def test_a_percent_value_is_shown_with_a_percent_sign_and_nothing_else_is(tmp_path: Path) -> None:
    assert check(tmp_path, README.replace("(12.5%)", "(12.5)"))  # a percent shown as a count
    assert check(tmp_path, README.replace("of 40 bullets", "of 40% bullets"))  # a count as one
    results = json.loads(json.dumps(RESULTS))
    results["scores"]["percent"] = 38.46
    readme = README.replace("(12.5%)", "(38.5%)")
    assert check(tmp_path, readme, results) == []
    assert check(tmp_path, readme.replace("of 40 bullets", "of 38 bullets"), results)


def test_a_date_must_be_a_date_of_the_sources(tmp_path: Path) -> None:
    assert check(tmp_path, README.replace("2026-03", "2026-04"))
    assert check(tmp_path, README.replace("to 2026-11", "to 2026-11-30")) == []


def test_numbers_outside_sections_in_code_links_identifiers_and_list_markers_are_not_checked(
    tmp_path: Path,
) -> None:
    numbers = [token for _, token in readme_numbers.numbers_in(readme_numbers.sections(README)[0])]
    assert numbers == ["5", "40", "12.5%", "183", "1,204", "2026-03", "2026-11", "3"]


def test_a_section_without_sources_or_never_closed_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(readme_numbers.CheckError, match="names no source"):
        check(tmp_path, README.replace(" results.json#scores other.json", ""))
    with pytest.raises(readme_numbers.CheckError, match="never closed"):
        check(tmp_path, README.replace("<!-- measured:end -->", ""))


def test_a_source_that_is_missing_or_a_path_that_is_not_there_is_an_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(readme_numbers.CheckError, match=r"nope\.json"):
        check(tmp_path, README.replace("other.json -->", "nope.json -->"))
    with pytest.raises(readme_numbers.CheckError, match=r"scores\.missing"):
        check(tmp_path, README.replace("#scores ", "#scores.missing "))


def test_a_readme_without_a_measured_section_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(readme_numbers.CheckError, match="no measured section"):
        check(tmp_path, "# Nothing measured\n\n12 things.\n")


def test_a_star_in_a_json_path_stands_for_every_key_or_item(tmp_path: Path) -> None:
    results = {"by": {"a": {"n": 5, "x": 31337}, "b": {"n": 40, "x": 1}}, "list": [{"n": 3}]}
    readme = "<!-- measured:begin s results.json#by.*.n results.json#list.*.n -->\n{}\n" + (
        "<!-- measured:end -->\n"
    )
    assert check(tmp_path, readme.format("5 of 40, seed 3"), results) == []
    assert check(tmp_path, readme.format("31337"), results)


def test_a_header_may_run_over_several_lines(tmp_path: Path) -> None:
    readme = README.replace(
        "<!-- measured:begin demo results.json#scores other.json -->",
        "<!-- measured:begin demo\n  results.json#scores\n  other.json\n-->",
    )
    (section,) = readme_numbers.sections(readme)
    assert section.sources == ("results.json#scores", "other.json")
    assert check(tmp_path, readme) == []


def test_the_readmes_checked_by_default_are_the_top_one_and_every_eval_readme() -> None:
    found = [path.relative_to(ROOT).as_posix() for path in readme_numbers.default_readmes(ROOT)]
    assert found[0] == "README.md"
    assert "eval/pipeline/README.md" in found
    assert "eval/verifier/README.md" in found
    assert "eval/verifier/blind/README.md" in found


def test_the_command_line_reports_and_exits_one_on_a_wrong_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = repo(tmp_path, README.replace("5 of 40", "6 of 40"))
    assert readme_numbers.main(["--root", str(root), str(root / "README.md")]) == 1
    assert "README.md:8: 6" in capsys.readouterr().err
    root = repo(tmp_path)
    assert readme_numbers.main(["--root", str(root), str(root / "README.md")]) == 0


# ---- the committed READMEs -------------------------------------------------------------------


def test_every_number_in_the_measured_sections_of_every_readme_is_a_value_of_its_sources() -> None:
    for readme in readme_numbers.default_readmes(ROOT):
        assert readme_numbers.check(readme, ROOT) == [], readme


def test_a_figure_changed_by_one_in_its_last_digit_fails_unless_it_is_a_value_too() -> None:
    report = readme_numbers.mutations(ROOT)
    assert report.total > 500
    survivors = [m for m in report.survivors if not m.is_source_value]
    assert survivors == [], [m.describe() for m in survivors]
    # Each source is narrowed to the values its passage quotes, so few changes land on another.
    assert report.killed / report.total >= 0.85
