"""Rendering a plan and a profile to ``.tex``: the golden file and what escaping guarantees."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from renderdata import entry, profile

from tailor.render.escape import escape_latex
from tailor.render.model import (
    InputError,
    ResumePlan,
    load_plan,
    load_profile,
    plan_from_dict,
)
from tailor.render.tex import (
    category_label,
    ordered_skills,
    project_name,
    render_tex,
    template_text,
)

DATA = Path(__file__).parent / "data" / "render"
GOLDEN = DATA / "golden.tex"


def _render() -> str:
    _, plan = load_plan(DATA / "plan.json")
    return render_tex(plan, load_profile(DATA / "profile.json"))


def test_the_tex_equals_the_golden_file() -> None:
    tex = _render()
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(tex, encoding="utf-8")
    assert tex == GOLDEN.read_text(encoding="utf-8")


def test_no_placeholder_or_region_marker_is_left() -> None:
    tex = _render()
    assert "@@" not in tex
    assert "%%BEGIN" not in tex
    assert "%%END" not in tex


def test_every_special_character_of_the_inputs_is_escaped_in_the_body() -> None:
    tex = _render()
    for escaped in (
        r"Acme \& Co. (100\% remote)",
        r"38\% $\rightarrow$ 138 ms for the \#1 endpoint, with a cache\_ttl",
        r"\{retry\}",
        r"C:\textbackslash{}jobs",
        r"\textasciitilde{}4 week",
        r"costs \$5 --- ``quoted''",
        r"Sam O'Neil \& Partners",
        r"C++",
    ):
        assert escaped in tex, escaped


def test_urls_keep_their_query_but_not_the_characters_tex_chokes_on() -> None:
    tex = _render()
    assert r"\codelink{https://example.com/code/queue_lab?a=1\%26b=2\#readme}" in tex
    assert r"\href{https://example.com/in/sam_o}{example.com/in/sam\_o}" in tex
    assert r"\href{mailto:sam_o@example.com}{sam\_o@example.com}" in tex


def test_work_and_projects_go_in_their_own_sections_in_plan_order() -> None:
    tex = _render()
    assert tex.index("Acme") < tex.index(r"\section{Projects}") < tex.index("Queue Lab")


def test_a_section_with_nothing_to_show_is_left_out() -> None:
    _, plan = load_plan(DATA / "plan.json")
    only_work = replace(plan, entries=tuple(e for e in plan.entries if e.type == "work"))
    tex = render_tex(only_work, load_profile(DATA / "profile.json"))
    assert r"\section{Projects}" not in tex
    assert r"\section{Experience}" in tex


def test_text_that_looks_like_a_placeholder_is_not_substituted() -> None:
    plan = ResumePlan(entries=(entry("a", "work", 5, ["Wrote @@EDUCATION@@ and @@SKILLS@@."]),))
    tex = render_tex(plan, replace(profile(), name="@@PROJECTS@@"))
    assert "Wrote @@EDUCATION@@ and @@SKILLS@@." in tex
    assert "\\textbf{\\Huge \\scshape @@PROJECTS@@}" in tex


def test_the_template_never_escapes_anything_itself() -> None:
    # The template is layout only: whatever text it holds is fixed, and every user string reaches
    # the document through escape_latex or escape_url in tex.py.
    template = template_text()
    for name in ("NAME", "CONTACT", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"):
        assert f"@@{name}@@" in template
    code = [ln for ln in template.splitlines() if not ln.lstrip().startswith("%")]
    assert "escape" not in "\n".join(code).lower()


def test_the_template_keeps_the_original_attribution_and_licence() -> None:
    head = template_text().split(r"\documentclass", 1)[0]
    assert "Jake Gutierrez" in head
    assert "License : MIT" in head
    assert "Permission is hereby granted, free of charge" in head


def test_project_name_is_what_comes_before_the_colon() -> None:
    assert project_name("Tidekv: an embedded key-value store in Go") == "Tidekv"
    assert project_name("No colon here") == "No colon here"


def test_skills_the_posting_names_come_first_in_each_category() -> None:
    items = ("Kafka", "Redis (cache)", "SQL", "Go")
    assert ordered_skills(items, ("Redis", "go services")) == [
        "Redis (cache)",
        "Go",
        "Kafka",
        "SQL",
    ]
    assert ordered_skills(items, ()) == list(items)
    assert ordered_skills(("C", "C++"), ("C++",)) == ["C++", "C"]


def test_category_labels() -> None:
    assert category_label("programming_languages") == "Languages"
    assert category_label("ml_tools") == "Ml tools"


def test_a_plan_without_heading_data_is_refused_with_a_hint() -> None:
    document = {"version": 1, "entries": [{"id": "x", "type": "work", "title": "T", "bullets": []}]}
    with pytest.raises(InputError, match="run `tailor plan` again"):
        plan_from_dict(document)


def test_a_plan_of_another_version_is_refused() -> None:
    with pytest.raises(InputError, match="version"):
        plan_from_dict({"version": 2, "entries": []})


def test_a_profile_needs_a_name(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"email": "a@example.com"}', encoding="utf-8")
    with pytest.raises(InputError, match="'name' is required"):
        load_profile(path)


def test_an_unreadable_or_malformed_profile_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="cannot read"):
        load_profile(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        load_profile(bad)


def test_escaping_is_applied_exactly_once() -> None:
    plan = ResumePlan(entries=(entry("a", "work", 5, ["100%"]),))
    tex = render_tex(plan, profile())
    assert escape_latex("100%") == r"100\%"
    assert r"100\%" in tex
    assert r"100\\%" not in tex
