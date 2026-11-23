"""The two renderings of a report: JSON for programs, Markdown for people."""

import json

from bankdata import PROJECT, WORK, bank

from tailor.bank import parse_bank
from tailor.verify.report import render_json, render_markdown
from tailor.verify.verifier import Bullet, verify

the_bank = parse_bank(bank(WORK, PROJECT))
BULLETS = [
    Bullet("Reduced p99 latency from 412 ms to 138 ms.", "acme-swe", id="good"),
    Bullet("Always cut p99 latency to 90 ms.", "acme-swe", id="bad"),
]


def test_json_has_a_summary_and_one_object_per_bullet() -> None:
    document = json.loads(render_json(verify(the_bank, BULLETS)))
    assert document["ok"] is False
    assert document["rounding"] == "conservative"
    assert document["summary"] == {
        "bullets": 2,
        "ok": 1,
        "violations": 1,
        "findings": {"escalation_absolute": 1, "fabricated_number": 1},
    }
    good, bad = document["bullets"]
    assert good == {
        "index": 0,
        "id": "good",
        "entry": "acme-swe",
        "text": "Reduced p99 latency from 412 ms to 138 ms.",
        "sources": ["acme-swe#1", "acme-swe#2", "acme-swe#scope"],
        "status": "ok",
        "findings": [],
    }
    assert bad["status"] == "violation"
    first = bad["findings"][0]
    assert set(first) == {"kind", "span", "text", "evidence"}
    assert bad["text"][first["span"][0] : first["span"][1]] == first["text"] == "Always"


def test_json_of_an_all_ok_report_says_ok() -> None:
    document = json.loads(render_json(verify(the_bank, BULLETS[:1])))
    assert document["ok"] is True
    assert document["summary"]["findings"] == {}


def test_json_is_stable_text() -> None:
    report = verify(the_bank, BULLETS)
    assert render_json(report) == render_json(report)
    assert render_json(report).endswith("\n")


def test_markdown_lists_each_bullet_and_each_finding() -> None:
    text = render_markdown(verify(the_bank, BULLETS))
    assert text.startswith("# Verification report")
    assert "2 bullets: 1 ok, 1 with violations" in text
    assert "## 1. ok" in text
    assert "## 2. 2 violations" in text
    assert "> Always cut p99 latency to 90 ms." in text
    assert "- **escalation_absolute** `Always`" in text
    assert "- **fabricated_number** `90 ms`" in text
    assert "no time in the sources equals" in text


def test_markdown_names_the_rounding_policy() -> None:
    text = render_markdown(verify(the_bank, BULLETS[:1], rounding="nearest"))
    assert "rounding: nearest" in text
    assert "all bullets ok" in text


def test_markdown_shows_the_cited_facts() -> None:
    text = render_markdown(verify(the_bank, BULLETS[:1]))
    assert "acme-swe#1" in text
