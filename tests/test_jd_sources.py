"""Reading a job description: files, the three job boards' JSON APIs, and plain pages."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tailor.jd import JDError, JobRef, canonicalize, fetch_job, html_to_text, load_job
from tailor.jd.sources import MAX_REDIRECTS

DATA = Path(__file__).parent / "data" / "jd"
GH_ID = "4001234"
LEVER_ID = "0a1b2c3d-1111-2222-3333-444455556666"
ASHBY_ID = "1f2e3d4c-5555-6666-7777-888899990000"


def fixture(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            f"https://boards.greenhouse.io/northwind/jobs/{GH_ID}?gh_src=abc",
            JobRef("greenhouse", "northwind", GH_ID),
        ),
        (
            f"https://job-boards.greenhouse.io/northwind/jobs/{GH_ID}/",
            JobRef("greenhouse", "northwind", GH_ID),
        ),
        (
            f"https://boards.greenhouse.io/embed/job_app?for=northwind&gh_jid={GH_ID}",
            JobRef("greenhouse", "embed", GH_ID),
        ),
        (
            f"https://jobs.lever.co/contoso/{LEVER_ID.upper()}/apply",
            JobRef("lever", "contoso", LEVER_ID),
        ),
        (
            f"https://jobs.ashbyhq.com/fabrikam/{ASHBY_ID}?utm_source=x",
            JobRef("ashby", "fabrikam", ASHBY_ID),
        ),
    ],
)
def test_job_urls_are_canonicalized_to_board_and_job_id(url: str, expected: JobRef) -> None:
    assert canonicalize(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/careers/42",
        "https://jobs.lever.co/contoso",
        "https://boards.greenhouse.io/northwind",
        "https://jobs.ashbyhq.com/fabrikam/not-a-uuid",
    ],
)
def test_other_urls_are_not_board_urls(url: str) -> None:
    assert canonicalize(url) is None


def test_the_api_urls_point_at_each_boards_public_json() -> None:
    assert JobRef("greenhouse", "b", "1").api_url == (
        "https://boards-api.greenhouse.io/v1/boards/b/jobs/1"
    )
    assert JobRef("lever", "b", "x").api_url == "https://api.lever.co/v0/postings/b/x"
    assert JobRef("ashby", "b", "x").api_url == ("https://api.ashbyhq.com/posting-api/job-board/b")


def test_a_greenhouse_job_is_read_from_the_api_with_its_escaped_html_reduced() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=fixture("greenhouse_job.json"))

    job = fetch_job(f"https://boards.greenhouse.io/northwind/jobs/{GH_ID}", client(handler))
    assert seen == [f"https://boards-api.greenhouse.io/v1/boards/northwind/jobs/{GH_ID}"]
    assert (job.title, job.company) == ("Backend Engineer", "Northwind Freight")
    assert "- Design and run Go services that track shipments" in job.text
    assert "PostgreSQL & Kafka" in job.text
    assert "var x" not in job.text
    assert "<" not in job.text


def test_a_lever_job_joins_the_description_and_its_lists() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v0/postings/contoso/{LEVER_ID}"
        return httpx.Response(200, json=fixture("lever_job.json"))

    job = fetch_job(f"https://jobs.lever.co/contoso/{LEVER_ID}", client(handler))
    assert job.title == "Platform Engineer"
    assert "What you will do\n- Build deployment tooling" in job.text
    assert "Experience with Kubernetes" in job.text
    assert "equal opportunity" in job.text


def test_an_ashby_job_is_picked_out_of_the_board_by_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/posting-api/job-board/fabrikam"
        return httpx.Response(200, json=fixture("ashby_board.json"))

    job = fetch_job(f"https://jobs.ashbyhq.com/fabrikam/{ASHBY_ID}", client(handler))
    assert job.title == "Data Engineer"
    assert "Build batch pipelines" in job.text
    assert "Not this one" not in job.text


def test_a_closed_ashby_job_is_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jobs": []})

    with pytest.raises(JDError, match="may be closed"):
        fetch_job(f"https://jobs.ashbyhq.com/fabrikam/{ASHBY_ID}", client(handler))


def test_any_other_url_is_fetched_and_reduced_to_text() -> None:
    page = (DATA / "page.html").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page)

    job = fetch_job("https://example.com/careers/sre", client(handler))
    assert job.text.startswith("Site Reliability Engineer\n")
    assert "Adatum Cloud runs a global network." in job.text
    assert "- Operate Kubernetes clusters" in job.text
    for noise in ("Home Jobs", "Cookies", "track()", "p{}"):
        assert noise not in job.text


def test_http_errors_and_empty_pages_are_errors() -> None:
    with pytest.raises(JDError, match="404"):
        fetch_job("https://example.com/x", client(lambda r: httpx.Response(404)))
    with pytest.raises(JDError, match="no text"):
        fetch_job(
            "https://example.com/x", client(lambda r: httpx.Response(200, text="<html></html>"))
        )


def test_a_text_file_is_read_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "jd.txt"
    path.write_text("\n  Backend Engineer\nGo and Kafka.\n", encoding="utf-8")
    job = load_job(str(path))
    assert job.text == "Backend Engineer\nGo and Kafka."
    assert job.source == str(path)


def test_a_missing_or_empty_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(JDError, match="cannot read"):
        load_job(str(tmp_path / "none.txt"))
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(JDError, match="empty"):
        load_job(str(empty))


def test_html_reduction_keeps_list_items_on_their_own_lines() -> None:
    assert html_to_text("<p>a&nbsp;b</p><ul><li>x</li><li>y</li></ul>") == "a b\n\n- x\n- y"


# ---- hosts, ids and board names --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://evilgreenhouse.io/northwind/jobs/{GH_ID}",
        f"https://boards.greenhouse.io.evil.example/northwind/jobs/{GH_ID}",
        f"https://evil.example/embed/job_app?for=x&gh_jid={GH_ID}",
        f"https://notgreenhouse.io/embed/job_app?for=x&gh_jid={GH_ID}",
        f"https://evil-jobs.lever.co.example/contoso/{LEVER_ID}",
        f"https://fakelever.co/contoso/{LEVER_ID}",
        f"https://xjobs.ashbyhq.com.evil.example/fabrikam/{ASHBY_ID}",
        f"https://evilashbyhq.com/fabrikam/{ASHBY_ID}",
    ],
)
def test_a_host_must_match_on_a_dot_boundary(url: str) -> None:
    assert canonicalize(url) is None


def test_the_service_domain_itself_and_its_subdomains_match() -> None:
    assert canonicalize(f"https://greenhouse.io/northwind/jobs/{GH_ID}") == JobRef(
        "greenhouse", "northwind", GH_ID
    )
    assert canonicalize(f"https://jobs.eu.lever.co/contoso/{LEVER_ID}") == JobRef(
        "lever", "contoso", LEVER_ID
    )
    assert canonicalize(f"https://ashbyhq.com/fabrikam/{ASHBY_ID}") == JobRef(
        "ashby", "fabrikam", ASHBY_ID
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://boards.greenhouse.io/northwind/jobs/abc",
        "https://boards.greenhouse.io/northwind/jobs/12ab",
        "https://boards.greenhouse.io/northwind/jobs/..",
        "https://boards.greenhouse.io/northwind/jobs/-1",
    ],
)
def test_a_greenhouse_job_id_must_be_all_digits(url: str) -> None:
    assert canonicalize(url) is None


@pytest.mark.parametrize("board", ["..", "a%2fb", "a%2e%2e", "a b", "-" * 101])
def test_a_board_name_with_odd_characters_is_an_error(board: str) -> None:
    with pytest.raises(JDError, match="board"):
        canonicalize(f"https://boards.greenhouse.io/{board}/jobs/{GH_ID}")
    with pytest.raises(JDError, match="board"):
        canonicalize(f"https://jobs.lever.co/{board}/{LEVER_ID}")


def test_a_board_name_is_checked_when_a_ref_is_built_by_hand() -> None:
    with pytest.raises(JDError, match="board"):
        JobRef("greenhouse", "../x", "1")
    with pytest.raises(JDError, match="job id"):
        JobRef("greenhouse", "ok", "1/../2")


# ---- the size of a body and the redirects ----------------------------------------------------


def test_a_body_over_the_cap_is_refused_before_it_is_all_read() -> None:
    chunks = 0

    def body() -> Any:
        nonlocal chunks
        for _ in range(100):
            chunks += 1
            yield b"x" * (64 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    with pytest.raises(JDError, match="larger than"):
        fetch_job("https://example.com/huge", client(handler))
    assert chunks < 100


def test_a_declared_length_over_the_cap_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Length": str(50 * 1024 * 1024)}, content=b"x")

    with pytest.raises(JDError, match="larger than"):
        fetch_job("https://example.com/huge", client(handler))


def test_the_cap_applies_to_the_board_apis_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{" + b" " * (3 * 1024 * 1024) + b"}")

    with pytest.raises(JDError, match="larger than"):
        fetch_job(f"https://boards.greenhouse.io/northwind/jobs/{GH_ID}", client(handler))


def test_a_redirect_is_followed_and_each_hop_is_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"Location": "/new"})
        return httpx.Response(200, text="<p>Backend Engineer at Northwind</p>")

    job = fetch_job("https://example.com/old", client(handler))
    assert "Backend Engineer" in job.text


@pytest.mark.parametrize("target", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/1"])
def test_a_redirect_to_another_scheme_is_refused(target: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": target})

    with pytest.raises(JDError, match="redirect"):
        fetch_job("https://example.com/old", client(handler))


def test_a_redirect_loop_ends_after_a_few_hops() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "/again"})

    with pytest.raises(JDError, match="redirect"):
        fetch_job("https://example.com/old", client(handler))
    assert len(seen) == MAX_REDIRECTS + 1


def test_a_board_api_redirect_may_not_leave_its_host() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "evil.example":
            return httpx.Response(200, json=fixture("greenhouse_job.json"))
        return httpx.Response(302, headers={"Location": "https://evil.example/x.json"})

    with pytest.raises(JDError, match="another host"):
        fetch_job(f"https://boards.greenhouse.io/northwind/jobs/{GH_ID}", client(handler))
    assert "evil.example" not in seen


# ---- malformed answers -----------------------------------------------------------------------


@pytest.mark.parametrize("lists", [None, "text", 5, {"text": "x"}, [None], ["x"], [5]])
def test_a_malformed_lever_lists_is_a_clear_error(lists: Any) -> None:
    document = {**fixture("lever_job.json"), "lists": lists}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=document)

    with pytest.raises(JDError, match="lists"):
        fetch_job(f"https://jobs.lever.co/contoso/{LEVER_ID}", client(handler))


def test_a_lever_answer_without_lists_is_fine() -> None:
    document = {k: v for k, v in fixture("lever_job.json").items() if k != "lists"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=document)

    job = fetch_job(f"https://jobs.lever.co/contoso/{LEVER_ID}", client(handler))
    assert job.title == "Platform Engineer"


@pytest.mark.parametrize("jobs", [None, "x", 5, {"id": "x"}, [None], ["x"]])
def test_a_malformed_ashby_board_is_a_clear_error(jobs: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jobs": jobs})

    with pytest.raises(JDError, match="jobs"):
        fetch_job(f"https://jobs.ashbyhq.com/fabrikam/{ASHBY_ID}", client(handler))
