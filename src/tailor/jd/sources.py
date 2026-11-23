"""Read a job description from a file or a URL.

Greenhouse, Lever and Ashby job URLs are turned into the board and job id first, then read from
each service's public JSON job-board API. Any other URL is fetched and reduced to text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from tailor.jd.text import html_to_text, unescape_html

USER_AGENT: Final = "jd-resume-tailor (+https://github.com/jason-te-sde/jd-resume-tailor)"
_UUID: Final = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
MAX_BODY: Final = 2 * 1024 * 1024
MAX_REDIRECTS: Final = 5
_REDIRECT_STATUSES: Final = frozenset({301, 302, 303, 307, 308})
_BOARD: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_JOB_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")


class JDError(Exception):
    """A job description could not be read."""


@dataclass(frozen=True, slots=True)
class JobRef:
    """A job on a known board: which service, the board and the job id."""

    service: str
    board: str
    job_id: str

    def __post_init__(self) -> None:
        """Refuse a board name or job id that could change the path of the API URL."""
        if not _BOARD.fullmatch(self.board):
            raise JDError(f"{self.board!r} is not a valid board name")
        if not _JOB_ID.fullmatch(self.job_id):
            raise JDError(f"{self.job_id!r} is not a valid job id")

    @property
    def api_url(self) -> str:
        """The public JSON endpoint for this job."""
        if self.service == "greenhouse":
            return f"https://boards-api.greenhouse.io/v1/boards/{self.board}/jobs/{self.job_id}"
        if self.service == "lever":
            return f"https://api.lever.co/v0/postings/{self.board}/{self.job_id}"
        return f"https://api.ashbyhq.com/posting-api/job-board/{self.board}"


@dataclass(frozen=True, slots=True)
class JobText:
    """The text of a posting, and what the source itself knew about it."""

    text: str
    source: str
    title: str = ""
    company: str = ""


def _on(host: str, domain: str) -> bool:
    """Whether ``host`` is ``domain`` or one of its subdomains (a match on a dot boundary)."""
    return host == domain or host.endswith(f".{domain}")


def canonicalize(url: str) -> JobRef | None:
    """Return the board and job id of a Greenhouse, Lever or Ashby job URL, else ``None``.

    Raises:
        JDError: The URL is a job URL of one of the services but its board name is not a plain
            name (it has characters that could change the path of the API request).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if (
        _on(host, "greenhouse.io")
        and len(parts) >= 3
        and parts[1] == "jobs"
        and re.fullmatch(r"\d+", parts[2])
    ):
        return JobRef("greenhouse", parts[0], parts[2])
    if _on(host, "lever.co") and len(parts) >= 2 and re.fullmatch(_UUID, parts[1]):
        return JobRef("lever", parts[0], parts[1].lower())
    if _on(host, "ashbyhq.com") and len(parts) >= 2 and re.fullmatch(_UUID, parts[1]):
        return JobRef("ashby", parts[0], parts[1].lower())
    if _on(host, "greenhouse.io") and len(parts) >= 1:
        job = parse_qs(parsed.query).get("gh_jid", [""])[0]
        if job.isdigit():
            return JobRef("greenhouse", parts[0], job)
    return None


@dataclass(frozen=True, slots=True)
class _Body:
    content: bytes
    encoding: str


def _read_capped(url: str, response: httpx.Response) -> bytes:
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and int(declared) > MAX_BODY:
        raise JDError(f"{url} is larger than {MAX_BODY // (1024 * 1024)} MiB")
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
        if len(body) > MAX_BODY:
            raise JDError(f"{url} is larger than {MAX_BODY // (1024 * 1024)} MiB")
    return bytes(body)


def _get(client: httpx.Client, url: str, *, stay_on: str | None = None) -> _Body:
    """Fetch ``url``, following at most a few redirects and reading at most ``MAX_BODY`` bytes.

    Every hop, the first included, must be http or https, and with ``stay_on`` must be on that
    host. The body is read as a stream and abandoned once it is over the cap.
    """
    current = url
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            scheme = urlparse(current).scheme.lower()
            if scheme not in ("http", "https"):
                raise JDError(f"{url} redirected to {current!r}: only http and https are followed")
            if stay_on is not None and (urlparse(current).hostname or "").lower() != stay_on:
                raise JDError(f"{url} redirected to another host ({current!r})")
            with client.stream(
                "GET", current, headers={"User-Agent": USER_AGENT}, follow_redirects=False
            ) as response:
                location = response.headers.get("Location")
                if response.status_code in _REDIRECT_STATUSES and location:
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                return _Body(_read_capped(url, response), response.encoding or "utf-8")
    except httpx.HTTPStatusError as error:
        raise JDError(f"{url} answered {error.response.status_code}") from error
    except httpx.HTTPError as error:
        raise JDError(f"cannot fetch {url}: {error}") from error
    raise JDError(f"{url} redirected more than {MAX_REDIRECTS} times")


def _json(client: httpx.Client, url: str) -> Any:
    host = (urlparse(url).hostname or "").lower()
    try:
        return json.loads(_get(client, url, stay_on=host).content)
    except ValueError as error:
        raise JDError(f"{url} did not return JSON") from error


def parse_greenhouse(data: dict[str, Any], ref: JobRef) -> JobText:
    """The job from a Greenhouse job-board API answer."""
    body = html_to_text(unescape_html(str(data.get("content", ""))))
    title = str(data.get("title", ""))
    return JobText(
        text=f"{title}\n\n{body}".strip(),
        source=ref.api_url,
        title=title,
        company=str(data.get("company_name") or ref.board),
    )


def parse_lever(data: dict[str, Any], ref: JobRef) -> JobText:
    """The job from a Lever postings API answer."""
    title = str(data.get("text", ""))
    chunks = [title, str(data.get("descriptionPlain", ""))]
    lists = data.get("lists", [])
    if not isinstance(lists, list) or not all(isinstance(item, dict) for item in lists):
        raise JDError(f"{ref.api_url}: the Lever answer has a malformed 'lists' (expected a list)")
    for section in lists:
        chunks.append(f"{section.get('text', '')}\n{html_to_text(str(section.get('content', '')))}")
    chunks.append(str(data.get("additionalPlain", "")))
    text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return JobText(text=text, source=ref.api_url, title=title, company=ref.board)


def parse_ashby(data: dict[str, Any], ref: JobRef) -> JobText:
    """The job with ``ref.job_id`` from an Ashby job-board API answer."""
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list) or not all(isinstance(item, dict) for item in jobs):
        raise JDError(f"{ref.api_url}: the Ashby answer has a malformed 'jobs' (expected a list)")
    for job in jobs:
        if str(job.get("id", "")).lower() == ref.job_id:
            title = str(job.get("title", ""))
            body = str(job.get("descriptionPlain") or html_to_text(str(job.get("descriptionHtml"))))
            return JobText(
                text=f"{title}\n\n{body}".strip(),
                source=ref.api_url,
                title=title,
                company=ref.board,
            )
    raise JDError(f"no job {ref.job_id} on the Ashby board {ref.board!r} (it may be closed)")


def fetch_job(url: str, client: httpx.Client | None = None) -> JobText:
    """Read the job at ``url``: through the board's JSON API when it has one, else as a web page."""
    owned = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        ref = canonicalize(url)
        if ref is None:
            page = _get(http, url)
            text = html_to_text(page.content.decode(page.encoding, errors="replace"))
            if not text:
                raise JDError(f"{url} has no text (it may need JavaScript to render)")
            return JobText(text=text, source=url)
        data = _json(http, ref.api_url)
        if not isinstance(data, dict):
            raise JDError(f"{ref.api_url} returned an unexpected document")
        parse = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby}
        return parse[ref.service](data, ref)
    finally:
        if owned:
            http.close()


def load_job(source: str, client: httpx.Client | None = None) -> JobText:
    """Read a job description from a text file, or from a URL when ``source`` starts with http."""
    if re.match(r"https?://", source):
        return fetch_job(source, client)
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise JDError(f"cannot read {source}: {error.strerror}") from error
    if not text.strip():
        raise JDError(f"{source} is empty")
    return JobText(text=text.strip(), source=str(path))
