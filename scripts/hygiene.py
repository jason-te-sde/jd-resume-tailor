"""Repository hygiene checks, shared by the CI hygiene job and scripts/preflight.sh.

Checks, each of which fails the run with exit code 1:

* no CJK text in any file git tracks (or would track: untracked files that are not ignored);
* no AI attribution in those files, and none in the commit messages of ``--range``;
* with ``--forbidden FILE``, none of the lines of FILE in any of those files except the top-level
  LICENSE (fixed-string, case-insensitive).

Findings print as ``path:line`` and never echo the offending text: the forbidden list is private,
and a finding that quoted it would copy it into a CI log.

Exit codes: 0 clean, 1 findings, 2 usage or environment error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Ideographs, kana, hangul, bopomofo, the CJK symbol and punctuation blocks and full-width forms.
# Written as raw-string escapes, which the formatter leaves alone, so this file holds no CJK text.
CJK = re.compile(
    r"[\u2e80-\u2fdf\u3000-\u303f\u3040-\u30ff\u3100-\u312f\u3190-\u31ff\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]"
)

# The bracketed letters stop these patterns from matching the lines that define them.
ATTRIBUTION = re.compile(
    r"Co-Authored-By:.*([C]laude|[A]nthropic|noreply@[a]nthropic)"
    r"|Generated with .*[C]laude",
    re.IGNORECASE,
)

LICENSE_NAME = "LICENSE"


class HygieneError(Exception):
    """The checks could not run (bad usage or environment), which is not the same as a finding."""


@dataclass(frozen=True)
class Finding:
    """One rule violation at a place, described without quoting the offending text."""

    where: str
    rule: str


def scan_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line number, rule)`` for every line of ``text`` that breaks a content rule."""
    findings: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if CJK.search(line):
            findings.append((number, "CJK character"))
        if ATTRIBUTION.search(line):
            findings.append((number, "AI attribution"))
    return findings


def _git(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        detail = error.stderr.decode("utf-8", "replace").strip() if hasattr(error, "stderr") else ""
        raise HygieneError(f"git {' '.join(args)} failed: {detail or error}") from error
    return result.stdout


def read_files(root: Path) -> dict[str, str]:
    """Return the text of every tracked or untracked-but-not-ignored text file under ``root``."""
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    files: dict[str, str] = {}
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8", "surrogateescape")
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            files[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return files


def check_files(files: dict[str, str]) -> list[Finding]:
    """Apply the content rules to each file."""
    return [
        Finding(f"{name}:{number}", rule)
        for name, text in files.items()
        for number, rule in scan_text(text)
    ]


def check_commits(root: Path, rev_range: str) -> list[Finding]:
    """Apply the content rules to every commit message in ``rev_range``."""
    log = _git(root, "log", "--format=%H%x1f%B%x1e", rev_range).decode("utf-8", "replace")
    findings: list[Finding] = []
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        sha, _, message = record.strip().partition("\x1f")
        findings.extend(
            Finding(f"commit {sha[:10]}, message line {number}", rule)
            for number, rule in scan_text(message)
        )
    return findings


def load_forbidden(path: Path) -> list[str]:
    """Read the forbidden-strings file: one string per line, blank lines ignored.

    Spaces at either end of a line are part of its string, so a short word that also occurs inside
    longer words can be written with a space on each side and match only where it stands alone.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HygieneError(f"cannot read the forbidden list: {error.strerror}") from error
    patterns = [line.lower() for line in lines if line.strip()]
    if not patterns:
        raise HygieneError("the forbidden list has no usable lines, so it would match nothing")
    return patterns


def check_forbidden(files: dict[str, str], patterns: Sequence[str]) -> list[Finding]:
    """Flag each file (the top-level LICENSE aside) that contains any pattern."""
    findings: list[Finding] = []
    for name, text in files.items():
        if name == LICENSE_NAME:
            continue
        lowered = text.lower()
        if any(pattern in lowered for pattern in patterns):
            findings.append(Finding(name, "forbidden string"))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print the findings; return the process exit code."""
    parser = argparse.ArgumentParser(description="Repository hygiene checks.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--range", dest="rev_range", help="git revision range whose messages to check"
    )
    parser.add_argument("--forbidden", type=Path, help="file of strings no file may contain")
    args = parser.parse_args(argv)

    try:
        files = read_files(args.root)
        findings = check_files(files)
        if args.rev_range:
            findings.extend(check_commits(args.root, args.rev_range))
        if args.forbidden:
            findings.extend(check_forbidden(files, load_forbidden(args.forbidden)))
    except HygieneError as error:
        print(f"hygiene: {error}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"{finding.where}: {finding.rule}")
    if findings:
        print(f"hygiene: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"hygiene: ok ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
