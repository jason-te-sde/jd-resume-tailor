"""Tests for scripts/hygiene.py, the checker behind the CI hygiene job and preflight.

A checker that is only tested on input it should flag cannot be told apart from one that flags
everything, so each rule has a case it must let through.
"""

import subprocess
import sys
from pathlib import Path

import hygiene
import pytest

SCRIPT = Path(hygiene.__file__)
THIS_TEST = Path(__file__)

# Assembled at run time: written out, these lines would trip the checks on this very file.
AI_NAME = "Cl" + "aude"
AI_HOST = "anthr" + "opic.com"
TRAILER = f"Co-Authored-By: {AI_NAME} <noreply@{AI_HOST}>"
GENERATED = f"Generated with [{AI_NAME} Code](https://example.com)"
CJK_CHAR = chr(0x4E2D)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_repo(
    root: Path, files: dict[str, str | bytes], messages: tuple[str, ...] = ("chore: start",)
) -> Path:
    """Create a git repo whose first commit holds ``files``, plus one empty commit per message."""
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    git(root, "add", "--all")
    git(root, "commit", "-q", "-m", messages[0])
    for message in messages[1:]:
        git(root, "commit", "-q", "--allow-empty", "-m", message)
    return root


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"README.md": "plain english\n", "src/a.py": "x = 1\n"})
    result = run(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cjk_in_a_file_fails_and_names_the_line_without_echoing_it(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"docs/note.md": f"fine\nbad {CJK_CHAR} here\n"})
    result = run(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "docs/note.md:2" in result.stdout, result.stdout
    assert CJK_CHAR not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "codepoint",
    [0x3042, 0x30AB, 0xAC00, 0x3002, 0xFF0C],
    ids=["hiragana", "katakana", "hangul", "ideographic-stop", "fullwidth-comma"],
)
def test_other_cjk_scripts_and_punctuation_fail(tmp_path: Path, codepoint: int) -> None:
    repo = make_repo(tmp_path / "repo", {"a.txt": f"x{chr(codepoint)}y\n"})
    assert run(repo).returncode == 1


def test_cjk_in_an_untracked_but_not_ignored_file_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.txt": "ok\n"})
    (repo / "new.txt").write_text(CJK_CHAR, encoding="utf-8")
    assert run(repo).returncode == 1


def test_cjk_in_an_ignored_file_is_not_a_finding(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {".gitignore": "scratch.txt\n", "a.txt": "ok\n"})
    (repo / "scratch.txt").write_text(CJK_CHAR, encoding="utf-8")
    result = run(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    payload = b"\x00\x01" + CJK_CHAR.encode("utf-8") + b"\xff\xfe"
    repo = make_repo(tmp_path / "repo", {"image.bin": payload})
    result = run(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("line", [TRAILER, GENERATED], ids=["trailer", "generated-line"])
def test_attribution_in_a_file_fails(tmp_path: Path, line: str) -> None:
    repo = make_repo(tmp_path / "repo", {"CONTRIBUTING.md": f"intro\n{line}\n"})
    result = run(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CONTRIBUTING.md:2" in result.stdout, result.stdout


def test_a_human_co_author_is_not_attribution(tmp_path: Path) -> None:
    line = "Co-Authored-By: Jane Doe <jane@example.com>"
    repo = make_repo(tmp_path / "repo", {"a.md": f"{line}\n"}, (f"feat: x\n\n{line}",))
    result = run(repo, "--range", "HEAD")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_word_alone_is_not_attribution(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.md": f"The {AI_NAME} model is mentioned here.\n"})
    result = run(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("line", [TRAILER, GENERATED], ids=["trailer", "generated-line"])
def test_attribution_in_a_commit_message_in_range_fails(tmp_path: Path, line: str) -> None:
    repo = make_repo(
        tmp_path / "repo", {"a.txt": "ok\n"}, ("chore: base", "feat: one", f"feat: two\n\n{line}")
    )
    result = run(repo, "--range", "HEAD~2..HEAD")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "commit" in result.stdout, result.stdout


def test_attribution_in_a_commit_outside_the_range_is_ignored(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path / "repo",
        {"a.txt": "ok\n"},
        (f"chore: base\n\n{TRAILER}", "feat: one", "feat: two"),
    )
    result = run(repo, "--range", "HEAD~2..HEAD")
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_commit_of_the_range_is_checked_not_only_the_last(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path / "repo",
        {"a.txt": "ok\n"},
        ("chore: base", f"feat: bad\n\n{TRAILER}", "feat: clean after"),
    )
    assert run(repo, "--range", "HEAD~2..HEAD").returncode == 1


def test_cjk_in_a_commit_message_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.txt": "ok\n"}, ("chore: base", f"feat: {CJK_CHAR}"))
    assert run(repo, "--range", "HEAD~1..HEAD").returncode == 1


def test_an_unknown_range_is_a_usage_error_not_a_pass(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.txt": "ok\n"})
    result = run(repo, "--range", "no-such-ref..HEAD")
    assert result.returncode == 2, result.stdout + result.stderr


def test_the_checker_and_its_tests_do_not_match_their_own_patterns() -> None:
    for path in (SCRIPT, THIS_TEST):
        findings = hygiene.scan_text(path.read_text(encoding="utf-8"))
        assert findings == [], f"{path.name} trips its own hygiene rules: {findings}"


# The forbidden-strings gate.


def write_forbidden(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "forbidden.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_forbidden_string_in_a_file_fails_case_insensitively_without_echoing_it(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo", {"a.md": "this mentions Sekrit Name once\n"})
    forbidden = write_forbidden(tmp_path, "other thing", "sekrit name")
    result = run(repo, "--forbidden", str(forbidden))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a.md" in result.stdout, result.stdout
    assert "sekrit" not in (result.stdout + result.stderr).lower()


def test_forbidden_string_only_in_license_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"LICENSE": "Copyright Sekrit Name\n", "a.md": "clean\n"})
    forbidden = write_forbidden(tmp_path, "sekrit name")
    result = run(repo, "--forbidden", str(forbidden))
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_license_file_in_a_subdirectory_is_not_exempt(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"docs/LICENSE": "Sekrit Name\n"})
    forbidden = write_forbidden(tmp_path, "sekrit name")
    assert run(repo, "--forbidden", str(forbidden)).returncode == 1


def test_blank_lines_in_the_forbidden_list_match_nothing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.md": "clean\n"})
    forbidden = write_forbidden(tmp_path, "", "   ", "sekrit name", "")
    result = run(repo, "--forbidden", str(forbidden))
    assert result.returncode == 0, result.stdout + result.stderr


def test_spaces_around_a_forbidden_string_are_kept_so_it_matches_only_a_whole_word(
    tmp_path: Path,
) -> None:
    forbidden = write_forbidden(tmp_path, " zork ")
    inside = make_repo(tmp_path / "inside", {"a.md": "the Bizorky parser\n"})
    alone = make_repo(tmp_path / "alone", {"a.md": "worked at Zork for a year\n"})
    assert run(inside, "--forbidden", str(forbidden)).returncode == 0
    assert run(alone, "--forbidden", str(forbidden)).returncode == 1


def test_an_empty_or_missing_forbidden_list_is_an_error_not_a_pass(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"a.md": "clean\n"})
    empty = write_forbidden(tmp_path, "", "  ")
    assert run(repo, "--forbidden", str(empty)).returncode == 2
    assert run(repo, "--forbidden", str(tmp_path / "missing.txt")).returncode == 2
