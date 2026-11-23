"""Shared pytest configuration."""

import os
import shutil

import pytest
from hypothesis import settings

# A failing property prints its reproduction blob, so a red run can be replayed from the log.
settings.register_profile("default", print_blob=True)
settings.load_profile("default")


def _tex_missing() -> str | None:
    missing = [tool for tool in ("pdflatex", "pdftotext") if shutil.which(tool) is None]
    return f"{' and '.join(missing)} not installed" if missing else None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the ``requires_tex`` tests, and say why, when there is no TeX to run them with."""
    reason = _tex_missing()
    if reason is None:
        return
    if os.environ.get("TAILOR_REQUIRE_TEX"):
        # CI's render job: a missing TeX there must fail the run, not skip its whole point.
        pytest.exit(f"TAILOR_REQUIRE_TEX is set but {reason}", returncode=1)
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "requires_tex" in item.keywords:
            item.add_marker(skip)
