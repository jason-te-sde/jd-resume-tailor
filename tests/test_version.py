import re

import tailor


def test_version_is_pep440_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", tailor.__version__), (
        f"unexpected version string {tailor.__version__!r}"
    )
