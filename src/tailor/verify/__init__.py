"""The fact-consistency verifier: what a bullet may say, given the facts it cites."""

from tailor.verify.findings import KINDS, Finding, FindingKind
from tailor.verify.report import BulletResult, Report, render_json, render_markdown
from tailor.verify.rounding import Rounding
from tailor.verify.verifier import Bullet, VerifyError, load_bullets, verify, verify_text

__all__ = [
    "KINDS",
    "Bullet",
    "BulletResult",
    "Finding",
    "FindingKind",
    "Report",
    "Rounding",
    "VerifyError",
    "load_bullets",
    "render_json",
    "render_markdown",
    "verify",
    "verify_text",
]
