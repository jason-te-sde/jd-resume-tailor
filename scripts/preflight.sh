#!/usr/bin/env bash
# The local gate: what CI runs for hygiene, lint and tests, in one command.
#
#   bash scripts/preflight.sh
#   TAILOR_FORBIDDEN=/path/to/list.txt bash scripts/preflight.sh
#
# When TAILOR_FORBIDDEN names a file, no file in the repository (the top-level LICENSE aside) may
# contain any line of it, compared as fixed strings and ignoring case. The list is private: it is
# kept outside the repository and the check reports file names only, never its contents.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s ==\n' "$1"; }

# The commits this branch adds: everything not on main. With no main to compare to, all of them.
if git rev-parse --verify --quiet origin/main >/dev/null; then
  range="origin/main..HEAD"
elif git rev-parse --verify --quiet main >/dev/null; then
  range="main..HEAD"
else
  range="HEAD"
fi

step "hygiene (commits ${range})"
hygiene_args=(--range "$range")
if [ -n "${TAILOR_FORBIDDEN:-}" ]; then
  hygiene_args+=(--forbidden "$TAILOR_FORBIDDEN")
else
  echo "TAILOR_FORBIDDEN is not set: the private forbidden-strings check is skipped"
fi
uv run python scripts/hygiene.py "${hygiene_args[@]}"

step "lint"
uv run ruff check .
uv run ruff format --check .
uv run mypy

step "tests"
uv run pytest

step "verifier gate"
uv run tailor eval verifier --check >/dev/null

step "scored files and README numbers"
uv run tailor eval pipeline --check
for scored in judge judge-runs sample first-sample residual first-rewrites; do
  uv run tailor eval "$scored" --check
done
uv run tailor eval pipeline --replay --jd ios-engineer --replay-seed 1
uv run python scripts/readme_numbers.py

step "README quickstart"
uv run tailor bank check examples/alex-rivera/bank.json

printf '\npreflight: all checks passed\n'
