# Contributing

The fact-bank format, the verifier (`tailor verify`) and its evaluation, `tailor plan`, the renderer
(`tailor run`), the judge and the end-to-end evaluation (`tailor eval pipeline`) all exist. Issues
and ideas are welcome; larger changes are easier to review after a short issue that states the
problem and how you would know it is solved.

## Prerequisites

Python 3.12 or newer (CI runs 3.12 and 3.13) and [uv](https://docs.astral.sh/uv/). `uv sync` installs
the runtime and the dev tools; run every command through `uv run`.

## Before you push

```sh
bash scripts/preflight.sh
```

That is the repository hygiene check, `ruff check`, `ruff format --check`, `mypy` in strict mode, the
test suite, the verifier gate (`tailor eval verifier --check`), the `--check` of every committed
scored file of the evaluation (each is recomputed from its committed inputs, with no model), the
replay of one evaluation run, and the check that every figure in the READMEs' measured sections is
a value of the files they name. All of it must pass with no warnings: pytest turns warnings into
errors.
Maintainers also set `TAILOR_FORBIDDEN` to a private list of strings that must never appear in the
repository; contributors can leave it unset.

## Workflow

1. Branch from `main` as `<type>/<slug>`, for example `fix/latex-tilde`.
2. Commit in small steps that each pass preflight.
3. Open a pull request. It merges once CI is green and the branch is up to date.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), with the module as the scope:

```
<type>(<scope>): <imperative summary, no trailing period>

<body: why this change, what was considered and rejected, what it does not do,
and how far it was verified>
```

Types in use: `feat`, `fix`, `test`, `docs`, `ci`, `build`, `chore`, and `eval` for data that a
measurement produced (runs, judge verdicts, samples, labels) committed on its own.

No AI attribution anywhere, commit messages included: the `hygiene` job fails on a co-author
trailer or a generated-by line that names an AI tool. Tools add them by default, so delete them
before committing.

## Tests

- A change in behaviour comes with a test that fails without it. Run it and watch it fail first.
- A checker needs a case it must flag and a case it must let through.
- Randomised tests use Hypothesis, whose failure output includes the blob that replays the run.

## Models in tests

No test calls a model or a network. The provider tests use scripted SDK clients and stored request
examples (`tests/data/llm/`), the job-board tests use invented fixtures through httpx's mock
transport (`tests/data/jd/`), and the pipeline tests use `tests/llmfakes.ScriptedLLM`. Two
end-to-end tests replay real runs against a local model: `tests/test_plan_replay.py` replays one
run of `tailor plan` from `examples/alex-rivera/cassettes`, and `tests/test_committed_results.py`
(with CI's `pipeline-replay` job) replays run 22 of the pipeline evaluation from
`eval/pipeline/cassettes` and compares it with the committed record field by field.

A cassette is named by the hash of its request, so any change to what the model is asked breaks a
replay with "no cassette for request <hash>": a prompt, a schema, the bank, a job description, the
selection, or the text of a verifier finding (the retry prompt shows the model its findings, and the
record keeps them). The two replays are fixed differently.

- **The example run** is a demonstration: record it again, with Ollama serving a model you have
  pulled, and commit the cassettes with the new plan (below).
- **The evaluation run** is part of a measurement of 36 runs made at one commit. Do not record its
  cassettes again on their own: one run recorded at another commit would no longer be the run the
  record scores. If your change is meant to change what the model is asked, say so in the pull
  request; the evaluation then has to be measured again in full, by its protocol
  (`eval/pipeline/README.md`), which a maintainer does. If it is not meant to, the failing replay
  has found a change you did not intend.

To record the example run again:

```sh
rm -r examples/alex-rivera/cassettes examples/alex-rivera/plans
TAILOR_CASSETTES=record TAILOR_CASSETTE_DIR=examples/alex-rivera/cassettes \
TAILOR_PROVIDER=openai TAILOR_MODEL=qwen3:8b TAILOR_BASE_URL=http://localhost:11434/v1 \
  uv run tailor plan --jd examples/alex-rivera/jds/backend-engineer.txt \
  --bank examples/alex-rivera/bank.json --profile examples/alex-rivera/profile.json \
  --out examples/alex-rivera/plans/backend-engineer.plan.json
```

The run takes a while on a laptop (every call is one request to the model). Say in the commit what
model produced it.

Recording refuses to write into a repository unless the bank is under that repository's
`examples/`: a cassette holds the bank's text, so a real bank must never be recorded into a
committed directory (set `TAILOR_CASSETTE_DIR` outside the repository for one).

## Rendering

The tests that build a PDF carry `@pytest.mark.requires_tex` and need `pdflatex` and poppler's
`pdftotext` (on Debian or Ubuntu: `texlive-latex-base texlive-latex-extra texlive-fonts-recommended
poppler-utils`; on macOS: BasicTeX and `brew install poppler`). Without them the marked tests skip
with that reason, and everything else still runs; set `TAILOR_REQUIRE_TEX=1` to make a missing TeX
an error instead (CI's render job does). The golden file `tests/data/render/golden.tex`
changes when the template or the escaping does: regenerate it with `UPDATE_GOLDEN=1 uv run pytest
tests/test_render_tex.py` and read the diff. Escaping goes through `tailor/render/escape.py` and
nothing else; the template must not contain text that needs escaping.

To refresh the committed example after a change to the template or the renderer:

```sh
TAILOR_CASSETTES=replay TAILOR_CASSETTE_DIR=examples/alex-rivera/cassettes \
TAILOR_PROVIDER=openai TAILOR_MODEL=qwen3:8b \
  uv run tailor run --jd examples/alex-rivera/jds/backend-engineer.txt \
  --bank examples/alex-rivera/bank.json --profile examples/alex-rivera/profile.json \
  --out examples/alex-rivera/out && rm examples/alex-rivera/out/plan.json
pdftoppm -r 100 -png -singlefile -x 0 -y 0 -W 850 -H 560 \
  examples/alex-rivera/out/resume.pdf examples/alex-rivera/out/resume-top
```

## Figures in the READMEs

A measured section of a README names its sources in its header, for example
`<!-- measured:begin gate eval/pipeline/results.json#scores.gate_off -->`, and every figure in it
must be a value under those paths (`scripts/readme_numbers.py`, run by preflight and CI). To quote
a new figure, have the scorer write it and name its file and path in the header; do not compute it
by hand. A percentage is quoted from a `percent` value, which the scorers write exactly from the
counts.

## The verifier and its dataset

A change to the rules of `src/tailor/verify/` is judged by `tailor eval verifier`. Tune on the dev
rows (`--split dev`); the holdout is for the final numbers, so run the full command once, when the
change is done, and commit the new `eval/verifier/results.json` with the reason. A change that
makes the holdout worse by more than 0.02 fails CI. To add rows, append them to
`eval/verifier/labeled.jsonl` (sources must be exact fact texts of the entry; see
`eval/verifier/README.md`) and run `tailor eval verifier --write-split`: the recorded sides are
kept and the new rows go to dev, so the holdout never changes. `tailor eval verifier --data
eval/verifier/blind --score-only` scores the blind set; it writes nothing and is not the gate.

## Code

- Every public module, class and function has a Google-style docstring; comments explain why, not
  what.
- No module-level mutable state, no wildcard imports, no `print` outside `src/tailor/cli.py` and
  `scripts/`.
- Everything in the repository is in English.
