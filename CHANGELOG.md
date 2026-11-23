# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/); before 1.0, the fact-bank format may change between
minor versions.

## Unreleased

### Added

- `tailor eval pipeline`: runs `tailor plan` live over the bank, twelve invented job descriptions
  (`eval/pipeline/jds/`) and several seeds with a given provider and model, records every model
  exchange into cassettes, and writes one results file: per run the bullets planned, calls, tokens,
  wall time and the 1-minute load average; per bullet the first rewrite's verifier findings (the
  gate-off outcome), the outcome after the gate and the final text; and the commit, tree state,
  model, provider, Ollama version, model digest and machine. It refuses to start or write from a
  dirty tree, and records a crashed run as failed. `--replay` rebuilds one committed run from
  cassettes with no model and checks it against the committed numbers.
- The scorer: gate-off violation rate per class and in total, fallback and passed-after-retry
  rate, time, tokens and load per resume (median and range), run totals, time split by load, per
  job description; each rate with an exact percent. `--halves N` scores the first N runs and the
  rest; `--citations` counts the fact ids copied into bullets, by shape.
- `--seed` and `TAILOR_SEED`: the OpenAI-compatible provider sends `seed` with every call. The
  Anthropic API takes no seed, and a seed given for it is reported on stderr. The seed is part of a
  cassette's hash only when set, so earlier cassettes keep their names.
- `tailor judge` and `tailor eval judge`: an LLM judge that checks a bullet against the facts it
  cites, per violation class, with a reason, and its precision and recall next to the verifier's on
  the verifier's holdout and the blind set.
- `tailor eval judge-runs`, `sample` and `residual`: the judge over every final bullet of the runs,
  a blind sample (every flagged bullet plus a seeded random draw, shuffled, with no outcome,
  verdict or finding shown) and the residual violation rate from its labels, with a Wilson interval.
- `tailor eval first-sample` and `first-rewrites`: a seeded blind sample of the first rewrites
  that reached the verifier, and the labelled first-rewrite violation rate with the verifier's
  precision and recall against those labels.
- `--check` on every command that writes a scored file (`pipeline`, `judge`, `judge-runs`,
  `sample`, `first-sample`, `residual`, `first-rewrites`): it recomputes the file from its
  committed inputs with no model call and compares; CI runs each. `--rescore` writes the scored
  part again from the same inputs.
- `scripts/readme_numbers.py`: each measured section of the top-level README and of every
  `eval/**/README.md` names the files it quotes, and every figure in it must be a value of those
  files, so a figure typed by hand or left behind by a re-run fails CI.
- The protocol of the evaluation and its measured record, `eval/pipeline/README.md`, with the
  results of the first full run: 36 runs of `qwen3:8b` over 12 postings and 3 seeds, and what the
  gate changed by two labellings from the facts, of first rewrites and of the bullets that passed
  it, with the verifier's precision and recall against them and the judge's accuracy next to the
  verifier's. CI re-scores every committed run and replays one from its cassettes.
- `tailor eval verifier --score-only`: score a whole dataset with no split (used for the blind
  set), writing nothing.
- 32 dev rows for percent reductions, number binding, scope numbers and production claims.
- `tailor run` and `tailor render`: render a plan to a one-page LaTeX resume (Jake Gutierrez's MIT
  template with placeholders, one tested escaping function), compile it with pdflatex or xelatex,
  check the layout (page count, overfull boxes, bullet gaps, wrapped bullets, page fill, ATS text,
  repeated openers) and fit it to one page by dropping bullets and entries in a fixed order,
  recording each drop in `provenance.md`. `plan.json` now carries each entry's heading data.
- Project scaffold: Python 3.12+, uv with a lockfile, ruff, mypy in strict mode, pytest with
  warnings as errors, and a CI workflow with hygiene, lint and test jobs.
- `scripts/hygiene.py` and `scripts/preflight.sh`: the checks CI runs, plus an optional private
  forbidden-strings gate for maintainers.
- The fact-bank format as a JSON Schema (draft 2020-12), including the optional structured `facts`
  field, and `load_bank` / `parse_bank`, which validate a bank and report every problem with its
  JSON path before returning typed, immutable entries.
- Views of a bank for the verifier: `facts_for`, `fact`, `do_not_claim`, `same_project_groups`,
  `candidates` and `warnings`, all in plain text with LaTeX escapes resolved by one tested
  function.
- `tailor bank check <path>`: validates a bank, prints a summary, and exits 0, 1 (invalid) or 2
  (usage or unreadable file).
- An invented example bank and profile in `examples/alex-rivera/`.
- `tailor verify --bank BANK --bullets BULLETS.json [--rounding conservative|nearest]
  [--format json|md]`: checks bullets against the facts they cite for fabricated numbers, number
  drift (rounding the flattering way, by a direction rule per unit), unbacked technologies and
  names, DO NOT CLAIM overlap, absolute words and unsupported causal links; exits 0, 1 (a
  violation) or 2 (could not run).
- Quantity extraction (K/M/B, `%`, `x`, durations, throughputs, sizes, `$`, `+`, ranges, arrow
  pairs) with Hypothesis round-trip properties, and a curated technology vocabulary shipped as
  data.
- `tailor eval verifier` and the labelled dataset in `eval/verifier/` (306 rows, fixed holdout of
  84), with a CI job that fails if the holdout numbers drop by more than 0.02.
- Two invented projects (Tidekv, Dispatch Desk) in the example bank, which now has 3 work entries
  and 10 projects.
- The model layer (`tailor.llm`): `complete_json(messages, schema, *, model)` with an Anthropic
  provider (one tool whose input schema is the target schema, forced where the model allows it) and
  an OpenAI-compatible provider (strict JSON-schema response format, which also reaches a local
  Ollama). Configuration comes from flags, `TAILOR_*` environment variables or `tailor.toml`; there
  is no default model. `TAILOR_CASSETTES=record|replay|off` stores model calls by the hash of the
  request and replays them without a network.
- Job descriptions (`tailor.jd`): from a text file, from a Greenhouse, Lever or Ashby job URL
  through the board's public JSON API, or from any page reduced to text; one model call extracts
  title, company, responsibilities, requirements, nice-to-haves and keywords.
- Selection and rewriting: a deterministic pre-filter and tie-breaks around one scoring call, two
  to three work entries and projects with two to four bullets each, and one rewrite call per entry
  whose bullets cite fact ids. Every rewrite goes through the verifier; failures get one retry with
  the findings shown, then fall back to the bank's own bullet with the reason recorded.
  `adaptable: false` entries are copied without a model call.
- `tailor plan --jd FILE|URL --bank BANK [--profile PROFILE] --out plan.json`: writes the plan (the
  chosen entries, final bullets with sources and verdicts, retries, fallbacks, token usage) and a
  Markdown provenance report next to it.
- Three invented job descriptions and a recorded local-model run of `tailor plan` in
  `examples/alex-rivera/`, replayed by the tests and by CI.

### Changed

- The verifier evaluation's split is recorded in `split.json` and no longer recomputed: rows added
  later go to dev, so the holdout (84 rows) never changes. The overclaim-pattern rows are now
  prefixed `pattern-`. The figures of the invented example bank were changed; the recorded run was made
  again.
- The holdout numbers are re-measured and the CI gate holds them: do-not-claim recall 0.56 to 0.67,
  any-violation recall 0.88 to 0.91, nothing worse. The blind set was re-scored (not re-labelled):
  faithful bullets flagged went from 8 of 27 to 11 of 27.

### Fixed

- A rewrite whose text carries a fact id copied from the prompt (`[id]`, `(id)`, `(sources:
  [id])`, `(fact: id)`, `(Cites id)`) is refused, even when the verifier passes it, so the gate
  retries it once and otherwise falls back to the bank's sentence.
- No option of the command line can be abbreviated: `tailor eval pipeline --seed 2` is an error,
  not `--seeds 2`.
- A label file that labels a row twice is an error, not a silent override; a residual sample with
  no random draw is an error, not a traceback.
- A percentage that is the size of a reduction ("reduced latency by 41%", "41% fewer", "a 41%
  reduction") now rounds the unflattering way: 40.6% may be written 40%, not 41%.
- A number in a bullet must share a metric word with the fact it matches, so a number cannot be
  borrowed from another cited fact (`verify/binding.py`).
- A scope fact is context, not evidence of an achievement: its numbers no longer count as sources
  for a bullet's numbers. `production` and `led the team` are flagged when no fact says them.
- `tailor plan --provider anthropic` without `ANTHROPIC_API_KEY` ends in a one-line error.
- `tailor plan --out plan.md` no longer lets the report overwrite the plan (the report is
  `plan.report.md`).
- A malformed score from the model (`covers: null`, a `scores` that is not a list) is an
  `LLMError`, not a `TypeError`.
- Job URLs: response bodies are capped at 2 MiB, hosts match on a dot boundary, redirects are
  followed by hand (http and https only, at most 5, a board API call stays on its host), board
  names and job ids are validated, and a malformed Lever `lists` or Ashby `jobs` is a clear error.
- `TAILOR_CASSETTES=record` refuses to write cassettes inside a repository unless the bank is
  under that repository's `examples/`.
