# Pipeline evaluation

This directory measures the whole pipeline (`tailor plan`: extract, select, rewrite, verify, retry,
fall back) on a real local model, and the LLM judge that checks the verifier's work. It is the
measurement behind the numbers in the top-level README. Everything here is invented: the bank is
the example bank, the postings are fictional.

This file is the **protocol**. It was committed before the first live model call, so the numbers
cannot have shaped it. Anything that changes it after that is listed under "Deviations" at the end,
with the reason.

## The job descriptions

Twelve fictional postings in `jds/`, each for an invented company, each the length and shape of a
real one (role, responsibilities, requirements, nice-to-haves). They were written for this
evaluation, by hand, from the bank's tags and tech lists and from the kinds of requirements real
postings make; none is copied or adapted from a real posting, and none comes from any person's job
search. Three (`backend-engineer`, `data-platform-engineer`, `distributed-systems-engineer`) are
the example postings in `examples/alex-rivera/jds/`.

They are spread over roles the bank can serve and roles it cannot, because the pressure to invent
is highest where the facts do not fit the posting:

| Posting | What the bank offers | What it is for |
| --- | --- | --- |
| `backend-engineer` | strong: the work entry, tidekv, the dock scheduler | the baseline |
| `data-platform-engineer` | strong: the data internship, the transit lake | the baseline |
| `distributed-systems-engineer` | strong: the rate limiter, the Raft project | the baseline; "exactly-once" pressure |
| `database-internals-engineer` | strong: tidekv, raftlet | storage and recovery claims |
| `site-reliability-engineer` | partial: Terraform, Prometheus, AWS; no Kubernetes | named tools the bank lacks |
| `ml-platform-engineer` | partial: MLflow, ONNX; no feature store, no Kubernetes | named tools the bank lacks |
| `llm-applications-engineer` | partial: one agent project | vector stores and frameworks the bank lacks |
| `frontend-engineer` | weak: a co-op and a dashboard; no Next.js or GraphQL | a different stack, a senior bar |
| `staff-backend-engineer` | weak: junior scope; no Java | a different seniority; leadership and compliance claims |
| `engineering-manager` | poor: no management | "led the team" pressure |
| `ios-engineer` | none | everything would have to be invented |
| `embedded-firmware-engineer` | none | the Rust is real, the firmware is not |

## What is run

- **Model:** `qwen3:8b` (8.2B parameters, Q4_K_M, Ollama image `500a1f067a9f`: Deviation 6), through
  Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`). Ollama 0.23.4 on an Apple M2
  with 16 GB of memory. It is the only measurement on the machine while it runs; the 1-minute load
  average at the start of every run is recorded anyway.
- **Server setting:** `ollama serve` is started with `OLLAMA_CONTEXT_LENGTH=8192`. Ollama's default
  here is 4096 tokens, and the selection prompt alone is about 3,000 tokens, so a default run risks
  a silently cut prompt, which would look like a weak model. This is a window size, not a sampling
  parameter.
- **Sampling:** the provider's defaults: no temperature, top-p, top-k or token limit is sent (the
  model's own defaults apply: temperature 0.6, top-p 0.95, top-k 20). The only parameter added is
  `seed`.
- **Bank and postings:** the example bank, all 12 postings.
- **Seeds:** 1, 2 and 3, each sent as `seed` with every model call of a run. That is 12 postings
  times 3 seeds, 36 runs. A seed is sent so that a run can be repeated on this machine (that was
  not measured: Deviation 6); it does not make the three runs of a posting independent of one
  another in any statistical sense, and nothing below treats them so.
- **Load and speed:** another project's build and test work may run on this machine during the
  runs. This evaluation is not a speed benchmark. Wall time per resume is recorded together with
  the 1-minute load average at the start of each run, and is reported as measured under that load,
  not as the model's speed. Tokens per resume (input and output) is the cost figure that does not
  depend on load.
- **Rounding policy:** the default, `conservative`.
- **Verifier and prompts:** frozen at the commit the results record. A change to the verifier, a
  prompt or the selection after the first live call starts the measurement over (see "Invalid runs").

One run gives both sides of the gate. The first rewrite the model gives for each bullet is checked
by the verifier and its findings are kept (what the resume would hold with the gate off); the
bullet that ends up on the resume (passed first time, passed after one retry, or the bank's own
wording) is what it holds with the gate on. The model is not run twice.

```sh
# the server, with the window above
OLLAMA_CONTEXT_LENGTH=8192 ollama serve

# the 36 runs: needs a clean tree; cassettes go outside the repository while it runs
TAILOR_PROVIDER=openai TAILOR_MODEL=qwen3:8b TAILOR_BASE_URL=http://localhost:11434/v1 \
  uv run tailor eval pipeline --cassettes "$SCRATCH/cassettes"
```

## What counts as a violation

The six classes of the verifier, as `eval/verifier/README.md` defines the labels:
`fabricated_number`, `number_drift`, `fabricated_entity`, `do_not_claim`, `escalation_absolute`,
`escalation_causal`, judged against the facts the bullet cites, the entry's scope note and its DO
NOT CLAIM list.

- **Gate-off violation:** a rewritten bullet whose *first* rewrite has at least one verifier
  finding. The rate is over bullets the model rewrote (a bullet of an `adaptable: false` entry is
  copied without a call and is in no rate). A rewrite that cites no fact, or one the entry does not
  have, never reaches the verifier; it is counted separately as `structural`, not as a violation.
- **Residual violation:** a bullet the verifier passed (first time or after a retry) that a blind
  label from the facts calls a violation of any class. The bank's own wording (fallbacks, verbatim
  entries) is faithful by definition and is outside the frame.
- A violation the verifier cannot see (its known blind spots are in the top-level README) is by
  construction a residual one, which is what the blind labelling is for.

## The judge

`tailor judge` asks the model the verifier's six questions for one bullet, with a reason. It sees
the bullet, the facts it cites, the entry's scope note, DO NOT CLAIM list and tech list, and never
a verifier finding or a label. It uses `qwen3:8b` with `seed` 0. There is no paid API key, so there
is no second judge: the judge is the local model only.

1. **Measured first.** It is run on the verifier's holdout split (84 rows) and on the blind set (86
   rows), and its per-class precision and recall are reported next to the verifier's on the same
   rows (`judge-accuracy.json`).
2. **Residual check.** It is run on every final bullet of every completed run, each distinct bullet
   once (`judge-runs.json`). A bullet in the frame (see below) that the judge flags is a candidate
   residual violation.

The judge is a second reader whose accuracy is measured, not an authority: a flag is a reason to
look, and only a label counts as a violation.

## The blind sample and the residual rate

- **Frame:** the bullets the model rewrote and the verifier passed, in completed runs.
- **Strata:** every bullet in the frame that the judge flagged (or failed to judge) goes in whole.
  From the rest of the frame, a **random sample of 60** is drawn without replacement with
  `random.Random(20261121)`. The sample is the two together, shuffled with the same generator under
  opaque ids `s-001`, `s-002`, ... The run is a unit: a bullet that two runs share appears once per
  run.
- **What the labeller sees** (`blind-sample.jsonl`): the id, the bullet, the facts it cites (text
  only), the entry's scope note, its DO NOT CLAIM list and its tech list. No verifier finding, no
  judge verdict, no outcome, no seed, no posting. The id to run mapping is in
  `blind-sample-map.json`, a separate file.
- **Labels** (`blind-labels.jsonl`, committed by the labeller): one line per id,
  `{"id": "s-001", "labels": ["ok"]}`, labels as in `eval/verifier/README.md`: `ok`, or one or more
  violation classes. It is a blind labelling from the facts, as the existing blind set describes its
  own. The labels are committed separately and after the sample.
- **The rate:** the violating bullets among the flagged ones are counted as labelled. The share of
  violating bullets in the random 60 is extrapolated to the unflagged rest of the frame with a
  Wilson interval (95%). The residual estimate is the confirmed flagged plus the extrapolation; the
  rate is that over the frame. If the unflagged rest has 60 bullets or fewer, all are labelled and
  nothing is extrapolated. With 60 drawn, the interval is wide: a rate near zero still has an upper
  bound of several percent. The interval ignores the finite population (conservative) and treats
  the flagged stratum as exact.

## Invalid runs

A run is invalid when it crashes, or when the endpoint returns an error after the client's own
retries (the OpenAI SDK retries a connection error or a 5xx twice before it raises), or when a
model answer is not usable JSON for the schema. It is **recorded as failed, with its error, and not
dropped or re-run**: the results list every one, the scores count them (`runs.failed`), and the
README reports how many there were. The rates are over completed runs. A failed run is not
replaced by a retry, since a retry would select the runs that happen to work. If the failures are a
fault of the harness (not of the model or the machine) the fix is made, the results are discarded,
and all 36 runs are made again; that is recorded under "Deviations".

## Files

| File | What it is |
| --- | --- |
| `jds/` | the twelve postings |
| `halves.json` | the scores of runs 1 to 21 and of runs 22 to 36 (Deviation 1) |
| `results.json` | every run (per bullet: first rewrite's findings, outcome, final text), the meta (commit, tree state, model, Ollama version, machine) and the scores computed from the runs |
| `citations.json` | the fact ids the model copied into its bullets, by shape, counted from the runs |
| `cassettes/` | the model exchanges of runs 22 to 36 (157 files, 1.4 MB), for replay. Those of runs 1 to 21 were lost (Deviation 2): runs 22 to 36 replay, runs 1 to 21 cannot, and CI replays one (`ios-engineer`, seed 1) |
| `judge-accuracy.json` | the judge and the verifier per class on the holdout and the blind set |
| `judge-runs.json` | the judge's verdict on every final bullet |
| `blind-sample.jsonl`, `blind-sample-map.json` | the sample for labelling and its id mapping |
| `blind-labels.jsonl` | the labels |
| `blind-judgement-calls.md` | the 28 borderline rows of the labelling, each with the other reading |
| `residual.json` | the residual rate scored from the labels |
| `first-sample.jsonl`, `first-sample-map.json` | a blind sample of first rewrites for labelling (Deviation 7), and its id mapping |
| `first-labels.jsonl` | its labels |
| `first-judgement-calls.md` | the 27 borderline rows of that labelling, each with the other reading |
| `first-rewrites.json` | the first-rewrite violation rate, and the verifier against those labels |

```sh
uv run tailor eval pipeline --check      # scores, halves and citations are what the runs give (CI)
uv run tailor eval pipeline --replay --jd ios-engineer --replay-seed 1   # one run rebuilt from cassettes, no model (CI)
uv run tailor eval residual --check      # likewise for each scored file: judge, judge-runs, sample,
                                         # first-sample, residual, first-rewrites (CI)
```

## Results

Measured on 2026-11-21 and 2026-11-22 at commit 157434151cdad72d9fd7de2928c2a12bb4260028 (the code
that ran; the tree was clean). The figures in the measured passages below are values that the
scorer wrote into `results.json`, `halves.json`, `citations.json`, `judge-accuracy.json`,
`judge-runs.json`, `blind-sample-map.json`, `residual.json` and `first-rewrites.json`, not figures
computed by hand: each passage names the files it quotes (in a comment at its start), and
`scripts/readme_numbers.py` checks every figure in it against them. Each of those files is in turn
checked against the runs and labels it was computed from (`--check`). Two counts of the results
stand outside those passages, each next to the file it was counted from: how often the judge's
class matched the label, and what the other reading of one labelling call would change. The
deviations at the end quote versions, sizes and run numbers as recorded at the time; the ones that
are values of `results.json` or `first-rewrites.json` are measured passages too.

<!-- measured:begin runs
  eval/pipeline/results.json#scores.runs
  eval/pipeline/results.json#scores.bullets
  eval/pipeline/results.json#scores.totals
  eval/pipeline/results.json#scores.gate_off.structural
-->
**36 of 36 runs completed, none failed.** The model rewrote 442 bullets (the one `adaptable: false`
entry is a project without code, so it is never a candidate and no bullet was copied verbatim) in
359 model calls, using 988,610 input and 77,124 output tokens in all. The structural count (a
rewrite that cites no fact or a fact the entry does not have) was 1 bullet.
<!-- measured:end -->

### With the gate off and on

<!-- measured:begin gate
  eval/pipeline/results.json#scores.bullets.rewritten
  eval/pipeline/results.json#scores.gate_off
  eval/pipeline/results.json#scores.gate_on
  eval/pipeline/residual.json#confidence_percent
  eval/pipeline/residual.json#residual.frame
  eval/pipeline/residual.json#residual.estimate
  eval/pipeline/residual.json#residual.percent
  eval/pipeline/residual.json#residual.interval_percent
  eval/pipeline/citations.json#first_rewrites.fabricated_number_only_id_digits
  eval/pipeline/citations.json#first_rewrites.flagged_only_for_id_digits
-->

The gate-off side is the first rewrite the model gave for each bullet. The gate-on side is where
the bullet ended up. Rates are over the 442 rewritten bullets.

| | Bullets | Rate |
| --- | --- | --- |
| First rewrite flagged by the verifier (gate off) | 238 | 53.8% |
| Final bullet passed first time | 203 | 45.9% |
| Final bullet passed after one retry | 122 | 27.6% |
| Final bullet fell back to the bank's own sentence | 117 | 26.5% |

Every final bullet either passed the verifier or is the bank's own sentence, by construction. The
rate at which a passed bullet is still a violation is the residual rate, measured by the blind
labelling below: an estimated 29.4 of the 325 passed bullets (9.0%, 95% interval 5.4% to 17.3%).

By class (a first rewrite can carry more than one finding, so the classes add to more than 238):

| Class | First rewrites flagged | Rate |
| --- | --- | --- |
| fabricated_number | 107 | 24.2% |
| number_drift | 1 | 0.2% |
| fabricated_entity | 12 | 2.7% |
| do_not_claim | 3 | 0.7% |
| escalation_absolute | 8 | 1.8% |
| escalation_causal | 152 | 34.4% |
| any class | 238 | 53.8% |

Many `fabricated_number` flags are not invented numbers: in 65 of the 107, every number the
verifier flagged is the digit of a fact id the model copied into its text (the digit of
`[marlow-intern#4]`, read as a number no fact states; see the output defect below and
`citations.json`), and for 49 of the 238 flagged first rewrites that digit is the only finding.

<!-- measured:end -->

### By posting

<!-- measured:begin postings
  eval/pipeline/results.json#scores.by_jd.*.bullets.rewritten
  eval/pipeline/results.json#scores.by_jd.*.gate_off.percent
  eval/pipeline/results.json#scores.by_jd.*.gate_on.*.percent
  eval/pipeline/results.json#scores.by_jd.*.per_resume.seconds.median
-->

Three runs of each posting. The postings are listed in the table of the first section; the ones the
bank fits badly are `frontend-engineer`, `staff-backend-engineer`, `engineering-manager`,
`ios-engineer` and `embedded-firmware-engineer`.

| Posting | Bullets | Gate off: flagged | Passed first time | After retry | Fell back | Median s per resume |
| --- | --- | --- | --- | --- | --- | --- |
| backend-engineer | 35 | 51.4% | 48.6% | 28.6% | 22.9% | 1353 |
| data-platform-engineer | 39 | 69.2% | 30.8% | 30.8% | 38.5% | 2077 |
| database-internals-engineer | 42 | 38.1% | 61.9% | 19.0% | 19.0% | 1551 |
| distributed-systems-engineer | 47 | 36.2% | 63.8% | 23.4% | 12.8% | 1695 |
| embedded-firmware-engineer | 24 | 75.0% | 25.0% | 45.8% | 29.2% | 2507 |
| engineering-manager | 25 | 64.0% | 36.0% | 40.0% | 24.0% | 1999 |
| frontend-engineer | 33 | 33.3% | 66.7% | 18.2% | 15.2% | 2197 |
| ios-engineer | 24 | 79.2% | 20.8% | 50.0% | 29.2% | 3025 |
| llm-applications-engineer | 47 | 63.8% | 36.2% | 31.9% | 31.9% | 2411 |
| ml-platform-engineer | 48 | 60.4% | 39.6% | 22.9% | 37.5% | 2096 |
| site-reliability-engineer | 34 | 61.8% | 38.2% | 23.5% | 38.2% | 1739 |
| staff-backend-engineer | 44 | 36.4% | 61.4% | 18.2% | 20.5% | 1204 |

The postings the bank serves badly do not all have the highest rates: `frontend-engineer` (33.3%)
and `staff-backend-engineer` (36.4%) are among the lowest, `ios-engineer` (79.2%) and
`embedded-firmware-engineer` (75.0%) the highest, and `data-platform-engineer`, a posting the bank
fits well, is third (69.2%). With 24 to 48 bullets per posting these are small samples.

<!-- measured:end -->

### Time, load and tokens per resume

<!-- measured:begin time
  eval/pipeline/results.json#scores.per_resume
  eval/pipeline/results.json#scores.by_load
-->

Wall time is reported as measured under the load of the machine, not as the model's speed. Median
1,965 s per resume (range 933 to 3,851 s). The 1-minute load average at the start of a run had a
median of 1.69 (range 0.86 to 19.94). The 22 runs that started at a load below 2 had a median of
1,717 s, the 14 that started at 2 or more a median of 2,106 s.

Tokens per resume (the figure that does not depend on load): median 27,885 input (range 17,286 to
38,002) and 2,114 output (1,578 to 3,105), in a median of 10 calls (7 to 13). The cost is zero for
a local model.

<!-- measured:end -->

### Before and after the operating-system update

<!-- measured:begin halves
  eval/pipeline/halves.json#split_after
  eval/pipeline/halves.json#*.runs.total
  eval/pipeline/halves.json#*.bullets.rewritten
  eval/pipeline/halves.json#*.gate_off.percent
  eval/pipeline/halves.json#*.gate_off.by_class.*.percent
  eval/pipeline/halves.json#*.gate_on.*.percent
  eval/pipeline/halves.json#*.per_resume.seconds.median
-->
The first 21 runs and the other 15, before and after the update (see the first deviation), from
`halves.json`:

| | Bullets | Gate off: flagged | fabricated_number | escalation_causal | Passed first time | After retry | Fell back | Median s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| First 21 runs | 245 | 50.2% | 19.2% | 33.9% | 49.8% | 27.8% | 22.4% | 1999 |
| Other 15 runs | 197 | 58.4% | 30.5% | 35.0% | 41.1% | 27.4% | 31.5% | 1932 |

The runs are in posting order, so the first half is the first seven postings (backend-engineer to
frontend-engineer) and the second half the last five (ios-engineer to staff-backend-engineer). A
difference between the halves is therefore a difference between postings as much as between
machines: it cannot be put down to the update.

<!-- measured:end -->

### The judge

<!-- measured:begin judge-holdout
  eval/pipeline/judge-accuracy.json#meta.usage
  eval/pipeline/judge-accuracy.json#meta.judge_seed
  eval/pipeline/judge-accuracy.json#datasets.holdout.rows
  eval/pipeline/judge-accuracy.json#datasets.holdout.verifier.violation_rows
  eval/pipeline/judge-accuracy.json#datasets.holdout.verifier.ok.support
  eval/pipeline/judge-accuracy.json#datasets.holdout.*.classes
-->
`qwen3:8b`, seed 0, 170 judge calls with no failures (250,417 input and 15,430 output tokens).
Precision / recall per class, the judge next to the verifier on the same rows. Holdout, 84 rows
(57 with a violation, 27 without):

| Class | Support | Judge | Verifier |
| --- | --- | --- | --- |
| fabricated_number | 5 | 1.00 / 0.60 | 1.00 / 1.00 |
| number_drift | 9 | 0.53 / 1.00 | 1.00 / 1.00 |
| fabricated_entity | 13 | 0.75 / 0.92 | 1.00 / 1.00 |
| do_not_claim | 18 | 0.75 / 0.67 | 1.00 / 0.67 |
| escalation_absolute | 10 | 1.00 / 0.50 | 1.00 / 1.00 |
| escalation_causal | 7 | 1.00 / 0.57 | 1.00 / 1.00 |
<!-- measured:end -->

<!-- measured:begin judge-blind
  eval/pipeline/judge-accuracy.json#datasets.blind.rows
  eval/pipeline/judge-accuracy.json#datasets.blind.verifier.violation_rows
  eval/pipeline/judge-accuracy.json#datasets.blind.verifier.ok.support
  eval/pipeline/judge-accuracy.json#datasets.blind.*.classes
-->
Blind set, 86 rows (59 with a violation, 27 without):

| Class | Support | Judge | Verifier |
| --- | --- | --- | --- |
| fabricated_number | 12 | 0.60 / 0.75 | 0.48 / 0.83 |
| number_drift | 9 | 0.27 / 0.78 | 0.89 / 0.89 |
| fabricated_entity | 10 | 0.75 / 0.90 | 1.00 / 1.00 |
| do_not_claim | 16 | 0.73 / 0.69 | 1.00 / 0.31 |
| escalation_absolute | 13 | 0.90 / 0.69 | 0.73 / 0.85 |
| escalation_causal | 11 | 0.88 / 0.64 | 0.50 / 0.64 |
<!-- measured:end -->

<!-- measured:begin judge-overall
  eval/pipeline/judge-accuracy.json#datasets.*.judge
  eval/pipeline/judge-accuracy.json#datasets.*.verifier
-->
Any violation caught: holdout, judge 55 of 57 (0.96), verifier 52 of 57 (0.91); blind set, judge 58
of 59 (0.98), verifier 51 of 59 (0.86). Clean rows flagged: holdout 0 of 27 for both; blind set,
judge 13 of 27 (48.1%), verifier 11 of 27 (40.7%). The two read differently. On the blind set the
judge finds far more of the `do_not_claim` rows (recall 0.69 against 0.31; on the holdout they tie
at 0.67) and has the higher `fabricated_number` precision (0.60 against 0.48); the verifier has the
better recall on `escalation_absolute` on both sets (1.00 against 0.50 on the holdout, 0.85 against
0.69 on the blind set) and, on the holdout, on every class except `number_drift` and `do_not_claim`,
where they tie. The judge's `number_drift` precision is low on both sets (0.53 and 0.27), for
different reasons: on the holdout none of its wrong `number_drift` flags is on a faithful row (each
names the wrong class on a violating row, most often a `do_not_claim` one), while on the blind set
about half of them are on faithful rows.
<!-- measured:end -->

<!-- measured:begin judge-runs
  eval/pipeline/judge-runs.json#summary
  eval/pipeline/blind-sample-map.json#strata.flagged
-->
The judge over the final bullets of the runs (`judge-runs.json`): 442 judged, none failed, **40
flagged** (do_not_claim 16, number_drift 12, fabricated_entity 8, escalation_causal 4,
fabricated_number 2; a bullet can carry two). Of the 40, 34 are in the frame (24 passed first time,
10 after a retry) and 6 are bank fallbacks. These are the judge's flags, not confirmed violations:
given the false-alarm rate on the blind set above, a share of them will be wrong.
<!-- measured:end -->

### The blind sample and the residual rate

<!-- measured:begin sample
  eval/pipeline/results.json#scores.gate_on.first_pass.bullets
  eval/pipeline/results.json#scores.gate_on.after_retry.bullets
  eval/pipeline/blind-sample-map.json#seed
  eval/pipeline/blind-sample-map.json#frame
  eval/pipeline/blind-sample-map.json#strata
  eval/pipeline/residual.json#labelled.rows
  eval/pipeline/residual.json#judgement_calls
-->
The frame is the 325 bullets the model rewrote and the verifier passed (203 first time, 122 after
a retry). The sample has 94 rows: the 34 the judge flagged, and 60 drawn at random (seed
20261121) from the 291 it did not. The labels are in `blind-labels.jsonl`; the 28 borderline rows,
each with the other reading, are in `blind-judgement-calls.md`.
<!-- measured:end -->

No label was changed after it was committed.

**How the labelling was done.** It is a blind labelling from the facts, as the verifier's blind
set describes its own. The labeller was given the label definitions (the top-level README's
table), the verifier blind set's README and judgement calls, and the sample (the bullet, the facts
it cites, the entry's scope note, its DO NOT CLAIM list and its tech list, with no verifier
finding, judge verdict, outcome, seed or posting). Reading the top-level README for the
definitions, the labeller also saw its aggregate counts, but no verdict on any row. Two rules were
settled by the labeller while labelling, and they move the class counts: a trailing clause that
claims a use no fact mentions is labelled `escalation_causal`, and placing a personal project in a
payment-processing platform is labelled `do_not_claim`. Both are in `blind-judgement-calls.md`.

<!-- measured:begin residual
  eval/pipeline/results.json#scores.runs.total
  eval/pipeline/blind-sample-map.json#strata
  eval/pipeline/residual.json
-->
**Labels.** 14 of the 94 rows carry a violation (one carries two labels); the rest are `ok`. Counts
by class: `do_not_claim` 7, `escalation_causal` 6, `fabricated_entity` 1, `escalation_absolute` 1,
`fabricated_number` 0, `number_drift` 0. The violations are from the two strata as follows: 4 of the
60 random rows (three `escalation_causal`, one `do_not_claim`) and 10 of the 34 flagged rows (six
`do_not_claim`, three `escalation_causal`, one `fabricated_entity`, one `escalation_absolute`).

**The residual rate.** The 10 confirmed flagged bullets are counted as they are. The 4 of 60 in
the random draw (6.7%, Wilson 95% interval 2.6% to 15.9%) is extrapolated to the 291 unflagged
bullets: 19.4 more, so an estimated 29.4 violating bullets among the 325 the verifier passed,
that is **9.0%, 95% interval 5.4% to 17.3%** (17.6 to 56.3 bullets). Over all 442 final bullets
of the 36 runs (the fallbacks being the bank's own sentences, faithful by definition) that is 6.65%.
The interval ignores the finite population, which makes it conservative, and treats the flagged
stratum as exact. With 60 bullets drawn the interval is wide.

**The judge on this sample.** Of its 34 flags in the frame the labels confirm 10 as violating
(29.4%) and call the rest `ok`. The estimated share of all violating bullets in the frame that the
judge flagged is 34.0%: 10 of 29.4.
<!-- measured:end -->

In 4 of the 10 the judge's class is among the labelled classes; in the other 6 it named a
different class from the label (for example `number_drift` or `fabricated_entity` on a bullet
labelled `do_not_claim`).

<!-- measured:begin citations
  eval/pipeline/citations.json
  eval/pipeline/results.json#protocol.seeds
-->
**An output defect the verifier passed.** The model sometimes copied a fact id from the prompt into
the bullet's text instead of its `sources` field (`citations.json`). 76 first rewrites did, in five
shapes: `[id]` in 38, `(id)` in 21, `(sources: [id])` in 7, `(fact: id)` in 7 and `(Cites id)` in
3. The verifier rejected all but 4 of them, mostly because it read the digit of an `entry#n` id as
a number no fact states. The 4 it passed are first-pass bullets of one run
(`distributed-systems-engineer`, seed 1, the Raftlet entry) ending in `(sources: [raftlet-...])`,
whose explicit fact ids carry no digit, and they are the 4 final bullets with a copied id; no
other shape reached a final bullet. Two are in the labelled sample (s-044 and s-048) and are
labelled `ok`, since the text is formatting and not a claim. This is a defect of the pipeline's
output, not a violation class. It was fixed after these runs, which predate the fix: the gate now
refuses a bullet that the verifier passes when its text carries a fact id of its entry, in any of
these shapes, so it is retried once and otherwise falls back. Such a refusal is recorded as its own
kind, `copied_citation`: not a verifier finding (the verifier passed the bullet) and not
structural (it reached the verifier). These runs have none.
<!-- measured:end -->

### First rewrites, labelled

<!-- measured:begin first-sample
  eval/pipeline/first-rewrites.json#seed
  eval/pipeline/first-rewrites.json#frame
  eval/pipeline/first-rewrites.json#sampled
  eval/pipeline/first-rewrites.json#labelled
  eval/pipeline/first-rewrites.json#judgement_calls
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.rows
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.same_labels
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.ok
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.violating
  eval/pipeline/first-rewrites.json#without_overlap.rows
  eval/pipeline/first-rewrites.json#without_overlap.violating
  eval/pipeline/first-rewrites.json#without_overlap.percent
  eval/pipeline/first-rewrites.json#without_overlap.interval_percent
  eval/pipeline/first-rewrites.json#without_overlap.gate_off_flag.flagged_violating
  eval/pipeline/first-rewrites.json#confidence_percent
-->
To measure the first rewrites themselves, and not only the verifier's flags on them, 60 of the 441
first rewrites that reached the verifier were drawn at random (seed 20261122; see the deviations)
and labelled from the facts, with the same definitions, precedents and reading as the sample of
passed bullets, and without the verifier's findings. 19 of the 60 carry a violation:
`escalation_causal` 10, `fabricated_number` 5, `fabricated_entity` 4, `do_not_claim` 4,
`escalation_absolute` 2, `number_drift` 0. The labels are in `first-labels.jsonl`, and the 27
borderline rows, each with the other reading, in `first-judgement-calls.md`.

**Not blind to the verdict for 13 rows.** 13 of the 60 are first-pass bullets (the first rewrite is
the final bullet) that are also rows of the sample of passed bullets, identical in text and cited
facts, and the same labeller labelled both samples. For those 13 the labeller could recognise a
bullet it had labelled as a passed one, and so know that the verifier had passed it. Their labels
are the ones the same bullets got in the passed sample, 13 of 13 (12 `ok`, 1 violating), given
before any first rewrite was seen. Without them, 18 of the other 47 carry a violation: 38.3% (95%
interval 25.8% to 52.6%), and the verifier flagged 16 of those 18 (`first-rewrites.json`,
`without_overlap`).
<!-- measured:end -->

One call is new to this labelling: a number taken only from a scope
note is labelled `fabricated_number` (f-053, f-056), since a scope is context and not evidence of a
result; the other reading, `ok`, would move `fabricated_number` from 5 to 3 and the violating rows
from 19 to 17. Two rows cite a fact that does not match the bullet, whose own text names another
fact id (f-009, f-053); they are labelled from the row as given.

<!-- measured:begin gate-effect
  eval/pipeline/first-rewrites.json#confidence_percent
  eval/pipeline/first-rewrites.json#frame
  eval/pipeline/first-rewrites.json#sampled
  eval/pipeline/first-rewrites.json#violation_rate
  eval/pipeline/residual.json#labelled.rows
  eval/pipeline/residual.json#residual.frame
  eval/pipeline/residual.json#residual.estimate
  eval/pipeline/residual.json#residual.percent
  eval/pipeline/residual.json#residual.interval_percent
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.rows
-->
**What the gate changed**, by the two labellings (95% intervals; the passed bullets' rate is
estimated from a stratified sample, as described above; the first-rewrite labelling was not blind
to the verdict for 13 rows, as said above):

| | Labelled | Violating | Rate |
| --- | --- | --- | --- |
| First rewrites (what the resume would hold with the gate off) | 60 of 441 | 19 | 31.7% (21.3% to 44.2%) |
| Rewritten bullets that passed the verifier (gate on) | 94 of 325 | 29.4 (estimated) | 9.0% (5.4% to 17.3%) |
<!-- measured:end -->

<!-- measured:begin first-verifier
  eval/pipeline/first-rewrites.json#gate_off_flag
  eval/pipeline/first-rewrites.json#verifier
  eval/pipeline/first-rewrites.json#violation_rate.percent
  eval/pipeline/results.json#scores.gate_off.percent
-->
**The verifier against these labels.** As the gate uses it, a finding of any class refuses the
rewrite. It flagged 28 of the 60: 16 of them are labelled violating (precision 0.57), and it
flagged 16 of the 19 violating rewrites (recall 0.84). It flagged 12 of the 41 rewrites labelled
`ok` (29.3%), and 3 of the 19 violations are among the 32 rewrites it passed. A flagged rewrite's
labelled class was among its findings in 13 cases (0.46 of the flags, 0.68 of the violations). Per
class (a class with no labelled row has no recall):

| Class | Support | Flagged | Precision | Recall |
| --- | --- | --- | --- | --- |
| escalation_causal | 10 | 17 | 0.24 | 0.40 |
| fabricated_number | 5 | 12 | 0.42 | 1.00 |
| fabricated_entity | 4 | 3 | 1.00 | 0.75 |
| do_not_claim | 4 | 1 | 1.00 | 0.25 |
| escalation_absolute | 2 | 1 | 0.00 | 0.00 |
| number_drift | 0 | 1 | 0.00 | n/a |

On these rewrites the gate-off flag rate of 53.8% therefore overstates how often the first
rewrite was wrong (31.7% labelled): most of the verifier's `escalation_causal` and
`fabricated_number` flags are on rewrites not labelled so (precision 0.24 and 0.42).
<!-- measured:end -->

### What these numbers do not show

- One small local model. Its absolute rates are for it; the verifier's job does not depend on the
  model's quality, which is why the same gate is measured on a model that gets it wrong often. A
  larger model would give different rates, and this is the only model measured (there is no paid
  API key, so no second model or second judge).
- The postings are fictional, written for this evaluation, and the bank is invented. Rates on real
  postings and a real bank are not measured here.
- Three seeds of a posting are repeats of one model, not independent samples, and 24 to 48 bullets
  per posting are few.
<!-- measured:begin limits
  eval/pipeline/results.json#scores.gate_off.percent
  eval/pipeline/first-rewrites.json#confidence_percent
  eval/pipeline/first-rewrites.json#violation_rate.percent
  eval/pipeline/first-rewrites.json#violation_rate.interval_percent
-->
- The gate-off figures (53.8% in all, and the rates by class) are the verifier's flag rates, not
  violation rates: on the labelled sample of first rewrites they overstate the violation rate (31.7%
  labelled, 95% interval 21.3% to 44.2%). The passed bullets are only as good as the verifier: that
  is what the blind labelling of them measures.
<!-- measured:end -->
<!-- measured:begin labellings
  eval/pipeline/residual.json#labelled.rows
  eval/pipeline/residual.json#judgement_calls
  eval/pipeline/residual.json#residual.interval_percent
  eval/pipeline/first-rewrites.json#sampled
  eval/pipeline/first-rewrites.json#judgement_calls
  eval/pipeline/first-rewrites.json#violation_rate.interval_percent
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.rows
-->
- Each rate rests on a single labelling by one labeller, of 94 passed bullets and of 60 first
  rewrites, and both intervals are wide (5.4% to 17.3%, and 21.3% to 44.2%). The first was blind
  to the verifier's verdict; the second was not for 13 rows that repeat bullets of the first (see
  "First rewrites, labelled"). 28 and 27 rows were borderline and are listed with the other
  reading in `blind-judgement-calls.md` and `first-judgement-calls.md`. Three labelling rules move
  the class counts: a use no fact mentions is `escalation_causal`; a personal project placed in a
  payment platform is `do_not_claim`; a number taken only from a scope note is `fabricated_number`
  (the first-rewrite labelling only).
<!-- measured:end -->
- The wall times are measured under the load of a machine doing other work (Deviations 1 and 3).

## Deviations

1. **The machine was upgraded mid-run.** Runs 1 to 21 ran on macOS 26.6.2. The machine then
   rebooted for an operating-system update to macOS 27.2.1 (build 26B131), and runs 22 to 36 ran
   there after `--resume`. Unchanged across the update: Ollama 0.23.4, `qwen3:8b` (500a1f067a9f),
   the context length (8192), the repository commit and the harness. `meta.machine.platform` in
   `results.json` records only the platform at the start. Neither part was dropped or re-run; the
   headline rates are given for all runs and for each part (`halves.json`). The split by runs is
   also a split by posting (see "Before and after the operating-system update").
2. **The cassettes of runs 1 to 21 were lost.** They were kept in a scratch directory under the
   system's temporary folder, which the reboot cleared. Their results are in `results.json`. The
   committed cassettes cover runs 22 to 36 (157 files); `--replay` in CI uses one of those runs
   (`ios-engineer`, seed 1), and all fifteen were replayed to the committed records and scores
   when they were committed.
<!-- measured:begin deviations
  eval/pipeline/results.json#protocol.seeds
  eval/pipeline/results.json#runs.0.wall_seconds
  eval/pipeline/results.json#runs.0.load_average_1m
  eval/pipeline/results.json#runs.21.load_average_1m
-->
3. **The first resumed run started at a 1-minute load near 20** (`ios-engineer`, seed 1: 19.94),
   from the post-update background indexing. Each run's load is recorded as before.
4. **The run was stopped after its first run and resumed.** The first run took 1,844 s under
   load 4.19 from other work, a pace at which the whole matrix would have taken most of a day, so
   the pipeline was stopped (not cut) and the other runs were made later with `--resume` on a
   quieter machine, at the same commit. The 3-seed protocol was kept.
<!-- measured:end -->
5. **The judge cassettes (2.9 MB) are not committed.** The judge's verdicts are in
   `judge-accuracy.json` and `judge-runs.json`; the judge calls were recorded outside the
   repository.
6. **Two statements of the protocol are not backed by the record.** The model build
   `500a1f067a9f` comes from the local Ollama manifest of `qwen3:8b` (the start of its sha256),
   read after the runs; `results.json` records only the tag and the Ollama version. Runs made
   after this record also store the digest (`meta.model_digest`). And "a seed makes a run
   repeatable on this machine" was not measured: no run was repeated live. What is recorded is
   that the seed was sent with every call (`meta.seed_applied`), and that every run with
   cassettes replays to its committed record.
<!-- measured:begin deviation-7
  eval/pipeline/first-rewrites.json#seed
  eval/pipeline/first-rewrites.json#frame
  eval/pipeline/first-rewrites.json#sampled
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.rows
  eval/pipeline/first-rewrites.json#overlap_with_passed_sample.same_labels
-->
7. **A labelled sample of first rewrites was added after the results.** The protocol labelled
   passed bullets only, while the design asks that the first-rewrite violation rate be measured
   by the verifier and partly reviewed by hand on a sample. After a review of this record, 60
   first rewrites were drawn at random (seed 20261122) from the 441 that reached the verifier
   (the run as the unit; the structural one is left out), and exported as the residual sample was,
   with no verifier finding, outcome, seed or posting (`first-sample.jsonl`, by `tailor eval
   first-sample --seed 20261122`). The frame was not made to exclude the bullets of the passed
   sample, and 13 of the 60 are such bullets, which the same labeller had labelled as passed
   ones: for those, the labelling was not blind to the verdict. Their labels equal the earlier
   ones, 13 of 13, and the scores are also given without them. The sample was labelled the same
   way (`first-labels.jsonl`, `first-judgement-calls.md`) and scored by `tailor eval
   first-rewrites` (`first-rewrites.json`); the results are under "First rewrites, labelled".
<!-- measured:end -->
