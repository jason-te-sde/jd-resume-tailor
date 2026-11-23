# The blind set

<!-- measured:begin rows eval/verifier/blind/results.json#rows eval/verifier/blind/results.json#judgement_calls -->
`rows.jsonl` holds 86 labelled bullets over the invented bank, written by an author who was given
only the bank and the label definitions in the top-level README: not the verifier's code, its
word lists, its tests or the main dataset. The author was asked to vary the phrasing on purpose
(spelled-out numbers, other units, paraphrased limits, claims in subordinate clauses), and to keep
the rows that were judgement calls rather than drop them; `judgement-calls.md` lists those, with
the author's reasoning: 14, of which 13 are rows of the set (the other is a row left out, see
below).
<!-- measured:end -->

The set was evaluated once, at commit 492a3fa, and `results.json` records that run under
`first_run`. It is not used to tune the verifier, and it is not part of the CI gate: once rules are
fitted to it, it stops measuring anything. A later change can be scored against it, and the
difference reported, but a fresh blind set is the honest way to measure the next version.

<!-- measured:begin rescored
  eval/verifier/history.json#blind_before
  eval/verifier/history.json#blind_after
-->
**Re-scored, not re-labelled.** After an independent review of the verifier (percent reductions,
number-to-claim binding, scope facts, `production` claims) the set was scored again with the new
rules, at bfc57d9, and `results.json` records that under `rescored`. No label was changed and no
rule was fitted to these rows. The only edit to the rows is that the figures of the invented bank
were changed (the same substitution in the bank, the main dataset and here, in the rows' sources
and bullets), which does not change what the verifier says about any row; that was checked row by
row. The numbers moved for the worse on false alarms (8 of 27 faithful rows flagged, now 11); see
the top-level README.
<!-- measured:end -->

To reproduce them, without writing anything:

```sh
uv run tailor eval verifier --data eval/verifier/blind --score-only
```

## How the rows were prepared

The author copied sources from the bank's raw JSON, with its LaTeX escapes (`\%`,
`$\rightarrow$`), while the verifier compares plain text. Before the run, each source was passed
through the bank's own `normalize_latex`, and a source that was a fragment of exactly one fact was
replaced by that whole fact (5 sources). Four rows whose sources matched no fact of their entry
were left out (`raftlet`, whose explicit `facts` replace its bullets as sources). No bullet and no
label was changed.
