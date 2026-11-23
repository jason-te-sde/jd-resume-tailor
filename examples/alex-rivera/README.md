# Alex Rivera: an invented example

Everything in this directory is invented. Alex Rivera is not a real person, and the employers,
school, projects, repositories, measurements and contact details are made up to exercise the tool.
Nothing in it describes real work, and any resemblance to a real person, company or project is a
coincidence.

- `bank.json` is a fact bank: 3 work entries and 10 projects in backend, distributed systems, data,
  machine learning and frontend engineering. Each entry has bullets with numbers, a `scope` note
  with provenance (`EVIDENCE:`, `ORIGIN:`) and, for every entry, a `DO NOT CLAIM:` list of what the
  bullets must not be stretched into. One entry carries structured `facts`, one pair of entries is
  the same project (`Same project as`), and two projects have `has_code` set to false.
- `profile.json` holds the contact details and education of the invented person. Every address
  uses `example.com` and the phone number is in the 555-01xx range that North American numbering reserves for fiction.

- `jds/` holds three short invented job descriptions (a backend role, a data-platform role and a
  distributed-systems role) to run `tailor plan` on. The companies do not exist.
- `cassettes/` and `plans/` hold a recorded run of `tailor plan` on `jds/backend-engineer.txt`
  against a local model, and the plan and provenance report it produced. They let CI and anyone
  else replay the run without a model; see CONTRIBUTING.md before changing a prompt.

- `out/` holds the output of `tailor run` on `jds/backend-engineer.txt`, replayed from the
  cassettes: the LaTeX, the PDF (one page, nothing dropped), the provenance report and a PNG of the
  top half of the page.

Check the bank:

```sh
uv run tailor bank check examples/alex-rivera/bank.json
```
