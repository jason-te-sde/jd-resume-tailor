# Judgement calls

- blind-003 (escalation_absolute): The source gives a minimum of 58 fps in 60 s runs on one laptop; 'never ... any machine' generalises. Could also be argued do_not_claim (browsers), but I kept it to the absolute.
- blind-010 (ok): 'every failing seed' paraphrases 'each failing seed'.
- blind-018 (fabricated_entity): Playwright is real but absent from this entry's stack (Cypress is there), so it is a plausible-looking fabricated entity.
- blind-020 (do_not_claim, escalation_absolute): 'exactly-once' is both on the DO NOT CLAIM list and in the absolute list; both labels apply.
- blind-028 (fabricated_number): p95 is a percentile label rather than a measured quantity; I count it as a number with no source match. A checker may reasonably treat percentile names as non-numbers.
- blind-036 (escalation_absolute, do_not_claim): 'every browser' breaks the Chrome/Firefox-only note; 'guarantees' is absolute.
- blind-041 (ok): 'every' here restates 'a check and its update are a single atomic step'; I consider it supported.
- blind-043 (escalation_causal): Plausible mechanism, but the sources never tie the latency result to the Lua script.
- blind-046 (do_not_claim): Paraphrase of 'distributed locking'; the source says the guarantee comes from one PostgreSQL instance.
- blind-047 (fabricated_number): 412/138 is about 3x, so 'fourfold' has no source; I labelled fabricated rather than drift because it is not a rounding of any stated number.
- blind-052 (do_not_claim): Paraphrase of 'zero downtime'. 'Without any' is also absolute-flavoured, but I labelled only do_not_claim since the DO NOT CLAIM list covers it directly.
- blind-059 (fabricated_number): 0.09 s = 90 ms, no match for 138 ms in any rounding; a unit-converted fabrication.
- blind-060 (escalation_absolute): Replaces the 2% threshold with zero tolerance; I read it as an absolute, though it is also arguably a changed number.
- blind-072 (ok): Converts 138 ms to 0.14 s (seconds, rounded up for a lower-is-better value); I consider it allowed.
