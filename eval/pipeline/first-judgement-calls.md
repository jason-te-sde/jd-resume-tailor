# Judgement calls

Labelled with the same reading as the first sample; 's-' ids refer to the calls recorded for it.

- f-003 (ok): Same call as s-029 and s-048 (the bullet is identical to s-048): 'covering leader crashes, partitions and message loss' unpacks 'controls the network and crashes', and the trailing '(sources: ...)' is an artefact, not a claim.
- f-006 (ok): 'using Docker' ties a stack technology to the CI run, which the fact does not do; it does not make the claim stronger. Same call as s-060.
- f-007 (escalation_causal): 'in a regression suite for improved evaluation of agent behavior' claims a use no fact mentions; the model is a ticket classifier. Same rule as s-018; a reader could pass it as posting vocabulary.
- f-008 (do_not_claim): 'using Docker for deployment' claims a deployment for a project that DO NOT CLAIM says has only run on test machines. The item names production use rather than deployment, and Docker is in the stack, so a reader could take it as packaging and call it ok.
- f-009 (fabricated_number, escalation_causal): The cited fact is the crash-injection fact, so 3.08x, 91 s, 29.5 s and '4 tables at a time instead of 2' have no source, and 'through merging 4 tables' is a mechanism no source states. The bullet's own '(fact: tidekv#4)' suggests a mis-cited fact; I labelled the row as given, and a reader could keep only fabricated_number since the whole claim is unsupported.
- f-013 (escalation_absolute, escalation_causal): 'ensuring data consistency' turns one recovery measurement into a guarantee (as s-027), and 'across services via recovery' names a use the embedded store never had and ties it to the recovery (as s-013). A reader could file the guarantee under the 'durability guarantees' DO NOT CLAIM item instead.
- f-015 (ok): 'improving form completion from 61% to 74%' makes the rebuild the cause where the fact uses a semicolon; I accepted it because the rise was measured in an A/B test of the rebuilt form, which is what establishes the cause. That is the difference from s-059 and s-067.
- f-016 (fabricated_entity): Swift is clear. I did not label 'real-time' (not on this entry's list) or 'ensuring reliable data handling in high-throughput environments' (deduplication by event id backs 'reliable'); a stricter reader could add escalation_absolute.
- f-018 (fabricated_entity, do_not_claim): 'Instruments' is not in the stack, and '41,000 requests per second with low latency' is the latency-bounded reading DO NOT CLAIM rules out. 'Optimized ... to handle' implies an optimization no fact states; I did not add escalation_causal for a verb with no mechanism named.
- f-021 (ok): 'ensuring consistent state across concurrent processes' restates the single atomic step in other words. Because it carries no condition, a reader could take it as the 'strict global limits while Redis is down' item.
- f-024 (fabricated_number, escalation_causal): 69% is derived from 31 to 9 (about 71%), and a derived number counts as fabricated; it understates, so a reader could pass the number. 'reducing' ties the drop to the alerts (as s-059).
- f-025 (escalation_absolute): 'ensuring consistent and timely data integration' asserts a guarantee no fact makes (as s-027). 'data integration' describes the pipeline itself rather than an outside use, so I did not add escalation_causal (as s-069); a reader could.
- f-029 (ok): Same call as s-069; the bullet is identical.
- f-031 (ok): Same call as s-054.
- f-032 (ok): 'to ensure reliable data handling' states the purpose of the deduplication the fact describes; unlike s-027, what is ensured follows from a mechanism, not from a test result. A stricter reader would label escalation_absolute.
- f-036 (fabricated_entity, escalation_causal): VoiceOver and Dynamic Type are clear. 'achieving' a score of 96 is the s-067 call.
- f-037 (ok): 'Led a 4-person team' for 'Led a 4-person effort': the fact states the leading, so the claim has a fact behind it, though 'team' can suggest people management that 'effort' does not. 'eliminating up to 5 hours' keeps 'up to'.
- f-038 (do_not_claim, escalation_causal): Same bullet and call as s-007.
- f-041 (ok): 'Improved scenario success rate by passing 35 of 40' has a 'by' clause that restates the result, and the improvement over 22 of 40 without the tool is in the fact.
- f-043 (escalation_causal): 'reducing' ties the drop to the alerts (as s-059). 'on-call incident frequency' renames pages as incidents; I read it as on-call load, not a different count, though a reader could call incidents a stronger claim than pages.
- f-047 (ok): Same call as s-029; the bullet is identical.
- f-050 (ok): 'ensuring accurate rate enforcement for system reliability' restates the within-2% result inside the same '10-minute tests' context, and 'for system reliability' is a purpose gloss (as s-082). A stricter reader would label escalation_absolute as in s-027.
- f-053 (fabricated_number): The cited fact is the compaction fact; '200 crash-injection tests' and 'no acknowledged writes lost' appear only in the scope, which is context, not evidence of a result. The claim is true per the scope and the '(fact: tidekv#3)' tag suggests a mis-cited fact, so a reader could call it ok.
- f-055 (escalation_causal): Same call as s-067.
- f-056 (fabricated_number): '37 of 40 scenarios using the retrieval tool' comes from the scope, not a cited fact. It is true and not flattering, so a reader could call it ok as provenance; I applied the scope rule as for f-053.
- f-057 (fabricated_entity): 'as part of the component library' names a thing the entry never mentions, as 'storage engine' did in s-022; it is not a technology, so a reader could put it outside the six classes. 'leading to an increase' I accepted for the A/B reason in f-015.
- f-060 (ok): Same call as s-088.
