# Provenance report

Job: Backend Engineer, Shipment Tracking at Harbor & Pine Logistics (examples/alex-rivera/jds/backend-engineer.txt).
Model: qwen3:8b (openai); rounding: conservative; 10 calls, 24406 input and 1470 output tokens.

10 bullets: 5 passed the verifier first time, 4 after one retry, 1 fell back to the bank's wording, 0 copied verbatim (adaptable: false).

## Software Engineer (`lantern-swe`)

Coverage score 10/10: Candidates directly addressed all responsibilities with Go, Kafka, PostgreSQL, and latency optimization.

1. Reduced the p99 latency of the shipment-tracking API in staging load tests from 412 ms to 138 ms by replacing a per-request fan-out of 7 queries with a single batched query and implementing a Redis read-through cache.

   - outcome: first_pass; verdict: ok
   - sources: lantern-swe#1
   - bank wording: Cut p99 latency of the shipment-tracking API from 412 ms to 138 ms by replacing a per-request fan-out of 7 queries with one batched query and adding a Redis read-through cache.

2. Designed an idempotent Kafka consumer for carrier status events that deduplicates by event ID, achieving 9,200 events/s throughput in staging load tests on a 6-partition topic.

   - outcome: first_pass; verdict: ok
   - sources: lantern-swe#2
   - bank wording: Built an idempotent Kafka consumer for carrier status events that deduplicates by event id and sustained 9,200 events/s on a 6-partition topic in staging load tests.

3. Added 23 Prometheus alerts and runbooks for the dispatch service; pages per on-call week fell from 31 to 9 over two quarters.

   - outcome: fallback; verdict: bank_original
   - sources: lantern-swe#4
   - why the bank wording was used: first attempt: escalation_causal: 'reducing on-call pages per week from 31 to 9 over two quarters' - the cited sources do not link 'reducing on-call pages per week from 31 to 9 over two quarters' to the result in one statement. second attempt: escalation_causal: 'with pages per on-call week decreasing from 31 to 9 over two quarters' - the cited sources do not link 'with pages per on-call week decreasing from 31 to 9 over two quarters' to the result in one statement

4. Led a team to replace the nightly billing-export cron job with an event-driven pipeline, removing up to 5 hours of data staleness daily between events and exports.

   - outcome: after_retry; verdict: ok
   - sources: lantern-swe#5
   - bank wording: Led a 4-person effort to replace the nightly billing-export cron job with an event-driven job, removing up to 5 hours of data staleness each day.
   - first attempt rejected: escalation_absolute: 'eliminating' - 'eliminating' is an absolute; none of the cited sources says it; escalation_causal: 'eliminating up to 5 hours of data staleness daily between events and exports' - the cited sources do not link 'eliminating up to 5 hours of data staleness daily between events and exports' to the result in one statement

## Data Engineering Intern (`marlow-intern`)

Coverage score 0/10: Experienced in Python/data engineering tools, not backend Go/Kafka.

1. Designed event-driven data pipelines using Airflow and dbt to automate report generation, replacing manual processes and ensuring data consistency for tracking dashboards.

   - outcome: first_pass; verdict: ok
   - sources: marlow-intern#1, marlow-intern#scope
   - bank wording: Built 12 dbt models and an Airflow DAG that replaced a manual weekly spreadsheet report, saving about 6 analyst-hours a week.

2. Reduced the nightly warehouse load time from 74 to 29 minutes by partitioning a 310 GB fact table and loading only the changed partitions.

   - outcome: after_retry; verdict: ok
   - sources: marlow-intern#2
   - bank wording: Cut the nightly warehouse load from 74 to 29 minutes by partitioning a 310 GB fact table by date and loading only the changed partitions.
   - first attempt rejected: fabricated_entity: 'PostgreSQL' - the technology 'PostgreSQL' appears in none of the cited facts and not in the entry's tech stack; escalation_causal: 'by partitioning a 310 GB fact table' - the cited sources do not link 'by partitioning a 310 GB fact table' to the result in one statement

## Dispatch Desk: a tool-using claim-triage agent (`dispatch-desk`)

Coverage score 0/10: Python and lightweight tools, lacks Go/Kafka backend stack.

1. Developed a Python-based tool that triages freight-damage claims using an Ollama model served locally and a retrieval tool querying a SQLite table of past claims.

   - outcome: first_pass; verdict: ok
   - sources: dispatch-desk#1
   - bank wording: Built a tool-using agent in Python that triages freight-damage claims with an 8B open-weights model served locally by Ollama and one retrieval tool over a SQLite table of 3,200 past claims.

2. Achieved 35 successful passes out of 40 hand-written scenarios with a retrieval tool, compared to 22 of 40 without it, using pytest cases and CI pipeline runs.

   - outcome: after_retry; verdict: ok
   - sources: dispatch-desk#2, dispatch-desk#4
   - bank wording: Passed 35 of 40 hand-written claim scenarios (0.875), against 22 of 40 (0.55) for the same model without the retrieval tool.
   - first attempt rejected: escalation_causal: 'through structured testing' - the cited sources do not link 'through structured testing' to the result in one statement

## Gridlock: a distributed rate limiter in Rust and Redis (`gridlock`)

Coverage score 0/10: Uses Rust/Redis, lacks Go expertise required for API backend.

1. Built a rate limiter in Rust that uses Redis and a Lua script to perform a single atomic step for checking and updating token buckets.

   - outcome: after_retry; verdict: ok
   - sources: gridlock#1
   - bank wording: Built a rate limiter in Rust that keeps token buckets in Redis and updates them with one Lua script, so a check and its update are a single atomic step.
   - first attempt rejected: escalation_causal: 'ensuring consistent checks and updates' - the cited sources do not link 'ensuring consistent checks and updates' to the result in one statement

2. Achieved 41,000 checks per second on a 4-vCPU VM with a p99 latency of 2.1 ms at 20,000 checks/s, using a single Redis instance.

   - outcome: first_pass; verdict: ok
   - sources: gridlock#2
   - bank wording: Handled 41,000 checks/s on one 4-vCPU VM against a single Redis instance, with a p99 of 2.1 ms at 20,000 checks/s.
