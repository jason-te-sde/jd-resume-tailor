"""Small banks for the tests, built fresh each time so a test can break them freely."""

from __future__ import annotations

import copy
from typing import Any

WORK: dict[str, Any] = {
    "id": "acme-swe",
    "type": "work",
    "title": "Software Engineer",
    "company": "Acme Corp",
    "location": "Remote",
    "date": "Jan. 2024 -- Present",
    "tech": "Go, PostgreSQL",
    "bullets": ["Cut p99 latency from 412 ms to 138 ms.", "Wrote 23 alerts."],
    "bullet_capabilities": ["performance", "observability"],
    "tags": ["backend", "latency"],
    "adaptable": True,
    "description": "Backend work.",
    "scope": "EVIDENCE: load test. DO NOT CLAIM: production p99; cost savings.",
}

PROJECT: dict[str, Any] = {
    "id": "queue-lab",
    "type": "project",
    "title": "Queue Lab",
    "date": "2025",
    "tech_stack": "Go, SQLite",
    "github": "https://example.com/code/queue-lab",
    "has_code": True,
    "bullets": ["Processed 9,200 jobs/s in a 3-node test."],
    "bullet_capabilities": ["throughput"],
    "tags": ["queue"],
    "adaptable": True,
    "description": "A job queue.",
    "scope": "EVIDENCE: bench/run.csv. DO NOT CLAIM: exactly-once delivery.",
}

NO_CODE: dict[str, Any] = {
    "id": "private-model",
    "type": "project",
    "title": "Private Model",
    "has_code": False,
    "bullets": ["Reached F1 0.87 on a held-out set."],
}


def bank(*entries: dict[str, Any]) -> dict[str, Any]:
    """Return a bank document holding deep copies of ``entries`` (default: work + project)."""
    chosen = entries or (WORK, PROJECT)
    return {"experiences": [copy.deepcopy(entry) for entry in chosen]}
