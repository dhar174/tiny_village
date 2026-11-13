# TASK027 — Stable graph entity identity (#799)

**Updated:** 2026-11-13

**Status:** Implemented and locally verified; awaiting review of the worktree.

Approved scope: [D035](../../docs/decisions.md#d035--preserve-object-graph-nodes-with-immutable-entity-uuids-799).
Worktree branch: `codex/799-stable-graph-identity`, based on
`c6f06c4001b581da54e945510948b3669d7fcc7e`.

All five retained probes reproduced their failures on the baseline and exited
zero with the repair. The action execution probe reported object and graph
energy both at 30, and the maintained regression asserts exact synchronization.
The four new regressions cover lookup and removal after mutation, graph
provenance, named-target identity/ambiguity, and rollback of a failed attribute.

Local receipts under `.test-artifacts/`: `799-reproduction-comparison.json`,
`799-red/report.json`, `799-final-core/summary.json`,
`799-final-dev/summary.json`, and `799-final-visual/report.json`.
On Linux x86-64 with CPython 3.12.14, core passed 76 tests and three CLI probes,
dev passed 20 tests, and the focused visual dependency environment passed 54
graph ownership, WorldState and Location integration tests.

The existing #738, #220, #602, #741 and #802 gates retain their dispositions.
The implementation does not establish whole-action transactions, callback
rollback, live inference, or broader gameplay readiness.
