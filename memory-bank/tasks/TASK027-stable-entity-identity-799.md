# TASK027 — Stable graph entity identity (#799)

**Updated:** 2026-11-13

**Status:** PR #815 merged at `67a8abd20d497397cd3c48006a85287d5f082ee2`. Issue #799 closed as completed.

Approved scope: [D035](../../docs/decisions.md#d035--preserve-object-graph-nodes-with-immutable-entity-uuids-799).
Worktree branch: `codex/799-stable-graph-identity`, based on
`c6f06c4001b581da54e945510948b3669d7fcc7e`.

All five retained probes reproduced their failures on the baseline and exited
zero with the repair. The action execution probe reported object and graph
energy both at 30, and the maintained regression asserts exact synchronization.
The maintained regressions cover lookup and removal after mutation, graph
provenance, named-target identity and ambiguity, rollback of a failed
attribute, and successful named-target application with two real Characters.
The successful case runs the retained probe and checks that duplicate visitor
entries result in a single effect synchronized to the object and graph.

The sources for the five reproductions are tracked in two directories. Four
[graph probes](../../docs/planning/issue-736/execution/blocked/) came from #736.
The [named-target probe](../../docs/planning/issue-734/evidence/audit_probes.py)
came from #734. Its original launcher used an absolute checkout path. Run
these from the repository root in the locked CPython 3.12 Linux environment:

```bash
for probe in graph_character_identity graph_state_provenance graph_action_provenance graph_executed_world; do
    timeout 25s python -B - < "docs/planning/issue-736/execution/blocked/$probe.py.txt"
    printf '%s exit=%s\n' "$probe" "$?"
done
timeout 25s python -B docs/planning/issue-734/evidence/audit_probes.py named_target
```

Use a separate checkout at `c6f06c4` for baseline comparison. Each probe
should exit 1 at the baseline and 0 with the repair. The four graph probes
take source on stdin to keep imports rooted at the checkout. The named-target
probe also runs through `python -B -m unittest tests.test_entity_identity_contracts`
on the updated PR branch.

As documented in D035, a direct Character write (such as weather energy
drain) can leave the graph stale. A later action uses the graph value when
present and may overwrite that direct write. Reconciliation of direct
gameplay writers is follow-up work.

Local receipts under `.test-artifacts/`: `799-reproduction-comparison.json`,
`799-red/report.json`, `799-final-core/summary.json`,
`799-final-dev/summary.json`, and `799-final-visual/report.json`.
On Linux x86-64 with CPython 3.12.14, core passed 76 tests and three CLI probes,
dev passed 20 tests, and the focused visual dependency environment passed 54
graph ownership, WorldState and Location integration tests.
The review follow-up passed 11 focused tests and the core profile with 77
tests and three CLI probes. The core receipt is
`.test-artifacts/799-review-core/summary.json`.

The existing #738, #220, #602, #741 and #802 gates retain their dispositions.
The implementation does not establish whole-action transactions, callback
rollback, live inference, or broader gameplay readiness.
