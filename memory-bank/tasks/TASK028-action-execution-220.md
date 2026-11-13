# TASK028 — Graph-backed action execution (#220)

**Updated** 2026-11-13

**Status** Local implementation and verification on `codex/220-action-execution`.

The approved scope selectively ports draft [#696](https://github.com/dhar174/tiny_village/pull/696)
at `8016a3f467719afa4b795207c14f40502b161410` onto
`67a8abd20d497397cd3c48006a85287d5f082ee2`, the [#815](https://github.com/dhar174/tiny_village/pull/815)
merge. [D035](../../docs/decisions.md#d035--preserve-object-graph-nodes-with-immutable-entity-uuids-799)
continues to govern identity and rollback.

Action resolves registered actors and targets, evaluates concrete conditions
against graph state and applies validated scalar changes in native units.
Controller and Plan invoke the selected action once. The controller owns its
success bookkeeping. Weather drain shares the synchronized attribute write,
which restores the failed attribute and preserves earlier committed effects.

The candidate disposition and local receipts are retained under
`.test-artifacts/`. The comparison covers the six original #220 regression
failures, #799 compatibility controls, and the two canonical Linux profiles.
The #816 weather case checks exact values of 20, 15 and 25 and two writes.
The wider #816 direct-writer audit and the #740/#802 planner changes remain
separate follow-ups. #741 retains the replanning work and #602 retains the
integrated gameplay acceptance gate.
