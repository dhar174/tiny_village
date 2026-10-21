# #734 — Evidence-based backlog triage

Approved for execution by dhar174: “Implement the proposed plan.”

## Goal and boundaries

Catalog the original 261 issues, reconcile the current backlog, and propose
evidence-based issue updates. Historical cutoff: 2026-10-13T07:30:00Z, when
roadmap issue #716 was created. Reconstruct membership from creation and
closed/reopened events. Pin current main, snapshot time, and artifact hashes.

This is investigation and tracking work. Do not repair runtime behavior, rebase
or merge PRs, approve Phase 3, or rerun the old Project publisher. AI remains
required in the final product. Existing design choices remain settled.

## Ordered work

1. Start a focused branch from refreshed main; save this plan, TASK021, the
   execution record, and the user's publication/ownership choices.
2. Retain paginated issue/PR snapshots, comments, native parents/dependencies,
   relevant PR revisions, and historical state evidence. Validate the exact
   original cohort and identify subsequent additions separately.
3. Read each original issue and relevant discussion; compare acceptance with
   current source and retained evidence. Each ledger row needs identity/cohort,
   state, actual assignees, accountable owner, canonical topic/native parent,
   prerequisites/readiness, disposition, acceptance/reproduction, evidence,
   verification status, and rationale. Review later additions for coverage and
   blockers. Use implement/verify/duplicate/defer; do not equate appearance with
   verified completion. Defer requires a condition; duplicates need a canonical
   target and preservation of unique acceptance requirements.
4. Audit #482 weather rendering; #355 action-queue tuple consumers/producers,
   ties and replanning; #602 named-target graph regressions; #596 live timing
   collection; #597 frame analysis; #609 GraphManager relationship authority.
   Keep #602 open while required checks/Phase 4 prerequisites remain blocked.
5. Use existing focused tests and real-object scratch probes; retain exact
   commands, source/environment, counts, failures/skips/timeouts. New cold
   probes use the durable capture helper with 20s diagnostic and 25s parent
   deadlines. Record failures as triage evidence, not an instruction to fix
   runtime code. No retries converted into passing evidence.
6. Prepare a complete ledger, summary, dependency order and exact proposed
   mutations in one documentation PR. Review the package once with dhar174
   before publishing issue metadata. Duplicate closures require approval of
   the specific issue numbers and evidence. Human decisions live only in
   docs/decisions.md.
7. After review, apply approved annotations/links/state changes only. Preserve
   source bodies, assignees, parents and decisions; compare current contents
   before writes, checkpoint and verify API readbacks. Project 10 HTTP 403
   means retain pending configuration changes for #737, without claiming sync.

## Ownership and organization

Preserve GitHub assignees. dhar174 is accountable ledger owner for unassigned
items; no mass GitHub assignments. Reuse existing subsystem epics, preserve
native parents, cross-link other canonical topics, and propose missing links
or epics for review. Validate dependencies for cycles and lost edges.

## Acceptance and handoff

All 261 original issues have complete evidence-backed rows; current additions
are reconciled; all six audits have results or explicit blockers; hierarchy and
dependency checks pass; no unapproved state changes occur. Full completion
requires reviewed documentation to merge and approved metadata to be verified.
Until then #734 remains open. #735 and #736 follow, then #737; #738 remains a
separate human gate. Use Refs #734, not an automatic closing keyword in the PR.
