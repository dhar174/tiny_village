# TASK025 — PR ownership and Project configuration (#737)

Status: In progress. Updated 2026-10-22 UTC.

## Request and approved scope
Implement the approved #737 coordination plan; review all 26 PRs, investigate five direct handoffs, reconcile 328 issue-backed tasks, and prepare safe Project synchronization. Runtime changes and PR dispositions require their own review; #738 is a separate human gate.

## Progress
- Base: merged main 48213783e2ee0864594af4a3f26667e1269d394a (#805).
- Approved plan: docs/superpowers/plans/2026-10-22-issue-737-coordination.md.
- Snapshot, matrix and Project preview: pending.
- Project access: previously HTTP 403; recheck and retain current evidence.
- Publication and #737 completion: pending, never inferred from offline tests.

## Execution ledger
- Human choices are recorded only in docs/decisions.md; technical implementation notes remain here.
- Preflight: matrix issue keys consume the union of frozen roadmap86, triage307 and runtime5 (328 unique). Native blockers are distinct from canonical parents and proposed execution prerequisites.
- Technical ruling: final PR dispositions remain recommendations until reviewed; future-phase code/rebases stay with their owning issue. This avoids broadening the approved coordination scope.

## Verified implementation progress
- Captured484 source issues/26 candidate PRs; managed332 after four scoped remediations. Ownership/order matrix and12-field/six-view preview prepared;112 native edges/387 separate planned activation references, no cycles/order conflicts.
- Created #806 access blocker (parent737), #807 helper contracts (parent585), #808 prompt failures (parent137), #809 graph coverage (parent222), preserving gate order. Owners737/585/137/222 are blocked by their respective children. Original assignees/parents/bodies preserved.
- Five disposition recommendations prepared; no PR rewrite/closure. Core72+3 CLI probes and cache27 pass. Offline synchronization32 and ordering11 pass; failed runtime criteria under807/808 and invalid original graph fixture remain captured.
- Independent read-only review supplied two Important findings. Fix pass added owner blockers and required/documented edit window; remote API has no CAS. A controlled race-boundary test explicitly demonstrates the remaining API limitation. New-item reads now avoid unrelated private Project permissions.
- Live prepare passes repository source guards then Project403. #737/#806 publication verification and #738 human decision remain pending. Portable package and focused review PR being prepared.
- Technical ruling: preserve non-managed Status; an open issue with Done-like Status blocks prepare for review. No automatic status or phase-decision repair.
- Technical ruling: a failed/unacknowledged write requires inspect and a newly reviewed plan/checkpoint; do not infer ownership of ambiguous writes or silently retry.

- Portable archive extracts and validates outside Git on Linux. Deliberately modified preview/capture and missing hash manifest are rejected; the missing-manifest weakness was first reproduced, then corrected. Native Windows and real Project adapter behavior remain unverified.
- All145 retained runtime/runner/lock file hashes were compared against git show4821378; zero mismatches.
