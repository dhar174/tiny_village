# TASK025 — PR ownership and Project configuration (#737)

Status: In progress. Updated 2026-10-23 UTC.

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
- Live prepare passes repository source guards then Project403. #737/#806 publication verification and #738 human decision remain pending. Portable package published in draft PR #810: https://github.com/dhar174/tiny_village/pull/810. Human disposition/configuration review and authenticated publication remain pending.
- Technical ruling: preserve non-managed Status; an open issue with Done-like Status blocks prepare for review. No automatic status or phase-decision repair.
- Technical ruling: a failed/unacknowledged write requires inspect and a newly reviewed plan/checkpoint; do not infer ownership of ambiguous writes or silently retry.

- Portable archive extracts and validates outside Git on Linux. Deliberately modified preview/capture and missing hash manifest are rejected; the missing-manifest weakness was first reproduced, then corrected. Native Windows and real Project adapter behavior remain unverified.
- All145 retained runtime/runner/lock file hashes were compared against git show4821378; zero mismatches.

- Draft PR #810 published and attached to the task; implementation commit ad58fd0. No #737/#806/#738 closure or PR disposition. Hosted checks were queued at publication; local checks/evidence are the verified result.

## Closeout execution — 2026-10-23 UTC
- D032 records the approved dispositions/local route; plan saved at docs/superpowers/plans/2026-10-23-issue-737-closeout.md. New branch codex/issue-737-closeout starts from merged main10569c0.
- Reopened737 while806 remains open; closed701/704 unmerged after unchanged-head and145 runtime/runner/lock hash checks; annotated709/711/715 and807/808/809, preserving heads/assignees. Before/after receipts retained.
- Ruling: retain the original810 package untouched; self-contained refreshed artifacts live in docs/planning/issue-737-closeout. Historical probe results remain historical.
- Ruling: this package supports publication stage only; reject completion flags until authenticated evidence and a separate final-stage review exist. This prevents offline tests from appearing to authorize closure.
- Closeout regressions9 RED (missing interfaces) then9 GREEN; existing32 publisher and11 ordering tests pass. Windows path normalization is tested; native Windows remains unverified.
- Pending: refreshed capture/validation, independent review, focused closeout PR and owner-authenticated local prepare.806/737 cannot close before actual publication/readback;738 remains Pending.
- Fresh snapshot484/332/24 and preview validate;112 native edges,387 planned activation references,333 source guards, zero cycles/conflicts. Portable extraction validates and53 tests pass; three tamper cases reject. Actual prepare reaches GraphQL FORBIDDEN after source guards, exit1, zero board writes.
- Independent reviewer identified a fork/branch collision in candidate exclusions. Fixed with canonical headRepository plus exact branch; real source-collector regression RED1 then GREEN. Full54 tests pass; null/foreign identities stay inventoried. No other findings.
- PR #811 published: https://github.com/dhar174/tiny_village/pull/811. No automatic closure keywords. Refreshed local PowerShell handoff awaits authenticated live-plan/digest; actual board delta remains unreviewed until supplied.
- Final snapshot includes25 open PRs with only canonical-repository closeout811 excluded, leaving the approved24 candidates. All332 managed issues and333 guards validate; final exact ZIP validates outside Git. Fresh prepare again passes source checks then403. Initial #811 hosted core/dev/visual/required/CodeQL checks all succeeded at6bdf736; later source/evidence commit requires its own check result.
- Remaining work requires owner-supplied authenticated live-plan/digest. Apply only after concrete delta review;806/737 closure and final readiness refresh remain blocked. No native Windows or successful Project readback claim.
