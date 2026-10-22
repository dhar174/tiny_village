# TASK023 — Review functional-test PRs (#736)

**Status:** In Progress — D030 accepted; runtime ownership/dependencies synchronized; #804 ready for review  
**Added:** 2026-10-21 America/New_York  
**Updated:** 2026-10-22 America/New_York

## Request

“Lets go through them for 736, then.” Review nine older functional-test PRs
against current main and merged #349; present keep/rebase/split/supersede choices
with evidence before implementation or closures.

## Approach and boundaries

1. Pin main and live PR heads; retain descriptions, discussions and review notes.
2. Compare actual source and maintained paths; identify useful contract assertions,
   weak/false-positive tests, overlaps and out-of-scope production changes.
3. Use narrow bounded probes where execution clarifies a recommendation; preserve
   failures and avoid model/network initialization. No new test implementation.
4. Walk through decisions one at a time, recording confirmed choices only in
   docs/decisions.md. Build a scoped consolidation plan after the walkthrough.
5. Specific closures and code changes await the relevant reviewed choices.
   #736/#737/#738 remain open; no phase-gate approval.

## Progress

- Main 11b6e93 includes #703/#798; verified merges and #735 closure.
- Retained current heads for #420/#417/#408/#406/#404/#397/#395/#391/#393.
  Reviewed #349's merged root file; current root copy is identical.
- #420: five original candidate methods pass against current main, no skips,
  errors, failures or timeout. Recommend salvage unique real State/Action/
  Condition assertions into maintained suites, then supersede old demonstration.
  User selected split/salvage then supersede after validation; recorded D029.
  Execution follows the consolidated walkthrough/plan; no port or closure yet.
- Independently inspected remaining source; #417's selected original utility
  method passes (one test), not a full-PR validation. Others remain unexecuted.
- [Walkthrough and evidence](../../docs/planning/issue-736/README.md).
  No implementation, PR closure/rebase or GitHub metadata mutation.

- User selected #417 split/salvage unique real-object utility assertions, then
  supersede after validation; recorded D029. #408 is the next disposition.

- Independent contract probes: #393's reduced calls fail current signature binding;
  #397's original helper reports passed=-1 for one method with two controlled
  failing subtests. Retained expected-failure capture exits zero as a diagnostic;
  it is not a claim those injected product tests pass. No implementation change.
- #408 superseding recommendation presented; user choice remains pending.

- User selected #408 supersede through verified consolidation, retaining any
  unique useful assertion found during mapping; recorded D029. #406 is next.

- User selected #406 salvage useful memory API assertions, then supersede;
  recorded D029. #404 has no current Python format_traceback caller; its obsolete
  wrapper recommendation is pending. No code/closure changes.

- User selected #404 supersede as obsolete; recorded D029. #397 counting
  disposition is pending, backed by the retained negative-passed-count probe
  and existing CI-runner subtest coverage. No implementation/closures.

- User selected #397 supersede with maintained counting-coverage links; recorded
  D029. #395 planner-scenario salvage recommendation is pending. Preserve real
  runtime defects and reject false-positive success paths; no code changes.

- User selected #395 salvage strict planner assertions then supersede; recorded
  D029. #391 happiness keyword-threshold disposition is pending. Preserve
  required behavior and legitimate absent-relationship cases; no implementation.

- User selected #391 behavior-focused happiness checks then supersede; recorded
  D029. Include applicable and absent relationship cases and preserve broken
  required behavior as failures or tracked blockers. #393 is the last disposition;
  no implementation, rebase or closure has been performed.

## Progress log — 2026-10-22 UTC

- Completed the nine-PR walkthrough. Recorded the final #393 choice under D029,
  including reconciliation of the two differently answered question versions.
- Prepared the scoped consolidation plan in
  docs/superpowers/plans/2026-10-22-issue-736-functional-test-consolidation.md.
  Source inspection also identifies Goal.check_completion signature/container
  inconsistencies and swallowed motive-path errors as risks to verify, not
  newly proven runtime failures. No new execution probe was run this turn.
- Awaiting plan review; no test/runtime implementation, closure or GitHub write.

| Subtask | Status | Updated |
| --- | --- | --- |
| Pin candidate sources and inspect maintained suites | Complete | 2026-10-21 |
| Record all nine human dispositions in D029 | Complete | 2026-10-22 |
| Prepare scoped consolidation plan | Complete; awaiting review | 2026-10-22 |
| Implement and validate assertion consolidation | Not started | 2026-10-22 |
| Publish replacement PR and verified closure crosslinks | Not started | 2026-10-22 |

## Approved execution — 2026-10-22 America/New_York

- User approved the plan and helpful subagents. Three scoped agents handled
  Action/utility, happiness and graph/planner ports; parent handled provenance,
  memory API, CI integration, report and publication.
- Main and all candidate heads unchanged. Retained24 source snapshots/190
  inventoried methods, distinguishing inherited assertions from PR deltas.
- Added21 maintained methods; core/dev isolated runs pass. Integrated local core
  72 tests+3 probes and dev20 tests pass. Existing40-method suite retains10
  failures identically reproduced on exact main; #774 remains open.
- Required runtime diagnostics remain failed: full Goal/Plan contracts #802,
  additional motive contribution #803; graph/world identity #799 and replan
  return contract #741 remain. No runtime repair, skip/xfail or fake planner.
- Full report: docs/testing/issue-736-functional-test-consolidation.md. Raw
  captures retained in a hashed archive; five initial graph-fixture failures,
  excluded candidate interface misuse and duplicate validations are explicit.
- #736 and all nine candidates stay open; replacement publication/review and
  explicit blocker disposition remain. #737/#738 retain separate approval.

| Execution subtask | Status |
| --- | --- |
| Task1 pin/map/source inventory | Complete |
| Tasks2–3 Action/utility + memory ports | Implemented; targeted checks pass |
| Tasks4–5 happiness + planner/graph | Passing ports retained; required acceptance partial |
| Task6 local CI and evidence report | Local checks complete; blocker disposition pending |
| Task7 publish/review and supersede | Publication pending; no closures |

## Publication — 2026-10-22 America/New_York

- Draft PR #804 published and attached. Fresh independent review found no
  Critical issues; its alternative-action mapping gap and stale baseline
  statement were corrected in bd513ce before publication. No test/runner
  content changed in the correction.
- Evidence comments link #736/#802/#803/#799/#741 and all nine candidates to
  immutable replacement sources. All candidates and #736 remain open.
- Required next human decision: review partial acceptance and select whether
  failed requirements carry forward explicitly, runtime order changes for
  scoped repairs, or consolidation merge is deferred. Recommended explicit
  carry-forward remains a proposal, not a recorded approval.
- Task7 publication complete; replacement merge, gap disposition and specific
  superseding closures remain pending. No runtime repair or acceptance
  exception is approved by this handoff.

## D030 carry-forward — 2026-10-22 America/New_York

- User selected Option1 and explicitly required gaps to remain in correct issues.
  Recorded in the single decision file under D030. Runtime requirements remain
  #802/#218, #803/#139, #799, #741 and utility debt #774, with failed evidence.
- Synchronize owning issue bodies and native parents/gate dependencies for the
  two new follow-ups, preserving original requirements and existing relations.
- Finalize #804 for review; no merge, runtime closure or #738 approval. Replacement
  merge and bounded post-merge verification precede superseding dispositions.

## Verified carry-forward publication — 2026-10-22 America/New_York

- Owning issues #802/#803/#799/#741/#774 and parents #218/#139 have additive
  carry-forward sections with immutable D030/report/reproduction links and
  unchecked runtime requirements; original bodies/assignees/states preserved.
- #802 is native child of #218 and #803 of #139. Both are blocked by #738;
  each parent is blocked by its child. Verified an acyclic affected prerequisite
  graph over28 issues and preserved #799/#741/#774 existing relations.
- #736 records scoped acceptance with merge/verification pending. #737 receives
  Project/ownership handoff and #738 receives the retained-gap review note.
- #804 description updated to D030 acceptance; finalize for human review without
  merging. Receipt files live in docs/planning/issue-736/execution/option1/.
- Documentation/tracking only; tested runtime/runner/test source remains unchanged.
