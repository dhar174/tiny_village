# TASK023 — Review functional-test PRs (#736)

**Status:** In Progress — all nine dispositions selected; consolidation plan awaiting review  
**Added:** 2026-10-21 America/New_York  
**Updated:** 2026-10-22 UTC

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
