# TASK023 — Review functional-test PRs (#736)

**Status:** In Progress — #420/#417 dispositions selected; reviewing #408  
**Added:** 2026-10-21 America/New_York  
**Updated:** 2026-10-21 America/New_York

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
