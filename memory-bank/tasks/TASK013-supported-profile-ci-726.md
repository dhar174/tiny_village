# SDD ledger — plan: docs/superpowers/plans/2026-10-20-issue-726-supported-profile-ci.md

**Status:** PR #789 ready for human review; #726 open pending post-merge rollout
**Added/Updated:** 2026-10-20

## Request / approved scope

Implement the reviewed #726 plan. Required core/dev/visual CI; NLP and CPU-LLM
checks manual only. No automatic merge, closure or phase approval. AI remains
required in the final product (D015).

## Tasks / shared interfaces

1. Post-merge baseline and plan: complete, base eb5f4c9 (#788 merged).
2. Bounded runner: complete. Produces --profile / --artifacts and exit/report contract.
3. Required workflow: implemented. Consumes runner profiles and success-only gate.
4. Manual ML workflow: implemented. Consumes nlp/llm-cpu profiles; never scheduled.
5. Validation/review/PR: complete for publication. Human review and hosted main/manual/enforcement rollout follow merge.

Pre-flight: tasks 3/4 consume task 2's exact profile names and artifact directory.
Both require existing hash locks; no application runtime API changes.

## Rulings and evidence

- Ruling: use a feature branch in the existing clone and this resumable ledger
  plus /workspace/scratch/issue-726 — managed cloud constraint and cloud-only
  skill helper scripts; cost is manual ledger bookkeeping rather than worktree
  isolation/helper automation. Preserve evidence rather than deleting it.
- Ruling: execute the approved explicit matrix rather than bare whole-repo
  pytest — unsupported/live-model/known legacy suites are outside this plan;
  cost is no whole-repository-green claim.

## Log

- 2026-10-20: user approved implementation. Branch from merged main eb5f4c9.
  Post-merge baseline running; no runtime or dependency-lock edits.

- Task 1: complete — baseline.json retains 47 passing methods and CLI exits
  0/0/2. Task 2 RED: 12 tests fail because the bounded runner does not exist.

- Task 2: initial 12-method RED→GREEN pass. New runner uses stdlib parent and
  separate session per batch, real unittest/pytest counts and explicit failing
  evidence for empty/skipped/incomplete/nonzero/timeout/cancellation cases.
  The existing subprocess_capture interface and diagnostic limits are unchanged.

- Task 2: complete — 13 runner methods pass. All five profiles pass locally:
  core 40 + 3 CLI probes, dev 15, visual 68, NLP 2, CPU-LLM 2 (127 methods/cases).
  Core/dev/visual used new hash-verified issue726 venvs and pip checks; optional
  local profiles reuse existing locked environments, not fresh-install claims.
- Workflow structural check: verified actionlint 1.7.12 archive SHA-256
  8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8.
  Initial check rejected runner.temp in job-level env; retained original output,
  moved cache assignment to a runner step, and both workflows validate.

- Independent whole-branch review at ab22ce2: three Important findings, no
  Critical findings or declines. All accepted: nested session cleanup, pytest
  non-strict XPASS accounting and missing/null probe exit codes.
- One fix pass: original review-red retains six failing observations across
  17 methods (including subcases); xpass-red separately proves false PASS.
  Intermediate group cleanup tests exposed asynchronous SIGKILL delivery;
  retained review-green1 failure, added bounded quiescence after signalling.
  All 18 runner methods now pass in review-green2.log.
- Ruling: track supervisor-created nested sessions in an inherited scoped
  registry, with bootstrap registration/closed-owner guards and Linux process
  identities — signal handlers cannot clean up after a watchdog hard exit;
  cost is Linux-specific lifecycle bookkeeping, within the approved target.
- Final: minor (deferred): original prewritten RUNNING test branch is deleted
  by the supervisor, duplicating missing-file coverage. New child-written
  partial-report regression covers the real contract; original duplicate
  assertion remains a test-cleanup opportunity, not an untested success path.

- Final: fixed all three Important findings — new nested timeout/watchdog,
  XFAIL/XPASS and child-written incomplete-code regressions RED→GREEN; 18/18
  runner methods pass. Full corrected five-profile matrix 132/132 cases and
  3/3 CLI probes pass, zero unexpected skips/profile failures/timeouts.
  Fresh required environments/pip checks and final actionlint pass retained.
  Runtime/test/workflow/lock hashes published in issue-726-local-results.json.

- Published PR #789: https://github.com/dhar174/tiny_village/pull/789 . Issue #726
  body and evidence comment updated; issue remains open, no closing PR keyword.
- First hosted PR run 39501297078 passes core/dev/visual and Python CI required:
  128 cases + 3 probes, no skips/failures/errors. All artifact ZIP digests match
  GitHub metadata. Publisher GitHub Actions app ID 15368. No enforcement change
  or manual workflow dispatch before merge. Hosted receipt published.
