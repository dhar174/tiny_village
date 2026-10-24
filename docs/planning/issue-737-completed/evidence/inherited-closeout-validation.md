# #737 closeout validation and handoff

## Current result

Repository dispositions are applied; board publication is **not verified**.
#737 is open with native blocker #806 open. #738 remains open/Blocked and
Decision Pending. No issue acceptance checkbox is marked on unverified evidence.

## Fresh closeout evidence (2026-10-23 UTC)

- Source main: `10569c06cc01075c70d5b39ec1b9e6804eb7f5cd`, merged #810.
  Snapshot timestamps retain the exact capture interval. All484 source issues,
  332 managed issues and24 reviewed candidate heads are retained.
- Read back #701/#704 closed and unmerged, #709/#711/#715 open, all five heads
  and assignees unchanged, #737 reopened with its prior body intact.
  `evidence/closeout/dispositions-before.json`, `dispositions-after.json` and
  `mutation-receipts.json` preserve actual responses and comment URLs.
-12 custom fields, six views,112 native edges,387 distinct planned activation
  references;333 source guards; zero cycles or ordering conflicts. Fresh preview,
  issue matrix and candidate matrix regenerate from the captured source inputs.
- CPython3.12/Linux, standard library:32 publisher tests,11 ordering tests and11
  closeout tests pass (54 total; zero failures/skips). Closeout tests initially
  failed9 missing-interface cases; retained RED and GREEN output is under
  `evidence/closeout/`. The live-inventory adapter tests confirm same-branch foreign-fork PRs and new
  unreviewed candidates remain visible while the exact canonical-repository closeout branch is excluded.
- Closeout checks cover changed heads/same-count substitutions, premature closure,
  runtime gap preservation, gate readiness, completion-flag rejection and Windows
  path normalization. Existing publisher tests retain interrupted/resumable and
  repeated-apply/no-duplicate coverage.
- All145 retained runtime/runner/lock source hashes match merged main; this is
  source comparison, **not** a rerun of game runtime checks. No runtime, lock,
  CI service or UI behavior changes were made.
- Extracted ZIP validation and all54 tests also pass outside Git on Linux.
  Deliberately changed preview, corrupted raw archive and missing manifest each
  cause validation failure; receipts: `evidence/closeout/portable-verification.json`.
- Current Project REST access returns403 “Resource not accessible by integration.”
  The actual CLI preparation passes all333 issue source guards,24 PR heads and
  main revision, then receives GraphQL FORBIDDEN (exit1); output retained under
  `evidence/closeout/prepare-*.txt`. No live before/after
  plan, board mutation or independent successful readback is claimed.

## Historical evidence

The copied core72 tests +3 CLI probes, cache27 checks and runtime probes retain
#810's original revisions, failures and limits. They are not new executions.
Original historical reports are labelled `evidence/historical-*`; canonical
original files under `../issue-737/` remain unchanged. Both hash manifests and
core archive member hashes verify their integrity. Initial invalid graph fixture
and failed helper/prompt criteria remain evidence rather than passing acceptance.

## Remaining acceptance

Run the portable package on the owner's authenticated computer and return the
read-only prepared live plan/digest. Review actual fields/views/conflicts before
apply, coordinate an edit window, then obtain independent readback. Only then
close806/737 and refresh/review/publish their final Tracking readiness and738's
Human review readiness with Decision Pending. Current stage rejects completion
without a separately reviewed evidence-backed final package. Native Windows,
real Project adapter mutation/readback, gameplay and live AI remain unverified.
