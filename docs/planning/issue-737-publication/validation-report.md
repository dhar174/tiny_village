# #737 publication tooling validation

**Tooling is under review; authenticated board publication remains unverified.**
D033 authorizes execution, not an unseen board delta or gate738 approval. The
frozen owner-facing ZIP is generated only after the tooling PR merges.

## Current development baseline and results

- Main/source5f2bf989c26ffdb0d40778b88947dd5d1d7a3599, merged #811. Fresh captured
  inputs cover484 issues,332 managed issues and24 unchanged reviewed candidate
  heads. Twelve fields, six views,112 native edges,387 planned activation references,
  333 source guards, zero detected cycles/order conflicts. Snapshot times retained.
- `python3.12 -B -m unittest discover -s docs/planning/issue-737-publication -p 'test_*.py'`:
  89 tests pass,0 failures/errors/skips. This includes the prior54 and35 receipt,
  closure-stage and CLI/lifecycle regressions. RED evidence and GREEN output are
  retained under `evidence/publication-tooling/`.
- The full332-row real builder is exercised through initial publication, simulated
  issue completion, completed-stage validation, final readiness publication and
  independent receipt verification using a controlled external API. It preserves
  unrelated items and keeps738 Pending. This is **offline simulation**, not live
  GitHub mutation or human acceptance.
- Locked core CPython3.12/Linux: canonical runner passes72 tests and3 CLI probes,
  zero failures/errors/skips. Fresh summary, full output and raw capture archive
  with member hashes are retained separately from inherited #810/#811 evidence.
- All145 retained runtime/runner/lock source hashes match current merged main;
  no runtime or dependency changes, live inference or gameplay acceptance claimed.
- Actual read-only `--prepare` passes repository source guards, then returns Project
  FORBIDDEN, exit1. No concrete live plan or board write occurred. Existing configured
  credentials are not requested or printed. Local authenticated owner evidence
  remains necessary for publication and closure.

## Receipt and stage boundaries

Receipts bind an exact desired preview, reviewed plan/baseline, project identity,
normalized full board state, source reads before/after inspection and timestamp.
Verification recomputes actual differences and preservation; it does not trust a
success label, checkpoint or digest alone. JSON records are not cryptographic
attestation: the human review still establishes their live provenance and approval.
The legacy CLI remains supported; closure requires the stronger retained receipt.

Publication stage keeps806/737 open and738 Pending/Blocked. Completed stage requires
verified prior initial publication, the same managed/candidate cohort, actual
completed issues closed after the receipt (806 no later than737), Tracking readiness
and738 open/Pending/Human review. A completed-stage package still needs its own
final readiness apply and independent inspection. Runtime gaps remain open.

Native Windows and actual board adapter mutation/readback remain unverified until
the local run. Windows path and portable extraction tests do not establish native
Windows support. Historical captures remain labelled and the prior packages are
unchanged. Only already permitted CI is used; no new paid service is introduced.

## Independent review

A read-only reviewer inspected the exact code, tests and documentation from the
coordinator's retained tool history; no reviewer commands or edits occurred.
Two Important findings were reproduced: an internally consistent wrong-board
receipt could satisfy completion, and a receipt's fixed temporary path could
replace its plan. Four regressions failed first, then passed after canonical
Project10 URL/reviewed node-ID binding and temporary/resolved-path guards. All89
tests pass after the fix. Review did not independently execute the real adapter,
native Windows or all raw artifacts. No other findings were reported.

The candidate archive extracts and validates outside Git; all89 tests pass there.
Altered preview, missing manifest and corrupted fresh core archive each reject.
This is portability/tooling validation only: the authenticated publication ZIP is
not frozen until after merge. Native Windows remains unexecuted.
