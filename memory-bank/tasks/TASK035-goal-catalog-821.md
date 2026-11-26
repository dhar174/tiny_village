# TASK035 — Complete #821 Goal and inventory contracts
Updated: 2026-11-25 America/New_York. Status: implementation and local verification complete; PR828 final publication/checks pending.
Authorization: user approved the complete proposed plan with “Please implement the plan.”
Base: freshly fetched main a600aee; isolated /workspace/tiny_village-821.

## Contract and execution ledger
- Empty UtilityGoal metadata is incomplete; full Goal empty Conditions remain complete.
- Preserve stable UtilityGoal alias and actual three-argument factory callbacks.
- Fix inventory counts/attribute matching and both Cook requirements.
- Derive employment from graph job or isolated explicit actor state; project supported Job names and track job dependencies in retained requests.
- Add contextual evaluation errors; preserve request ERROR versus NO_PLAN/SATISFIED.
- Goal/Plan structural observations are pure; both mutable types become unhashable.
- Coordinator alone edits/runs commands. Architecture scout reconfirmed boundaries; domain/final reviewers remain read-only.
- Pre-flight: completion error interface is consumed by scoring and Strategy; job normalization is shared by Character, Goal, planner and fingerprint paths. No interface conflict.
- Ruling: use this canonical Memory Bank task as ledger; skill helper scripts are not installed locally. Preserve existing dependency profiles, deadlines and nonscalar planning exclusions.
- No automatic motive-catalog activation, portfolio engine, AI readiness or later human-gate approval.
- Prior main receipts core245+3/dev20/visual70 are baseline only; three additional-suite failures and parser ERROR output remain disclosed.

## Verified implementation progress
- Initial regression command ran32 tests (existing tests were accidentally inherited twice), with exactly6 new failing groups. Corrected discovery to8 independent new tests.
- First implementation bootstrap failed because actual Characters store JobRoles, not only Job subclasses. Shared normalization now accepts the existing JobRoles base; no permissive substitute.
- Eight real-world acceptance tests pass (11.075s), including actor/partner request fingerprints, explicit request ERROR, typed collision, and energy20->30 execution.
- Existing13 production Goal tests PASS (22.518s). Final8 acceptance tests PASS (13.036s); canonical core253+3/dev20/visual70 all PASS with every batch exit0 and no failures/errors/skips. Eight committed source hashes match the tested snapshot.
- Fresh independent reviewer and repository architecture/planning/world/test-CI roles found no established blocking concern in supplied source/evidence. No independent checkout/command/nativeZcode review is claimed.
- Ruling: combine the dependent completion, scoring and inventory repairs into one source commit6744d37, with separate evidence closeout; shared bootstrap and error interfaces required integrated verification. The original RED and failed-bootstrap receipts remain retained.
- Additional102-check suites still have the same3 known exact-base failures; parser ERROR stderr persists. Neither is counted as passing.
- PR828 is attached and targets main; implementation head6744d37 passed all8 hosted checks. Final delivery/hosted state is tracked by GitHub, with no merge authorization.
- Full receipt: docs/testing/evidence/issue-821-review.json. Boundaries and accepted contracts: docs/testing/issue-821-goal-contracts.md.
