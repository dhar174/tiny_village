# #737 ownership and Project synchronization review package

**Ready for disposition/configuration review; Project publication is blocked.**
This package inventories26 open PRs, investigates the five direct handoffs, and
previews332 issues: the approved328-issue cohort plus four scoped follow-ups.
It changes no game runtime, dependency locks or UI. #737 and #738 remain open;
#738's Decision remains Pending. Execution scope is approved under D031.

## Review entry points

- [Five disposition recommendations](disposition-review.md), including alternatives,
  evidence and the review-before-closure boundary.
- [PR ownership/order matrix](pr-matrix.md); `pr-matrix.json` retains full heads,
  changed files, reviews, checks, closing references, prerequisites and acceptance.
- `project-preview.json`: all12 custom fields for every managed issue and all six
  saved-view specifications. **This is desired configuration, not a live delta.**
  Project403 prevents capturing current before-values; `--prepare` must produce
  the concrete live before/after plan for review before applying anything.
- `issue-matrix.json.gz`: full ownership, parent/dependency/readiness/sequence
  mapping. `snapshot.json.gz` retains484 source issues and26 candidate PRs.
- [Validation and limitations](validation-report.md).
- [Portable package](portable-package.zip): Python standard-library tools, inputs,
  evidence and hashes; GitHub CLI is needed for online operations.

Accountable owner is dhar174; Codex supplies technical review; dhar174 retains
human acceptance. Existing GitHub assignees and review requests are preserved.
These roles are not submitted reviews. GitHub does not allow a PR author to
request their own review.

## Published coordination changes

| Issue | Scope / order | Native parent and ownership blocker |
|---|---|---|
| #806 | Phase3: restore authenticated Project access and verify approved publication | Child of #737; after approved #732; blocks #737. |
| #807 | Phase9: retain useful #709 helper contracts, including actual unavailable-capability reporting | Child of #585; after #584/#768; blocks #585. |
| #808 | Phase5: distinguish valid empty context from failed goal/needs calculations | Child of #137; after #743/#746; blocks #137. |
| #809 | Phase4: retain useful #715 graph-query coverage with real shared state | Child of #222; after #738/#799; blocks #222. |

These four parents and11 prerequisite edges were added and read back. Creating
them does not activate implementation or approve the PR recommendations. All
original issue bodies, existing assignees and parents were preserved. No PR was
closed, rebased, rewritten or merged by this work.

Runtime carry-forward remains explicit: #799 stable identity/action provenance;
#800 location-manager lifecycle; #801 animation lifecycle; #802 Goal/Plan
contracts; #803 motive contribution; #741 replanning; #774 utility/falsy-ID debt.
Their native relationships and unchecked requirements remain intact.

## Delivery order and field policy

The immediate order is #806 publication evidence → #737 completion → separate
#738 human walkthrough. Future runtime work retains its phase gates. Phase4's
main chain places #799 before #220 (candidate #696), then #740 and #802 before
#218 (candidate #689), followed by #741/#602 and the engineering/human reviews.
The matrix also retains independent Character/query work and later event,
economic, persistence, presentation and AI tasks; sequence is not a substitute
for an explicit dependency or activation decision.

- Stable Task keys are preserved for the original86 roadmap items; other rows
  use `ISSUE-<number>`. Sequence numbers are proposed from a stable topological
  order within the existing phases, with phase gates last. Existing sequence
  values are not overwritten without a reviewed live plan.
- `Depends on` labels **native prerequisites** separately from **planned
  activation**. Proposed activation is not silently converted into a native edge.
  Parent-child relationships do not become prerequisites.
- Readiness evaluates actual issue state and recorded human decisions. Closing a
  gate alone never approves it. Closed scoped tasks use Tracking; deferred scope
  remains Deferred. Open runtime gaps remain unresolved, not Done.
- `approval-records.json` reflects D011/D012 and D023–D025, including accepted
  planning tasks #727–#729. No decision is inferred from an automated check.
  New human decisions belong only in `docs/decisions.md`.
- Preserve populated Scope/Review owner, stable keys, all non-managed fields and
  unrelated items/views. Surface conflicting Decisions, archived/duplicate rows,
  schema changes and an open issue with a Done-like Project Status for review.
  Status itself is not automatically rewritten or treated as completion evidence.
- Retain12 custom fields and six existing views. Missing/type-conflicting fields
  or missing views require explicit schema review; the updater does not create
  them. View deltas include layout/filter/grouping/sorting, preserving existing
  field visibility. Original “View1” and unrelated views are untouched.

## Local preparation and publication

Extract the portable archive and open its `tiny-village-project-737` directory.
Use Python3.12 and an already authenticated `gh`. Project calls select the
configured `GH_PROJECT_TOKEN` if available, without displaying its value.
Native Windows execution is not yet verified; the lock uses platform-specific
imports inside the relevant branch rather than importing Unix-only `fcntl`.

```shell
python validate_package.py
python test_sync_project.py
python test_build_preview.py
python sync_project.py --prepare --plan live-plan.json
```

`--prepare` performs no GitHub mutations. It checks all reviewed source signatures,
heads and main revision, reads the entire Project with pagination, then saves the
live before/after plan and prints its approval digest. Review that file, including
human edits and view changes. Keep Project edits paused during application:

```shell
python sync_project.py --apply --plan live-plan.json --approval <reviewed-digest> --edit-window-confirmed
python sync_project.py --inspect
```

The remote field API has **no compare-and-swap/version condition**. Target rereads
catch observed drift, but a concurrent same-field edit inside a mutation can be
overwritten without detection. `--edit-window-confirmed` acknowledges an agreed
window with no other Project writers; the local lock only serializes publishers
sharing that checkpoint. Offline tests explicitly demonstrate this limitation.

The checkpoint records confirmed operations. A failed/unacknowledged mutation is
never retried automatically: inspect actual state and prepare a new reviewed plan
with a **new checkpoint path**. Confirmed checkpoints can resume the same exact
plan without replaying writes. Inspect verifies sources, item fields and view
configuration, and returns nonzero when incomplete; it never closes #737/#806 or
approves #738.

If any source/head changes, stop and review it. The following commands refresh
local artifacts only, producing a new review bundle, not new approval:

```shell
python capture_snapshot.py
python package.py --rebuild
```

Unexpected PRs, missing owners, changed cohort, cycles and ordering conflicts
stop regeneration rather than silently expanding scope. The coordinator's own
review PR is retained separately and excluded from the26-PR candidate cohort.
Recheck old runtime evidence against changed source before accepting dispositions.

Current `--prepare` passes the retained source guards and then receives Project
FORBIDDEN. Both the access receipt and actual CLI failure are retained. No live
before/after plan, Project write or successful readback is claimed. Finish the
reviewable work here while #806 retains that external authorization blocker.
