# TASK021 — Evidence-based backlog triage (#734)

**Status:** In Progress — awaiting consolidated publication review  
**Added:** 2026-10-21  
**Updated:** 2026-10-21

## Original request

“Let's begin #734.” Then: “Implement the proposed plan.”

## Plan and constraints

Approved plan: [#734 triage](../../docs/superpowers/plans/2026-10-21-issue-734-backlog-triage.md).
Review the complete ledger and exact mutation proposals before changing issue
metadata; approve individual duplicate closures. Preserve assignees; use
dhar174 as accountable ledger owner where unassigned. D026 records approval
and these choices. No runtime fixes or Phase 3 gate approval.

## Progress

| Step | State | Evidence |
|---|---|---|
| Baseline and snapshots | Complete | Branch codex/issue-734-backlog-triage from main 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7 |
| Original and current ledger | Complete | 475 issues/158 comments, 37 PRs, 50 review origins; original 261 plus 46 later open |
| Six mandatory audits | Complete with failures retained | Four scoped passes; #602/#609 fail with real objects; 12 existing tests pass; isolated map/animation failure |
| Integrity checks and PR | Complete for review | PR #797; independent review complete, #530 acceptance restored, catalog checks pass |
| Approved issue publication | Awaiting later review | No issue mutations authorized before package review |

## Execution record

### 2026-10-21

- Refreshed main and created a clean focused branch. Environment observations
  current; unrestricted HTTP policy enforced; configured GH_PROJECT_TOKEN
  binding ready. This does not establish Project API authorization.
- Pre-flight: snapshot → ledger → mutation proposals → reviewed publication;
  issue state and proposed disposition must remain separate throughout.
- Execution scope: this is documentation/evidence work, so use existing tests and scratch
  probes, without new production behavior or a new test framework. Failures are
  findings, not a request for out-of-scope fixes. Cost if wrong: follow-up
  implementation must add any missing durable regression coverage.
- Workspace choice: the approved focused branch in this clean cloud checkout provides
  task isolation; no extra linked worktree or dependency reinstall. Cost if
  wrong: work must be relocated if concurrent tracked edits appear.

- Cataloged 307 issues with explicit per-issue acceptance, source/discussion links,
  accountability, preserved native hierarchy and proposed scheduling. Inventoried
  all 37 open PR heads for later reconciliation. No issue mutations performed.
- Retained all 12 captures: 5 zero exits / 7 nonzero exits; 12 existing focused
  tests pass, four mandatory probes pass their scope and two expose real defects.
  Four initial setup/probe errors remain recorded; corrected runs are not a claim
  that the original attempts passed. No timeouts or runtime fixes.
- Probe correction: frame-analysis suggestions below budget are existing policy; correct
  the probe to distinguish stage hints from over-budget warnings. Cost if wrong:
  a later performance-policy task must revisit what users should be told.
- Isolated map/animation failures with two real Characters. Map failure is the
  reset LocationManager, not evidence of another global GraphManager deadlock.
- Three new remediation drafts and 80 missing-parent additions await review.
  Narrow closure proposals: #355/#482/#594/#596/#597; duplicate proposals:
  #351→#350 and #382→#380. #602/#609 stay open. No state changes approved yet.
- Project read still denied HTTP 403; #737 owns pending synchronization.

- Independent fresh-context review found no Critical/Important issue and checked
  snapshot/proposal/CSV/Markdown parity. Preserved #530's omitted named-constant
  acceptance through a failing→passing data check; full catalog checks pass.
- Final: fixed #530 scope omission — named-constant criterion RED→GREEN across
  ledger, proposed annotation and Markdown; complete catalog validation passes.
- Final: minor (deferred): validator full artifact-hash/field parity automation;
  all current hashes and canonical/readable parity independently verified.
- Final: minor (deferred): reusable Markdown renderer; current rendered package
  verified, but one-shot generator remains scratch-only.

- Published review PR #797: https://github.com/dhar174/tiny_village/pull/797.
  It references #734 without an automatic closing keyword. Full completion
  still requires human review, approved issue publication/readback and merge.
  No issue/Project metadata or issue state was changed.

### 2026-10-21 — PR #797 review correction

- Investigated [Codex comment](https://github.com/dhar174/tiny_village/pull/797#discussion_r4342402200)
  from review 5606749353. It correctly identified that only the two snapshot
  hashes were checked; other publication artifacts could change undetected.
- The validator now checks every full-manifest hash and exact input-file
  coverage, reporting changed/missing/unlisted paths before catalog success.
  Manifest self-hashing, derived validation.json output and bytecode caches are
  explicit exclusions. Other field-parity/renderer enhancements remain deferred.
- Disposable-copy verification: old validator falsely accepted ten mutations;
  corrected validator accepts the pristine bundle and rejects all ten altered
  or incomplete copies. Exact before/after output and scratch driver are retained
  in evidence/manifest-validation-review.json. No original captures modified.
- Rehashed the bundle and refreshed its validation receipt. No gameplay/runtime
  changes, triage publication, issue closures or phase approval were performed.
