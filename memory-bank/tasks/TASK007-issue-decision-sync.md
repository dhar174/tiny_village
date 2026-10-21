# [TASK007] - Synchronize review decisions to GitHub issues

**Status:** Completed

**Added:** 2026-10-15

**Updated:** 2026-10-15

## Original Request

"Can you update the actual repo issues (722 and children) with our decisions?
Also, should they be closed now?"

## Approach

Inspect actual GitHub hierarchy, publish the single decision Markdown record,
and add targeted status/scope references to existing issues. Preserve their
original bodies/checklists and current states. A completed investigation cycle
does not resolve #779 or supply the required human gate approvals.

## Implementation Plan

1. Read current issue hierarchy and affected future work.
2. Publish decision/evidence documentation as a draft PR that closes no issues.
3. Synchronize scope/status notes without altering acceptance requirements.
4. Verify issue states, dependency order, hierarchy and published bodies.

## Progress Tracking

**Overall Status:** Complete — 18 issue bodies verified; no issue states changed.

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 7.1 | Inspect native hierarchy and states | Complete | 2026-10-15 | #717 parents #718–#722; #722 has no children |
| 7.2 | Publish canonical decisions and evidence | Complete | 2026-10-15 | Draft PR #780, attached; no automatic issue closure |
| 7.3 | Synchronize related issue scope/status | Complete | 2026-10-15 | 18 dedicated blocks; all previous bodies and states preserved |
| 7.4 | Verify relationships and closure guidance | Complete | 2026-10-15 | #779 is child/blocker of #721; #721 blocks #722; #718–#720 remain closed |

## Progress Log

### 2026-10-15

- Confirmed #722 is a gate with no native children. The actual Phase 1 parent
  is #717; #718–#722 are its sub-issues.
- Read the later carrier tasks for Windows/profile validation, optional live
  models and real gameplay scenarios. Preserve their current phases and gates.
- Published/attached draft PR #780 with the decision file, review reports and
  investigation plan. The PR contains documentation only and closes no issues.
- Canonical decision snapshot:
  https://github.com/dhar174/tiny_village/blob/cce26ac9a2aad64c3534fce2a5e7a11f5fb1ccf0/docs/decisions.md.
- Issue changes use a dedicated Markdown block and read-back checks; previous
  acceptance criteria and historical text remain unchanged.
- Updated #717–#722, #779, #773–#776, #724–#726, #654, #751, #658 and #585.
  All 18 bodies were read back; original text outside the new block and each
  issue's state match the pre-update values. No closure or reopening occurred.
- #779 previously had no parent; added it as a native sub-issue of #721 without
  replacing an existing parent. Verified child and blocker relationships and
  transitive review/gate ordering through #722. #722 still has no children.
- Draft PR #780 is attached; verified open/draft status and documentation-only
  source scope. Project fields were not changed or claimed synchronized.
- Verification receipt: /workspace/scratch/issue-721-issue-sync-2026-10-15/verified.json.
