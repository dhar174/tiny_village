# [TASK006] - Plan startup timing investigation (#779)

**Status:** Pending — awaiting plan approval

**Added:** 2026-10-15

**Updated:** 2026-10-15

## Original Request

"Please develop an organized plan for investigating #779 and wait for plan approval."

## Approach

Read the current issue, saved failing runs, diagnostic results and existing
test/supervisor source. Plan bounded, sequential diagnosis that separates
imports, graph operations, process shutdown and aggregate deadlines. Read-only
planning does not authorize executing experiments or making a production fix.

## Plan

[Proposed investigation plan](../../docs/superpowers/plans/2026-10-15-issue-779-startup-investigation.md).
Human decisions and approval status belong in
[the single decision record](../../docs/decisions.md).

## Progress Tracking

**Overall Status:** Planning complete — execution not started.

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 6.1 | Review issue, constraints and historical evidence | Complete | 2026-10-15 | #779 open; main remains 93079fd; no reproduction runs |
| 6.2 | Draft and self-review investigation plan | Complete | 2026-10-15 | Hypotheses, bounded experiments, outputs and stopping points specified |
| 6.3 | Obtain explicit plan approval | Pending | 2026-10-15 | Do not execute the plan before approval |
| 6.4 | Execute investigation and report findings | Not started | 2026-10-15 | Diagnostics only; production remedy requires separate review |

## Progress Log

### 2026-10-15

- Read #779 and retained post-merge logs/results; inspected existing tests and
  supervisor without rerunning checks or creating executable diagnostics.
- Identified two planning distinctions: child versus aggregate deadlines, and
  integration fixtures overriding the runner's shared Matplotlib cache path.
  Neither observation establishes the cause of the timing failures.
- Saved the proposed plan with fixed scope, unchanged baseline limits,
  per-attempt outcome accounting, a diagnostic budget and repair review boundary.
- Scope and review decisions remain in docs/decisions.md; #721/#722 and Project
  field status are unchanged. Await explicit plan approval.
