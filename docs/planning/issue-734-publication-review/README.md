# PR #798 — Stale metadata and completion review corrections

Date: 2026-10-21 America/New_York. Review baseline:
`244daf3f800dcba5b0cddf8ccf06bc30c9072431` (the user updated #798 from main
before this fix; the two reviewed scripts are unchanged from reviewed 6969cbf).

## Findings and changes

- [Stale publication metadata](https://github.com/dhar174/tiny_village/pull/798#discussion_r4343068091):
  preflight formerly compared only bodies and selected parents. It now compares
  the frozen reviewed state/state reason, assignee identities, native parent,
  dependency identities and each proposal's `base_updated_at`. Ordering/display
  differences in connections do not masquerade as identity changes. Timestamp-only
  drift also stops for inspection. A failed preflight cannot become `before.json`.
- Publication rechecks metadata immediately before annotation, parent, dependency
  and closure writes. Readbacks checkpoint its own timestamps and narrowly approved
  relationship/state operations. Checkpoint timestamps never authorize unrelated
  state/assignment/dependency drift. An interrupted write without its readback
  checkpoint requires inspection, rather than silently adopting the live state.
- [Premature completion](https://github.com/dhar174/tiny_village/pull/798#discussion_r4343068095):
  complete.py now validates the complete published issue set and all existing
  body/assignment/relationship/state, new-issue and cycle invariants before
  either the completion comment or closing #734. The same validator checks
  the final state afterward, with the expected #734 close as the only change.
  The exact 307-issue annotation set is also checked against the approved proposals.

No original issue bodies, archived snapshots, checkpoint, publication receipts
or human decisions were rewritten. The historical pre-publication metadata was
also compared to the reviewed snapshot: [307 matches, zero differences](historical-metadata-audit.json).
The original completion evidence stands; this correction does not replay it or
supply a new phase approval. #703's later merge/state changes are separate from
these historical captures.

## Verification and limits

[Results](results.json): **78 offline cases pass**, comprising the same 39
scenarios against original and fixed code. The fixed code rejects **33 stale
inputs with zero attempted writes** and preserves six valid execution controls.
Cases cover state/reason, assignees, parent, dependencies, timestamp-only and body
changes before preflight or an annotation batch; resume after owned annotations,
parents, dependency additions and a closure; and external changes after those
owned operations. Completion cases cover edited/reassigned/reparented issues,
removed prerequisites, reopened completions, closed gates, changed new issues,
and an existing completion comment.

The publisher's valid controls stop at the first controlled mutation, rather
than replaying bulk writes. Completion controls assert exact ordering: live
read/validation → comment (unless present) → close → live read/validation.
Stale completion cases attempted two writes with the original code and none
with the fixed code. Original publication probes demonstrate the missed
metadata checks; existing body rejection is retained as a control.

The [scratch driver](probe-driver.txt) executes the actual run/completion bodies
against retained production API snapshots. API boundaries are controlled counters
and failure injection; **zero GitHub issue/Project writes** occurred. The already
validated bundle check is bypassed only inside these isolated fixtures; the real
87-input hash validator runs separately. No new repository test suite or runtime
feature was added. These probes do not claim end-to-end publication through live
GitHub or an atomic transaction across API requests.

All four attempts are retained under captures/: initial harness failure (missing
DEVNULL on the subprocess boundary), passing 66-case matrix, extended-resume
fixture failure (the fixture omitted #355's already-applied parent from its
checkpoint), and corrected passing 78-case matrix. The two harness/fixture errors
were corrected without changing production code; neither is passing evidence.
Each child used the existing durable subprocess capture with a 20s diagnostic
and 25s parent deadline. No attempt timed out. Exact source/capture receipts and
the earlier passing 66-case result remain alongside the final result.

The original frozen bundle still verifies all 87 artifact hashes. The separate
publication manifest is refreshed for the two fixed scripts and updated README;
archived data files remain byte-for-byte unchanged. Review evidence has its own
manifest, excluding that manifest itself.
