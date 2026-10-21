# #734 — Approved publication and verified completion

Dhar174 approved the complete review package after #797 merged. [D027](../../decisions.md#d027--approve-and-publish-the-734-backlog-triage-package)
records that approval, including the specific duplicate closures. PR #797 is
verified merged at `2c3bc5753b15aad4a1e345c020c5872d6ba8b25c`.

**#734 is closed/completed**, with its [completion comment](https://github.com/dhar174/tiny_village/issues/734#issuecomment-6328866835)
and independent API readback retained in [completion.json](completion.json).

## Published and read back

| Operation | Verified outcome |
| --- | --- |
| Marked issue annotations | 307; every original body retained byte-for-byte outside the new block |
| Missing native parent links | 80 existing issues; no existing parent replaced |
| Narrow completed closures | #355, #482, #594, #596, #597 |
| Approved duplicate closures | #351 → #350; #382 → #380; both not planned |
| New scoped remediation issues | #799, #800, #801; three new native parent links |
| Remediation dependencies | Seven new edges, all read back with real issue IDs |
| Existing assignments/dependencies | Preserved; no mass assignments or native prerequisite removal |
| Relationship cycles | Zero in the complete final issue graph |
| Completion closure | #734 closed only after publication verified and #797 merge confirmed |

The preflight read all 475 current issues and found no material body or parent
drift from the approved 307 proposals. Five-issue annotation batches reread
current bodies immediately before ordered writes and checkpoint successful
results. Parent additions rechecked that children were still unparented and
that target capacity remained available. New issue markers/checkpoints prevent
duplicate creation on resume. Each new dependency was cycle-checked before
writing. Seven explicit state changes were individually read back. Final
independent checks cover all 307 original bodies, assignments, parents,
prerequisites and preserved states, the specific closures, new issues and edges.

## Scoped remediation handoff

| Issue | Parent | Blocked by | Blocks |
| --- | --- | --- | --- |
| [#799](https://github.com/dhar174/tiny_village/issues/799) — Stable Character/Location identity and action targets | #222 | #738 | #220, #602 |
| [#800](https://github.com/dhar174/tiny_village/issues/800) — Map location-manager lifecycle | #56 | #738 | #658 |
| [#801](https://github.com/dhar174/tiny_village/issues/801) — Animation update lifecycle | #56 | #738 | #761 |

These are scheduled remediation issues, not runtime fixes or approved identity
and interface designs. #602/#609 remain open with the real failed evidence.
Phase parent #733 and human gate #738 remain open; no later human gate is
approved. Project 10 configuration remains #737's disclosed pending handoff
(last retained access observation: HTTP 403); this publication made no Project
writes. The new issues still need the corresponding Project configuration in
that task. D014/D015's original product and final AI requirements are intact.

## Evidence and reproduction

- [Immutable review input](../issue-734/README.md): unchanged from merged #797;
  all 87 manifest hashes still pass. Its proposed/not-applied flags, original
  snapshots and zero-mutation validation receipt describe the historical review
  stage, not the current live publication status.
- [Preflight](preflight.json), [before](before.json), [checkpoint](checkpoint.json),
  [publication readback](after.json), [verification](verification.json), and
  [final readback after completion](final-readback.json). Bodies are retained
  intentionally for preservation checks and recovery; no credentials captured.
- [Publisher](publish.py) and [completion/readback](complete.py): exact bounded
  publication scope using the existing gh credential. The applied publisher
  was committed at `3449f0f`; the final publisher/completion versions additionally
  refuse further writes once completion exists. The obsolete last pending annotation batch was removed
  after verification. Successful operations remain in the checkpoint.
- [Artifact hashes](artifact-hashes.json): covers this publication folder except
  itself and bytecode caches. This is separate from the frozen input manifest.

Executed from repository root:

```sh
python docs/planning/issue-734/validate_bundle.py
python docs/planning/issue-734-publication/publish.py
python docs/planning/issue-734-publication/publish.py --apply
python docs/planning/issue-734-publication/complete.py
```

The first two commands validate/read only. Apply is a resumable one-time action
under D027, not a general backlog regeneration command. Completion now prevents
further applies; do not reset checkpoints or rerun the old Project publisher.
The initial verification receipt precedes #734's completion closure; completion
and final-readback receipts provide the subsequent authoritative state.

## Next ordered work

Under [D028](../../decisions.md#d028--retain-pr-703-as-the-735-utility-consolidation-carrier),
#735 is implemented in the selected existing [PR #703](https://github.com/dhar174/tiny_village/pull/703),
head `c65496d828ae3d45d17226ce013d2b4355682d20`: rebased onto main, one utility
test module, documented retained behavior and core CI inclusion. #714 is closed
as superseded with a crosslink. [Readback](utility-handoff.json) confirms all
eight checks green. Local validation is 51 core tests + three CLI probes and
18 CI-runner tests; exact evidence/source mapping is in #703. #735 remains open
until #703 merges. #736 reconciles the older functional-test PRs, then #737
handles PR ownership, merge ordering and Project synchronization before #738's
separate human review.
