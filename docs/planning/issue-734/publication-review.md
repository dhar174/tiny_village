# Proposed issue publication — review required

**Not approved or applied.** The complete, exact marked annotations, snapshot update timestamps/body hashes, parent proposals and state operations are in [proposed-issue-updates.json](proposed-issue-updates.json). They are data, not a script that writes to GitHub.

## Specific state changes for review

| Issue | Proposed operation | Evidence and boundary |
|---|---|---|
| [#351](https://github.com/dhar174/tiny_village/issues/351) | CLOSED:NOT_PLANNED (duplicate); duplicate of #350 | Exact duplicate: identical original issue body and same original review-comment URL as #350. No unique acceptance was found in either issue discussion. |
| [#355](https://github.com/dhar174/tiny_village/issues/355) | CLOSED:COMPLETED | Queue probe passes and both actual Plan.execute heap-pop consumers unpack four fields. Replan retains dependencies but uses id(action) for ties; no cross-process deterministic-order claim. |
| [#382](https://github.com/dhar174/tiny_village/issues/382) | CLOSED:NOT_PLANNED (duplicate); duplicate of #380 | Exact duplicate: original #380 and #382 bodies are byte-for-byte identical and neither has an issue discussion. Preserve #380 as canonical. |
| [#482](https://github.com/dhar174/tiny_village/issues/482) | CLOSED:COMPLETED | Real SDL-dummy rainy panel rendering changes actual Surface pixels and exits zero. This proves the reported missing-constant path only, not weather behavior/audio. |
| [#594](https://github.com/dhar174/tiny_village/issues/594) | CLOSED:COMPLETED | All ten #113–#122 native parents are #594; #598/#599 also have parent #594 and #594 has parent #56. Its organizational acceptance is already met, independently of feature implementation. |
| [#596](https://github.com/dhar174/tiny_village/issues/596) | CLOSED:COMPLETED | One actual SDL-dummy game-loop frame records all five timing stages. Bounded/disabled collection passes the corrected probe. The same frame logs map/animation/recovery defects; this is not healthy-gameplay or frame-budget evidence. |
| [#597](https://github.com/dhar174/tiny_village/issues/597) | CLOSED:COMPLETED | Corrected probe validates empty/normal/slow/dominant-stage samples, ten-record eviction and disabled collection. Current policy can suggest dominant-stage optimization below budget; that is documented behavior, not a measured performance failure. |

All other issue states remain unchanged, including #602, #609, #734, phase parent #733 and human gate #738. #654 stays closed under its existing completed scope.

## Metadata changes for review

- Append/prepend one marked triage annotation to each of 307 cataloged issues, preserving every existing body section, decision and assignee.
- Add 80 parent links only where the snapshot has no parent; retain every existing native parent, including the 23 open children of historically closed parents.
- Existing blockers remain intact. Proposed execution-prerequisite text is not a native dependency write. Canonical-topic links do not silently reparent existing children.
- Create the three separately scoped [follow-up issues](proposed-new-issues.md) only after approval, then link their approved blockers using real issue IDs.
- No mass assignments, labels, priority/scope resets, PR rebases/merges or automated human decisions.

## Publication procedure after review

1. Record the human decision and exact approved subsets in the single decision file. Approval must identify duplicate closures explicitly; excluded or deferred changes stay pending.
2. Refresh each affected issue body, state, assignees, parent and dependencies. Compare body hash and updatedAt with the proposal. Preserve intervening content and re-review material proposal changes; never overwrite a stale body.
3. Apply an idempotent marked annotation only once. Add a parent only if still unparented and still approved; never use replace-parent behavior. Add approved new-issue/dependency links after resolving real database IDs and rechecking cycles.
4. Checkpoint each successful operation; on failure retain the partial state and read back before resuming. Duplicate/completed closures occur only in the approved subset.
5. Verify bodies, native relationships, state reasons and unchanged assignees through API readback. #734 closes only after documentation merge and reviewed publication finish; the PR deliberately uses Refs #734.

## Project handoff

The current Project 10 read returns HTTP 403, retained in project-access.json. #737 owns later Phase/Sequence/Readiness/Depends on synchronization and PR ownership/merge ordering. Retain pending configuration explicitly; do not rerun the old publisher or reset prior human decisions.

