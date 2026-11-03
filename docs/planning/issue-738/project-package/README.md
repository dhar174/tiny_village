# Phase3 approval: owner-authenticated Project10 synchronization

Dhar174 explicitly approved #738 under [D034](https://github.com/dhar174/tiny_village/blob/6890cde019ba93f748454ddfb4709b3fd8440e35/docs/decisions.md#d034--approve-phase-3-gate-738-with-retained-issue-backed-follow-ups). #738 and parent #733 are CLOSED/completed in that order, after native prerequisite checks. The13 originalPhase3 tickets and runtime requirements remain OPEN with their acceptance/relationships intact. This bundle synchronizes the recorded decision and readiness; it performs no runtime implementation.

Main is pinned to `16528f2130fc4b890558a1fccd590ad29e9bae7a` (merged813). Source inputs freshly cover484 issues,332 managed rows,333 source guards and24 unchanged candidate heads/states. All12 custom fields and six delivery views retain their schema. The known Project10 node ID is `PVT_kwHOAE_6m84Bm2UC`.

## Fresh read-only plan

Extract this ZIP in a new folder and open its inner `tiny-village-phase3-approval` directory in PowerShell with Python3.12 and your existing authenticated gh. Run:

```powershell
New-Item -ItemType Directory -Force ..\phase3-evidence
python sync_project.py --prepare --plan ..\phase3-evidence\phase3-plan.json
```

Return the complete `phase3-plan.json` and printed approval digest for review. Do not apply an unseen live delta. Source/main drift stops preparation and requires refreshed inputs. Keep outputs outside the frozen package. Do not reuse the previous737 plan/checkpoint. No credentials need to be shared.

The prospective comparison against the prior verified board predicts23 item operations:738 DecisionPending→Approved and ReadinessHuman review→Tracking, plus22 Phase4 rows ReadinessBlocked→Ready. All22 have no unsatisfied native/planned prerequisites after the explicit gate approval. Other fields/rows/views and all later pending gates remain unchanged. The exact list and per-row changes are in `validation.json`. This is a comparison against a historical authenticated board, not a fresh live plan. Preserve live human Scope/Review owner, Decisions, unrelated cards, Status and views through actual preparation/review.

## Apply only after actual plan review

After the exact plan/digest is reviewed, coordinate a quiet edit window and use a separate checkpoint:

```powershell
python sync_project.py --apply --plan ..\phase3-evidence\phase3-plan.json --checkpoint ..\phase3-evidence\phase3-checkpoint.json --approval REVIEWED_PHASE3_DIGEST --edit-window-confirmed
python sync_project.py --inspect --plan ..\phase3-evidence\phase3-plan.json --receipt ..\phase3-evidence\phase3-inspect.json
```

Run separately and stop on nonzero. If a reviewed plan has zero operations, skip apply. Return the complete plan/checkpoint/inspection and terminal results. The API has no compare-and-swap; a concurrent edit can race a remote write. Pending/unacknowledged writes require inspection/reconciliation and a newly reviewed plan/checkpoint, without blind retry. Never substitute an unreviewed digest.

## Provenance and boundaries

The two publisher/receipt Python files and capture helper are unchanged copies of the merged reviewed737 tooling. The historical737 builder and its explicit “738 approval is outside this737 package” guard remain unchanged. The coordinator used that builder with Pending738 to regenerate the source-backed base, then a separate recorded D034 transformation changed Decision738 and recomputed readiness from actual issue states, native/planned prerequisites and explicit human decisions. Its exact computation is retained as evidence text in `evidence/decision-overlay.py.txt`; it performs no GitHub mutation. Other managed values are independently compared against the prior full332-row preview.

`validation.json` proves artifact/scope/preservation consistency, not live publication. The prior full receipt/plan/desired are retained under evidence, with exact original/reproduced provenance from the737 final record. Current cloud Project read still returns FORBIDDEN. A read-only capture initially hit upstream503; that failed attempt is retained, and one successful read-only retry supplied the snapshot. No publication failure was relabelled success.

Keep main unchanged while preparing/applying this package. The approval/closure evidence is in PR814; do not merge it during the local cycle. A later source merge requires new source comparison and, for a claim of current-main board verification, fresh authenticated readback. Historical receipts retain their original revisions.
