# #737 final readiness package

The initial authenticated Project10 readback is verified. #806 and #737 are now CLOSED/completed, in that order after the receipt. This completed-stage bundle regenerates all332 managed rows from those actual closures, retains the exact initial evidence, and keeps #738 OPEN/Pending. **Final readiness has not yet been published or independently read back.**

Merged tooling/main: `dc9fcc023b4ee37b3efd579195caaab80993c64a`. Project10 node ID: `PVT_kwHOAE_6m84Bm2UC`. Authority: D032/D033; no new runtime/dependency code or phase approval. The earlier `issue-737`, `issue-737-closeout` and `issue-737-publication` packages remain unchanged. Python tooling here is byte-identical to the reviewed merged publication tooling. Stage-specific regression scripts remain in that original publication package; this bundle validates completed-stage artifacts directly. Historical test/probe reports remain labelled evidence.

## Owner: fresh read-only preparation

Extract this ZIP into a **new directory**, separate from the initial publication package. Its inner directory remains `tiny-village-project-737-publication` because it uses the unchanged merged packager. Open that extracted directory in PowerShell with Python3.12 and your existing authenticated `gh`. Keep evidence outside the frozen bundle:

```powershell
python validate_package.py
New-Item -ItemType Directory -Force ..\final-evidence
python sync_project.py --prepare --plan ..\final-evidence\final-plan.json
```

Stop on nonzero. Return the complete `final-plan.json` and printed approval digest for concrete delta review. The prospective review against the initial verified board predicts only Readiness changes: #806 Ready→Tracking, #737 Blocked→Tracking, #738 Blocked→Human review. Actual live drift, including GitHub Status automation following closures, must be reviewed in the fresh plan; no Status change is requested. All332 rows and six delivery views are regenerated, preserving populated Scope/Review owner, human Decisions, stable keys, assignments, parents, runtime blockers and unrelated board data.

## Owner: after concrete live-plan review

Coordinate a quiet edit window; the remote API has no compare-and-swap. Use only the exact newly reviewed digest with a separate checkpoint:

```powershell
python sync_project.py --apply --plan ..\final-evidence\final-plan.json --checkpoint ..\final-evidence\final-checkpoint.json --approval REVIEWED_FINAL_DIGEST --edit-window-confirmed
python sync_project.py --inspect --plan ..\final-evidence\final-plan.json --receipt ..\final-evidence\final-inspect.json
```

If the reviewed plan has zero operations, skip apply and run inspection. Return the complete plan, any apply checkpoint and fresh inspection receipt plus terminal result. Receipt validation independently recomputes source bindings, desired state and preservation. A source/main change requires fresh preparation; do not reuse an initial-stage plan/checkpoint or merge source changes during the local cycle. No credentials are needed in the response.

## Completion boundary

`validate_package.py` verifies initial publication evidence, actual ordered issue closures, the full cohort and completed-stage generated values. It reports `prior_publication_verified_final_readback_pending`; it does not establish final live publication. That still requires the fresh owner-local readback. Failed/ambiguous writes require inspection and a newly reviewed plan/checkpoint, without blind retry.

#738 remains OPEN/Pending; #733 remains OPEN and Phase4 blocked. Use `human-walkthrough.md` after final readback. The human must separately choose Approved / Changes requested / Deferred; record a subsequent human choice only in `docs/decisions.md`.
