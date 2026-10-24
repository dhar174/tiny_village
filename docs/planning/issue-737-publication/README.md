# #737 authenticated publication tooling

**Frozen publication-stage bundle. Tooling PR #812 merged at `dc9fcc023b4ee37b3efd579195caaab80993c64a`.
Board publication and #806/#737 completion remain unverified; the next step is
owner-authenticated read-only preparation.**
Approval for this workflow is D033 in `docs/decisions.md`; it does not approve an
unseen live delta or #738. This package extends the merged #811 publisher with
independently recomputable inspection receipts and evidence-backed completed-stage
validation. The original `../issue-737/` and `../issue-737-closeout/` folders remain
unchanged. Copied probes/reports are historical, not fresh runtime executions.

## Coordinator: freeze after tooling merge

Keep the reviewed24 candidate numbers/heads and332 managed issues. Refresh against
actual merged main using `capture_snapshot.py`, run all four test scripts, then
`package.py --rebuild`. Supply the ZIP's checksum and immutable download link.
Artifact-only commits can remain on the publication branch while the owner runs
publication; do not merge a new source revision between prepare and apply.
Any main/source change invalidates an unused plan and requires a new reviewed
preparation. A verified receipt remains historical evidence of its exact revision.

Only the three explicitly listed coordination branches from canonical repository
`dhar174/tiny_village` are excluded from the candidate inventory. A fork using the
same branch name remains visible. New candidates or changed heads require review.

## Owner: read-only preparation

Extract the frozen ZIP; open `tiny-village-project-737-publication` in PowerShell.
Use Python3.12 and authenticated `gh`. Project calls select an existing configured
`GH_PROJECT_TOKEN`; never send credentials. Native Windows execution is not yet
verified. Run these commands one at a time and stop if any returns nonzero:

```powershell
python validate_package.py
python test_sync_project.py
python test_build_preview.py
python test_closeout.py
python test_publication_evidence.py
New-Item -ItemType Directory -Force ..\publication-evidence
python sync_project.py --prepare --plan ..\publication-evidence\publication-plan.json
```

Return `publication-plan.json` and its printed approval digest for concrete delta
review. It contains the actual board baseline and proposed changes. Do not apply
until that exact plan/digest has been reviewed. Missing schema, archived/duplicate
rows, conflicting human Decisions or Done-like Status on open requirements stop
preparation for resolution. Existing assignments, populated Scope/Review owner,
unmanaged fields/items/views and prior approvals must remain intact.

## Owner: approved apply and independent inspection

Agree a quiet board-edit window. The remote API has no compare-and-swap; rereads
and a local lock do not prevent a concurrent edit inside a remote write. Use the
reviewed digest in place of REVIEWED_DIGEST:

```powershell
python sync_project.py --apply --plan ..\publication-evidence\publication-plan.json --checkpoint ..\publication-evidence\publication-checkpoint.json --approval REVIEWED_DIGEST --edit-window-confirmed
python sync_project.py --inspect --plan ..\publication-evidence\publication-plan.json --receipt ..\publication-evidence\publication-inspect.json
```

Stop on nonzero. Return the plan, checkpoint, inspection receipt and console
outputs. Inspect uses fresh source reads before/after a fresh board read, performs
zero mutations and does not trust checkpoint success. The coordinator recomputes
receipt integrity, plan/source bindings, desired field/view values and preservation
of unrelated state. A success label alone is insufficient. Receipts are retained
observations, not cryptographic proof of provenance; live origin and the human
approval remain part of review. Do not share credentials.

`--receipt` works only with `--inspect` and must be outside the frozen package and
distinct from the plan, preview and checkpoint; the same checks cover its atomic
temporary file and resolved symlink target. Existing CLI modes remain supported.
Receipt files bind the observed project identity/state, source checks, timestamp,
plan/desired/project digests and remaining differences. Incomplete inspection exits2;
API/validation failure exits1 and cannot produce a verified receipt.

Confirmed checkpoints resume without duplicate writes. An unacknowledged write
requires inspection, reconciliation and a new reviewed plan with a new checkpoint;
preserve failed evidence rather than blindly retrying.

## Coordinator: closure and completed-stage readiness

After authenticated initial publication/readback and human delta review:

1. Complete/close806, then reconcile acceptance and complete/close737 with evidence.
   Preserve bodies, parents, assignments and unresolved requirements.
2. In a separately frozen completed-stage bundle, retain the reviewed initial
   `desired`, `plan` and verified `receipt` under `evidence/approved-publication/`.
   Record the Project10 node ID established by the authenticated initial read in
   `review-scope.json` as `project_id`; it is unknown in this tooling package.
   Set its stage to `completed`, with `publication_evidence` mapping
   those three keys to package-relative paths. These files become hashed artifacts.
3. Capture the actual closed/completed states and regenerate all332 rows. Validation
   requires the same prior managed cohort/candidate heads, initial receipts from
   before issue closure,806 closed no later than737, closed items using Tracking,
   and738 open/Pending with Human review readiness. Runtime gaps remain open.
4. Review the final readiness plan, apply with a separate plan/checkpoint, and run
   fresh inspection using `--receipt`, with paths prefixed `final-` instead of
   `publication-`. Review all changed rows and saved views, not only three cards.
5. Publish final evidence. After any later merge, refresh source inputs and perform
   another read-only comparison against actual main. New differences need review;
   a source merge does not retroactively erase earlier point-in-time evidence.

The completed-stage validator proves prerequisite evidence consistency; it does
not claim that the final readiness delta is already published. That requires its
own successful inspection. The tool never closes issues or approves a phase gate.

## #738 human walkthrough

Use `human-walkthrough.md` after publication and closure evidence is ready. Discuss
734–737, ownership/order, runtime gaps, applicable UX/UI/aesthetic records and known
limits. Present Approved / Changes requested / Deferred with recommendation.
Record the human decision only in `docs/decisions.md`; retain738 open/Pending until
explicit approval and keep Phase4 blocked. D033 supplies no such approval.
