# #737 closeout: TinyVillage roadmap board

**Publication-stage package; live synchronization remains unverified.**
This separately refreshed package follows merged #810 at
`10569c06cc01075c70d5b39ec1b9e6804eb7f5cd`. The original
[`../issue-737/`](../issue-737/) folder is unchanged historical evidence.
D032 in `docs/decisions.md` records the approved human choices. This package
manages the same 332 issues, 12 custom fields and six delivery views, with the
24 remaining candidate PRs pinned by number and head in `review-scope.json`.

## Current delivery state

- #701/#704 were closed unmerged as superseded; retained implementation/test
  evidence is in merged #778/#785/#788/#810. Before/after API receipts are in
  `evidence/closeout/`. Existing heads and assignees were preserved.
- #709/#711/#715 remain open. Follow-ups #807/#808/#809 carry the approved split
  dispositions; verify and review replacements before any later supersession.
  Existing phase prerequisites, native parents and unresolved criteria remain.
- #737 was reopened because #806 still blocks authenticated board publication.
  #738 remains open, Blocked and **Decision Pending**. No phase approval occurred.
- Runtime gaps #799–#803, #741 and #774 remain open in their established order.
  Future dependent PR rebases happen only after their prerequisites land.

`pr-matrix.md` and its JSON retain ownership, activation gates, exact heads,
checks and required validation. Historical green checks do not establish
integration with today's main. Accountable owner/human acceptance: dhar174;
technical review: Codex. Existing assignments/review requests are preserved;
these role records are not submitted GitHub reviews or self-review requests.

## Run the read-only local preparation

Download/extract [portable-package.zip](portable-package.zip) and open the
`tiny-village-project-737-closeout` directory in PowerShell. Use Python 3.12
and your authenticated GitHub CLI. Do not paste or send a token. Project calls
select `GH_PROJECT_TOKEN` if configured, otherwise existing `gh` authentication.

```powershell
python validate_package.py
if ($LASTEXITCODE -ne 0) { throw "Package validation failed" }
python test_sync_project.py
if ($LASTEXITCODE -ne 0) { throw "Publisher tests failed" }
python test_build_preview.py
if ($LASTEXITCODE -ne 0) { throw "Ordering tests failed" }
python test_closeout.py
if ($LASTEXITCODE -ne 0) { throw "Closeout tests failed" }
python sync_project.py --prepare --plan live-plan.json
if ($LASTEXITCODE -ne 0) { throw "Preparation blocked; share the error, not credentials" }
```

Send `live-plan.json` and its printed digest for concrete delta review. Preparation
reads sources and the board but performs **zero GitHub mutations**. It may surface
conflicting human edits or missing schema; stop for review rather than guessing.
The original #810 publisher/snapshot should not be used for this refreshed cohort.

A merged closeout PR, changed issue body/state, new candidate or changed head
invalidates the exact source guard. The coordinator must refresh and review the
snapshot/package before another prepare; regenerating data does not approve it.
`capture_snapshot.py` followed by `package.py --rebuild` refreshes local artifacts
only. Same-count PR substitutions and changed heads are rejected against the
reviewed cohort. Only the two exact coordination branches listed in
`review-scope.json`, from the canonical `dhar174/tiny_village` repository, are excluded from candidate inventory; all other PRs count.

## Apply only the reviewed live delta

After concrete plan review, arrange a quiet board edit window and use the exact
digest. No other Project writers should edit during application. The remote API
has no compare-and-swap: a concurrent edit inside a write can be overwritten.
Local locking and rereads do not eliminate that limitation.

```powershell
python sync_project.py --apply --plan live-plan.json --approval REVIEWED_DIGEST --edit-window-confirmed
if ($LASTEXITCODE -ne 0) { throw "Partial/blocked publication; inspect before retrying" }
python sync_project.py --inspect
if ($LASTEXITCODE -ne 0) { throw "Publication is not verified" }
```

Preserve the plan, console outputs and checkpoint for independent readback.
Confirmed checkpoints resume without duplicate writes. An unacknowledged write
requires inspection and a newly reviewed plan with a **new checkpoint path**;
do not delete evidence or automatically retry ambiguous mutations.

The updater preserves unrelated items/views, non-managed fields and populated
Scope/Review owner. It rejects duplicate/archived rows, schema drift, conflicting
human Decisions and an open issue with a Done-like Status. Status is never
rewritten automatically. Planned activation is labelled separately from native
prerequisites; parents are not prerequisites. Existing approvals remain intact.

## Completion sequence and acceptance boundary

1. Review/apply the concrete publication-stage plan and independently read back
   all managed values and view layout/filter/grouping/sorting.
2. Only after that evidence, close #806, update #737's acceptance evidence/checklist
   and close #737. No closure is inferred from offline checks or #810's merge.
3. Refresh a separately reviewed final-stage package: completed #806/#737 use
   Tracking, #738 becomes Human review **with Decision Pending**, and runtime
   requirements retain their open states and phase gates. Prepare/review/apply
   that readiness delta and obtain final authenticated readback.
4. Begin #738's separate human walkthrough; Phase 4 waits for that decision.

This publisher does not close issues or approve gates. The current validator
accepts only the publication stage and rejects a switched completion flag: a
final-stage package requires actual authenticated receipts and separate review.
Windows-style manifest paths are tested, but native Windows execution and real
Project mutation/readback remain unverified until the local run supplies evidence.

## Evidence provenance

`evidence/historical-*`, core/cache captures and seven runtime probes are copied
unchanged from #810. Their successful and failed results remain historical.
Fresh closeout receipts/tests, source timestamps, reviewed heads and hashes are
separate. See [validation-report.md](validation-report.md). Rebuilding the ZIP or
passing Linux/offline tests supplies no publication, Windows or product approval.
