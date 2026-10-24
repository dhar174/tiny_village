# TASK025 — PR ownership and Project configuration (#737)

Status: In progress. Updated 2026-10-23 UTC.

## Request and approved scope
Implement the approved #737 coordination plan; review all 26 PRs, investigate five direct handoffs, reconcile 328 issue-backed tasks, and prepare safe Project synchronization. Runtime changes and PR dispositions require their own review; #738 is a separate human gate.

## Progress
- Base: merged main 48213783e2ee0864594af4a3f26667e1269d394a (#805).
- Approved plan: docs/superpowers/plans/2026-10-22-issue-737-coordination.md.
- Snapshot, matrix and Project preview: pending.
- Project access: previously HTTP 403; recheck and retain current evidence.
- Publication and #737 completion: pending, never inferred from offline tests.

## Execution ledger
- Human choices are recorded only in docs/decisions.md; technical implementation notes remain here.
- Preflight: matrix issue keys consume the union of frozen roadmap86, triage307 and runtime5 (328 unique). Native blockers are distinct from canonical parents and proposed execution prerequisites.
- Technical ruling: final PR dispositions remain recommendations until reviewed; future-phase code/rebases stay with their owning issue. This avoids broadening the approved coordination scope.

## Verified implementation progress
- Captured484 source issues/26 candidate PRs; managed332 after four scoped remediations. Ownership/order matrix and12-field/six-view preview prepared;112 native edges/387 separate planned activation references, no cycles/order conflicts.
- Created #806 access blocker (parent737), #807 helper contracts (parent585), #808 prompt failures (parent137), #809 graph coverage (parent222), preserving gate order. Owners737/585/137/222 are blocked by their respective children. Original assignees/parents/bodies preserved.
- Five disposition recommendations prepared; no PR rewrite/closure. Core72+3 CLI probes and cache27 pass. Offline synchronization32 and ordering11 pass; failed runtime criteria under807/808 and invalid original graph fixture remain captured.
- Independent read-only review supplied two Important findings. Fix pass added owner blockers and required/documented edit window; remote API has no CAS. A controlled race-boundary test explicitly demonstrates the remaining API limitation. New-item reads now avoid unrelated private Project permissions.
- Live prepare passes repository source guards then Project403. #737/#806 publication verification and #738 human decision remain pending. Portable package published in draft PR #810: https://github.com/dhar174/tiny_village/pull/810. Human disposition/configuration review and authenticated publication remain pending.
- Technical ruling: preserve non-managed Status; an open issue with Done-like Status blocks prepare for review. No automatic status or phase-decision repair.
- Technical ruling: a failed/unacknowledged write requires inspect and a newly reviewed plan/checkpoint; do not infer ownership of ambiguous writes or silently retry.

- Portable archive extracts and validates outside Git on Linux. Deliberately modified preview/capture and missing hash manifest are rejected; the missing-manifest weakness was first reproduced, then corrected. Native Windows and real Project adapter behavior remain unverified.
- All145 retained runtime/runner/lock file hashes were compared against git show4821378; zero mismatches.

- Draft PR #810 published and attached to the task; implementation commit ad58fd0. No #737/#806/#738 closure or PR disposition. Hosted checks were queued at publication; local checks/evidence are the verified result.

## Closeout execution — 2026-10-23 UTC
- D032 records the approved dispositions/local route; plan saved at docs/superpowers/plans/2026-10-23-issue-737-closeout.md. New branch codex/issue-737-closeout starts from merged main10569c0.
- Reopened737 while806 remains open; closed701/704 unmerged after unchanged-head and145 runtime/runner/lock hash checks; annotated709/711/715 and807/808/809, preserving heads/assignees. Before/after receipts retained.
- Ruling: retain the original810 package untouched; self-contained refreshed artifacts live in docs/planning/issue-737-closeout. Historical probe results remain historical.
- Ruling: this package supports publication stage only; reject completion flags until authenticated evidence and a separate final-stage review exist. This prevents offline tests from appearing to authorize closure.
- Closeout regressions9 RED (missing interfaces) then9 GREEN; existing32 publisher and11 ordering tests pass. Windows path normalization is tested; native Windows remains unverified.
- Pending: refreshed capture/validation, independent review, focused closeout PR and owner-authenticated local prepare.806/737 cannot close before actual publication/readback;738 remains Pending.
- Fresh snapshot484/332/24 and preview validate;112 native edges,387 planned activation references,333 source guards, zero cycles/conflicts. Portable extraction validates and53 tests pass; three tamper cases reject. Actual prepare reaches GraphQL FORBIDDEN after source guards, exit1, zero board writes.
- Independent reviewer identified a fork/branch collision in candidate exclusions. Fixed with canonical headRepository plus exact branch; real source-collector regression RED1 then GREEN. Full54 tests pass; null/foreign identities stay inventoried. No other findings.
- PR #811 published: https://github.com/dhar174/tiny_village/pull/811. No automatic closure keywords. Refreshed local PowerShell handoff awaits authenticated live-plan/digest; actual board delta remains unreviewed until supplied.
- Final snapshot includes25 open PRs with only canonical-repository closeout811 excluded, leaving the approved24 candidates. All332 managed issues and333 guards validate; final exact ZIP validates outside Git. Fresh prepare again passes source checks then403. Initial #811 hosted core/dev/visual/required/CodeQL checks all succeeded at6bdf736; later source/evidence commit requires its own check result.
- Remaining work requires owner-supplied authenticated live-plan/digest. Apply only after concrete delta review;806/737 closure and final readiness refresh remain blocked. No native Windows or successful Project readback claim.

## Authenticated publication tooling — 2026-10-23 America/New_York
- D033 authorizes the approved four-step plan; isolated branch codex/issue-737-publication starts from merged5f2bf989. Historical issue-737 and issue-737-closeout packages remain unchanged.
- Tooling PR must merge before the fresh authenticated ZIP is frozen. No board delta or phase gate is approved by D033.806/737/738 remain open; owner local evidence is required.
- Preflight: receipt CLI produces state/digest evidence consumed by completed-stage validator; existing apply/checkpoints remain the only mutation path. Runtime receipt files stay outside the frozen package.
- Receipt/stage tooling: prior54 tests plus31 regressions pass85 total; locked core72+3 pass. Full332-row controlled-API lifecycle covers both stages and keeps738Pending. Actual prepare reachesFORBIDDEN after source checks, no live plan/writes. Fresh evidence retained under issue-737-publication/evidence/publication-tooling.
- Ruling: stronger optional receipt checks bind the original plan and verify unrelated-state preservation; legacy inspect output alone is not closure evidence. Receipts record observations and require human provenance review.
- Independent final review found wrong-board evidence acceptance and atomic temporary-file collisions. Four regressions RED then GREEN after canonicalProject10/reviewed node-ID and resolved destination/temp guards. Entire89-test suite passes; no other findings. Actual board readback/Windows remain unverified.
- Candidate archive validates outside Git;89 tests pass in extraction and three corruption cases reject. Tooling source/evidence ready for focused PR; the authenticated owner bundle will freeze only after tooling merge perD033. Prior folders unchanged.
- #812 merged under the explicitly approved review/merge workflow atdc9fcc023b4ee37b3efd579195caaab80993c64a, exact reviewed heada200880; all8 hosted checks pass and full Git trees match. No rule bypass or self-review request. Authenticated bundle now freezes after merge; only artifact commits follow, not another source merge.
- Source issue comments record progress without closing737/806 or approving738. Await owner-local read-only live plan/digest; current cloud access remainsFORBIDDEN, so publication, closure and final readiness are still pending.
- Frozen-source bundle validates/extracts outside Git and passes89 tests; every packaged Python file equals reviewed merged main dc9fcc0. Merged-main core/dev/visual/required/analyze checks succeed. Fresh prepare stillFORBIDDEN after source checks. No live plan/readback yet; proceed with owner-local read-only prepare, then concrete delta review.

## Authenticated initial readback and ordered closure — 2026-10-23 America/New_York
- Original owner-uploaded plan358987 bytes/SHA db5028bc... and receipt389472 bytes/SHA f01b40e9... independently verified with merged verify_receipt; exact bytes/hashes/command retained in docs/planning/issue-737-publication-readback at c7a070b. Project10 nodePVT_kwHOAE_6m84Bm2UC; initial desired4d77d37a..., approval10db8480..., observed2026-10-24T02:43:39.646747+00:00. All332 rows/12 fields/6 views match, zero plan operations/remaining differences/mutations and full preservation. No apply/checkpoint needed. Fresh coordinator333-issue/24-PR/main source comparison matches.
- Conditional D032/D033 closeout executed:806 CLOSED/completed02:50:29Z, then73702:50:33Z, after initial receipt. Original bodies preserved except checked acceptance plus evidence/qualification appendix; parents/assignees/labels unchanged. Future rebases remain activation-time work, not claimed performed.738 untouched OPEN/Pending;733/runtime gaps open. Mutation responses and independent GET readbacks retained.
- Original publication package remains byte-for-byte unchanged. New docs/planning/issue-737-completed bundle retains initial desired/plan/receipt, actual closure evidence and fresh484-issue/332-managed/24-PR snapshot. All parents/assignments/labels/native edges and unrelated bodies/states preserved. Prospective comparison against verified board changes only806/737 Readiness→Tracking and738→Human review, DecisionPending. Actual final live plan/readback still required.
- All final-bundle Python tools equal merged #812 source; original fixture-specific89 regression tests rerun successfully. Completed bundle omits those fixture-specific scripts and uses full completed-stage evidence validator. No new runtime suite or native board-write verification claimed.
- Final ZIP frozen at d940356ca12fe6b7c8d62989a2a4150a4b085ac8:816889 bytes, SHA25640b03674c3e6b32fc1e8e1c85276167d75c9d91ce693a175cd658382e6712e18. Evidence draft PR #813 published/attached: https://github.com/dhar174/tiny_village/pull/813. Fresh333-issue/24-PR/main guards match completed preview after PR creation; main stilldc9fcc0. Do not merge while owner prepares/applies final readiness. Next owner command: sync_project.py --prepare --plan ..\final-evidence\final-plan.json from newly extracted final ZIP; actual final plan/digest must be reviewed before apply. Final receipt and #738 walkthrough/decision remain pending.
- Owner pasted final plan: desired0a85047a..., exactly806Ready→Tracking/737Blocked→Tracking/738Blocked→Human review;738DecisionPending. Baseline now preserves automated StatusDone on737/806 and unrelated PR813 item. No original file/digest attached yet. Reconstructed exact full expected plan from original authenticated baseline+visible changes and regenerated/validated with unchanged merged tooling: expected approval8e7d27f6f9dbdd2b83be6d14ed146146122197b5ad625e5f70adaba250161acf. Fresh333issue/24PR/main guards match. Evidence docs/planning/issue-737-final-readiness-review explicitly labels reconstruction; never claim original parsed file. Exact apply digest fails closed before writes if local plan differs. Owner-local apply/independent receipt and actual original plan/checkpoint upload remain pending;738OPEN/Pending.
