# #736 post-merge verification and scoped closeout

PR #804 merged at `6cc576edcb465c154ba7f56a508116ec1b247975` on
2026-10-22. The agreed bounded post-merge checks pass on that exact clean source:
**72 tests + three CLI probes in core, 20 tests in dev**, with no test failures,
errors, skips or timeouts. All 15 previously validated test/runner/runtime source
hashes match. The intervening changes are agent-team configuration and documents;
no additional runtime repair is inferred.

This supplies the verification prerequisite for the human-approved
[D030 scoped reconciliation](../decisions.md#d030--accept-736-reconciliation-and-retain-runtime-requirements).
It does not make the known failed runtime requirements pass or approve #738.

## Commands and retained evidence

Run once from the repository on merged commit `6cc576e`:

```sh
TINY_TEST_ARTIFACT_DIR=/workspace/scratch/issue-736/post-merge-6cc576e/core-captures /workspace/.venvs/issue726-core/bin/python scripts/run_ci_checks.py --profile core --artifacts /workspace/scratch/issue-736/post-merge-6cc576e/core
TINY_TEST_ARTIFACT_DIR=/workspace/scratch/issue-736/post-merge-6cc576e/dev-captures /workspace/.venvs/issue726-dev/bin/python scripts/run_ci_checks.py --profile dev --artifacts /workspace/scratch/issue-736/post-merge-6cc576e/dev
```

Both processes exit zero. Original cold-child diagnostic/parent deadlines remain
20/25 seconds; existing batch deadlines remain unchanged. The first dev command
failed during shell redirection before Python launched because directory
creation had been incorrectly parallelized. No tests started; the setup failure
is retained as `dev-not-started.json`, and the previously unrun profile was then
started once. It is not a passing test or an erased product failure.

[Structured summaries, candidate preflight and raw archive](../planning/issue-736/post-merge)
retain counts, commands, environments, revisions, child sources, stdout/stderr
and terminal metadata. `raw-member-hashes.json` checks every member of
`raw-captures.tar.gz`. Earlier failing runtime reproductions and existing utility
failures remain in [the original consolidation report](issue-736-functional-test-consolidation.md);
they were not rerun or reclassified by this bounded check. Other optional model,
platform and gameplay acceptance is outside this verification.

## Authorized dispositions

All nine candidate heads still match their reviewed pinned SHAs. After this
verification, apply the selected superseding closures for #420/#417/#408/#406,
#404/#397/#395/#391/#393 with links to merged replacement #804 and the exact
assertion map. Complete #736 for its accepted reconciliation scope under D030;
original runtime requirements remain outstanding in their owner issues.

- #802 under #218: full Goal completion/planning/evaluation/feasibility.
- #803 under #139: missing basic motive-satisfaction contribution.
- #799: graph identity, state/action provenance and synchronized world updates.
- #741: replanning result/failure contract and actual alternative behavior.
- #774: existing utility Goal/fixture and explicit falsy-ID test debt.

The issue bodies, native parent/gate relationships, evidence and unchecked
runtime requirements remain intact. #737 is the next task: reconcile PR
ownership/reviewers, merge sequence and Project configuration. #738 remains a
separate human decision gate. Closure readbacks are retained alongside the
post-merge evidence after publication; no phase approval or runtime fix is
supplied by scoped #736 completion.

## Verified closure outcome

#420/#417/#408/#406/#404/#397/#395/#391/#393 are CLOSED as superseded, with
unchanged reviewed heads and no merge. #736 is CLOSED/completed for the accepted
D030 scope; its original requirement text and runtime carry-forward remain.
[Candidate closure receipts](../planning/issue-736/post-merge/candidate-closure-receipts.json),
[scoped completion](../planning/issue-736/post-merge/issue-736-completion.json)
and [final states](../planning/issue-736/post-merge/final-states.json) retain
readback evidence. #737 is OPEN with zero open native prerequisite blockers;
#733/#738 and all five runtime/test-debt owner issues remain OPEN.

The functional task file is now TASK024, matching the merged task-index link;
the separate Zcode-team TASK023 is preserved. Duplicate in-progress index rows
for the verified completed #734/#735 predecessors were removed, preserving their
completed records. No runtime, test, dependency or team configuration changed.
