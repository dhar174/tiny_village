# #737 validation report

## Result

The coordination/review package is prepared. **Project synchronization and #737
completion are not verified.** Both credential routes receive FORBIDDEN for
Project10; #806 blocks #737. All five candidate PR dispositions are pending human
review, and #738 remains open/Pending. No runtime or dependency changes were made.

## Source and scope

- Runtime baseline: merged main `48213783e2ee0864594af4a3f26667e1269d394a` (#805).
- Working branch: `codex/issue-737-coordination`; input capture start/end timestamps
  and exact source/PR heads are retained in `snapshot.json.gz`.
- Full source inventory:484 issues; managed scope332 issues = original roadmap /
  reviewed triage union323 + #799–#803 + #806–#809. All26 remaining candidate PRs
  are inventoried for ownership/order; five direct handoffs received substantive
  investigation. This does not claim26 substantive code reviews.
- Runtime/root-module, runner and lock hashes are retained in
  `evidence/runtime-lock-source-hashes.json`; all145 files match the unchanged
  baseline. The existing core lock hash remains
  `2a037e6923b1f4e0f19d14109adc4217a5e4b7e0583a780639dd476a1451c908`.
-12 existing custom fields, six saved views,112 managed native prerequisite
  edges and387 separately labelled planned-activation references. Zero detected
  cycles or phase/sequence conflicts across the complete retained graph.

## Executed checks

| Command / environment | Actual result |
|---|---|
| Locked core CPython3.12.14, Linux x86-64: `python -B scripts/run_ci_checks.py --profile core --artifacts /workspace/scratch/issue-737/core-baseline` |72 tests +3 CLI probes pass;0 failures/errors/skips/timeouts. Full runner summary and203 raw files retained. |
| `python -B docs/planning/issue-737/test_sync_project.py` |32 offline service-boundary checks pass. Covers typed fields, duplicates, foreign items, stale source/heads/schema, human edits, interrupted/ambiguous writes, restart/idempotence, new-row defaults, explicit clearing, local locks and Status/gate conflicts. |
| `python -B docs/planning/issue-737/test_build_preview.py` |11 offline checks pass. Covers missing inputs, cycles, parent/dependency separation, ordering, closed-but-unapproved gates and the explicit #738 boundary. |
| Bounded `tests.test_model_cache` in the locked core interpreter |27 tests pass,0 failures/errors/skips. Separate retained probe; no model downloads/inference. |
| `build_preview.py` and `validate_package.py` | Regeneration matches retained data, all12 fields validate, stable keys are unique, and #737 remains blocked / #738 Pending. Artifact and raw-member hashes are verified. A clean extracted package validates outside Git; modified preview/raw evidence and a missing hash manifest are rejected. |
| Actual `sync_project.py --prepare` with configured Project credential | Reviewed repository source checks pass; Project read returns FORBIDDEN. Nonzero exit, no live plan and no Project mutation. |

Each new cold probe uses the existing durable capture helper with a20-second
diagnostic deadline and25-second parent deadline. Seven captures: four exit0,
two exit1 on unresolved runtime criteria, one exit1 on a documented invalid
fixture assumption. None timed out. Do not combine these into a green runtime
suite: the helper capability and prompt visibility requirements remain failed.

## Retained failures and corrections

- #807: real helper configuration propagates the controlled original OSError but
  leaves `use_llm=True`, `llm_status="disabled"`, provider absent. Requested state
  and actual availability are inconsistent. Real Character selection passes.
- #808: two controlled goal/needs failures return empty context with zero prompt
  diagnostics. Original cause/visibility/caller semantics remain unresolved.
- #809: the initial Character-distance probe incorrectly assumed an unconnected
  controller graph. Its failure is retained as a fixture error. A separate
  isolated real-Location probe passes the documented query checks; NetworkX-free
  parity remains unverified. No retry erases the original capture.
- #787 retains pre-existing job-catalog diagnostics emitted by real startup.
- Original implementation RED captures include an absent module, empty-text /
  checkpoint/default/schema regressions and review-fix failures. Current GREEN
  logs retain actual counts rather than replacing those failures.

## Independent review and limits

The independent read-only reviewer examined the supplied implementation patches
and command evidence; it ran no commands and made no edits. Two Important
findings were addressed in the coordinator's fix pass: owner blockers were
added/read back, and the no-CAS edit window was made explicit and required by
the CLI. Fresh graph validation is acyclic; CLI/status tests were observed RED
then GREEN. A controlled test demonstrates that the remote same-field race
cannot be detected, rather than claiming it is protected.

Additional coordinator checks removed broad Issue.projectItems lookup, which
could require unrelated private-Project permissions. New-item lookup stays in
Project10 and subsequently uses the acknowledged item ID; its boundary test was
observed RED then GREEN. Subprocess output uses explicit UTF-8 for portability.

Unverified: authenticated Project before-values, real mutation acknowledgement /
readback, saved-view update behavior, native Windows execution, renderer/gameplay
and live AI inference. Three read-query shapes pass GitHub schema validation on
incompatible public nodes; that is not a live Project adapter test. A fourth
diagnostic lookup was denied and is no longer used by the updater. The updater
requires a quiet edit window because GitHub's field API offers no conditional
version writes; the local lock alone cannot protect concurrent remote edits.

No original issue acceptance is checked off, no #737/#738 closure is performed,
and no PR disposition is applied by offline success. After the configuration and
disposition review, complete authenticated publication/readback, merge the
focused PR, and verify #737's criteria before closing it. Then conduct #738's
separate human walkthrough.
