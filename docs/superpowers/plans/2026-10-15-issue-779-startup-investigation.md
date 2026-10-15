# Issue #779 Startup Timing Investigation Plan

**Status:** Proposed — awaiting explicit plan approval. No investigation runs,
diagnostic helpers or runtime changes are authorized by preparing this document.

**Goal:** Explain the intermittent startup timeouts, distinguish individual
process failures from batch-budget exhaustion, and produce evidence for a
scoped remedy and renewed human review of #721.

**Approach:** Investigate sequentially in the existing checkout and prepared
Linux Python environment. Preserve the original failures, measure component
boundaries, and test one hypothesis at a time. Stop before production repairs
or changes to the verification contract; investigation approval alone does
not approve those changes.

**Tools:** Python 3.12, existing unittest cases, subprocess/monotonic/resource
timing, faulthandler, `-X importtime`, and available read-only Linux process
observations. No new dependencies or model downloads.

**Requirements:** [#779](https://github.com/dhar174/tiny_village/issues/779),
[the agreed decisions](../../decisions.md), and
[the merged verification report](../../testing/issue-721-startup-review.md).
Execution should use `superpowers:executing-plans` sequentially, without
delegation. All execution checkboxes below remain unchecked pending approval.

## 1. Scope and constraints

- Investigate merged runtime/test revision
  `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`. GitHub main still points there
  at plan preparation. The current branch adds review documentation only.
- Use `/workspace/.venvs/tiny_village/bin/python`, the existing Python 3.12.14
  CPU environment. Record actual versions at execution; do not silently
  substitute another interpreter or dependency profile.
- Preserve graph child limits of **20 seconds fatal / 25 seconds parent**,
  cache integration limits of **25/30 seconds**, and original verification
  batch limits of **110/120 seconds**. Timed-out or incomplete runs stay failed
  or incomplete. Successful diagnostics never replace their outcomes.
- Keep existing assertions, eight import-order subcases, mixed eight-thread
  singleton checks, ownership boundaries, cache checks and failure/retry paths.
- Per decisions D002/D004/D005/D006, exclude repairs to #773–#776, native
  Windows validation, real model downloads/inference, visual/gameplay-turn
  validation and the full repository suite.
- Do not install packages, alter dependency declarations, restructure runtime
  imports, change singleton/cache behavior, modify saved cloud settings, or
  change deadlines during this investigation. Such remedies require a scoped
  proposal supported by the resulting evidence.
- Do not delete shared caches, clear host page caches, chmod unrelated paths,
  dump credentials/environment contents, or disable verification settings.
- Keep #721 Deferred, #779 open, and #722 open. Project field access remains
  a separate outstanding 403; it does not block local diagnosis.

## 2. What the existing evidence establishes

| Observation | Established result | What remains unknown |
| --- | --- | --- |
| Graph-manager-first import | Fatal timer fires during pandas imports through `tiny_memories` | Why the import sometimes exceeds 20 seconds |
| Character-first import | Parent reaches 25 seconds without a child traceback | Whether delay is import, native code, scheduling, shutdown or output handling |
| Lifecycle batch | Seven methods report OK, then a recursion/retry subcase fails before the 110-second aggregate cutoff | Cause of the child failure; complete results for remaining methods |
| Ownership batch | Five methods report OK, then a FoodItem subcase fails before aggregate cutoff | Cause of the child failure; complete results for remaining methods |
| Cache suite | 26/27 pass; loader integration hits 25 seconds with no Python frame | Where the loader-test process stalls |
| Later import diagnostics | Both affected orders pass in 10.19s / 5.11s | Why outcomes vary; no root cause is established |

The five other complete suites passed 30 tests. Goal fallback, cold getter and
minimal mode passed; minimal mode executed zero actions. These are historical
results, not new checks performed while writing this plan.

Two harness details matter: `validate.py` imposes its aggregate fatal timer
on suites that spawn many fresh interpreters; `TestModelCacheIntegration.setUp`
sets a fresh per-test `MPLCONFIGDIR`, overriding the runner's shared plotting
cache. Neither detail alone proves a defect or justifies a deadline change.

## 3. Artifacts and file boundaries

| Artifact | Responsibility |
| --- | --- |
| Existing `/workspace/scratch/issue-721-post-merge-93079fd/` | Immutable historical logs, commands, diagnostics and corrected result interpretation |
| New `/workspace/scratch/issue-779-investigation/<run-id>/manifest.json` | SHA, source hashes, interpreter/dependency versions, permitted non-secret settings and resource observations |
| New `investigate.py` in that run directory | Temporary bounded diagnostic supervisor, written only after approval |
| New `events.jsonl`, `results.json`, `hypotheses.json`, per-attempt stdout/stderr | Process/phase timing, complete attempt history, hypotheses and classifications |
| New `docs/testing/issue-779-startup-investigation.md` after execution | Shareable findings, evidence, limitations and proposed remedy; not a second human decision log |
| Existing `docs/decisions.md` | Only maintained human decision record, including later approval or scope decisions |
| Memory Bank task/index | Resumability pointers and progress, not duplicated decision records |

Inspect `tests/test_graph_initialization.py`, `tests/test_graph_ownership.py`,
`tests/test_model_cache.py`, `tiny_globals.py`, `tiny_graph_manager.py`,
`tiny_memories.py`, `tiny_brain_io.py`, and `tiny_cache.py` as evidence requires.
No production or existing test-file edits belong to this investigation plan.

## 4. Ordered investigation

### Phase A — Freeze inputs and make evidence trustworthy

- [ ] Record checkout SHA, base SHA and runtime/test-file hashes. If main or
  those files have changed, report the difference before mixing baselines.
- [ ] Record Python/dependency versions using metadata rather than importing
  heavy libraries merely to discover their versions. Capture CPU quota, memory
  events, filesystem type, process counts and tool availability without secrets.
- [ ] Recover exact original command/environment setup from saved `validate.py`
  and `final-commands.json`. Retain offline flags, dummy SDL, two BLAS/OpenMP
  threads and bytecode policy. Record child-specific overrides, especially
  the integration test's plotting and cache directories.
- [ ] Create a new owned evidence directory; preserve old logs and shared
  caches. Distinguish a fresh process, a fresh application cache and an OS-cold
  filesystem; do not describe them interchangeably.
- [ ] Define structured outcomes: `PASS`, `ASSERTION_FAILURE`,
  `CHILD_FATAL_TIMEOUT`, `PARENT_TIMEOUT`, `BATCH_INCOMPLETE`, `NOT_RUN`, and
  `INFRASTRUCTURE_INTERRUPTION`. Unknown final counts are null, never zero.

**Deliverable:** A reproducible manifest and command map. No cause inferred yet.

### Phase B — Reproduce once with the original contract

- [ ] Run the four original affected targets once, sequentially, under the
  unchanged original supervisor and limits:

  ```bash
  /workspace/.venvs/tiny_village/bin/python -m unittest -v tests.test_graph_initialization.TestGraphImports
  /workspace/.venvs/tiny_village/bin/python -m unittest -v tests.test_graph_initialization.TestGraphLifecycle
  /workspace/.venvs/tiny_village/bin/python -m unittest discover -s tests -p test_graph_ownership.py -v
  /workspace/.venvs/tiny_village/bin/python -m unittest -v tests.test_model_cache
  ```

  The supervisor must retain the saved 110/120-second batch timers; bare shell
  commands alone do not reproduce that batch contract.
- [ ] Retain separate full stdout/stderr, parent wall/CPU timing and exit status
  for every batch. Count test methods and subcases separately. A cutoff must
  list interrupted and unrun cases rather than fabricate a completed summary.
- [ ] Check for surviving processes started by these runs. Clean up only
  identified investigation descendants before the next experiment; never use
  broad process-name kills. If descendant cleanup cannot be assured, stop.

**Deliverable:** Fresh baseline, including failures if present. Do not loop
until it passes or replace the prior evidence if the failure does not recur.

### Phase C — Locate the boundary that consumes time

- [ ] Add temporary diagnostic instrumentation outside the repository. Emit
  monotonic and UTC markers for process start, before/after imports, before/
  after the tested operation, assertions complete, entry to atexit, and parent
  observation of process exit. Atexit entry alone does not prove clean shutdown.
- [ ] Observe PID/parent PID, elapsed and CPU time, thread count and available
  wait-state data for investigation-owned processes before existing deadlines.
  Write diagnostics to files rather than relying on truncated unittest output.
  Keep instrumented runs distinct: measurement can alter timing.
- [ ] Cover these five previously affected scenarios using original source and
  assertions: graph-manager-first, Character-first,
  `TestGraphLifecycle.test_recursion_is_prompt_and_retryable`,
  `TestGraphOwnership.test_items_with_configured_effects_preserve_the_supplied_action_system`,
  and `TestModelCacheIntegration.test_real_model_loaders_receive_selected_cache_and_preserve_verification`.
  Preserve both constructor-entry and all item-constructor subcases.
- [ ] Preserve child fatal/parent limits. Invoke existing single methods when
  useful; import-order diagnostics must retain the no-world-construction
  assertions. Do not convert a smaller diagnostic into a full-suite pass.
- [ ] If supported, obtain a Python stack dump before the fatal timer through
  a diagnostic-only signal registration. If a native-only stall remains, use
  already-available permitted process/syscall observations; do not install or
  escalate profiling tools. Missing access is an explicit evidence limit.
- [ ] Ensure diagnostic supervisor cleanup terminates only its own process
  group after capture, reaps it, and retains partial output. Never extend the
  original time budget merely to obtain a passing exit.

**Deliverable:** Classify delay as import, tested operation, shutdown/pipe
handling, resource/infrastructure interruption, or still unlocalized.

### Phase D — Test evidence-led hypotheses

Choose the supported branch first; do not execute every experiment by default.
For each experiment, record prediction, single changed variable, comparison,
result and what would falsify the hypothesis.

| Hypothesis | Controlled comparison | Evidence needed before recommending a remedy |
| --- | --- | --- |
| Heavy imports or cache initialization dominate | `-X importtime` plus dependency-only controls for the implicated library; if plotting is implicated, compare fresh versus reused investigation-owned plotting cache while holding HF settings fixed | Repeated concentration of time at the same boundary and a matching one-variable timing effect; not simply a slow library name in one stack |
| Graph lifecycle blocks after imports | Separate import-complete and constructor/recursion/retry markers, preserving original graph assertions | Stable wait/lock or operation evidence after import; do not infer a graph deadlock from a parent waiting in `communicate` |
| Child budget is adequate but aggregate budget expires | Compare captured per-method/child durations and suite overhead with the 110-second cutoff; use individual-method diagnostics for methods the batch never reached | Explicit budget arithmetic and separate identification of any genuinely failing child; splitting batches does not validate the original batch contract |
| Native shutdown, threads or inherited pipes delay exit | Compare assertion/atexit markers with parent-observed exit and thread/process state; use file versus pipe output only as a separately labeled experiment if indicated | Delay localized after the tested work, or a documented descriptor/descendant keeping capture open |
| Environment or storage scheduling contributes | Compare wall/CPU timing and before/after cgroup/process observations; use a lightweight fresh-interpreter control alongside serial, not competing, runs | Correlated resource/wait evidence; aggregate historical throttling or no OOM events alone is insufficient |

- [ ] Limit targeted diagnostic runs to **16 launched top-level cases**, each
  retaining its native child limits and at most the original 120-second batch
  parent limit. Subprocesses/subcases count separately in recorded statistics.
- [ ] Predeclare at most **three trials per chosen variant** before running it;
  retain all trials and stop early only for safety or a confirmed hypothesis.
  Stop at the cap if evidence remains insufficient; do not retry until green.
- [ ] Treat multiple independent causes separately. One explained batch cutoff
  does not resolve the cache-loader child timeout or another unexplained stall.
- [ ] If no failure recurs, report the fixed observation window and remaining
  uncertainty. Do not call the cause identified or #779 resolved.

**Deliverable:** Supported causal explanation, or a bounded inconclusive
report naming the missing evidence and next useful observation.

### Phase E — Present the finding before any repair

- [ ] Prepare a finding with exact evidence, affected files/environment,
  alternatives ruled out, confidence limits and the smallest proposed remedy.
- [ ] Distinguish an environment/setup correction, a runtime repair, a test
  harness repair and a verification-budget contract change. Propose only what
  the evidence supports; do not combine unrelated repairs.
- [ ] Stop for human review before applying a remedy. In particular, import
  restructuring, lifecycle/cache changes, package changes, persistent cache
  configuration and changed deadlines are outside investigation approval.
  Record any subsequent human decision only in `docs/decisions.md`.

**Deliverable:** A concrete repair proposal or a documented reason no repair
is justified. Investigation completion is not issue resolution.

### Phase F — Verification handoff and report

- [ ] If investigation alone establishes a supported environment explanation
  without changing the contract, run one final original focused matrix. If a
  repair is required, leave this verification pending until that repair has
  separate approval and implementation; do not waste a full rerun on an
  unchanged known failure merely for a green summary.
- [ ] Required final targets total **81 existing test methods**: imports 1
  (eight subcases), lifecycle 12, ownership 11, core-global 9, named-target 6,
  actions 5, event payloads 5, GOAP errors 5 and cache 27. Use the commands in
  the saved matrix, original limits and unmodified assertions. Also exercise
  the cold getter, production Goal fallback and minimal mode probes.
- [ ] Report executed/passed/failed/errored/skipped/unrun results and child/
  batch timeouts separately, with the exact SHA. Confirm mixed-thread graph
  access still executes. Preserve the zero-action minimal-mode limitation.
- [ ] Write `docs/testing/issue-779-startup-investigation.md` with the manifest,
  experiment table, retained failures, cause or uncertainty, proposed remedy
  and verification still needed. Link raw local evidence as local evidence;
  retain substantive findings in a shareable report.
- [ ] After execution, add the findings to #779 and link them on #721 without
  closing either issue or changing the human disposition. Keep #722 blocked.
  A denied Project API update remains an explicit outstanding action.

## 5. Completion, stopping points and expected effort

The investigation is complete when inputs and failed runs are preserved,
affected boundaries have been examined, and the report either supports a
specific remedy or explains why the evidence remains inconclusive. #779 is
not complete until its broader correction/verification acceptance criteria
are actually satisfied; #721/#722 still require their human decisions.

Plan for **roughly 1–3 hours of investigation after approval**, depending on
reproducibility and available process observations. This excludes production
repair and the subsequent human review. There is one initial affected-matrix
run, a capped diagnostic phase, and at most one final focused verification
matrix under this plan. Stop earlier for missing required capability,
unreaped processes, changed inputs or exhausted diagnostic budget; report the
specific blocker rather than widen scope or relax checks.

## Approval requested

Approve this **investigation-only** scope and bounded experiment sequence.
Approval starts the diagnostic work; it does not authorize production fixes,
deadline changes, release claims, issue closure or phase advancement. Until
approval arrives, no execution steps above will begin.
