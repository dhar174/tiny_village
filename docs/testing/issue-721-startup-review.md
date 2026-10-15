# Issue #721: merged startup engineering review

Review prepared on 2026-10-15 for dhar174. **Human disposition: pending.**
This is verification evidence, not approval of #721 or phase gate #722.

## Revision and prerequisites

- Merged main: `93079fd6ef7ef7607153b95e2b66a48a957a9ee6` (PR #778).
- Exact source tree matches reviewed PR head `f796b33a8e140f6bebc2b0bad14c6b300eb26ee8`.
- #718, #719 and #720 are closed; #721 and #722 remain open.
- Both merged-main CodeQL language checks succeed. Static analysis does not
  establish startup, model or gameplay readiness.

## What the repairs cover

1. #718 / #706: preserve the production Goal fallback when the character import
   is unavailable, avoiding the undefined-Goal import failure.
2. #719 / #777: remove sample-world construction during imports, serialize
   allocation and initialization, reject recursive initialization promptly,
   roll back failed attempts, and preserve injected graph/action ownership.
   Getter/class paths share one ready graph. Reset/replacement is documented
   for quiescent boundaries; this is not a guarantee of concurrent hot swapping.
3. #720 / #778: configure model caches before optional model imports; preserve
   explicit HF settings; validate paths and effective-user ownership; make
   selected POSIX directories private without deleting data or changing parents.
   This supersedes the narrow intent of draft #701 without executing its
   destructive reproduction script.

## Fresh merged-main verification

Prepared Python 3.12.14 CPU environment, Linux, no new dependencies or models.
Existing assertions and deadlines are unchanged. Sequential batches use
110-second fatal / 120-second parent limits; graph children use 20/25 seconds,
cache integration children 25/30 seconds. Owned isolated HF/TMP paths, offline
model flags, dummy SDL and two BLAS/OpenMP threads match the previous matrix.

| Existing command target | Current result |
| --- | --- |
| `tests.test_graph_initialization.TestGraphImports` | One test method fails: six of eight import orders pass; two time out |
| `tests.test_graph_initialization.TestGraphLifecycle` | Incomplete at 110-second aggregate timer; seven methods report OK, recursion/retry subcase fails before interruption |
| discovery `test_graph_ownership.py` | Incomplete at 110-second aggregate timer; five methods report OK, FoodItem subcase fails before interruption |
| `test_core_global_graph_manager` | 9/9 pass, zero skips |
| discovery `test_action_named_target_resolution.py` | 6/6 pass, zero skips |
| discovery `test_actions.py` | 5/5 pass, zero skips |
| discovery `test_event_handler_payloads.py` | 5/5 pass, zero skips |
| discovery `test_goap_error_handling.py` | 5/5 pass, zero skips |
| `tests.test_model_cache` | 27 executed: 26 pass, one loader-integration timeout, zero errors/skips |
| `main.py --mode minimal --no-llm --headless` | Exit 0 in 10.15s; **zero actions executed** |
| Production Goal fallback with character import deliberately unavailable | Pass, exit 0 |
| Cold global getter, repeated identity and `G is world_state.graph` | Pass, exit 0 in 9.73s |

Unittest targets use `/workspace/.venvs/tiny_village/bin/python -m unittest -v`;
discovery targets use `-m unittest discover -s tests -p <filename> -v`.
The local `validate.py` wraps these exact targets in the fatal and parent timers.
Incomplete suites do not have trustworthy final executed/failure counts; the
runner's zero-valued summary counters for them must not be interpreted as green.
The earlier 81 pre-merge checks are historical and do not replace this run.

## Timing evidence and diagnosis limits

- Graph-manager-first import reaches its 20-second fatal timer while importing
  pandas through `tiny_memories`; this trace does not show the original graph
  initialization lock cycle.
- Character-first import reaches the 25-second parent deadline without a
  traceback. Its captured output includes optional embedding initialization
  failing because Transformers is unavailable.
- Lifecycle and ownership suites stop at their aggregate timers after partial
  success and subcase failures. They are **not** complete passing suites.
- Cache loader integration reaches its 25-second fatal timer with no Python
  frame. No cache-path or ownership assertion failure is established by that
  timeout; the integration behavior still lacks a passing result in this run.
- Separate `-X importtime` diagnostics of the two affected import orders, with
  unchanged 20/25-second limits and before/after/assertion/atexit markers, both
  exit 0 in 10.19s and 5.11s. Assertions establish no graph construction on those
  diagnostic imports. Cumulative memory imports take about 8.42s / 3.34s, with
  sklearn and matplotlib contributing materially. This is timing evidence,
  not an established root cause or a replacement full-suite pass.
- No OOM events are reported. A transient execution transport disconnect also
  occurred. Neither fact establishes why individual processes stalled.

No timeouts were increased, tests weakened, runtime code changed or failed
runs discarded. **Recommendation: defer human approval pending timing diagnosis.**
Distinguish an environment/dependency import delay from a runtime regression
before choosing a repair. Passing one diagnostic is insufficient.

Investigation is tracked in [#779](https://github.com/dhar174/tiny_village/issues/779).
Its gate/dependency treatment awaits the human walkthrough; creating that issue
does not record a disposition or approve implementation.

## Existing limits to decide explicitly

- #773 integration fixtures and #774 Goal contracts remain open; their earlier
  9/11 and 16/26 outcomes were not rerun in this verification.
- #775 missing Open Door effects remains open. Ownership fixtures supplying
  effects do not establish default Door construction/action behavior.
- #776 unavailable-spaCy memory remains open. Character ownership checks stop
  at the documented prompt boundary and do not verify full construction.
- Native Windows ACL/execution, real model downloads/inference, visual gameplay
  and the full repository suite remain unverified.
- Imports still attempt optional embedding initialization; absence of graph
  construction is not a claim that imports have no side effects at all.

## Human walkthrough

1. Decide whether current failed/incomplete verification requires deferring
   #721. Recommended: **Deferred** until the startup timing is explained and
   required checks complete with retained evidence.
2. Review repair scope and decide which wider limits remain scheduled followups
   versus prerequisites. Any required remediation needs its own linked issue.
3. Record reviewer identity, date, exact revision/evidence, rationale, followups
   and exactly one disposition: Approved / Changes requested / Deferred.
   Only explicit human Approved permits closing #721.
4. Consider #722 separately afterward; never infer its approval from #721,
   merged PRs, passing tests, silence or elapsed time.

## Evidence retained

Raw commands/results/logs, corrected incomplete-suite interpretations,
instrumented import logs, revision and GitHub snapshots are in
`/workspace/scratch/issue-721-post-merge-93079fd/`. The review comment on #721
retains the substantive results remotely; the scratch paths are local evidence.
Earlier repair reports: [#718](issue-718-goal-fallback-validation.md),
[#719](issue-719-graph-initialization-validation.md),
[#720](issue-720-owned-cache-validation.md).
