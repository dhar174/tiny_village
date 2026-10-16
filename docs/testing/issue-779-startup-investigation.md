# Issue #779: bounded startup timing investigation

**Result: inconclusive for the historical timeout cause.** The four affected
suites pass unchanged in this investigation, and none of the 16 predeclared
diagnostic cases reproduces the historical failure. We measured import and
plotting-cache costs, but that is not a verified explanation of the earlier
20/25-second stalls. No production repair or deadline change was made.

Investigation authorized on 2026-10-15 under D007 in the single
[decision record](../decisions.md), following the
[approved plan](../superpowers/plans/2026-10-15-issue-779-startup-investigation.md).
#779 remains open and blocks #721; #721 remains Deferred and #722 remains open.

Findings are retained on [#779](https://github.com/dhar174/tiny_village/issues/779#issuecomment-6199183299)
and linked from [#721](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6199184274).
Both comment bodies, open issue states and the native #779 blocker were read
back and verified after publication. No human disposition was changed.

## Inputs and reproducibility

- Runtime/test baseline: `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`.
- Investigation starts from documentation commit `8ceb7cc` on
  `codex/issue-721-review`. GitHub main still points to the runtime baseline
  at execution start. All 388 captured Python/manifest source hashes match
  again after diagnostics; no runtime, test or dependency source changed.
- Interpreter: `/workspace/.venvs/tiny_village/bin/python`, Python 3.12.14 CPU,
  Linux, overlay filesystem. No installs, models or new dependency profile.
- Recorded versions include NumPy 1.26.4, SciPy 1.12.0, pandas 3.0.6,
  scikit-learn 1.3.2, Matplotlib 3.11.2, NetworkX 3.1, pygame 2.6.1 and
  faiss-cpu 1.15.1. Torch, Transformers, spaCy, NLTK and huggingface-hub are
  unavailable in this prepared environment.
- Original runner settings are preserved: dummy SDL, two BLAS/OpenMP threads,
  no bytecode writes, offline model flags and owned HF/TMP paths. Cache
  integration fixtures still create distinct HF/Hub/Transformers paths and
  their own fresh plotting directory. No shared cache was deleted or chmodded.
- Graph child timers remain 20 seconds fatal / 25 seconds parent; cache
  integration timers remain 25/30 seconds; original batch timers remain
  110/120 seconds. Original assertions and subcases are unchanged.

Original failure evidence remains untouched in
`/workspace/scratch/issue-721-post-merge-93079fd/` and the
[earlier review report](issue-721-startup-review.md). New evidence lives in
`/workspace/scratch/issue-779-investigation/2026-10-15-93079fd/`.
These are local paths, not remotely accessible artifact URLs.

## Original affected baseline: 51/51 methods pass

The saved supervisor's command construction, environment, fatal timer and
parent timeout were reused. Each batch runs in its own process group; an outer
observer captures resources, exit status and descendant cleanup. It does not
change the existing test process deadlines. Historical output format is retained.

| Existing target | Executed | Passed | Failures/errors/skips | Inner batch seconds |
| --- | ---: | ---: | --- | ---: |
| `tests.test_graph_initialization.TestGraphImports` | 1 | 1 | 0/0/0 | 35.58 |
| `tests.test_graph_initialization.TestGraphLifecycle` | 12 | 12 | 0/0/0 | 59.06 |
| discovery `test_graph_ownership.py` | 11 | 11 | 0/0/0 | 54.73 |
| `tests.test_model_cache` | 27 | 27 | 0/0/0 | 20.62 |

The import method retains all eight import-order subcases. Lifecycle retains
mixed eight-thread access for ten rounds, direct/provider identity, recursion,
failure/retry, reset/replacement and import-cause checks. Ownership retains
the documented Character prompt and Door definition boundaries. Cache retains
all 27 tests and controlled model-loading boundaries. No methods were skipped,
disabled or removed. All investigation descendants were checked and reaped.

Example invocation, from repository root:

```bash
/workspace/.venvs/tiny_village/bin/python /workspace/scratch/issue-779-investigation/2026-10-15-93079fd/investigate.py baseline
```

`baseline-copy.py` retains the historical runner; `baseline-case.py` selects
one of the four targets without running the other matrix targets. Exact
commands and per-batch summaries are retained beside it.

## Instrumented scenarios and fixed diagnostic budget

Exactly **16 top-level diagnostic cases** were predeclared and executed. All
exit 0; no timeout or assertion failure recurs. These are experiments, not
16 additional unique regression tests. There were no retries until green.

| Diagnostic group | Cases | Purpose |
| --- | ---: | --- |
| Graph-manager-first and Character-first imports | 2 | Preserve no-world-construction assertions and locate import time |
| Original recursion/retry method | 1 | Preserve both constructor-entry subcases and distinguish post-import work |
| Original item ownership method | 1 | Preserve ItemObject, FoodItem and configured Door subcases |
| Original cache loader method | 1 | Observe four controlled loader calls and settings assertions |
| Fresh/reused plotting-cache loader variants | 6 | Three predeclared trials per variant; change only plotting cache selection |
| One syscall-traced loader case | 1 | Observe native/process exit and request a predeadline Python stack |
| pandas, sklearn and pyplot imports in isolation | 3 | Measure library-only import cost without constructing a graph |
| Lightweight interpreter control | 1 | Check basic process startup/exit |

Diagnostic code lives outside the repository. It adds statement-boundary,
assertion-completion and atexit markers to original child source in memory,
plus `-X importtime`, full child stdout/stderr and parent-observed exit timing.
It retains the source statements and assertions. Instrumentation can affect
timing, so its passes do not replace either baseline or historical failures.
Only the plotting comparison overrides a fixture setting, and those results
are explicitly classified as controlled experiments.

The five initial scenarios execute eight instrumented test children. All reach
`original_source_complete`, enter atexit and exit 0. Their measured source work
takes 2.88–4.94 seconds; the additional interval from atexit entry to parent
observation is 0.43–0.74 seconds. No original scenario hangs during import,
graph work, loader calls or shutdown in these observations. Atexit entry alone
was not treated as proof of exit; the parent result supplies that evidence.

## Supported finding: fresh plotting-cache work adds import overhead

Cache fixtures override `MPLCONFIGDIR`, so the runner's shared plotting cache
does not remove this cost. The controlled variants preserve fresh per-test
HF caches and the real test's assertions; the reused variant alone redirects
the plotting cache to an investigation-owned directory.

| Trial | Fresh plotting cache, child seconds | Reused directory, child seconds |
| --- | ---: | ---: |
| 1 | 7.261 | 5.337 |
| 2 | 6.176 | 4.225 |
| 3 | 5.728 | 4.212 |
| Median | 6.176 | 4.225 |

The reused directory starts empty in trial 1; only trials 2–3 reuse a populated
font cache. The median difference is about **1.95 seconds**. Matplotlib's
`font_manager` import accounts for about 2.07–2.30 seconds in fresh trials,
versus 0.036–0.055 seconds in populated-cache trials. Other library timings also
vary, so the total median difference is a small-sample measurement, not a
universal performance guarantee.

This explains a measurable part of normal startup cost. It does **not** explain
the historical 20/25-second child timeouts, and it does not justify changing
fixture isolation, making imports lazy, or increasing deadlines.

## Native/process and resource observations

- Isolated import work took about 0.723 seconds for pandas, 1.241 for sklearn
  and 2.706 for pyplot in their recorded controls. Controls do not establish
  whole-application or OS-cold behavior.
- The syscall-traced loader child exits 0 in 16.896 seconds, versus roughly
  5–7 seconds for untraced fresh-cache variants. Tracing substantially perturbs
  this workload. Its source work takes 16.100 seconds, with 15.866 seconds in
  `tiny_memories` import; post-atexit exit takes 0.598 seconds.
- At approximately 15 seconds, the permitted diagnostic signal captures the
  main thread importing Matplotlib classes, including garbage collection in
  `matplotlib.artist`. This is a requested diagnostic dump, **not** a timeout
  failure. The original 25/30-second loader limits remain unchanged and pass.
- Long futex waits in background/watchdog threads are not evidence that the
  main thread hung. The longest recorded single syscall in the traced main
  loader thread is a 0.118-second poll. This trace shows a successful run,
  not the cause of the historical native-only/no-frame failure.
- CPU/process/wait-state samples and cgroup observations are retained.
  Memory events report no OOM or OOM kills. No matching resource failure occurs
  during this cycle. That cannot reconstruct earlier scheduling/storage state
  or rule it out as a contributor to the historical failures.
- All process groups and descendants created by the investigation are checked
  and reaped. No profiling packages or privileged capabilities were added.

## Hypothesis disposition

| Hypothesis | What this cycle supports | Remaining uncertainty |
| --- | --- | --- |
| Import/cache preparation dominates normal startup | Loader phase markers and import timings localize most normal work to imports; plotting-cache cost is measured | No reproduction showing that cost causes a 20/25-second failure |
| Graph initialization regressed into a lock cycle | Original lifecycle/concurrency and instrumented recursion/ownership cases pass | A passing observation window cannot exclude a rarer defect |
| Aggregate deadline alone explains all failures | Current complete lifecycle/ownership batches fit below 110 seconds | Historical aggregate cutoffs also followed failing child subcases; arithmetic alone cannot explain those child failures |
| Shutdown/native wait caused the no-frame timeout | Successful cases exit shortly after atexit; traced case exits normally | No failing native-only execution was captured |
| Resource/environment interruption caused stalls | No OOM or comparable interruption recurs | Earlier environment state is unavailable; no causal attribution justified |

The baseline consumes 35.58 / 59.06 / 54.73 / 20.62 seconds in its respective
batches. It establishes completion under current inputs, not a newly approved
budget contract or proof that historical timing failures are harmless.

## Remedy assessment and handoff

**No runtime or verification-contract repair is justified by the causal
evidence collected so far.** Reusing the plot cache or raising timeouts would
change behavior/coverage without explaining the original stalls. No such
change was applied. #779 is not resolved by passing this cycle.

The next useful evidence is a failing execution captured with child phase
markers, complete stdout/stderr, resource observations and cleanup records.
The temporary supervisor and predeadline stack mechanism are prepared for that
purpose. A further diagnostic cycle needs an explicitly reviewed observation
window beyond this exhausted 16-case budget, preferably tied to a meaningful
environment change or an actual recurrence, rather than more identical retries.
If durable test-harness observability is proposed instead, its scope and
overhead must be reviewed separately before editing existing tests.

The plan's conditional final **81-method** matrix was not run: neither an
environmental root cause nor an approved remedy was established. The 51-method
affected baseline above is fresh; the other 30 tests and cold-getter/Goal/
minimal probes were not rerun in this cycle. Their earlier results are historical.
No complete new 81/81, full simulation, native Windows or live-model readiness
claim is made. The zero-action minimal-mode limitation remains unchanged.

Investigation report complete; cause unconfirmed, further verification and
any remedy pending. #779 stays open; decisions D001–D007 and the #721/#722
human review order remain unchanged. Project Decision synchronization remains
outstanding from its previously recorded 403; no Project update was claimed.

## Evidence inventory

The new local evidence directory contains `manifest.json`, immutable source
hashes, `progress.md`, copied baseline commands/supervisor, per-batch JSON/logs,
`events.jsonl`, `results.json`, predeclared `hypotheses.json`, child source and
stdout/stderr, phase JSONL, import profiles, syscall traces,
`diagnostic-analysis.json`, `completion-audit.json` and process cleanup records.
Supervisor elapsed times include up to 0.5 seconds of polling granularity;
the child timings in the plotting table use the more precise parent-run
markers. Raw local files are retained, not deleted on completion.

The investigator reviewed the report against raw results and the approved plan;
no delegated review was used, as specified by the plan. Generic software
implementation workflow steps were adapted to this diagnostic task: no repair
or RED/GREEN implementation cycle was performed. The cost is that any later
remedy still needs its own review and verification. Retaining raw local evidence
costs disk space and is not a remote backup; substantive findings are preserved
in the GitHub comments above.
