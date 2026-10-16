# Issue #779 Failure Capture Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this
> plan sequentially after human approval. Preserve the existing execution
> method; do not delegate or create a worktree. Track the steps below.

**Status:** Approved; implementation in progress. See D008 in the single
decision record for authorization and scope. Human approvals belong only in
[docs/decisions.md](../../decisions.md).

**Goal:** Preserve useful evidence from the next startup failure without
changing the behavior, assertions or deadlines being investigated.

**Architecture:** Add one standard-library subprocess capture helper used by
the existing graph and cache integration helpers. Write child output and
phase events directly to durable files, preserve existing unittest outcomes,
and run one bounded verification matrix after implementation.

**Tech Stack:** Prepared Python 3.12.14, unittest, subprocess, faulthandler,
pathlib, JSON and optional read-only Linux `/proc` observations. No installs.

**Spec:** [#779](https://github.com/dhar174/tiny_village/issues/779), the
[completed investigation](../../testing/issue-779-startup-investigation.md),
and decisions D001–D007 in the single decision record. This is the proposed
observability follow-up, not a claim to satisfy all of #779's acceptance criteria.

## Global constraints

- Baseline runtime/tests: `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`.
  Planning branch starts at documentation commit `29063c3`. Recheck upstream
  and source differences before implementation; do not mix changed baselines.
- Graph children retain **20 seconds fatal / 25 seconds parent** at the
  default call site; the existing `timeout` parameter still gives
  `fatal_timeout = timeout - 5`. Cache integration retains **25/30 seconds**.
  Verification batches retain **110/120 seconds**.
- Preserve all current source assertions, eight import orders, graph identity,
  concurrency, recursion/retry, injection/ownership and cache/loader checks.
- Preserve offline model flags, dummy SDL, two BLAS/OpenMP threads and all
  existing cache fixture overrides. Do not change runtime modules, dependencies,
  import structure, plotting-cache selection or cloud configuration.
- Windows, live models, gameplay turns, #773–#776, CI creation and the full
  repository suite remain outside this task. Existing later roadmap tasks
  retain that work.
- Keep #779 open and #721 Deferred; keep #722 open. Passing capture checks
  does not establish the historical cause or approve a phase.

## Evidence and current gap

The completed cycle passed 51 affected methods and all 16 predeclared
diagnostic cases. The historical timeout did not recur. Fresh plotting-cache
work added about 1.95 seconds median in the measured comparison, which does
not explain the earlier 20/25-second failures.

`FreshPythonTestCase.run_python` in `tests/test_graph_initialization.py`
currently captures output in pipes and reports bounded tails. Ownership tests
inherit that helper. `TestModelCacheIntegration.run_python` in
`tests/test_model_cache.py` has a separate pipe-capture implementation.
An aggregate fatal exit can prevent a final test summary and orderly output
collection. Cache fixture cleanup also removes its temporary directory, making
that directory unsuitable for retained diagnostics.

File capture changes output transport and adds small instrumentation costs.
Document that boundary: a successful instrumented run cannot prove the
original pipe-based timeout fixed. Do not add import-time profiling, syscall
tracing, statement rewriting or a signal-handshake subsystem by default.

## File boundaries and dependency order

| File | Responsibility |
| --- | --- |
| New `tests/subprocess_capture.py` | One reusable child launcher, durable files, phase events, timeout preservation and bounded resource observations |
| New `tests/test_subprocess_capture.py` | Real-process checks for this new capture behavior; no game fixtures or model dependencies |
| `tests/test_graph_initialization.py` | Replace only `FreshPythonTestCase.run_python` plumbing; ownership tests inherit it |
| `tests/test_model_cache.py` | Replace only `TestModelCacheIntegration.run_python` plumbing; retain fixture setup and all assertions |
| `.gitignore` | Ignore the proposed root `.test-artifacts/` directory |
| New `docs/testing/issue-779-failure-capture.md` | Usage, verified outcomes, evidence paths, limitations and handoff |
| `docs/decisions.md`, Memory Bank task/index/current summaries | Canonical approval status and resumability pointers |
| Owned `/workspace/scratch/issue-779-capture/<run-id>/` | Temporary verification supervisor, manifest, batch logs and results; not application code |

**Order:** Task 1 → Task 2 → Task 3 → human findings review. Each task depends
on the previous task's verified interface. Submit one scoped implementation PR
after approval and execution; do not alter ready documentation PR #780.

## Review focus

1. Abrupt parent exit: retain emitted child output and unfinished metadata;
   never report an incomplete run as passing. Task 1 owns the check.
2. Temporary-cache cleanup: diagnostics survive outside fixture `TMPDIR` and
   cannot overwrite earlier attempts. Task 1 owns the check.
3. High-volume output and non-ASCII text: retain full raw bytes, provide useful
   console tails, and avoid pipe deadlock. Task 1 owns the check.
4. Deadline/process cleanup: keep fatal and parent timers distinct; terminate
   and reap only owned processes. Tasks 1 and 3 own the checks.
5. Missing diagnostics capability: an unwritable artifact path is an actionable
   error; unavailable `/proc` data is explicitly unavailable. Task 1 owns the check.

### Task 1: Durable child capture

**Files:** Create `tests/subprocess_capture.py` and
`tests/test_subprocess_capture.py`; modify `.gitignore`.

**Interfaces:**

```python
@dataclass(frozen=True)
class PythonCapture:
    completed: subprocess.CompletedProcess[str]
    artifact_dir: Path

class CapturedTimeout(subprocess.TimeoutExpired):
    artifact_dir: Path

def capture_python(
    source: str, *, cwd: Path, env: Mapping[str, str], test_id: str,
    fatal_timeout: float, timeout: float,
) -> PythonCapture: ...
```

Consumes the caller's already-dedented source, working directory, environment
and unchanged time limits. Returns captured output and status without asserting
success; a parent timeout raises `CapturedTimeout`, still catchable as
`subprocess.TimeoutExpired`. Invalid limits fail before launching a process.

- [x] Add real-process checks before implementing the helper. Cover successful
  output/return code; nonzero exit; output larger than current 3000/6000-character
  console tails; non-ASCII output; two calls with the same test ID; and artifact
  survival after removal of a separate temporary cache directory. Assert raw
  files retain all bytes emitted by each controlled child.
- [x] Add controlled fatal-watchdog and parent-timeout cases. Only these new
  synthetic cases use short limits; existing graph/cache limits stay unchanged.
  The fatal case must retain a faulthandler dump and nonzero exit. The parent
  case deliberately cancels its child watchdog, then verifies `CapturedTimeout`,
  retained output, and child termination/reaping. Avoid tight timing assertions.
- [x] Add an abrupt-parent-exit check using an owned outer supervisor. Confirm
  that flushed child output and started metadata survive `os._exit`, without
  inventing a final exit code. The supervisor must clean up/reap its own process
  group even if an assertion fails. Add unwritable-root and unavailable-resource
  checks; do not turn missing capture into a silent passing fallback.
- [x] Run the new checks once and confirm they fail for the missing helper.
  Implement the interface with these artifact rules:
  - Root: `TINY_TEST_ARTIFACT_DIR` if provided, otherwise
    `cwd / '.test-artifacts' / 'subprocess'`; resolve relative overrides against
    `cwd`. Never derive this location from fixture `TMPDIR` or model caches.
  - Unique owned directory per invocation; new leaf directories/files are
    private on POSIX. Do not chmod parents, change the process umask, follow
    pre-existing leaf symlinks or delete old artifacts.
  - `stdout.log` and `stderr.log` are direct child file targets, not buffers
    saved only after completion. Preserve raw bytes; decode returned strings
    with explicit UTF-8 replacement for invalid diagnostic bytes.
  - `run.json` records schema version, test ID, source SHA-256, interpreter,
    cwd, PID when available, deadlines, timestamps, return code and observed
    outcome. Write initial metadata before launch and replace it atomically on
    updates. A started record with no terminal outcome means incomplete.
  - `events.jsonl` records flushed `child_ready`, `source_complete` and an
    `atexit_marker` with monotonic times. Register the existing fatal watchdog
    before added diagnostic work; do not reset it after instrumentation.
    Preserve the source's `__main__` execution context and append completion
    only after its original statements finish successfully.
  - Parent observations distinguish spawn error, exited-zero, exited-nonzero
    and parent-timeout. Record watchdog evidence separately; do not infer a
    particular failure cause from elapsed time or exit code alone.
  - If still running two seconds before its fatal deadline, collect one
    bounded Linux process/resource snapshot into `resources.json`. Read PID
    status, CPU/wait information and available cgroup counters; do not read
    environment or command-line contents. Missing/denied values remain unknown.
    Diagnostic reads must not extend the original parent timeout.
  - Use the existing process group: separating every child into a new session
    would let it escape the verification supervisor's group cleanup. On a
    normal helper parent-timeout, kill/reap the direct owned child, matching
    the existing launcher. The outer verification supervisor owns descendant
    group cleanup. Do not promise standalone descendant cleanup after parent
    death; preserve that limitation in usage documentation.
  - Store no environment dump, credentials, raw source or model contents in
    metadata. Raw child output can contain user data; review it before uploading.
    Do not automatically publish raw artifacts.
- [x] Run `/workspace/.venvs/tiny_village/bin/python -m unittest -v
  tests.test_subprocess_capture` and verify every intended case completed.
  Buffered application output not yet written to a descriptor is unrecoverable;
  `atexit_marker` is not proof of final process exit. Document both limits.
- [x] Commit the helper, its focused behavior checks and `.gitignore` entry.

### Task 2: Integrate existing startup helpers

**Files:** Modify the two helper methods identified above, without changing
their test bodies, fixtures or the separate stdlib-import guard in
`tests/test_model_cache.py`.

**Interfaces:** Consume Task 1's `capture_python` and `CapturedTimeout`.
`FreshPythonTestCase.run_python(source, timeout=25)` continues returning a
`CompletedProcess[str]`. Cache `run_python(source)` retains its current caller
contract. Supply `self.id()` plus the source digest/unique directory so
different subcases of one test remain distinguishable.

- [x] Replace graph pipe-launch plumbing with `capture_python`, passing the
  current environment, `timeout - 5`, `timeout` and original logging setup.
  Keep caught parent timeouts as unittest failures. Include the artifact path
  and bounded output tails in failure messages.
- [x] Replace only cache integration pipe-launch plumbing, passing its existing
  fixture environment and 25/30-second limits. Keep nonzero exits as assertion
  failures and uncaught parent timeouts as unittest errors. Include the artifact
  path in both diagnostics; preserve the exception's standard timeout fields.
- [x] Review the source diff to confirm all embedded assertions, subcases,
  concurrency rounds and loader-verification settings are unchanged. Verify
  the shared helper imports under both existing module and discovery commands;
  do not introduce a `tests/__init__.py` solely for this change.
- [x] Use the first four batches of Task 3 as integration verification rather
  than running an additional duplicate affected matrix. If they reveal a
  capture implementation defect, repair it and explicitly report any required
  rerun; never replace the original failed attempt in the evidence.
- [x] Commit integration after those batches complete, or report the specific
  failed/incomplete boundary. No deadline adjustment is a permitted fix.

### Task 3: One bounded observation window and evidence handoff

**Files:** Create `docs/testing/issue-779-failure-capture.md`; update task
tracking and link approval/outcome status from `docs/decisions.md`.
Temporary run files stay in the owned scratch directory listed above.

**Interfaces:** Consume Task 2's integrated helpers and Task 1 artifacts.
Produce an exact input manifest, all attempt records and an evidence report.

- [x] Record the exact checkout/runtime diff, interpreter, dependency metadata,
  platform, effective non-secret test settings and hashes. Preserve previous
  evidence directories. Do not call a fresh Python process or cache a fresh
  cloud instance; record which condition actually applies.
- [x] Reuse the historical verification target definitions and limits in a
  temporary sequential supervisor. Give each batch its own process group,
  retain batch output directly to files, and clean up/reap only that group's
  descendants. Missing unittest completion counts are null/incomplete, not zero.
- [x] Run this single ordered matrix, with no unrelated suite expansion:

  | Order | Target after `python -m unittest` | Existing methods |
  | --- | --- | ---: |
  | 1 | `-v tests.test_graph_initialization.TestGraphImports` | 1, eight subcases |
  | 2 | `-v tests.test_graph_initialization.TestGraphLifecycle` | 12 |
  | 3 | `discover -s tests -p test_graph_ownership.py -v` | 11 |
  | 4 | `-v tests.test_model_cache` | 27 |
  | 5 | `-v test_core_global_graph_manager` | 9 |
  | 6 | `discover -s tests -p test_action_named_target_resolution.py -v` | 6 |
  | 7 | `discover -s tests -p test_actions.py -v` | 5 |
  | 8 | `discover -s tests -p test_event_handler_payloads.py -v` | 5 |
  | 9 | `discover -s tests -p test_goap_error_handling.py -v` | 5 |

  These total **81 existing methods**; count the new capture-helper checks
  separately. First four batches are the 51-method affected subset, not an
  extra preliminary matrix. Run the three existing cold-getter, production
  Goal-fallback and minimal-mode probes once with the saved original assertions
  and 110/120-second limits. Minimal mode still proves zero actions only.
- [x] Treat this as a new, explicitly approved budget of nine batches plus
  three probes, separate from the exhausted 16-case investigation. Stop on a
  startup timeout/aggregate cutoff, changed inputs, unsafe cleanup or unavailable
  required capture. Record later targets as unrun. There are no automatic
  retries, additional cold-host trials or new profiling experiments; a captured
  failure leads to a finding and scoped next proposal. Development checks for
  the new helper are not extra application-observation attempts.
- [x] Review artifact usability: a failure message identifies an existing path;
  metadata identifies the test/source; output is complete for emitted bytes;
  phase markers and parent exit remain distinct; interrupted counts are honest.
  Report overhead/transport limitations without claiming an unmeasured speedup.
- [ ] Write the outcome report and prepare a scoped PR referencing #779 with
  no automatic closing keyword. Add a concise findings/evidence link on #779
  and #721 after verifying published links. Do not merge or approve reviews.

## Approval and completion boundaries

**Before implementation:** Review this plan, including test-harness edits,
focused capture checks and the new bounded observation budget. Record approval
only in `docs/decisions.md`; do not reinterpret D007 as approval of this work.

**After implementation:** Human review assesses the capture change and findings.
Tooling is complete when its controlled failure checks work and the bounded
application observations are accurately reported. #779 remains unresolved if
the historical cause remains unknown, even if the 81 methods pass. Any later
runtime/environment correction or deadline proposal needs evidence and its
own review. #721's review and #722's phase decision remain separate steps.

**Rollback:** Revert the capture/helper integration commits to restore original
launch plumbing; leave prior raw evidence and human decisions intact. No
runtime migrations, dependency changes or shared-cache cleanup are involved.
