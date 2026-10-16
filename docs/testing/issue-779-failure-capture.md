# Issue #779: durable startup failure capture

**Capture implementation verified; historical timeout cause still unknown.**
All 11 new capture checks, 81 existing test methods and three original probes
pass. No startup timeout recurred in this single approved observation window.
This implements the observability follow-up; it does not resolve #779 or
approve startup review #721 or phase gate #722.

Authorization is D008 in the single [decision record](../decisions.md).
The [implementation plan](../superpowers/plans/2026-10-16-issue-779-failure-capture.md)
defines the unchanged deadlines, scope and fixed observation budget.

Published as [draft PR #781](https://github.com/dhar174/tiny_village/pull/781),
stacked on documentation PR #780. Findings were posted and read back on
[#779](https://github.com/dhar174/tiny_village/issues/779#issuecomment-6205767601)
and [#721](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6205768056).
The dedicated #779 status block was updated without changing its original
acceptance text or open state. #780's head and ready-for-review state are unchanged.

## Changes and usage

[The shared helper](../../tests/subprocess_capture.py) replaces only the launch
plumbing in `FreshPythonTestCase.run_python` and
`TestModelCacheIntegration.run_python`. Ownership tests inherit the graph
helper. Existing test bodies, fixtures, assertions, subcases and the independent
model-cache stdlib import guard are unchanged. No runtime, dependency, model,
cloud configuration or shared-cache changes were made.

Run the existing unittest commands from the repository root. Each affected
child now gets a unique directory under `.test-artifacts/subprocess/`, ignored
by Git. To retain diagnostics elsewhere, set `TINY_TEST_ARTIFACT_DIR`; relative
values resolve against the repository working directory. That root is
independent of fixture `TMPDIR`, HF cache paths and their cleanup.

For example, in a POSIX shell with the prepared cloud interpreter:

```bash
TINY_TEST_ARTIFACT_DIR=/workspace/scratch/startup-capture \
  /workspace/.venvs/tiny_village/bin/python -m unittest -v \
  tests.test_graph_initialization.TestGraphImports
```

This example runs the target directly. The approved verification window below
additionally used its original 110/120-second batch supervisor.

| Artifact | Meaning |
| --- | --- |
| `run.json` | Test ID, source digest, interpreter/cwd, PID, deadlines, timestamps and parent-observed outcome |
| `stdout.log`, `stderr.log` | Full bytes emitted to the child descriptors, written directly rather than retained only in parent memory |
| `events.jsonl` | Flushed child-ready, successful source-completion and atexit markers with monotonic times |
| `resources.json`, when available | One near-deadline process/cgroup snapshot; missing observations remain unknown |

Failure messages include the artifact directory plus bounded output tails.
Graph parent timeouts remain unittest failures; cache integration parent timeouts
remain unittest errors. The graph helper still returns a `CompletedProcess`.
`CapturedTimeout` remains a `subprocess.TimeoutExpired` subclass with its usual
command, timeout, output and stderr fields, plus the artifact path.

The parent distinguishes exited-zero, exited-nonzero, parent-timeout and spawn
error. A STARTING/STARTED record without a terminal outcome is incomplete,
never passing. Observing a watchdog dump is recorded separately from cause.
The helper does not infer a graph deadlock from a nonzero exit or elapsed time.

Unique leaf directories/files are private on POSIX; existing parents and caches
are not chmodded or cleaned. Metadata excludes environment dumps and raw source.
Raw child output may contain application/user data; review it before sharing.
No raw logs are automatically uploaded.

## Revision and environment

- Runtime baseline: `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`.
- Validated code commit: `31e8cdb0cdc89db996e0a5469fd76749a5a558c7`.
  Execution began at helper commit `ed19d61` with the two integration helpers
  uncommitted; `validated-commit.json` confirms the committed code is byte-for-byte
  equivalent to that tested tree. Later report-only changes do not alter it.
- Prepared Python 3.12.14 CPU/Linux instance; no new instance was provisioned.
  Fresh interpreters and fixture-owned cache directories are not an OS-cold host.
- Dependency metadata and 390 Python/manifest file hashes are retained in the
  run manifest. Hashes match before and after the window and the code commit.
- Offline HF/Transformers flags, owned HF/TMP paths, dummy SDL, two BLAS/OpenMP
  threads, no bytecode writes and the existing plotting-cache configuration are
  unchanged. Cache integration still creates its own plotting-cache directory.
- Child limits remain graph **20/25 seconds** and cache **25/30 seconds**;
  batch fatal/parent limits remain **110/120 seconds**.

AST comparisons against merged main verify every existing method/fixture is
unchanged apart from the two launch helpers. Ownership test source is byte-identical.

## New capture checks: 11/11 pass

`python -m unittest -v tests.test_subprocess_capture` completes in 2.658 seconds
with no failures, errors or skips. Checks cover:

- Main-module globals and future imports; successful Unicode output and markers.
- Nonzero exits, output beyond console tails and invalid diagnostic bytes.
- Repeated test IDs, separate artifact roots and survival after cache cleanup.
- A deliberately triggered fatal watchdog and retained Python traceback.
- A deliberately triggered parent timeout and direct-child reaping.
- Abrupt parent exit with durable output/incomplete metadata and owned orphan cleanup.
- Invalid deadlines, unusable artifact paths, real spawn failure and unknown
  process/resource observations.

These synthetic cases intentionally use short limits. They do not change graph
or cache deadlines and are not application-timeout reproductions. The initial
red run was one loader error for the missing helper, not 11 executed failures;
the green run executed all 11 methods after implementation.

## Approved observation window: 81/81 plus three probes

One sequential window ran 2026-10-16 01:07:35–01:09:46 UTC. There were no
automatic retries or extra application observations. The first four batches
are the 51-method affected subset within this 81-method matrix, not a duplicate run.

| Target | Methods passed/executed | Batch seconds |
| --- | ---: | ---: |
| Graph imports, including all eight orders | 1/1 | 23.277 |
| Graph lifecycle, including mixed-thread access and retry | 12/12 | 45.984 |
| Graph/action ownership | 11/11 | 37.500 |
| Model-cache helpers and loader integration | 27/27 | 14.265 |
| Core global graph manager | 9/9 | 2.304 |
| Named-target resolution | 6/6 | 0.250 |
| Actions | 5/5 | 0.270 |
| Event payloads | 5/5 | 0.273 |
| GOAP errors | 5/5 | 0.571 |

All nine batches complete: **zero failures, errors, skips, child timeouts or
incomplete batches**. All owned process groups and adopted descendants were
checked and reaped. Exact commands are retained in `results.json` and the
approved plan.

The integrated helpers create **42 child artifact sets**: 37 graph children
with 20/25-second limits and five cache children with 25/30-second limits. Every
set has both output files, all three phase markers and a parent-observed zero
exit. No watchdog dump is observed. Parent-observed child durations range from
0.115 to 5.103 seconds; this is not an instrumentation-overhead benchmark.

| Original probe | Result | Limit of the result |
| --- | --- | --- |
| Cold getter with repeated identity and graph alias assertions | Pass, exit 0, 2.680s | Optional embedding initialization logs failure in this dependency profile |
| Production Goal fallback with character import unavailable | Pass, exit 0, 0.169s | Controlled unavailable-character boundary, not full Character integration |
| `main.py --mode minimal --no-llm --headless` | Pass, exit 0, 3.040s | **Zero actions executed**; startup/fallback smoke only |

Minimal-mode output still reports a MapController initialization failure from
its mock-based demo boundary, undefined `ENGLISH_STOP_WORDS` sentiment fallback,
and updates to missing `char_emma` nodes. Its own success banner and tracked-action
count do not establish actual executed gameplay actions. These messages remain
explicit limitations; no full simulation, display, model or gameplay readiness
claim is made.

## PR #781 deadline review correction — 2026-10-16

The review identified a harness regression: starting the parent deadline before
`Popen` and the STARTED metadata write deducted setup time from the configured
child wait budget. The helper now records launch time after process creation
and starts the parent wait deadline after capture setup. Resource sampling
remains tied to launch time; the child's fatal watchdog and configured limits
are unchanged. Completed metadata records `wait_started_monotonic` separately.

Two regression cases advance only the launcher's clock across process creation
or STARTED metadata persistence while running real children. Both failed before
the correction. All **13 focused capture checks pass** afterward, with no skips,
failures or errors (3.352 seconds). The real-process review reproduction with a
controlled 0.7-second pre-launch delay also now exits 0: the previous launcher
and corrected helper finish in 1.448 and 1.478 seconds overall, respectively,
using the same 1.2-second parent and 0.9-second fatal limits.

Correction evidence is local at `/workspace/scratch/pr-781-deadline-fix/`:
`red.log`, `green.log`, `real-launch.log` and the reproduction script/results.
The 81-method application matrix and three probes above remain historical
evidence for `31e8cdb`; they were **not rerun for this correction**. No additional
application observation window, runtime remedy or issue closure is claimed.

## Evidence, review and remaining work

Local evidence: `/workspace/scratch/issue-779-capture/2026-10-16/`:

- `manifest.json`, `source-integrity.json`, `validated-commit.json`.
- `verify.py`, `results.json`, `window.log`, per-target stdout/stderr files.
- `children/` with the 42 verified artifact sets.
- `helper-red.log`, `helper-green.log`, and the execution ledger.

These paths are local, not remotely accessible artifact URLs. This report
retains the substantive results; prior failed/incomplete runs and the earlier
16-case investigation remain untouched.

The code received a separate author review against the approved plan; no
independent agent review was performed because the approved execution method
excludes delegation. Reviewers should assess capture semantics and process
ownership before merging.

File output changes the former pipe transport. Phase markers and resource
observation also affect timing; this window does not measure that overhead or
prove the original timeout fixed. Output still buffered inside the child cannot
be recovered after a fatal exit. An atexit marker is not proof of final process
exit. A daemon resource observer cannot delay the parent deadline; if its file
is absent, the resource observation is unknown.

Children retain the supervisor's process group. The helper kills/reaps its
direct child on parent timeout; the outer supervisor owns group/descendant
cleanup. An arbitrary standalone parent's abrupt death does not guarantee
automatic descendant cleanup, although emitted logs and started metadata survive.
No power-loss recovery or remote artifact backup was tested.

The D008 observation budget is now consumed. No causal explanation or justified
runtime correction follows from another passing window. #779 stays open;
#721 stays Deferred and #722 stays open. A future recurrence can now leave
durable evidence, but further application observations, a runtime/environment
remedy or a deadline-contract change need their own reviewed scope. Project
Decision synchronization remains outstanding from the previously recorded 403.
