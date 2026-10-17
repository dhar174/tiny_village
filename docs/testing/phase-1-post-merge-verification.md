# Phase 1 post-merge verification — 2026-10-16

**All 94 test methods report success and all three probes exit 0, with one
supervisor completion gap.** This is not a fully confirmed green supervisor
run: the lifecycle suite prints `Ran 12 tests ... OK`, but its supervisor's
final exit/cleanup record was lost during an execution-server disconnect.
That gap remains explicit; no target was rerun to replace it.

This window was authorized under D009 in the single
[decision record](../decisions.md). Subsequently D010 explicitly accepted the
unknown historical cause and supervisor gap and authorized #779's closure as
unable to reproduce; diagnostics added. D011 subsequently records the
maintainer's approval of #721 for Linux/offline startup. #721 is closed as
completed; #722 remains open for its separate phase-gate decision.

Published in [draft evidence PR #782](https://github.com/dhar174/tiny_village/pull/782).
Findings were posted and read back on
[#779](https://github.com/dhar174/tiny_village/issues/779#issuecomment-6227420596)
and [#721](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6227421138).
Their dedicated status blocks were refreshed while preserving all original
acceptance text. The initial evidence publication changed no human disposition;
the later D010 accepted-risk decision supplies #779's closure rationale.

Disposition read-back confirms [#779's closure](https://github.com/dhar174/tiny_village/issues/779#issuecomment-6227664671)
with reason `not_planned` and [#721's unchanged Deferred status](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6227665778).
Those receipts precede D011's separate approval of #721; its earlier Deferred
disposition remains historical evidence. #717 and #722 remain open.
The [subsequent #721 approval](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6229486733)
was posted and read back; #721 is closed with reason `completed`. Native
#779 → #721 → #722 relationships are retained; #722 remains open.

## Inputs and boundaries

- Exact merged main: `77e4b9437cc00970def933d992140e6da37eae8e`, including
  merged PRs #780 and #781 and the corrected subprocess wait budget.
- Started **2026-10-16 at 20:28:29 EDT** in the existing prepared Python 3.12.14
  CPU/Linux instance. Fresh interpreters and fixture caches are not a fresh host.
- No runtime/test/manifest edits, installs, model downloads, persistent cache
  changes or deadline adjustments. All 390 captured Python/manifest hashes
  match before the window, at continuation and after completion.
- Existing offline HF/Transformers settings, owned temporary/cache directories,
  dummy SDL and two BLAS/OpenMP threads remain in place. Cache integration retains
  its own HF/Hub/Transformers and plotting-cache overrides.
- Graph fatal/parent limits: **20/25 seconds**; cache integration: **25/30 seconds**;
  every test/probe batch: **110/120 seconds**. Original assertions are unchanged.
- One attempt per target: 13 capture methods, 81 existing matrix methods and
  three original probes. No new diagnosis, profiling or extra application runs.

## Reported results

| Target | Methods reported successful | Seconds | Parent/supervisor evidence |
| --- | ---: | ---: | --- |
| Capture checks, including launch/metadata delay regressions | 13/13 | 5.564 | Exit 0, cleanup recorded |
| Graph imports, all eight orders | 1/1 | 36.990 | Exit 0, cleanup recorded |
| Graph lifecycle, including concurrency and retry | 12/12 | 37.169* | Unittest OK and 15 child zero-exits; final supervisor record missing |
| Graph/action ownership | 11/11 | 42.796 | Exit 0, cleanup recorded |
| Cache helpers and loader integration | 27/27 | 16.000 | Exit 0, cleanup recorded |
| Core global graph manager | 9/9 | 4.393 | Exit 0, cleanup recorded |
| Named-target resolution | 6/6 | 1.470 | Exit 0, cleanup recorded |
| Actions | 5/5 | 0.539 | Exit 0, cleanup recorded |
| Event payloads | 5/5 | 0.603 | Exit 0, cleanup recorded |
| GOAP errors | 5/5 | 1.559 | Exit 0, cleanup recorded |

*Lifecycle seconds come from its unittest summary; the other values are
parent-observed batch durations. **82 methods** have confirmed supervisor
completion; the remaining **12** have a successful test summary and complete
child evidence, with the supervisor completion gap. No test failure, error or
skip is reported in any completed unittest summary.

| Original probe | Result | Scope limit |
| --- | --- | --- |
| Cold getter, repeated identity and graph alias | Exit 0, 5.217s | Prepared dependency profile; optional embedding initialization still logs failure |
| Production Goal fallback with character import unavailable | Exit 0, 0.172s | Controlled unavailable-character boundary |
| `main.py --mode minimal --no-llm --headless` | Exit 0, 2.887s | **Zero actions executed**, not a gameplay-turn demonstration |

Minimal mode retains the mock-based MapController initialization error,
undefined `ENGLISH_STOP_WORDS` sentiment fallback, and missing `char_emma`
graph-node update messages. Its banner does not establish full simulation
readiness. Native Windows, live models, display/gameplay and the full repository
suite remain outside the agreed Phase 1 scope.

## Connection loss and bounded continuation

The execution-server transport disconnected and its session could not be resumed.
Saved results confirmed the capture and import batches. The lifecycle stderr
retained its complete 12-test OK summary, and all 15 lifecycle child records
contained parent-observed zero exits. The original supervisor's terminal record
was absent, so its lifecycle outcome remains `SUPERVISOR_INTERRUPTED`, with
**null parent exit code and cleanup confirmation**. It was not relabeled PASS.

After checking retained files and finding no live process holding output files
for this run, execution continued **only the NOT_RUN targets**, once each. The
continuation supervisor was detached from the tool connection and recorded its
batch PIDs/groups before waiting. It completed ownership, cache, five smaller
suites and the three probes. Their exit/cleanup records are retained.

Original `results-before-resume.json` and the original logs remain untouched.
The continuation deliberately returns a nonzero overall status because the
original supervisor gap persists; its completion marker and results distinguish
that reporting gap from an application assertion failure. The disconnect does
not establish the cause of the historical startup timeouts.

All **42 integration child artifact sets** were checked: 37 graph and five cache
children, each with stdout/stderr, ready/source-complete/atexit markers and a
parent-observed zero exit. No child watchdog dump is observed. An atexit marker
alone was not treated as proof of exit. The missing lifecycle batch supervisor
record remains a separate limitation despite its complete child evidence.

## Evidence and decision handoff

Local raw evidence lives at
`/workspace/scratch/phase1-post-merge/2026-10-16-main-77e4b94/`:

- `manifest.json`, `resume-manifest.json`, `source-integrity.json`, `evidence-audit.json`.
- `verify.py`, `continue-unstarted.py`, original and continuation logs.
- `results-before-resume.json`, `results.json`, `continuation-complete.json`.
- Per-target stdout/stderr and the 42 retained child artifact sets.

These are local paths, not remote artifact links. Earlier failed runs and both
previous bounded investigation windows remain preserved.

The historical timeout did not recur; its cause remains unknown. The application
summaries support revisiting #779, while the supervisor gap prevents claiming an
entirely clean supervised run. D010 records the maintainer's subsequent explicit
acceptance of both limitations and authorization of closure as **"unable to
reproduce; diagnostics added"**. #779 is closed with reason **not planned**;
its original root-cause criteria are not claimed complete. Reopen on recurrence
and retain the diagnostic artifacts.

The #779 decision and #721 approval are recorded under D010/D011. Next obtain
#722's separate Phase 1 decision. Do not close #717 or advance to
Phase 2 until those required human decisions are recorded in `docs/decisions.md`.
Project Decision synchronization remains outstanding from the previous 403;
no new access attempt or synchronization is claimed here.
