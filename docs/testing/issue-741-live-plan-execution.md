# #741 live simulation plan execution

Implementation approval is D037. Prerequisite #823 merged as
`437c6898eaed4c1728be7ac1dc3951190e0208fd`; freshly fetched origin/main matched
before creating the clean isolated `codex/issue-741-live-plan-ticks` worktree.
The [approved plan](../superpowers/plans/2026-11-16-issue-741-live-plan-ticks.md)
and [machine-readable receipt](evidence/issue-741-local-results.json) retain scope,
commands, executed counts, source hashes and state traces.

## Runtime behavior

Strategy requests preserve canonical UUIDs and fixed native Goal contracts.
The controller retains ordered action occurrences and a cursor. Routine, native
and dictionary events, LLM decisions and legacy application aliases share one
simulation-time dispatch boundary. Default cadence is one second; pause freezes
it, delayed updates coalesce whole intervals and preserve the fractional remainder.

Higher-priority work interrupts between steps; equal/lower requests wait. An
interrupted goal resumes through fresh planning. Complete goals are checked against
actual state; queue exhaustion or an action returning True cannot fabricate success.
Before execution, validate the remaining tail from graph state after weather. After
execution, verify modeled effects against graph, object and seeded aliases before
crediting success. Partial failures retain real partial writes and do not replay.

Failure handoff uses fresh StrategyManager planning and semantic action exclusions,
including regenerated candidates. Unchanged blocked work does not search repeatedly;
a five-second cooldown plus changed goal/action dependencies or catalog starts a new
bounded failure epoch. Unrelated scalar/other-actor writes do not wake it. Reset,
load, actor removal and world/manager replacement invalidate retained bindings.

Legacy Plan.replan now returns a truthful boolean, preserves full replacements and
occurrence dependencies, reevaluates previously completed predicates, and clears
unusable queues. Manufactured random-success alternatives are removed. Synchronous
Plan.execute remains a manual compatibility API, not the live scheduler.

## Executed verification

Final canonical commands use the existing Python3.12 dependency-profile venvs:

```sh
python -B scripts/run_ci_checks.py --profile core --artifacts <core-receipts>
python -B scripts/run_ci_checks.py --profile dev --artifacts <dev-receipts>
python -B scripts/run_ci_checks.py --profile visual --artifacts <visual-receipts>
```

The linked receipt records their exact interpreter and artifact paths.

| Profile | Tests | CLI probes | Result |
| --- | ---: | ---: | --- |
| core | 224 | 3 | PASS, exit0 |
| dev | 20 | 0 | PASS, exit0 |
| visual | 70 | 0 | PASS, exit0 |

Every batch exited0; zero test failures, errors or skips. The dedicated core
active-plans batch retains existing child25s/fatal20s/supervisor120s limits. Visual
includes one dummy-SDL scheduler/speed parity test. No dependencies or locks changed.

Observed live traces through actual update_game_state:

- Rest then Work on separate eligible intervals: energy20->35->5,
  wealth5->5->25; actual production Goal satisfied, graph/object agree.
- Default daily plan: Rest4 then Leisure4 across eight intervals;
  energy80/satisfaction80 and COMPLETE. No repeated per-render prefix.
- 480 updates of1/60s and80 updates of.1s produce the same eight-action trace.

The retained D030 reproduction now exits0 and emits replan_result=true with a
real Rest replacement. Separate regressions execute repeated real replacements,
prove goal outcomes, and check explicit no-alternative/exhausted-limit behavior.

## Independent review and corrections

The repo-defined architecture, planning/actions, world-state, UI/simulation,
LLM-contracts and test/CI specialists supplied read-only investigations. A fresh
independent reviewer examined actual source/readbacks supplied in the coordinator
history; it ran no commands or disk-inspection tools. Native Zcode discovery was
not exercised. The coordinator alone edited files and executed checks.

The reviewer identified five issues: untargeted native Events with empty participant
lists; nested typed legacy decisions; repeated-occurrence dependencies; historical
completion reported as current satisfaction; and silently dropped preempted legacy
sequences. It also identified recovery delay lost across preemption. Each received
an observed failing real-world regression before its fix. All seven targeted
review/UUID regressions passed; the final profiles above ran after the fixes.

Implementation rulings:

- Goal-less legacy sequence tails cannot be replanned toward an unknown goal.
  Urgent preemption emits CANCELLED_PREEMPTED with cursor/remaining action names.
  Goal-bearing requests retain the approved preserve-and-replan behavior.
- Basic save/load testing uses valid JSON input: save_game_state already fails on
  UUID dictionary keys. #741 clears scheduler state on successful basic load and
  does not add a persistence format or claim that serialization defect is repaired.
- A partial-write fixture's original expensive alternative exceeded the existing
  1000-expansion bound with the complete catalog, despite working alone. Fixture
  costs.1/.2 retain actual failure/replacement execution within unchanged bounds.
- Fixtures must declare initiator effect targets and required Character personality/
  time-manager inputs. Quest fixtures need assigned_time because real updates expire
  old quests. Corrected fixtures preserve production semantics and exact-once checks.
- GraphManager construction is singleton-based; the replacement regression explicitly
  resets that singleton to create and assert a distinct real world.

## Limits

Existing job-parser ERROR stderr remains, including the retained reproduction's
117-count/116-parsed mismatch; passing assertions do not imply clean stderr.
Offline provider-boundary tests establish cached availability, routing and reuse;
they do not establish live inference. Existing catalog/inventory/callback limits,
full save restoration, native Windows/GPU and later product/phase acceptance remain
separate. This change does not approve merging #741 or any later human gate.
