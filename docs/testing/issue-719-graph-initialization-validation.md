# Issue #719: Graph initialization and ownership validation

Implementation follows the [approved plan](../superpowers/plans/2026-10-14-issue-719-graph-initialization.md) and was merged in [PR #777](https://github.com/dhar174/tiny_village/pull/777). The branch-validation record below is retained; see the post-merge verification at the end. Human phase decisions remain separate.

## Revisions and repaired behavior

- Base main: `610b4f871694a380f3fedfd48664c188a169cbee` (merged #706; #718 closed).
- Final runtime revision tested: `6667a0b615d615054d6b2701494e220df45ef2d4`.
- Subsequent documentation commits do not change runtime/test code.

Cold global access held a non-reentrant lock while importing graph → utility → characters. The character module constructed Festival/Apple example objects, whose action systems called the global getter again. The planning baseline reproduced the hang in cold graph access and the named-target suite: both reached their 10-second fatal traceback deadlines.

The baseline stack included:

```text
tiny_globals.py:226  get_global_graph_manager (second lock acquisition)
actions.py:1234      ActionGenerator.__init__
actions.py:1285      ActionSystem.__init__
tiny_event_handler.py:53 Event.__init__
tiny_characters.py:1065 module-level Festival example
tiny_utility_functions.py:35 character import
tiny_graph_manager.py:73 utility import
tiny_globals.py:230  get_global_graph_manager (outer import while holding lock)
```

Object-valued examples now require `build_example_world_criteria(graph_manager)`. Bare imports construct no sample world. Global initialization imports the graph outside its lock and publishes the result only after complete construction. A private construction gate serializes direct `GraphManager()` calls as well; same-thread recursive calls fail promptly, failed candidates are rolled back, and later calls can retry. Reset inspects an already-loaded module instead of importing it.

Explicit graph/action-system ownership is forwarded through characters, items, locations, events, controller/strategy/GOAP composition, strategy recovery, and generated strategy actions. Existing lazy global defaults remain available. No effect values, planning thresholds, node-key semantics, or LLM policy were changed.

## Environment and commands

Python 3.12.14 in the existing CPU environment (`/workspace/.venvs/tiny_village/bin/python`), repository root. Dummy SDL, two BLAS/OpenMP threads, and existing owned cache locations were selected through environment variables. No new runtime dependency or heavy model installation was performed.

New regression cases use a fresh interpreter for each scenario, a 20-second fatal traceback timer, and a 25-second parent deadline. Existing suites/entry points were launched by a Python subprocess orchestrator with a 110-second traceback timer and a 120-second parent deadline. Timeouts fail required checks. The runner preserves exit statuses and stdout/stderr.

The executed targets are reproduced by:

```bash
python -m unittest discover -s tests -p test_graph_initialization.py -v
python -m unittest discover -s tests -p test_graph_ownership.py -v
python -m unittest -v test_core_global_graph_manager
python -m unittest discover -s tests -p test_action_named_target_resolution.py -v
python -m unittest discover -s tests -p test_actions.py -v
python -m unittest discover -s tests -p test_event_handler_payloads.py -v
python -m unittest discover -s tests -p test_goap_error_handling.py -v
python -m unittest -v tests.test_graph_initialization tests.test_graph_ownership
python main.py --mode minimal --no-llm --headless
python main.py --mode test --no-llm --headless
python -m unittest discover -s tests -p test_tiny_utility_functions.py -v
```

Use the prepared interpreter and dummy SDL environment for these commands. Wrapping suites in `subprocess.run(..., timeout=120, capture_output=True, text=True)` reproduces the outer bounds without depending on POSIX timeout utilities. The new test modules themselves enforce their individual child deadlines on Windows and Linux.

## Final results

| Target | Executed | Passed | Failures | Errors | Skipped | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| New initialization/lifecycle suite | 13 | 13 | 0 | 0 | 0 | Passed |
| New ownership suite | 10 | 10 | 0 | 0 | 0 | Passed |
| Existing core global-manager suite | 9 | 9 | 0 | 0 | 0 | Passed |
| Existing named-target suite | 6 | 6 | 0 | 0 | 0 | Passed |
| Existing action suite | 5 | 5 | 0 | 0 | 0 | Passed |
| Existing event payload suite | 5 | 5 | 0 | 0 | 0 | Passed |
| Existing GOAP error suite from #706 | 5 | 5 | 0 | 0 | 0 | Passed |
| Combined new-suite rerun (repeat coverage) | 23 | 23 | 0 | 0 | 0 | Passed |
| Main integration test mode | 11 | 9 | 1 | 1 | 0 | Failed; #773 unchanged |
| Existing utility suite | 26 | 16 | 1 | 9 | 0 | Failed; #774 unchanged |

**53 required unique tests passed.** The combined run repeats 23 of those tests; it is not added to the unique count. Across the unique required and broader suites, 90 tests executed: 78 passed, 2 failures, 10 errors, 0 skipped. No required scenario timed out.

Additional functional probes:

- Cold global getter: exit 0 in 1.70 seconds; repeated access returns the same ready graph, and `G is world_state.graph`.
- Actual Goal fallback, with the optional character import deliberately unavailable: exit 0. Name, effects, priority/score, urgency, attributes, and explicit zero score retain #718 behavior.
- Minimal entry point: exit 0 in 1.75 seconds. The reduced demo reports **zero executed actions**; this does not establish a complete gameplay turn. The real graph-backed named-target regression separately executes an action and verifies the intended graph update.

Visual rendering, heavy NLP/LLM execution, and the full repository suite were not executed. CPU-only optional imports still emit existing embedding/cache diagnostics; this patch does not repair #720 or remove all module side effects.

## Required scenario coverage

- Eight cold import orders: globals, graph, characters, utility, GOAP, actions/locations, items/events, and strategy/controller. All leave the global reference and class singleton uninitialized.
- Mixed first access: eight barrier-synchronized getter/initializer/direct-constructor callers over ten reset rounds. Each round constructs one WorldState, returns one object/UUID, and exposes ready graph, analytics, and social-model references.
- Getter-first, constructor-first, and direct-constructor recursion: prompt descriptive RuntimeError; clean singleton references; a subsequent valid attempt succeeds.
- Constructor/import failure: original failure/cause retained; guards and locks released; retry succeeds. Waiting callers progress after one failed construction and share the successful result.
- Cold reset does not import the graph. Ready reset clears both references; reinjecting a ready real graph aligns provider/class identity. Controlled duck-typed provider injection remains compatible. Recursive lifecycle mutation is rejected, and readiness queries from the constructing thread do not deadlock.
- Domain/controller ownership: real Action/ActionSystem/Location/Event/item/GraphManager objects retain supplied dependencies. A falsy supplied action system remains supplied. Controller composition and recovery use one owner; the test intentionally creates zero characters to isolate composition from unrelated full-character behavior.
- Strategy generation: the strategy/GOAP owner and all generated actions remain with the supplied graph even when the global provider differs. All five strategy helper action constructors accept and forward an optional graph; all 14 strategy action creation sites forward the manager's owner.
- Existing six named-target assertions finish. An additional case uses actual WorldState graph UUID nodes and lightweight demo characters, verifying one intended effect, no bystander change, duplicate-target de-duplication, and the authoritative graph update. Canonical character/node-key reconciliation remains separate work.

## Test-first and independent review evidence

- Import baseline: seven failing subcases out of eight; after removing object-valued globals, all eight pass.
- Lifecycle baseline: ten failure reports across the initial ten test methods; additional direct-recursion and waiting-caller probes fail. After repair, the expanded 13-test suite passes.
- Ownership baseline: eight failure reports across six methods. After forwarding, six methods pass. Door's definition boundary is disclosed below; reverting only its forwarding proves the configured ownership regression fails, then restoring it passes.
- Controller/strategy baseline: two failures across nine methods. Composition/recovery forwarding makes all nine pass.
- A fresh-context reviewer independently ran all 13 lifecycle tests successfully and reported one Important defect: strategy-generated actions ignored the injected owner. A new regression first failed with `Generated action resolved a different global world`; forwarding then made the 10-test ownership suite pass, followed by the complete 53-test matrix and 23-test combined rerun. No Critical or Minor findings were reported. The final fix was verified by tests rather than a second reviewer dispatch.

The reviewer did not judge full Character readiness, default Door feature behavior, #773/#774 repairs, concurrent live reset/replacement, or human approvals. Those exclusions do not count as approvals or passing checks.

## Remaining defects and execution decisions

1. Workflow helper/reference files were unavailable through the skill resource reader. Equivalent task bases, commands, results, and rulings were maintained in a local execution ledger. This substitutes bookkeeping only; runtime and validation requirements were retained.
2. The approved scoped matrix governs verification rather than the workflow skill's generic full-suite requirement. Heavy/visual/full-suite behavior remains unverified and cannot be claimed ready.
3. Default Door construction still raises `KeyError: 'Open'` because the production effect map lacks that definition. Ownership tests supply explicit effect-definition data and run the real Door/action/system/graph objects. This proves configured ownership, not default Door availability. [#775](https://github.com/dhar174/tiny_village/issues/775) tracks the repair under #195, with native blockers #719 and human gate #748.
4. Character ownership tests exercise the actual constructor through graph, action-system, twelve real actions and GOAP setup, then deliberately stop at the optional PromptBuilder boundary. A separate full-constructor diagnostic proceeds further and fails in unchanged `FlatMemoryAccess` at `spacy.tokens.token.Token` because spaCy is None. These tests do not establish full Character readiness. [#776](https://github.com/dhar174/tiny_village/issues/776) tracks that defect under #654, with native blockers #719, #725 and human gate #722.
5. The review's declined scenarios remain reported rather than silently treated as working. #773 retains the `Mock` inventory/state integration failures; #774 retains nine real/fallback Goal constructor mismatches and one falsy fixture ID failure. Reset/replacement remains a quiescent test/setup operation, not a concurrent live-world feature. Human merge/phase approval remains pending.

## Human review and next work

Review graph lock order/publication, fault recovery, explicit ownership including generated actions, and the fixture/integration boundaries above. #719 is ready for code review on this branch; main remains unchanged until merge. After approval and merge, #720 is next. The planned #721 startup engineering review and #722 phase decision still require their own human evidence and decisions.

## Post-merge verification — 2026-10-14

PR #777 merged as `0ba9061a90db776d49a5959b6eb35f711754530c` and closed #719.
The merged head also includes the direct graph-ownership fallback regression
added after the original 53-test branch validation. The exact merged code was
verified before beginning #720:

| Target | Executed | Passed | Failures/errors/skips |
| --- | ---: | ---: | --- |
| Initialization/lifecycle | 13 | 13 | 0/0/0 |
| Ownership, including direct owner fallback | 11 | 11 | 0/0/0 |
| Core global manager | 9 | 9 | 0/0/0 |
| Named-target | 6 | 6 | 0/0/0 |
| Actions | 5 | 5 | 0/0/0 |
| Event payloads | 5 | 5 | 0/0/0 |
| GOAP errors | 5 | 5 | 0/0/0 |

**54/54 tests passed**, with no skips or timeouts. The same commands listed
above were run in bounded subprocesses. Cold getter exited 0 in 2.40 seconds;
the actual Goal fallback and minimal mode also exited 0. Minimal mode continues
to report zero executed actions. No broader integration, full-model or visual
readiness is inferred.

TMPDIR was isolated under an owned validation directory so the pre-#720 code
could not chmod an existing shared cache. Model offline flags, dummy SDL and
the prepared CPU interpreter were used. Raw logs and the command/result JSON
are retained in `/workspace/scratch/issue-719-post-merge/`. #720 implementation
started only after this verification passed; #721/#722 remain human steps.
