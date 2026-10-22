# #736 assertion map

Execution approved under [D029](../../decisions.md#d029--review-736-functional-test-pr-dispositions-step-by-step). This is the initial source map, committed before ports. Test names are intended destinations until execution evidence is added.

## Provenance

Main remains `11b6e93595a70ce9f24cfb239a31ac873847ff56`. All nine candidate heads match the walkthrough; all are OPEN. GitHub currently reports #408 mergeable and the other eight conflicting. This is metadata, not validation or a reason to merge stale branches.

[Refreshed metadata](execution/refreshed-prs.json) retains titles, authors, heads, file lists, reviews and recent discussion. [Source index](execution/source-index.json) inventories 24 exact changed Python file snapshots and 190 test methods, including inherited methods. Assertions outside a candidate delta are identified below; their presence is not credited as newly added coverage. No old test file is restored to discovery.

## Action, State and utility

| Source at pinned head | Assertion / disposition and maintained destination |
| --- | --- |
| #420 verify_genuine_failures.py:65–70 | State dict read/get/default99/write energy8: unique port to tests/test_action_contracts.py. Standalone current action resolution main() covers reads only. |
| #420 same:102–114 | Empty preconditions and real apply_effects(State(energy5))→7: unique direct API port. Existing test_actions executes effects against fixture objects. |
| #420 same:116–146 | Real Condition/Action threshold energy10 passes12, fails5: port with complete real Character; no StatefulTarget fake. |
| #420 same:72–77,148–153 and other three methods | Invented missing names, arithmetic, file/source existence and absent paths: exclude non-product demonstrations. |
| #417 test_fix_validation.py:262–327 | Four exact edge states; real Action and lightweight production utility Goal; finite action/plan score, retained magnitude bound, input isolation: tests/test_utility_contracts.py. Remove optional hasattr and swallowed missing-method paths. |
| #417 same:526–552 (inherited) | Real Action + utility Goal golden29.1: preserve in new utility contracts. Existing real-Goal tests assert positivity; mock-only golden is insufficient equivalent. |
| #417 same:556–610 (inherited) | Custom test-action golden and mock parity: redundant with real-Action golden; exclude copied implementation. |
| #417 verify_genuine_failures.py:255–315 | Hungry>satiated, aligned action>misaligned work, repeated simulated eating<2×initial utility: preserve real-Action relational checks and input isolation. Existing real-class plan suite only checks positivity. |
| #417 test_issue_425_fix.py | Four numeric requirements remain in maintained tests/test_issue_425_fix.py; exclude new copied UtilityTestAction precondition/effect logic. |
| #417/#408 demonstrations | Local algorithms, arithmetic, JSON/parsing, invented absent imports and generic filesystem examples: exclude. No unique #408 Action/utility contract found. |
| #417/#408 Character import/class surface | Subsumed by complete Character construction in tests/test_lightweight_runtime.py and new target/behavior tests. |
| #417/#408 CreateBuilding construction/map retention | Compare with root TestCompletedImplementations.test_building_coordinate_selection (real construction and deterministic non-overlap). Preserve root building coverage; map any genuinely additional supported API assertion before closure. |

## Memory and runner

| Source | Assertion / disposition and maintained destination |
| --- | --- |
| #406 test_tiny_memories.py TestImport delta | Three new MemoryManager/SpecificMemory/GeneralMemory class-export checks: direct imports/class identity in tests/test_memory_runtime.py without model initialization. These are surface assertions only. All inherited memory tests remain intact. |
| #406 test_fix_validation.py examples | Nine Python/local-repository/service examples do not exercise TinyVillage memory; exclude. Session log is not a test artifact to port. |
| #417/#408 memory demonstration | Obsolete Memory(content=..., memory_type=..., timestamp=None) call is incompatible with current approved Memory(description, creation_time, importance_score). Preserve its record/read intent through existing TestBasicMemory.test_real_records_retain_fields_and_evict_oldest and complete Character independent-memory test; do not invent legacy aliases or falsely label semantic coverage. Verify exact source before closure. |
| #404 format_traceback helper | Retired tests/test_completed_implementations.py wrapper has no current Python caller. Supersede only with maintained full-output failure evidence; no helper restoration. Linked runtime issues remain separate. |
| #397 run_unittest_with_proper_counting | Retained diagnostic: testsRun1/failing-subtests2/passed−1. Current tests/test_ci_checks.py covers real method/subtest counts, failure/error/skip/import and incomplete reports. Run maintained coverage and retain exact outcome; no negative-pass helper port. |

## Happiness

| Source | Assertion / disposition and maintained destination |
| --- | --- |
| #391 two added TestHappinessCalculation methods | Three-of-four keyword threshold/warnings do not prove behavior. Replace with paired motive/social/romantic/family outcomes plus legitimate absent relationships in tests/test_character_happiness_contracts.py. |
| #395 TestHappinessCalculation:98–221 | Inherited, not #395 delta; map same four required features to Task4; do not attribute as new work in #395. |
| #417/#408 keyword/TODO scans and current root happiness scan | Same behavior map. Keep current root scan until all required behavior passes or a demonstrated blocker receives a separate reviewed disposition. No weakened feature requirement. |
| Current PersonalMotives vs calculate_happiness | Source risk: contribution calls absent food/safety/social getters under bare except; execute strict real-object motive pair and retain any failure as blocker, not a fake getter or runtime fix. |

## Planner and graph

| Source | Assertion / disposition and maintained destination |
| --- | --- |
| #395 test_goap_implementations delta | Preserve unique Plan queue/evaluate/replan/alternative and planner utility/feasibility requirements after comparison with current suites. None/exception cannot prove reachable success. Record any nonfunctional required path as failing reproduction; no simulated repair. |
| #395 find_alternative_action:353–367 | Explicit exclusion after comparison: accepts None unconditionally or checks only a dict/action/name shape using MockAction. It specifies no available alternative, applicable precondition, effect or achieved goal, so a None-only implementation passes. Do not port this as evidence of real alternative selection. Real alternative/replanning behavior remains an unresolved requirement under #741; no execution claim is made for this excluded assertion. |
| #393 unified/integration tests, plus approved strengthening | Already satisfied→empty plan; reachable→nonempty plan plus actual final predicate; unreachable→documented no-plan; blocked preconditions→unchanged state. Intended tests/test_goap_graph_contracts.py; use production Goal and complete Character for full-domain claims. |
| #393 dynamic state/action retrieval | Actual GraphManager/MultiDiGraph, Character registration, authoritative state/action provenance. Fallback character-only state or Rest/Idle must not count as graph retrieval. Intended strict graph tests or explicitly blocked reproductions. |
| #393 cost and utility integration | Compare equally effective actions reaching initially unsatisfied energy target; expensive input first, cheaper selected and final energy target achieved. Preserve current four-argument cost and three-argument heuristic context. |
| #393 MockGraphManager / MockCharacter / simulated plan_actions | Exclude copied production logic and dict substitute for real MultiDiGraph. |
| #393 tiny_goap_system.py signature reductions | Exclude incompatible production edits. Retained signature diagnostic proves mismatch against current main; not full old-branch execution. |
| Current full Goal | Source risk: check_completion accepts no state; constructor expects lists while completion calls dictionary values directly. Reproduce current public APIs; no monkeypatched completion method or local Goal adapter masquerading as domain integration. |
| Current real graph lookup | Source risk: object-key registration versus name-key lookup and type casing. Preserve #799 and any additional distinct defects; no name aliases or invented return values. |

## Execution and closure rules

Each row gains exact executed selectors/commands and results in the validation report before it counts as verified. New regression tests are allowed to pass existing correct behavior; no production change is needed merely to manufacture RED/GREEN. Required broken behavior remains a failed reproduction under a named blocker, not skip/xfail or fabricated green evidence. All nine dispositions are selected; none is applied. #736 remains open until required coverage gaps receive an explicit disposition. Candidate closures follow replacement merge and fresh-head checks; #737/#738 remain separate.

## Execution outcome — 2026-10-22 America/New_York

The [validation report](../../testing/issue-736-functional-test-consolidation.md) supplies exact commands, result counts, limitations and blocker classification for each domain above. It is the evidence reference for this map; no candidate is closed.

| Candidate / domain | Executed destination and disposition status |
| --- | --- |
| #420 | tests.test_action_contracts (3 methods) core/dev pass; all unique State/Action/Condition assertions accounted for. |
| #417 | tests.test_utility_contracts (5 methods) core/dev pass; existing Action/real-class utility/#425 cases pass. Existing tiny-utility 10 failures reproduce on exact main and remain #774; happiness/memory/building requirements map separately below. |
| #408 | No extra Action/utility behavior found. Character construction is exercised by the real fixtures; memory record intent is covered by 14-method memory/minimal selection; building map identity now explicit in existing root test (4/4 visual). Happiness remains blocked as recorded. |
| #406 | New TestLazyMemory.test_public_memory_classes_are_available_without_initializing_services, with full memory/minimal selection14/14 core/dev. Semantic/full-memory readiness is unverified. |
| #404 | Controlled nested-error worker correctly FAILS; full traceback retained and outer evidence check exits0. Maintained runner profile tests pass. Obsolete helper is not restored. |
| #397 | Maintained CI-runner method/subtest/error counting executes in dev profile20/20; original expected failing-subtest diagnostic retained. |
| #391 | Seven real happiness methods pass core/dev; strict added motive-satisfaction requirement fails at unchanged numeric result. New #803 retains required gap; root source check is not removed. |
| #395 | Five supported Plan/planner primitives pass core/dev. Full Goal evaluation/feasibility gaps are #802; replan None expectation is tracked under #741. evaluate_utility(Plan) is excluded candidate misuse, not a production defect. The conditional None-or-named find_alternative_action shape check is explicitly excluded as described above; real alternatives remain #741 and unverified. |
| #393 | Supported contextual cost/heuristic and real action selection pass. Full Goal satisfied/reachable/cost outcomes remain #802; real graph state/action/world/identity failures remain #799. Negative no-plan observations meet their expectation but fail positive completion controls. |

Core integrated profile:72 tests+3 CLI probes pass; dev profile20 tests pass. These green profiles do not include the explicitly failed text reproductions and do not satisfy those requirements. Every required failure remains named; closure needs the separate reviewed acceptance described in the report.

## Accepted gap disposition — D030

On 2026-10-22 dhar174 selected scoped reconciliation with explicit runtime
carry-forward. See D030 in docs/decisions.md and the report's accepted
carry-forward section. Required failed outcomes remain #802/#803/#799/#741;
existing utility test debt remains #774. Replacement merge and post-merge
verification still precede candidate closures or scoped #736 completion. No
failed behavior is reclassified as passed.
