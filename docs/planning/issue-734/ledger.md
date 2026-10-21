# Backlog ledger — proposed dispositions

**Review pending. No issue metadata or state changes have been applied.**

261 original issues plus 46 later open issues. The JSON/CSV retain full titles, original checklists, owners, source anchors, discussions, native dependencies and exact update proposals. Proposed execution prerequisites are distinct from native edges; handoffs coordinate work and are not extra blockers. Existing human approvals and the single already-closed original issue #654 remain intact.

## Original 261

### Actions and GOAP contracts

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#166](https://github.com/dhar174/tiny_village/issues/166) | implement | 4 | dhar174 |
| [#167](https://github.com/dhar174/tiny_village/issues/167) | implement | 4 | dhar174 |
| [#182](https://github.com/dhar174/tiny_village/issues/182) | implement | 4 | dhar174 |
| [#183](https://github.com/dhar174/tiny_village/issues/183) | implement | 6 | dhar174 |
| [#184](https://github.com/dhar174/tiny_village/issues/184) | implement | 4 | dhar174 |
| [#185](https://github.com/dhar174/tiny_village/issues/185) | implement | 4 | dhar174 |
| [#186](https://github.com/dhar174/tiny_village/issues/186) | implement | 4 | dhar174 |
| [#187](https://github.com/dhar174/tiny_village/issues/187) | implement | 4 | dhar174 |
| [#188](https://github.com/dhar174/tiny_village/issues/188) | implement | 6 | dhar174 |
| [#218](https://github.com/dhar174/tiny_village/issues/218) | implement | 4 | dhar174 |
| [#220](https://github.com/dhar174/tiny_village/issues/220) | implement | 4 | dhar174 |
| [#354](https://github.com/dhar174/tiny_village/issues/354) | implement | 4 | dhar174 |
| [#355](https://github.com/dhar174/tiny_village/issues/355) | verify | 4 | dhar174 |
| [#356](https://github.com/dhar174/tiny_village/issues/356) | implement | 4 | dhar174 |
| [#358](https://github.com/dhar174/tiny_village/issues/358) | verify | 4 | dhar174 |
| [#602](https://github.com/dhar174/tiny_village/issues/602) | implement | 4 | dhar174 |

#### #166 — GOAP Integration

**Acceptance:** Produce valid multi-step GOAP plans from real goals/state/actions and show effects achieve the goal; single-action utility ranking is not sufficient.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #165, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #218. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:584, tiny_strategy_manager.py:796; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #166.

#### #167 — Goal Management

**Acceptance:** Construct supported production Goals from real state/needs, pass them to the planner and verify completion/failure against actual effects and scales.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #165, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #218. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:584, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #167.

#### #182 — actions.py & Action System

**Acceptance:** Complete the original action catalog, preconditions/effects, real execution/graph ownership, resolver alignment, cost and skill contracts; preserve all child evidence rather than closing the umbrella from one action.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:36, tiny_goap_system.py:63, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #182.

#### #183 — Expansion of Actions

**Acceptance:** Provide concrete job/item/building/social actions with valid targets/preconditions/effects; keep catalog expansion distinct from fixing Action execution.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #196. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:302, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #183.

#### #184 — Comprehensive Preconditions/Effects

**Acceptance:** Check passing and failing preconditions and actual effect ranges/units for each supported catalog action; assert authoritative graph updates and failure integrity.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #184.

#### #185 — Action.execute() Implementation

**Acceptance:** Execute production Action objects and confirm intended effects and graph updates once, including invalid targets/preconditions; method existence is not completion.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:302, tiny_goap_system.py:63, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #185.

#### #186 — Alignment with ActionResolver

**Acceptance:** Resolve equivalent dictionary/template/Action inputs to correct production behavior, preserving graph ownership, target and initiator.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:302, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #186.

#### #187 — Cost Balancing

**Acceptance:** Reconcile costs, need scales and utility/planning units; exercise boundary cases and verify meaningful plans rather than tune arbitrary constants to a fixture.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #187.

#### #188 — Skill Integration

**Acceptance:** Use real Skill containers for action preconditions/progression and demonstrate supported skills improve only through valid actions.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #196. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1248, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #188.

#### #218 — StrategyManager and GOAPPlanner Interaction

**Acceptance:** Connect StrategyManager to real multi-step GOAP with supported Goal/Action contracts; execute plans across ticks and replan on failure.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #217, canonical topic #182; native prerequisites #740; proposed execution prerequisites #738; handoff #218. blocked by native #740.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:584, tiny_strategy_manager.py:95; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #218.

#### #220 — Action.execute()

**Acceptance:** Apply effects and graph updates through the supported action contract with consistent units/identities, including real named targets and failed preconditions.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the actions and goap contracts acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #217, canonical topic #182; native prerequisites #738; proposed execution prerequisites none; handoff #220. blocked by native #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:302, tiny_goap_system.py:63, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #220.

#### #354 — The magic number 1000 for max iterations should be defined as a class constant or configurable parameter to make it easier to tune and more maintainable.

**Acceptance:** Define and exercise configurable/bounded planner search limits, including exhaustion outcome; current max_iterations=1000 remains local policy rather than a class/config contract.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. The current planner still uses local max_iterations=1000. Keep this low-priority policy improvement separate from the proven tuple compatibility issue.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #218. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_goap_system.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #354.

#### #355 — This change from 3-tuple to 4-tuple format will break existing code that expects the original (priority, action, dependencies) format when popping from the queue. All existing heappop operations throughout the codebase need to be updated to handle the new tuple structure.

**Acceptance:** Verify all action-queue producers and consumers accept four fields; equal-priority insertion and replanning preserve action/dependency data, with explicit limits on tie determinism.

**Disposition/evidence:** verify; `passed_scoped_probe`. Queue probe passes and both actual Plan.execute heap-pop consumers unpack four fields. Replan retains dependencies but uses id(action) for ties; no cross-process deterministic-order claim.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #218. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_goap_system.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #355.

#### #356 — Creating a mock class that always returns True in execute() violates the coding guideline against over-mocking. This mock will make tests pass even when the actual functionality doesn't work correctly, which goes against the principle of designing tests that fail when there are real errors.

**Acceptance:** Replace success-shaped/random MockAlternativeAction production fallback with a real valid alternative or explicit failure; assert actual precondition/effect behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Current production fallback still defines MockAlternativeAction, always accepts preconditions and returns a random success without applying real effects. This is not resolved by queue compatibility.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #741. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_goap_system.py:63; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #356.

#### #358 — Creating dynamic classes using type() makes the code harder to understand and debug. Consider defining a proper ActionWrapper class or ensuring graph manager returns proper Action objects instead of creating anonymous classes.

**Acceptance:** Verify dictionary actions use the production ActionWrapper interface with meaningful effects/preconditions; do not count anonymous type construction as a supported action contract.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current actions and goap contracts code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #218. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_goap_system.py:50; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #358.

#### #602 — Support named effect targets such as target_character_in_location

**Acceptance:** Pass all six existing named-target cases AND real-world co-location, requested identity, absence, deduplication and graph updates after required Phase 4 dependencies; retain current real-object failure and keep open.

**Disposition/evidence:** implement; `failed_real_object_probe`. Six legacy regressions pass using DemoCharacter/RecordingGraphManager fixtures. The real-object probe fails: Action passes UUID to a graph keyed by Character objects; mutating current_visitors then breaks Location hashing (dict_or_obj). Phase 4 prerequisites remain open; never close from the fixture-only pass.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #182, canonical topic #182; native prerequisites #741; proposed execution prerequisites #738; handoff #602. blocked by native #741.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #602.

### Architecture and boundaries

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#14](https://github.com/dhar174/tiny_village/issues/14) | defer | 9 | dhar174 |
| [#49](https://github.com/dhar174/tiny_village/issues/49) | defer | 9 | dhar174 |
| [#534](https://github.com/dhar174/tiny_village/issues/534) | defer | 9 | dhar174 |
| [#535](https://github.com/dhar174/tiny_village/issues/535) | defer | 9 | dhar174 |

#### #14 — Refactor: Decouple tightly coupled modules and improve architecture for maintainability

**Acceptance:** Inventory coupling at graph, rendering and AI boundaries; implement scoped interfaces/configuration/refactors with preserved behavior and updated architecture evidence, rather than a blanket rewrite.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader architecture and boundaries requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect goap_evaluator.py:1, tiny_globals.py:26, world_state.py:449; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #14.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #49 — tiny_graph_manager.py TODOs

**Acceptance:** Demonstrate goal-completion evaluation across action sequences and graph-derived difficulty; compare documented future enhancements with current graph/GOAP implementations.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader architecture and boundaries requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect goap_evaluator.py:1, tiny_globals.py:1, world_state.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #49.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #534 — Another instance of circular dependency where GoapEvaluator delegates back to GraphManager. This pattern should be avoided as it defeats the purpose of extracting GOAP logic. The graph operations should be implemented directly in GoapEvaluator or extracted to a shared utility.

**Acceptance:** Review the viability-cost delegation against actual call cycles and graph ownership; decouple only with preserved cached graph semantics and behavior evidence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader architecture and boundaries requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect goap_evaluator.py:102; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #534.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #535 — This delegation back to GraphManager creates a circular dependency pattern that undermines the refactoring goals. The GoapEvaluator should not depend on GraphManager methods. Consider extracting the graph analysis logic into a separate utility class or implementing it directly in GoapEvaluator.

**Acceptance:** Review the goal-difficulty delegation separately from viability costs; preserve authoritative graph analytics while removing any demonstrated recursive dependency.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader architecture and boundaries requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect goap_evaluator.py:102; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #535.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Authoritative world and graph identity

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#222](https://github.com/dhar174/tiny_village/issues/222) | implement | 4 | dhar174 |
| [#245](https://github.com/dhar174/tiny_village/issues/245) | verify | 4 | dhar174 |
| [#609](https://github.com/dhar174/tiny_village/issues/609) | implement | 4 | dhar174 |

#### #222 — GraphManager as Single Source of Truth

**Acceptance:** Trace shared graph ownership and actual entity/relationship reads/writes across systems; fix identity/adapter defects without creating a second authoritative store.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the authoritative world and graph identity acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #217, canonical topic #222; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:737, tiny_globals.py:1, world_state.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #222.

#### #245 — Single global instance of GraphManager

**Acceptance:** Verify shared manager identity across supported simulation callers and composition boundaries; preserve #719 startup ownership evidence without extending it to unexecuted call paths.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Merged #719 proves scoped initialization/ownership. Broader caller coverage remains a verification item. The visual-loop None.locations error is specifically MapController.location_manager being reset to None, not proof of a new global GraphManager initialization failure.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #222; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:737, tiny_globals.py:1, world_state.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #245.

#### #609 — Move relationship state toward GraphManager as source of truth

**Acceptance:** Read/write actual keyed graph relationship attributes through controller and GraphManager, including decay; no swallowed API error, default-filled result or duplicate authority.

**Disposition/evidence:** implement; `failed_real_object_probe`. Real MultiDiGraph relationship reader logs unhashable dict and returns empty; keyed edge attributes do not decay. Static code passes {} as get_edge_data key and treats multiedge adjacency as flat attributes. Controller delegation/status wording does not establish correctness.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #212, canonical topic #222; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:737, tiny_globals.py:1, world_state.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #609.

### Behavioral test quality

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#315](https://github.com/dhar174/tiny_village/issues/315) | defer | 9 | dhar174 |
| [#319](https://github.com/dhar174/tiny_village/issues/319) | defer | 9 | dhar174 |
| [#333](https://github.com/dhar174/tiny_village/issues/333) | defer | 9 | dhar174 |
| [#350](https://github.com/dhar174/tiny_village/issues/350) | verify | 3 | dhar174 |
| [#351](https://github.com/dhar174/tiny_village/issues/351) | duplicate | 3 | dhar174 |
| [#352](https://github.com/dhar174/tiny_village/issues/352) | verify | 3 | dhar174 |
| [#353](https://github.com/dhar174/tiny_village/issues/353) | verify | 3 | dhar174 |
| [#357](https://github.com/dhar174/tiny_village/issues/357) | verify | 3 | dhar174 |
| [#359](https://github.com/dhar174/tiny_village/issues/359) | verify | 3 | dhar174 |
| [#478](https://github.com/dhar174/tiny_village/issues/478) | verify | 8 | dhar174 |
| [#479](https://github.com/dhar174/tiny_village/issues/479) | verify | 8 | dhar174 |
| [#481](https://github.com/dhar174/tiny_village/issues/481) | verify | 8 | dhar174 |
| [#493](https://github.com/dhar174/tiny_village/issues/493) | defer | 9 | dhar174 |
| [#494](https://github.com/dhar174/tiny_village/issues/494) | defer | 9 | dhar174 |
| [#495](https://github.com/dhar174/tiny_village/issues/495) | defer | 9 | dhar174 |
| [#496](https://github.com/dhar174/tiny_village/issues/496) | defer | 9 | dhar174 |
| [#501](https://github.com/dhar174/tiny_village/issues/501) | verify | 3 | dhar174 |
| [#502](https://github.com/dhar174/tiny_village/issues/502) | verify | 3 | dhar174 |
| [#503](https://github.com/dhar174/tiny_village/issues/503) | verify | 3 | dhar174 |
| [#504](https://github.com/dhar174/tiny_village/issues/504) | verify | 3 | dhar174 |
| [#505](https://github.com/dhar174/tiny_village/issues/505) | verify | 8 | dhar174 |
| [#506](https://github.com/dhar174/tiny_village/issues/506) | verify | 8 | dhar174 |
| [#514](https://github.com/dhar174/tiny_village/issues/514) | defer | 9 | dhar174 |
| [#515](https://github.com/dhar174/tiny_village/issues/515) | defer | 9 | dhar174 |
| [#516](https://github.com/dhar174/tiny_village/issues/516) | defer | 9 | dhar174 |
| [#517](https://github.com/dhar174/tiny_village/issues/517) | defer | 9 | dhar174 |
| [#519](https://github.com/dhar174/tiny_village/issues/519) | defer | 9 | dhar174 |
| [#520](https://github.com/dhar174/tiny_village/issues/520) | defer | 9 | dhar174 |
| [#522](https://github.com/dhar174/tiny_village/issues/522) | implement | 4 | dhar174 |
| [#539](https://github.com/dhar174/tiny_village/issues/539) | defer | 9 | dhar174 |
| [#540](https://github.com/dhar174/tiny_village/issues/540) | defer | 9 | dhar174 |
| [#541](https://github.com/dhar174/tiny_village/issues/541) | defer | 9 | dhar174 |
| [#682](https://github.com/dhar174/tiny_village/issues/682) | verify | 8 | dhar174 |
| [#683](https://github.com/dhar174/tiny_village/issues/683) | verify | 3 | dhar174 |

#### #315 — The torch.rand stub always returns 0, which could mask issues where randomness is expected. This violates the principle of writing tests that fail when functions don't work as expected.

**Acceptance:** Exercise prompt descriptor randomness with actual supported dependencies or a clearly bounded RNG fixture; no torch.rand stub fixed at zero may stand in for production behavior.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_enhanced_decision_prompt.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #315.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #319 — Setting Character to the base object class creates an overly simplified mock that doesn't represent the real Character interface. This could mask integration issues where the real Character class has different behavior or attributes.

**Acceptance:** Use a Character-compatible fixture or real basic-memory Character and demonstrate that a wrong interface fails; do not replace the production class with object in sys.modules.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_decision_prompt_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #319.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #333 — Using a hardcoded increment value in the mock defeats the purpose of making the test more robust. The test should use the actual respond_to_talk implementation or be designed to work regardless of the specific increment value. This mock implementation may not accurately reflect the real method's behavior.

**Acceptance:** Call the real respond_to_talk method and verify its documented effect for more than one context, instead of copying a fixed increment into the test double.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_actions.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #333.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #350 — This test calls the real GOAP planner but doesn't validate that the returned plan would actually achieve the goal. The test should verify that if a plan is returned, it contains actions that logically lead to the goal state, not just that some value is returned.

**Acceptance:** Replay each returned GOAP plan's actual effects from the initial state and assert the requested goal is reached; reconcile the removed historical file with current tests under #736.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #350.

#### #351 — This test calls the real GOAP planner but doesn't validate that the returned plan would actually achieve the goal. The test should verify that if a plan is returned, it contains actions that logically lead to the goal state, not just that some value is returned.

**Acceptance:** Identical original body and identical source discussion_r2217197555 as #350; preserve the canonical goal-achievement acceptance and historical link before approved duplicate closure.

**Disposition/evidence:** duplicate; `source_reviewed_not_executed`. Exact duplicate: identical original issue body and same original review-comment URL as #350. No unique acceptance was found in either issue discussion.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #736. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #351.

#### #352 — The test calls the real GOAP planning system but only checks if a plan exists, not whether the plan would actually work. Consider adding validation that any returned plan contains actions that can achieve the goal from the current state.

**Acceptance:** Validate that the planned actions actually achieve the goal in the debug scenario; a nonempty plan or printed action list is insufficient.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_simple_debug.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #352.

#### #353 — While this test uses more realistic scenarios, it still doesn't validate that the returned plans would actually achieve the stated goals. The test should verify that if a plan contains a 'Sleep' action for an energy goal, that action actually increases energy in the expected way.

**Acceptance:** Execute the energy-restoration sequence and assert the actual Sleep action raises energy to the goal using supported units; retain this scenario distinct from generic #350.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #353.

#### #357 — The test creates a planner with None graph_manager but then tests functionality that may depend on having actual components. Consider using mock objects that simulate the expected behavior rather than None, or ensure the test validates that the function works correctly when graph_manager is None.

**Acceptance:** Reconcile the removed historical planner test with current real-graph coverage and explicit unavailable-graph behavior; a no-error None-manager run is insufficient.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #357.

#### #359 — The test initializes planner without graph manager but doesn't validate that the resulting behavior is correct when components are missing. The test should verify that fallback mechanisms work as intended, not just that methods complete without errors.

**Acceptance:** Verify documented fallback results and real state changes with absent graph services, separately from real-graph integration; retain the original removed-file context.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #359.

#### #478 — This test manipulates the controller's ui_panels attribute directly rather than testing the actual fallback scenario. Consider testing with a controller that naturally has no ui_panels, or ensure this manipulation accurately reflects the real fallback condition.

**Acceptance:** Induce an actual supported UI fallback condition on a real controller/Surface and assert a distinguishable rendering result without rewriting ui_panels to force success.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #478.

#### #479 — This assertion will always pass and doesn't verify that the modular UI system is actually functioning correctly. The test should verify specific expected behaviors or state changes after rendering, not just the absence of exceptions.

**Acceptance:** Assert meaningful pixels/content/state from real modular panels rather than assertTrue(True) or only absence of exceptions.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #479.

#### #481 — This assertion will always pass and doesn't verify that the legacy fallback is actually being used. The test should verify specific behaviors that distinguish legacy rendering from modular rendering.

**Acceptance:** Demonstrate that real fallback output differs from modular output and remains usable under the supported failure condition.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #481.

#### #493 — Mocking preconditions_met to always return True prevents testing of actual precondition logic. Consider testing both success and failure scenarios with real precondition evaluation.

**Acceptance:** Exercise actual social preconditions in passing and failing states, proving a failed precondition prevents the action and its effects.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_interactions.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #493.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #494 — Integration tests should use real character objects rather than mock implementations to properly validate the end-to-end functionality. The current mock duplicates production logic, preventing detection of actual bugs.

**Acceptance:** Use real Characters, locations and graph in the social integration path and assert actual effects; copied production logic in MockCharacter is not evidence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #494.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #495 — This mock method duplicates the exact logic from the real implementation, which means tests will pass even if the real respond_to_talk method is broken. Mock methods should be minimal and focus on verifying calls rather than reimplementing functionality.

**Acceptance:** Call the real respond_to_talk implementation and assert documented outcomes, including cases that distinguish it from the former copied mock.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_interactions.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #495.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #496 — The extensive use of mock classes (MockPersonalityTraits, MockCharacter) may not properly test the actual social interaction functionality. Consider using real character classes or reducing mocking to ensure tests fail when the actual functions don't work as expected.

**Acceptance:** Exercise real personality/Character interactions and outcomes with representative traits; keep doubles only at explicitly external boundaries.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_interactions.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #496.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #501 — Passing None for GraphManager in comprehensive tests undermines the test's ability to detect real integration issues. This could cause tests to pass even when the GraphManager integration is broken.

**Acceptance:** Locate replacement coverage for the removed comprehensive GOAP tests and prove actual graph-derived plan behavior, preserving historical missing-dependency failures.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #501.

#### #502 — Using None for graph_manager may not properly test GOAPPlanner functionality. Consider creating a mock GraphManager to verify the planner behaves correctly with dependencies, rather than bypassing them entirely.

**Acceptance:** Verify graph-present and explicitly absent-graph planner results with meaningful goals/effects, not only no-error completion of the focused script.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_focused.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #502.

#### #503 — Mock actions that always return True/False regardless of actual conditions will not properly test plan execution logic. The test should verify that execute() is called with correct parameters and fails appropriately when real conditions aren't met.

**Acceptance:** Execute real supported Actions with the intended target/initiator and demonstrate both valid effects and real precondition failure; reconcile the removed historical file.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:112, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #503.

#### #504 — This test design validates the failure handling logic correctly, but the success/failure simulation should be based on realistic scenarios rather than predetermined outcomes to ensure the failure handling actually works.

**Acceptance:** Reproduce retry/replan failure handling from actual changing conditions and prove resulting world state, instead of predetermined True/False action outcomes.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #504.

#### #505 — The test is using extensive mocking with MockPygame which may not accurately test the real functionality. Consider testing with actual pygame objects or at minimum ensure the mocks behave identically to real pygame components, particularly for collision detection and rendering.

**Acceptance:** Verify real Pygame collision/hit testing/selection/render behavior; retain external-boundary doubles only when their scope is explicit.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_map_interactivity.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #505.

#### #506 — The method `is_building` is duplicated at lines 363-368 and 586-591. This duplicate code should be removed to maintain code clarity.

**Acceptance:** Confirm one authoritative is_building implementation handles production Building/dictionary cases and produces correct hit tests; retain historical duplicate-definition context.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Only one is_building definition remains in the current module. This removes the static duplicate-definition symptom but does not establish current Building/dictionary hit-test correctness.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:1318; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #506.

#### #514 — The MockGraphManager class creates extensive mock infrastructure that may not accurately test the real SocialModel functionality. Consider using the actual GraphManager with minimal test data instead of creating complex mock classes that could hide integration issues.

**Acceptance:** Run SocialModel through real GraphManager state and assert actual relationships; no substituted tiny_characters module or graph-only mock may establish integration.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_model_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #514.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #515 — The MockWorldState class provides minimal functionality that may not reveal edge cases or integration issues. The empty neighbors() method and basic edge checking could lead to tests passing when the real implementation might fail.

**Acceptance:** Build an actual WorldState graph and verify SocialModel edge/neighborhood behavior for nonempty and absent relations, rather than an empty canned graph.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_model.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #515.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #516 — The test only validates prompt length and string content but doesn't test if the PromptBuilder actually generates valid, functional prompts that would work with an LLM. This could pass even if the prompt generation logic is broken, violating the principle of designing tests that fail when functionality is incorrect.

**Acceptance:** Assert semantic prompt/context/action/schema correctness for representative real contexts; string length, type and a character-name substring are insufficient.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_llm_interface_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #516.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #517 — This test only verifies the return type and structure but doesn't validate that TinyBrainIO actually processes inputs correctly. The test could pass even if the brain_io fails to process prompts properly, which goes against the guideline to design tests that fail when the tested function doesn't work as expected.

**Acceptance:** Validate actual provider input/output behavior and explicit unavailable-model errors under the supported profile; output list type alone does not prove inference.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_llm_interface_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #517.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #519 — Mocking the decide_action_with_llm method defeats the purpose of testing the LLM integration pipeline. This test will pass even if the actual LLM integration is broken. Consider testing with real LLM components or implementing a more realistic test double.

**Acceptance:** Exercise the actual decide_action_with_llm pipeline with a controlled external provider boundary and separate live inference evidence; do not mock the method under test.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_llm_integration_pipeline.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #519.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #520 — Extensive mocking of LLM components (mock_brain_io, mock_prompt_builder, mock_output_interpreter) means this test won't detect real integration failures. The test should use more realistic implementations or at least verify the actual method calls and data flow.

**Acceptance:** Connect actual prompt construction, parsing and executable actions with recorded provider input/output; distinguish controlled model boundaries from live model readiness.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_llm_integration_pipeline.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #520.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #522 — The test mocks the strategy_manager to always return successful responses, which means the test will pass even if the actual strategy update logic is broken. Design the test to verify real integration behavior rather than mocked success.

**Acceptance:** Exercise a real controller/strategy/action/state pipeline; never substitute a strategy that always returns success or infer gameplay readiness from mocked display objects.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the behavioral test quality acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_system_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #522.

#### #539 — Similar to the social prompt test, this only verifies that the destination string appears in the output but doesn't test if the travel prompt is properly constructed with correct character state information and prompt formatting.

**Acceptance:** Generate travel prompts from supported Character/location/state and assert complete destination/context/role formatting, including missing destination behavior.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_travel_prompt.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #539.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #540 — This test only checks if action strings are present in the prompt but doesn't verify that the prompt is properly formatted or that the character information is correctly included. The test should validate the complete prompt structure to ensure it functions as expected.

**Acceptance:** Generate social prompts with eligible actual action options, correct Character context and complete role formatting; substring-only assertions do not establish correctness.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_travel_prompt.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #540.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #541 — Using `object` as a stub for `Character` creates a mock that doesn't accurately represent the actual Character class behavior. This test design may pass even if the code doesn't work correctly with real Character objects that have specific attributes and methods.

**Acceptance:** Replace object or permissive stand-ins with actual Character-compatible data and prove unsupported attributes fail rather than being fabricated by the test setup.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader behavioral test quality requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_social_travel_prompt.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #541.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #682 — [TESTING] Duplicate and Conflicting Test Files for Building Loading

**Acceptance:** Compare the two current building-loading suites case by case, retain distinct unit/integration behavior and update referenced commands without losing unique assertions.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #682.

#### #683 — [TESTING] MagicMock/Mock Usage Contradicts Memory and Build Testing Best Practices

**Acceptance:** Audit permissive memory/build/Character stand-ins, retain only deliberate external boundaries/negative examples and assert actual behavior; selected Phase 3 test-PR work is only part of this broader task.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current behavioral test quality code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #681, canonical topic #683; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #683.

### Buildings and ownership

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#171](https://github.com/dhar174/tiny_village/issues/171) | verify | 6 | dhar174 |
| [#174](https://github.com/dhar174/tiny_village/issues/174) | defer | 6 | dhar174 |
| [#530](https://github.com/dhar174/tiny_village/issues/530) | implement | 6 | dhar174 |

#### #171 — tiny_buildings.py & Building System

**Acceptance:** Verify type-specific building functionality, valid interactions, ownership effects and robust custom loading through real graph/action state; constructor success alone is insufficient.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current buildings and ownership code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #171; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #751. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_buildings.py:493, tiny_building_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #171.

#### #174 — Ownership

**Acceptance:** Resolve the existing in-scope decision, then prove owner-based access/income rules and graph/economic effects with non-owner cases; do not silently add this to the MVP.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader buildings and ownership requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent #171, canonical topic #171; native prerequisites #198; proposed execution prerequisites #748; handoff #174. blocked by native #198.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_buildings.py:1, tiny_building_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #174.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

#### #530 — The energy threshold of 5 is a magic number. Consider defining this as a named constant (e.g., MIN_ENERGY_FOR_INTERACTIONS = 5) to improve maintainability and make it easier to adjust this value in the future.

**Acceptance:** Document the supported interaction energy scale/threshold, test values below/at/above it with real Characters and preserve meaningful action gating.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the buildings and ownership acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #171; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #196. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_buildings.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #530.

### Canonical scenario and release evidence

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#223](https://github.com/dhar174/tiny_village/issues/223) | implement | 8 | dhar174 |
| [#474](https://github.com/dhar174/tiny_village/issues/474) | verify | 8 | dhar174 |
| [#652](https://github.com/dhar174/tiny_village/issues/652) | verify | 8 | dhar174 |
| [#654](https://github.com/dhar174/tiny_village/issues/654) | verify | 2 | dhar174 |
| [#655](https://github.com/dhar174/tiny_village/issues/655) | implement | 8 | dhar174 |
| [#656](https://github.com/dhar174/tiny_village/issues/656) | implement | 8 | dhar174 |
| [#658](https://github.com/dhar174/tiny_village/issues/658) | implement | 8 | dhar174 |

#### #223 — General Recommendations & Priorities for Demo

**Acceptance:** Use the accepted #724 fixture and #751/#658 trace to verify core planning, economic actions, events and understandable UI; startup-only success is not MVP readiness.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the canonical scenario and release evidence acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #658; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #658. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:158, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #223.

#### #474 — The demo script relies heavily on mock classes that don't represent real functionality. While this is a demo, it creates fake implementations that may not accurately demonstrate how the system works with real components, potentially misleading users about actual capabilities.

**Acceptance:** Make the specifically cited map/location demo exercise real map/location/action objects and honestly label limits; #654 fixed a different official minimal entrypoint.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #658; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #658. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect demo_map_location_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #474.

#### #652 — Review and calculate how close to MVP tiny village is

**Acceptance:** Report actual evidence per accepted MVP criterion and unresolved blockers; do not invent an overall completion percentage or equate startup with economic gameplay.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #658; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #658. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #652.

#### #654 — Reduce demo dependency on heavy modules

**Acceptance:** Preserve completed #788/#726 startup/basic-memory acceptance evidence and existing closed state; no gameplay, live-inference or wider phase approval is inferred.

**Disposition/evidence:** verify; `historical_completed_evidence_preserved`. Already CLOSED/COMPLETED through merged #788 and preserved #726 validation; original cohort membership is historical and does not reopen it. Existing pre-merge wording may be contextualized without changing accepted scope.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #723, canonical topic #658; native prerequisites #725; proposed execution prerequisites none; handoff #726. closed: preserve existing disposition.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #654.

#### #655 — Enhance demo logs for narrative clarity

**Acceptance:** Print actual per-Character intents/actions/outcomes, triggering events and final who-did-what summary from the canonical scenario; startup logs alone do not suffice.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the canonical scenario and release evidence acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #658; proposed execution prerequisites #759; handoff #658. blocked by native #658.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #655.

#### #656 — Update and reconcile MVP/demo documentation

**Acceptance:** Reconcile README/Quickstart/status claims with current minimal/visual/canonical commands, archive labels and executed scenario evidence.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the canonical scenario and release evidence acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #759; proposed execution prerequisites none; handoff #658. blocked by native #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #656.

#### #658 — Define and freeze the MVP demo path

**Acceptance:** Publish one documented deterministic real gameplay scenario with fixed fixture/seed, events/actions/outcomes and final result; preserve both headless and Pygame presentations.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the canonical scenario and release evidence acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #656; proposed execution prerequisites #759; handoff #658. blocked by native #656.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:158, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #658.

### Character behavior and goals

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#139](https://github.com/dhar174/tiny_village/issues/139) | implement | 4 | dhar174 |
| [#140](https://github.com/dhar174/tiny_village/issues/140) | defer | 9 | dhar174 |
| [#141](https://github.com/dhar174/tiny_village/issues/141) | defer | 9 | dhar174 |
| [#142](https://github.com/dhar174/tiny_village/issues/142) | defer | 9 | dhar174 |
| [#143](https://github.com/dhar174/tiny_village/issues/143) | implement | 4 | dhar174 |
| [#144](https://github.com/dhar174/tiny_village/issues/144) | implement | 6 | dhar174 |
| [#600](https://github.com/dhar174/tiny_village/issues/600) | implement | 6 | dhar174 |
| [#601](https://github.com/dhar174/tiny_village/issues/601) | defer | 9 | dhar174 |

#### #139 — Minimum completion of TODOs in tiny_characters.py

**Acceptance:** Verify real Character composition and track history/preferences/banking/goals/skills individually; lightweight construction does not complete broader behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the character behavior and goals acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:2087; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #139.

#### #140 — History

**Acceptance:** Retain structured long-term experience with relevant temporal/causal context and show influence on recall or decisions beyond a single recent_event field.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader character behavior and goals requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #155. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #140.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #141 — Likes/Dislikes

**Acceptance:** Define explicit preferences and verify their source and influence on decisions without contradicting traits or remembered experience.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader character behavior and goals requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #141.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #142 — Bank Accounts

**Acceptance:** Define deposit/withdrawal/loan rules and demonstrate conservation, rejected invalid transactions and persistence; ordinary wealth changes are not a banking system.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader character behavior and goals requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #142.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #143 — Behavioral Methods:

**Acceptance:** Verify actual activity animation, memory-informed decisions and contextual goal generation, preserving their separate Phase 4/8/9 implementation boundaries.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the character behavior and goals acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:2468; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #143.

#### #144 — Skill Usage

**Acceptance:** Prove skills affect action eligibility/success/efficiency or unlocks, and successful supported activities update the real skill representation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the character behavior and goals acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #196. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #144.

#### #600 — Support goal-aware required-item extraction from decisions and completion criteria

**Acceptance:** Derive required items from real goal completion/decision context and verify resulting eligible work/trade/consume choices; do not hardcode the fixture item name as a general solution.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the character behavior and goals acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #750. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:2087; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #600.

#### #601 — Add luxury calculation from recent environmental history

**Acceptance:** Compute luxury/happiness from a documented recent environmental history and verify sustained versus transient surroundings produce expected outcomes.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader character behavior and goals requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #139, canonical topic #139; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_characters.py:1730; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #601.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Configuration validation and migration

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#62](https://github.com/dhar174/tiny_village/issues/62) | implement | 7 | dhar174 |
| [#93](https://github.com/dhar174/tiny_village/issues/93) | implement | 7 | dhar174 |
| [#94](https://github.com/dhar174/tiny_village/issues/94) | implement | 7 | dhar174 |
| [#95](https://github.com/dhar174/tiny_village/issues/95) | defer | 9 | dhar174 |
| [#96](https://github.com/dhar174/tiny_village/issues/96) | defer | 9 | dhar174 |
| [#97](https://github.com/dhar174/tiny_village/issues/97) | implement | 7 | dhar174 |
| [#98](https://github.com/dhar174/tiny_village/issues/98) | implement | 7 | dhar174 |
| [#99](https://github.com/dhar174/tiny_village/issues/99) | implement | 7 | dhar174 |
| [#100](https://github.com/dhar174/tiny_village/issues/100) | implement | 7 | dhar174 |
| [#101](https://github.com/dhar174/tiny_village/issues/101) | implement | 7 | dhar174 |

#### #62 — Add configuration validation and schema checking (export/import configuration)

**Acceptance:** Validate import/export configuration against a documented schema; invalid input leaves the existing configuration/world intact and produces an actionable error.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #62.

#### #93 — TODO: Add configuration validation and schema checking (in export_configuration)

**Acceptance:** Reject invalid export configuration before writing, document the supported schema and round-trip valid exported sections; keep distinct import coverage in #98.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #93.

#### #94 — TODO: Add configuration versioning and migration support (in export_configuration)

**Acceptance:** Version the exported schema and test supported migrations plus rejection of unsupported versions without partial state mutation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #94.

#### #95 — TODO: Add configuration encryption for sensitive data (in export_configuration)

**Acceptance:** Identify which configuration fields require encryption and a key-management contract before implementation; demonstrate round-trip and failed-decryption handling without recording secrets.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader configuration validation and migration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #95.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #96 — TODO: Add configuration templates and presets (in export_configuration)

**Acceptance:** Provide documented, validated presets that preserve accepted defaults; reject invalid/unsupported profiles with a clear capability explanation.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader configuration validation and migration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #96.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #97 — TODO: Add proper data sanitization (in export_configuration)

**Acceptance:** Exclude unsupported/sensitive serialized values from export according to a documented policy and validate that import cannot inject unintended runtime objects.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #97.

#### #98 — TODO: Add configuration validation and safety checks (in import_configuration)

**Acceptance:** Parse and validate imports before replacing active configuration; malformed/unsafe input leaves the active simulation intact.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #98.

#### #99 — TODO: Add configuration migration for version compatibility (in import_configuration)

**Acceptance:** Migrate each explicitly supported configuration version and reject unknown versions without changing the current world/configuration.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #99.

#### #100 — TODO: Add backup creation before importing new configuration (in import_configuration)

**Acceptance:** Create and verify a recoverable backup before configuration replacement; backup failure must not destroy the only known-good state.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #100.

#### #101 — TODO: Add selective configuration import (only specific sections) (in import_configuration)

**Acceptance:** Import only selected supported sections, validate dependencies across sections, and leave unselected sections unchanged.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the configuration validation and migration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #62, canonical topic #62; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #101.

### Contributor tooling and experiments

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#640](https://github.com/dhar174/tiny_village/issues/640) | defer | 9 | dhar174 |
| [#643](https://github.com/dhar174/tiny_village/issues/643) | verify | 3 | dhar174 |
| [#644](https://github.com/dhar174/tiny_village/issues/644) | defer | 9 | dhar174 |
| [#645](https://github.com/dhar174/tiny_village/issues/645) | defer | 9 | dhar174 |
| [#646](https://github.com/dhar174/tiny_village/issues/646) | defer | 9 | dhar174 |
| [#647](https://github.com/dhar174/tiny_village/issues/647) | verify | 3 | dhar174 |
| [#649](https://github.com/dhar174/tiny_village/issues/649) | defer | 9 | dhar174 |
| [#651](https://github.com/dhar174/tiny_village/issues/651) | verify | 3 | dhar174 |

#### #640 — Build hierarchical Markdown navigation system for agents and contributors

**Acceptance:** Implement the original multi-tier Markdown routing/metadata/ambiguity conventions and validate paths; it is broader than flat map #630 and not a duplicate.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader contributor tooling and experiments requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #640.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #643 — Check Github Actions with GitHub Actions Expert

**Acceptance:** Review the actual supported-profile workflow, effective required checks and retained failures; preserve #726 evidence without approving unrelated legacy/AI workflows.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. #726 supplies current supported-profile CI evidence, not approval of optional AI Mentor or legacy workflows. Inventory/queue decisions belong in #737/#770.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #643.

#### #644 — Check frontend with experts

**Acceptance:** Identify an actual frontend target before React/Next review; approved Pygame UX/UI review remains separate, and this does not authorize a web rewrite.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. The repository has no supported React/Next frontend; preserve the request as deferred until there is a concrete target. Pygame implementation/review remains in #761/#763–#766.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #644.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #645 — Modernize codebase

**Acceptance:** Replace the vague modernization request with evidence-backed bounded changes after core contracts stabilize; no blanket framework migration or feature removal.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader contributor tooling and experiments requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #645.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #646 — Test web app with Playwright

**Acceptance:** Require an actual supported browser application before Playwright work; current native Pygame cannot be validated as a web application.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. There is no browser application target in the supported runtime; Playwright cannot supply native Pygame acceptance. Deferred re-scope requires a concrete browser feature, not a new framework by assumption.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #646.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #647 — System Architecture Review

**Acceptance:** Assess actual subsystem boundaries/risks against code and current engineering evidence, preserving #731's scoped approval without extending it to untested gameplay.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current contributor tooling and experiments code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #647.

#### #649 — Chat about simple app idea

**Acceptance:** Define a concrete TinyVillage-related product question/use case before invoking an unrelated app-idea workflow; retain the original expansive game vision.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader contributor tooling and experiments requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #649.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #651 — Task Researcher and planner

**Acceptance:** Provide scoped issue-backed tasks, prerequisites, acceptance/evidence and human gates; compare the existing 86-item roadmap and this ledger before inventing another plan.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current contributor tooling and experiments code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #736. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #651.

### Documentation and status accuracy

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#217](https://github.com/dhar174/tiny_village/issues/217) | verify | 8 | dhar174 |
| [#630](https://github.com/dhar174/tiny_village/issues/630) | verify | 8 | dhar174 |
| [#633](https://github.com/dhar174/tiny_village/issues/633) | verify | 8 | dhar174 |
| [#634](https://github.com/dhar174/tiny_village/issues/634) | verify | 8 | dhar174 |
| [#635](https://github.com/dhar174/tiny_village/issues/635) | verify | 8 | dhar174 |
| [#636](https://github.com/dhar174/tiny_village/issues/636) | verify | 8 | dhar174 |
| [#641](https://github.com/dhar174/tiny_village/issues/641) | verify | 9 | dhar174 |
| [#642](https://github.com/dhar174/tiny_village/issues/642) | verify | 9 | dhar174 |
| [#648](https://github.com/dhar174/tiny_village/issues/648) | verify | 8 | dhar174 |
| [#650](https://github.com/dhar174/tiny_village/issues/650) | verify | 8 | dhar174 |
| [#653](https://github.com/dhar174/tiny_village/issues/653) | verify | 8 | dhar174 |
| [#681](https://github.com/dhar174/tiny_village/issues/681) | verify | 8 | dhar174 |
| [#684](https://github.com/dhar174/tiny_village/issues/684) | implement | 8 | dhar174 |
| [#685](https://github.com/dhar174/tiny_village/issues/685) | implement | 8 | dhar174 |
| [#686](https://github.com/dhar174/tiny_village/issues/686) | verify | 8 | dhar174 |

#### #217 — Discrepancies with Documentation

**Acceptance:** Map each documentation discrepancy to current code and executed evidence without silently replacing original requirements or declaring the whole integration complete.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #217.

#### #630 — Create organized mapping and visualization for all markdown files

**Acceptance:** Enumerate all tracked Markdown documents with valid relative paths and an actual navigable visualization; a partial root-file list is not complete coverage.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #630.

#### #633 — Step 1: Generate Context Map and Analyze File System Structure

**Acceptance:** Validate the existing context map/folder blueprint against current tracked files/modules and state explicit risks/relationships; stale generated artifacts are not current evidence.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (existing human assignee); native parent #632, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #633.

#### #634 — Step 2: Code Pattern, Dependency, and Architecture Analysis

**Acceptance:** Trace current code patterns and OO/subsystem flows, documenting real boundaries and dependencies rather than copying intended architecture.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #632, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #634.

#### #635 — Step 3: Architecture, Data Flow, and Control Flow Diagram Generation

**Acceptance:** Check existing architecture/control/data diagrams against actual source flows, identify unsupported arrows and publish a readable collected report.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #632, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #635.

#### #636 — Step 4: Gap Analysis and Reporting (Project State vs. Documentation/Specs)

**Acceptance:** Compare documented capability claims with current executed evidence and map each gap to an issue; preserve historical failures and planning/implementation distinctions.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #632, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #636.

#### #641 — Create ADR

**Acceptance:** Retain rationale, alternatives, consequences and human decisions in the existing single decision file; identify any missing architectural decisions without splitting accepted decisions into parallel records.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #641.

#### #642 — Create Product Requirements Document

**Acceptance:** Reconcile the accepted MVP/support/budget contract and broader vision into explicit product requirements; do not claim the bounded fixture exhausts the final product.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #642.

#### #648 — Review Technical Documentation

**Acceptance:** Audit authoritative technical documentation against real interfaces/commands and current evidence, clearly marking archival material.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #648.

#### #650 — Review and improve specs

**Acceptance:** Compare implementation against the already approved UX/visual/UI and runtime specifications; reopen only concrete discrepancies, not settled preference choices.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #650.

#### #653 — Generate Blueprints

**Acceptance:** Validate existing folder/flow/Copilot-instruction blueprints against the current repository, documenting gaps and a maintainable update path.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #653.

#### #681 — Incongruities Between Documentation and Codebase: Master Tracking Issue

**Acceptance:** Track documentation/test/status/architecture gaps with evidence-backed children, keeping current versus archived authority clear and later scope intact.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #681.

#### #684 — [STATUS] Outdated Project Status and Demo Readiness Claims in Documentation

**Acceptance:** Correct present-tense readiness claims using retained evidence and archive labels; explicitly preserve D015 final-product AI rather than repeating the old nice-to-have wording.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Original wording calls LLM/full memory nice-to-have. D015 supersedes that for the final product; only lightweight testing may omit AI dependencies.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #681, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #684.

#### #685 — [ARCHITECTURE] Discrepancies Between Code and Architecture/Integration Docs

**Acceptance:** Document actual building/checkpoint initialization, failure paths and supported public methods; do not infer complete persistence from robust checkpoint scaffolding.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the documentation and status accuracy acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #681, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #685.

#### #686 — [POLISH] Final Alignment and Minor Fixes for Documentation and Code

**Acceptance:** Align entrypoints/method names/status links and segregate negative mock examples; preserve useful historical text with clear archival context instead of rewriting evidence.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current documentation and status accuracy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #681, canonical topic #681; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #656. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect docs/README.md:1, README.md:1, docs/decisions.md:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #686.

### Events and consequences

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#51](https://github.com/dhar174/tiny_village/issues/51) | implement | 5 | dhar174 |
| [#53](https://github.com/dhar174/tiny_village/issues/53) | implement | 5 | dhar174 |
| [#54](https://github.com/dhar174/tiny_village/issues/54) | implement | 5 | dhar174 |
| [#55](https://github.com/dhar174/tiny_village/issues/55) | defer | 9 | dhar174 |
| [#106](https://github.com/dhar174/tiny_village/issues/106) | implement | 5 | dhar174 |
| [#107](https://github.com/dhar174/tiny_village/issues/107) | implement | 5 | dhar174 |
| [#108](https://github.com/dhar174/tiny_village/issues/108) | implement | 5 | dhar174 |
| [#109](https://github.com/dhar174/tiny_village/issues/109) | implement | 5 | dhar174 |
| [#110](https://github.com/dhar174/tiny_village/issues/110) | implement | 5 | dhar174 |
| [#111](https://github.com/dhar174/tiny_village/issues/111) | implement | 5 | dhar174 |
| [#112](https://github.com/dhar174/tiny_village/issues/112) | implement | 5 | dhar174 |
| [#133](https://github.com/dhar174/tiny_village/issues/133) | implement | 5 | dhar174 |
| [#169](https://github.com/dhar174/tiny_village/issues/169) | implement | 5 | dhar174 |
| [#190](https://github.com/dhar174/tiny_village/issues/190) | implement | 5 | dhar174 |
| [#191](https://github.com/dhar174/tiny_village/issues/191) | implement | 5 | dhar174 |
| [#192](https://github.com/dhar174/tiny_village/issues/192) | implement | 5 | dhar174 |
| [#193](https://github.com/dhar174/tiny_village/issues/193) | defer | 9 | dhar174 |
| [#219](https://github.com/dhar174/tiny_village/issues/219) | implement | 5 | dhar174 |
| [#564](https://github.com/dhar174/tiny_village/issues/564) | implement | 5 | dhar174 |
| [#565](https://github.com/dhar174/tiny_village/issues/565) | defer | 9 | dhar174 |
| [#566](https://github.com/dhar174/tiny_village/issues/566) | implement | 5 | dhar174 |
| [#567](https://github.com/dhar174/tiny_village/issues/567) | implement | 5 | dhar174 |
| [#568](https://github.com/dhar174/tiny_village/issues/568) | implement | 5 | dhar174 |
| [#608](https://github.com/dhar174/tiny_village/issues/608) | defer | 9 | dhar174 |

#### #51 — Event System (tiny_event_handler.py)

**Acceptance:** Trigger a real event, route its strategy/action, apply character/world effects once, and retain the graph and memory trace; complete outstanding child acceptance without treating status strings as evidence.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1743; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #51.

#### #53 — Enhance Event Effects: Diversify and Deepen Game Impact

**Acceptance:** Demonstrate character, location and relationship effects with conditions/chaining, bounded changes and before/after logging; preserve all detailed child requirements.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #51, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #53.

#### #54 — Event-Driven AI: Ensure StrategyManager and characters react meaningfully to events generated by EventHandler.

**Acceptance:** A real triggered event changes the selected character strategy or plan, and its executable outcome changes authoritative state.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #51, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #219. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #54.

#### #55 — Content in tiny_event_handler.py

**Acceptance:** Provide documented, executable event templates that produce distinct emergent outcomes through the supported schema; additional content follows the validated event pipeline.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader events and consequences requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #51, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #55.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #106 — TODO: Implement proper event processing system (in _process_pending_events)

**Acceptance:** Process a real pending event through the authoritative handler and executable strategy outcome, not a placeholder queue drain.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #106.

#### #107 — TODO: Add event types (social, economic, environmental, etc.) (in _process_pending_events)

**Acceptance:** Validate supported social/economic/environmental event payloads and dispatch actual effects; invalid event types fail clearly without partial mutation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #107.

#### #108 — TODO: Add event consequences and ripple effects (in _process_pending_events)

**Acceptance:** Demonstrate documented event consequences and ripple effects with bounded propagation and actual graph changes.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #108.

#### #109 — TODO: Add event prioritization and scheduling (in _process_pending_events)

**Acceptance:** Order and schedule events deterministically by documented time/priority rules, including ties and overdue events, without repeated application.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #109.

#### #110 — TODO: Add cross-character event interactions (in _process_pending_events)

**Acceptance:** Resolve event participants to real Characters and verify effects target the intended participants and authoritative world.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #110.

#### #111 — TODO: Add event persistence and memory (in _process_pending_events)

**Acceptance:** Record supported pending/completed event context and memory once; validate save/load continuation without duplicating effects.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #111.

#### #112 — TODO: Add event-driven story generation (in _process_pending_events)

**Acceptance:** Demonstrate an actual event advances story state and records an understandable outcome; fabricated story success does not prove simulation effects.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #133, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #112.

#### #133 — Event Processing: Key Area for Completion

**Acceptance:** Trace handler detection → strategy → character-aware action → graph/memory outcome with declared dispatch and exactly-once semantics.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:4643; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #133.

#### #169 — Event Handling

**Acceptance:** A real handler event influences strategy selection and an executed Character plan; retain event-driven behavior beyond a method call or event registration.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #165, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #219. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #169.

#### #190 — Integration with Gameplay Loop

**Acceptance:** Trigger handler events during the real gameplay update, consume returned strategies and apply character-aware action outcomes.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #189, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #219. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:305, tiny_gameplay_controller.py:1743; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #190.

#### #191 — Event Impact

**Acceptance:** Preserve all event-impact acceptance and link the broader #53 effect work; compare character/world/relationship outcomes before considering consolidation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #189, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #191.

#### #192 — Event-Driven AI

**Acceptance:** Verify the event-driven AI requirement through real strategy/action changes; overlap with #54 is retained pending parent/PR reconciliation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #189, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #219. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #192.

#### #193 — Content

**Acceptance:** Demonstrate executable storytelling event content through the supported handler; overlap with #55 is retained without automatic closure.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader events and consequences requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #189, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #193.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #219 — EventHandler Integration

**Acceptance:** Route detected event strategies through the proper Character and action executor, asserting authoritative outcomes and exactly-once feedback.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #217, canonical topic #51; native prerequisites #743; proposed execution prerequisites none; handoff #219. blocked by native #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:305, tiny_gameplay_controller.py:1743; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #219.

#### #564 — 3) World / GraphManager Effects (Environment/Resources/Regions/Weather)

**Acceptance:** Implement four supported world/location effect types through real EventHandler→GraphManager mutation, including missing nodes/unknown fields and a real integration trace; retain the schema-v2 prerequisite.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #53, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #564.

#### #565 — 4) Relationship / Faction Effects (Trust/Alliance/Rivalry/Reputation)

**Acceptance:** Meet the original five relationship-effect types plus missing-edge, invalid-target and symmetry contracts against actual graph edges; basic MVP events do not imply this broader faction scope is complete.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader events and consequences requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #53, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #565.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #566 — 5) Conditional + Stacking + Chaining Effects (Prereqs + Multi-step Outcomes)

**Acceptance:** Demonstrate two effect-condition patterns, two stacking policies and guarded chaining with true/false conditions, cap behavior, application order and termination.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #53, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #566.

#### #567 — 6) Effect Logging + Balance Hooks (Audit Trail + Tuning Support)

**Acceptance:** Retain structured event/effect/target/before/after/time records for every applied effect, configurable caps/scales and a debug diff check using actual outcomes.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #53, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #567.

#### #568 — 7) Authoring Examples + Docs (Golden Examples + Dev Notes)

**Acceptance:** Provide 5–10 schema-v2 event examples, explicit attribute/target mapping, an authoring guide and an automated golden-path assertion of real deltas and logs.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the events and consequences acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #53, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #745. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #568.

#### #608 — Add social-event generation

**Acceptance:** Generate and resolve relationship-building/conflict/turning-point events from actual social context with bounded effects.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader events and consequences requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #212, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #608.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Gameplay orchestration

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#56](https://github.com/dhar174/tiny_village/issues/56) | implement | 4 | dhar174 |
| [#138](https://github.com/dhar174/tiny_village/issues/138) | defer | 9 | dhar174 |
| [#615](https://github.com/dhar174/tiny_village/issues/615) | defer | 9 | dhar174 |
| [#616](https://github.com/dhar174/tiny_village/issues/616) | defer | 9 | dhar174 |

#### #56 — Core Gameplay Controller (tiny_gameplay_controller.py)

**Acceptance:** Reconcile controller child tasks against executed runtime behavior; preserve later features and report unresolved failures explicitly instead of closing the umbrella from startup success.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the gameplay orchestration acceptance as ordered implementation/integration work in Phase 4; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #56; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #220. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #56.

#### #138 — Feature Implementations

**Acceptance:** Replace feature-status claims with evidence per subsystem; preserve weather/social/quest/procedural/audio scope and their distinct later gates.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader gameplay orchestration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #56; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #138.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #615 — Implement procedural content generation foundations

**Acceptance:** Select a concrete procedural-content target, seed/constraints and validity checks, then prove reproducible generated content; no generic generator or status string counts as completion.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader gameplay orchestration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #56; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #615.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #616 — Implement sound and music system

**Acceptance:** Demonstrate supported music/SFX playback, volume/stop controls, event routing and cleanup with explicit unavailable-device behavior; keep UI/event hooks consistent.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader gameplay orchestration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #56; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #616.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Input and accessibility

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#60](https://github.com/dhar174/tiny_village/issues/60) | implement | 8 | dhar174 |
| [#61](https://github.com/dhar174/tiny_village/issues/61) | implement | 8 | dhar174 |
| [#71](https://github.com/dhar174/tiny_village/issues/71) | implement | 8 | dhar174 |
| [#74](https://github.com/dhar174/tiny_village/issues/74) | defer | 9 | dhar174 |
| [#75](https://github.com/dhar174/tiny_village/issues/75) | defer | 9 | dhar174 |
| [#76](https://github.com/dhar174/tiny_village/issues/76) | defer | 9 | dhar174 |
| [#77](https://github.com/dhar174/tiny_village/issues/77) | implement | 8 | dhar174 |
| [#78](https://github.com/dhar174/tiny_village/issues/78) | implement | 8 | dhar174 |
| [#79](https://github.com/dhar174/tiny_village/issues/79) | defer | 9 | dhar174 |
| [#80](https://github.com/dhar174/tiny_village/issues/80) | implement | 8 | dhar174 |
| [#81](https://github.com/dhar174/tiny_village/issues/81) | implement | 8 | dhar174 |
| [#82](https://github.com/dhar174/tiny_village/issues/82) | defer | 9 | dhar174 |
| [#83](https://github.com/dhar174/tiny_village/issues/83) | implement | 8 | dhar174 |
| [#135](https://github.com/dhar174/tiny_village/issues/135) | implement | 8 | dhar174 |
| [#380](https://github.com/dhar174/tiny_village/issues/380) | defer | 9 | dhar174 |
| [#382](https://github.com/dhar174/tiny_village/issues/382) | duplicate | 9 | dhar174 |

#### #60 — Add sophisticated input handling (handle_events)

**Acceptance:** Exercise mouse and keyboard routes against real UI controls, with focus ownership and one action per input under the approved interaction specification.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #60.

#### #61 — Add accessibility features (handle_events)

**Acceptance:** Verify keyboard navigation, readable focus/state indicators, contrast and supported accessibility behavior using the agreed UI and the dedicated human accessibility review.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #61.

#### #71 — TODO: Add keyboard shortcuts configuration (in handle_events)

**Acceptance:** Route configurable shortcuts through the focused control/dialog before global handlers; retain the approved S/L/R, Step and speed defaults.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #71.

#### #74 — TODO: Add mouse gesture recognition (in handle_events)

**Acceptance:** Specify gestures and demonstrate recognized/unrecognized input without conflicting with click/drag/selection behavior.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #74.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #75 — TODO: Add multi-touch support for mobile (in handle_events)

**Acceptance:** Choose a supported touch platform and demonstrate selection, inspection and UI control without shrinking essential text or relying on hover.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #75.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #76 — TODO: Add gamepad/controller support (in handle_events)

**Acceptance:** Specify controller mapping and demonstrate focus/activation/cancellation on a supported device without changing simulation semantics.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #76.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #77 — TODO: Add accessibility features (screen reader, high contrast mode) (in handle_events)

**Acceptance:** Verify the explicitly listed screen-reader/high-contrast requirements and keyboard support; keep unique acceptance beyond the broader accessibility header #61.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #77.

#### #78 — TODO: Add customizable key bindings (in handle_events)

**Acceptance:** Persist remapped bindings, prevent conflicts, retain text-entry focus, and expose the active shortcut labels with safe defaults.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #78.

#### #79 — TODO: Add input recording and playback for testing (in handle_events)

**Acceptance:** Record input with logical tick context and replay to the same supported scenario outcome; distinguish UI history from a complete canonical trace.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #79.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #80 — TODO: Add mouse interaction handling (in handle_events, under MOUSEBUTTONDOWN)

**Acceptance:** Click a real villager/building and verify selection, inspection and map/UI hit testing; clicks consumed by a panel must not reach the map.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #80.

#### #81 — TODO: Add right-click context menus (in handle_events, under MOUSEBUTTONDOWN)

**Acceptance:** Verify context-menu positioning, keyboard focus, dismissal and enabled/disabled actions; informational inspection must remain live unless the user pauses.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #81.

#### #82 — TODO: Add zoom functionality (in handle_events, under MOUSEWHEEL)

**Acceptance:** Define camera zoom limits and whole-number world-art scaling; demonstrate hit testing and readable independent HUD text at each supported setting.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #82.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #83 — TODO: Add scroll-based UI navigation (in handle_events, under MOUSEWHEEL)

**Acceptance:** Scroll the focused entity/history view, preserving selection and reading position; Page Up/Down must not also change playback speed.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #135, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:3027, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #83.

#### #135 — Input Handling: Needs to be made more sophisticated, with support for various input methods and customization.

**Acceptance:** Implement the agreed keyboard/mouse/focus model and track optional device/input extensions separately; do not close from a single shortcut working.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the input and accessibility acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #135.

#### #380 — Feature: Implement Player Interaction and Controls

**Acceptance:** Define and implement player building placement and direct movement commands with valid/invalid placement and state effects; these exceed the accepted inspection/playback MVP.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader input and accessibility requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #380.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #382 — Feature: Implement Player Interaction and Controls

**Acceptance:** Same building-placement and direct-command requirements as #380; close only after approval of this exact duplicate and canonical link.

**Disposition/evidence:** duplicate; `source_reviewed_not_executed`. Exact duplicate: original #380 and #382 bodies are byte-for-byte identical and neither has an issue discussion. Preserve #380 as canonical.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #135; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #770. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #382.

### Items, inventory and economy

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#194](https://github.com/dhar174/tiny_village/issues/194) | implement | 6 | dhar174 |
| [#195](https://github.com/dhar174/tiny_village/issues/195) | implement | 6 | dhar174 |
| [#196](https://github.com/dhar174/tiny_village/issues/196) | implement | 6 | dhar174 |
| [#197](https://github.com/dhar174/tiny_village/issues/197) | implement | 6 | dhar174 |
| [#198](https://github.com/dhar174/tiny_village/issues/198) | implement | 6 | dhar174 |

#### #194 — Item System (tiny_items.py) Completion

**Acceptance:** Complete concrete item types/interactions/inventory/economic child contracts; item class definitions alone do not prove the economic loop.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the items, inventory and economy acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #194; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #195. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:139, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #194.

#### #195 — More Item Types

**Acceptance:** Construct supported food/tool/resource/clothing/quest items with required properties and categorization; preserve the unresolved default Door action #775.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the items, inventory and economy acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #194, canonical topic #194; native prerequisites #748; proposed execution prerequisites none; handoff #195. blocked by native #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #195.

#### #196 — Item Interactions

**Acceptance:** Execute actual equip/wear/craft/consume interactions against valid state, retaining effects and explicit failure behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the items, inventory and economy acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #194, canonical topic #194; native prerequisites #195; proposed execution prerequisites #748; handoff #196. blocked by native #195.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #196.

#### #197 — Inventory Management

**Acceptance:** Expose inventory categories, needs and valid trade/drop options to decisions/UI, reflecting real quantities and invalid operations.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the items, inventory and economy acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #194, canonical topic #194; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #750. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #197.

#### #198 — Economic Integration

**Acceptance:** Demonstrate goods produced, traded and consumed with conserved supported balances and real job/need consequences.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the items, inventory and economy acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #194, canonical topic #194; native prerequisites #199, #196; proposed execution prerequisites #748; handoff #198. blocked by native #199,#196.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #198.

### Jobs and economic production

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#199](https://github.com/dhar174/tiny_village/issues/199) | implement | 6 | dhar174 |
| [#200](https://github.com/dhar174/tiny_village/issues/200) | implement | 6 | dhar174 |
| [#201](https://github.com/dhar174/tiny_village/issues/201) | implement | 6 | dhar174 |
| [#202](https://github.com/dhar174/tiny_village/issues/202) | implement | 6 | dhar174 |
| [#203](https://github.com/dhar174/tiny_village/issues/203) | implement | 6 | dhar174 |
| [#204](https://github.com/dhar174/tiny_village/issues/204) | implement | 6 | dhar174 |

#### #199 — Job System (tiny_jobs.py)

**Acceptance:** Verify job assignment/leaving, prerequisites, work outcomes, progression and income with valid catalog data and real Characters; resolve #787 diagnostics separately.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent #749, canonical topic #199; native prerequisites #748; proposed execution prerequisites none; handoff #199. blocked by native #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:5, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #199.

#### #200 — Character Assignment

**Acceptance:** Apply for, assign and leave supported jobs with validated requirements and graph state; reject unavailable/invalid roles without process exit.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #199, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #199. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:1, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #200.

#### #201 — Job Performance

**Acceptance:** Show actual work results affect job performance and downstream income/progression, with documented scales and unsuccessful-work behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #199, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #199. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:259, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #201.

#### #202 — Job Actions

**Acceptance:** Execute type-specific work actions that consume time/energy and produce documented resources/value through authoritative state.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #199, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #199. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:259, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #202.

#### #203 — Career Progression

**Acceptance:** Meet documented promotion/skill conditions, advance a real career and persist it; do not manufacture progression through a mock.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #199, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #199. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:1, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #203.

#### #204 — Economic Simulation

**Acceptance:** Execute the work → income/production → market → buy/eat chain with scarcity and conservation; keep broad economy expansion beyond the bounded fixture visible.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the jobs and economic production acceptance as ordered implementation/integration work in Phase 6; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #199, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #199. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:1, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #204.

### Loop instrumentation and performance

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#58](https://github.com/dhar174/tiny_village/issues/58) | defer | 9 | dhar174 |
| [#59](https://github.com/dhar174/tiny_village/issues/59) | defer | 9 | dhar174 |
| [#67](https://github.com/dhar174/tiny_village/issues/67) | defer | 9 | dhar174 |
| [#68](https://github.com/dhar174/tiny_village/issues/68) | implement | 8 | dhar174 |
| [#69](https://github.com/dhar174/tiny_village/issues/69) | defer | 9 | dhar174 |
| [#134](https://github.com/dhar174/tiny_village/issues/134) | verify | 8 | dhar174 |
| [#595](https://github.com/dhar174/tiny_village/issues/595) | verify | 8 | dhar174 |
| [#596](https://github.com/dhar174/tiny_village/issues/596) | verify | 8 | dhar174 |
| [#597](https://github.com/dhar174/tiny_village/issues/597) | verify | 8 | dhar174 |

#### #58 — Add frame rate adjustment based on performance (game_loop)

**Acceptance:** Under sustained measured load, adjust frame pacing within declared limits while preserving logical simulation outcomes; profile before selecting an adaptive policy.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader loop instrumentation and performance requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #762. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #58.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #59 — Add network synchronization for multiplayer (game_loop)

**Acceptance:** Define multiplayer authority, synchronization and failure contracts, then demonstrate two clients agree on logical state; this extends beyond the accepted single-process MVP.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader loop instrumentation and performance requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #59.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #67 — TODO: Add mod system integration (in game_loop)

**Acceptance:** Specify a supported mod boundary and demonstrate loading, rejecting and isolating incompatible content after the core interfaces stabilize.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader loop instrumentation and performance requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #67.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #68 — TODO: Add automated testing hooks (in game_loop)

**Acceptance:** Expose a bounded deterministic loop/scenario entrypoint whose result asserts real events, actions and state changes; startup-only smoke is insufficient.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the loop instrumentation and performance acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #68.

#### #69 — TODO: Add real-time configuration updates (in game_loop)

**Acceptance:** Define which settings may change live; apply them only at a safe boundary, reject invalid changes and retain current state on failure.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader loop instrumentation and performance requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #69.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #134 — Game Loop Enhancements: Key area

**Acceptance:** Track remaining loop enhancements individually; verify instrumentation separately from adaptive FPS, configuration, networking and profiling/optimization.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current loop instrumentation and performance code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #134.

#### #595 — Follow-up: Add performance profiling and optimization to game_loop

**Acceptance:** Provide retained real-loop profiling/stage evidence identifying actionable hotspots and compare any optimization; instrumentation in #596 alone does not prove optimization.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current loop instrumentation and performance code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #762. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #595.

#### #596 — Add runtime performance monitoring in game_loop

**Acceptance:** Record real-loop stage/frame timings and expose bounded rolling summaries, including collection disabled state; do not infer frame budget compliance or healthy gameplay.

**Disposition/evidence:** verify; `passed_scoped_probe`. One actual SDL-dummy game-loop frame records all five timing stages. Bounded/disabled collection passes the corrected probe. The same frame logs map/animation/recovery defects; this is not healthy-gameplay or frame-budget evidence.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #762. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:2853; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #596.

#### #597 — Add frame-time analysis and optimization suggestions

**Acceptance:** Exercise empty, normal and above-budget samples and dominant-stage suggestions; verify bounded history and disabled monitoring without claiming benchmark acceptance.

**Disposition/evidence:** verify; `passed_scoped_probe`. Corrected probe validates empty/normal/slow/dominant-stage samples, ten-record eviction and disabled collection. Current policy can suggest dominant-stage optimization below budget; that is documented behavior, not a measured performance failure.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #134, canonical topic #134; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #762. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #597.

### Map movement and weighted paths

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#177](https://github.com/dhar174/tiny_village/issues/177) | defer | 9 | dhar174 |
| [#178](https://github.com/dhar174/tiny_village/issues/178) | defer | 9 | dhar174 |
| [#180](https://github.com/dhar174/tiny_village/issues/180) | defer | 9 | dhar174 |
| [#475](https://github.com/dhar174/tiny_village/issues/475) | verify | 9 | dhar174 |

#### #177 — Pathfinding Robustness:

**Acceptance:** Compare weighted A* results against a correctness reference on dynamic obstacles/complex maps, including unreachable goals, before optimizing membership queues.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader map movement and weighted paths requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #176, canonical topic #177; native prerequisites #768; proposed execution prerequisites none; handoff #177. blocked by native #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:785, tiny_locations.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #177.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #178 — Terrain Impact

**Acceptance:** Validate weighted terrain changes path cost/choice and movement pacing; compare against a correctness reference before optimization.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader map movement and weighted paths requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #176, canonical topic #177; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #177. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:1450, tiny_locations.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #178.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #180 — Points of Interest

**Acceptance:** Demonstrate a non-building POI with actual location, capacity, preconditions and effects; invalid/missing coordinates fail clearly.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader map movement and weighted paths requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #176, canonical topic #177; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #177. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:1, tiny_locations.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #180.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #475 — The code assumes `character.location.coordinates_location` exists and is indexable, but there's no validation. If the character's location structure is different or None, this will cause an AttributeError or IndexError.

**Acceptance:** Exercise POI interaction with missing/None/invalid/valid location coordinates and capacity; valid input must produce the intended real action eligibility.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current map movement and weighted paths code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #177; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #177. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_locations.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #475.

### Memory and semantic services

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#11](https://github.com/dhar174/tiny_village/issues/11) | defer | 9 | dhar174 |
| [#152](https://github.com/dhar174/tiny_village/issues/152) | defer | 9 | dhar174 |
| [#153](https://github.com/dhar174/tiny_village/issues/153) | defer | 9 | dhar174 |
| [#154](https://github.com/dhar174/tiny_village/issues/154) | defer | 9 | dhar174 |
| [#155](https://github.com/dhar174/tiny_village/issues/155) | defer | 9 | dhar174 |
| [#156](https://github.com/dhar174/tiny_village/issues/156) | defer | 9 | dhar174 |
| [#157](https://github.com/dhar174/tiny_village/issues/157) | defer | 9 | dhar174 |
| [#518](https://github.com/dhar174/tiny_village/issues/518) | verify | 9 | dhar174 |
| [#528](https://github.com/dhar174/tiny_village/issues/528) | verify | 9 | dhar174 |

#### #11 — Performance: Optimize memory management and prevent memory leaks in MemoryManager

**Acceptance:** Bound embedding/model/index storage over a repeated record/recall/delete lifecycle; retain memory measurements and confirm supported cleanup without claiming basic-mode bounds prove full AI memory bounds.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent #769, canonical topic #152; native prerequisites #155; proposed execution prerequisites #768; handoff #11. blocked by native #155.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:336, tiny_characters.py:204; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #11.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #152 — tiny_memories.py (Memory Management)

**Acceptance:** Verify semantic recording/retrieval, BST role, coherence, decision influence, fact extraction and resource bounds against the original memory requirements; basic recent recall is a completed prerequisite, not the full feature.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:614, tiny_characters.py:2087; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #152.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #153 — BST Usage

**Acceptance:** Demonstrate sorted/filtered retrieval within a GeneralMemory or document the exact retained BST role; do not infer functionality from class existence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #152, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:676, tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #153.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #154 — Memory Coherence

**Acceptance:** Specify contradictory-memory handling and demonstrate changed belief/recall behavior while preserving original experiences and provenance.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #152, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #155. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:614, tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #154.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #155 — Integration with Decision-Making

**Acceptance:** Show relevant memories change actual goal/plan choice through StrategyManager/GOAP; recording/recall or prompt inclusion alone is insufficient.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #152, canonical topic #152; native prerequisites #607; proposed execution prerequisites #768; handoff #155. blocked by native #607.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:614, tiny_characters.py:2087; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #155.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #156 — Fact Extraction Robustness

**Acceptance:** Use a documented extraction corpus to check correct facts, invalid inputs and unavailable-model behavior; preserve actual model/evaluation provenance.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #152, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:1, tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #156.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #157 — Computational Cost

**Acceptance:** Measure full NLP record/recall cost and memory usage before optimizing; retain core/basic versus full-AI capability boundaries.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader memory and semantic services requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #152, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #11. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:1, tiny_characters.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #157.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #518 — Hardcoding the matplotlib backend to 'Agg' could cause issues in environments where other backends are required. Consider making this configurable or detecting the appropriate backend based on the environment.

**Acceptance:** Check full-profile plotting respects the caller's selected backend; basic/headless imports must not force a matplotlib backend or initialize rendering.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. The #654 lazy-service boundary removed eager matplotlib import/backend setup from the supported memory path. Full-profile caller-selected plotting behavior is not exercised by this audit.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #518.

#### #528 — The function returns True even when normalization setting fails (line 2196-2198). This could mask important errors. If normalization is critical for the application, consider returning False when normalization fails, or at least document why this is acceptable.

**Acceptance:** Induce actual semantic-index normalization failure and verify the result/capability accurately reports failure or a documented partial-load state, retaining the original cause.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current memory and semantic services code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #152; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #528.

### Panels, inspection and player feedback

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#123](https://github.com/dhar174/tiny_village/issues/123) | verify | 8 | dhar174 |
| [#124](https://github.com/dhar174/tiny_village/issues/124) | defer | 9 | dhar174 |
| [#125](https://github.com/dhar174/tiny_village/issues/125) | defer | 9 | dhar174 |
| [#126](https://github.com/dhar174/tiny_village/issues/126) | implement | 8 | dhar174 |
| [#128](https://github.com/dhar174/tiny_village/issues/128) | implement | 8 | dhar174 |
| [#129](https://github.com/dhar174/tiny_village/issues/129) | defer | 9 | dhar174 |
| [#130](https://github.com/dhar174/tiny_village/issues/130) | implement | 8 | dhar174 |
| [#131](https://github.com/dhar174/tiny_village/issues/131) | defer | 9 | dhar174 |
| [#132](https://github.com/dhar174/tiny_village/issues/132) | verify | 8 | dhar174 |
| [#381](https://github.com/dhar174/tiny_village/issues/381) | implement | 8 | dhar174 |
| [#383](https://github.com/dhar174/tiny_village/issues/383) | implement | 8 | dhar174 |
| [#507](https://github.com/dhar174/tiny_village/issues/507) | implement | 8 | dhar174 |
| [#508](https://github.com/dhar174/tiny_village/issues/508) | implement | 8 | dhar174 |

#### #123 — TODO: Implement modular UI system with panels (in _render_ui)

**Acceptance:** Verify real modular panels, focus, wrapping/scrolling and approved right-sidebar/bottom-drawer layout; existing legacy panels alone do not complete the new specification.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current panels, inspection and player feedback code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #123.

#### #124 — TODO: Add character relationship visualization (in _render_ui)

**Acceptance:** Display actual graph-backed relationships with meaningful labels and accessible selection; no duplicated or invented relationship state.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader panels, inspection and player feedback requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #124.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #125 — TODO: Add village statistics dashboard (in _render_ui)

**Acceptance:** Display documented village metrics sourced from real simulation records, with units and unavailable states; keep the MVP HUD distinct from a full dashboard.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader panels, inspection and player feedback requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #125.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #126 — TODO: Add interactive building information panels (in _render_ui)

**Acceptance:** Inspect Building objects and dictionaries with correct type-specific properties, readable wrapping and consistent keyboard/click selection.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the panels, inspection and player feedback acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #126.

#### #128 — TODO: Add save/load game functionality UI (in _render_ui)

**Acceptance:** Present named saves/load dialogs, quicksave, overwrite confirmation and unsaved-progress protection; backend round-trip correctness remains a prerequisite.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the panels, inspection and player feedback acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #128.

#### #129 — TODO: Add settings and configuration panels (in _render_ui)

**Acceptance:** Define which settings are safe in-session; validate controls, persistence and defaults without conflicting with approved session controls.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader panels, inspection and player feedback requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #129.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #130 — TODO: Add help and tutorial overlays (in _render_ui)

**Acceptance:** Provide the dismissible launch hint, always-available Help and visible shortcut labels; verify focus and Escape dismissal behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the panels, inspection and player feedback acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #130.

#### #131 — TODO: Add drag-and-drop interaction hints (in _render_ui)

**Acceptance:** Specify supported drag/drop actions and display valid/invalid placement without stealing unrelated input; retain cancellation and state integrity.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader panels, inspection and player feedback requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #131.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #132 — TODO: Add notification system for important events (in _render_ui)

**Acceptance:** Use the approved five-second non-modal status lane for routine confirmations, persistent actionable errors and retained event history, without stealing focus.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current panels, inspection and player feedback code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #123, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:4068, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #132.

#### #381 — Feature: Implement UI for Information Displays

**Acceptance:** Show selected Character name, needs, goal, inventory and persistent time/speed/population controls using the approved readable layout and real state.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the panels, inspection and player feedback acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_map_controller.py:31; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #381.

#### #383 — Feature: Implement Core Game Menus (Main Menu, Save/Load)

**Acceptance:** Implement New Game/Load/Quit and actual save/load backend plus UI, preserving named saves, dirty-progress protection, paused load and approved Escape/focus behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the panels, inspection and player feedback acceptance as ordered implementation/integration work in Phase 8; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #383.

#### #507 — Hard-coded screen dimensions (800x600) should be configurable parameters or obtained from the actual screen/surface dimensions to support different screen sizes.

**Acceptance:** Clamp InfoPanel to actual Surface dimensions at 1024×768 and 1280×720, preserving readable wrapping rather than the current hardcoded 800×600 bounds.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. The inline finding concerns InfoPanel.show; current lines 36–37 still clamp to 800×600. #508 targets a different class/method, so matching titles are not duplicate evidence.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #507.

#### #508 — Hard-coded screen dimensions (800x600) should be configurable parameters or obtained from the actual screen/surface dimensions to support different screen sizes.

**Acceptance:** Clamp ContextMenu to actual Surface dimensions at both supported sizes, with correct keyboard/click targets; distinct from InfoPanel #507 despite matching titles.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. The inline finding concerns ContextMenu.show; current lines 123–124 still clamp to 800×600. Keep separate from InfoPanel #507.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #123; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #508.

### Prompts and final-product AI

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#158](https://github.com/dhar174/tiny_village/issues/158) | defer | 9 | dhar174 |
| [#161](https://github.com/dhar174/tiny_village/issues/161) | defer | 9 | dhar174 |
| [#164](https://github.com/dhar174/tiny_village/issues/164) | defer | 9 | dhar174 |
| [#168](https://github.com/dhar174/tiny_village/issues/168) | defer | 9 | dhar174 |
| [#206](https://github.com/dhar174/tiny_village/issues/206) | verify | 9 | dhar174 |
| [#207](https://github.com/dhar174/tiny_village/issues/207) | verify | 9 | dhar174 |
| [#208](https://github.com/dhar174/tiny_village/issues/208) | defer | 9 | dhar174 |
| [#209](https://github.com/dhar174/tiny_village/issues/209) | defer | 9 | dhar174 |
| [#256](https://github.com/dhar174/tiny_village/issues/256) | defer | 9 | dhar174 |
| [#260](https://github.com/dhar174/tiny_village/issues/260) | defer | 9 | dhar174 |
| [#262](https://github.com/dhar174/tiny_village/issues/262) | defer | 9 | dhar174 |
| [#263](https://github.com/dhar174/tiny_village/issues/263) | defer | 9 | dhar174 |
| [#264](https://github.com/dhar174/tiny_village/issues/264) | defer | 9 | dhar174 |
| [#284](https://github.com/dhar174/tiny_village/issues/284) | verify | 9 | dhar174 |
| [#583](https://github.com/dhar174/tiny_village/issues/583) | defer | 9 | dhar174 |
| [#584](https://github.com/dhar174/tiny_village/issues/584) | defer | 9 | dhar174 |
| [#585](https://github.com/dhar174/tiny_village/issues/585) | defer | 9 | dhar174 |
| [#586](https://github.com/dhar174/tiny_village/issues/586) | defer | 9 | dhar174 |

#### #158 — tiny_prompt_builder.py & LLM Integration

**Acceptance:** Provide dynamic goal/need/context prompts, diverse situations, robust parsed outputs and strategic provider routing through actual production interfaces; retain final-product AI scope.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:387, tiny_strategy_manager.py:95, tiny_brain_io.py:71; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #158.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #161 — Diverse Prompt Templates

**Acceptance:** Generate context-correct social/crisis/exploration prompts with dynamic valid actions and role formatting, including absent-context failures.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #158, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:2867, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #161.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #164 — Conceptual LLM_Interface

**Acceptance:** Trace prompt → provider → parser → executable action through one documented production interface, with explicit unavailable/fallback behavior.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #158, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:2190, tiny_strategy_manager.py:1, tiny_brain_io.py:71; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #164.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #168 — LLM Integratio

**Acceptance:** Document and test strategic LLM activation criteria and actual provider/fallback routing; AI remains required for the final product.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #165, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:95, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #168.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #206 — Dynamic Action Options in Prompts

**Acceptance:** Generate prompt options from currently eligible executable actions and verify state changes alter options; no invented or stale choices.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #205, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #206.

#### #207 — Robust Output Parsing

**Acceptance:** Parse valid and malformed actual model outputs into supported actions with clear fallback/unavailable behavior; no live inference is claimed by controlled parsers.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #205, canonical topic #158; native prerequisites #155; proposed execution prerequisites #768; handoff #583. blocked by native #155.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #207.

#### #208 — Strategic Invocation

**Acceptance:** Verify documented strategic invocation triggers select the provider or deterministic fallback and preserve requested versus available capability state.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #205, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:95, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #208.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #209 — Contextual Richness

**Acceptance:** Inject relevant actual memories, relationships and event/world context into bounded prompts; retain provenance, formatting and context limits.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #205, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #209.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #256 — Optimization and Performance Improvements for tiny_prompt_builder.py

**Acceptance:** Validate descriptor/need caching and invalidation, modular assembly, configurable prompt length and documented safe concurrency against meaningful prompt outputs and performance evidence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:965, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #256.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #260 — Testing, Validation, and CI Integration for tiny_prompt_builder.py

**Acceptance:** Meet the original 80-percent module coverage target, schema integration, documented fixtures and CI/coverage reporting; existing supported-profile CI is partial evidence only.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #585. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:23, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #260.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #262 — Comprehensive Documentation and Usage Examples for tiny_prompt_builder.py

**Acceptance:** Provide current PromptBuilder overview/quickstart, public docstrings, three real usage examples, documented contributor workflow and the specified docs-site pipeline; keep deployment requirements separate from a passing startup guide.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #262.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #263 — Integration and Extensibility Improvements for tiny_prompt_builder.py

**Acceptance:** Specify and validate extension components, model-specific prompt variants, debugging visibility and configuration-driven assembly after stable prompt/provider contracts.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #263.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #264 — Advanced Features for tiny_prompt_builder.py

**Acceptance:** Define and verify emotion-aware context, narrative continuity, system/user roles and configurable game-context constraints as scoped final-product AI increments.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (existing human assignee); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #264.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #284 — Consider using a logging framework instead of print statements for error reporting to allow better control over log outputs in production.

**Acceptance:** Check error-reporting branches use actionable logging with exception context and controllable levels; do not replace useful user-facing console output indiscriminately.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #284.

#### #583 — Implement and Refine Criteria for Strategic LLM Invocation in StrategyManager

**Acceptance:** Cover each documented strategic LLM trigger in production decision routing and representative demos/tests with updated code rationale and guides; no live inference is inferred from routing tests.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #163, canonical topic #158; native prerequisites #614, #207; proposed execution prerequisites #768; handoff #583. blocked by native #614,#207.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:95, tiny_brain_io.py:16; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #583.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #584 — Instrument and Document All LLM Calls with Context and Reason

**Acceptance:** Record character/context/trigger/goal/fallback reason for every strategic provider call and expose interpretable demo/log examples with accurate documentation.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #163, canonical topic #158; native prerequisites #583; proposed execution prerequisites #768; handoff #583. blocked by native #583.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:95, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #584.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #585 — Comprehensive Integration and Unit Tests for Strategic LLM Logic

**Acceptance:** Exercise every strategic route with meaningful prompt/provider/parser/action assertions and separately retained real download/model/inference evidence; controlled external boundaries do not establish live readiness.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #163, canonical topic #158; native prerequisites #584; proposed execution prerequisites #768; handoff #583. blocked by native #584.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #585.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #586 — Update Documentation to Explain LLM Strategic Invocation and Codepoints

**Acceptance:** Link current criteria, call sites, operational logs, supported configuration, actual scenario tests and failure behavior in the LLM reference/usage guides.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #163, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #583. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #586.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Quests and progression

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#213](https://github.com/dhar174/tiny_village/issues/213) | defer | 9 | dhar174 |
| [#610](https://github.com/dhar174/tiny_village/issues/610) | defer | 9 | dhar174 |
| [#611](https://github.com/dhar174/tiny_village/issues/611) | defer | 9 | dhar174 |
| [#612](https://github.com/dhar174/tiny_village/issues/612) | defer | 9 | dhar174 |
| [#613](https://github.com/dhar174/tiny_village/issues/613) | defer | 9 | dhar174 |

#### #213 — Quest System

**Acceptance:** Generate contextual multi-step quests, progress/reward them through real events/state and make them inspectable; status labels do not prove this.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader quests and progression requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #213; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1743; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #213.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #610 — Generate context-aware quests

**Acceptance:** Generate valid quests from actual needs/world/resources and reject impossible contexts rather than assigning unrelated random templates.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader quests and progression requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #213, canonical topic #213; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #610.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #611 — Add multi-step quest objectives with meaningful rewards

**Acceptance:** Advance ordered quest stages through real state/events and award documented items/money/reputation/relationships once on valid completion.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader quests and progression requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #213, canonical topic #213; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #611.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #612 — Add quest UI

**Acceptance:** Inspect active/completed/progressing quests from actual state with readable navigation and no invented completion.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader quests and progression requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #213, canonical topic #213; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #612.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #613 — Integrate quests with events

**Acceptance:** Trigger quests from supported events and emit milestone/completion events without loops or repeated rewards.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader quests and progression requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #213, canonical topic #213; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #613.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Relationships and social simulation

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#212](https://github.com/dhar174/tiny_village/issues/212) | defer | 9 | dhar174 |
| [#606](https://github.com/dhar174/tiny_village/issues/606) | defer | 9 | dhar174 |
| [#607](https://github.com/dhar174/tiny_village/issues/607) | defer | 9 | dhar174 |

#### #212 — Social Network System

**Acceptance:** Demonstrate meaningful relationship dynamics and decision influence from one graph-backed authority; #609 exposes remaining graph read/update defects.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader relationships and social simulation requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #212; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #606. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:737, social_model.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #212.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #606 — Implement meaningful relationship dynamics

**Acceptance:** Apply actual social action outcomes to real relationship strengths with documented ranges and asymmetric/symmetric policy.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader relationships and social simulation requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #212, canonical topic #212; native prerequisites #768; proposed execution prerequisites none; handoff #606. blocked by native #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:1, social_model.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #606.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #607 — Use relationships in decision-making

**Acceptance:** Show changing real relationships changes selected goals/actions/plans in otherwise comparable contexts; a prompt mentioning friendship is insufficient.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader relationships and social simulation requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #212, canonical topic #212; native prerequisites #606; proposed execution prerequisites #768; handoff #606. blocked by native #606.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_graph_manager.py:1, social_model.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #607.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Rendering and art integration

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#113](https://github.com/dhar174/tiny_village/issues/113) | defer | 9 | dhar174 |
| [#114](https://github.com/dhar174/tiny_village/issues/114) | defer | 9 | dhar174 |
| [#115](https://github.com/dhar174/tiny_village/issues/115) | defer | 9 | dhar174 |
| [#116](https://github.com/dhar174/tiny_village/issues/116) | defer | 9 | dhar174 |
| [#117](https://github.com/dhar174/tiny_village/issues/117) | defer | 9 | dhar174 |
| [#118](https://github.com/dhar174/tiny_village/issues/118) | defer | 9 | dhar174 |
| [#119](https://github.com/dhar174/tiny_village/issues/119) | defer | 9 | dhar174 |
| [#120](https://github.com/dhar174/tiny_village/issues/120) | defer | 9 | dhar174 |
| [#121](https://github.com/dhar174/tiny_village/issues/121) | defer | 9 | dhar174 |
| [#122](https://github.com/dhar174/tiny_village/issues/122) | verify | 8 | dhar174 |
| [#594](https://github.com/dhar174/tiny_village/issues/594) | verify | 8 | dhar174 |
| [#598](https://github.com/dhar174/tiny_village/issues/598) | defer | 9 | dhar174 |
| [#599](https://github.com/dhar174/tiny_village/issues/599) | defer | 9 | dhar174 |

#### #113 — TODO: Add render quality settings (low, medium, high, ultra) (in render)

**Acceptance:** Define measured render-quality presets consistent with pixel-art scaling, verify configuration and compare supported performance budgets.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #762. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #113.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #114 — TODO: Add dynamic resolution scaling based on performance (in render)

**Acceptance:** Demonstrate adaptive resolution only if it preserves approved integer-scaled pixel art and readable independent HUD text; decide from measured frame evidence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #762. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #114.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #115 — TODO: Add anti-aliasing options (in render)

**Acceptance:** Scope antialiasing to approved elements; preserve crisp nearest-neighbor world art, measure performance and retain required visual review before enabling it.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites #177; proposed execution prerequisites #768; handoff #115. blocked by native #177.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #115.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #116 — TODO: Add post-processing effects (bloom, shadows, etc.) (in render)

**Acceptance:** Define bounded post-processing that fits the approved warm pixel-art direction and compare enabled/disabled rendering against measured budgets.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #116.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #117 — TODO: Add level-of-detail (LOD) system for distant objects (in render)

**Acceptance:** Measure a need for distant-object LOD and demonstrate visual/state consistency across supported viewport sizes before introducing complexity.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #762. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #117.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #118 — TODO: Add particle effects system (in render)

**Acceptance:** Demonstrate a bounded particle pass with lifecycle cleanup, correct world/UI layering and measured load; retain approved restrained motion.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #118.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #119 — TODO: Add lighting and shadow system (in render)

**Acceptance:** Implement scoped lighting/shadow layers without obscuring selected villagers, violating pixel scaling or exceeding measured budgets.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #119.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #120 — TODO: Add weather and atmospheric effects (in render)

**Acceptance:** Render weather-specific world effects through the layer pipeline and prove they correspond to actual weather state; weather text alone is insufficient.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #120.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #121 — TODO: Add screenshot and video recording functionality (in render)

**Acceptance:** Capture the supported viewport/UI with a clear output result and failure reporting; validate recording resource cleanup and documented formats.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #121.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #122 — TODO: Add fallback rendering for when map fails (in render)

**Acceptance:** Induce a real map-rendering failure and verify a distinguishable safe fallback on a real Surface; report failure rather than silently claiming full rendering.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. Reconcile the historical claim against current rendering and art integration code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #761. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #122.

#### #594 — Rendering and Graphics Enhancements

**Acceptance:** Verify #594 is natively under #56, all #113–#122 are under #594, and #598/#599 are attached; this is organization only, not delivery of graphics features.

**Disposition/evidence:** verify; `verified_native_organization`. All ten #113–#122 native parents are #594; #598/#599 also have parent #594 and #594 has parent #56. Its organizational acceptance is already met, independently of feature implementation.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #761. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #594.

#### #598 — Add VR/AR rendering support

**Acceptance:** Select supported VR/AR use cases/devices and concrete extension contracts before implementation; preserve as deferred expansion without imposing it on the Pygame MVP.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #598.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #599 — Add render effect layers pipeline

**Acceptance:** Define an ordered world/effects/UI pipeline and demonstrate layer ordering and cleanup; broad lighting/post-processing is later, while #761 retains required local occluder fading.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader rendering and art integration requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #594, canonical topic #594; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_map_controller.py:44, tiny_gameplay_controller.py:89; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #599.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Save/load and state restoration

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#102](https://github.com/dhar174/tiny_village/issues/102) | implement | 7 | dhar174 |
| [#103](https://github.com/dhar174/tiny_village/issues/103) | implement | 7 | dhar174 |
| [#104](https://github.com/dhar174/tiny_village/issues/104) | implement | 7 | dhar174 |
| [#105](https://github.com/dhar174/tiny_village/issues/105) | implement | 7 | dhar174 |
| [#214](https://github.com/dhar174/tiny_village/issues/214) | implement | 7 | dhar174 |
| [#614](https://github.com/dhar174/tiny_village/issues/614) | defer | 9 | dhar174 |

#### #102 — TODO: Restore character states (requires more complex character system) (in load_game_state)

**Acceptance:** Restore complete supported Character attributes, inventory, goals and identities; compare continued execution with an uninterrupted run, loaded paused.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the save/load and state restoration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #214, canonical topic #214; native prerequisites #103; proposed execution prerequisites #753; handoff #102. blocked by native #103.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5066, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #102.

#### #103 — TODO: Restore map state and building conditions (in load_game_state)

**Acceptance:** Restore map/building/world entities by stable identity and validate ownership/resources rather than rebuilding default scenery.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the save/load and state restoration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #214, canonical topic #214; native prerequisites #755; proposed execution prerequisites #753; handoff #103. blocked by native #755.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5066, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #103.

#### #104 — TODO: Restore relationship networks (in load_game_state)

**Acceptance:** Restore actual graph-backed relationship edges and verify read/update behavior after load; no parallel social-state authority.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the save/load and state restoration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #214, canonical topic #214; native prerequisites #102; proposed execution prerequisites #753; handoff #104. blocked by native #102.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5066, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #104.

#### #105 — TODO: Restore economic state (in load_game_state)

**Acceptance:** Restore economic balances, inventories, production and trade bookkeeping without double applying transactions; validate continuation.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the save/load and state restoration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #214, canonical topic #214; native prerequisites #104; proposed execution prerequisites #753; handoff #105. blocked by native #104.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5066, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #105.

#### #214 — Save/Load System

**Acceptance:** Round-trip all supported world/Character/relationship/economic/event/memory/time/random state, with versioning, corruption handling and paused/Finished load behavior.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the save/load and state restoration acceptance as ordered implementation/integration work in Phase 7; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #214; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #755. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1743, tiny_memories.py:614; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #214.

#### #614 — Persist memory subsystem state during save/load

**Acceptance:** Persist or rebuild supported semantic memory/index state with faithful recall after load; #756 separately owns basic record persistence and is not a duplicate.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Full semantic/index persistence remains Phase 9 under existing dependencies. Basic memory/pending events/time/randomness have separate Phase 7 acceptance in #756; retain both.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #214, canonical topic #214; native prerequisites #11; proposed execution prerequisites #768; handoff #756. blocked by native #11.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5015, tiny_memories.py:2521; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #614.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### State-preserving recovery

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#85](https://github.com/dhar174/tiny_village/issues/85) | implement | 5 | dhar174 |
| [#86](https://github.com/dhar174/tiny_village/issues/86) | implement | 5 | dhar174 |
| [#87](https://github.com/dhar174/tiny_village/issues/87) | implement | 5 | dhar174 |
| [#88](https://github.com/dhar174/tiny_village/issues/88) | implement | 5 | dhar174 |
| [#89](https://github.com/dhar174/tiny_village/issues/89) | implement | 5 | dhar174 |
| [#90](https://github.com/dhar174/tiny_village/issues/90) | implement | 5 | dhar174 |
| [#91](https://github.com/dhar174/tiny_village/issues/91) | implement | 5 | dhar174 |
| [#92](https://github.com/dhar174/tiny_village/issues/92) | implement | 5 | dhar174 |
| [#137](https://github.com/dhar174/tiny_village/issues/137) | implement | 5 | dhar174 |

#### #85 — TODO: Add detailed memory usage monitoring (in get_system_health_report)

**Acceptance:** Expose actual memory observations with units/source and explicit unknown values; demonstrate growth/cleanup without substituting guessed resource values.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #85.

#### #86 — TODO: Add performance metrics collection (in get_system_health_report)

**Acceptance:** Expose measured runtime performance in health reports with collection state, sample count and meaningful units; startup success is not sustained-load evidence.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #86.

#### #87 — TODO: Add predictive failure detection (in get_system_health_report)

**Acceptance:** Define predictive failure signals using actual retained history; demonstrate alerts and false-positive behavior without presenting guesses as established failures.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #87.

#### #88 — TODO: Add automated system recovery recommendations (in get_system_health_report)

**Acceptance:** Generate recovery recommendations from identified failures/dependencies and show why each action is safe for the affected state.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #88.

#### #89 — TODO: Add intelligent recovery strategies for different failure types (in attempt_system_recovery)

**Acceptance:** Choose recovery by failure class and prove preserved world/ownership; never replace a failed real Character with a success-shaped substitute.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #89.

#### #90 — TODO: Add system dependency analysis and recovery ordering (in attempt_system_recovery)

**Acceptance:** Respect subsystem initialization/ownership prerequisites during recovery and prove recovery ordering cannot create duplicate authoritative graphs.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #90.

#### #91 — TODO: Add partial recovery support (recover what can be recovered) (in attempt_system_recovery)

**Acceptance:** Recover only trusted components without losing world state; a state-integrity failure stops advancement until an explicitly safe recovery completes.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #91.

#### #92 — TODO: Add user notification system for recovery attempts (in attempt_system_recovery)

**Acceptance:** Show recovery attempt, outcome and actionable failure without stealing focus or hiding an unresolved error; retain the event-history record.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #137, canonical topic #137; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #137. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #92.

#### #137 — Error Handling and Recovery: Key area

**Acceptance:** Demonstrate failure-appropriate recovery preserves authoritative state; fatal/state-integrity failures stop advancement and offer only safe actions.

**Disposition/evidence:** implement; `source_reviewed_not_executed`. Retain the state-preserving recovery acceptance as ordered implementation/integration work in Phase 5; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #56, canonical topic #137; native prerequisites #746; proposed execution prerequisites #743; handoff #137. blocked by native #746.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1544; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #137.

### Storytelling and narrative

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#491](https://github.com/dhar174/tiny_village/issues/491) | verify | 9 | dhar174 |
| [#492](https://github.com/dhar174/tiny_village/issues/492) | defer | 9 | dhar174 |
| [#499](https://github.com/dhar174/tiny_village/issues/499) | defer | 9 | dhar174 |
| [#500](https://github.com/dhar174/tiny_village/issues/500) | defer | 9 | dhar174 |

#### #491 — References arc.character_roles but this attribute is not defined in the StoryArc class. Additionally, CharacterRole enum is referenced but not defined anywhere in the module.

**Acceptance:** Exercise StoryArc character-role lookup and supported role values with real story objects; current character_roles declaration is static evidence, not proof of arc behavior.

**Disposition/evidence:** verify; `source_reviewed_not_executed`. StoryArc now declares character_roles and the lookup uses string defaults. The old missing-field/enum finding is a verification candidate; meaningful arc behavior was not executed here.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #112; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_storytelling_system.py:59; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #491.

#### #492 — Extensive mocking of the graph_manager and its methods may not accurately test the integration. Consider using a more realistic test setup or integration test that uses actual implementations where possible.

**Acceptance:** Drive story integration with actual graph/events/story objects and assert narrative progress corresponds to real state; no permissive graph mock may supply the result.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader storytelling and narrative requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #112; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect test_storytelling_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #492.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #499 — Creating a mock for _current_narrative_context may not accurately test the importance scaling functionality. Consider using actual NarrativeImpact values or testing the scaling logic more directly to ensure it fails when the function doesn't work as expected.

**Acceptance:** Use real NarrativeImpact values and actual current narrative context to assert importance scaling at relevant boundaries, not a canned context mock.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader storytelling and narrative requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #112; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_story_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #499.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #500 — Using Mock() for story arc elements may not properly test arc progression. The test should use real element objects or at minimum verify that the arc progression logic actually evaluates element counts correctly, rather than mocking the elements themselves.

**Acceptance:** Advance a real arc using actual story elements; assert element-count/progression conditions and failure/non-progress cases rather than accepting any Mock element.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader storytelling and narrative requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #112; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_story_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #500.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

### Weather and environment

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#211](https://github.com/dhar174/tiny_village/issues/211) | defer | 9 | dhar174 |
| [#482](https://github.com/dhar174/tiny_village/issues/482) | verify | 8 | dhar174 |
| [#603](https://github.com/dhar174/tiny_village/issues/603) | defer | 9 | dhar174 |
| [#604](https://github.com/dhar174/tiny_village/issues/604) | defer | 9 | dhar174 |
| [#605](https://github.com/dhar174/tiny_village/issues/605) | defer | 9 | dhar174 |

#### #211 — Weather System

**Acceptance:** Verify behavior/world impact, presentation and seasonal progression separately; weather text/rendering does not complete weather simulation.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader weather and environment requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #138, canonical topic #211; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1743, tiny_event_handler.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #211.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #482 — WEATHER_UI_MESSAGES is referenced but not defined or imported in this file. This will cause a NameError at runtime.

**Acceptance:** Render the rainy-weather panel with real Pygame/fonts/Surface; WEATHER_UI_MESSAGES resolves and observable panel pixels are drawn.

**Disposition/evidence:** verify; `passed_scoped_probe`. Real SDL-dummy rainy panel rendering changes actual Surface pixels and exits zero. This proves the reported missing-constant path only, not weather behavior/audio.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #211; native prerequisites none in snapshot; proposed execution prerequisites none; handoff #761. await human review.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #482.

#### #603 — Make weather affect character decisions and world state

**Acceptance:** Show weather changes character decisions/actions, event weather checks and authoritative world state consistently; rainy-panel success #482 does not supply this evidence.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader weather and environment requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #211, canonical topic #211; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317, tiny_event_handler.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #603.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #604 — Add weather visual/audio presentation

**Acceptance:** Render weather-specific visual/audio cues from actual weather state through supported presentation boundaries; coordinate audio with #616.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader weather and environment requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #211, canonical topic #211; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317, tiny_event_handler.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #604.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #605 — Implement seasonal weather progression

**Acceptance:** Advance seasons deterministically and verify temperature/weather probabilities and their documented gameplay effects over supported time progression.

**Disposition/evidence:** defer; `source_reviewed_not_executed`. Retain this broader weather and environment requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #211, canonical topic #211; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #770. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:5317, tiny_event_handler.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #605.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

## Later open issues

### Actions and GOAP contracts

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#739](https://github.com/dhar174/tiny_village/issues/739) | defer | 4 | dhar174 |
| [#740](https://github.com/dhar174/tiny_village/issues/740) | defer | 4 | dhar174 |
| [#741](https://github.com/dhar174/tiny_village/issues/741) | defer | 4 | dhar174 |
| [#742](https://github.com/dhar174/tiny_village/issues/742) | verify | 4 | dhar174 |
| [#743](https://github.com/dhar174/tiny_village/issues/743) | verify | 4 | dhar174 |

#### #739 — [E4] Phase 4: Align actions, world state, and GOAP

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H4 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader actions and goap contracts requirement for Phase 4; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #182; native prerequisites none in snapshot; proposed execution prerequisites #738; handoff #739. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:740, tiny_goap_system.py:63, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #739.

**Revisit:** After the recorded Phase 4 prerequisites and explicit scope decision, retaining current acceptance.

#### #740 — [P4-02] Document and reconcile need/resource scales at runtime boundaries

**Acceptance:** Choose documented units per canonical field and explicit conversions at planner/prompt boundaries.; Validate clamps, thresholds, costs, and target effects on real scenarios.; Record representative before/after states so apparent action success cannot hide unchanged or incorrectly clamped values.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader actions and goap contracts requirement for Phase 4; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #739, canonical topic #182; native prerequisites #220; proposed execution prerequisites #738; handoff #740. blocked by native #220.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #740.

**Revisit:** After the recorded Phase 4 prerequisites and explicit scope decision, retaining current acceptance.

#### #741 — [P4-04] Execute active plans across simulation ticks and replan on failure

**Acceptance:** Track a per-character active plan/cursor and execute the intended next step at each tick.; Invalidate or replan when preconditions/world state change and hand failures back to StrategyManager.; Assert end-to-end actor identity, plan progress, goal achievement, and consistent graph state.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader actions and goap contracts requirement for Phase 4; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #739, canonical topic #182; native prerequisites #218; proposed execution prerequisites #738; handoff #741. blocked by native #218.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:740, tiny_goap_system.py:63, tiny_strategy_manager.py:95; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #741.

**Revisit:** After the recorded Phase 4 prerequisites and explicit scope decision, retaining current acceptance.

#### #742 — [P4-06] Human contract review: actions, state units, planning, and execution

**Acceptance:** Inspect actual before/after character and graph state, not only mock call counts.; Review identity/units/preconditions/effects, cache validity, bounded failures, and plan continuation.; Record an explicit approval or linked rework before the event-integration phase.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current actions and goap contracts code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #739, canonical topic #182; native prerequisites #602; proposed execution prerequisites #738; handoff #742. blocked by native #602.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:126, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #742.

#### #743 — [H4] Human decision gate 4: Align actions, world state, and GOAP

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current actions and goap contracts code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #739, canonical topic #182; native prerequisites #742; proposed execution prerequisites #738; handoff #743. blocked by native #742.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect actions.py:1, tiny_goap_system.py:1, tiny_strategy_manager.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #743.

### Behavioral test quality

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#773](https://github.com/dhar174/tiny_village/issues/773) | defer | 4 | dhar174 |
| [#774](https://github.com/dhar174/tiny_village/issues/774) | defer | 4 | dhar174 |

#### #773 — Repair minimal integration test fixtures and verify fallback action behavior

**Acceptance:** - Reproduce both failures with a bounded fresh process.
- Replace permissive fixture responses with realistic state/inventory data and retain meaningful behavioral assertions.
- Determine and fix any runtime defect that remains with those fixtures; do not catch/assert away failures.
- Complete all 11 integration tests, reporting executed, failed, errored, and skipped counts.

Scheduling: follow-up to #718; triage with #734 and coordinate runtime changes with #218. This issue does not approve or advance the human review in #721.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader behavioral test quality requirement for Phase 4; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites #718; proposed execution prerequisites #738; handoff #773. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:1, tests/test_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #773.

**Revisit:** After the recorded Phase 4 prerequisites and explicit scope decision, retaining current acceptance.

#### #774 — Reconcile utility test Goal contracts and preserve falsy action IDs

**Acceptance:** - Document and reconcile the supported Goal interface used by utility evaluation and its fixtures.
- Exercise real production objects where feasible; keep explicitly targeted fallback coverage and avoid silently masking real-Goal incompatibilities.
- Preserve explicit falsy action IDs in the fixture.
- Run all 26 utility tests in a fresh bounded subprocess and report actual counts; distinguish import-order and constructor compatibility from the #718 handler fix.

Scheduling: follow-up to #718; triage with #734 and coordinate goal/utility contracts with #740 and #218. Preserve the existing phase gates and review requirements.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader behavioral test quality requirement for Phase 4; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #683; native prerequisites #718; proposed execution prerequisites #738; handoff #774. scheduled after #738.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tests/test_goap_planner_integration_real.py:13, tests/test_gameplay_controller.py:316; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #774.

**Revisit:** After the recorded Phase 4 prerequisites and explicit scope decision, retaining current acceptance.

### Canonical scenario and release evidence

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#716](https://github.com/dhar174/tiny_village/issues/716) | implement | 3 | dhar174 |
| [#760](https://github.com/dhar174/tiny_village/issues/760) | defer | 8 | dhar174 |
| [#761](https://github.com/dhar174/tiny_village/issues/761) | defer | 8 | dhar174 |
| [#762](https://github.com/dhar174/tiny_village/issues/762) | defer | 8 | dhar174 |
| [#763](https://github.com/dhar174/tiny_village/issues/763) | verify | 8 | dhar174 |
| [#764](https://github.com/dhar174/tiny_village/issues/764) | verify | 8 | dhar174 |
| [#765](https://github.com/dhar174/tiny_village/issues/765) | verify | 8 | dhar174 |
| [#766](https://github.com/dhar174/tiny_village/issues/766) | verify | 8 | dhar174 |
| [#767](https://github.com/dhar174/tiny_village/issues/767) | verify | 8 | dhar174 |
| [#768](https://github.com/dhar174/tiny_village/issues/768) | verify | 8 | dhar174 |

#### #716 — [PROGRAM] Tiny Village: ordered MVP delivery and human review program

**Acceptance:** Every work item and review has a repository issue and Project row.; Phase, Sequence, Readiness, Scope, Depends on, Review owner, and Decision communicate execution order.; A phase cannot advance until its human gate records Approved; no automated process approves a gate.; MVP release is Phase 8; Phase 9 and optional tasks do not block the Phase 8 release.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the canonical scenario and release evidence acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #658; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #716. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #716.

#### #760 — [E8] Phase 8: Validate UX, UI, aesthetics, and MVP release

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H8 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader canonical scenario and release evidence requirement for Phase 8; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #658; native prerequisites none in snapshot; proposed execution prerequisites #759; handoff #760. scheduled after #759.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #760.

**Revisit:** After the recorded Phase 8 prerequisites and explicit scope decision, retaining current acceptance.

#### #761 — [P8-04] Implement the approved MVP UI and UX specification

**Acceptance:** Implement agreed HUD/inspection/notification/control and save/load feedback behavior.; Provide legible selection, focus, paused/error/recovery states and supported-resolution handling.; Review PR #710 for graceful map-render failure feedback; track discovered UI fixes as separate linked issues.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader canonical scenario and release evidence requirement for Phase 8; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #655; proposed execution prerequisites #759; handoff #761. blocked by native #655.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #761.

**Revisit:** After the recorded Phase 8 prerequisites and explicit scope decision, retaining current acceptance.

#### #762 — [P8-09] Measure MVP stability, frame times, memory, and failure recovery

**Acceptance:** Run the fixed scenario, then a bounded multi-villager soak using the Phase 2 budgets.; Record frame/decision timings, memory growth, crashes, stuck plans, and recovery behavior.; Create issues for actionable findings; verify #596/#597 from evidence instead of duplicating implemented instrumentation.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader canonical scenario and release evidence requirement for Phase 8; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #761; proposed execution prerequisites #759; handoff #762. blocked by native #761.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #762.

**Revisit:** After the recorded Phase 8 prerequisites and explicit scope decision, retaining current acceptance.

#### #763 — [P8-05] Human aesthetic review of the village, HUD, and visual states

**Acceptance:** Review palette, typography, spacing, hierarchy, map/character legibility, and selection/error visual language.; Check supported resolutions/backgrounds and asset consistency/provenance.; Record approved visual decisions or individual remediation issues with examples.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #761; proposed execution prerequisites #759; handoff #763. blocked by native #761.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #763.

#### #764 — [P8-06] Human UX usability walkthrough of all supported MVP journeys

**Acceptance:** Walk through launch, observe, inspect, pause, save/load, recover, and exit.; Check discoverability, feedback timing, narrative clarity, and meaningful user control.; Record observations, friction, approved decisions, and issue-backed fixes.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #761; proposed execution prerequisites #759; handoff #764. blocked by native #761.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #764.

#### #765 — [P8-07] Human UI interaction and state consistency review

**Acceptance:** Verify panels, controls, labels, state transitions, and unavailable/error states.; Ensure displayed needs, inventory, goals, and save/load outcomes match authoritative state.; Review resolution/layout behavior and record required fixes separately.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #761; proposed execution prerequisites #759; handoff #765. blocked by native #761.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #765.

#### #766 — [P8-08] Human accessibility and inclusive-interaction review

**Acceptance:** Check text/selection contrast, non-color status cues, readable sizes, keyboard operation, and focus visibility.; Review configurable controls, motion settings where applicable, and actionable error/recovery feedback.; Document platform constraints honestly; do not claim browser/ARIA support for a Pygame interface.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #761; proposed execution prerequisites #759; handoff #766. blocked by native #761.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #766.

#### #767 — [P8-10] Human modern-practices and release-readiness review

**Acceptance:** Review cold startup, state contracts, behavioral test results, bounds/caches, configuration, and error handling.; Confirm design/UX/UI/accessibility decisions and performance results are recorded.; Verify reproducible commands, clean changes, optional-feature boundaries, and any remaining release blockers.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #762, #766, #765, #764, #763; proposed execution prerequisites #759; handoff #767. blocked by native #762,#766,#765,#764,#763.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #767.

#### #768 — [H8] Human decision gate 8: Validate UX, UI, aesthetics, and MVP release

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current canonical scenario and release evidence code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #760, canonical topic #658; native prerequisites #767; proposed execution prerequisites #759; handoff #768. blocked by native #767.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect main.py:1, demo_minimal_integration.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #768.

### Contributor tooling and experiments

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#733](https://github.com/dhar174/tiny_village/issues/733) | implement | 3 | dhar174 |
| [#734](https://github.com/dhar174/tiny_village/issues/734) | implement | 3 | dhar174 |
| [#735](https://github.com/dhar174/tiny_village/issues/735) | implement | 3 | dhar174 |
| [#736](https://github.com/dhar174/tiny_village/issues/736) | implement | 3 | dhar174 |
| [#737](https://github.com/dhar174/tiny_village/issues/737) | implement | 3 | dhar174 |
| [#738](https://github.com/dhar174/tiny_village/issues/738) | verify | 3 | dhar174 |

#### #733 — [E3] Phase 3: Consolidate backlog and pull requests

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H3 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the contributor tooling and experiments acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #14; native prerequisites none in snapshot; proposed execution prerequisites #732; handoff #733. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #733.

#### #734 — [P3-01] Triage the original issue backlog into canonical subsystem epics

**Acceptance:** Give each original item an owner, reproduction/acceptance criteria, parent topic, prerequisite, and implement/verify/duplicate/defer disposition.; Recheck apparently implemented #482/#355/#602/#596/#597/#609; do not close #602 while its regression checks are blocked.; Preserve useful historical context and require human-approved evidence before duplicate closures.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the contributor tooling and experiments acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #733, canonical topic #14; native prerequisites #732; proposed execution prerequisites none; handoff #734. ready for scoped investigation; publication review pending.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #734.

#### #735 — [P3-02] Consolidate duplicate utility changes in PRs #703 and #714

**Acceptance:** Avoid creating two competing tests/test_tiny_util_funcs.py files.; Confirm tweener, numeric conversion, and clamping expectations reflect documented semantics.; Rebase the selected change and close/supersede the duplicate only after a maintainer decision.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the contributor tooling and experiments acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #733, canonical topic #14; native prerequisites #734; proposed execution prerequisites #732; handoff #735. blocked by native #734.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #735.

#### #736 — [P3-03] Reconcile older functional-test PRs with current test locations

**Acceptance:** Review #420/#417/#408/#406 and #404/#397/#395/#391 plus #393 against current main and merged #349.; Salvage tests that validate real behavior, actual goal completion, and graph contracts; avoid copied production logic in mocks.; Document each PR's keep/rebase/split/supersede decision and targeted validation evidence.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the contributor tooling and experiments acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #733, canonical topic #14; native prerequisites #734; proposed execution prerequisites #732; handoff #736. blocked by native #734.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #736.

#### #737 — [P3-04] Reconcile PR ownership, merge sequence, and dependency configuration

**Acceptance:** Assign a reviewer and owner to each selected implementation PR and verify current CI/mergeability.; Rebase each dependent PR after its prerequisite lands; retain existing native parents and dependency links.; Update Phase/Sequence/Readiness/Depends on and create a distinct issue for every newly discovered remediation.

**Disposition/evidence:** implement; `existing_roadmap_or_remediation_acceptance_preserved`. Retain the contributor tooling and experiments acceptance as ordered implementation/integration work in Phase 3; source presence alone does not prove its acceptance.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #733, canonical topic #14; native prerequisites #736, #735; proposed execution prerequisites #732; handoff #737. blocked by native #736,#735.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #737.

#### #738 — [H3] Human decision gate 3: Consolidate backlog and pull requests

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current contributor tooling and experiments code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #733, canonical topic #14; native prerequisites #737; proposed execution prerequisites #732; handoff #738. blocked by native #737.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect AGENTS.md:1, .github/workflows/python-ci.yml:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #738.

### Events and consequences

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#744](https://github.com/dhar174/tiny_village/issues/744) | defer | 5 | dhar174 |
| [#745](https://github.com/dhar174/tiny_village/issues/745) | defer | 5 | dhar174 |
| [#746](https://github.com/dhar174/tiny_village/issues/746) | defer | 5 | dhar174 |
| [#747](https://github.com/dhar174/tiny_village/issues/747) | verify | 5 | dhar174 |
| [#748](https://github.com/dhar174/tiny_village/issues/748) | verify | 5 | dhar174 |
| [#786](https://github.com/dhar174/tiny_village/issues/786) | defer | 5 | dhar174 |

#### #744 — [E5] Phase 5: Complete event-driven execution

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H5 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader events and consequences requirement for Phase 5; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #744. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #744.

**Revisit:** After the recorded Phase 5 prerequisites and explicit scope decision, retaining current acceptance.

#### #745 — [P5-02] Validate event payloads, prioritization, consequences, and scheduling

**Acceptance:** Typed and event-like payloads trigger deterministic, bounded processing.; Priority and consequence dispatch preserve actor/location context and required order.; No duplicate handling, unbounded cascades, or unexpected cross-character effects occur.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader events and consequences requirement for Phase 5; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #744, canonical topic #51; native prerequisites #219; proposed execution prerequisites #743; handoff #745. blocked by native #219.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #745.

**Revisit:** After the recorded Phase 5 prerequisites and explicit scope decision, retaining current acceptance.

#### #746 — [P5-03] Record event/action outcomes in memory and accounting once

**Acceptance:** Count and remember each successful action once across daily and event-driven routes.; Failed actions produce accurate failure evidence without successful reward/memory accounting.; A trace connects event, actor, selected plan, applied effect, and resulting state.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader events and consequences requirement for Phase 5; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #744, canonical topic #51; native prerequisites #745; proposed execution prerequisites #743; handoff #746. blocked by native #745.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #746.

**Revisit:** After the recorded Phase 5 prerequisites and explicit scope decision, retaining current acceptance.

#### #747 — [P5-05] Human walkthrough: event-driven decisions and understandable feedback

**Acceptance:** Walk through success, invalid input, actor-specific failure, and recovery.; Confirm understandable event/decision/action feedback without exposing irrelevant implementation details.; Record approval or issue-backed remediation before economic integration.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current events and consequences code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #744, canonical topic #51; native prerequisites #137; proposed execution prerequisites #743; handoff #747. blocked by native #137.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #747.

#### #748 — [H5] Human decision gate 5: Complete event-driven execution

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current events and consequences code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #744, canonical topic #51; native prerequisites #747; proposed execution prerequisites #743; handoff #748. blocked by native #747.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:1, tiny_gameplay_controller.py:1970; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #748.

#### #786 — Event graph registration fails for dictionary-valued impact

**Acceptance:** - Decide the supported impact shapes and event identity/equality contract before changing hashing.
- If dictionary impacts are supported, make registration, lookup, updates and removal work without mutable-hash corruption.
- Add real graph regressions for supported impact types and updates.
- Preserve existing event recurrence and participant behavior.

#654 uses the already supported neutral numeric impact `0` for its startup fixture. This separate defect is not repaired or hidden by that smoke check. Retained original observation: `minimal-green.log` in the #654 evidence directory. Related gameplay/scenario work: #751/#658.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader events and consequences requirement for Phase 5; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #51; native prerequisites none in snapshot; proposed execution prerequisites #743; handoff #786. scheduled after #743.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_event_handler.py:23, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #786.

**Revisit:** After the recorded Phase 5 prerequisites and explicit scope decision, retaining current acceptance.

### Items, inventory and economy

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#749](https://github.com/dhar174/tiny_village/issues/749) | defer | 6 | dhar174 |
| [#750](https://github.com/dhar174/tiny_village/issues/750) | defer | 6 | dhar174 |
| [#751](https://github.com/dhar174/tiny_village/issues/751) | defer | 6 | dhar174 |
| [#752](https://github.com/dhar174/tiny_village/issues/752) | verify | 6 | dhar174 |
| [#753](https://github.com/dhar174/tiny_village/issues/753) | verify | 6 | dhar174 |
| [#775](https://github.com/dhar174/tiny_village/issues/775) | defer | 6 | dhar174 |

#### #749 — [E6] Phase 6: Deliver the village economic loop

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H6 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader items, inventory and economy requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #194; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #749. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #749.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

#### #750 — [P6-05] Expose inventory needs and trade/drop choices to decisions

**Acceptance:** Review PR #693 against the chosen inventory and economic APIs.; Unavailable items/actions are not offered as valid choices.; Decision and prompt contexts show consistent quantities and useful choices without duplicate aliases.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader items, inventory and economy requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #749, canonical topic #194; native prerequisites #198; proposed execution prerequisites #748; handoff #750. blocked by native #198.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #750.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

#### #751 — [P6-07] Demonstrate the full work-to-food-to-recovery scenario

**Acceptance:** Goals drive work, wages, food acquisition, eating, and recovery rather than name-based fake wealth increments.; Different needs/scarcity produce explainable changes in chosen actions.; Verify actor/graph/inventory state and conserved totals over multiple ticks.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader items, inventory and economy requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #749, canonical topic #194; native prerequisites #750; proposed execution prerequisites #748; handoff #751. blocked by native #750.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #751.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

#### #752 — [P6-08] Human review: economic behavior, balance, and data integrity

**Acceptance:** Walk through success, insufficient money, unavailable goods, and failed work/trade.; Inspect wage/item conservation, meaningful need changes, and the scope decision for ownership.; Record approved tuning/behavior or create specific remediation issues.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current items, inventory and economy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #749, canonical topic #194; native prerequisites #751; proposed execution prerequisites #748; handoff #752. blocked by native #751.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #752.

#### #753 — [H6] Human decision gate 6: Deliver the village economic loop

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current items, inventory and economy code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #749, canonical topic #194; native prerequisites #752; proposed execution prerequisites #748; handoff #753. blocked by native #752.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:1, tiny_gameplay_controller.py:1970; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #753.

#### #775 — Define the missing Open Door effects and validate production construction

**Acceptance:** - Agree on the intended Open Door behavior and register a complete supported effect definition.
- Construct a production Door without patching effect_dict and verify it uses the supplied action system/graph.
- Execute the real action against a valid target/state and assert the intended state and authoritative graph update.
- Retain missing/invalid-definition failure coverage; do not turn Open into an undocumented no-op.

During #719, ownership alone is tested with explicit effect-definition fixture data and real Door/Action/System/GraphManager objects. That test does not claim the default Door constructor works.

Scheduling: child of #195, Phase 6; depends on #719 and human phase gate #748. Preserve the existing parent/task review requirements. No human phase decision is supplied by this issue.

Observed on runtime revision cc53105a8250485639c13e1f49abddbe77d5de7e; the missing definition also exists in main 610b4f871694a380f3fedfd48664c188a169cbee.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader items, inventory and economy requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #195, canonical topic #194; native prerequisites #748, #719; proposed execution prerequisites none; handoff #775. blocked by native #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_items.py:553, tiny_gameplay_controller.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #775.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

### Jobs and economic production

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#787](https://github.com/dhar174/tiny_village/issues/787) | defer | 6 | dhar174 |

#### #787 — Investigate job catalog count and missing-role diagnostics during real startup

**Acceptance:** - Reconcile expected top-level roles, nested names and any intentional aliases in `job_roles.json`.
- Correct parser/validation diagnostics to distinguish invalid records from aliases consistently.
- Add focused coverage for the four reported role names and valid existing jobs.
- Demonstrate that real Character construction preserves supported jobs and no longer reports spurious errors; retain explicit errors for genuinely invalid inputs.

Related job-system roadmap task: #199. #654 changes unsupported sample jobs to existing catalog roles and makes invalid Character jobs raise rather than exiting the process; catalog-wide reconciliation is outside that dependency/runtime scope. Original logs are retained in its validation evidence.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader jobs and economic production requirement for Phase 6; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent none, canonical topic #199; native prerequisites none in snapshot; proposed execution prerequisites #748; handoff #787. scheduled after #748.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_jobs.py:162, job_roles.json:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #787.

**Revisit:** After the recorded Phase 6 prerequisites and explicit scope decision, retaining current acceptance.

### Prompts and final-product AI

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#769](https://github.com/dhar174/tiny_village/issues/769) | defer | 9 | dhar174 |
| [#770](https://github.com/dhar174/tiny_village/issues/770) | verify | 9 | dhar174 |
| [#771](https://github.com/dhar174/tiny_village/issues/771) | verify | 9 | dhar174 |
| [#772](https://github.com/dhar174/tiny_village/issues/772) | verify | 9 | dhar174 |

#### #769 — [E9] Phase 9: Expand optional features deliberately

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H9 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader prompts and final-product ai requirement for Phase 9; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #158; native prerequisites none in snapshot; proposed execution prerequisites #768; handoff #769. scheduled after #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #769.

**Revisit:** After #768 plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.

#### #770 — [P9-12] Human disposition review of optional tooling and experimental PRs

**Acceptance:** Record keep/separate/defer/supersede decisions for #705/#707/#708/#592/#702/#700.; Treat AI Algorithm Mentor as optional automation with distinct permission/provider prerequisites, not required test CI.; Do not ship generated dummy fixtures, unrelated LangGraph tooling, or experimental sleep changes as MVP runtime work.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #769, canonical topic #158; native prerequisites #768; proposed execution prerequisites none; handoff #770. blocked by native #768.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #770.

#### #771 — [P9-13] Human review of optional feature behavior and operational readiness

**Acceptance:** Review social/memory/LLM fallbacks, persistence, instrumentation, budgets, and rendering correctness.; Run the unchanged supported MVP as a regression control.; Record remaining deferred increments and approval/rework decisions with evidence.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #769, canonical topic #158; native prerequisites #770, #115, #585; proposed execution prerequisites #768; handoff #771. blocked by native #770,#115,#585.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #771.

#### #772 — [H9] Human decision gate 9: Expand optional features deliberately

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current prompts and final-product ai code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #769, canonical topic #158; native prerequisites #771; proposed execution prerequisites #768; handoff #772. blocked by native #771.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_prompt_builder.py:1, tiny_strategy_manager.py:1, tiny_brain_io.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #772.

### Save/load and state restoration

| Issue | Disposition | Phase | Accountable owner |
|---|---|---|---|
| [#754](https://github.com/dhar174/tiny_village/issues/754) | defer | 7 | dhar174 |
| [#755](https://github.com/dhar174/tiny_village/issues/755) | defer | 7 | dhar174 |
| [#756](https://github.com/dhar174/tiny_village/issues/756) | defer | 7 | dhar174 |
| [#757](https://github.com/dhar174/tiny_village/issues/757) | defer | 7 | dhar174 |
| [#758](https://github.com/dhar174/tiny_village/issues/758) | verify | 7 | dhar174 |
| [#759](https://github.com/dhar174/tiny_village/issues/759) | verify | 7 | dhar174 |

#### #754 — [E7] Phase 7: Restore saved simulation state

**Acceptance:** All phase work is tracked as individual issues.; Phase gate H7 records the human decision and evidence.; Optional/deferred work is explicitly documented and does not silently become a release prerequisite.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader save/load and state restoration requirement for Phase 7; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #716, canonical topic #214; native prerequisites none in snapshot; proposed execution prerequisites #753; handoff #754. scheduled after #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #754.

**Revisit:** After the recorded Phase 7 prerequisites and explicit scope decision, retaining current acceptance.

#### #755 — [P7-01] Define versioned snapshots and atomic save behavior

**Acceptance:** Specify stable IDs, schema version, world/actor state, inventories, relationships, events, time, and supported memory data.; Validate input before replacing live state and write saves atomically.; Document incompatible/corrupt save handling and supported migrations.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader save/load and state restoration requirement for Phase 7; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #754, canonical topic #214; native prerequisites #753; proposed execution prerequisites none; handoff #755. blocked by native #753.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #755.

**Revisit:** After the recorded Phase 7 prerequisites and explicit scope decision, retaining current acceptance.

#### #756 — [P7-06] Persist basic memory records, pending events, time, and randomness

**Acceptance:** Restore supported plain memory records and pending event/plan state.; Preserve simulation time and deterministic RNG state needed by the approved scenario.; Document that full NLP/FAISS model/index persistence is addressed separately by #614 in Phase 9.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader save/load and state restoration requirement for Phase 7; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #754, canonical topic #214; native prerequisites #105; proposed execution prerequisites #753; handoff #756. blocked by native #105.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #756.

**Revisit:** After the recorded Phase 7 prerequisites and explicit scope decision, retaining current acceptance.

#### #757 — [P7-07] Validate restart/load continuation and corrupted saves

**Acceptance:** Save, restart, load, and continue produce equivalent state and subsequent behavior.; Missing/invalid/version-mismatched data fail clearly before mutating valid live state.; Verify restored actor/graph/inventory identities and no double application of events or rewards.

**Disposition/evidence:** defer; `existing_roadmap_or_remediation_acceptance_preserved`. Retain this broader save/load and state restoration requirement for Phase 7; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #754, canonical topic #214; native prerequisites #756; proposed execution prerequisites #753; handoff #757. blocked by native #756.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #757.

**Revisit:** After the recorded Phase 7 prerequisites and explicit scope decision, retaining current acceptance.

#### #758 — [P7-08] Human review: save/load correctness and recoverability

**Acceptance:** Review snapshot versions, actual restored values, identity consistency, and corruption behavior.; Inspect user-facing recovery instructions and ensure errors do not conceal partial restores.; Record approval or linked remediation before release-facing UI validation.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current save/load and state restoration code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #754, canonical topic #214; native prerequisites #757; proposed execution prerequisites #753; handoff #758. blocked by native #757.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #758.

#### #759 — [H7] Human decision gate 7: Restore saved simulation state

**Acceptance:** Review every required task, current PR head, commands/results, and the phase-specific evidence.; Walk through the supported behavior, failure cases, and applicable design/engineering artifacts.; Choose exactly Approved, Changes requested, or Deferred; record reviewer, date, rationale, evidence, and issue-backed remediation.; Only close this gate after a human records Approved; keep rework/deferred gates open and downstream work blocked.; Do not auto-approve based on a passing suite, a bot action, elapsed time, or a completed checklist.; Human reviewer: dhar174, or a maintainer explicitly delegated in the Project.; Check prerequisite issues and the previous phase's recorded decision.; Review artifacts, current commit/PR heads, executed checks, known failures, and proposed scope.; Walk through the phase's success/failure scenarios and relevant design/engineering evidence.; Record reviewer, date, evidence links, rationale, and exactly one decision: Approved / Changes requested / Deferred.; Create a separate issue for each required remediation and add it as a blocker.; Set Project Decision to the human's recorded decision. Close this issue only after Approved.

**Disposition/evidence:** verify; `existing_roadmap_or_remediation_acceptance_preserved`. Reconcile the historical claim against current save/load and state restoration code and evidence before proposing completion; unexecuted behavior remains unverified.

**Ownership/order:** dhar174 (ledger-only maintainer accountability; GitHub assignments preserved); native parent #754, canonical topic #214; native prerequisites #758; proposed execution prerequisites #753; handoff #759. blocked by native #758.

**Reproduce/review:** At source 13b8e374d07e7e7ea6f8f562ea6a290fcab967f7, inspect tiny_gameplay_controller.py:1970, tiny_memories.py:1; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #759.

