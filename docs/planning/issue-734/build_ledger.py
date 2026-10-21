import ast,collections,csv,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'docs/planning/issue-734'
manifest=json.loads((P/'snapshot-manifest.json').read_text());issues=json.loads((P/'issues-snapshot.json').read_text())['issues'];by={i['number']:i for i in issues};orig=set(manifest['historical_open_issues']);prs=json.loads((P/'prs-snapshot.json').read_text())['pull_requests'];review=json.loads((P/'review-origin-snapshot.json').read_text());source_review=json.loads((P/'source-review.json').read_text());review_by={n:r for r in review for n in r['issues']};sources_by={n:r for r in source_review for n in r['issues']};SHA=manifest['source_sha']
# Curated subsystem and execution handoffs. Canonical topic is a cross-reference,
# never an instruction to replace the native parent.
TOPICS={
 'architecture':(14,'Architecture and boundaries',['goap_evaluator.py','tiny_globals.py','world_state.py']),
 'graph':(222,'Authoritative world and graph identity',['tiny_graph_manager.py','tiny_globals.py','world_state.py']),
 'actions':(182,'Actions and GOAP contracts',['actions.py','tiny_goap_system.py','tiny_strategy_manager.py']),
 'characters':(139,'Character behavior and goals',['tiny_characters.py']),
 'events':(51,'Events and consequences',['tiny_event_handler.py','tiny_gameplay_controller.py']),
 'memory':(152,'Memory and semantic services',['tiny_memories.py','tiny_characters.py']),
 'llm':(158,'Prompts and final-product AI',['tiny_prompt_builder.py','tiny_strategy_manager.py','tiny_brain_io.py','tiny_output_interpreter.py']),
 'buildings':(171,'Buildings and ownership',['tiny_buildings.py','tiny_building_manager.py']),
 'items':(194,'Items, inventory and economy',['tiny_items.py','tiny_gameplay_controller.py']),
 'jobs':(199,'Jobs and economic production',['tiny_jobs.py','job_roles.json']),
 'movement':(177,'Map movement and weighted paths',['tiny_map_controller.py','tiny_locations.py']),
 'controller':(56,'Gameplay orchestration',['tiny_gameplay_controller.py']),
 'input':(135,'Input and accessibility',['tiny_gameplay_controller.py','tiny_map_controller.py']),
 'ui':(123,'Panels, inspection and player feedback',['tiny_gameplay_controller.py','tiny_map_controller.py']),
 'performance':(134,'Loop instrumentation and performance',['tiny_gameplay_controller.py']),
 'recovery':(137,'State-preserving recovery',['tiny_gameplay_controller.py']),
 'config':(62,'Configuration validation and migration',['tiny_gameplay_controller.py']),
 'persistence':(214,'Save/load and state restoration',['tiny_gameplay_controller.py','tiny_memories.py']),
 'weather':(211,'Weather and environment',['tiny_gameplay_controller.py','tiny_event_handler.py']),
 'social':(212,'Relationships and social simulation',['tiny_graph_manager.py','social_model.py']),
 'quests':(213,'Quests and progression',['tiny_gameplay_controller.py']),
 'docs':(681,'Documentation and status accuracy',['docs/README.md','README.md','docs/decisions.md']),
 'testing':(683,'Behavioral test quality',['tests/test_goap_planner_integration_real.py','tests/test_gameplay_controller.py']),
 'render':(594,'Rendering and art integration',['tiny_map_controller.py','tiny_gameplay_controller.py']),
 'mvp':(658,'Canonical scenario and release evidence',['main.py','demo_minimal_integration.py']),
 'story':(112,'Storytelling and narrative',['tiny_storytelling_system.py']),
 'tooling':(14,'Contributor tooling and experiments',['AGENTS.md','.github/workflows/python-ci.yml']),
}
S={}
def group(nums,topic,phase,disposition='implement'):
 for n in nums.split():S[int(n)]={'topic':topic,'phase':phase,'disposition':disposition}
group('11 152 153 154 155 156 157','memory',9,'defer')
group('14 49 534 535','architecture',9,'defer')
group('51 53 54 133 169 190 191 192 219 564 566 567 568','events',5)
group('55 193 565 608','events',9,'defer')
group('56','controller',4)
group('58 59 67 69','performance',9,'defer')
group('68','performance',8)
group('60 61 71 77 78 80 81 83 135','input',8)
group('74 75 76 79 82','input',9,'defer')
group('62 93 94 97 98 99 100 101','config',7)
group('95 96','config',9,'defer')
group('85 86 87 88 89 90 91 92 137','recovery',5)
group('102 103 104 105 214','persistence',7)
group('106 107 108 109 110 111 112','events',5)
group('113 114 115 116 117 118 119 120 121 598 599','render',9,'defer')
group('122 594','render',8,'verify')
group('123 132','ui',8,'verify')
group('124 125 129 131','ui',9,'defer')
group('126 128 130 381 383','ui',8)
group('134 595 596 597','performance',8,'verify')
group('138','controller',9,'defer')
group('139','characters',4)
group('140 141 142 601','characters',9,'defer')
group('143','characters',4)
group('144 600','characters',6)
group('158 161 164 168 208 209 256 260 262 263 264 583 584 585 586','llm',9,'defer')
group('206 207','llm',9,'verify')
group('166 167 182 184 185 186 187 218 220 354 356','actions',4)
group('183 188','actions',6)
group('171','buildings',6,'verify')
group('174','buildings',6,'defer')
group('177 178 180','movement',9,'defer')
group('194 195 196 197 198','items',6)
group('199 200 201 202 203 204','jobs',6)
group('211 603 604 605','weather',9,'defer')
group('212 606 607','social',9,'defer')
group('213 610 611 612 613','quests',9,'defer')
group('217 630 633 634 635 636 648 650 653 681 686','docs',8,'verify')
group('223 655 656 658','mvp',8)
group('222 245 609','graph',4)
group('284','llm',9,'verify')
group('315 319 333 492 493 494 495 496 499 500 514 515 516 517 519 520 539 540 541','testing',9,'defer')
group('350 352 353 357 359 501 502 503 504 683','testing',3,'verify')
group('351','testing',3,'duplicate')
group('355 358','actions',4,'verify')
group('380','input',9,'defer')
group('382','input',9,'duplicate')
group('474','mvp',8,'verify')
group('475','movement',9,'verify')
group('478 479 481 505 506','testing',8,'verify')
group('482','weather',8,'verify')
group('491','quests',9,'verify')
group('507 508','ui',8)
group('518 528','memory',9,'verify')
group('522','testing',4)
group('530','buildings',6)
group('602','actions',4)
group('614','persistence',9,'defer')
group('615 616','controller',9,'defer')
group('640 644 645 646 649','tooling',9,'defer')
group('641 642','docs',9,'verify')
group('643 647 651','tooling',3,'verify')
group('652','mvp',8,'verify')
group('654','mvp',2,'verify')
group('682','testing',8,'verify')
group('684 685','docs',8)
for n in [491,492,499,500]: S[n]['topic']='story'
S[245]['disposition']='verify'
assert orig == set(S),{'missing':sorted(orig-set(S)),'extra':sorted(set(S)-orig)}
# Observable completion criteria for short historical issues and key overlaps.
A={}
def acceptance(text):
 for line in text.strip().splitlines():
  n,description=line.split('|',1);A[int(n)]=description
acceptance('''
11|Bound embedding/model/index storage over a repeated record/recall/delete lifecycle; retain memory measurements and confirm supported cleanup without claiming basic-mode bounds prove full AI memory bounds.
14|Inventory coupling at graph, rendering and AI boundaries; implement scoped interfaces/configuration/refactors with preserved behavior and updated architecture evidence, rather than a blanket rewrite.
49|Demonstrate goal-completion evaluation across action sequences and graph-derived difficulty; compare documented future enhancements with current graph/GOAP implementations.
51|Trigger a real event, route its strategy/action, apply character/world effects once, and retain the graph and memory trace; complete outstanding child acceptance without treating status strings as evidence.
53|Demonstrate character, location and relationship effects with conditions/chaining, bounded changes and before/after logging; preserve all detailed child requirements.
54|A real triggered event changes the selected character strategy or plan, and its executable outcome changes authoritative state.
55|Provide documented, executable event templates that produce distinct emergent outcomes through the supported schema; additional content follows the validated event pipeline.
56|Reconcile controller child tasks against executed runtime behavior; preserve later features and report unresolved failures explicitly instead of closing the umbrella from startup success.
58|Under sustained measured load, adjust frame pacing within declared limits while preserving logical simulation outcomes; profile before selecting an adaptive policy.
59|Define multiplayer authority, synchronization and failure contracts, then demonstrate two clients agree on logical state; this extends beyond the accepted single-process MVP.
60|Exercise mouse and keyboard routes against real UI controls, with focus ownership and one action per input under the approved interaction specification.
61|Verify keyboard navigation, readable focus/state indicators, contrast and supported accessibility behavior using the agreed UI and the dedicated human accessibility review.
62|Validate import/export configuration against a documented schema; invalid input leaves the existing configuration/world intact and produces an actionable error.
67|Specify a supported mod boundary and demonstrate loading, rejecting and isolating incompatible content after the core interfaces stabilize.
68|Expose a bounded deterministic loop/scenario entrypoint whose result asserts real events, actions and state changes; startup-only smoke is insufficient.
69|Define which settings may change live; apply them only at a safe boundary, reject invalid changes and retain current state on failure.
71|Route configurable shortcuts through the focused control/dialog before global handlers; retain the approved S/L/R, Step and speed defaults.
74|Specify gestures and demonstrate recognized/unrecognized input without conflicting with click/drag/selection behavior.
75|Choose a supported touch platform and demonstrate selection, inspection and UI control without shrinking essential text or relying on hover.
76|Specify controller mapping and demonstrate focus/activation/cancellation on a supported device without changing simulation semantics.
77|Verify the explicitly listed screen-reader/high-contrast requirements and keyboard support; keep unique acceptance beyond the broader accessibility header #61.
78|Persist remapped bindings, prevent conflicts, retain text-entry focus, and expose the active shortcut labels with safe defaults.
79|Record input with logical tick context and replay to the same supported scenario outcome; distinguish UI history from a complete canonical trace.
80|Click a real villager/building and verify selection, inspection and map/UI hit testing; clicks consumed by a panel must not reach the map.
81|Verify context-menu positioning, keyboard focus, dismissal and enabled/disabled actions; informational inspection must remain live unless the user pauses.
82|Define camera zoom limits and whole-number world-art scaling; demonstrate hit testing and readable independent HUD text at each supported setting.
83|Scroll the focused entity/history view, preserving selection and reading position; Page Up/Down must not also change playback speed.
85|Expose actual memory observations with units/source and explicit unknown values; demonstrate growth/cleanup without substituting guessed resource values.
86|Expose measured runtime performance in health reports with collection state, sample count and meaningful units; startup success is not sustained-load evidence.
87|Define predictive failure signals using actual retained history; demonstrate alerts and false-positive behavior without presenting guesses as established failures.
88|Generate recovery recommendations from identified failures/dependencies and show why each action is safe for the affected state.
89|Choose recovery by failure class and prove preserved world/ownership; never replace a failed real Character with a success-shaped substitute.
90|Respect subsystem initialization/ownership prerequisites during recovery and prove recovery ordering cannot create duplicate authoritative graphs.
91|Recover only trusted components without losing world state; a state-integrity failure stops advancement until an explicitly safe recovery completes.
92|Show recovery attempt, outcome and actionable failure without stealing focus or hiding an unresolved error; retain the event-history record.
93|Reject invalid export configuration before writing, document the supported schema and round-trip valid exported sections; keep distinct import coverage in #98.
94|Version the exported schema and test supported migrations plus rejection of unsupported versions without partial state mutation.
95|Identify which configuration fields require encryption and a key-management contract before implementation; demonstrate round-trip and failed-decryption handling without recording secrets.
96|Provide documented, validated presets that preserve accepted defaults; reject invalid/unsupported profiles with a clear capability explanation.
97|Exclude unsupported/sensitive serialized values from export according to a documented policy and validate that import cannot inject unintended runtime objects.
98|Parse and validate imports before replacing active configuration; malformed/unsafe input leaves the active simulation intact.
99|Migrate each explicitly supported configuration version and reject unknown versions without changing the current world/configuration.
100|Create and verify a recoverable backup before configuration replacement; backup failure must not destroy the only known-good state.
101|Import only selected supported sections, validate dependencies across sections, and leave unselected sections unchanged.
102|Restore complete supported Character attributes, inventory, goals and identities; compare continued execution with an uninterrupted run, loaded paused.
103|Restore map/building/world entities by stable identity and validate ownership/resources rather than rebuilding default scenery.
104|Restore actual graph-backed relationship edges and verify read/update behavior after load; no parallel social-state authority.
105|Restore economic balances, inventories, production and trade bookkeeping without double applying transactions; validate continuation.
106|Process a real pending event through the authoritative handler and executable strategy outcome, not a placeholder queue drain.
107|Validate supported social/economic/environmental event payloads and dispatch actual effects; invalid event types fail clearly without partial mutation.
108|Demonstrate documented event consequences and ripple effects with bounded propagation and actual graph changes.
109|Order and schedule events deterministically by documented time/priority rules, including ties and overdue events, without repeated application.
110|Resolve event participants to real Characters and verify effects target the intended participants and authoritative world.
111|Record supported pending/completed event context and memory once; validate save/load continuation without duplicating effects.
112|Demonstrate an actual event advances story state and records an understandable outcome; fabricated story success does not prove simulation effects.
113|Define measured render-quality presets consistent with pixel-art scaling, verify configuration and compare supported performance budgets.
114|Demonstrate adaptive resolution only if it preserves approved integer-scaled pixel art and readable independent HUD text; decide from measured frame evidence.
115|Scope antialiasing to approved elements; preserve crisp nearest-neighbor world art, measure performance and retain required visual review before enabling it.
116|Define bounded post-processing that fits the approved warm pixel-art direction and compare enabled/disabled rendering against measured budgets.
117|Measure a need for distant-object LOD and demonstrate visual/state consistency across supported viewport sizes before introducing complexity.
118|Demonstrate a bounded particle pass with lifecycle cleanup, correct world/UI layering and measured load; retain approved restrained motion.
119|Implement scoped lighting/shadow layers without obscuring selected villagers, violating pixel scaling or exceeding measured budgets.
120|Render weather-specific world effects through the layer pipeline and prove they correspond to actual weather state; weather text alone is insufficient.
121|Capture the supported viewport/UI with a clear output result and failure reporting; validate recording resource cleanup and documented formats.
122|Induce a real map-rendering failure and verify a distinguishable safe fallback on a real Surface; report failure rather than silently claiming full rendering.
123|Verify real modular panels, focus, wrapping/scrolling and approved right-sidebar/bottom-drawer layout; existing legacy panels alone do not complete the new specification.
124|Display actual graph-backed relationships with meaningful labels and accessible selection; no duplicated or invented relationship state.
125|Display documented village metrics sourced from real simulation records, with units and unavailable states; keep the MVP HUD distinct from a full dashboard.
126|Inspect Building objects and dictionaries with correct type-specific properties, readable wrapping and consistent keyboard/click selection.
128|Present named saves/load dialogs, quicksave, overwrite confirmation and unsaved-progress protection; backend round-trip correctness remains a prerequisite.
129|Define which settings are safe in-session; validate controls, persistence and defaults without conflicting with approved session controls.
130|Provide the dismissible launch hint, always-available Help and visible shortcut labels; verify focus and Escape dismissal behavior.
131|Specify supported drag/drop actions and display valid/invalid placement without stealing unrelated input; retain cancellation and state integrity.
132|Use the approved five-second non-modal status lane for routine confirmations, persistent actionable errors and retained event history, without stealing focus.
133|Trace handler detection → strategy → character-aware action → graph/memory outcome with declared dispatch and exactly-once semantics.
134|Track remaining loop enhancements individually; verify instrumentation separately from adaptive FPS, configuration, networking and profiling/optimization.
135|Implement the agreed keyboard/mouse/focus model and track optional device/input extensions separately; do not close from a single shortcut working.
137|Demonstrate failure-appropriate recovery preserves authoritative state; fatal/state-integrity failures stop advancement and offer only safe actions.
138|Replace feature-status claims with evidence per subsystem; preserve weather/social/quest/procedural/audio scope and their distinct later gates.
139|Verify real Character composition and track history/preferences/banking/goals/skills individually; lightweight construction does not complete broader behavior.
140|Retain structured long-term experience with relevant temporal/causal context and show influence on recall or decisions beyond a single recent_event field.
141|Define explicit preferences and verify their source and influence on decisions without contradicting traits or remembered experience.
142|Define deposit/withdrawal/loan rules and demonstrate conservation, rejected invalid transactions and persistence; ordinary wealth changes are not a banking system.
143|Verify actual activity animation, memory-informed decisions and contextual goal generation, preserving their separate Phase 4/8/9 implementation boundaries.
144|Prove skills affect action eligibility/success/efficiency or unlocks, and successful supported activities update the real skill representation.
153|Demonstrate sorted/filtered retrieval within a GeneralMemory or document the exact retained BST role; do not infer functionality from class existence.
154|Specify contradictory-memory handling and demonstrate changed belief/recall behavior while preserving original experiences and provenance.
155|Show relevant memories change actual goal/plan choice through StrategyManager/GOAP; recording/recall or prompt inclusion alone is insufficient.
156|Use a documented extraction corpus to check correct facts, invalid inputs and unavailable-model behavior; preserve actual model/evaluation provenance.
157|Measure full NLP record/recall cost and memory usage before optimizing; retain core/basic versus full-AI capability boundaries.
161|Generate context-correct social/crisis/exploration prompts with dynamic valid actions and role formatting, including absent-context failures.
164|Trace prompt → provider → parser → executable action through one documented production interface, with explicit unavailable/fallback behavior.
166|Produce valid multi-step GOAP plans from real goals/state/actions and show effects achieve the goal; single-action utility ranking is not sufficient.
167|Construct supported production Goals from real state/needs, pass them to the planner and verify completion/failure against actual effects and scales.
168|Document and test strategic LLM activation criteria and actual provider/fallback routing; AI remains required for the final product.
174|Resolve the existing in-scope decision, then prove owner-based access/income rules and graph/economic effects with non-owner cases; do not silently add this to the MVP.
178|Validate weighted terrain changes path cost/choice and movement pacing; compare against a correctness reference before optimization.
180|Demonstrate a non-building POI with actual location, capacity, preconditions and effects; invalid/missing coordinates fail clearly.
183|Provide concrete job/item/building/social actions with valid targets/preconditions/effects; keep catalog expansion distinct from fixing Action execution.
184|Check passing and failing preconditions and actual effect ranges/units for each supported catalog action; assert authoritative graph updates and failure integrity.
185|Execute production Action objects and confirm intended effects and graph updates once, including invalid targets/preconditions; method existence is not completion.
186|Resolve equivalent dictionary/template/Action inputs to correct production behavior, preserving graph ownership, target and initiator.
187|Reconcile costs, need scales and utility/planning units; exercise boundary cases and verify meaningful plans rather than tune arbitrary constants to a fixture.
188|Use real Skill containers for action preconditions/progression and demonstrate supported skills improve only through valid actions.
190|Trigger handler events during the real gameplay update, consume returned strategies and apply character-aware action outcomes.
191|Preserve all event-impact acceptance and link the broader #53 effect work; compare character/world/relationship outcomes before considering consolidation.
192|Verify the event-driven AI requirement through real strategy/action changes; overlap with #54 is retained pending parent/PR reconciliation.
193|Demonstrate executable storytelling event content through the supported handler; overlap with #55 is retained without automatic closure.
194|Complete concrete item types/interactions/inventory/economic child contracts; item class definitions alone do not prove the economic loop.
195|Construct supported food/tool/resource/clothing/quest items with required properties and categorization; preserve the unresolved default Door action #775.
196|Execute actual equip/wear/craft/consume interactions against valid state, retaining effects and explicit failure behavior.
197|Expose inventory categories, needs and valid trade/drop options to decisions/UI, reflecting real quantities and invalid operations.
198|Demonstrate goods produced, traded and consumed with conserved supported balances and real job/need consequences.
199|Verify job assignment/leaving, prerequisites, work outcomes, progression and income with valid catalog data and real Characters; resolve #787 diagnostics separately.
200|Apply for, assign and leave supported jobs with validated requirements and graph state; reject unavailable/invalid roles without process exit.
201|Show actual work results affect job performance and downstream income/progression, with documented scales and unsuccessful-work behavior.
202|Execute type-specific work actions that consume time/energy and produce documented resources/value through authoritative state.
203|Meet documented promotion/skill conditions, advance a real career and persist it; do not manufacture progression through a mock.
204|Execute the work → income/production → market → buy/eat chain with scarcity and conservation; keep broad economy expansion beyond the bounded fixture visible.
206|Generate prompt options from currently eligible executable actions and verify state changes alter options; no invented or stale choices.
207|Parse valid and malformed actual model outputs into supported actions with clear fallback/unavailable behavior; no live inference is claimed by controlled parsers.
208|Verify documented strategic invocation triggers select the provider or deterministic fallback and preserve requested versus available capability state.
209|Inject relevant actual memories, relationships and event/world context into bounded prompts; retain provenance, formatting and context limits.
211|Verify behavior/world impact, presentation and seasonal progression separately; weather text/rendering does not complete weather simulation.
212|Demonstrate meaningful relationship dynamics and decision influence from one graph-backed authority; #609 exposes remaining graph read/update defects.
213|Generate contextual multi-step quests, progress/reward them through real events/state and make them inspectable; status labels do not prove this.
214|Round-trip all supported world/Character/relationship/economic/event/memory/time/random state, with versioning, corruption handling and paused/Finished load behavior.
217|Map each documentation discrepancy to current code and executed evidence without silently replacing original requirements or declaring the whole integration complete.
218|Connect StrategyManager to real multi-step GOAP with supported Goal/Action contracts; execute plans across ticks and replan on failure.
219|Route detected event strategies through the proper Character and action executor, asserting authoritative outcomes and exactly-once feedback.
220|Apply effects and graph updates through the supported action contract with consistent units/identities, including real named targets and failed preconditions.
222|Trace shared graph ownership and actual entity/relationship reads/writes across systems; fix identity/adapter defects without creating a second authoritative store.
223|Use the accepted #724 fixture and #751/#658 trace to verify core planning, economic actions, events and understandable UI; startup-only success is not MVP readiness.
245|Verify shared manager identity across supported simulation callers and composition boundaries; preserve #719 startup ownership evidence without extending it to unexecuted call paths.
284|Check error-reporting branches use actionable logging with exception context and controllable levels; do not replace useful user-facing console output indiscriminately.
354|Define and exercise configurable/bounded planner search limits, including exhaustion outcome; current max_iterations=1000 remains local policy rather than a class/config contract.
355|Verify all action-queue producers and consumers accept four fields; equal-priority insertion and replanning preserve action/dependency data, with explicit limits on tie determinism.
356|Replace success-shaped/random MockAlternativeAction production fallback with a real valid alternative or explicit failure; assert actual precondition/effect behavior.
358|Verify dictionary actions use the production ActionWrapper interface with meaningful effects/preconditions; do not count anonymous type construction as a supported action contract.
380|Define and implement player building placement and direct movement commands with valid/invalid placement and state effects; these exceed the accepted inspection/playback MVP.
381|Show selected Character name, needs, goal, inventory and persistent time/speed/population controls using the approved readable layout and real state.
382|Same building-placement and direct-command requirements as #380; close only after approval of this exact duplicate and canonical link.
474|Make the specifically cited map/location demo exercise real map/location/action objects and honestly label limits; #654 fixed a different official minimal entrypoint.
475|Exercise POI interaction with missing/None/invalid/valid location coordinates and capacity; valid input must produce the intended real action eligibility.
478|Induce an actual supported UI fallback condition on a real controller/Surface and assert a distinguishable rendering result without rewriting ui_panels to force success.
479|Assert meaningful pixels/content/state from real modular panels rather than assertTrue(True) or only absence of exceptions.
481|Demonstrate that real fallback output differs from modular output and remains usable under the supported failure condition.
482|Render the rainy-weather panel with real Pygame/fonts/Surface; WEATHER_UI_MESSAGES resolves and observable panel pixels are drawn.
491|Exercise StoryArc character-role lookup and supported role values with real story objects; current character_roles declaration is static evidence, not proof of arc behavior.
505|Verify real Pygame collision/hit testing/selection/render behavior; retain external-boundary doubles only when their scope is explicit.
506|Confirm one authoritative is_building implementation handles production Building/dictionary cases and produces correct hit tests; retain historical duplicate-definition context.
507|Clamp InfoPanel to actual Surface dimensions at 1024×768 and 1280×720, preserving readable wrapping rather than the current hardcoded 800×600 bounds.
508|Clamp ContextMenu to actual Surface dimensions at both supported sizes, with correct keyboard/click targets; distinct from InfoPanel #507 despite matching titles.
518|Check full-profile plotting respects the caller's selected backend; basic/headless imports must not force a matplotlib backend or initialize rendering.
522|Exercise a real controller/strategy/action/state pipeline; never substitute a strategy that always returns success or infer gameplay readiness from mocked display objects.
528|Induce actual semantic-index normalization failure and verify the result/capability accurately reports failure or a documented partial-load state, retaining the original cause.
530|Document the supported interaction energy scale/threshold, test values below/at/above it with real Characters and preserve meaningful action gating.
534|Review the viability-cost delegation against actual call cycles and graph ownership; decouple only with preserved cached graph semantics and behavior evidence.
535|Review the goal-difficulty delegation separately from viability costs; preserve authoritative graph analytics while removing any demonstrated recursive dependency.
565|Meet the original five relationship-effect types plus missing-edge, invalid-target and symmetry contracts against actual graph edges; basic MVP events do not imply this broader faction scope is complete.
594|Verify #594 is natively under #56, all #113–#122 are under #594, and #598/#599 are attached; this is organization only, not delivery of graphics features.
595|Provide retained real-loop profiling/stage evidence identifying actionable hotspots and compare any optimization; instrumentation in #596 alone does not prove optimization.
596|Record real-loop stage/frame timings and expose bounded rolling summaries, including collection disabled state; do not infer frame budget compliance or healthy gameplay.
597|Exercise empty, normal and above-budget samples and dominant-stage suggestions; verify bounded history and disabled monitoring without claiming benchmark acceptance.
598|Select supported VR/AR use cases/devices and concrete extension contracts before implementation; preserve as deferred expansion without imposing it on the Pygame MVP.
599|Define an ordered world/effects/UI pipeline and demonstrate layer ordering and cleanup; broad lighting/post-processing is later, while #761 retains required local occluder fading.
600|Derive required items from real goal completion/decision context and verify resulting eligible work/trade/consume choices; do not hardcode the fixture item name as a general solution.
601|Compute luxury/happiness from a documented recent environmental history and verify sustained versus transient surroundings produce expected outcomes.
602|Pass all six existing named-target cases AND real-world co-location, requested identity, absence, deduplication and graph updates after required Phase 4 dependencies; retain current real-object failure and keep open.
603|Show weather changes character decisions/actions, event weather checks and authoritative world state consistently; rainy-panel success #482 does not supply this evidence.
604|Render weather-specific visual/audio cues from actual weather state through supported presentation boundaries; coordinate audio with #616.
605|Advance seasons deterministically and verify temperature/weather probabilities and their documented gameplay effects over supported time progression.
606|Apply actual social action outcomes to real relationship strengths with documented ranges and asymmetric/symmetric policy.
607|Show changing real relationships changes selected goals/actions/plans in otherwise comparable contexts; a prompt mentioning friendship is insufficient.
608|Generate and resolve relationship-building/conflict/turning-point events from actual social context with bounded effects.
609|Read/write actual keyed graph relationship attributes through controller and GraphManager, including decay; no swallowed API error, default-filled result or duplicate authority.
610|Generate valid quests from actual needs/world/resources and reject impossible contexts rather than assigning unrelated random templates.
611|Advance ordered quest stages through real state/events and award documented items/money/reputation/relationships once on valid completion.
612|Inspect active/completed/progressing quests from actual state with readable navigation and no invented completion.
613|Trigger quests from supported events and emit milestone/completion events without loops or repeated rewards.
614|Persist or rebuild supported semantic memory/index state with faithful recall after load; #756 separately owns basic record persistence and is not a duplicate.
615|Select a concrete procedural-content target, seed/constraints and validity checks, then prove reproducible generated content; no generic generator or status string counts as completion.
616|Demonstrate supported music/SFX playback, volume/stop controls, event routing and cleanup with explicit unavailable-device behavior; keep UI/event hooks consistent.
630|Enumerate all tracked Markdown documents with valid relative paths and an actual navigable visualization; a partial root-file list is not complete coverage.
633|Validate the existing context map/folder blueprint against current tracked files/modules and state explicit risks/relationships; stale generated artifacts are not current evidence.
634|Trace current code patterns and OO/subsystem flows, documenting real boundaries and dependencies rather than copying intended architecture.
635|Check existing architecture/control/data diagrams against actual source flows, identify unsupported arrows and publish a readable collected report.
636|Compare documented capability claims with current executed evidence and map each gap to an issue; preserve historical failures and planning/implementation distinctions.
640|Implement the original multi-tier Markdown routing/metadata/ambiguity conventions and validate paths; it is broader than flat map #630 and not a duplicate.
641|Retain rationale, alternatives, consequences and human decisions in the existing single decision file; identify any missing architectural decisions without splitting accepted decisions into parallel records.
642|Reconcile the accepted MVP/support/budget contract and broader vision into explicit product requirements; do not claim the bounded fixture exhausts the final product.
643|Review the actual supported-profile workflow, effective required checks and retained failures; preserve #726 evidence without approving unrelated legacy/AI workflows.
644|Identify an actual frontend target before React/Next review; approved Pygame UX/UI review remains separate, and this does not authorize a web rewrite.
645|Replace the vague modernization request with evidence-backed bounded changes after core contracts stabilize; no blanket framework migration or feature removal.
646|Require an actual supported browser application before Playwright work; current native Pygame cannot be validated as a web application.
647|Assess actual subsystem boundaries/risks against code and current engineering evidence, preserving #731's scoped approval without extending it to untested gameplay.
648|Audit authoritative technical documentation against real interfaces/commands and current evidence, clearly marking archival material.
649|Define a concrete TinyVillage-related product question/use case before invoking an unrelated app-idea workflow; retain the original expansive game vision.
650|Compare implementation against the already approved UX/visual/UI and runtime specifications; reopen only concrete discrepancies, not settled preference choices.
651|Provide scoped issue-backed tasks, prerequisites, acceptance/evidence and human gates; compare the existing 86-item roadmap and this ledger before inventing another plan.
652|Report actual evidence per accepted MVP criterion and unresolved blockers; do not invent an overall completion percentage or equate startup with economic gameplay.
653|Validate existing folder/flow/Copilot-instruction blueprints against the current repository, documenting gaps and a maintainable update path.
654|Preserve completed #788/#726 startup/basic-memory acceptance evidence and existing closed state; no gameplay, live-inference or wider phase approval is inferred.
655|Print actual per-Character intents/actions/outcomes, triggering events and final who-did-what summary from the canonical scenario; startup logs alone do not suffice.
656|Reconcile README/Quickstart/status claims with current minimal/visual/canonical commands, archive labels and executed scenario evidence.
658|Publish one documented deterministic real gameplay scenario with fixed fixture/seed, events/actions/outcomes and final result; preserve both headless and Pygame presentations.
681|Track documentation/test/status/architecture gaps with evidence-backed children, keeping current versus archived authority clear and later scope intact.
682|Compare the two current building-loading suites case by case, retain distinct unit/integration behavior and update referenced commands without losing unique assertions.
683|Audit permissive memory/build/Character stand-ins, retain only deliberate external boundaries/negative examples and assert actual behavior; selected Phase 3 test-PR work is only part of this broader task.
684|Correct present-tense readiness claims using retained evidence and archive labels; explicitly preserve D015 final-product AI rather than repeating the old nice-to-have wording.
685|Document actual building/checkpoint initialization, failure paths and supported public methods; do not infer complete persistence from robust checkpoint scaffolding.
686|Align entrypoints/method names/status links and segregate negative mock examples; preserve useful historical text with clear archival context instead of rewriting evidence.
''')
acceptance('''
315|Exercise prompt descriptor randomness with actual supported dependencies or a clearly bounded RNG fixture; no torch.rand stub fixed at zero may stand in for production behavior.
319|Use a Character-compatible fixture or real basic-memory Character and demonstrate that a wrong interface fails; do not replace the production class with object in sys.modules.
333|Call the real respond_to_talk method and verify its documented effect for more than one context, instead of copying a fixed increment into the test double.
350|Replay each returned GOAP plan's actual effects from the initial state and assert the requested goal is reached; reconcile the removed historical file with current tests under #736.
351|Identical original body and identical source discussion_r2217197555 as #350; preserve the canonical goal-achievement acceptance and historical link before approved duplicate closure.
352|Validate that the planned actions actually achieve the goal in the debug scenario; a nonempty plan or printed action list is insufficient.
353|Execute the energy-restoration sequence and assert the actual Sleep action raises energy to the goal using supported units; retain this scenario distinct from generic #350.
357|Reconcile the removed historical planner test with current real-graph coverage and explicit unavailable-graph behavior; a no-error None-manager run is insufficient.
359|Verify documented fallback results and real state changes with absent graph services, separately from real-graph integration; retain the original removed-file context.
492|Drive story integration with actual graph/events/story objects and assert narrative progress corresponds to real state; no permissive graph mock may supply the result.
493|Exercise actual social preconditions in passing and failing states, proving a failed precondition prevents the action and its effects.
494|Use real Characters, locations and graph in the social integration path and assert actual effects; copied production logic in MockCharacter is not evidence.
495|Call the real respond_to_talk implementation and assert documented outcomes, including cases that distinguish it from the former copied mock.
496|Exercise real personality/Character interactions and outcomes with representative traits; keep doubles only at explicitly external boundaries.
499|Use real NarrativeImpact values and actual current narrative context to assert importance scaling at relevant boundaries, not a canned context mock.
500|Advance a real arc using actual story elements; assert element-count/progression conditions and failure/non-progress cases rather than accepting any Mock element.
501|Locate replacement coverage for the removed comprehensive GOAP tests and prove actual graph-derived plan behavior, preserving historical missing-dependency failures.
502|Verify graph-present and explicitly absent-graph planner results with meaningful goals/effects, not only no-error completion of the focused script.
503|Execute real supported Actions with the intended target/initiator and demonstrate both valid effects and real precondition failure; reconcile the removed historical file.
504|Reproduce retry/replan failure handling from actual changing conditions and prove resulting world state, instead of predetermined True/False action outcomes.
514|Run SocialModel through real GraphManager state and assert actual relationships; no substituted tiny_characters module or graph-only mock may establish integration.
515|Build an actual WorldState graph and verify SocialModel edge/neighborhood behavior for nonempty and absent relations, rather than an empty canned graph.
516|Assert semantic prompt/context/action/schema correctness for representative real contexts; string length, type and a character-name substring are insufficient.
517|Validate actual provider input/output behavior and explicit unavailable-model errors under the supported profile; output list type alone does not prove inference.
519|Exercise the actual decide_action_with_llm pipeline with a controlled external provider boundary and separate live inference evidence; do not mock the method under test.
520|Connect actual prompt construction, parsing and executable actions with recorded provider input/output; distinguish controlled model boundaries from live model readiness.
539|Generate travel prompts from supported Character/location/state and assert complete destination/context/role formatting, including missing destination behavior.
540|Generate social prompts with eligible actual action options, correct Character context and complete role formatting; substring-only assertions do not establish correctness.
541|Replace object or permissive stand-ins with actual Character-compatible data and prove unsupported attributes fail rather than being fabricated by the test setup.
564|Implement four supported world/location effect types through real EventHandler→GraphManager mutation, including missing nodes/unknown fields and a real integration trace; retain the schema-v2 prerequisite.
566|Demonstrate two effect-condition patterns, two stacking policies and guarded chaining with true/false conditions, cap behavior, application order and termination.
567|Retain structured event/effect/target/before/after/time records for every applied effect, configurable caps/scales and a debug diff check using actual outcomes.
568|Provide 5–10 schema-v2 event examples, explicit attribute/target mapping, an authoring guide and an automated golden-path assertion of real deltas and logs.
''')
acceptance("""
152|Verify semantic recording/retrieval, BST role, coherence, decision influence, fact extraction and resource bounds against the original memory requirements; basic recent recall is a completed prerequisite, not the full feature.
158|Provide dynamic goal/need/context prompts, diverse situations, robust parsed outputs and strategic provider routing through actual production interfaces; retain final-product AI scope.
169|A real handler event influences strategy selection and an executed Character plan; retain event-driven behavior beyond a method call or event registration.
171|Verify type-specific building functionality, valid interactions, ownership effects and robust custom loading through real graph/action state; constructor success alone is insufficient.
177|Compare weighted A* results against a correctness reference on dynamic obstacles/complex maps, including unreachable goals, before optimizing membership queues.
182|Complete the original action catalog, preconditions/effects, real execution/graph ownership, resolver alignment, cost and skill contracts; preserve all child evidence rather than closing the umbrella from one action.
256|Validate descriptor/need caching and invalidation, modular assembly, configurable prompt length and documented safe concurrency against meaningful prompt outputs and performance evidence.
260|Meet the original 80-percent module coverage target, schema integration, documented fixtures and CI/coverage reporting; existing supported-profile CI is partial evidence only.
262|Provide current PromptBuilder overview/quickstart, public docstrings, three real usage examples, documented contributor workflow and the specified docs-site pipeline; keep deployment requirements separate from a passing startup guide.
263|Specify and validate extension components, model-specific prompt variants, debugging visibility and configuration-driven assembly after stable prompt/provider contracts.
264|Define and verify emotion-aware context, narrative continuity, system/user roles and configurable game-context constraints as scoped final-product AI increments.
383|Implement New Game/Load/Quit and actual save/load backend plus UI, preserving named saves, dirty-progress protection, paused load and approved Escape/focus behavior.
583|Cover each documented strategic LLM trigger in production decision routing and representative demos/tests with updated code rationale and guides; no live inference is inferred from routing tests.
584|Record character/context/trigger/goal/fallback reason for every strategic provider call and expose interpretable demo/log examples with accurate documentation.
585|Exercise every strategic route with meaningful prompt/provider/parser/action assertions and separately retained real download/model/inference evidence; controlled external boundaries do not establish live readiness.
586|Link current criteria, call sites, operational logs, supported configuration, actual scenario tests and failure behavior in the LLM reference/usage guides.
""")
assert set(A)==orig, {'missing_acceptance':sorted(orig-set(A)),'extra':sorted(set(A)-orig)}
# Native roadmap tasks retain their own ordered requirements and human gates.
phase_gate={3:732,4:738,5:743,6:748,7:753,8:759,9:768}
phase_epic={2:723,3:733,4:739,5:744,6:749,7:754,8:760,9:769}
# Additional handoffs are textual proposals, not invented native dependency writes.
HANDOFF={3:736,4:220,5:219,6:751,7:755,8:761,9:770,2:726}
for n in [350,351,352,353,357,359,501,502,503,504,683]:S[n]['handoff']=736
for n in [11,157]:S[n]['handoff']=11
for n in [155,140,154]:S[n]['handoff']=155
for n in [583,584,585,586,158,161,164,168,208,209,256,260,262,263,264,206,207,284,315,319,516,517,519,520,539,540,541]:S[n]['handoff']=585 if n in [260,315,319,516,517,519,520,539,540,541] else 583
for n in [166,167,218,354,355,356,358]:S[n]['handoff']=218 if n!=356 else 741
for n in [182,184,185,186,187,220,222,245,609]:S[n]['handoff']=220
for n in [51,53,54,133,169,190,191,192,106,107,108,109,110,111,112,564,566,567,568]:S[n]['handoff']=745 if n not in [54,169,190,192] else 219
for n in [85,86,87,88,89,90,91,92,137]:S[n]['handoff']=137
for n in [102,103,104,105,214,614]:S[n]['handoff']=756 if n==614 else (n if n in [102,103,104,105] else 755)
for n in [194,195,196,197,198,199,200,201,202,203,204,530,144,183,188,600]:S[n]['handoff']=199 if n in [199,200,201,202,203,204] else (750 if n in [197,600] else 195 if n in [194,195] else 198 if n==198 else 196)
for n in [177,178,180,475]:S[n]['handoff']=177
for n in [606,607,212,333,493,494,495,496,514,515]:S[n]['handoff']=606
for n in [211,603,604,605,213,610,611,612,613,615,616,49,14,534,535]:S[n]['handoff']=770
for n in [595,596,597,58,113,114,117]:S[n]['handoff']=762
for n in [655,656,658,223,652,474]:S[n]['handoff']=658
for n in [217,630,633,634,635,636,648,650,653,681,684,685,686,682]:S[n]['handoff']=656
S[602]['handoff']=602;S[174]['handoff']=174;S[115]['handoff']=115
DUPLICATES={351:350,382:380}
COMPLETE_CANDIDATES={355:'queue',482:'weather',594:'native hierarchy snapshot',596:'live_timing and frame_analysis',597:'frame_analysis'}
SPECIAL={
 351:'Exact duplicate: identical original issue body and same original review-comment URL as #350. No unique acceptance was found in either issue discussion.',
 382:'Exact duplicate: original #380 and #382 bodies are byte-for-byte identical and neither has an issue discussion. Preserve #380 as canonical.',
 507:'The inline finding concerns InfoPanel.show; current lines 36–37 still clamp to 800×600. #508 targets a different class/method, so matching titles are not duplicate evidence.',
 508:'The inline finding concerns ContextMenu.show; current lines 123–124 still clamp to 800×600. Keep separate from InfoPanel #507.',
 356:'Current production fallback still defines MockAlternativeAction, always accepts preconditions and returns a random success without applying real effects. This is not resolved by queue compatibility.',
 355:'Queue probe passes and both actual Plan.execute heap-pop consumers unpack four fields. Replan retains dependencies but uses id(action) for ties; no cross-process deterministic-order claim.',
 482:'Real SDL-dummy rainy panel rendering changes actual Surface pixels and exits zero. This proves the reported missing-constant path only, not weather behavior/audio.',
 594:'All ten #113–#122 native parents are #594; #598/#599 also have parent #594 and #594 has parent #56. Its organizational acceptance is already met, independently of feature implementation.',
 596:'One actual SDL-dummy game-loop frame records all five timing stages. Bounded/disabled collection passes the corrected probe. The same frame logs map/animation/recovery defects; this is not healthy-gameplay or frame-budget evidence.',
 597:'Corrected probe validates empty/normal/slow/dominant-stage samples, ten-record eviction and disabled collection. Current policy can suggest dominant-stage optimization below budget; that is documented behavior, not a measured performance failure.',
 602:'Six legacy regressions pass using DemoCharacter/RecordingGraphManager fixtures. The real-object probe fails: Action passes UUID to a graph keyed by Character objects; mutating current_visitors then breaks Location hashing (dict_or_obj). Phase 4 prerequisites remain open; never close from the fixture-only pass.',
 609:'Real MultiDiGraph relationship reader logs unhashable dict and returns empty; keyed edge attributes do not decay. Static code passes {} as get_edge_data key and treats multiedge adjacency as flat attributes. Controller delegation/status wording does not establish correctness.',
 245:'Merged #719 proves scoped initialization/ownership. Broader caller coverage remains a verification item. The visual-loop None.locations error is specifically MapController.location_manager being reset to None, not proof of a new global GraphManager initialization failure.',
 654:'Already CLOSED/COMPLETED through merged #788 and preserved #726 validation; original cohort membership is historical and does not reopen it. Existing pre-merge wording may be contextualized without changing accepted scope.',
 614:'Full semantic/index persistence remains Phase 9 under existing dependencies. Basic memory/pending events/time/randomness have separate Phase 7 acceptance in #756; retain both.',
 684:'Original wording calls LLM/full memory nice-to-have. D015 supersedes that for the final product; only lightweight testing may omit AI dependencies.',
 644:'The repository has no supported React/Next frontend; preserve the request as deferred until there is a concrete target. Pygame implementation/review remains in #761/#763–#766.',
 646:'There is no browser application target in the supported runtime; Playwright cannot supply native Pygame acceptance. Deferred re-scope requires a concrete browser feature, not a new framework by assumption.',
 643:'#726 supplies current supported-profile CI evidence, not approval of optional AI Mentor or legacy workflows. Inventory/queue decisions belong in #737/#770.',
 491:'StoryArc now declares character_roles and the lookup uses string defaults. The old missing-field/enum finding is a verification candidate; meaningful arc behavior was not executed here.',
 506:'Only one is_building definition remains in the current module. This removes the static duplicate-definition symptom but does not establish current Building/dictionary hit-test correctness.',
 354:'The current planner still uses local max_iterations=1000. Keep this low-priority policy improvement separate from the proven tuple compatibility issue.',
 518:'The #654 lazy-service boundary removed eager matplotlib import/backend setup from the supported memory path. Full-profile caller-selected plotting behavior is not exercised by this audit.',
}
# Runtime source census: definitions, TODOs and hashes, never import/initialize modules.
tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
paths=set(f for t in TOPICS.values() for f in t[2])
paths.update(f for r in source_review for f in r['current_paths'])
paths.update(['tiny_locations.py','tiny_animation_system.py','actions.py','tests/test_action_named_target_resolution.py','tests/test_goap_planner_integration_real.py','.github/workflows/python-ci.yml','.github/workflows/python-ml-profiles.yml'])
source={}
for f in sorted(paths):
 p=ROOT/f
 if not p.is_file():continue
 raw=subprocess.check_output(['git','show',f'{SHA}:{f}'],cwd=ROOT);text=raw.decode(errors='replace');lines=text.splitlines();defs=[]
 if f.endswith('.py'):
  try:defs=[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno} for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
  except SyntaxError as e:defs=[{'parse_error':str(e)}]
 source[f]={'sha256':hashlib.sha256(raw).hexdigest(),'definitions':defs,'todos':[{'line':n,'text':s.strip()} for n,s in enumerate(lines,1) if 'TODO' in s]}
(P/'source-anchors.json').write_text(json.dumps(source,indent=2)+'\n')
def checklist(body):
 section=body.split('### Acceptance criteria\n')[-1] if '### Acceptance criteria\n' in body else body
 if '### Related' in section:section=section.split('### Related')[0]
 return re.findall(r'^\s*- \[[ xX]\]\s*(.+)',section,re.M)
def evidence_paths(n,topic):
 paths=sources_by.get(n,{}).get('current_paths',[]) or TOPICS[topic][2]
 paths=[f for f in paths if f in source]
 found=[]
 text=by[n]['body']+' '+by[n]['title']
 for f in paths[:3]:
  defs=[d for d in source[f]['definitions'] if d.get('name') and re.search(r'\b'+re.escape(d['name'])+r'\b',text)]
  line=defs[0]['line'] if defs else 1
  found.append(f'{f}:{line}')
 return found
rows=[]
for n in sorted(orig | {i['number'] for i in issues if i['state']=='OPEN'}):
 i=by[n];native=[b['number'] for b in i['blockedBy']['nodes']];original=n in orig
 if original:
  spec=S[n];topic=spec['topic'];phase=spec['phase'];disp=spec['disposition'];acc=A[n]
 else:
  phase_match=re.search(r'- Phase: (\d+)',i['body']) or re.search(r'\[(?:E|H)(\d+)\]',i['title']);phase=int(phase_match.group(1)) if phase_match else {716:3,773:4,774:4,775:6,786:5,787:6}[n]
  topic='tooling' if phase==3 else {4:'actions',5:'events',6:'items',7:'persistence',8:'mvp',9:'llm'}[phase]
  if n==775:topic='items'
  if n==787:topic='jobs'
  if n in [773,774]:topic='testing'
  if n==716:topic='mvp'
  disp='defer' if phase>3 else 'implement'
  if re.match(r'\[H\d|\[P\d-\d+\] Human',i['title']):disp='verify'
  acc='; '.join(checklist(i['body'])) or re.split(r'## Acceptance|## Scoped follow-up|Acceptance:',i['body'])[-1].strip()
  spec={'handoff':n}
 gate=phase_gate.get(phase);handoff=spec.get('handoff',HANDOFF.get(phase))
 # Textual execution prerequisites are not native edges and must not silently
 # impose one sibling task as another's blocker. Native edges are authoritative.
 suggested=[] if i['state']=='CLOSED' or n in COMPLETE_CANDIDATES or n in DUPLICATES else ([gate] if gate and gate not in native and n!=gate else [])
 open_native=[b for b in native if by[b]['state']=='OPEN']
 readiness='closed: preserve existing disposition' if i['state']=='CLOSED' else ('blocked by native '+','.join('#'+str(b) for b in open_native) if open_native else ('await human review' if disp=='duplicate' or n in COMPLETE_CANDIDATES else ('scheduled after #'+str(gate) if suggested and by[gate]['state']=='OPEN' else 'ready for scoped investigation; publication review pending')))
 default_rationale={
 'implement':f'Retain the {TOPICS[topic][1].lower()} acceptance as ordered implementation/integration work in Phase {phase}; source presence alone does not prove its acceptance.',
 'verify':f'Reconcile the historical claim against current {TOPICS[topic][1].lower()} code and evidence before proposing completion; unexecuted behavior remains unverified.',
 'defer':f'Retain this broader {TOPICS[topic][1].lower()} requirement for Phase {phase}; activate only after its stated prerequisites and scoped acceptance are reviewed. Deferral is not completion or removal.',
 'duplicate':'Exact duplicate candidate requires human approval of this specific issue and canonical evidence before closure.'}[disp]
 evidence=evidence_paths(n,topic)
 status='source_reviewed_not_executed'
 if n in [355,482,596,597]:status='passed_scoped_probe'
 if n==594:status='verified_native_organization'
 if n in [602,609]:status='failed_real_object_probe'
 if n==654:status='historical_completed_evidence_preserved'
 if not original:status='existing_roadmap_or_remediation_acceptance_preserved'
 row={'number':n,'url':i['url'],'title':i['title'],'cohort':'original_261' if original else 'current_only_open','state':i['state'],'state_reason':i['stateReason'],'snapshot_updated_at':i['updatedAt'],'snapshot_body_sha256':hashlib.sha256(i['body'].encode()).hexdigest(),'actual_assignees':[a['login'] for a in i['assignees']['nodes']],'accountable_owner':next((a['login'] for a in i['assignees']['nodes'] if a['login'] not in ['Copilot','copilot-swe-agent']), 'dhar174'),'owner_basis':'existing human assignee' if any(a['login'] not in ['Copilot','copilot-swe-agent'] for a in i['assignees']['nodes']) else 'ledger-only maintainer accountability; GitHub assignments preserved','canonical_topic':TOPICS[topic][1],'canonical_topic_issue':TOPICS[topic][0],'native_parent':(i['parent'] or {}).get('number'),'native_parent_state':by[i['parent']['number']]['state'] if i['parent'] else None,'native_prerequisites':native,'proposed_execution_prerequisites':suggested,'readiness':readiness,'disposition':disp,'execution_phase':phase,'handoff_issue':handoff,'acceptance':acc,'original_checklist':checklist(i['body']),'reproduction':f'At source {SHA}, inspect '+', '.join(evidence)+f'; exercise the stated acceptance with actual supported objects/profile. Historical reproduction and discussion are retained in issues-snapshot.json entry #{n}.' if evidence else f'Review snapshot issue #{n}, its native hierarchy/dependencies, original acceptance and linked immutable roadmap/decision evidence; execute the stated scenarios when its phase activates.','source_anchors':evidence,'source_review_refs':[r['data']['html_url'] for r in review if n in r['issues']],'issue_discussions':[c['url'] for c in i['comments']['nodes']],'verification':status,'rationale':SPECIAL.get(n,default_rationale),'duplicate_of':DUPLICATES.get(n),'deferral_condition':(f'After #{gate} plus preserved native prerequisites; review the concrete acceptance and applicable human activation before implementation. AI is final-product scope, not optional product functionality.' if phase==9 else f'After the recorded Phase {phase} prerequisites and explicit scope decision, retaining current acceptance.') if disp=='defer' else None,'proposed_state':'CLOSED:COMPLETED' if n in COMPLETE_CANDIDATES else 'CLOSED:NOT_PLANNED (duplicate)' if n in DUPLICATES else 'preserve','proposed_parent':TOPICS[topic][0] if (original or n in [786,787]) and not i['parent'] and TOPICS[topic][0]!=n and n not in DUPLICATES else None}
 rows.append(row)
(P/'ledger.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
with (P/'ledger.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
 for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
# Exact non-executing body annotations. A later publication must require review,
# fresh compare-and-merge, matching body hashes, and API readback.
proposals=[]
for r in rows:
 n=r['number'];prereqs=r['native_prerequisites'] or r['proposed_execution_prerequisites'];links=lambda xs:', '.join(f'#{x}' for x in xs) or 'none in snapshot'
 annotation=f'''<!-- issue-734-triage:v1:start -->
## Reviewed backlog triage — #734

- Disposition: **{r['disposition']}**. {r['rationale']}
- Accountable ledger owner: **{r['accountable_owner']}**; existing assignees preserved.
- Canonical topic: #{r['canonical_topic_issue']} ({r['canonical_topic']}); existing native parent preserved.
- Order: Phase {r['execution_phase']}; native prerequisites {links(r['native_prerequisites'])}. Proposed execution prerequisites {links(r['proposed_execution_prerequisites'])}; handoff #{r['handoff_issue']} (coordination, not an additional blocker).
- Acceptance/reproduction: {r['acceptance']}
- Evidence: #734's reviewed ledger and retained snapshot at `{SHA}`; outcome `{r['verification']}`. Unexecuted behavior is not verified.
- Closure and gate boundary: {('Duplicate of #'+str(r['duplicate_of'])+'; closure only under the specific recorded human approval.') if r['duplicate_of'] else 'No later human gate or broader product readiness is inferred.'}

<!-- issue-734-triage:v1:end -->'''
 proposals.append({'number':n,'base_updated_at':r['snapshot_updated_at'],'base_body_sha256':r['snapshot_body_sha256'],'body_operation':'prepend marked block; preserve all prior content','proposed_annotation':annotation,'parent_operation':{'add_if_unparented':r['proposed_parent']} if r['proposed_parent'] else 'preserve','native_dependencies_operation':'preserve; textual execution suggestions are proposals only','state_operation':r['proposed_state'],'assignees_operation':'preserve','requires_review':True})
(P/'proposed-issue-updates.json').write_text(json.dumps({'status':'PROPOSED_NOT_APPROVED_NOT_APPLIED','mutations_performed':0,'updates':proposals},indent=2,ensure_ascii=False)+'\n')
# Every open PR is associated with a handoff only. No keep/rebase/merge decision.
pr_tasks={735:[703,714],736:[420,417,408,406,404,397,395,391,393],218:[689],219:[690],174:[691],196:[692],750:[693],198:[694],195:[695],220:[696],199:[697],761:[710],115:[713],177:[712],155:[363],656:[687],633:[688],770:[592,700,702,705,707,708],737:[701,704,709,711,715]}
pr_rows=[]
for pr in prs:
 hand=next((n for n,nums in pr_tasks.items() if pr['number'] in nums),737)
 pr_rows.append({'number':pr['number'],'url':pr['url'],'title':pr['title'],'head_sha':pr['headRefOid'],'base':pr['baseRefName'],'draft':pr['isDraft'],'actual_assignees':[a['login'] for a in pr['assignees']['nodes']],'accountable_ledger_owner':'dhar174','handoff_issue':hand,'closing_issue_references':[i['number'] for i in pr['closingIssuesReferences']['nodes']],'files':[f['path'] for f in pr['files']['nodes']],'disposition':'inventory only; handoff is not merge approval','review_and_ci':'snapshot head/mergeability/reviews retained; current CI/ownership/rebase judgment reserved for #735–#737 or the linked implementation task'})
(P/'pr-handoffs.json').write_text(json.dumps(pr_rows,indent=2,ensure_ascii=False)+'\n')
counts=collections.Counter(r['disposition'] for r in rows);print(json.dumps({'rows':len(rows),'original':sum(r['cohort']=='original_261' for r in rows),'current_only':sum(r['cohort']=='current_only_open' for r in rows),'dispositions':dict(counts),'proposed_parent_additions':sum(r['proposed_parent'] is not None for r in rows),'closure_candidates':sorted(COMPLETE_CANDIDATES),'duplicate_candidates':DUPLICATES,'open_pr_handoffs':len(pr_rows)},indent=2))
