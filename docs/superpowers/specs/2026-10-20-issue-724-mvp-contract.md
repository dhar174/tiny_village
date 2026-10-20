# Tiny Village MVP scenario, support matrix and release budgets

**Status: Planning contract approved with vision-preservation constraint —
2026-10-20 (D014).** This is the planning deliverable for
[#724](https://github.com/dhar174/tiny_village/issues/724). It defines agreed
implementation and release-validation targets, not implemented behavior or
measured performance. D013 selects both presentations; D014 accepts the
contract while preserving the original TinyVillage vision. #724 awaits the
integration PR because #783 merged into #782's old branch after #782 had merged.
Human decisions belong only in [docs/decisions.md](../../decisions.md).

## 1. Purpose, scope and recommendation

A player should watch a small village make understandable decisions: work to
earn money, acquire food, eat and recover. A contributor should reproduce the
same outcomes and inspect the authoritative state without a display, GPU,
model download or hosted service. Success means real state changes and a
verifiable explanation of them, rather than a successful startup banner.

Use **one simulation with two presentations**: a headless acceptance run
and a small Pygame observer interface. Both use the same scenario, action
execution, logical clock and state; rendering cannot choose or apply actions.
The maintainer explicitly selected both surfaces on 2026-10-20; D013 in the
single decision record captures that choice. D014 subsequently accepts the
scenario/support/budget contract with the long-term scope guardrails below.

| Approach | Benefit | Cost / scope consequence |
| --- | --- | --- |
| Headless proof plus Pygame observer, selected (D013) | Reproducible behavior and a visible game, with one correctness contract | Requires both renderer-free initialization and the existing UI/design work |
| Headless-only release | Smallest validation surface | Defers the player-facing Pygame experience; requires an explicit roadmap scope change |
| Pygame-first release | Earliest visual demonstration | More display dependencies; console diagnostics alone would not satisfy reproducible behavioral acceptance |

Release scope includes the economic loop, a small fixed map, explanations,
pause/inspection, basic save/load and honest failure recovery. Save/load remains
Phase 7 work, not an extra task to implement in #724. Defer quests, VR,
multiplayer/network play, mod/editor tooling, broad social simulation and
advanced analytics from this MVP. Optional live ML/LLM features cannot become
core requirements through imports or silent downloads. Existing later-phase
issues remain open; exclusion from this scenario is not their completion.

### Preserve the expansive TinyVillage vision

The MVP is the first verifiable slice of the existing project. Autonomous
characters, personalities and histories, relationships and social behavior,
careers and a richer economy, memory and storytelling, optional LLM-assisted
decisions, persistence and a growing village remain part of the broader vision
in the project brief, product context and design documents. Their existing
roadmap issues and review requirements remain intact.

The three villagers, fixed map, fixed seed and episode rules are a versioned
acceptance fixture. They are not limits on simulation population, map size,
available actions or eventual emergent behavior. Put fixture-specific values
in scenario data; the shared engine must continue to accept other actors,
worlds and supported action definitions without name-based branches.

Separating core/visual/ML dependencies changes installation requirements, not
the availability or ambition of optional capabilities. Preserve their explicit
integration boundaries and existing feature paths. Do not delete subsystems,
replace the shared world with a demo-only implementation, or narrow public
contracts solely to fit this episode. Any proposed feature removal or broader
scope reduction needs its own explicit decision and issue. Keep extensions
grounded in current interfaces; this does not require a speculative rewrite.

## 2. Research baseline and current gaps

Reviewed source: runtime/main `77e4b9437cc00970def933d992140e6da37eae8e`.
Decision/evidence base: PR [#782](https://github.com/dhar174/tiny_village/pull/782)
at `68da3afd01e40c114e0392fa1bf22e797871d18f`. This draft is stacked on that
unmerged documentation branch. Research was static; no application runs,
benchmarks, dependency installations or new tests were performed for #724.

| Observed source or evidence | Consequence for this proposal |
| --- | --- |
| [`main.py`](../../../main.py), `main`, `check_dependencies`, `run_minimal_demo` | CLI has modes/characters/FPS/headless, but no scenario, seed, tick limit or result-file arguments. Dependency checks include Pygame and FAISS even for minimal mode; minimal's config is not passed to its demo. Proposed commands below do not work yet. |
| [`demo_minimal_integration.py`](../../../demo_minimal_integration.py), `mock_heavy_dependencies`, `create_minimal_character` | Demo patches `sys.modules`, uses simplified characters and catches action failures. A real core profile needs #725/#654, including #776's unavailable-spaCy construction path. |
| [Phase 1 report](../../testing/phase-1-post-merge-verification.md) | 94 method summaries and three probes succeeded, with the accepted lifecycle supervisor gap. Minimal mode executed zero actions. This establishes neither the economic scenario nor clean-install, visual or performance readiness. |
| [`tiny_strategy_manager.py`](../../../tiny_strategy_manager.py), `get_character_state_dict`, `get_daily_actions`; [`actions.py`](../../../actions.py), `WorkAction` | Need normalization and action effects mix 0–1, 0–10 and larger values; work may add money without a payer. #740 and the economic tasks must reconcile this, not a special demo-only state shim. |
| [`tiny_gameplay_controller.py`](../../../tiny_gameplay_controller.py), `game_loop`, `_record_performance_sample`, `_update_economic_state` | Loop uses frame delta, frame samples include FPS limiting, and market/trade events may add random wealth. The scenario needs logical time, causal transfers and a deterministic configuration. Existing timing instrumentation is a starting point, not percentile evidence. |
| [`tiny_time_manager.py`](../../../tiny_time_manager.py), `GameCalendar`; [`tiny_characters.py`](../../../tiny_characters.py), `Character.__init__` | Wall-clock time, random initialization and UUID creation need explicit deterministic boundaries and stable scenario identities. A seed alone is insufficient. |
| [`tiny_gameplay_controller.py`](../../../tiny_gameplay_controller.py), `_handle_keydown`; [`tiny_map_controller.py`](../../../tiny_map_controller.py), `InfoPanel` | Existing pause/reset/save/load/help and selection paths can be refined; a new UI framework is unnecessary. Their existence is not verified UX or save/load correctness. |
| [`requirements.txt`](../../../requirements.txt) | GPU/NLP/model packages and core dependencies are mixed. #725 must establish verified install profiles before any support claim. |

Other retained gaps: #773/#774 integration and Goal/fixture contracts, #775's
Door effects, and #776's optional memory initialization. D002 keeps their
scheduling intact. #724 does not repair or retest them and does not reopen the
Phase 1 gate. #223 supplies the economic-loop intent; #751 proves the integrated
loop later; #658 publishes the final canonical demo in Phase 8.

## 3. Versioned deterministic scenario: `village-economy-v1`

### Inputs and units

- Seed: **724**. Three villagers with stable IDs and fixed attributes below.
  Use deterministic tie-breaking by stable action/target ID. Record RNG
  algorithm/state, scenario version and dependency lock hash. Optional features
  must not consume the scenario RNG or change action ordering.
- **120 logical ticks**, each representing five game minutes, starting at
  scenario time `2023-01-01T08:00:00` with no host timezone dependency. Headless
  runs without sleeps; visual playback defaults to two ticks/second. Frame
  timing, pause duration and wall-clock timestamps cannot change simulation
  results. Each actor completes at most one world action per tick, in stable
  actor-ID order; a movement step is one such action.
- Canonical scenario needs: hunger and energy are **0–10**; higher hunger is
  worse and higher energy is better. Money is integer credits; food is integer
  meal units. Planner adapters may normalize to 0–1 once at a documented
  boundary. This proposes a contract for #740; it does not describe current
  runtime consistency or require an unrelated global field migration.
- Fixed unobstructed grid: home `(1,1)`, workshop `(1,4)`, market `(4,1)`;
  bidirectional cardinal movement, one tile/tick, horizontal before vertical
  on equal-cost routes. No random obstacles or unimplemented Door traversal.
  Moving changes position, not hunger/energy, in this bounded fixture.
- Employer starts with **100 credits**; market till **0**, stock **6 meals**.
  All three villagers start at home. Other attributes required by production
  constructors must be explicitly fixed in the versioned fixture, not generated
  from the host or filled by permissive mocks. Jobs and locations use real
  supported objects; background quests/weather/rewards and passive need decay
  are disabled through supported scenario configuration.

| Stable ID / display name | Hunger | Energy | Credits | Owned meals | Decision to demonstrate |
| --- | ---: | ---: | ---: | ---: | --- |
| `villager_emma` / Emma | 7 | 6 | 0 | 0 | Earn, buy, eat, then recover |
| `villager_bob` / Bob | 7 | 2 | 5 | 0 | Address critical fatigue before shopping |
| `villager_casey` / Casey | 7 | 6 | 0 | 1 | Eat available food without an unnecessary purchase |

Only Emma needs an assigned workshop job for the nominal episode; Bob and Casey
are explicitly unemployed. Scenario initialization must not assign random jobs.

### Actions, causality and completion

These are proposed public behavioral rules, not instructions to drive characters
through hard-coded action lists. Strategy/GOAP must select real actions from
needs, locations, available resources and preconditions. Names cannot cause
money or inventory changes. Intermediate failures must remain observable.

| Operation | Preconditions and location | Atomic committed result |
| --- | --- | --- |
| Work and wage settlement | Assigned job; at workshop; energy at least 3; employer can pay | Employer −10 credits, worker +10, worker energy −2. One settlement per successful work action, never repeated on event replay. |
| Acquire meal | At market; buyer has at least 5 credits; stock at least 1 | Buyer −5, market till +5; market stock −1, buyer meal quantity +1. Reject partial/negative transfers. |
| Eat | Own at least one meal; any location | Consume exactly one meal; hunger −6 clamped at 0, energy +1 clamped at 10. Record consumed quantity. |
| Rest/recover | At home | Energy +3 clamped at 10; no money, inventory or hunger change. |

Critical fatigue (energy below 3) takes priority over shopping/work. Hungry
actors prefer owned food before purchasing it. An actor lacking food and money
must pursue a feasible income-producing plan. The complete loop is mandatory
for Emma; Bob and Casey demonstrate different decisions without actor-name
conditionals. Travel and changed preconditions are part of the plan.

At tick 0 enqueue one fixed `scenario_started` event, then let state/needs drive
decisions. No ambient wealth grants, restocks, LLM responses or scripted action
success events occur during the nominal episode. Work completion may produce
an explanatory event, but cannot pay the wage a second time. Once an actor has
hunger at most 2 and energy at least 6, it waits without changing resources for
the remainder of the 120 ticks. Emit one final result at tick 120.

**Required nominal outcome:** Emma completes work/wage → purchase → eat → rest;
Bob rests before purchase/eating; Casey eats without buying or working. All
finish with hunger 1, no owned meals, and energy Emma 8 / Bob 6 / Casey 7.
Final credits are Emma 5, Bob 0, Casey 0, employer 90, market 10. Market stock
is 4; three meals are consumed. Exact movement/event traces must also be stable
across repeated runs of the same version and dependency profile.

**Invariant checks at every commit:** authoritative actor and graph state
agree, inventory identities/quantities agree, no resource becomes negative,
and no failed action changes state. Credits remain **105 total**. Market stock
+ owned meals + consumed meals remain **7 total**. Action/event IDs prevent
duplicate application. UI and result writers read these states; they do not
reconstruct a success from action names or log messages.

### Failure and variation cases

| Fixed variation | Required observable behavior |
| --- | --- |
| Zero market stock | No purchase/charge/negative quantity; report unavailable food and blocked goal. No fabricated replenishment or successful nominal result. |
| Employer cannot pay | No work settlement or free income; explain the unmet resource precondition and leave balances unchanged. |
| Purchase fails before commit | All balances, stock and inventory unchanged; a later valid action may proceed without duplicate rewards. |
| Same nominal seed with renderer attached or paused | Same logical trace and final state; render timing is excluded from equivalence. |
| Save mid-episode, restart, load, continue | Same next actions/final state, logical time and RNG state; no duplicated events/wages. Corrupt saves leave the valid live world unchanged and give a recovery action. |

A variation is an expected failure-path check, not a successful nominal run.
The final summary distinguishes completed, blocked, rejected, failed and
cancelled outcomes. An unmet goal with no progress for 30 logical ticks is
reported as blocked; it must not spin indefinitely or be marked complete.
Expected waits after a satisfied goal are not stuck plans.

## 4. Planned commands and observable results

**The commands in this section are future interface requirements.** Only
`visual/minimal/test`, `--headless`, `--no-llm`, character count and FPS exist
today. Do not publish these examples as a working Quickstart before #658
implements and validates the scenario flags and behavior.

```bash
# Proposed canonical headless acceptance command, not implemented yet:
python main.py --mode minimal --scenario village-economy-v1 --seed 724 --ticks 120 --no-llm --headless --result artifacts/mvp-result.json

# Proposed Pygame view of the same scenario, not implemented yet:
python main.py --mode visual --scenario village-economy-v1 --seed 724 --ticks 120 --no-llm --fps 30 --result artifacts/mvp-visual-result.json
```

The scenario fixes actor count; conflicting `--characters` overrides fail with
an actionable explanation. Missing optional display dependencies affect only
the visual flow. Missing core dependencies, invalid scenario data or an
unwritable result path fail before world mutation. Existing legacy demo modes
must not silently claim to run the new scenario.

Headless output shows launch/seed, each meaningful decision and its reason,
successful/rejected outcomes with actual state changes, and an honest summary.
Structured results contain schema/scenario version, seed, tick count, profile
and source revision, per-actor initial/final values, goals, committed/rejected
actions, transfers, consumed quantities, invariant results and outcome.
Diagnostic wall times, platform identifiers and presentation details are
separate from the stable logical trace/digest used for repeatability checks.
Exit codes: 0 only for a completed nominal scenario with all invariants met;
2 for input/setup errors; 1 for runtime, invariant or unmet-goal failure;
130 for cancellation, with partial evidence labeled incomplete. There is no
successful fallback to a zero-action demo.

Pygame supports start, observe, select/inspect actor and building, pause/resume,
restart the identical scenario, basic save/load, recover from errors and exit.
Reuse existing Space/R/S/L/help controls where coherent; #729 resolves the
current Escape conflict between clearing selection and quitting. Display
current needs/units, inventory, credits, goal, action and reason, tick/pause
state, transfer outcomes and the final summary. A save/load error must show
what happened and what the user can do, not an exception dump.

## 5. Agreed support targets and presentation matrix

| Profile / surface | MVP target | Current evidence and release prerequisite |
| --- | --- | --- |
| Linux x86-64, Ubuntu 24.04, CPython 3.12, CPU-only core, headless | Required automated behavior/installation profile | Phase 1 used prepared Python 3.12.14 CPU/Linux; clean installation and full scenario are unverified. #725/#654/#776/#726 supply the core profile and CI. No Pygame, FAISS, spaCy, Transformers, GPU or network/model service required for core. |
| Same OS/Python plus Pygame visual profile, local display | Required presentation (D013); platform target agreed under D014 | No current visual release proof. #727–#730 define/review UX/UI/aesthetics; later implementation and Phase 8 validation must pass. SDL dummy rendering supports automation but cannot substitute for a human display/controls review. |
| Native Windows | Planned/unverified; no MVP support promise | Preserve D004: require native startup, paths/cache permissions and supported flow evidence before claiming support. Installing a wheel or simulating metadata is insufficient. |
| macOS, ARM, other Python versions | Unverified; outside the initial support claim | Require a separately recorded support expansion and native evidence. |
| Live ML/LLM, GPU, remote services | Optional later profile; not a core release condition | D005 and existing ML/LLM work, including #585, retain separate download/inference validation. |

The reference performance profile is a declared Linux x86-64 machine with two
available CPU cores, 8 GiB RAM and no GPU, using the approved locked core/visual
profiles. Record CPU model, OS/kernel, Python, package/asset hashes, renderer,
resolution and resource limits. Virtual core count alone is not comparable
hardware. Baselines must use the same declared machine/profile; unmeasured
machines are not presumed to meet the budgets below. #725 selects the exact
lock artifact, and #726 publishes the concrete runner configuration.

Visual targets: a **1280×720** default client area and supported **1024×768**
minimum; keyboard and mouse at both, with readable system scaling tested.
Propose a quiet village map, clear actor selection, and a compact needs/plan
panel. #728 owns the actual palette, typography, spacing, asset provenance and
visual references; #729 owns layouts and interaction states. #724 does not
approve a mockup or expand into a web UI. Essential status must have text/icon
cues as well as color; text needs at least 4.5:1 contrast and essential selection
cues 3:1 against adjacent backgrounds. Full journeys need visible keyboard
focus and discoverable controls, with no hover-only or color-only information.
Motion effects must be reducible and fit the frame budget. These are proposed
Pygame review criteria, not claims of browser/ARIA or general accessibility
conformance. #730 and Phase 8's #765/#766 review the actual experience.

## 6. Agreed release budgets and measurement contract

**Agreed targets under D014, not benchmark results.** These bound the three-villager
MVP on the declared reference profile. They neither change Phase 1's graph
20/25-second, cache 25/30-second or batch 110/120-second diagnostic limits nor
authorize repeated tests now. A target miss becomes an issue or explicit
maintainer revision, never a silent timeout increase or retry until green.

| Metric | Agreed acceptance budget | Definition / sample requirement |
| --- | --- | --- |
| Fresh-process ready time | At most 20 s, each of 5 launches per profile | Process launch to constructed real world and available console/UI. Packages/assets already installed, no process/model cache reuse or network downloads; OS file caches are not flushed. Report every launch, not a best run. |
| Headless episode | At most 30 s after readiness | All 120 ticks and final result written; excludes setup, includes actual planning/actions. Record all 5 complete episodes. |
| Simulation latency | p95 at most 100 ms; no tick over 500 ms | Each complete logical tick including all 3 actors, within the 5 episodes. Sleeping/rendering is excluded and measured separately. |
| Visual frame pacing | 30 FPS cap; p95 frame at most 40 ms, p99 at most 66.7 ms; no active frame over 250 ms | At least 300 active seconds at each supported resolution. Includes input, simulation, rendering, FPS limiter and routine save activity; user-paused intervals are reported separately. |
| Control feedback | p95 at most 100 ms; maximum 250 ms | At least 30 timestamped pause/inspect/help inputs per resolution, from input receipt to visible acknowledgment. Quit/cancel must release owned processes within 2 s. |
| Save / load | Each at most 2 s; responsive acknowledgment within 250 ms | At least 5 saves and 5 loads, including invalid-save rejection. Time through durable save or validated restoration, without a partial world change. |
| Memory | Process-tree peak RSS at most 1 GiB | Core and visual measured separately; include owned descendants, exclude unrelated processes. Report startup peak and active-run peak. |
| Stability | Zero unhandled exceptions, invariant violations, hidden action failures or unexpected watchdog exits | All nominal runs plus the bounded soak; missing terminal/cleanup evidence is incomplete rather than PASS. |
| Soak growth | At most 64 MiB increase after warm-up; still below 1 GiB peak | Ten real actors, 20 minutes at 2 logical ticks/s; compare process-tree RSS median over minutes 3–5 with median over minutes 18–20. |

Use monotonic measurements and retain raw samples. For percentiles, use the
nearest-rank value after sorting, state N, and report min/median/p95/p99/max.
For frames include the FPS limiter, matching `game_loop`; also report active
update/render stage costs separately so intentional pacing is not mistaken for
CPU work. Track simulation ticks separately from frames and label profiler
overhead. Report frames with active goals separately from satisfied/waiting
frames, and require both populations to meet the frame limits; idle intervals
cannot hide slow planning or action frames. No averages in place of required
tails; report every failure/skip.

The bounded soak repeats a versioned ten-worker derivative of the nominal
fixture: IDs `villager_01`–`villager_10`, each hunger 7 / energy 6 / credits 0 /
meals 0, all at home with workshop jobs; employer 1000, market 20 meals and
zero till, otherwise identical rules. Each 120-tick episode must complete ten
economic/recovery loops. Repeat 20 episodes through the supported scenario
reset path, holding each tick at 0.5 s (2400 ticks, 20 active minutes).
Record reset costs separately and in memory peaks. Fresh-world fixture resets
are explicit episode boundaries, never unlogged resource creation within an
episode. Expect 200 full loops, 200 consumed meals and 20 separate balanced
episode ledgers. An idle world does not satisfy this workload. Verify reset
is deterministic and releases old owned world/resources. #762 owns execution
and budget findings; this soak does not replace #751's nominal/variation proof.

## 7. Delivery ownership, reviews and completion criteria

| Existing issue(s) | Required handoff from this proposal |
| --- | --- |
| #724 | Agree the scenario, two surfaces, support targets and numeric budgets; record approval of the exact specification revision. Documentation only. |
| #725 → #654 (including #776) → #726 | Lock core/optional profiles; remove fake production dependency substitutions; demonstrate real reduced-capability construction, then required bounded CI. |
| #727 → #728/#729 → #730 | Flesh out journeys, visual direction and interaction states; review aesthetics, controls, legibility and failure feedback before UI implementation. |
| #731 → #732 | Human engineering-practices review and separate Phase 2 gate, using the scoped contract and actual Phase 2 evidence. |
| #740 and existing graph/GOAP/action work | Resolve units, aliases, goals, deterministic ordering, graph ownership and causal action/effect contracts. No extra copy of runtime state for the demo. |
| #195/#196/#199/#198/#750 → #751 | Implement inventory/jobs/market/atomic transactions and valid choices in existing order; prove nominal, scarcity, failure and state-conservation behavior. |
| #755/#756/#757 → #758 | Stable snapshots, logical clock/RNG/event restoration, atomic saves, restart equivalence and human recovery review. |
| #656/#658/#655 | Reconcile release docs, implement/validate the canonical invocation and result contract, and explain real decisions/outcomes. |
| #762, #765/#766/#767 and Phase 8 gate | Measure budgets and soak, perform UI/accessibility/engineering reviews and decide release readiness. |

These are acceptance-contract mappings, not new native blockers, approvals or
schedule changes. Preserve current parents/dependencies and D002's follow-ups.
Any newly discovered implementation work needs its own issue; reuse existing
issues rather than duplicate them. Project custom-field synchronization remains
unconfirmed after the earlier access denial; this specification cannot assert
that Project readiness/Decision fields have changed.

The maintainer accepted the fixture/actions, state units, invariants/outcome,
presentations, support targets and budget workload under D014, conditional on
preserving the original project's expansiveness. The guardrails in section 1
make that condition explicit. Record the approved source revision and human
rationale in `docs/decisions.md`. #724 completes as a planning issue when this
contract and its decision reach main; implementation, actual support claims
and downstream design/engineering/release reviews remain separate work.

Future release evidence must contain source/spec/lock/asset hashes, commands,
machine/profile metadata, seeds and fixture version, initial/final authoritative
states, logical trace/digest, resource ledgers, action counts/reasons, raw timing
and memory samples, exit/cleanup records, all failed/incomplete attempts and
human control/visual review artifacts. Record explicit rework or waiver decisions
in the single decision file; neither a screenshot nor an exit-0 banner proves
this contract.
