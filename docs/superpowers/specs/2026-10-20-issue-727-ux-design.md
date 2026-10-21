# TinyVillage MVP journeys, controls and simulation feedback

**Status:** Approved for planning under D023. Fifteen interaction choices
were confirmed individually by dhar174; their authoritative record, alternatives
and rationale are [D018](../../decisions.md#d018--select-mvp-ux-behavior-for-727).
This document applies those choices to #727. It authorizes no product-code work,
supplies no phase approval and does not claim these journeys work today.
The remaining walkthrough choice is now recorded in
[D021 R01](../../decisions.md#d021--review-remaining-design-proposals-for-727729);
D020 settles the overlapping menu/save/shortcut proposals. This consolidated
written artifact still awaits a complete disposition.

**Subsequent disposition:** Approved by dhar174 on 2026-10-21
(America/New_York) under [D023](../../decisions.md#d023--approve-design-planning-727729-and-combined-review-730),
including the complete written artifact and #730's distinct combined review.
Draft/pending wording below describes the original publication, superseded by
this disposition. Production trials, native evidence and downstream
implementation remain required; #731/#732 are separate decisions.

## 1. Purpose, boundaries and research

Help a first-time observer understand autonomous villagers, let a returning user
inspect and recover a session, and let a contributor reproduce outcomes without
a display. Success is an understandable connection between a goal, a recorded
decision, its committed or rejected action, and the real state change.

Preserve the [accepted #724 contract](2026-10-20-issue-724-mvp-contract.md): one
authoritative simulation, a non-interactive headless acceptance run and a Pygame
observer; `village-economy-v1`, seed 724, 120 logical ticks; the existing support,
accessibility and performance targets. D014 preserves the expansive village
vision and D015 requires final-product AI. Lightweight verification does not
establish gameplay or AI readiness. Clicking, inspecting and rendering do not
select or execute world actions. Direct character commands/building placement
remain in #382's wider scope; this document does not complete or remove them.

Research base: merged main `969f6b985f3d64d0388ffcbd523ed2e0b8816ae7` (#790).
Static inspection covered `main.py`, `demo_minimal_integration.py`,
`tiny_gameplay_controller.py` and `tiny_map_controller.py`, the #724 contract,
and issues #381/#382/#658/#727–#732/#755–#758/#765–#767. No application runs,
benchmarks, dependency installations or implementation tests were performed.

| Current source finding | Required downstream change |
| --- | --- |
| Controller initializes `paused=False`; minimal demo explicitly reports zero gameplay actions. | Add the agreed paused visual start in the canonical scenario; keep startup smoke and real episode acceptance distinct. |
| Controller handles Escape as quit; map controller has a separate Escape-dismiss path. Event routing chooses which handler receives input. | #729 must define one consumed-input route implementing D018's dismiss/menu policy; do not invoke both actions for one input. |
| Existing Space/R/S/L/H/F1 bindings provide a starting point; Help largely logs to the console. | Retain coherent shortcuts, add visible in-window guidance, keyboard focus and a session menu. |
| Existing save writes partial character/global data directly; load does not recreate characters or restore the full world. | #755–#757 must provide versioned, atomic, complete restoration before this UX can claim success. |
| Character/building selection and information panels exist, but their existence is not a usability result. | #729/#381 must expose the agreed state and evidence; Phase 8 verifies real interactions. |

## 2. Journeys and information

### Launch, observe and inspect

Visual launch validates dependencies/scenario inputs, constructs the real world,
then shows the village **paused at tick 0**, with a clear Play control and a
short dismissible controls hint. Help is always available through a button and
H/F1. No launch screen or mandatory tutorial is required. A setup failure gives
an actionable explanation instead of entering a partially initialized session.

Play starts at 1×, defined as two logical ticks/second. Provide Play/Pause,
single-tick advance while paused, and 0.5×/1×/2×/4× presets, corresponding to
1/2/4/8 ticks/second. Step commits exactly one whole logical tick using the
same ordered simulation as playback, then remains paused. Disable Step while
running, finished, recovering, or applying a save/load operation. Speed changes
wall-clock pacing only; it cannot alter logical decisions, RNG use or results.

Select an entity by clicking the map or choosing it from a stable keyboard list.
Provide villagers and buildings, with names and disambiguating stable identity.
Arrow keys navigate the focused list; Enter selects. Movement does not reorder
entries or change focus. Selection highlights the same entity on the map and
opens its inspector without pausing. While paused, selection and inspection
remain usable and do not mutate the world. If an entity disappears or a world
is replaced, clear invalid selection and explain it rather than showing stale
data. Search, a spatial keyboard cursor and direct world commands are not needed
for this inspection journey; their alternatives are recorded in D018.

| Surface | Information users must be able to find |
| --- | --- |
| Session status | Scenario, current tick/game time, population, playback speed and an explicit Paused / Running / Finished / Recovering state. |
| Character inspector | Name/identity, location, hunger and energy with units/direction, credits, inventory quantities, active goal/action, short reason and latest actual outcome. Basic recent memory is distinguishable from semantic/AI services. |
| Building inspector | Name/identity, purpose/location, relevant stock/resources and available or unavailable interactions with reasons. Information is read-only for this observer scope. |
| Event history | Ordered meaningful decisions, transfers, rejected actions, blocked goals and recovery events, with tick and actor/target identity. Selection can link to the relevant event; important status is also visible without opening history. |
| Expanded details | Recorded preconditions, rejection cause, before/after values and action/event identifiers where the engine supplies them. Missing evidence is labeled unavailable. |

Use short, evidence-based explanations by default, with expandable details.
For example, “Going to the workshop to earn money for food” is permitted only
when the recorded goal/plan supports it. Report committed transfers from real
state, not action names or narrative text. Label a planned action separately
from an attempted, completed, rejected or blocked one. Do not invent motives,
model reasoning, sentiment or successful effects to fill a display.

An ordinary blocked goal keeps other villagers running. Show a persistent
blocked status, its unmet condition and a history entry. No global pause,
automatic resource injection or false completion follows. The scenario's
existing blocked-goal/final-result rules remain in force. A state-integrity or
execution failure uses the recovery journey below instead.

### Pause, menus and input

Space and a visible control explicitly toggle normal playback. Selection and
opening an inspector do not toggle it. Escape dismisses the topmost dismissible
panel/dialog; with none open it opens the session menu and pauses advancement.
The menu exposes Resume, Quicksave (S), Save… for named saves, Load, Restart and
Quit. Each is labelled and focusable; menu Quicksave uses the same dedicated
slot and state checks as S, while keeping the menu open and playback stopped.
An unfinished menu session stays paused; a completed world retains Finished.
Final-state inspection and recovery cannot use Resume to advance an ended or
untrusted world.

Closing a dialog must not silently start playback. Text input and focused
controls consume their keys before global shortcuts; typing a save name cannot
trigger restart, load or save. OS window-close uses the same unsaved-progress
protection as Quit. A recovery notice cannot be dismissed into unsafe playback.

| Operation | Required discoverable input |
| --- | --- |
| Play/Pause | Visible control and Space outside text input. |
| Select/inspect | Left-click map entity; focused entity list with arrows/Enter. |
| Step and speed | Labeled controls reachable by Tab/Shift+Tab and Enter/Space. Additional direct shortcuts are optional #729 design, not a new #727 requirement. |
| Help | Visible Help and H/F1. |
| Session/back | Escape with the ordered dismiss/menu semantics above. |
| Save/Load/Restart/Quit | Visible session-menu actions, usable by keyboard and mouse. Existing S/L/R mappings must be reconciled in #729 with naming input and confirmation, rather than retained as unsafe immediate world changes. |

Input focus is visible, ordered and contained within an active modal dialog,
then returns to its invoking control. Help documents the actual bindings and
current mode; essential actions are never hover-only, color-only or available
only through memorized keys. #729 owns exact panel placement and final shortcut
mapping; #728 owns palette/type/spacing/assets. Section 5 identifies the
confirmed selections and the remaining complete written-review boundary.

### Save, load, restart and exit

Save opens an in-game list of named saves, with scenario, logical tick and save
time. Users can create a named save or explicitly replace one after confirmation.
Quicksave uses a dedicated single slot without repeated overwrite prompts;
always show success or failure. Keep diagnostic save time separate from logical
simulation time. Persistence operations use a coherent tick boundary and signal
busy/success/failure; a spinner or request acknowledgment is not save success.

Before Load, Restart or Quit discards unsaved progress, offer **Save and Continue,
Discard and Continue, Cancel**. Progress means unsaved authoritative simulation
changes, not selection, help, speed or a pause toggle. Save and Continue advances
to the requested transition only after durable success; failure leaves that
transition blocked with actionable choices. Cancel preserves the current world.
A failed overwrite preserves the previous valid save. Do not silently create a
new recovery/version history or fall back to different content.

Load lets users identify a save before replacing the world. Validate the complete
snapshot and scenario/version compatibility before committing restoration. A
failed or corrupt load leaves the live world intact and offers another save,
cancel or diagnostic details. A successful load restores the real world, pending
events, memory, logical time and RNG state, with the save and restored tick
identified. An unfinished world loads **paused**; a completed world loads as
**Finished**, with inspection available and Play/Step disabled.
Play is explicit for unfinished worlds. This presentation choice
does not change the saved causal trace or continuation equivalence.

Restart requests the identical scenario/seed from tick 0, using the same unsaved
guard, then returns to the paused launch state. It does not regenerate random
characters. Quit uses the guard, ends the session, releases owned resources and
preserves any promised result/diagnostics; cancellation is not nominal success.
Phase 7 owns persistence correctness, not this planning deliverable.

### Failure recovery and scenario completion

Recoverable file/UI errors give a visible explanation, affected operation and
next action while retaining valid world/save data. Runtime or state-integrity
failure stops simulation advancement and preserves diagnostics. Explain what
failed and expose only actions safe for that failure: Load, Restart, Export
Diagnostics and Quit as applicable. Retry is available only for a known-safe
repeatable operation. Never offer ordinary Save for untrusted state, or silently
resume/restore/retry in a way that fabricates a successful episode. Diagnostic
export failure is itself reported; unavailable actions explain why.

At the agreed final tick, stop advancing and keep the village inspectable.
Display **Finished**, distinct from Paused, and a persistent non-modal summary:
completed/blocked/failed outcome, goals, committed/rejected actions, transfers,
consumption and invariant results, plus the written report's location/status.
Do not imply success if reporting failed. Restart, Load and Quit remain available;
Play/Step cannot extend this bounded episode. The final tick is a fixture bound,
not a restriction on future open-ended village simulations.

### Headless contributor journey

The future canonical invocation remains the one specified by #724/#658; its
scenario/result flags must not be advertised as working until implemented.
Headless execution never waits for Play, confirmation, save names or keyboard
selection. It prints concise meaningful decision/reason/outcome lines and a
readable final summary, writing structured trace/results to the requested file.
Verbose output adds diagnostics, not required user-facing explanations. No
extra live structured stream or graphical error dialog is required.

Retain the accepted result contract: 0 only for nominal completion with all
invariants met, 2 for input/setup errors, 1 for runtime/invariant/unmet-goal
failure and 130 for cancellation with incomplete evidence. Current startup
smoke and existing legacy exit behavior are not proof of this future contract.

## 3. Human usability walkthrough and acceptance evidence

These are proposed executable review scenarios, not executed results. Use real
domain objects and actual state; no staged successes or simplified replacement
characters. Under D021 R01, a maintainer exercises every scenario without
coaching beyond in-product guidance; include first-time observers when available
and report their experience separately. Each row must pass with mouse and
keyboard access at 1280×720 and 1024×768 where relevant. The human records
pass/fail, unexpected behavior and assistance required; SDL automation alone
cannot approve human usability.

| Scenario | Observable success criterion |
| --- | --- |
| First launch and discoverability | From paused tick 0, find Play, Help and inspection using only the hint/visible controls. Dismiss hint, reopen Help, and return without accidentally starting time. |
| Inspect before and during playback | Select each entity type by map and keyboard list, identify needs/units/resources and the goal/reason/outcome. Selection alone never pauses; explicit Pause freezes world advancement while inspection still works. |
| Time controls | Step advances exactly one full tick and remains paused. Every speed preset changes pacing only; repeated traces/final state match the canonical headless run. |
| Understand causality | Explain why the fixture villagers work, buy/eat or rest, and identify the actual balances/inventory changes from UI evidence without a developer supplying the answer. Expanded details agree with authoritative state. |
| Blocked goal | Run the zero-stock or unpaid-employer variation: explain the unmet condition from inspector/history while other feasible activity continues. No partial transfer or false success appears. |
| Navigation and menus | Complete essential journeys with visible keyboard focus and without hover. Escape consumes one intended dismiss/menu action; naming input never triggers global shortcuts; opening the session menu stops advancement. |
| Save and overwrite | Create a named milestone, cancel then confirm replacement, and quicksave twice. Metadata identifies each successful save. Inject write failure: previous valid save survives; no success message or pending transition occurs. |
| Load and continuation | From unsaved progress exercise all three guard choices. An unfinished valid load returns paused at the saved tick; after Play its continuation matches uninterrupted execution. Completed saves load as Finished with inspection and disabled Play/Step. Corrupt/incompatible saves preserve the live world and explain recovery. |
| Restart and quit | Cancel leaves the world intact; confirmed restart returns the same fixture/seed paused at tick 0. Menu Quit and window-close honor the same guard; owned resources terminate within the accepted budget. |
| Runtime recovery | Controlled safe failure stops advancement and preserves evidence. Only safe actions appear, an unavailable reason is understandable, and no unsafe Save/Resume/Retry or unannounced rollback is possible. |
| End of episode | At tick 120, distinguish Finished from Paused; inspect final entities and find the non-modal outcome/invariants/report. Tick does not increase; Restart/Load/Quit remain accessible. |
| Headless outcomes | Follow readable events and final summary without prompts; verify actual structured output and accepted exits for nominal, setup error, blocked/failure and cancellation cases. |

Measure against #724's existing budgets, without inventing easier ones: control
feedback p95 ≤100 ms/max 250 ms over at least 30 pause/inspect/help inputs per
resolution; save/load each ≤2 s and acknowledgment ≤250 ms over at least five
saves and five loads including invalid-save rejection; owned-process quit/cancel
cleanup ≤2 s. Retain existing startup, simulation, frame, memory and stability
budgets too. Text contrast is at least 4.5:1, essential selection cues 3:1;
pair color with text/icon cues and review reduced motion/focus at both sizes.
These are Pygame targets, not a claim of browser accessibility conformance.

Retain reviewed spec/source/lock/asset revisions, native OS/display/scaling,
commands, event/state traces, screenshots or recordings, actual timing samples,
terminal/cleanup evidence and the human's rationale. Missing evidence is
incomplete. Link each required remediation to its own issue/blocker; a passing
CI run cannot supply the later human approval.

## 4. Ownership and review boundaries

| Issue(s) | Handoff |
| --- | --- |
| #727 | Review and freeze this journey/feedback specification; no runtime implementation or usability-pass claim. |
| #728 | Visual direction, palette, typography, spacing, legibility, asset licensing and references, respecting these interactions. |
| #729 / #381 | Wireframes, HUD/inspectors/history, input ownership/focus, menu/dialog/busy/error states and exact shortcuts. |
| #730 | Distinct human UX/UI/aesthetic review of #727–#729 before UI implementation authorization. |
| #731 / #732 | Engineering-practices review and separate Phase 2 decision gate. |
| Existing simulation tasks / #751 | Real action causality, logical ticks, units, blocked goals and headless/visual trace equivalence. |
| #755–#757 / #758 | Snapshot schema, complete atomic save/load and continuation, then human persistence/recovery review. |
| #658 / #655 | Canonical command, readable events, structured report and truthful documentation of implemented behavior. |
| #765 / #766 / #767 | Human interaction/state, inclusive-interaction and release-readiness evidence using the walkthrough above. |

Keep #727 open until the human reviews the written specification. Its acceptance
is a planning deliverable; it does not close #381/#382, approve #730/#732 or
demonstrate future features. No new runtime API/schema, dependency, lock, CI
configuration or platform-support claim is introduced here.

## 5. Consolidated selections and written-specification review

The 15 initial choices are recorded in D018. The earlier connective proposals
now have explicit selections in D020 and D021; those single-file records retain
their alternatives and rationale. Their implementation requirements are:

| Detail | Selected requirement | Record |
| --- | --- | --- |
| Leaving menus/dialogs | Remain paused; Play/Resume is explicit. | D020 U07 |
| Data during save/load | Hold a coherent tick boundary until saving finishes, with responsive UI; load validates before replacement and remains non-advancing. | D020 U08 |
| Session/Step/speed shortcuts | S Quicksave, L Load selection, R guarded Restart, period single Step and Page Up/Down speed, with focusable controls. Focused paging consumes Page Up/Down before global speed. | D020 U05/U11; D021 R09 |
| Walkthrough method | Maintainer exercises every scenario with only in-product guidance; include first-time observers when available and report their results separately. | D021 R01 |

Review this complete specification, including the journey and evidence tables,
before recording #727's written disposition in docs/decisions.md. #728 and #729
are also consolidated for review; keep their dispositions distinct. UI
implementation still awaits the roadmap's combined design and later review
prerequisites. No executed usability evidence is supplied by these selections.
