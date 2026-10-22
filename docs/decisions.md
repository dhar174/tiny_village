# Project decisions

This is the single maintained record of human decisions from the project review
walkthrough. Add future decisions here. Evidence reports and task summaries
should link to this file rather than maintain separate decision records.

## Recorded decisions

### D001 — Defer startup engineering review #721

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Disposition:** Deferred
- **Reviewed revision:** `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`
  (merged PR #778).
- **Decision:** Keep #721 open until #779 explains the startup timing failures
  and required verification completes with retained evidence. Keep phase gate
  #722 open and downstream phase advancement blocked.
- **Reason:** Merged verification has import timeouts, incomplete lifecycle and
  ownership batches, and one cache-loader integration timeout. Passing targeted
  diagnostics show variability without establishing a cause or replacing the
  failed and incomplete runs.
- **Confirmation:** The user answered “Yes” to recording Deferred pending #779.
- **Evidence:** [Merged verification](testing/issue-721-startup-review.md) and
  [recorded GitHub disposition](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6194738536).
- **Follow-up:** [#779](https://github.com/dhar174/tiny_village/issues/779) is a
  verified native blocker of [#721](https://github.com/dhar174/tiny_village/issues/721).

### D002 — Keep wider repairs in their existing phases

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep the following issues in their existing phases and preserve
  their existing dependencies and review requirements:

  | Issue | Scope retained |
  | --- | --- |
  | [#773](https://github.com/dhar174/tiny_village/issues/773) | Integration fixtures and fallback action behavior |
  | [#774](https://github.com/dhar174/tiny_village/issues/774) | Goal contracts and explicit falsy action IDs |
  | [#775](https://github.com/dhar174/tiny_village/issues/775) | Missing Open Door effects and production action behavior |
  | [#776](https://github.com/dhar174/tiny_village/issues/776) | Character memory behavior when spaCy is unavailable |

- **Boundary:** These issues remain open follow-ups; this scope decision does
  not claim their behavior works or their acceptance criteria are complete.
  Require #779 before approving #721. Any newly diagnosed prerequisite must be
  made explicit rather than silently expanding or relaxing review scope.
- **Reason:** Separate the focused startup investigation from the broader
  integration, contract and feature repairs already tracked in the roadmap.
- **Confirmation:** The user replied “I agree” to keeping #773–#776 in their
  existing phases while requiring #779 before approving #721.

### D003 — Maintain decisions in one Markdown file

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Maintain review decisions in this file, including their scope,
  rationale, confirmation, evidence and outstanding actions.
- **Confirmation:** “And, just put all decisions in a single MD file.”
- **Existing records:** Preserve historical GitHub comments and validation
  evidence as provenance. Use this file for subsequent decision updates.

### D004 — Defer native Windows validation to platform work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep native Windows execution and cache-permission validation
  in the later platform work, rather than add it as a prerequisite for #721.
  The current review remains focused on the Linux startup evidence and #779.
- **Boundary:** Windows support and Windows cache isolation remain unverified.
  Require native Windows validation before claiming Windows support; Linux
  tests and simulated Windows metadata do not establish it.
- **Reason:** Resolve the current startup verification failures within their
  existing environment and address platform-specific guarantees separately.
- **Confirmation:** “Yes, save it for later,” in response to deferring Windows
  validation to the later platform work.

### D005 — Defer live model validation to ML/LLM work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep real model downloads and inference validation in the
  later ML/LLM work. Use offline startup checks for #721, including the
  existing cache integration checks with controlled model-loading boundaries.
- **Boundary:** Offline checks do not establish live download, transport,
  artifact verification or inference behavior. Preserve existing verification
  settings; this decision does not permit disabling TLS or checksum checks.
  The cache-loader timeout remains part of required investigation #779.
- **Reason:** Validate startup and cache configuration within the prepared
  environment without expanding this review into live model integration.
- **Confirmation:** The user answered “Yes” to keeping real downloads and
  inference in later ML/LLM work while #721 uses offline startup checks.

### D006 — Defer real gameplay-turn validation to integration work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep completion of a real gameplay turn in the later demo and
  integration work, rather than make it a prerequisite for Phase 1 approval.
  Phase 1 remains focused on reliable startup within the agreed environment.
- **Boundary:** The minimal-mode probe exits successfully but executes zero
  actions. Its result establishes no completed gameplay turn or full simulation
  readiness. Later integration validation must demonstrate actual behavior.
- **Reason:** Separate the startup repairs from end-to-end simulation behavior,
  retaining the latter as explicit follow-up work.
- **Confirmation:** “Later, yes,” in response to scheduling gameplay-turn
  validation in later demo/integration work.

### D007 — Approve the bounded #779 investigation

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Execute the [investigation plan](superpowers/plans/2026-10-15-issue-779-startup-investigation.md)
  sequentially within its diagnostic budget and unchanged verification limits.
- **Boundary:** Investigation only. Bring findings back before applying a
  runtime, dependency, persistent configuration or deadline-contract remedy.
  #721 remains Deferred; #722 remains open.
- **Confirmation:** “Approved, proceed,” in response to the investigation-only
  plan approval question.

### D008 — Approve durable failure capture and bounded verification for #779

- **Date:** 2026-10-16
- **Decision-maker:** dhar174
- **Decision:** Implement the [failure-capture plan](superpowers/plans/2026-10-16-issue-779-failure-capture.md)
  sequentially, including its focused real-process checks and single nine-batch,
  three-probe application observation window.
- **Boundary:** Test-harness diagnostics only; preserve original assertions,
  deadlines, runtime, dependency and cache behavior. No automatic retries,
  issue closure, phase approval or delegation. Bring any unexplained timeout
  or proposed runtime/environment correction back with evidence.
- **Confirmation:** “Looks good. Approved.” in response to the saved plan.

### D009 — Approve one post-merge verification window

- **Date:** 2026-10-16
- **Decision-maker:** dhar174
- **Decision:** Verify merged `main` once with the 13 capture checks,
  81 existing startup-matrix methods and three original probes. Preserve all
  original assertions, child deadlines and 110/120-second batch limits.
- **Boundary:** One sequential window, no automatic retries or expanded
  diagnosis. Bring results back for the #779 closure/risk decision and the
  separate human reviews of #721 and #722. This execution approval does not
  accept the unexplained timeout risk, close issues or approve either gate.
- **Confirmation:** “Proceed,” following the ordered post-merge next steps.

### D010 — Accept residual uncertainty and close investigation #779

- **Date:** 2026-10-17 UTC (2026-10-16 EDT)
- **Decision-maker:** dhar174
- **Decision:** Accept the unknown historical timeout cause and the D009
  lifecycle supervisor completion gap; authorize closing #779 as **unable to
  reproduce; diagnostics added**. This is an accepted-risk disposition, not
  a demonstrated root-cause fix or fulfillment of every original criterion.
- **Evidence:** [Post-merge report](testing/phase-1-post-merge-verification.md)
  at main `77e4b9437cc00970def933d992140e6da37eae8e`: 94 successful method
  summaries (82 with confirmed supervisor completion, 12 with the documented
  supervisor gap), three zero-exit probes and 42 verified child artifact sets.
  Diagnostics merged in #781; evidence is published in #782.
- **Boundary:** Preserve the original failures and missing terminal record.
  Reopen #779 if the timeout recurs and retain the diagnostic artifacts before
  proposing further investigation. This decision supplies no approval of #721
  or #722 and authorizes no additional observation window or phase advancement.
- **Confirmation:** “yes” to: “Do you accept the unknown historical timeout
  cause and this reporting gap, and authorize closing #779 as ‘unable to
  reproduce; diagnostics added’?”

### D011 — Approve startup engineering review #721

- **Date:** 2026-10-16 (America/New_York)
- **Decision-maker:** dhar174
- **Disposition:** Approved
- **Reviewed revision:** `77e4b9437cc00970def933d992140e6da37eae8e`
  (merged startup repairs and capture diagnostics).
- **Decision:** Approve #721 for the agreed Linux/offline startup scope and
  close the engineering-review issue. This supersedes D001's Deferred
  disposition for the current review; the original failed runs remain evidence.
- **Rationale and evidence:** Startup fallback, graph lifecycle/ownership and
  cache repairs are merged. The [post-merge report](testing/phase-1-post-merge-verification.md)
  retains 94 successful method summaries and three zero-exit probes. 82 methods
  have confirmed supervisor completion; the other 12 retain successful summaries
  and child exits with the supervisor reporting gap accepted under D010.
  The historical timeout cause remains unknown; #779's accepted-risk closure
  does not establish a root-cause fix.
- **Boundary:** Native Windows, live models and real gameplay remain deferred
  under D004–D006; #773–#776 retain their existing scope under D002. This does
  not approve phase gate #722, close parent #717 or advance downstream work.
- **Confirmation:** “Yes, approved. **#721’s engineering review**,” explicitly
  for the agreed Linux/offline startup scope and documented limitations.

### D012 — Approve Phase 1 gate #722 and complete phase #717

- **Date:** 2026-10-17 (America/New_York)
- **Decision-maker:** dhar174
- **Disposition:** Approved
- **Reviewed revision:** `77e4b9437cc00970def933d992140e6da37eae8e`,
  with the [post-merge evidence](testing/phase-1-post-merge-verification.md)
  published in PR #782.
- **Decision:** Approve #722 for the agreed Linux/offline startup scope,
  complete Phase 1 parent #717, and allow Phase 2 to begin. Close #722 and
  #717 as completed; preserve the native dependency links and later gates.
- **Rationale:** The #718–#720 repairs are merged and closed. D011 records
  human approval of engineering review #721. The evidence retains 94 successful
  method summaries and three zero-exit probes; D010 explicitly accepts the
  unknown historical timeout cause and the 12-method lifecycle supervisor
  reporting gap. This decision accepts Phase 1 within those documented limits.
- **Boundary:** Windows validation, live models and real gameplay stay in
  later work under D004–D006; #773–#776 remain in their existing phases under
  D002. This gate approval does not establish broader platform, inference or
  gameplay readiness or approve any subsequent phase gate. The next ordered
  task is #724, defining the MVP scenario, support matrix and release budgets.
- **Confirmation:** “Yes” to: “Do you approve completing Phase 1 within the
  same Linux/offline scope and accepted limitations, allowing Phase 2 to begin?”

### D013 — Require headless and Pygame presentations for the MVP

- **Date:** 2026-10-20 (America/New_York)
- **Decision-maker:** dhar174
- **Decision:** Require a reproducible headless scenario plus a small Pygame
  view of that same simulation for the MVP release.
- **Reason:** Preserve automated behavioral proof and a player-facing village
  view, sharing the authoritative simulation rather than separate demo logic.
- **Confirmation:** Selected “Headless scenario + Pygame view (recommended)”
  in response to the #724 presentation-scope question.
- **Boundary:** This selects the two required presentations. It does not
  approve the draft's detailed fixture, new CLI, support claims, numeric budgets,
  visual design or implementation plan, nor complete #724 or later reviews.
- **Artifact:** [Draft #724 MVP contract](superpowers/specs/2026-10-20-issue-724-mvp-contract.md).

### D014 — Accept the MVP contract while preserving the original vision

- **Date:** 2026-10-20 (America/New_York)
- **Decision-maker:** dhar174
- **Reviewed artifact:** PR #783 source `a42c447d05f3248ce27863fda36b604cb93ac17e`,
  including the [MVP contract](superpowers/specs/2026-10-20-issue-724-mvp-contract.md).
- **Decision:** Accept #724's planning contract with the condition that the
  MVP preserves the expansiveness of the original TinyVillage plans. Treat the
  small deterministic scenario as the first release's proof of behavior, not
  a replacement for the broader product vision or a cap on the shared engine.
- **Condition made concrete:** Preserve autonomous characters, personalities,
  relationships, jobs/economy, memory/storytelling, optional LLM integration,
  persistence and village expansion in their existing planned scope. Keep the
  fixture data-driven and optional features available through explicit profiles.
  Do not remove features or narrow the long-term scope without a separate
  explicit decision. Section 1 of the contract carries these guardrails.
- **Confirmation:** “Merged, sounds good as long as the MVP doesn't dilute
  the expansiveness of the original TinyVillage plans.” This explicit qualified
  acceptance supplies the disposition; a PR merge alone would not.
- **Boundary:** Targets remain future requirements, not verified functionality
  or benchmark results. #730/#731/#732 and later release gates remain required.
- **Integration finding:** #782 merged into main as `79e5683`; #783 subsequently
  merged into #782's old branch as `a9ed214`. The MVP contract therefore needs
  a focused integration PR to main. Subsequently #784 merged as
  `8d6b254357fadff390e3dae228acd7f1e55f5b4a`, bringing the contract to main and
  closing #724. The next ordered delivery task is #725.

### D015 — Require AI capabilities in the final product

- **Date:** 2026-10-20
- **Decision-maker:** dhar174
- **Disposition:** Product requirement clarified
- **Decision:** AI capabilities are a core component of the final TinyVillage
  product. Their optionality applies to lightweight testing; it does not make
  AI an optional final-product feature.
- **Effect on #654/#776:** Support testing without the heavy AI dependencies
  while preserving real AI integration and explicit capability reporting.
  Successful execution with AI disabled does not establish final-product
  readiness. Keep dependency profiles and separate AI integration validation.
- **Confirmation:** “The AI capabilities are only optional for testing
  purposes; they will be a core component of the final product.”

### D016 — Approve #654/#776 lightweight testing implementation

- **Date:** 2026-10-20
- **Decision-maker:** dhar174
- **Disposition:** Implementation approved
- **Decision:** Implement the [reviewed plan](superpowers/plans/2026-10-20-issue-654-lightweight-runtime.md)
  with real basic memory recording/recent recall, real startup and event
  registration, and explicit disabled semantic operations. Gameplay-turn proof
  remains #751/#658. Require explicit minimal mode; reject visual/headless CLI
  combinations before initialization. D015's final-product AI requirement stands.
- **Confirmation:** The user selected these three choices during planning and
  explicitly requested implementation of the complete plan.
- **Limits:** This is implementation authorization, not phase-gate approval,
  gameplay acceptance, live-model validation, or permission to merge the PR.

### D017 — Implement supported-profile CI; ML checks manual only

- **Date:** 2026-10-20 (America/New_York).
- **Decision-maker:** dhar174.
- **Decision:** Implement the [reviewed #726 plan](superpowers/plans/2026-10-20-issue-726-supported-profile-ci.md)
  with required core/dev/visual checks and manually triggered NLP/CPU-LLM checks.
  No ML schedule. Preserve D015's required final-product AI scope.
- **Confirmation:** Selected “Manual only,” then “Yes, implement this plan.”
- **Boundary:** Human PR review precedes merge. Verify hosted main/manual runs
  and effective required-check enforcement before closing #726. No automatic
  merge or phase-gate approval; gameplay and inference remain separate.

### D018 — Select MVP UX behavior for #727

- **Date:** 2026-10-20 (America/New_York).
- **Decision-maker:** dhar174.
- **Disposition:** Fifteen interaction choices confirmed individually;
  consolidated written specification awaits review.
- **Artifact:** [Draft #727 UX specification](superpowers/specs/2026-10-20-issue-727-ux-design.md).
- **Purpose:** Make autonomous activity understandable and inspectable through
  the two accepted presentations, preserving D014's wider vision and D015's
  required final-product AI. This remains observer UX for the bounded scenario.
- **Confirmation:** Each selected entry below is the user's submitted answer
  to its corresponding multiple-choice question. The recommended choice was
  explicitly selected each time; no selection is inferred from silence.

| ID / topic | Confirmed selection | Other options presented | Recommendation rationale |
| --- | --- | --- | --- |
| UX01 — Visual launch | Start paused with the village visible and a clear Play control. Headless remains automatic/non-interactive. | Start running immediately; show a Start/Load/Help launch screen. | Orient and inspect before events advance, without adding another screen. |
| UX02 — Inspection | Keep inspection live; pause/resume explicitly. | Pause automatically when the inspector opens; offer separate live/paused inspection modes. | Observe changing autonomous activity without selection repeatedly interrupting it. |
| UX03 — Explanations | Brief plain-language reasons and outcomes, with technical details expandable. | Full decision/state details by default; compact status with explanations only on demand. | Explain causality without overwhelming observation; use recorded evidence and label missing reasons. |
| UX04 — Playback | Play/Pause, advance one tick while paused, fixed 0.5×/1×/2×/4× speeds. | Play/Pause only at two ticks/s; Play/Pause/Step with a continuous slider. | Combine precise inspection and reproducible pacing. 1× stays two ticks/s; pacing cannot change logical outcomes. |
| UX05 — Save organization | Named saves in an in-game list, plus quicksave. | Fixed numbered slots; one quicksave slot only. | Preserve identifiable milestones with quick repeated saving. Show scenario, tick and save time. |
| UX06 — Escape | Dismiss the topmost open panel/dialog; otherwise open a paused session menu. | Always open the session menu; dismiss UI then otherwise request quit confirmation. | Familiar backing out with explicit access to Resume/Save/Load/Restart/Quit. |
| UX07 — Unsaved progress | Offer Save and Continue / Discard and Continue / Cancel only when Load/Restart/Quit would lose unsaved progress. | Always confirm; automatically create a recovery save and continue without a prompt. | Protect actual progress without habitual unnecessary prompts. A failed requested save blocks the transition. |
| UX08 — Blocked goals | Keep other villagers running; highlight the blocked goal/reason in inspector and history. | Pause on every newly blocked goal; make pause-on-blockage configurable. | Retain autonomous observation while clearly explaining unmet conditions and honest final outcomes. |
| UX09 — Runtime failure | Stop advancement, preserve diagnostics, offer only recovery actions known safe for that failure. | Automatically restore a validated checkpoint; end the session with a report and Load/Restart options. | Avoid unannounced rollback or unsafe continuation. Retry only when safe; no ordinary Save of untrusted state. |
| UX10 — Headless output | Readable decision/outcome lines and final summary, structured results in a file. | JSON Lines events plus readable summary; readable console plus a separate structured event stream and result file. | Serve observers and automated verification without another live output destination. Accepted result/exit contracts remain unchanged. |
| UX11 — Keyboard selection | Stable entity list with arrow navigation and Enter selection. Map clicking remains available. | Spatial keyboard cursor; entity-name search overlay. | Keep moving villagers discoverable and selectable, with logical Tab order and visible focus for ordinary controls. |
| UX12 — Final tick | Keep the village inspectable with a persistent non-modal summary; stop advancing. | Modal result dialog before inspection; close window after writing results. | Connect final outcome/invariants to real village state; distinguish Finished from Paused. |
| UX13 — After load | Always load paused and require explicit Play. | Restore saved running/paused state; offer Resume after loading in the dialog. | Let users confirm the restored world; do not change saved causal state or continuation. |
| UX14 — Replacement | Confirm named-save replacement; quicksave replaces its dedicated slot without a prompt and reports success/failure. | Version every named save/quicksave; confirm every replacement including quicksave. | Protect milestones while keeping quicksave useful. Failed writes preserve the previous valid save. |
| UX15 — Guidance | Short dismissible launch hint, always-available Help and visible shortcut labels. | Guided first-use walkthrough plus Help; Help/labels without launch guidance. | Make essentials discoverable without a mandatory tutorial; Help is available by button and H/F1. |

- **Boundary:** These answers approve the interaction choices and their stated
  rationale, not a written artifact that did not yet exist. They allow the
  choices to be consolidated for review. The specification's explicit proposed
  defaults and walkthrough protocol are not separately approved by these
  answers. #727 remains open; #728/#729, human design review #730, engineering
  review #731 and gate #732 remain separate. No runtime implementation,
  persistence correctness, live AI, usability pass or automatic merge is
  authorized or established by this record. All subsequent human dispositions
  remain in this file under D003.

### D019 — Select pixel-art direction for #728

- **Date:** 2026-10-20 (America/New_York).
- **Decision-maker:** dhar174.
- **Disposition:** Pixel-art preference and 13 individual visual choices
  confirmed; consolidated written specification and proposed defaults await review.
- **Artifact:** [Draft #728 visual specification](superpowers/specs/2026-10-20-issue-728-visual-design.md).
- **Starting preference:** “It should be pixel art though I think. Lets go
  through the options.” Retain the previously agreed warm, readable 2D direction.
- **Confirmation:** The user explicitly submitted the selected answer for each
  row below after reviewing alternatives and recommendations. The palette
  comparison was generated after the user selected visual comparisons, then
  the user chose A. No choice is inferred from silence or a PR merge.

| ID / topic | Confirmed selection | Other options presented | Recommendation rationale |
| --- | --- | --- | --- |
| V01 — Perspective | Angled top-down: building fronts, roofs and faces on a straightforward 2D map. | Strict overhead; isometric; side-view/cutaway. | Balance expressive village art and readable geography with the existing 2D runtime. |
| V02 — Source scale | 32×32 ground tiles and moderately detailed character sprites. | 16×16 simplified art; 48×48/64×64 higher-detail art. | Recognizable objects and expression with manageable asset work. Buildings may span tiles; this sets no village-size limit. |
| V03 — Atmosphere | Warm, grounded village life: natural colors, expressive villagers and believable materials. | Whimsical storybook; rustic/subdued. | Support everyday activity and emergent stories without flattening hardship or important status. |
| V04 — HUD treatment | Restrained pixel-art frames, quiet high-contrast backgrounds and readable text. | Decorative wood/parchment panels; minimally decorated flat panels. | Keep the interface cohesive while prioritizing explanations and controls. |
| V05 — Typography | Conventional readable information text; pixel lettering only for short headings/decorative labels. | Pixel fonts throughout; conventional fonts throughout. | Preserve readability of explanations, punctuation and numbers at both supported sizes. |
| V06 — Palette | A — Meadow & Terracotta: welcoming greens, warm timber/clay roofs and restrained teal accents. | B Forest & Ochre; C Sage & Slate, shown on the same sample scene/HUD. | Match the selected warm, grounded atmosphere. The generated board is a palette reference, not an exact spritesheet, wireframe or measured contrast result. |
| V07 — Selection | Contrasting ground ring plus thin sprite outline; selected identity also appears in the inspector. | Sprite outline alone; overhead selection marker/chevron. | Locate the selected villager against varied ground using shape and light/dark contrast alongside color. |
| V08 — Occlusion | Fade occluding foreground art locally while the villager is selected. | Opaque scenery with through-scenery silhouette; temporary position marker above the occlusion. | Keep the actual selected sprite/activity visible. Requires foreground layering and a stable paused/reduced-motion treatment. |
| V09 — Individuality | Shared proportions/reusable art templates with curated hair, clothing and accessory variation. | Bespoke sprite/animation sets per villager; a small shared set identified mainly through names/details. | Give growing populations recognizable appearances without requiring runtime customization or identity-by-color alone. |
| V10 — Animation | Short walk cycles, clear activity poses and a small amount of ambient motion. | Minimal movement frames/static activity/no ambient motion; rich activity/scenery/weather/lighting animation. | Communicate activity and life with controlled asset/performance scope; retain static cues and reduced motion. |
| V11 — Sourcing | Coherent free/permissively licensed foundation plus original art where needed. | Entirely original; one purchased asset family; several deliberately restyled packs. | Avoid mandatory purchases while preserving style and verified source/notice records. Prefer suitable CC0; evaluate other permitted terms explicitly. |
| V12 — Scaling | Whole-number world scaling, flexible viewport and independently sized HUD text. | Fixed low-resolution canvas with borders; fractional fill scaling. | Keep world pixels crisp and information readable; this does not require camera zoom controls. |
| V13 — Map status | Small contextual icons for the selected villager; explanations and village-wide alerts in the HUD. | Icons over every villager; activity/status entirely in inspectors/HUD. | Connect visible activity to explanations without crowding larger villages. Use distinct shapes/labels for blocked/error states. |

- **Boundary:** These selections permit consolidation and asset/font research,
  not product implementation or approval of a written artifact that did not yet
  exist. Specific production packs/fonts, exact colors, type/spacing values and
  layer/frame details are explicitly proposed for written review. The generated
  reference and candidate metadata supply no actual renderer, performance,
  font-coverage or accessibility pass. #728 stays open; outstanding #727 written
  disposition, #729, #730, #731/#732 and later gates remain separate. No changes
  to D014's broader vision or D015's final-product AI requirement.

### D020 — Select interface behavior for #729

- **Date:** 2026-10-20 (America/New_York).
- **Decision-maker:** dhar174.
- **Disposition:** Twelve interface choices explicitly selected; consolidated
  written specification and additional defaults await review.
- **Artifact:** [Draft #729 UI specification](superpowers/specs/2026-10-20-issue-729-ui-design.md).
- **Purpose:** Turn D018's journeys and D019's visual direction into a stable,
  readable Pygame observer interface at both supported sizes, retaining the
  wider product vision and final-product AI requirement.
- **Confirmation:** Each row is the user's submitted choice after review of
  options/tradeoffs. The generated layout comparison was shown before selecting
  A. No final written acceptance, prerequisite closure or phase approval is
  inferred from these selections or the earlier PR merges.

| ID / topic | Confirmed selection | Other options presented | Recommendation rationale |
| --- | --- | --- | --- |
| U01 — Main layout | A: docked right entity list/inspector with expandable event strip. | Bottom information dock; collapsible map overlays. | Stable inspection location without covering villagers; exact dimensions remain proposed. |
| U02 — Inspector organization | Compact overview with expandable Inventory, Plan and Recent Memory sections. | Separate subject tabs; one scrolling inspector with all main sections open. | Keep essential state/reason/outcome together while preserving technical expansion. |
| U03 — Event history | Docked bottom drawer reduces map height and keeps inspector visible. | Non-modal map overlay; dedicated full-window history view. | Retain village/inspection context and explicit playback. Preserve reading position as new events arrive, with count/Jump to Latest. |
| U04 — Smaller window | Keep right sidebar at 1024×768, adapt width and use wrapping/local scrolling. | Move to bottom dock; collapse secondary details into drawers. | Preserve positions and keyboard navigation without shrinking essential text. |
| U05 — Session shortcuts | S Quicksave, L Load selection, R Restart request. | Ctrl+S/Ctrl+O/Ctrl+R; focused menu controls without direct shortcuts. | Retain existing pattern while routing through text focus and unsaved-progress protection; named saves stay in the menu. |
| U06 — Save/Load placement | Centered modal over the village with focused save list/naming controls. | Full-window save/load screen; temporary sidebar view. | Provide metadata/confirmation space and retained village context; contain focus and restore it afterward. |
| U07 — Closing menus/dialogs | Remain paused until explicit Play/Resume. Successful loads already remain non-advancing. | Restore prior playback; ask whether to resume each time. | Avoid surprise advancement. Live inspector/history behavior remains unchanged. |
| U08 — Actual save consistency | Hold the world at a complete tick boundary until saving finishes, keeping UI responsive. Load remains paused and validates replacement. | Save an immutable snapshot while simulation keeps advancing. | Clear saved state and bounded correctness scope; this does not permit blocking the UI thread. |
| U09 — Routine confirmation | Record in history and show a brief non-modal confirmation. | Dedicated persistent notification area; history/badge without temporary confirmation. | Immediate acknowledgment plus retained evidence, without stealing focus. Required persistent errors/blocked/recovery behavior is unchanged. |
| U10 — Unavailable capabilities | Keep related section visible with plain-language reason and access to details/requirements. | Hide sections and explain in Feature Status; group unavailable functions into one entry. | Explain profile limits in context; distinguish empty, disabled, uninitialized and unavailable/failed without inventing results. |
| U11 — Step/speed keys | Period advances one paused tick; Page Up/Down moves between speed presets. | Period plus 1–4 direct preset mapping; focusable controls without additional direct keys. | Retain familiar speed keys and quick step inspection; text fields consume their input and visible controls remain available. |
| U12 — Final results | Bottom drawer with Summary and Events views, compact outcome when collapsed. | Sidebar Results section; persistent map banner with details on demand. | Keep result/evidence adjacent while preserving inspection. Finished is distinct from Paused; Play/Step cannot extend the episode. |

- **Earlier proposals clarified:** U07 explicitly selects D018's proposed
  menu-close behavior; U08 selects its proposed held-tick save policy; U05/U11
  specify the previously deferred shortcut choices. Other written defaults,
  including the walkthrough method, are not accepted by implication.
- **Boundary:** These choices permit consolidation into a written UI proposal,
  not product implementation. Exact geometry, input precedence, quicksave's
  post-operation playback policy, confirmation duration, history buffer and
  terminal-save/close details are explicitly proposed for written review.
  The comparison's example inventory/artwork/text is not scenario evidence.
  #727/#728 final dispositions and #729 remain open; #730, #731/#732 and later
  reviews remain separate. No runtime, font-fit, accessibility or performance
  pass is supplied. Continue recording all human decisions only in this file.

### D021 — Review remaining design proposals for #727–#729

- **Date:** 2026-10-21 (UTC; later selections on the same date in America/New_York).
- **Decision-maker:** dhar174.
- **Disposition:** All thirteen remaining selections confirmed; consolidated
  complete written-specification dispositions remain pending.
- **Artifacts:** [UX specification](superpowers/specs/2026-10-20-issue-727-ux-design.md),
  [visual specification](superpowers/specs/2026-10-20-issue-728-visual-design.md),
  and [UI specification](superpowers/specs/2026-10-20-issue-729-ui-design.md),
  reviewed from merged main `3e1c7ef` after PR #793.
- **Purpose:** Settle remaining proposals without repeating choices already
  confirmed in D018–D020, then obtain the separate written dispositions and
  proceed to #730's combined design review.

| ID / topic | Confirmed selection | Other options presented | Recommendation rationale |
| --- | --- | --- | --- |
| R01 — Later UX walkthrough method (#727) | A maintainer exercises every scenario using only in-product guidance; include first-time observers when available and report their results separately. | Maintainer only (less coordination, weaker newcomer evidence); require an external participant group (stronger newcomer evidence, recruitment effort/delays). | Keep the required walkthrough practical while distinguishing maintainer familiarity from newcomer usability. |
| R02 — Environment artwork foundation (#728) | Trial Slates with matching original artwork for missing pieces; retain CC BY 4.0 attribution and modification notices. Audit coverage/style and a compatible villager family before production adoption. | CC0-only source assets plus more original buildings/props (fewer attribution obligations, more art work); entirely original environment artwork (maximum control, largest workload). | Best researched 32×32 village foundation for the selected direction without mandatory asset purchases. The user explicitly accepted the recommendation. |
| R03 — Exact palette approach (#728) | Trial the existing section 3 Meadow & Terracotta color values, refining them against real scenes and contrast checks. | Revise the palette within the selected mood and compare visually before implementation; develop an original constrained palette from scratch (more design work and another visual review). | Provide a reproducible starting point while allowing refinements supported by rendered-scene evidence. The user accepted the recommendation; no rendered palette validation has been executed. |
| R04 — Font pairing trial (#728) | Trial Atkinson Hyperlegible for information and Pixelify Sans for short headings; use conventional headings if pixel lettering fails readability testing. Verify real Pygame rendering and character coverage at both supported sizes. | Noto Sans information text with Pixelify Sans headings (more neutral appearance); conventional text throughout, starting with Atkinson Hyperlegible (simpler typography, pixel identity carried by artwork/frames). | Distinguishable letters/numbers for information with restrained pixel-art character in headings. The user explicitly selected this pairing; no font has been imported or renderer/font-coverage pass established. |
| R05 — Type-size and spacing baseline (#728) | Trial 18px body/control text, 16px secondary text and 24px headings; 4px spacing increments, 8px control padding, 12px panel padding and 16px group gaps. Check actual fonts at both supported sizes; wrap or scroll essential text rather than shrink it to fit. | Same text sizes with a roomier 8px spacing grid (less information visible/more scrolling); larger text such as 20px body, 18px secondary and 28px headings (requires revised panel fit/wrapping/scrolling). | Use a concrete readable starting point that balances information density and map area, subject to actual font-fit checks. The user explicitly selected this baseline. |
| R06 — Animation/layer production scope (#728) | Complete a small, coherent set: reuse compatible templates, draw missing directional/activity frames, and separate foreground pieces needed for selected-villager fading. Retain the agreed short walks, clear activity poses and limited ambient motion; exact frame counts/fade values belong in the production plan. | Larger bespoke animation set (more individualized detail, substantially more artwork/performance validation); revisit the earlier choice to reduce initial coverage (requires revised design/rework, preserving complete visibility/status cues). | Complete the selected behavior with controlled art scope instead of expanding animation detail. The user explicitly selected this approach. |
| R07 — Initial panel dimensions (#729) | Trial the dimensioned specification: 304px sidebar at 1280 width, 280px at 1024; 40px collapsed event strip and 240px expanded drawer. Retain configurable dimensions and revise if actual font-fit checks require it. | Wider sidebar (more details/less wrapping, less map width); taller expanded event/results drawer (more history/summary, less map height). Either alternative requires revised tables and bounds checks. | Keep a concrete, statically checked starting layout at both supported sizes while reserving real font-fit validation. The user explicitly selected this starting point. |
| R08 — Quicksave return outside menu (#729) | After S quicksave outside the session menu, restore prior playback when safe: running returns to running, paused stays paused. An ordinary save I/O error shows a persistent error and preserves prior playback only while the world remains valid. Menu quicksave remains paused; runtime/state-integrity failure always stops advancement. | Always leave paused and require explicit Play; ask whether to resume whenever quicksave interrupted running playback (more interruption). | Preserve the convenience of quicksave without surprise changes to paused sessions or unsafe advancement. The user explicitly selected this policy. |
| R09 — Page Up/Down focus precedence (#729) | Page the focused scrollable view first; change speed only when the focused control/dialog has not consumed the key, such as with map focus. Text editing/dialog input remains local; one keypress cannot scroll and change speed. Visible speed controls remain available. | Prioritize speed outside text fields/dialogs (focused lists need other paging navigation); revise the earlier speed mapping to reserve Page Up/Down for scrolling (requires a follow-up shortcut choice). | Preserve predictable local keyboard navigation while retaining the agreed speed shortcut in unconsumed contexts. The user explicitly selected this precedence. |
| R10 — Routine-confirmation duration/placement (#729) | Show routine confirmations for five seconds in the existing HUD status lane; retain each message in event history and do not steal focus. Actionable errors remain persistent. | Longer configurable duration in the same lane (more reading time, longer occupancy); separate reserved notification lane (requires revised layout and consumes space without covering villagers/controls). | Provide consistent acknowledgment without another interface region or map obstruction. The user explicitly selected this policy. |
| R11 — Displayed event-history buffer (#729) | Keep 200 recent UI entries with a labelled count and access to the complete canonical result trace; preserve reading position and indicate new events. Trimming the UI never deletes the separately retained simulation evidence. | 1,000 recent entries (more immediate scrollback, greater memory/display-management cost); paginated history (older entries accessible through additional navigation/implementation complexity). | Bound immediate UI work while keeping the complete trace available and the visible subset honest. The user explicitly selected this policy. |
| R12 — Completed-scenario saves (#729) | Allow completed saves and load them as Finished, keeping the village, final summary and history inspectable with Play/Step disabled. Never advance beyond the scenario endpoint. | Load as Paused with an explicit completion notice (less clear state label); restrict saves to unfinished sessions (requires a design revision and removes completed-world save/reload inspection). | Preserve the authoritative terminal state and useful inspection without implying that playback can resume. The user explicitly selected this behavior. |
| R13 — Window-close during commit (#729) | Remember the close request during a non-cancellable save/load commit, show closing as pending, finish or report the operation, then apply unsaved-progress protection and failure-appropriate recovery rules at the safe boundary before exiting. Keep UI event handling responsive. | Decline the request while committing, explain why, and require another Quit request after completion (simpler pending state, extra user action). | Honor the user's request without interrupting an in-flight commit or bypassing state/unsaved-progress safeguards. The user explicitly selected this policy. |

- **Boundary:** R01 selects the later working-interface test method; R02 selects
  an environment-art trial. No human walkthrough or asset trial has been
  executed. These selections do not approve the complete
  #727/#728/#729 written artifacts, close issues, authorize UI implementation,
  or approve #730/#731/#732. R03 selects a palette-trial approach, not a rendered
  contrast pass. R04 selects the font pairing to trial, not final validated
  font files. R05 selects initial sizes/spacing, not a measured font-fit result.
  R06 selects production scope, not completed artwork or renderer validation.
  R07 selects initial layout dimensions, not a native readability/fit pass.
  Exact imported assets and the compatible villager-family choice remain work
  for the later production plan and audit, not unresolved changes to the
  selected sourcing strategy. No save/load, history or input behavior was run.

**Complete written review now requested:** read the three linked specifications
together, including their journey/state tables, documented focus traversal,
input ownership/Escape ordering, recovery restrictions and later evidence
requirements. The thirteen selections settle the individually presented
proposals; they do not yet supply acceptance of the complete documents.
Record a distinct Approved / Changes requested / Deferred disposition for each
of #727, #728 and #729 after that review. #730 remains a separate combined
UX/UI/aesthetic review; #731/#732 remain separate engineering/phase decisions.

### D022 — Walk through the consolidated design journeys

- **Date:** 2026-10-21 (America/New_York).
- **Reviewer:** dhar174.
- **Reviewed revision:** merged main `752bc2e`, following PR #794.
- **Disposition:** In progress; record only the journey explicitly reviewed
  in each response. Complete #727/#728/#729 dispositions and #730's combined
  review decision remain pending.
- **Artifacts:** [UX journeys](superpowers/specs/2026-10-20-issue-727-ux-design.md),
  [visual direction](superpowers/specs/2026-10-20-issue-728-visual-design.md)
  and [UI specification](superpowers/specs/2026-10-20-issue-729-ui-design.md).

| Journey | Human disposition | Reviewed scope and rationale | Confirmation |
| --- | --- | --- | --- |
| J01 — Launch and first inspection | Accepted as written | Validated setup, paused tick-0 start, discoverable Play/Help/hint, top controls/right inspector/bottom event strip, mouse/keyboard entity selection, ring/outline identity, truthful expandable information and inspection without unintended advancement. Warm pixel art and readable text retain later native checks. | User: “Its good.” after the launch-and-inspection walkthrough and recommendation to accept. |
| J02 — Playback and live inspection | Accepted as written | Play starts at 1×/two logical ticks per second; 0.5×/1×/2×/4× affects pacing only. Pause keeps inspection responsive; Step advances one complete paused tick and is disabled while running/finished/recovering. Inspection/history preserve focus and reading position; focused paging precedes speed. | User: “yes. We didnt already do all these? Im sure they're good...” |

- **Review method correction:** the user identified that this walkthrough
  repeated already selected behavior. Stop repeating the choice questionnaire;
  retain D018–D021 and prepare one consistency review of the complete artifacts.
  Present only actual gaps and the consolidated final disposition. The informal
  reassurance above is retained verbatim, not expanded into an unasked phase
  approval or executed-usability claim.

- **Boundary:** J01/J02 accept these planned journeys, not an executed usability
  result, the complete specifications or a phase decision. Other journeys,
  complete planning acceptance, #730/#731/#732 and runtime implementation are
  not approved by this response. All human decisions remain in this file.

### D023 — Approve design planning #727–#729 and combined review #730

- **Date:** 2026-10-21 (America/New_York).
- **Reviewer / decision-maker:** dhar174.
- **Reviewed artifacts:** complete UX, visual and UI specifications at merged
  main `752bc2e` (PR #794); D018–D021 selections; the consolidated consistency
  review in PR #795 at `6bc999d` and D022's scoped responses.
- **Explicit confirmation:** the user answered “Yes please.” to: “may I record
  #727–#729's planning specifications and #730's combined design review as
  approved, then begin #731's engineering review?”

| Issue | Human disposition | Scope and rationale |
| --- | --- | --- |
| #727 — UX journeys | Approved | Accept the complete written journeys, controls, truthful feedback, persistence/recovery requirements and later human walkthrough method. |
| #728 — Visual direction | Approved | Accept the complete pixel-art direction, asset/font/palette trials, spacing/animation constraints, licensing audit and later native-validation requirements. Trial approval is not production-file adoption. |
| #729 — UI specification | Approved | Accept the complete layout/state specification, documented focus traversal/input ownership/Escape behavior, visible session controls, save/load/history and terminal/close rules. |
| #730 — Combined design review | Approved | The three artifacts agree with the frozen MVP and broader product requirements; the consistency review identified no required design rework. Accept its explicit evidence limitations. |

- **Result:** synchronize and close #727/#728/#729/#730 as completed under this
  explicit human approval. Begin #731's engineering review preparation.
- **Limits:** no executed native usability, font-fit, renderer/performance,
  gameplay-turn or live-AI acceptance is implied. Existing downstream asset,
  implementation, persistence, episode and release work remains required.
  D014's expansive scope and D015's final-product AI remain intact.
  #731 is not approved by permission to begin it; #732 and parent #723 remain
  open. This decision does not authorize implementation beyond the existing
  reviewed scope or skip either later review/gate.
- **Project synchronization:** the configured API still returns HTTP 403,
  “Resource not accessible by integration,” for Project 10. Issue updates and
  the decision record are authoritative; Project Decision/Status fields cannot
  be claimed synchronized. Preserve that administrative limitation explicitly.

### D024 — Approve Phase 2 engineering review #731

- **Date:** 2026-10-21 (America/New_York).
- **Reviewer / decision-maker:** dhar174.
- **Disposition:** Approved.
- **Reviewed artifacts:** [engineering review](testing/issue-731-engineering-review.md)
  and [receipt](testing/evidence/issue-731-engineering-review.json), including
  PR #795's replayable-command correction at `17bed6d`. Runtime foundation is
  unchanged from verified #726 revision `ac0e25d`; the report preserves the
  separate source revisions, hosted results and evidence limitations.
- **Explicit confirmation:** the user answered “Approved” after the request
  to approve #731 for the reviewed Python 3.12/Linux foundation, retaining
  the documented UI, gameplay and live-AI validation limits.
- **Rationale:** accept import/renderer/service boundaries, hashed dependency
  profiles, configuration/error contracts, behavioral/package checks and CI
  result/enforcement evidence within that supported foundation. No required
  new engineering remediation or wholesale framework rewrite was identified.
- **Action:** publish this record and close #731 as completed. Prepare the
  separate #732 Phase 2 decision using the completed planning/foundation work.
- **Boundary:** this does not approve #732, close parent #723, unlock Phase 3
  or establish native UI/font-fit/performance, complete persistence, economic
  gameplay, GPU/native Windows/macOS or live-model readiness. The original
  expansive product and final-product AI requirements remain intact.
  Project 10 fields remain unsynchronized after HTTP 403; issue dispositions
  and this file are authoritative. No Project update is claimed.

### D025 — Approve Phase 2 gate #732 and complete parent #723

- **Date:** 2026-10-21 (America/New_York).
- **Reviewer / decision-maker:** dhar174.
- **Disposition:** Approved.
- **Explicit confirmation:** the user answered “Yes, approve” to “Do you
  approve Phase 2 gate #732?” after the recommendation for the reviewed
  planning/foundation scope, retained validation limits, Phase 3 handoff and
  the disclosed Project API limitation.
- **Reviewed artifacts:** [Phase 2 readiness](testing/phase-2-gate-readiness.md),
  complete design/engineering reports and D013–D024; PR #795 head `10667f6`,
  subsequently merged as main `a7dea4c32907d1fea6a6f4a7888d0337f17ad3e9`.
  Previous gate #722 and all nine required Phase 2 work/review issues
  (#724/#725/#654/#726/#727/#728/#729/#730/#731) were verified closed.
- **Rationale:** the agreed MVP/support/budget contract, reproducible supported
  profiles, real lightweight runtime, required CI and approved UX/visual/UI
  and engineering foundations meet this phase's planning/foundation scope.
  No required new phase remediation was identified. Earlier failures and the
  explicitly retained evidence limitations remain part of the record.
- **Action:** publish this decision; close #732 and Phase 2 parent #723 as
  completed. Unlock Phase 3 #733, beginning with #734's evidence-based backlog
  triage. Keep Phase 3 tasks and gate #738 open until their own work/decisions.
- **Limits:** this supplies no implemented UI, native font-fit/usability or
  performance acceptance, economic-turn/episode proof, complete persistence,
  live-model inference or native Windows/macOS/GPU readiness. D014 preserves
  the original expansiveness and D015 requires final-product AI. Existing
  later tasks and D002's issue-backed dispositions are not removed or widened.
  No arbitrary downstream code implementation or PR merge is authorized by
  recording this gate decision alone.
- **Project administration:** Project 10 field synchronization remains blocked
  by HTTP 403. Preserve issue/decision records as authoritative; do not claim
  Decision/Status/readiness fields were updated. The disclosed tracking limit
  is retained with the human phase approval.

### D026 — Execute #734 triage with review before issue publication

- **Date:** 2026-10-21.
- **Decision-maker:** dhar174.
- **Disposition:** Implementation plan approved, not backlog dispositions.
- **Explicit instruction:** “Implement the proposed plan.”
- **Publication choice:** “Review first”: prepare the complete ledger and exact
  proposed issue updates, review the package once, then publish approved changes.
  Duplicate closures require approval of the specific proposals and evidence.
- **Ownership choice:** “Ledger owner”: preserve existing GitHub assignees;
  record dhar174 as accountable maintainer for unassigned items, without mass
  GitHub assignments.
- **Boundary:** this does not approve triage results, issue closures, PR merges,
  reparenting, Project synchronization, or Phase 3 gate #738. #602 retains its
  blocked-regression and Phase 4 prerequisite limits. D014/D015 remain intact.


### D027 — Approve and publish the #734 backlog triage package

- **Date:** 2026-10-21.
- **Decision-maker:** dhar174.
- **Confirmation:** “Approved and merged. Lets move forward.” after review PR
  [#797](https://github.com/dhar174/tiny_village/pull/797), verified merged at
  `2c3bc5753b15aad4a1e345c020c5872d6ba8b25c` (reviewed head
  `fb0a0d834e48598b66f139ec268cd7d42aa008b8`).
- **Approved publication:** the complete [review package](planning/issue-734/publication-review.md):
  307 marked issue annotations, 80 missing-parent additions only if still
  unparented, and all three separately scoped remediation drafts with their
  specified parent/blocker links. Preserve existing bodies, assignees, parents
  and dependencies; checkpoint and verify publication against current GitHub.
- **Specific closures approved:** #355, #482, #594, #596 and #597 as completed;
  #351 as duplicate of #350 and #382 as duplicate of #380, both not planned.
  These are the reviewed narrow acceptances, not broader gameplay readiness.
- **Next work:** finish publication/readback and then close #734; proceed to
  #735 utility consolidation, followed by the remaining Phase 3 work. Preserve
  #602/#609 as open and retain all captured failures. New identity/interface
  repairs remain after gate #738, with their own scoped implementation review.
- **Boundary:** #733 and human phase gate #738 remain open; no gate approval,
  duplicate-PR selection, runtime repair or Project synchronization is implied.
  Project configuration remains #737's disclosed pending handoff. The frozen
  review bundle retains its historical proposed/not-applied status; current
  publication evidence is recorded separately.

### D028 — Retain PR #703 as the #735 utility consolidation carrier

- **Date:** 2026-10-21.
- **Decision-maker:** dhar174.
- **Choice:** “Keep #703; consolidate useful tests from #714 (recommended).”
- **Scope:** Rebase #703 onto current main, retain its shared `is_numeric`
  TypeError fix and useful numeric/score tests, add #714's reversed-range
  interpolation coverage, and document existing utility semantics and limits.
  Validate the selected result before closing #714 as superseded with crosslinks.
- **Boundary:** This chooses the PR carrier and duplicate disposition; it does
  not merge #703, approve human phase gate #738 or authorize changing the
  existing interpolation/score-scaling behavior into a different contract.


### D029 — Review #736 functional-test PR dispositions step by step

- **Date:** 2026-10-21 America/New_York.
- **Decision-maker:** dhar174.
- **Review baseline:** merged main `11b6e93595a70ce9f24cfb239a31ac873847ff56`.
- **#420 choice:** “Split: salvage useful assertions, then supersede #420
  (recommended).” Preserve unique real State/Action/Condition contract assertions
  in maintained test modules, validate the result, then supersede the old PR with
  a precise crosslink. Do not restore its unrelated demonstration checks merely
  because its five methods pass on current main.
- **Status:** #420 disposition selected; execution follows the consolidated
  walkthrough/plan. No port, closure or rebase has been performed. The other
  eight dispositions are pending and are not approved by this choice.
- **Boundary:** Keep existing #736 acceptance and recorded failures intact;
  this supplies no approval of #737, phase gate #738, runtime repairs or broader
  gameplay/AI readiness. Evidence is in [the walkthrough](planning/issue-736/README.md).

## Review status

The current scope questions have been answered (D002, D004–D006). Review #721
is **Approved** under D011 and closed for the agreed Linux/offline startup
scope. D010 accepts #779's residual uncertainty and closure. D001 remains the
historical disposition of the earlier failed verification. D012 separately
approves phase gate #722 and completes parent #717 within the same scope.
Phase 2 is unblocked and #724's planning contract is integrated.
The [#724 contract](superpowers/specs/2026-10-20-issue-724-mvp-contract.md)
has D013's two presentations and D014's qualified acceptance. The broader
TinyVillage vision remains intact. #784 integrated the accepted contract to
main and closed #724. #725 merged through #785; #654/#776 merged through #788.
D017's #726 work is complete: #789/#790 merged, hosted main/manual results and
effective required-check enforcement verified; #726 closed. D018 records the
15 confirmed #727 interaction choices. #791 merged the written artifact at
7391861; its final written-review disposition and #727 remain open. The user
explicitly requested the subsequent preliminary #728 design walkthrough. D019
records those visual choices; #792 merged the artifact at 4c9bbfd, while #728's
final disposition remains open. The user requested #729 next and selected the
12 choices in D020; its consolidated written specification/defaults await review.
#729 and later human gates remain open. The explicitly selected overlapping
#727 defaults are clarified in D020 without approving the remaining artifacts.
D021 now records thirteen further selections and the consolidated written
review request. #727/#728/#729 remain open pending complete written acceptance;
the PR merges and individual choices do not supply #730/#731/#732 approval.
D023 subsequently records the user's explicit approval of the complete
#727/#728/#729 planning artifacts and #730 combined review. Close those four
issues and prepare #731; #731/#732 remain separate pending human decisions.
D024 subsequently approves #731 within its reviewed scope. Close that review;
#732 and Phase 2 parent #723 remain open for the separate phase decision.
D025 subsequently records explicit Phase 2 gate approval and authorizes
#732/#723 completion. Phase 3 #733 is unblocked, with #734 next. Its work and
later gate #738 are not complete or approved by this Phase 2 disposition.

The [#779 investigation plan](superpowers/plans/2026-10-15-issue-779-startup-investigation.md)
was approved under D007, and its bounded cycle has concluded. Its original
16-case diagnostic budget remains exhausted; no production repair is approved.

The [failure-capture follow-up](superpowers/plans/2026-10-16-issue-779-failure-capture.md)
was approved under D008. Its implementation merged and its bounded verification
window concluded. This did not change #721's disposition or approve a runtime
correction.

The [D008 implementation and verification report](testing/issue-779-failure-capture.md)
records 11 new capture checks, 81 existing methods and three original probes
passing in the single approved window. Its nine-batch/three-probe budget is
consumed. The historical timeout did not recur, so its cause remains unknown;
#779 stayed open at that handoff; D010 subsequently accepts closure with the
cause still unknown. That handoff did not change #721's disposition; its later
approval is recorded separately under D011.
Merged [PR #781](https://github.com/dhar174/tiny_village/pull/781) contains this
follow-up; merging it is not a human approval or issue closure.

The [D009 post-merge report](testing/phase-1-post-merge-verification.md) on merged
main `77e4b94` contains 94 successful test-method summaries and three zero-exit
probes. One lifecycle supervisor completion record is missing after a transport
disconnect; its successful 12-test summary and all child exits are retained.
Only previously unstarted targets were continued, once each. No target was
rerun, and the missing supervisor result is not labeled PASS. The authorized
window is consumed. D010 subsequently accepts the residual risk and #779
closure. D011 subsequently approves #721 and D012 approves #722 separately.
Neither merging the PRs nor "Proceed" supplied those dispositions.
The evidence is published in [draft PR #782](https://github.com/dhar174/tiny_village/pull/782)
and linked on #779/#721; their original acceptance text is preserved.

The [approved observation cycle](testing/issue-779-startup-investigation.md)
has concluded: 51 affected regression methods pass and all 16 predeclared
diagnostic cases exit successfully. The historical timeout cause remains
unconfirmed. No remedy or new review disposition is approved by those results;
the diagnostic budget is exhausted. The later D010 closure accepts this
uncertainty without claiming a proven remedy.

## Outstanding actions and limits

- #779 is closed under D010 as unable to reproduce; diagnostics added. Reopen
  if the timeout recurs and retain failure artifacts. Further observations
  require a new scope; no root cause or production fix has been established.
- Include native Windows execution and cache-permission checks when planning
  the later platform work, before any Windows-support claim (D004).
- Include live model download and inference checks in the later ML/LLM work
  before making readiness claims for those features (D005).
- Demonstrate a real gameplay turn during the later demo/integration work;
  retain the zero-action limitation of the current minimal-mode probe (D006).
- Project 10's #721 and #722 **Decision** fields still need synchronization to
  **Approved** under D011/D012, and Phase 1/Phase 2 readiness needs reflecting
  in the Project where applicable.
  The configured credential received HTTP 403, “Resource not accessible by
  integration,” on the last attempt. The native issue dependency and recorded
  issue disposition were verified; a Project field update was not.
- #721 is approved and closed under D011 using the current evidence and D010's
  accepted limitations. No additional observation window is authorized.
- #722 is approved and #717 is complete under D012. Begin Phase 2 with #724's
  MVP scenario, support matrix and measurable budgets; subsequent reviews and
  gates retain their own approval requirements. No platform, gameplay or
  model-inference readiness is implied by Phase 1 completion.

## Issue closure and roadmap references

- #718, #719 and #720 remain closed: their scoped implementation repairs merged.
- #779 is closed with reason **not planned** under D010's non-reproducible,
  accepted-risk disposition. Original root-cause criteria remain unfulfilled.
- #721 is closed as completed under the explicit D011 human approval.
- #717 and #722 are closed as completed under the separate D012 phase-gate
  approval. Phase 2 parent #723 remains open; #724 is closed after #784's merge.
- #773–#775 remain open in their existing phases; deferral is not completion.
- #776 and parent #654 closed after #788 merged; lightweight memory/startup
  readiness does not establish gameplay or final AI readiness.
- Native Windows support/validation is carried into support-matrix and CI work
  (#724/#726); core/optional dependency profiles remain in #725. Live-model
  integration evidence remains separate in #585. Later lightweight-demo,
  real-turn scenario and canonical demo work remain in #654/#751/#658.
- GitHub hierarchy: #717 is the Phase 1 parent of #718–#722. #722 is a gate,
  not the parent issue. #779 is now a native child and required blocker of #721;
  #721 remains a required blocker of #722.

### Repository synchronization — 2026-10-15

Updated 18 existing issue bodies with targeted status/scope references to this
decision record: #717–#722, #779, #773–#776, #724–#726, #654, #751, #658 and
#585. Preserved their previous bodies, acceptance checklists and issue states;
no issues were closed or reopened. Read-back verification confirms the native
relationships and the open review/gate/remediation issues.

The decision file and evidence are published in
[documentation PR #780](https://github.com/dhar174/tiny_village/pull/780).
That PR closes no issues and supplies no new human review or phase approval.
Project Decision synchronization remains outstanding as described above.
