# Issue #654: Lightweight runtime and basic memory implementation plan

**Status:** Approved by dhar174 on 2026-10-20; implementation authorized.

**Goal:** Run the official minimal demo on the locked core profile with real
simulation objects and basic memory, without Pygame, AI packages, downloads,
or production mocks. AI is a final-product requirement under D015.

**Architecture:** Retain the shared runtime and production AI integrations.
Separate rendering and model initialization; introduce explicit basic memory
for lightweight testing. Execute sequentially using executing-plans.

**Requirements:** #654 and child #776. Baseline: merged main 8a899650.

## Binding interfaces

- `Memory(description, creation_time=None, importance_score=0)` retains the
  supplied fields and creates timestamps per instance.
- `MemoryManager(gametime_manager, index_load_filename=None, *, mode="full")`
  accepts basic/full; unknown modes and basic/index-load conflicts are errors.
- Append `Character(..., *, memory_mode="full")` without changing positional APIs.
- `MemoryManager.get_capabilities()` reports recording, recent recall and
  semantic service states: disabled/not_initialized/ready/unavailable.
- `demonstrate_minimal_integration(config=None)` returns the real controller
  only after its required checks; failures propagate to a nonzero CLI exit.
- Basic memory stores real independent records, oldest first, with a 50-record
  limit. Semantic requests raise MemoryFeatureUnavailableError, never dummy
  scores, sentiment, vectors or retrieval results.
- Full mode retains AI paths with lazy, idempotent initialization and explicit
  missing-dependency/model errors that preserve their causes.
- Minimal config selects headless, basic memory and no LLM decisions. Rendering
  and AI selection remain independent in the controller API.
- Visual/headless CLI combinations, including default visual, are rejected
  before dependency imports with instructions to select minimal mode.

## Ordered tasks

### Task 1: Baseline and decisions

- [ ] Branch from current main; preserve the four uncommitted D015 doc edits.
- [ ] Save the plan and task ledger; record approved selections in decisions.md.
- [ ] Retain cold core-only controller and memory failures with exact revision,
  commands, environment and exceptions. Leave dependency locks unchanged.

### Task 2: Lazy AI service boundaries

- [ ] Remove import-time model, sentiment and spaCy loading and NLTK downloads.
- [ ] Defer heavy dependencies until an enabled feature requests them.
- [ ] Preserve model identifiers, cache configuration and public feature paths.
- [ ] Reuse initialized services and expose actual capability failure states.
- [ ] Eliminate dummy dependency implementations on the supported memory path.
- [ ] Defer strategy LLM imports/model construction until requested; distinguish
  requested enablement from availability. Restore memory without eager AI setup.

### Task 3: Basic memory and real Characters (#776)

- [ ] Initialize mode and per-instance state before FlatMemoryAccess.
- [ ] Support real recording/recent recall without AI/index engines.
- [ ] Remove the broken spaCy constructor debug scan and malformed tuple test.
- [ ] Guard semantic entry points; repair Memory/Character argument mismatch.
- [ ] Pass mode into real Character composition and complete constructors using
  real graph, actions, location, traits, inventory and time manager.
- [ ] Verify separate Character memory stores and retained world ownership.

### Task 4: Renderer-free controller

- [ ] Defer Pygame/MapController imports and evaluated rendering annotations.
- [ ] Separate rendering from simulation construction; skip all rendering/UI
  setup in headless mode, including indirect recovery and notifications.
- [ ] Use a consistent stdlib monotonic clock for nonvisual operations.
- [ ] Retain real core services; required failures cannot create fake Characters.
- [ ] Provide idempotent cleanup restricted to resources this controller owns.

### Task 5: Truthful bounded minimal demonstration

- [ ] Remove Mock, global sys.modules replacement, simplified Characters and
  fake displays. Pass CLI config and honor the requested Character count.
- [ ] Construct the lightweight profile, register an actual Event in handler
  and graph, then record/recall actual memory for each Character.
- [ ] Validate ownership and required results; fail nonzero on any failure.
- [ ] Report capabilities and explicitly state gameplay was not exercised;
  remove successful-turn/action claims. Disable automatic checkpoints and
  generated config files; isolate unavoidable temporary resources.
- [ ] Reject visual/headless combinations before dependencies initialize.

Canonical command:

```bash
python main.py --mode minimal --headless --no-llm --characters 2
```

`python main.py --mode minimal` must also work on core alone.

## Verification

Write focused failing regressions before each implementation boundary. Use real
domain objects. Doubles are limited to external-service guards/failure injection.

- Core-only startup without installed Pygame/spaCy/Transformers/Torch/FAISS/etc.
- AI packages installed but no model/download/semantic initialization in basic.
- Complete actual Character constructors and graph/action ownership.
- Memory description/time/importance, equal importance, 50-record eviction,
  per-instance timestamps and independent stores.
- Explicit disabled semantic errors and missing full-mode dependency/model errors.
- Real spaCy-present/no-model construction, without claiming NLP model readiness.
- Actual event registration in handler and graph; nonzero failure propagation.
- Minimal with/without headless; invalid visual/headless before dependency imports.
- SDL-dummy real Pygame setup/cleanup with valid assets.
- Existing focused actions/GOAP/event/graph lifecycle/ownership/model-cache and
  subprocess-capture regressions. Retain original deadlines and failed evidence.

Use fresh core, dev without Pygame, dev+visual, and locked NLP environments.
New cold subprocess probes use the existing capture helper with 20-second fatal
diagnostic / 25-second parent deadlines. Report actual counts, failures, skips,
timeouts and commands; do not label retried failures as clean evidence.

## Completion and review

Update dependency/Quickstart/project context and publish a revisioned validation
report mapping every #654/#776 acceptance criterion to evidence. Open one focused
PR with separate service/basic-memory/headless-integration commits; request human
review of retained AI paths, boundaries, console guidance and failures. Issues
close only after acceptance and merge; no automatic phase approval. Next: #726.

Excluded: economic scenario, full gameplay turns/action-effect correctness
(#751/#658), live pretrained inference (#585), GPU/platform claims, CI changes,
visual redesign, broad #14 architecture and unrelated legacy failures.
