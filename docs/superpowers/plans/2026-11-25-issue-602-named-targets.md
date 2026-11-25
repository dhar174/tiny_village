# Complete #602 with #816 synchronization first

Approved by dhar174: "Implement the proposed plan." Full proposal is retained
in the task conversation; this implementation contract preserves its decisions.

## Goal and base

Use latest origin/main in an isolated worktree. The inspected base is
7b2c913dc9bf76f1e7cfd5b7a6f419f67173500c. Preserve D035 immutable typed identities,
GraphManager authority, per-attribute rollback, D013 presentations and D015 AI.
Prerequisites #220/#741/#799/#738 are closed, with preceding approval D034.

## Task 1: #816 synchronized direct updates

- Preserve `apply_attribute_change(target, graph_manager, attribute, calculate)`;
  canonicalize native fields, validate finite scalar results, use graph-first
  calculations and existing native clamps. Credits remain unscaled.
- Add private `Character._mutate_scalar(canonical_field, calculate)`; preserve
  setter APIs and construction behavior. Initialized removed Characters fail
  without object mutation or automatic registration. Zero is a valid update.
- Route Character scalar setters/social increments, economic rewards, building
  fees/scalar effects, mapped events and legacy event scalar writes through the
  existing mutation boundary. Building affordability uses graph-backed wealth.
  Preserve event-template mappings and managerless/nested-skills compatibility.
- Same manager/typed entity/field callback nesting stages object values and
  commits once at the outer boundary; other fields commit independently. Always
  clear transient context. Legacy method return values stay ignored; explicit
  calculate(None) restores staging without a commit.
- Failure restores the failed field's original object/graph/alias values and
  presence, retaining earlier commits. Service synchronization failure propagates
  to the existing failure result; no service-wide resource transaction is added.
- Regressions: real Actions energy20->15->25 (also stale object40/graph20),
  existing weather20->15->25, market20->23->33, setters/zero/social/building/events,
  callback exact-once, failed writes/aliases/absence, initialization/removal and
  authoritative planner snapshots. Verify and independently review before Task2.

## Task 2: #602 canonical contextual targets

- Canonicalize eligible native visitors to registered Characters before self
  exclusion, proximity, requested identity and deduplication.
- Preserve visitor fast path/empty-list fallback, ordered first eligible target,
  object/UUID/UUID-string/name requests, optional named absence success/no-op,
  required literal target failure, ambiguity rejection and prepare-before-write.
- Real-object regressions: copied initiator handle, requested partner after a
  third nearby Character, identifiers/fallback/absence/elsewhere, duplicate and
  overlapping selectors, location identity after visitor mutation, rejected
  preconditions/required targets, per-field rollback and named20->15->25.
- Retain six legacy checks. Do not broaden actor-local GOAP or the scheduler.

## Specialist strategy and verification

Use .zcode/agents/tiny-village-{architecture-scout,world-state,planning-actions,
test-ci,reviewer}.md through available Codex agents without claiming native
Zcode invocation. Architecture first; domain/test investigation concurrently;
independent milestone and whole-branch review. Coordinator alone writes/runs
commands; specialists read supplied source/evidence and recommend checks.

Use existing CPython3.12 bounded harness, dependency locks and timeouts. Run
focused action/identity/building/event checks, then canonical core/dev/visual
profiles with fresh artifacts. Record exact source hashes, commands, counts,
exits, stdout/stderr, numeric traces and review findings. Known parser ERROR
output is a limitation, not clean verification. Preserve historical evidence.

Deliver two logical implementation commits (#816 first), evidence report,
Memory Bank task/index and PR against main. Merge needs separate authorization.
#742/#743 approval, #821 inventory/catalog work, model/product readiness and new
test/lint/typecheck infrastructure remain outside scope.
