# #741 live simulation tick execution — approved implementation plan

Approved by the user's implementation request; merge #823 explicitly authorized
in the follow-up selection. Base: latest origin/main437c689 (verified after merge).
Spec: GitHub #741, including the D030/#736 carry-forward. Coordinator alone writes;
repo .zcode/agents specialists investigate/review read-only.

## Task 1: Goal-bearing strategy requests

Add PlanRequest(actor UUID, fixed goal, source/request identity, priority, extras)
and PlanProposal(request, ordered tuple, PLAN/SATISFIED/NO_PLAN/ERROR, reason).
Pure create_daily_request/create_event_requests stage without GOAP/model/execution;
plan_request uses existing planning APIs and failure handoff. Retain list APIs.
Resolve object/dict event participants; explicit bad identities fail, untargeted
village events use explicitly supplied registered actors, never default Emma.
Freeze goal containers without copying graph/entities. Test purity, identity,
fixed predicates, all outcomes, native event goals/actions and old compatibility.

## Task 2: Controller cadence and staging

UUID execution records retain goal/sequence/cursor/phase/recovery/pending. Actor
cadence is separate. Config planning defaults: action_interval_seconds1.0,
retry_cooldown_seconds5.0, max_recovery_attempts3, max_pending_requests16; validate.
Already-scaled finite nonnegative dt only; pause freezes. Consume whole intervals
without catch-up bursts, retain fraction, one action and one search/decision per
actor/update. New actors start zero. Automatic routine/event/decision aliases
stage; no update_strategy([]). Central _run_selected_action invokes/bookkeeps once;
low-level manual _execute_single_action remains explicit.

## Task 3: Continuation, priorities and completion

Re-resolve same registered actor/world; check actual fixed goal, validate full
remaining tail+terminal goal+candidate availability, execute current occurrence.
Initial plan+first action may share interval; replacement waits next interval.
Map before character; synchronized weather before eligible action validation.
Priority (normalized importance, abs impact); routine(0,0); equal-priority FIFO.
Higher preempts between actions: preserve request/arrival/exclusions, discard tail,
resume with fresh plan. Deduplicate actor/fixedgoal/source/candidate equivalents.
Pending cap16: higher incoming displaces newest at lowest priority; otherwise
reject incoming. REJECTED_CAPACITY observable. Pop urgent before requeueing old.
Daily completed stays satisfied until unmet. Cursor exhaustion is not completion.

## Task 4: Recovery, compatibility and lifecycle

False/exception/mismatched declared effects: cursor stays, fresh partial-write
state, semantic exclusions survive regenerated instances. Signature uses concrete
implementation/name/canonical conditions/effects/actor-target UUIDs, not generated
IDs/timestamps/addresses. Invalidation replans without blacklisting temporarily
inapplicable actions. One fresh search/update: defer recovery if budget spent.
Optional GOAP post-verifier runs after sole invocation before success bookkeeping;
record raw return vs verified progress. Pure projection, graph-first actual values.
NO_PLAN blocks immediately; third consecutive failure/invalidation without verified
step blocks; verified step resets count. BLOCKED owns slot, no unrelated fallback.
Retry only after 5 simulation seconds AND relevant state/catalog change; baseline
captured after partial failure writes, new epoch reconsiders exclusions. Errors
remain distinct. Active tail validation is never skipped by fingerprints.

Legacy Plan.replan returns bool; graph from Plan or Goal.environment; reject
conflicts. Include retained real candidates, preserve repeats/dependencies; install
whole validated replacement or actual satisfied empty queue; clear on no-plan,
missing context or exhausted limit. Preserve contract errors. Alternative dict
retains action/priority/dependencies plus full actions tuple. No mock/random
alternatives or incidental reorder on replan_count0. Legacy execute remains manual.
LLM idle decisions only at due slots without GOAP owner; ranked choices yield one
action, not a GOAP plan. Preserve cached availability/provider reuse. No direct
energy mutation/success-by-default fallback. Legacy bare lists stage ordered
non-goal sequences; terminate truthfully on failure. Clear state at reset/load,
actor/manager replacement/removal and close; no persistence expansion. Read-only
execution status accessor feeds tests/existing selected-character UI.

## Task 5: Acceptance and verification

Real update_game_state and registered full objects/actions. Rest->Work at two
intervals (energy20->35->5, money5->5->25); daily Rest4+Leisure4 achieves energy80,
satisfaction80. Repeats, fractional partitions, delayed .5+2.75->one/.25 then.74
none/.01 next, pause, speed once, config invalid. Events share per-actor budget;
priority/resume/capacity, UUID/name ambiguity, graph-only changes, weather, partner
conditions/catalog changes. False/exception/partial writes/no-effect True/verifier
error exactly once; real multi-step alternatives and explicit no-alternative;
three failures, unchanged clone exclusion, cooldown+relevant change epoch. Lifecycle
no stale calls. LLM boundary tests only external load/output fault injection.
Legacy boolean repro, repeated names, full alternative, empty/no-plan/exhaustion.
Migrate eager orchestration assertions, preserve counters/memory/quests/rollback.
Separate bounded core active-plan batch, one dummy-SDL parity case. Preserve child
25/fatal20/supervisor120 limits. Canonical core/dev/visual runners; actual commands,
counts, source hashes, traces, stdout/stderr; existing job-parser errors disclosed.
Independent final reviewer, resolve findings, update Memory Bank task/index and
D037 with actual approval/evidence. No later gate/model/product readiness inferred.

## Recorded implementation rulings

Unknown-goal legacy sequences interrupted by urgent work produce an explicit
CANCELLED_PREEMPTED receipt rather than inventing a goal for replanning. Structured
goals retain preserve/fresh-resume behavior. Full review, fixture corrections and
verified limits are in docs/testing/issue-741-live-plan-execution.md.
