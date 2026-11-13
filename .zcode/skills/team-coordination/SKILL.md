---
name: team-coordination
description: Coordinate Zcode contributor work in tiny_village with its read-only subsystem specialists, GraphManager and decision-loop contracts, Memory Bank, mem0ry4ai handoffs, CI evidence boundaries, and independent review.
---

# Tiny Village project coordination

This skill is the Zcode adapter to `docs/agents/team-coordination.md`, the canonical contributor-team policy. Read root `AGENTS.md` and that policy. Read the generic `~/.zcode/skills/team-coordination/SKILL.md` for universal delegation/MCP procedures; load this local file by path, not just the shared skill name. Apply this repository's stricter one-writer policy rather than generic permissions for other projects.

## Start with the right context

Identify the actual root, revision, local changes, authorized task, relevant accepted decisions, and evidence gates. Read `memory-bank/memory-bank-instructions.md`, the six core Memory Bank summaries, relevant task records, and the applicable entries in `docs/decisions.md`. Verify remembered or summarized claims against the current code.

For historical retrieval, pass the active root and explicit scope `project:tiny_village` to `mem0ry4ai:memory-scout`. For a multi-subsystem change, shared contract change, or unclear ownership, route current-source investigation to `tiny-village-architecture-scout` after giving it the relevant memory brief. For a known single-subsystem question, route directly to the owner. Avoid duplicate retrieval and repeated architecture investigation within one task.

The coordinator alone edits repository files, executes commands, dispatches specialists, integrates recommendations, and reports to the user. Every `tiny-village-*` specialist has Read/Grep/Glob-only work: no edits, shell, network, or agent spawning. Commands proposed by test/review specialists are for the coordinator to run. Multiple implementation writers are not enabled.

If acting as a specialist, keep the parent's assignment and restrictions. Do not reinterpret these startup rules as an instruction to invoke another scout or steward. Report missing evidence or a contract conflict to the parent.

## Choose the project specialist

Inspect the current Zcode `Agent` catalog and effective definition; use the exact advertised `subagent_type`. Repository templates are under `.zcode/agents/`, with possible user-level copies under `~/.zcode/agents/`; see `.zcode/README.md`. Preserve profile model/reasoning/turn limits. Existence, installation, discovery, and actual invocation are separate evidence states. Do not silently install or synchronize profiles.

| Agent | Use for | Required task-specific context |
| --- | --- | --- |
| `tiny-village-architecture-scout` | Cross-system architecture, ownership, current constraints and active work | Objective/snapshot, relevant decisions and Memory Bank excerpts, suspected call path, specific ownership question |
| `tiny-village-planning-actions` | GOAP/utility/goals, action preconditions/effects, plan validation/caching/replanning | `tiny_goap_system.py`, `actions.py`, strategy/evaluator paths, input world state, expected action/plan, actual failure or proposed contract |
| `tiny-village-world-state` | GraphManager/WorldState ownership, identity, graph coherence, time/events, resource and save/load boundaries | `tiny_graph_manager.py`, `world_state.py`, `tiny_globals.py`, entity/event/time consumers; identity and mutation/recovery scenario |
| `tiny-village-llm-contracts` | Strategy routing, prompts, model I/O, parsing/normalization, fallback and decision-loop integration | Prompt, model output example, parser/action mapping, relevant strategy/helpers, expected effects and dependency capabilities |
| `tiny-village-runtime-memory` | In-game memory recording/recall, basic/full capability, retrieval, lazy init and performance | `tiny_memories.py` and consumers, dependency profile, recording/query path, observed capability/fallback and expected result |
| `tiny-village-social-narrative` | Relationships, social actions, economy/jobs, events, storytelling and continuity | `social_model.py`, jobs/event/narrative paths, authoritative starting state and expected causal consequences |
| `tiny-village-ui-simulation` | Pygame/controller/map/animation, headless boundaries, display of simulation state | Controller/map/main/assets paths, headless or visual mode, input/render lifecycle, authoritative state, actual reproduction/evidence |
| `tiny-village-test-ci` | Check design, regression analysis, dependency profiles, CI failure diagnosis | Changed contract, relevant tests, `scripts/run_ci_checks.py`, selected profile/platform, actual coordinator-supplied receipts and gaps |
| `tiny-village-reviewer` | Independent final review of scope, contracts, permissions, integration, claims, agent setup | Exact final diff/snapshot, approved scope, relevant decisions/invariants, actual checks and explicitly unverified layers |

Parallelize distinct read-only questions against the same stated snapshot: planning semantics and graph ownership, for example, or prompt/output alignment and meaningful regression design. Keep dependencies sequential. Do not invoke all nine specialists automatically or assign one agent to “review everything” without a bounded question.

## Keep the three memory roles separate

- **Contributor records:** `docs/decisions.md` owns human decisions; `memory-bank/` owns resumable project/task context. Do not create competing decision logs or duplicate status records.
- **External retrieval/curation:** `mem0ry4ai:memory-scout` and `mem0ry4ai:memory-steward` use the existing user-level MCP server for permitted reusable contributor knowledge. They do not own the repository records.
- **Game runtime:** `tiny-village-runtime-memory` investigates the villagers' implemented memory pipeline. This is separate from both contributor systems and does not authorize writing memories about the user's task.

Use server `plugin:mem0ry4ai:mem0ry4ai` and its advertised tools (normally prefix `mcp__plugin_mem0ry4ai_mem0ry4ai__`). Follow the generic tool guide for resume/search/get/list/session search and gated writes. Do not derive scope from `.zcode`, a cache directory, or an isolated checkout name if an established project scope is supplied.

### Scout packet

```text
Repository root: [absolute active Tiny Village checkout].
Memory scope: project:tiny_village.
Task: [objective, important concepts, and exact retrieval question].
Relevant paths/decisions: [files, D0xx entries, subsystem names].
Current evidence/restrictions: [verified facts; no-write or other task boundaries].
Use scoped resume and targeted task/file searches; retrieve precise records
only as needed. Return IDs, useful context/gotchas, open work, staleness/conflicts,
and what current-source checks remain necessary. No writes, shell, or delegation.
```

## Assignment, integration, and review contracts

Each project specialist receives this packet, completed with concrete values:

```text
Task ID and objective: [TV identifier; one bounded question/deliverable].
Repository root/revision: [absolute path; commit or working-tree snapshot].
Paths and evidence packet: [relevant files/symbols, exact excerpts if needed].
Instructions: [root AGENTS; canonical policy sections; applicable .github/instructions].
Accepted decisions: [relevant D0xx entries and evidence limitations].
Scope/exclusions: [ownership; neighboring contracts and deferred work excluded].
Allowed operations: Read/Grep/Glob only; no edits, commands, network, or delegation.
Dependencies: [relevant memory brief, earlier findings, coordinator command receipts].
Acceptance and effort bound: [what makes the answer usable; bounded pass].
Escalate if: [specific missing input, conflicting contract, or evidence need].
Return: ID/status; paths/symbols examined; findings with path/line evidence;
observed/inferred/proposed labels; recommendations and owners; checks not run
or supplied results; uncertainties/dependencies/blockers; checked/unchecked review scope.
```

Give agents the minimum relevant source and task context rather than the entire conversation. They must not assume another specialist's report, approval, or provider access. Include exact command outputs when the question depends on them. A specialist's proposed command is not an executed check.

The coordinator reconciles recommendations against the actual code and accepted decisions, then makes edits. Preserve unrelated changes and shared contracts. Review is independent: send `tiny-village-reviewer` the final artifact, criteria, and real check receipts without asking it to approve a preferred answer. It reports defects; the coordinator fixes them and obtains another review when the substantive changes require it. If specialist invocation is unavailable, do the analysis directly and state that no independent specialist review occurred.

## Integration invariants and check selection

Preserve one authoritative GraphManager graph; do not expand the known `social_model.py`/WorldState exception. Keep GOAP/strategy, prompt/model output, parser/action IDs, execution/effects, memory/events, and graph feedback aligned. Preserve explicit module boundaries and meaningful capability/failure reporting; avoid fabricated success and permissive mocks.

AI remains required under D015. D013 requires both headless and Pygame presentations, and D014 preserves the broader product vision. Lightweight startup/basic-memory evidence does not establish a gameplay turn, visual readiness, live inference, Windows/GPU support, or final readiness. Review the relevant decision gate before proposing expanded execution.

Use existing profile-appropriate checks. `python -B scripts/run_ci_checks.py --profile core` is the canonical runner example; choose `dev` or `visual` as appropriate and check current selected offline ML coverage separately. It gates CPython 3.12 on Linux x86-64. On a native Windows checkout, report that supported-environment gate as unavailable unless the task provides an appropriate environment. Do not install models, rewrite locks, or run full game/provider trials for a coordination-doc task. Do not add tests/tooling for an ad-hoc request unless authorized.

Check documentation links and skill/profile metadata for coordination changes; inspect actual source/behavior checks for code changes. Retain command, directory, exit code, stdout/stderr, and receipt paths. Keep local checks, hosted CI, inference, gameplay, and platform claims distinct. Use the generic checkpoint/heartbeat procedure for long commands when no independent work remains.

## Closeout and steward packet

Preserve authoritative decisions in `docs/decisions.md`; do not fabricate owner acceptance. When Memory Bank updates are part of the task, update the existing task file and `tasks/_index.md` consistently on the same date with verified progress and remaining work. Do not create another task registry, copy private reasoning, or turn routine work into durable knowledge.

If external memory writes are permitted and a useful nonduplicative candidate remains, give `mem0ry4ai:memory-steward`:

```text
Root/scope: [absolute checkout]; project:tiny_village.
Write policy: [permitted operations and any no-write/proposals-only restrictions].
Final verified outcome: [settled behavior or reusable finding and rationale].
Authority/evidence: [docs/decisions, Memory Bank/source references, check receipts].
Candidates: [atomic gotcha/decision-reference/fact/command/preference/procedure;
why useful later; existing IDs and duplicate risks].
Exclude: [routine activity, unverified claims, repeated task status/decision text].
Resumption: [actual remaining need; authoritative Memory Bank task reference].
Search first, skip equivalents, and confirm any supported supersession/promotion.
Return actual tool-confirmed IDs/actions, skipped candidates, and limitations.
No repository edits, Memory Bank changes, shell, consolidation, or delegation.
```

Skip writing stewardship for a no-memory-write/read-only task or when no worthwhile candidate exists, regardless of Stop-hook reminders. Finish with the result, exact verification and independent-review status, material limitations, and the next authorized action if any.
