# Tiny Village Specialist Team Coordination

This is the single canonical coordination policy for the Tiny Village contributor
specialist team. All platform entry points (root `AGENTS.md`, root `GEMINI.md`,
`.agents/rules/team-coordination.mdc`, and the Zcode agent definitions themselves)
point here and must stay consistent with it. Change this document first, then the
entry points.

These specialists are **development assistants for maintaining this repository**.
They are separate from the villagers, planners, LLM integrations, and other agents
implemented inside the game runtime.

## 1. Platform entry points and loading paths

| Platform | Entry point | What it loads | Where definitions live |
| --- | --- | --- | --- |
| Zcode (primary) | root `AGENTS.md` | Injected into the primary session and (with `injectAgentsMd: true`) into each specialist | Source templates: `.zcode/agents/*.md` (repository-owned). Documented runtime location: `~/.zcode/agents/<name>.md`; current Zcode versions also discover the workspace templates (observed 2026-10-22). See `.zcode/README.md` for installation and version notes. |
| Antigravity / Gemini CLI | root `GEMINI.md` + `.agents/rules/team-coordination.mdc` | Rule is always-on; both route to this document | No Antigravity subagent adapters exist for this team by design; Antigravity compatibility means consistent instructions, not shared definition files. |
| GitHub Copilot | `.github/agents/*.agent.md` | Copilot's own discovery (`.agent.md` schema, unrelated to Zcode) | Copilot definitions are separate and unaffected by this policy. |

Zcode does not recursively discover instruction files or expand include syntax.
Nothing loads automatically: any entry point that wants this policy must say so
in plain text, which is why every entry point names this file explicitly.

## 2. Primary coordinator contract

The primary Zcode conversation is the repository coordinator and the accountable
integrator. It owns:

1. Interpreting the user's objective and the existing accepted decisions.
2. Recovering relevant context and identifying the current source of truth.
3. Decomposing work into bounded tasks with acceptance criteria.
4. Selecting specialists using the routing table in section 4.
5. Scheduling independent investigations concurrently and dependent work in order.
6. Reconciling conflicting specialist findings against repository evidence.
7. Making repository edits, integrating specialist recommendations, and running
   appropriate checks. **The coordinator is the only repository writer.**
8. Obtaining an independent final review for substantive changes.
9. Recording verified outcomes and producing the final report.

A specialist may recommend additional investigation, but the primary session
decides whether to dispatch it. Zcode workers cannot spawn workers, so all
dispatch stays in the primary session.

## 3. One writer and independent specialists

- **One writer.** Only the coordinator edits repository files. Specialists
  investigate, propose designs or patches (as diffs or recommendations in their
  reports), design meaningful checks, diagnose failures, and review results.
- **Read-only specialists.** Every specialist is restricted to `Read`, `Grep`,
  and `Glob`. A shell can write files, so no specialist gets `Bash`; when a
  specialist needs command evidence (for example a CI check), it proposes the
  exact command, the coordinator runs it, and the coordinator returns the
  evidence. There is currently no command-running validation role; if one is
  ever justified, its permitted commands, isolated output locations, and side
  effects must be documented in this file first.
- **No parallel implementation workers.** If a later authorized task needs
  them, require explicit file ownership or isolated worktrees, a named
  integration owner, and a rule against concurrent changes to shared contracts
  — recorded here before the work starts.
- **Review independence.** `tiny-village-reviewer` is separate from
  implementation and must not repair the material under review. Its job is to
  identify defects, unsupported claims, regressions, and missing evidence. The
  coordinator resolves findings; it does not delegate fixes back to the
  reviewer.

## 4. Routing table

| Agent | Purpose | Focus paths | Dispatch triggers | Exclusions | Tools | Expected output |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny-village-architecture-scout` | Map current architecture, decisions, subsystem ownership, active work, and compatibility constraints before substantive changes | Whole repo; `docs/decisions.md`, `memory-bank/`, root `*.py` inventory | Any task that will touch multiple subsystems, change shared contracts, or where current ownership is uncertain | Not for single-subsystem design work once the owning specialist is known | Read, Grep, Glob | Architecture map, ownership table, constraint list, open risks |
| `tiny-village-planning-actions` | GOAP planner, goals, utility scoring, action preconditions/effects, plan validation, caching, invalidation, replanning | `tiny_goap_system.py`, `actions.py`, `tiny_strategy_manager.py`, `goap_evaluator.py`, `tiny_utility_functions.py` | Plan quality, action catalog changes, goal/scoring behavior, replanning bugs | LLM prompt/parse contracts (route to llm-contracts); graph storage internals (route to world-state) | Read, Grep, Glob | Findings on planner behavior, proposed patches, test ideas |
| `tiny-village-world-state` | GraphManager/WorldState ownership, entity identity, shared graph coherence, time/events, resource lifecycle, save/load boundaries | `tiny_graph_manager.py`, `world_state.py`, `tiny_globals.py`, `tiny_characters.py`, `tiny_event_handler.py`, `tiny_time_manager.py`, `tiny_locations.py`, `tiny_jobs.py`, `CheckpointManager` in `tiny_gameplay_controller.py` | Graph schema or singleton changes, entity identity questions, event/time/persistence behavior, second-source-of-truth risks | GOAP scoring and LLM routing; presentation-layer rendering | Read, Grep, Glob | Ownership/coherence findings, invariants at risk, patch proposals |
| `tiny-village-llm-contracts` | Strategy routing thresholds, prompt generation, model I/O, output interpretation, action normalization, failure handling, decision-loop integration | `tiny_prompt_builder.py`, `tiny_brain_io.py`, `tiny_output_interpreter.py`, LLM wiring in `tiny_strategy_manager.py`, `llm_integration_utils.py`, `llm_character_utils.py` | Prompt template changes, parser/fallback bugs, LLM enable/disable behavior, contract drift between prompt/parse/action | Planner internals; memory retrieval quality | Read, Grep, Glob | Contract alignment findings, failure-mode analysis, patch proposals |
| `tiny-village-runtime-memory` | In-game memory recording/recall, basic vs full memory, semantic retrieval, capability reporting, lazy initialization, memory performance | `tiny_memories.py`, consumers in `tiny_characters.py`, `tiny_prompt_builder.py`, `cosine_threshold_query.py` | Memory behavior, embedding/retrieval issues, memory-performance questions | The contributor Memory Bank under `memory-bank/` (different system); LLM I/O contracts | Read, Grep, Glob | Memory pipeline findings, capability/fallback analysis, patch proposals |
| `tiny-village-social-narrative` | Relationships, social actions, jobs/economy interactions, events, storytelling, continuity | `social_model.py`, `tiny_jobs.py`, `tiny_event_handler.py`, `tiny_storytelling_system.py`, `tiny_storytelling_engine.py`, `tiny_story_arc.py`, `storytelling_integration.py` | Social mechanics, relationship effects on decisions, event consequences, story arc behavior | Graph storage ownership; presentation of social state | Read, Grep, Glob | Subsystem map, coupling findings, patch proposals |
| `tiny-village-ui-simulation` | Pygame presentation, controller/map interaction, animation, headless boundaries, presentation of authoritative simulation state | `tiny_gameplay_controller.py`, `tiny_map_controller.py`, `main.py`, `assets/`, design contracts in `docs/` | UI behavior, renderer init/recovery, headless/visual parity, input handling | Simulation logic ownership (graph, planning, memory); D013 requires both presentations — never propose dropping one | Read, Grep, Glob | Presentation-boundary findings, headless/visual parity analysis, patch proposals |
| `tiny-village-test-ci` | Independent test design, regression analysis, dependency profiles, deterministic verification, CI failure diagnosis | `tests/`, root `test_*.py`, `scripts/run_ci_checks.py`, `.github/workflows/python-ci.yml`, `.github/workflows/python-ml-profiles.yml`, `docs/guides/DEPENDENCY_PROFILES.md` | Designing meaningful checks for a change, diagnosing CI failures, profile questions | Running commands (proposes them; coordinator runs them and returns evidence); loosening assertions or adding permissive mocks | Read, Grep, Glob | Test design, expected outcomes, exact commands for the coordinator to run, failure analysis |
| `tiny-village-reviewer` | Independent final review of scope, contracts, permissions, integration, evidence, and remaining risks | The task's changed files plus the contracts they touch (for this team's setup work: `.zcode/`, `docs/agents/`, `AGENTS.md`, `GEMINI.md`, `.agents/`) | Substantive changes before closeout; agent-configuration changes; evidence review | Repairing anything it reviews; implementing fixes; re-reviewing after its own edits | Read, Grep, Glob | Defect/claim/regression/evidence-gap findings, checked and unchecked items |

Routing rules: pick the specialist whose focus paths contain the change. For
cross-cutting changes, dispatch `tiny-village-architecture-scout` first, then
the domain specialists in parallel. Combine small questions rather than
dispatching per file. Do not invoke every specialist automatically; trivial
changes need none.

Omitted role: a dedicated documentation/context agent. Context recovery is part
of the scout's assignment and closeout curation is a coordinator duty under
section 8; the Memory Bank already owns durable context, and a second curator
would compete with it.

## 5. Five-stage lifecycle

1. **Context recovery.** Read applicable instructions, accepted decisions in
   `docs/decisions.md`, relevant Memory Bank context, current source, and
   active work.
2. **Planning and scoping.** Define objective, exclusions, file ownership,
   dependencies, invariants, acceptance criteria, and verification approach.
3. **Specialist investigation and coordinator implementation.** Dispatch the
   necessary specialists, synthesize findings, then make focused changes.
4. **Independent verification and review.** Evaluate the actual resulting files
   against the task and relevant contracts, resolve findings, and recheck
   affected behavior.
5. **Knowledge closeout.** Update the existing context mechanism (Memory Bank
   task files; `docs/decisions.md` only for genuine human decisions) with
   verified outcomes, evidence, limitations, and remaining work.

Use the stages proportionately. Trivial changes do not need the full roster,
and context recovery is not repeated for every small subtask within one task.

## 6. Assignment template

Give each specialist enough context to work independently; it does not see the
conversation history, user intent, or another specialist's findings.

```text
Task ID: <short id, e.g. TV-123>
Question or outcome: <one specific question or deliverable>
Repository root: <supplied workspace root>; revision: <commit or working-tree snapshot note>
Relevant paths: <files/directories, repository-relative>
Applicable instructions: <AGENTS.md sections, docs/agents/team-coordination.md section, .github/instructions/*>
Accepted decisions: <relevant D0xx entries from docs/decisions.md, with boundary notes>
Scope: <what to examine>; Exclusions: <what is out of scope or owned elsewhere>
Allowed operations: read-only investigation (Read/Grep/Glob); no edits, no commands
Dependencies / earlier findings: <summaries this assignment may rely on>
Acceptance criteria: <what makes the answer usable>
Expected response: standard return contract (docs/agents/team-coordination.md §7)
Budget: <turn/effort bound>; Escalate if: <conditions>
```

## 7. Standard return contract

Every specialist report contains, concisely:

1. Task ID and status: complete, partial, or blocked.
2. Scope examined: relevant paths and symbols.
3. Findings, each supported by concrete evidence (path and line or symbol).
4. Which conclusions are observed, inferred, or proposed.
5. Recommended changes or next actions, with likely owning files.
6. Commands actually executed and results, or explicitly "not run" (specialists
   run none; if the coordinator supplied command evidence, it is cited).
7. Remaining uncertainty, dependencies, and blockers.
8. For reviews: what was checked and what remains unverified — a review with no
   defects still reports both.

Raw tool dumps are not findings. The coordinator records why conflicting
recommendations were accepted or rejected, and requests additional
investigation only to resolve a concrete uncertainty.

## 8. Context and memory discipline

- Durable human decisions live only in `docs/decisions.md` (D003); resumable
  task context lives in `memory-bank/`. Do not create a competing decision log
  or status record.
- Historical context is evidence, but it cannot override current verified
  source or explicitly accepted decisions. A specialist's recommendation is
  not human acceptance.
- Closeout updates the Memory Bank task file and `_index.md` on the same date;
  it does not fabricate decisions, approvals, or verification that did not
  occur.

## 9. Repository invariants

These apply to the coordinator and to every specialist, in addition to the
working agreements in root `AGENTS.md`:

1. Shared world state stays in the GraphManager graph; do not introduce a
   second authoritative simulation. `social_model.py` operating on `WorldState`
   is a known accepted exception — flag, do not extend.
2. Prompt contracts, output interpretation, executable actions, and action
   effects must remain aligned; a change to any one of them is a change to the
   contract.
3. Planning, execution, memory, and event feedback require explicit integration
   boundaries; keep them in their existing modules.
4. AI capabilities are required in the final product under
   [D015](../decisions.md#d015--require-ai-capabilities-in-the-final-product);
   dependency optionality exists for lightweight testing only.
5. Minimal startup/event/basic-memory success does not prove gameplay turns,
   live inference, visual readiness, or final-product readiness.
6. The accepted broader product vision and both headless and Pygame
   presentations are preserved under
   [D013](../decisions.md#d013--require-headless-and-pygame-presentations-for-the-mvp) and
   [D014](../decisions.md#d014--accept-the-mvp-contract-while-preserving-the-original-vision).
7. Preserve meaningful failure behavior and capability reporting. Do not
   replace real behavior with permissive mocks or fabricated success.
8. Use the current dependency profiles and CI evidence; do not assume native
   Windows/GPU/model behavior is verified by the Linux/offline checks
   (`scripts/run_ci_checks.py` hard-gates CPython 3.12 on Linux x86-64).
9. Preserve accepted decisions, review requirements, and existing task
   boundaries without reopening settled choices.

## 10. Escalation and blockers

A specialist escalates by returning `blocked` with the specific missing input;
it never expands its own scope, writes files, or runs commands to unblock
itself. The coordinator stops only the affected work stream when blocked:
completes independent deliverables, states the unresolved dependency, and
presents a concrete next step. Requirements are not silently relaxed and
approvals are not invented.

## 11. Fallback without specialist invocation

In environments where specialist invocation is unavailable (for example, a
session started before the definitions were installed, or a non-Zcode tool),
the primary agent performs the needed checks itself and reports plainly that no
independent specialist review occurred. It does not claim specialist
participation that did not happen.

## 12. Verification layers

Static checks (parsing definitions, path and link existence, consistency
between documents) never establish runtime loading or successful delegation.
Report each layer separately: defined, installed, discovered, invoked. Runtime
verification of Zcode definitions requires a fresh session after any definition
change; until then it is incomplete, not assumed.
