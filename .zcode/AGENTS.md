# Tiny Village: contributor instructions

## Start every Zcode session

Read the user-level `~/.zcode/AGENTS.md` and `~/.zcode/skills/team-coordination/SKILL.md`, then this root guide and the project [`.zcode/skills/team-coordination/SKILL.md`](.zcode/skills/team-coordination/SKILL.md). Read skills by file path because user and project copies may share the name `team-coordination`.

[`docs/agents/team-coordination.md`](docs/agents/team-coordination.md) is the canonical contributor-team policy. The local skill adapts it to Zcode with the roster and handoff packets. The primary session is the coordinator and **only repository writer and command executor**. Project specialists investigate/review with Read/Grep/Glob, return proposals and evidence, and never dispatch agents. A delegated specialist retains that role even when these instructions are injected.

Establish the actual checkout, revision, local changes, task, and approval boundary. Preserve unrelated work and current accepted decisions. Follow the user's requested scope without turning a documentation, startup, or dependency task into broader gameplay or model work.

## Project and authority map

Tiny Village is a Python simulation of autonomous characters using GOAP, utility evaluation, graph state, memories, social relationships, events, and LLM-backed decision making. Runtime modules live primarily at the repository root; do not invent a `src/` package layout.

AI capabilities are required in the final product under D015. Both headless and Pygame presentations remain required under D013; D014 preserves the broader vision. Optional dependencies and reduced-capability fallbacks support lightweight checks; they do not reduce product scope or prove gameplay/live-model readiness.

| Area | Main paths and ownership |
| --- | --- |
| Entry, gameplay orchestration, presentation | `main.py`, `tiny_gameplay_controller.py`, `tiny_map_controller.py`, `tiny_animation_system.py`, `assets/` |
| Planning, goals, actions, utility | `tiny_goap_system.py`, `actions.py`, `tiny_strategy_manager.py`, `goap_evaluator.py`, `tiny_utility_functions.py` |
| Shared state and entity identity | `tiny_graph_manager.py`, `world_state.py`, `tiny_globals.py`, `tiny_characters.py` |
| Prompt/model/output contract | `tiny_prompt_builder.py`, `tiny_brain_io.py`, `tiny_output_interpreter.py`, `llm_integration_utils.py`, `llm_character_utils.py` |
| Runtime memory | `tiny_memories.py`, its character/prompt consumers, `cosine_threshold_query.py` |
| Social, events, time, world entities, narrative | `social_model.py`, `tiny_event_handler.py`, `tiny_time_manager.py`, `tiny_locations.py`, `tiny_buildings.py`, `tiny_jobs.py`, `tiny_storytelling_*`, `tiny_story_arc.py`, `storytelling_integration.py` |
| Human decisions | `docs/decisions.md` — the single maintained decision record |
| Contributor resumption | `memory-bank/` and its task registry; distinct from runtime memory and external retrieval |
| Checks and dependency profiles | `scripts/run_ci_checks.py`, `requirements/`, `tests/`, root `test_*.py`, `.github/workflows/` |

Use `docs/README.md` to find current documentation, `design_docs/` for architecture, `docs/testing/` and `docs/guides/DEPENDENCY_PROFILES.md` for checks. `docs/archived/` is historical. `docs/reference/AGENTS.md` and `.github/copilot-instructions.md` are supplemental; this root file is the canonical contributor guide.

## Recover context without competing stores

At task start, read [`memory-bank/memory-bank-instructions.md`](memory-bank/memory-bank-instructions.md) and the core summaries: `projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, and `progress.md`. Consult relevant `memory-bank/tasks/` records and [`docs/decisions.md`](docs/decisions.md). Verify summaries against current code and task evidence.

For historical retrieval, give `mem0ry4ai:memory-scout` the active root and scope **`project:tiny_village`**, task, paths, and restrictions. Its brief complements `tiny-village-architecture-scout`, which inspects current source and ownership. Neither substitutes for the Memory Bank or accepted decisions.

`tiny-village-runtime-memory` concerns **in-game** memory; it is not the contributor memory scout or steward. User-level mem0ry4ai is a retrieval aid, not a second project decision log, task registry, or world-state store. Product decisions remain in `docs/decisions.md`; resumable contributor task status remains in the Memory Bank. When a task calls for Memory Bank updates, keep the task file and `tasks/_index.md` dated consistently; record verified context, not fabricated approvals or private reasoning.

Use `mem0ry4ai:memory-steward` only for permitted useful durable knowledge after verification, supplying authoritative references and a precise write policy. Avoid duplicating Memory Bank status or decision text. The steward may curate permitted MCP memory; it cannot edit repository files or update the Memory Bank itself. A no-memory-write task overrides hook reminders.

## Architecture and behavior rules

- Shared simulation state belongs in the `GraphManager` graph exposed through `tiny_globals`; do not introduce another authoritative world model. `social_model.py` operating on `WorldState` is a known accepted exception: flag it when relevant, do not expand it.
- Keep the decision chain aligned: gameplay orchestration → strategy/GOAP → prompt → model I/O → parsed executable action → effects → memory/events/shared graph feedback. Prompt, parser, action IDs/definitions, preconditions, and effects form one contract.
- Keep planning, execution, memory, events, and presentation in their existing modules with explicit integration boundaries. Rendering presents authoritative state; it does not own another simulation.
- Preserve optional-import handling and meaningful reduced-capability/failure reporting. Per-character LLM routing (`use_llm_decisions`, integration helpers) must degrade safely when dependencies are unavailable without pretending inference occurred.
- Minimal startup, basic memory, or event checks do not establish a real gameplay turn, live inference, visual readiness, native Windows/GPU support, or final-product readiness. Respect the separate decision/evidence gates.
- Prefer real behavior checks over permissive mocks, fake classes, or assertions weakened merely to pass. Preserve public behavior unless the task authorizes changing it.

## Conventions, skills, and agent configuration

Use `PascalCase` classes, `snake_case` functions/variables, and `UPPER_SNAKE_CASE` constants. Root `tiny_*.py` modules hold systems; `demo_*`, `test_*`, `validate_*`, and `verify_*` retain their established roles. Reuse neighboring helpers before adding cross-cutting files. Reusable modules use `logging.getLogger(__name__)`; entry points configure logging. Explain transitions, failures, and fallback reasons rather than logging every line.

For LangChain/LangGraph/agent/RAG/tool/vector/tracing changes, read `.github/instructions/langchain-python.instructions.md`. Relevant framework and LangSmith skills live in `.github/skills/`; read the ones needed by the task rather than assuming Zcode discovers that directory automatically. Prefer the connected LangChain documentation search capability when advertised; otherwise use current primary documentation. Do not invent a missing tool name.

Zcode source profiles live in `.zcode/agents/tiny-village-*.md`; user-level copies may be at `~/.zcode/agents/`. See `.zcode/README.md`. Inspect the current session's advertised roster and effective permissions instead of assuming workspace discovery/reload from file presence. GitHub Copilot definitions use `.github/agents/*.agent.md`; Antigravity uses `GEMINI.md` and `.agents/rules/team-coordination.mdc`. Keep these systems distinct; do not overwrite or synchronize profiles as a side effect of an ordinary task.

## Verification and working agreements

Python 3.12 is pinned by `.python-version`. Dependencies use profile inputs under `requirements/` and hash-locked outputs under `requirements/locks/`; the root `requirements.txt` points into that system. Do not regenerate locks or install/download models without task scope. Existing installation guidance uses `pip install --require-hashes --only-binary=:all: -r requirements.txt`; lock compilation lives in `scripts/compile_requirements.py`.

The canonical bounded check runner is `python -B scripts/run_ci_checks.py --profile core` (choose `dev` or `visual` for the corresponding scope). CI also covers selected offline ML profiles through `.github/workflows/python-ml-profiles.yml`. The runner gates CPython 3.12 on Linux x86-64: this Windows checkout cannot establish that gate by a native run. Use the appropriate supported environment when authorized, and report unavailable checks honestly. There is no root lint/format/type-check configuration to invent.

For ad-hoc requests, do not add tests or lint/format/type-check tooling unless requested; follow a contributor-agent mission's explicit test requirements when applicable. Select existing checks proportionate to code changes. Documentation-only coordination changes need definition/link/discovery checks, not gameplay or model execution. Inspect actual exit codes, stdout/stderr, file readback, and diffs before reporting success. Keep local checks, hosted CI, inference, gameplay, and platform evidence separate.

Use the canonical policy's proportionate lifecycle and independent review for substantive changes. Specialists propose commands; the coordinator executes them. State when independent review or a verification layer was unavailable. Respond in the user's preferred language, keep code and technical identifiers intact, and report concrete outcomes, limitations, and remaining authorized work.
