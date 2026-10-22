# AGENTS.md

## 1. Overview
Tiny Village is a single-repo Python simulation where contributor work usually touches an autonomous character decision loop backed by GOAP, graph state, memory, and optional LLM routing. This guide is for coding agents and maintainers: use it to place changes in the right layer, extend agent-specific tooling safely, and keep repository documentation aligned with the current root-level layout.

AI capabilities are required in the final product under [D015](docs/decisions.md#d015--require-ai-capabilities-in-the-final-product).
Dependency optionality and basic memory support lightweight testing; they do not
reduce product scope or establish AI/gameplay readiness.

## 2. Folder Structure
- `.agents`: Antigravity rule entry points (`rules/team-coordination.mdc`) that load the same coordination policy as the Zcode entry points.
- `.github`: repository automation and GitHub-specific metadata.
    - `agents`: GitHub Copilot contributor-agent definitions in `.agent.md` format; use these for narrowly scoped task agents.
    - `instructions`: repository-specific Copilot instructions, memory-bank rules, and file-pattern guidance.
    - `issue_templates`: issue templates and intake structure.
    - `skills`: reusable agent skills and bundled workflow resources.
- `assets`: screenshots and other game-facing assets.
- `docs`: current project documentation.
    - `agents`: the specialist-team coordination policy (`team-coordination.md`) — routing table, lifecycle, and delegation contracts for the contributor agent team.
    - `guides`: end-user and getting-started docs (including the dependency-profile guide).
    - `reference`: technical reference docs; `docs/reference/AGENTS.md` is supplemental background, not the canonical root guide.
    - `testing`: testing guidance, anti-pattern notes, and related references.
    - `planning`: per-issue planning and approval records.
    - `superpowers`: bundled plans and specs used by repository workflows.
    - `archived`: historical or superseded documentation.
- `design_docs`: architecture deep dives and system-design notes.
- `critical_analysis`: implementation plans, code analysis, and audit-style writeups.
- `memory-bank`: resumable project context for contributors and agents.
    - `tasks`: task registry and per-task progress tracking files.
- `requirements`: dependency-profile system with one `.in` file per profile (`core`, `dev`, `visual`, plus offline ML profiles `nlp`, `llm-cpu`, `llm-gpu`, `retrieval`, `analytics` and shared `llm-common`) and hash-locked outputs under `locks/`. The root `requirements.txt` is a pointer file installed with `pip install --require-hashes --only-binary=:all: -r requirements.txt`; regenerate locks with `scripts/compile_requirements.py`.
- `scripts`: repository tooling, notably `run_ci_checks.py` (the bounded check runner CI invokes per profile) and `compile_requirements.py` (dependency lock compilation).
- `tests`: organized test suite, migration helpers, fixtures, and focused regression coverage.
- `.zcode`: Zcode workspace configuration. `agents/` holds repository-owned source templates for the `tiny-village-*` specialist subagents (current Zcode versions discover them from the workspace; the documented user-level location is `~/.zcode/agents/` — see `.zcode/README.md`), and `plans/` holds per-session plan files.
- Repo root Python modules: the main runtime and many project entry points live at the repository root.
    - `main.py`: top-level entry point, logging setup, and demo-mode selection.
    - `actions.py`: action, state, and condition types plus the action catalog used by planning and execution.
    - `tiny_*.py`: core runtime systems such as characters, gameplay control, graph state, GOAP, memory, prompts, output interpretation, time, locations, and storytelling.
    - `demo_*.py`: subsystem demos and integration experiments.
    - `test_*.py`: additional root-level tests, often integration or regression focused.
    - `validate_*.py` and `verify_*.py`: ad hoc validation and smoke-check scripts for recent work.

## 2.1 AI Documentation, Skills, and MCP Resources
- **Memory Bank:** Treat the Memory Bank as the resumability layer for human contributors and coding agents. Start with [`memory-bank/memory-bank-instructions.md`](memory-bank/memory-bank-instructions.md), then read the current summaries in [`memory-bank/`](memory-bank/). The broader instruction contract in [`.github/instructions/memory-bank.instructions.md`](.github/instructions/memory-bank.instructions.md) explains why the Memory Bank exists, the required read order, task tracking, and the `/update memory bank` workflow.
- **LangChain/LangGraph Python guidance:** When editing Python code related to LangChain, LangGraph, agents, RAG, tool calling, retrievers, vector stores, or tracing, consult [`.github/instructions/langchain-python.instructions.md`](.github/instructions/langchain-python.instructions.md) for repository-level framework expectations.
- **LangChain/LangGraph skill inventory:** The discoverable framework skills in this clone currently live under [`.github/skills/`](.github/skills/) rather than a top-level `skills/` folder: [`langchain`](.github/skills/langchain/SKILL.md), [`langgraph-project-setup`](.github/skills/langgraph-project-setup/SKILL.md), [`langgraph-agent-patterns`](.github/skills/langgraph-agent-patterns/SKILL.md), [`langgraph-state-management`](.github/skills/langgraph-state-management/SKILL.md), [`langgraph-error-handling`](.github/skills/langgraph-error-handling/SKILL.md), and [`langgraph-testing-evaluation`](.github/skills/langgraph-testing-evaluation/SKILL.md). Closely related LangSmith skills are [`langsmith-fetch`](.github/skills/langsmith-fetch/SKILL.md), [`langsmith-trace`](.github/skills/langsmith-trace/SKILL.md), [`langsmith-evaluator`](.github/skills/langsmith-evaluator/SKILL.md), and [`langsmith-dataset`](.github/skills/langsmith-dataset/SKILL.md).
- **Built-in `langchain-docs` MCP server:** In agent-capable environments for this repository, the LangChain docs MCP capability is exposed through the `docs-langchain-search_docs_by_lang_chain` tool. Use it before generic web search whenever you need current LangChain or LangGraph syntax, API behavior, guides, or code examples. Best results come from natural-language queries that name the API, concept, or task, such as `LangChain Runnable batch example`, `LangGraph StateGraph conditional edges`, or `LangGraph checkpointer persistence`; the tool returns titled matches with links and snippets.
- **External references:** [LangChain Docs](https://python.langchain.com/docs/), [LangGraph Docs](https://langchain-experimental.github.io/langgraph/), [LangChain API Reference](https://python.langchain.com/docs/api_reference), [LangGraph API Reference](https://langchain-experimental.github.io/langgraph/api/), [Model Context Protocol Docs](https://modelcontextprotocol.info/docs/), [MCP Specification](https://modelcontextprotocol.io/specification/latest), and [Model Context Protocol GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol).

## 3. Core Behaviors & Patterns
- **State Management**: `GraphManager` is the central world model and is exposed through `tiny_globals` helpers. Core systems should read or update shared simulation state through that graph instead of creating parallel sources of truth.
- **Decision Loop**: Character runtime work typically flows through gameplay orchestration, strategy selection, prompt generation, LLM I/O, output interpretation, action execution, and then memory and event updates back into shared graph state. Keep prompt contracts, parser expectations, and executable action definitions aligned when changing any part of this chain.
- **Hybrid Planning**: Routine behavior combines GOAP, utility evaluation, and optional LLM routing. Per-character LLM participation is toggled through `use_llm_decisions` and helper utilities such as `setup_full_llm_integration(...)`; when LLM components are unavailable, the system is expected to degrade gracefully rather than fail hard.
- **Dependency Fallbacks**: Many modules wrap optional imports in `try/except ImportError` and provide reduced-capability fallbacks. Preserve that pattern when touching NLP, ML, graph, or LLM-adjacent code so optional features do not become mandatory for unrelated workflows.
- **Logging**: Entry points and demos often configure logging with `logging.basicConfig(...)`, while reusable modules typically use `logging.getLogger(__name__)`. Add logs that explain state transitions, fallback reasons, or parsing failures instead of noisy line-by-line tracing.
- **Agent Lifecycle & Team Coordination**: Gameplay-facing agent behavior belongs in runtime modules, not markdown. Contributor-agent tooling has three distinct homes: GitHub Copilot definitions under `.github/agents/*.agent.md` (Copilot's `.agent.md` schema); Zcode specialist-team source templates under `.zcode/agents/*.md` (Zcode frontmatter, installed to the user-level `~/.zcode/agents/` per `.zcode/README.md`); and Antigravity, which loads the same coordination policy through root `GEMINI.md` and `.agents/rules/team-coordination.mdc`. The canonical policy is [`docs/agents/team-coordination.md`](docs/agents/team-coordination.md) — read it before substantive delegated work. In Zcode, the primary session is the coordinator and the only repository writer: the read-only `tiny-village-*` specialists investigate, review, and return findings for the coordinator to integrate and verify. A specialist reading injected AGENTS.md retains its assigned restrictions, scope, and return contract, and reports requests for additional work back to the primary session. Typical specialist use cases map to subsystems: GOAP and actions, graph/world state, LLM prompt/output contracts, runtime memory, social/narrative, UI/simulation presentation, test/CI design, architecture scouting, and independent review.

Illustrative contributor-agent template based on existing `.github/agents/*.agent.md` files:

```md
---
name: Example Agent
description: >
  Focus this agent on one subsystem and keep its mission narrowly scoped.
infer: false
tools:
  - read
  - search
  - edit
  - execute
metadata:
  component: example
  repo_area: ai_runtime
---
```

Illustrative runtime setup using existing LLM helper utilities:

```python
from llm_integration_utils import setup_full_llm_integration

enabled_characters, strategy_manager = setup_full_llm_integration(
    characters,
    llm_character_names=["Alice", "Bob"],
)
```

Both examples are illustrative only; verify the current helper signatures and agent front matter in the repository before copying them into new work.

## 4. Conventions
- **Naming**: Classes use `PascalCase` (`StrategyManager`, `OutputInterpreter`), functions and variables use `snake_case` (`setup_full_llm_integration`, `get_global_graph_manager`), and constants use `UPPER_SNAKE_CASE` (`MAX_SPEED`, `NOTIFICATION_PRIORITIES`).
- **File Naming**: Core runtime modules stay at the repository root and usually use the `tiny_` prefix. Use `demo_*` for demos, `test_*` for tests, and `validate_*` or `verify_*` for ad hoc verification scripts.
- **Module Boundaries**: Keep decision orchestration, prompt generation, output parsing, action execution, graph state, and memory in their existing modules. Extend nearby code before introducing a new cross-cutting file.
- **Comments and Docstrings**: Use comments and docstrings for architectural context, behavioral constraints, or subtle implementation details, not for obvious line-by-line narration. If a path, subsystem name, or workflow changes, update the prose immediately rather than leaving stale guidance behind.
- **Agent Config Placement**: Put new GitHub Copilot contributor-agent definitions in `.github/agents/*.agent.md`. Put Zcode specialist-team definitions in `.zcode/agents/*.md` (discovered from the workspace by current Zcode versions, and installable to `~/.zcode/agents/` per `.zcode/README.md`), and keep the routing table in `docs/agents/team-coordination.md` in sync with the definitions. Keep the root `AGENTS.md` as the canonical contributor guide instead of duplicating that content across new markdown files.
- **Examples Must Match Reality**: Use the present root-level layout in examples and docs. Do not invent alternate directory trees or placeholder package structures that the repository does not actually contain.

## 5. Working Agreements
- Respond in the user's preferred language; keep technical terms in English and never translate code blocks.
- Before changing agent-related logic, inspect neighboring flows and reuse existing helpers, templates, and recurring patterns.
- Prefer simple, minimal changes; preserve public behavior unless the user explicitly asks for a behavior change.
- For ad-hoc user requests, do not add tests, lint steps, formatting, or type-check work unless the user explicitly asks. For contributor-agent missions defined in `.github/agents/*.agent.md`, follow the mission spec: if it calls for adding or updating tests or tooling, do so.
- Be conservative with mocks and fake classes. This repository repeatedly favors tests that fail on real behavior mismatches over tests that are easy to satisfy.
- Keep prompt contracts, parser logic, and action definitions in sync when touching the LLM-backed decision loop.
- Python 3.12 is the pinned runtime (`.python-version`). The canonical test path is `python -B scripts/run_ci_checks.py --profile <core|dev|visual>`, which is what `.github/workflows/python-ci.yml` runs; the offline ML profiles (`nlp`, `llm-cpu`) run via `python-ml-profiles.yml`. Prefer this runner over ad hoc test invocations when validating changes.
- There is no `pyproject.toml`, `pytest.ini`, or lint/format/type-check configuration at the repository level, and CI has no lint or type-check step; state that plainly rather than inventing commands.
- Treat `docs/reference/AGENTS.md` and `.github/copilot-instructions.md` as supplemental background. The root `AGENTS.md` is the canonical contributor guide for future agent work.
