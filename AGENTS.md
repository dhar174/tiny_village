# AGENTS.md

## 1. Overview
Tiny Village is a single-repo Python simulation where contributor work usually touches an autonomous character decision loop backed by GOAP, graph state, memory, and optional LLM routing. This guide is for coding agents and maintainers: use it to place changes in the right layer, extend agent-specific tooling safely, and keep repository documentation aligned with the current root-level layout.

## 2. Folder Structure
- `.github`: repository automation and GitHub-specific metadata.
    - `agents`: specialized contributor-agent definitions in `.agent.md` format; use these for narrowly scoped task agents.
    - `issue_templates`: issue templates and intake structure.
- `assets`: screenshots and other game-facing assets.
- `docs`: current project documentation.
    - `guides`: end-user and getting-started docs.
    - `reference`: technical reference docs; `docs/reference/AGENTS.md` provides additional background and may be linked from other docs, but this root `AGENTS.md` is the primary up-to-date contributor guide while references are being consolidated.
    - `testing`: testing guidance, anti-pattern notes, and related references.
    - `archived`: historical or superseded documentation.
- `design_docs`: architecture deep dives and system-design notes.
- `critical_analysis`: implementation plans, code analysis, and audit-style writeups.
- `tests`: organized test suite, migration helpers, fixtures, and focused regression coverage.
- Repo root Python modules: the main runtime and many project entry points live at the repository root.
    - `main.py`: top-level entry point, logging setup, and demo-mode selection.
    - `actions.py`: action, state, and condition types plus the action catalog used by planning and execution.
    - `tiny_*.py`: core runtime systems such as characters, gameplay control, graph state, GOAP, memory, prompts, output interpretation, time, locations, and storytelling.
    - `demo_*.py`: subsystem demos and integration experiments.
    - `test_*.py`: additional root-level tests, often integration or regression focused.
    - `validate_*.py` and `verify_*.py`: ad hoc validation and smoke-check scripts for recent work.

## 3. Core Behaviors & Patterns
- **State Management**: `GraphManager` is the central world model and is exposed through `tiny_globals` helpers. Core systems should read or update shared simulation state through that graph instead of creating parallel sources of truth.
- **Decision Loop**: Character runtime work typically flows through gameplay orchestration, strategy selection, prompt generation, LLM I/O, output interpretation, action execution, and then memory and event updates back into shared graph state. Keep prompt contracts, parser expectations, and executable action definitions aligned when changing any part of this chain.
- **Hybrid Planning**: Routine behavior combines GOAP, utility evaluation, and optional LLM routing. Per-character LLM participation is toggled through `use_llm_decisions` and helper utilities such as `setup_full_llm_integration(...)`; when LLM components are unavailable, the system is expected to degrade gracefully rather than fail hard.
- **Dependency Fallbacks**: Many modules wrap optional imports in `try/except ImportError` and provide reduced-capability fallbacks. Preserve that pattern when touching NLP, ML, graph, or LLM-adjacent code so optional features do not become mandatory for unrelated workflows.
- **Logging**: Entry points and demos often configure logging with `logging.basicConfig(...)`, while reusable modules typically use `logging.getLogger(__name__)`. Add logs that explain state transitions, fallback reasons, or parsing failures instead of noisy line-by-line tracing.
- **Agent Lifecycle**: Specialized contributor agents belong under `.github/agents/*.agent.md`; gameplay-facing agent behavior belongs in runtime modules, not markdown. Typical contributor-agent use cases here include subsystem-specific agents for GOAP, output interpretation, storytelling, and end-to-end decision-loop work.

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

Both examples are examples only; check the current helper signatures and agent front matter before copying them into new work.

## 4. Conventions
- **Naming**: Classes use `PascalCase` (`StrategyManager`, `OutputInterpreter`), functions and variables use `snake_case` (`setup_full_llm_integration`, `get_global_graph_manager`), and constants use `UPPER_SNAKE_CASE` (`MAX_SPEED`, `NOTIFICATION_PRIORITIES`).
- **File Naming**: Core runtime modules stay at the repository root and usually use the `tiny_` prefix. Use `demo_*` for demos, `test_*` for tests, and `validate_*` or `verify_*` for ad hoc verification scripts.
- **Module Boundaries**: Keep decision orchestration, prompt generation, output parsing, action execution, graph state, and memory in their existing modules. Extend nearby code before introducing a new cross-cutting file.
- **Comments and Docstrings**: Use comments and docstrings for architectural context, behavioral constraints, or subtle implementation details, not for obvious line-by-line narration. If a path, subsystem name, or workflow changes, update the prose immediately rather than leaving stale guidance behind.
- **Agent Config Placement**: Put new specialized contributor-agent definitions in `.github/agents/*.agent.md`. Keep the root `AGENTS.md` as the canonical contributor guide instead of duplicating that content across new markdown files.
- **Examples Must Match Reality**: Use the present root-level layout in examples and docs. Do not invent alternate directory trees or placeholder package structures that the repository does not actually contain.

## 5. Working Agreements
- Respond in the user's preferred language; keep technical terms in English and never translate code blocks.
- Before changing agent-related logic, inspect neighboring flows and reuse existing helpers, templates, and recurring patterns.
- Prefer simple, minimal changes; preserve public behavior unless the user explicitly asks for a behavior change.
- For ad-hoc user requests, do not add tests, lint steps, formatting, or type-check work unless the user explicitly asks. For contributor-agent missions defined in `.github/agents/*.agent.md`, follow the mission spec: if it calls for adding or updating tests or tooling, do so.
- Be conservative with mocks and fake classes. This repository repeatedly favors tests that fail on real behavior mismatches over tests that are easy to satisfy.
- Keep prompt contracts, parser logic, and action definitions in sync when touching the LLM-backed decision loop.
- No repo-level type-check command is currently discoverable from project config; state that plainly rather than inventing one.
- Treat `docs/reference/AGENTS.md` as supplemental background. The root `AGENTS.md` is the canonical contributor guide for future agent work.
