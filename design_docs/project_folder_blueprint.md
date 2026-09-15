# Project Folder Structure Blueprint

## Configuration Summary

- **Project Type:** Python simulation / AI environment
- **Monorepo:** `false`
- **Microservices:** `false`
- **Frontend Included:** `false` (pygame-backed visual mode rather than a
  separate web frontend)
- **Visualization Style:** Markdown list with targeted tables
- **Depth Level:** 2 for the primary tree, with deeper notes for important areas
- **Generated Folders Included:** only when they are committed and relevant

## Structural Overview

TinyVillage uses a root-oriented modular-monolith layout:

- Core runtime modules live directly at the repository root.
- Important non-code material is grouped into dedicated top-level directories:
  `.github/`, `assets/`, `critical_analysis/`, `design_docs/`, `docs/`,
  `memory-bank/`, and `tests/`.
- The root is intentionally mixed: production modules, demos, regression tests,
  validation scripts, screenshots, serialized data, and runtime artifacts all
  coexist there.
- Current documentation is split between:
  - `docs/` for living guides, reference, and testing guidance
  - `design_docs/` for architecture deep dives and point-in-time analysis
  - `memory-bank/` for persistent contributor/agent context

## Directory Visualization

- **`/`** (root runtime and support surface)
  - **`.github/`** — repository automation and Copilot configuration
    - `agents/`
    - `hooks/`
    - `instructions/`
    - `issue_templates/`
    - `prompts/`
    - `skills/`
  - **`assets/`** — game-facing assets
  - **`critical_analysis/`** — implementation and code analysis snapshots
  - **`design_docs/`** — architecture and connectivity documents
  - **`docs/`** — curated current documentation
    - `archived/`
    - `guides/`
    - `reference/`
    - `testing/`
  - **`memory-bank/`** — persistent repo context and task tracking
    - `tasks/`
  - **`tests/`** — organized regression suite plus helpers and fixtures
  - **Root Python/runtime files**
    - `main.py`
    - `actions.py`
    - `world_state.py`
    - `tiny_*.py`
    - `demo_*.py`
    - `test_*.py`
    - `validate_*.py`
    - `verify_*.py`
  - **Root data and artifacts**
    - `*.json`
    - `*.txt`
    - `*.csv`
    - `*.bin`
    - `*.pkl`
    - `*.png`

## Key Directory Analysis

### `/` (repository root)

**Purpose:** Hosts the runtime entry points, core simulation modules, demos,
top-level tests, validation scripts, and committed data/artifacts.

**Current shape:**

- 183 root files
- 29 `tiny_*.py` modules
- 26 `demo_*.py` scripts
- 49 root `test_*.py` files
- 9 `validate_*.py` / `verify_*.py` scripts

**Key conventions:**

- Runtime and domain modules use `tiny_` prefixes.
- Demo and issue-verification scripts stay at the root for direct execution.
- Not all root files are source code; many are reference data or committed
  runtime artifacts.

### `/.github/`

**Purpose:** Repository automation, agent configuration, reusable skills, and
workflow instructions.

**Observed structure:**

- `agents/` contains contributor-agent definitions
- `instructions/` contains repository-specific Copilot rules and workflows
- `skills/` contains reusable skills plus bundled assets/templates/examples
- `prompts/`, `hooks/`, and `issue_templates/` support repository workflows

**Caveat:** This directory has many nested files because skills bundle example
assets and templates; raw file counts here are not a signal of runtime
complexity in the main application.

### `/assets/`

**Purpose:** Static game assets.

**Observed contents:** currently small and focused, with `default_map.png` as
the primary committed asset.

### `/critical_analysis/`

**Purpose:** Point-in-time audits, implementation summaries, and analysis
documents.

**Observed contents:** controller/graph/utility analyses and issue-oriented
implementation summaries.

**Usage guidance:** helpful for historical reasoning, but should not override
current runtime and current docs when conflicts appear.

### `/design_docs/`

**Purpose:** System-level architecture and design deep dives.

**Representative files:**

- `high_level_architecture.md`
- `module_connectivity_map.md`
- `graph_manager_deep_dive.md`
- `memory_manager_deep_dive.md`
- `project_folder_blueprint.md`

**Usage guidance:** best location for architecture analysis and structural
blueprints; however, these documents are still analysis artifacts and should be
cross-checked against live code.

### `/docs/`

**Purpose:** Current documentation intended for ongoing use.

**Subdirectories:**

- `guides/` — end-user and getting-started documents
- `reference/` — current technical reference
- `testing/` — testing practices and anti-pattern guidance
- `archived/` — historical, superseded, or issue-completion writeups

**Placement rule:** new living navigational documentation belongs here before
it belongs in `critical_analysis/` or `docs/archived/`.

### `/memory-bank/`

**Purpose:** Persistent context layer for contributors and coding agents.

**Observed contents:**

- core repo summaries (`projectbrief.md`, `productContext.md`, etc.)
- `memory-bank-instructions.md`
- `tasks/` for task-level tracking

**Usage guidance:** keep task tracking and resumability context synchronized with
current repository state.

### `/tests/`

**Purpose:** Organized regression coverage, subsystem tests, smoke tests,
fixtures, and helper utilities.

**Current shape:**

- 98 committed files
- 83 `test_*.py` files
- helper/support files such as `mock_character.py`,
  `social_action_test_utils.py`, `skill_test_utils.py`, test JSON/NPY/TXT
  artifacts, and migration/demo helpers

**Important caveat:** this is not the repo's only test surface. Many active
regressions also live at the repository root as `test_*.py`.

## File Placement Patterns

### Runtime and domain logic

- `main.py` is the documented top-level entry point.
- Core domain/runtime modules remain at the root and typically use `tiny_`
  prefixes.
- Shared world-state and orchestration code stays near the root rather than in a
  nested Python package.

### Demos and validations

- `demo_*.py` files live at the root and are intended for direct execution.
- `validate_*.py` and `verify_*.py` files provide issue-focused verification and
  smoke-check paths.

### Tests

- Root `test_*.py` files often cover integration flows, issue regressions, or
  quick validation scripts.
- `tests/` contains broader subsystem coverage plus support files and fixtures.
- Test helpers may live in `tests/` even when they are not named `test_*.py`.

### Documentation

- User-facing usage docs go under `docs/guides/`.
- Current technical reference belongs in `docs/reference/`.
- Testing guidance belongs in `docs/testing/`.
- Historical summaries or superseded reports belong in `docs/archived/`.
- Architecture deep dives and analysis-driven structural documents belong in
  `design_docs/`.

## Naming and Organization Conventions

| Surface | Convention | Examples |
| --- | --- | --- |
| Root runtime modules | `tiny_*.py` | `tiny_graph_manager.py`, `tiny_prompt_builder.py` |
| Demo scripts | `demo_*.py` | `demo_goap_integration.py` |
| Root tests | `test_*.py` | `test_storytelling_integration.py` |
| Validation scripts | `validate_*` / `verify_*` | `validate_storytelling_integration.py`, `verify_fixes.py` |
| Directories | `snake_case` or `kebab-case` | `design_docs`, `memory-bank` |
| Python symbols | standard Python style | `PascalCase` classes, `snake_case` functions/variables |

## Navigation and Development Workflow

### Entry points

- Start with `README.md` for run commands.
- Use `AGENTS.md` for the canonical contributor-facing folder map.
- Use `design_docs/module_connectivity_map.md` when tracing subsystem
  relationships.
- Use `docs/README.md` to decide where new documentation should live.

### Common development tasks

| Task | Best starting point |
| --- | --- |
| Change startup or mode selection | `main.py` |
| Change game-loop cadence or system orchestration | `tiny_gameplay_controller.py` |
| Change shared world-state behavior | `tiny_graph_manager.py`, `world_state.py` |
| Add or modify actions | `actions.py`, then relevant strategy/interpreter files |
| Change planning heuristics | `tiny_strategy_manager.py`, `tiny_goap_system.py`, `tiny_utility_functions.py` |
| Change prompt or LLM behavior | `tiny_prompt_builder.py`, `tiny_brain_io.py`, `tiny_output_interpreter.py` |
| Add current technical documentation | `docs/reference/` |
| Add architecture analysis | `design_docs/` |
| Add regression tests | either root `test_*.py` or `tests/`, depending on neighboring patterns |

## Build and Output Organization

- Python version target is documented as 3.12.
- Dependency installation is driven by root `requirements.txt`.
- No canonical repo-wide build step is documented; this is an interpreted Python
  project.
- The dependency set includes optional, heavyweight AI/NLP/GPU-adjacent
  packages, so environment compatibility should be treated carefully.
- The repository includes committed `.bin`, `.pkl`, `.png`, and dataset files,
  so output-like files are not universally disposable or ignored.

## Technology-Specific Organization Notes

- Graph-backed simulation state is centered on `tiny_graph_manager.py` and
  `world_state.py`.
- GOAP and action-selection logic are split across `actions.py`,
  `tiny_strategy_manager.py`, `tiny_goap_system.py`, and utility modules.
- The optional LLM path is distributed across prompt, model-I/O, and output
  interpretation modules rather than a single adapter file.
- Narrative and memory systems have their own top-level modules rather than
  nested packages.

## Extension and Evolution Guidance

- Prefer extending the existing root-oriented module layout before introducing
  new cross-cutting directories.
- When adding documentation, follow the `docs/README.md` taxonomy instead of
  creating new ad hoc top-level doc folders.
- When adding tests, first inspect neighboring files to decide whether the new
  test belongs in the root or in `tests/`.
- Any future structural cleanup should preserve current import and execution
  patterns until a deliberate package-migration plan exists.

## Structure Enforcement and Review

- Re-check structure documents against the live tree before using them as the
  basis for broad refactors.
- Avoid claiming that `/tests` is the only test location; include root tests in
  future analyses.
- Treat `docs/archived/` and many `critical_analysis/` documents as historical
  context rather than canonical structure references.

## Maintenance

- **Last updated:** 2026-04-08
- Revisit this blueprint whenever:
  - a new top-level directory is introduced
  - runtime modules move out of the repository root
  - test organization changes materially
  - documentation taxonomy changes in `docs/README.md`
