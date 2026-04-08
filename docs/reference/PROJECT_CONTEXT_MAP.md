# Project Context Map

## Purpose

This document captures the current repository context for Step 1 of the
codebase-analysis workflow: surveying the live file system, identifying the
main modules and their relationships, locating tests and documentation, and
highlighting follow-on risk areas for deeper architecture work.

It complements:

- `README.md` for top-level run commands and broad module descriptions
- `AGENTS.md` for the canonical contributor-facing folder summary
- `design_docs/high_level_architecture.md` for subsystem concepts
- `design_docs/module_connectivity_map.md` for concrete runtime relationships
- `design_docs/project_folder_blueprint.md` for the file-system blueprint

## Structural Snapshot

- Repository style: root-oriented Python modular monolith
- Target Python: 3.12
- Root-level core modules: 29 `tiny_*.py` files
- Root-level demos: 26 `demo_*.py` files
- Root-level tests: 49 `test_*.py` files
- Root-level validation scripts: 9 `validate_*.py` / `verify_*.py` files
- `tests/` contents: 98 files total, including 83 `test_*.py` files plus helper
  scripts, fixtures, and test artifacts

## Important Directories

| Path | Purpose | Notable contents | Notes |
| --- | --- | --- | --- |
| `/.github` | Automation and Copilot configuration | `agents/`, `instructions/`, `skills/`, `prompts/`, `issue_templates/` | Large directory because bundled skills include example assets and templates |
| `/assets` | Game-facing assets | `default_map.png` | Small but runtime-relevant |
| `/critical_analysis` | Point-in-time implementation analysis | issue summaries, code analysis, plans | Useful background, not always current |
| `/design_docs` | Architecture and design deep dives | connectivity map, architecture docs, blueprint | Best source for system-level analysis |
| `/docs` | Current documentation tree | `guides/`, `reference/`, `testing/`, `archived/` | `docs/README.md` defines the documentation taxonomy |
| `/memory-bank` | Persistent contributor/agent context | core summaries, `tasks/` registry | Should stay synchronized with current repo state |
| `/tests` | Organized regression and subsystem coverage | unit/integration tests, helpers, artifacts | Flat layout; not the repo's only test surface |
| `/` root | Runtime modules, demos, tests, data, and scripts | `main.py`, `actions.py`, `tiny_*.py`, `demo_*.py`, `test_*.py` | Mixed surface with both production and support files |

## Root Runtime and Script Surfaces

| Group | Representative files | Role |
| --- | --- | --- |
| Entry points and orchestration | `main.py`, `tiny_gameplay_controller.py`, `tiny_event_handler.py`, `tiny_globals.py` | Start the application, wire systems, drive the update loop |
| World-state and domain model | `tiny_graph_manager.py`, `world_state.py`, `tiny_characters.py`, `tiny_locations.py`, `tiny_buildings.py`, `tiny_building_manager.py`, `tiny_items.py`, `tiny_jobs.py`, `social_model.py`, `tiny_types.py` | Hold shared simulation entities and backing state |
| Planning and action execution | `actions.py`, `tiny_strategy_manager.py`, `tiny_goap_system.py`, `goap_evaluator.py`, `tiny_utility_functions.py`, `tiny_util_funcs.py` | Evaluate goals, generate plans, and execute world changes |
| LLM and prompt pipeline | `tiny_prompt_builder.py`, `tiny_brain_io.py`, `tiny_output_interpreter.py`, `llm_integration_utils.py`, `llm_character_utils.py` | Optional prompt generation, model I/O, and response interpretation |
| Memory and narrative systems | `tiny_memories.py`, `tiny_memories_alpha.py`, `tiny_storytelling_system.py`, `tiny_storytelling_engine.py`, `tiny_story_arc.py`, `storytelling_integration.py` | Store memories and support narrative/story systems |
| Visual and simulation support | `tiny_map_controller.py`, `tiny_animation_system.py`, `tiny_time_manager.py` | Manage map, display-adjacent behavior, and simulation time |
| Demo and verification scripts | `demo_*.py`, `validate_*.py`, `verify_*.py` | Provide isolated experiments, smoke checks, and issue validation |

## Current Module Relationship Map

| File | Reads from / depends on | Updates or calls into | Why it matters |
| --- | --- | --- | --- |
| `main.py` | CLI args, config | `GameplayController`, minimal demo entry points, integration test entry points | Selects the runtime mode |
| `tiny_gameplay_controller.py` | strategy, time, graph, buildings, map, story systems | per-tick character/event updates and rendering flow | Real integration hub for the full loop |
| `tiny_globals.py` | singleton state | `GraphManager` creation/access | Cross-file service locator |
| `tiny_graph_manager.py` | `WorldState`, registered entities, analytics inputs | graph-backed world state and derived queries | Shared source of truth |
| `world_state.py` | entity dictionaries and graph backing store | `GraphManager` delegated state | Underlying storage layer |
| `tiny_characters.py` | graph manager, actions, memory, time | character-local behavior and graph-backed state | Primary domain agent object |
| `actions.py` | preconditions, effects, graph manager | object state plus graph state changes | Execution layer for chosen actions |
| `tiny_strategy_manager.py` | graph manager, GOAP, utilities, optional LLM path | action ranking, plan generation, prompt/LLM pipeline | Decision orchestration |
| `tiny_goap_system.py` | character state, graph manager, action definitions | plan generation and evaluation | Search/planning engine |
| `tiny_prompt_builder.py` | character state, environment, memories | formatted prompt strings | LLM context assembly |
| `tiny_brain_io.py` | prompt text | model call boundary | LLM I/O layer |
| `tiny_output_interpreter.py` | model output and candidate actions | action resolution/fallback behavior | Converts text back into executable choices |
| `tiny_memories.py` | character experiences, optional NLP/embedding stack | memory storage and retrieval | Long-term context and retrieval |

## Test Coverage Map

| Test surface | Coverage shape | Examples | Notes |
| --- | --- | --- | --- |
| Root `test_*.py` files | Integration, issue validation, subsystem regression | `test_global_graph_manager.py`, `test_map_interactivity.py`, `test_storytelling_integration.py` | Important because many structure docs understate root-level tests |
| `tests/` flat suite | Broader organized coverage plus helpers and fixtures | `tests/test_actions.py`, `tests/test_building_manager.py`, `tests/test_minimal_demo_smoke.py` | Contains both tests and support files |
| Smoke / subprocess tests | End-to-end runtime validation | `tests/test_minimal_demo_smoke.py` | Useful for checking live entry points |
| Memory-focused guidance | Preferred test-design patterns | `docs/testing/MEMORY_TESTING_BEST_PRACTICES.md`, `docs/testing/MEMORY_TESTING_GUIDELINES.md` | Strong guidance against permissive over-mocking |
| Social/action helpers | Shared realistic test utilities | `tests/social_action_test_utils.py` | Evidence of reusable test support patterns |

## Documentation and Analysis Sources

| Source | Best use |
| --- | --- |
| `README.md` | Run commands, broad module list, top-level orientation |
| `AGENTS.md` | Canonical contributor guidance and folder summary |
| `docs/README.md` | Current documentation taxonomy and destination rules |
| `design_docs/high_level_architecture.md` | Conceptual subsystem overview |
| `design_docs/module_connectivity_map.md` | Concrete file/function/class connectivity |
| `docs/testing/TEST_FILES_README.md` | Historical notes on some test files and verification scripts |
| `memory-bank/*.md` | Current persistent repo context for contributors and agents |

## Reference Patterns

| File | Pattern to reuse |
| --- | --- |
| `AGENTS.md` | Accurate, current high-level folder summary grounded in the live repo |
| `docs/README.md` | Documentation placement rules and current-vs-archived distinction |
| `design_docs/module_connectivity_map.md` | Implementation-grounded subsystem relationship mapping |
| `docs/testing/MEMORY_TESTING_BEST_PRACTICES.md` | Realistic test-object guidance for memory-related tests |

## Complexity and Ambiguity Hotspots

- **Mixed root surface:** The repository root contains production modules, demos,
  tests, data files, screenshots, binary artifacts, and validation scripts.
- **Two active test surfaces:** Both root `test_*.py` files and `tests/`
  contain important coverage; neither can be ignored during later analysis.
- **Heavy optional dependencies:** `requirements.txt` includes NLP, embedding,
  and CUDA-adjacent packages such as `bitsandbytes` and `cupy-cuda11x`, so some
  AI paths are environment-sensitive.
- **Point-in-time analysis docs:** `design_docs/` and `critical_analysis/`
  contain useful analysis, but some documents may lag behind the current code.
- **LLM integration boundaries:** Prompt generation, model I/O, fallback logic,
  and action execution are split across multiple files and do not always follow
  a single fully completed path.
- **Committed runtime artifacts:** The repo includes committed `.bin`, `.pkl`,
  `.png`, `.json`, and text outputs, so not every root file is source code.

## Known Risks for Steps 2–3

- Updating architecture docs without checking the live controller path can
  overstate the completeness of the direct LLM-driven turn flow.
- Assuming `/tests` is the only test location will miss many root regressions.
- Treating archived or analysis documents as canonical can propagate stale
  structure claims.
- Any follow-on module grouping work should preserve the current root-oriented
  layout rather than inventing a `src/` package that does not exist.

## Validation Notes

- Baseline repository validation before this document was added:
  `python -m unittest test_map_controller_building_info`
- Result: passed locally in this clone
