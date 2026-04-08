## Project Scope

- TinyVillage is a single-repo Python simulation game centered on autonomous AI
  characters living in a dynamic village environment.
- The repository's primary runtime lives in root-level Python modules:
  `main.py`, `tiny_gameplay_controller.py`, `tiny_characters.py`,
  `tiny_graph_manager.py`, `tiny_goap_system.py`, `actions.py`, and the
  broader set of `tiny_*.py` systems.
- The project combines simulation gameplay, world-state tracking via a graph
  model, event handling, Goal-Oriented Action Planning (GOAP), memory systems,
  optional LLM-assisted decision making, and visual/map-oriented runtime
  components.

## Primary Goals

- Simulate believable autonomous character behavior using a combination of
  graph-backed world state, GOAP planning, utility evaluation, and optional LLM
  routing.
- Maintain a cohesive shared world model covering characters, items, jobs,
  locations, relationships, events, and time progression.
- Support both richer visual/pygame gameplay flows and simpler demo or fallback
  modes when optional dependencies are unavailable.
- Keep the repository operable as a Python project with clear docs, tests,
  demos, and contributor guidance.

## Repository Boundaries

- This repository is focused on the TinyVillage simulation and its supporting
  tooling and documentation — not on notebook-generation, LangGraph scaffolding,
  or any other unrelated workflow.
- Optional ML/NLP/LLM features should degrade gracefully when unavailable
  rather than blocking unrelated simulation flows.
- Historical docs exist under `docs/archived/`; current Memory Bank summaries
  should prioritize the root `README.md`, root `AGENTS.md`, current reference
  docs under `docs/reference/`, and design docs under `design_docs/`.
