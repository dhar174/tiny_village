## System Architecture

- TinyVillage uses a modular, root-oriented Python architecture rather than a
  `src/` package layout.
- The main application spans `main.py` and `tiny_gameplay_controller.py`,
  coordinating gameplay flow, time, and system initialization.
- Core domain state is centralized in `tiny_graph_manager.py`, which acts as
  the shared world model for characters, relationships, locations, items, and
  derived simulation data.
- Character behavior flows through `tiny_strategy_manager.py`,
  `tiny_goap_system.py`, `actions.py`, prompt generation, optional LLM
  interaction, output interpretation, and then memory updates.

## Key Technical Decisions

- **Graph-centered world state:** `GraphManager` is the shared source of truth
  for simulation entities and relationships, accessed via
  `tiny_globals.get_global_graph_manager()`. Core systems read and update state
  through that graph rather than maintaining parallel models.
- **Hybrid decision system:** Character behavior combines GOAP planning,
  utility-style evaluation, and optional LLM-assisted reasoning rather than
  depending on only one method.
- **Graceful degradation for optional features:** LLM, NLP, and related advanced
  dependencies are wrapped in `try/except ImportError` fallback patterns so
  optional features do not break unrelated simulation flows.
- **Root-level module organization:** Core runtime behavior lives at the
  repository root in files like `actions.py` and `tiny_*.py`, so docs and
  contributor guidance should reflect that actual layout.
- **Event-driven orchestration:** Events trigger strategy updates, which drive
  GOAP planning, action selection, execution, graph updates, and memory
  recording.

## Design Patterns in Use

- **Main loop orchestration:** `main.py` / `tiny_gameplay_controller.py`
  initialize systems and drive per-tick or per-event progression.
- **Strategy-to-action pipeline:** `EventHandler` triggers updates that move
  through `StrategyManager`, GOAP planning, and action execution.
- **Prompt / model / interpreter pipeline:** Prompt construction
  (`tiny_prompt_builder.py`), model I/O (`tiny_brain_io.py`), and output
  interpretation (`tiny_output_interpreter.py`) are separated across dedicated
  modules instead of being mixed into gameplay orchestration.
- **Memory integration after action outcomes:** Character experiences are
  recorded through `tiny_memories.py` after actions and world updates occur.
- **Fallback paths at every optional boundary:** Each LLM/NLP integration point
  provides a non-LLM code path so the simulation degrades gracefully.

## Component Relationships

- `EventHandler` detects or generates game events and routes them into the main
  application flow.
- `StrategyManager` gathers character state and action options, prioritizes
  goals, and invokes `GOAPPlanner`.
- `GOAPPlanner` evaluates goals and constructs candidate plans using action
  definitions and graph-derived difficulty/context.
- `actions.py` defines executable actions with preconditions, effects, and
  execution behavior.
- `tiny_prompt_builder.py`, `tiny_brain_io.py`, and `tiny_output_interpreter.py`
  form the optional LLM decision path.
- `tiny_memories.py` records and retrieves memory data influenced by character
  experiences and simulation events.
- `tiny_map_controller.py` manages visual/map presentation for gameplay flows
  that use a pygame display.
