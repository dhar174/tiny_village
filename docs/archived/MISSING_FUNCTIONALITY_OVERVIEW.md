# Missing Functionality Needed for Minimal Tiny Village Demo

This document summarizes notable gaps that prevent running Tiny Village as a playable demo. Analysis is based on repository inspection and the existing `TODO_report.md`.

## Missing Assets and Entry Point
- The project documentation expects an `assets/` directory and a `maps/` folder as well as a `main.py` entry point【F:AGENTS.md†L11-L12】. None of these exist in the repository:
  - `assets/` directory: `ls assets` returns "No such file or directory"【7a99a0†L1-L3】
  - `maps/` directory: `ls maps` returns "No such file or directory"【4e53de†L1-L3】
  - `main.py` is also absent【8fc1cc†L1-L3】

Without map images, sprites, and a main script to start the systems, the game loop in `tiny_gameplay_controller.py` cannot be launched by users.

## Incomplete Core Systems
The `critical_analysis/TODO_report.md` file outlines numerous subsystems that remain unfinished. Key areas include:

- **Gameplay Controller** – Many TODOs in `tiny_gameplay_controller.py` show missing event processing, input handling, rendering features, and performance tools. Event-driven storytelling, mod support, and multiplayer are not started【F:critical_analysis/TODO_report.md†L5-L80】.
- **GOAP Planner & Strategy Manager** – The GOAP planning algorithm is partially implemented. Integration between `StrategyManager` and `GOAPPlanner` is unclear, and action execution relies on placeholder logic【F:critical_analysis/TODO_report.md†L121-L159】.
- **Memory System** – `tiny_memories.py` has advanced structures but lacks full integration with decision making and efficient data structures usage【F:critical_analysis/TODO_report.md†L131-L138】.
- **Action Definitions** – `actions.py` contains basic templates; additional actions with comprehensive preconditions/effects are needed for meaningful gameplay【F:critical_analysis/TODO_report.md†L181-L189】.
- **Event, Item, Job Systems** – These systems are stubbed and need expanded content, interactions, and connections to the economy and quests【F:critical_analysis/TODO_report.md†L191-L220】.

## Testing
Running `python -m unittest discover tests` fails because the `tests/` directory is missing. Note that if the directory existed but lacked proper structure (e.g., missing `__init__.py` files), `unittest` discovery would fail with the error "Start directory is not importable."

## Conclusion
To run a minimal Tiny Village demo, the project requires at least:
1. A main entry script and map/asset resources.
2. A functioning event loop linking `EventHandler`, `StrategyManager`, and `GOAPPlanner`.
3. More complete implementations of action, event, memory, and job systems.
4. Basic UI and rendering assets.

Addressing these gaps will enable a playable demonstration of the Tiny Village simulation.

