# Missing Elements for Minimal Tiny Village Demo

This document summarizes key functionality gaps discovered in the repository which currently prevent the game from running as a basic playable demo.

## 1. Main Entry Point
- `README.md` and `AGENTS.md` describe a `main.py` that should initialize the game, but no such file exists. The closest runnable script is `tiny_gameplay_controller.py` with a `__main__` block, but it lacks configuration handling and a clear startup routine.

## 2. Assets & Maps
- The repository does not include any images or map data. `tiny_map_controller.py` attempts to load a map image (e.g., `assets/default_map.png`) when instantiated, but the `assets/` directory and required files are missing. Without these resources the game cannot render a map or place characters/buildings.

## 3. Incomplete Core Systems
Several systems referenced in documentation are only partly implemented or exist as stubs:
- **Event System** -- `tiny_event_handler.py` has basic classes but the controller’s `_process_pending_events` method is largely TODOs. Events are not integrated into character decision making.
- **GOAP Planner Integration** -- `tiny_goap_system.py` includes planning logic, yet `StrategyManager` mostly selects single actions by utility rather than producing multi-step plans. The documented interaction of `StrategyManager → GOAPPlanner → ActionSystem` is incomplete.
- **Action Execution** -- many actions have placeholder `execute()` implementations. `ActionResolver` converts simple dictionaries into actions instead of utilizing richer definitions from `actions.py`.
- **LLM Interface** -- components like `PromptBuilder`, `TinyBrainIO` and `OutputInterpreter` exist, but their orchestration within the strategy loop is inconsistent. Prompt generation relies on hardcoded action lists and the pipeline lacks error handling for unpredictable LLM output.
- **Weather, Quest and Social systems** -- `GameplayController` includes `implement_weather_system`, `implement_quest_system`, and `implement_social_network_system`, yet these return basic stubs with no update methods.

## 4. World Data & Economy
- No sample building or character data is bundled (besides minimal defaults in the controller). Files such as `custom_buildings.json` and `job_roles.json` suggest external resources are expected, but they are not referenced during initialization.
- The economic simulation is labelled `STUB_IMPLEMENTED` in `critical_analysis/controller_analysis.md` and lacks item production/trading logic necessary for even simple buy/sell interactions.

## 5. Save/Load & Persistence
- `GameplayController` has `save_game_state` and `load_game_state` methods but they only handle a subset of character properties and do not persist complex state such as inventories, memories or the graph structure.

## 6. Tests & Dependencies
- The provided tests fail to run because required packages (e.g., `pyparsing`, `attrs`) are not included. This prevents validation of existing functionality.

## Conclusion
To achieve a minimal playable demo, the repository needs at least:
1. A real `main.py` or startup script that configures and launches `GameplayController`.
2. Basic assets (map image, building data, initial characters).
3. Working integrations between `EventHandler`, `StrategyManager`, `GOAPPlanner` and `ActionSystem` for simple goal-oriented behavior.
4. Implemented execution logic for core actions and minimal economy/quest/weather subsystems.
5. Passing unit tests with essential dependencies installed.
