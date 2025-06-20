## Overall Architecture

The TinyVillage system is an agent-based simulation. Key components include:
- **`MainApplication` / `tiny_gameplay_controller.py`**: Initializes systems, runs the game loop, manages time.
- **`EventHandler` (`tiny_event_handler.py`)**: Detects and queues game events.
- **`StrategyManager` (`tiny_strategy_manager.py`)**: Orchestrates high-level character decision-making, using GOAP and LLM prompts.
- **`GraphManager` (`tiny_graph_manager.py`)**: Central repository for game world state (entities, relationships), calculates derived data.
- **`Character` (`tiny_characters.py`)**: Represents AI agents, holds personal state, makes decisions.
- **`ActionSystem` (`actions.py`)**: Defines rules for actions (preconditions, effects, costs).
- **`GOAPPlanner` (`tiny_goap_system.py`)**: Generates action plans to achieve character goals.
- **`LLM_Interface` (Conceptual: `PromptBuilder`, `TinyBrainIO`, `OutputInterpreter`)**: Handles LLM communication for decision augmentation.
- **`MemoryManager` (`tiny_memories.py`)**: Manages character memories, including NLP processing and retrieval.
- **`MapController` (`tiny_map_controller.py`)**: Manages visual representation and map interactions.

The architecture is event-driven, with clear layers for event handling, control, strategy, planning, utility, and data.

## Data Flow for Character Decision-Making (e.g., "New Day" event)

1.  **Event Trigger**: `EventHandler` detects an event (e.g., "New Day") and sends it to `MainApplication`.
2.  **Strategy Initiation**: `MainApplication` triggers `StrategyManager.update_strategy()` for a character.
3.  **State Gathering**: `StrategyManager` fetches character state and available actions from `GraphManager` and `ActionSystem`.
4.  **Goal Evaluation**: `StrategyManager` or `Character` (via `GOAPPlanner`) evaluates and prioritizes goals based on needs, motives, and world state.
5.  **Action Planning (GOAP)**: `GOAPPlanner.goap_planner()` generates a sequence of actions (a `Plan`) to achieve the prioritized goal. This involves checking preconditions and simulating effects.
6.  **LLM Prompting (Optional/Implicit)**: If GOAP doesn't suffice or for more nuanced decisions, `PromptBuilder` generates a prompt for the LLM based on character context, needs, and potential plans/actions.
7.  **LLM Interaction**: `TinyBrainIO` sends the prompt to the LLM and receives a response.
8.  **Output Interpretation**: An `OutputInterpreter` (conceptual) parses the LLM response into executable game actions.
9.  **Action Execution**: `GameplayController` (or `Character` directly) executes the action(s) using `ActionSystem.execute()`. This involves final precondition checks and applying effects, which update the `GraphManager` and `Character` state.
10. **Memory Update**: The `Character` records significant events or action outcomes as memories via `MemoryManager.add_memory()`, which processes and indexes them.

## Strategies and Memories

**Strategies:**
- Orchestrated by `StrategyManager`.
- Involve selecting and prioritizing goals using `GOAPPlanner.evaluate_goal_importance()`. This considers character needs, motives, personality, relationships, and environmental factors from `GraphManager`.
- `StrategyManager` can plan daily activities, respond to job offers, etc.
- It uses `GOAPPlanner` to generate action sequences (plans) to achieve these goals.
- Utility functions (`tiny_utility_functions.py`) provide quantitative measures for decision-making and goal/plan evaluation.

**Memories:**
- Managed by `MemoryManager` (`tiny_memories.py`).
- **Structure**:
    - `Memory` (base class).
    - `GeneralMemory`: Broad categories (e.g., "Social Interactions"), contains an index for specific memories.
    - `SpecificMemory`: Individual experiences (e.g., "Met John"), includes description, timestamp, importance, sentiment, emotion, extracted facts, SVO triples. Undergoes NLP analysis.
    - `MemoryBST`: AVL trees for sorting specific memories by attributes.
- **NLP Pipeline**: `MemoryManager.analyze_query_context()` processes text (for new memories or queries) using spaCy, sentence embeddings, sentiment/emotion analysis, keyword/fact extraction.
- **Storage & Retrieval**:
    - Specific memories are linked to General memories.
    - `FlatMemoryAccess` provides global search across all specific memories using a FAISS index for semantic similarity search on embeddings.
    - Retrieval involves processing a query, searching the FAISS index, and returning relevant `SpecificMemory` objects.
- **Purpose**: To give characters a rich, human-like recall ability, influencing their understanding and decisions.

## Action System Design

- **`actions.py`**:
    - `State`: Represents the current state of game entities.
    - `Condition`: Defines criteria for preconditions or goal states (attribute, target value, operator).
    - `Action`: Core unit. Contains `name`, `preconditions` (dict of `Condition`s), `effects` (list of state changes), `cost`, `target`, `initiator`.
        - `preconditions_met()`: Checks if an action can be performed.
        - `apply_effects()`: Modifies a `State`.
        - `execute()`: Checks preconditions, applies effects, updates `GraphManager`.
    - `ActionTemplate` / `ActionGenerator`: For creating instances of generic actions.
    - `ActionSystem`: Manages action definitions and provides execution methods.
- **`tiny_goap_system.py`**:
    - `Plan`: A sequence of actions to achieve goals. Has an action queue and can be executed. `replan()` is a placeholder.
    - `GOAPPlanner`:
        - `goap_planner()`: Static method. Performs state-space search to find an action sequence (names) to satisfy a goal's completion conditions by simulating action effects.
        - `evaluate_goal_importance()`: Selects which goal to pursue based on character and world state.
        - `calculate_goal_difficulty()`: Assesses goal difficulty (delegates to `GraphManager`).

**Interaction**: Goals are selected, then `GOAPPlanner` finds a plan (sequence of action names). The `Plan` object then executes these actions, with each `Action` object verifying its own preconditions and applying its effects to the game state via `GraphManager`.

## Major Components & Interactions for Gap Identification

- **Integration of LLM**: The `LLM_Interface` is conceptual. The actual implementation of `PromptBuilder`, `TinyBrainIO`, and especially `OutputInterpreter` (parsing LLM responses into concrete `Action` objects) will be critical.
- **`StrategyManager` - `GOAPPlanner` - `ActionSystem` Loop**: This is the core decision-making engine. The seamless flow of goals, plans, and actions is vital.
    - How `StrategyManager` decides *when* to use LLM vs. pure GOAP.
    - How `GOAPPlanner` gets its list of available `Actions` and their current costs/feasibility.
- **`MemoryManager` Influence**: How retrieved memories actively influence `StrategyManager`'s goal formulation, `GOAPPlanner`'s goal evaluation, or `PromptBuilder`'s context. The documents describe storage and retrieval but less about the direct feedback mechanism into decision-making.
- **`GraphManager` as Single Source of Truth**: All components rely on `GraphManager` for world state. Consistency and timely updates after actions are crucial.
- **`EventHandler` and System Responsiveness**: How events propagate and trigger appropriate responses in `StrategyManager` or individual characters.
- **`Plan.replan()`**: Noted as a placeholder. Robust replanning in response to failures or changing environments is a complex but important feature for believable AI.
- **`OutputInterpreter`**: This conceptual component is key. Translating potentially creative/abstract LLM outputs into specific, executable `Action` instances with defined targets and parameters is a major challenge.
- **Action Definition Completeness**: The `ActionSystem` relies on a comprehensive set of actions with well-defined preconditions and effects. Gaps here would limit character behavior.
- **Utility & Goal Evaluation Logic**: The effectiveness of `tiny_utility_functions.py` and `GOAPPlanner.evaluate_goal_importance()` in guiding characters towards believable and engaging behavior.
- **State Representation**: Ensuring the `State` object in `actions.py` and the character/world state in `GraphManager` are always synchronized and provide sufficient detail for condition checking and effect application.
