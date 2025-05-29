# Component Deep Dive: Action System (`actions.py` & `tiny_goap_system.py`)

The Action System, primarily defined in `actions.py` and utilized by `tiny_goap_system.py`, governs how characters can interact with the world and how they plan to achieve their objectives.

## `actions.py`: Defining Action Primitives

*   **`State` Class:**
    *   **Purpose:** Represents the current state of game entities (characters, objects). Allows attribute access and can call methods of the underlying entity.
    *   **Key Feature:** `compare_to_condition()` for checking if a state attribute meets a condition.

*   **`Condition` Class:**
    *   **Purpose:** Defines criteria for actions (preconditions) or goal states.
    *   **Attributes:** `name`, `attribute` (to check in `State`), `target` (entity), `satisfy_value`, `op` (comparison operator), `weight`.
    *   **Key Feature:** `check_condition(state)` evaluates the condition.

*   **`Action` Class:**
    *   **Purpose:** Represents a single, executable game action.
    *   **Attributes:** `name`, `preconditions` (dictionary of `Condition` objects), `effects` (list of state-change dictionaries), `cost`, `target`, `initiator`, `related_skills`, `impact_ratings`.
    *   **Key Functionality:**
        *   `preconditions_met()`: Checks if all preconditions are satisfied.
        *   `apply_effects()`: Modifies a `State` based on defined effects. Can change attributes or invoke methods.
        *   `execute()`: Main execution logic. Checks preconditions, applies effects, and includes integration points for updating `GraphManager` (e.g., relationship changes).

*   **`ActionTemplate`, `ActionGenerator`:**
    *   **Purpose:** Allow for defining generic action blueprints (`ActionTemplate`) that can be instantiated by an `ActionGenerator`, promoting extensibility.

*   **`Skill`, `JobSkill`, `ActionSkill`:** Classes for representing character skills.

*   **`ActionSystem` Class:**
    *   **Purpose:** Manages action definitions and provides a simplified execution path.
    *   **Key Functionality:** `setup_actions()` (defines templates), `generate_actions()`, `execute_action()` (likely used by GOAP for internal state simulation during planning).

## `tiny_goap_system.py`: Planning Action Sequences

*   **`Plan` Class:**
    *   **Purpose:** Represents a sequence of actions to achieve goals.
    *   **Attributes:** `name`, `goals` (list of `Goal` objects), `action_queue` (priority queue of actions with dependencies), `completed_actions`.
    *   **Key Functionality:** `add_goal()`, `add_action()`, `execute()` (iterates through goals, executes actions from queue if dependencies and preconditions met). `replan()` is a placeholder.

*   **`GOAPPlanner` Class:**
    *   **Purpose:** Core Goal-Oriented Action Planning engine.
    *   **Key Functionality:**
        *   **`goap_planner(character, goal, char_state, actions)` (Static Method):** The main planning algorithm. It performs a state-space search (similar to A* or BFS) to find a sequence of action names leading from the `char_state` to a state satisfying the `goal`'s completion conditions. It simulates action effects to explore states.
        *   **`evaluate_goal_importance(character, goal, graph_manager, **kwargs)`:** Crucial for selecting which goal to pursue. It assesses a goal's relevance based on the character's needs, motives, personality, and the current world state (obtained from `GraphManager`).
        *   **`calculate_goal_difficulty(character, goal)`:** Delegates to `GraphManager` to determine how hard it is to achieve a goal, factoring in resource availability, pathfinding, etc.

## Interaction and Data Flow

1.  **Goal Selection:** A `Character` (often guided by `StrategyManager`) uses `GOAPPlanner.evaluate_goal_importance` to pick a high-priority `Goal`. This evaluation uses data from the `Character` object and `GraphManager`.
2.  **Plan Generation:** `GOAPPlanner.goap_planner` is invoked with the character's current `State`, the chosen `Goal`, and available `Actions`. It searches for an optimal sequence of action *names*.
3.  **Plan Execution:** The action names are typically used to populate a `Plan` object. `Plan.execute()` then iterates through this sequence. For each `Action` in the sequence:
    *   The actual `Action` object is retrieved/instantiated.
    *   `Action.execute()` is called.
        *   This first checks `preconditions_met()` against the current game state (from `GraphManager`).
        *   If successful, `apply_effects()` updates the `State` of involved entities. These state changes are then propagated to the `GraphManager` to reflect the action's impact on the world.

## Strengths

*   **Purposeful Behavior:** GOAP enables characters to act logically towards achieving their goals.
*   **Decoupling:** Action definitions (`actions.py`) are separate from planning logic (`tiny_goap_system.py`).
*   **Contextual Goal Prioritization:** `evaluate_goal_importance` allows for dynamic and nuanced goal selection.

## Potential Challenges

*   **Planner Performance:** GOAP's search can be computationally intensive.
*   **Plan Robustness:** The current replanning capability is basic; robust handling of failures or environmental changes is complex.
*   **Defining Actions/Goals:** Crafting comprehensive and balanced actions, preconditions, effects, and goal conditions requires significant design effort.

The Action System provides a strong framework for character agency and intelligent behavior within TinyVillage, driven by goals and constrained by the rules of the game world.
