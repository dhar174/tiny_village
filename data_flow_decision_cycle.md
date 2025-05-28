# Data Flow Diagram - Character Decision Cycle ("New Day" Event)

This diagram illustrates the data flow when a character makes decisions, typically triggered by an event like "New Day".

1.  **Event Trigger:**
    *   **Source:** `EventHandler`
    *   **Action:** Detects/generates a "New Day" event.
    *   **Data:** `Event` object (e.g., `{type: "new_day", date: "YYYY-MM-DD"}`)
    *   **Destination:** `MainApplication`

2.  **Strategy Initiation:**
    *   **Source:** `MainApplication`
    *   **Action:** Receives event, triggers character planning.
    *   **Data:** Character ID, `Event` object.
    *   **Destination:** `StrategyManager.update_strategy()`

3.  **Character State & Context Gathering:**
    *   **Source:** `StrategyManager`
    *   **Data Request 1:** Character's current state.
    *   **Destination Request 1:** `GraphManager.get_character_state()`
    *   **Data Response 1:** `CharacterState` (dictionary of attributes).
    *   **Source Response 1:** `GraphManager`.
    *   **Data Request 2:** Available actions.
    *   **Destination Request 2:** `StrategyManager.get_daily_actions()` (ideally queries `GraphManager` or `ActionSystem`).
    *   **Data Response 2:** List of `Action` templates/instances.
    *   **Source Response 2:** `ActionSystem` / `GraphManager`.

4.  **Goal Evaluation & Prioritization:**
    *   **Source:** `StrategyManager` or `Character`
    *   **Action:** Evaluate importance of active/new goals.
    *   **Data In:** `CharacterState` (needs, motives), `Goal` objects.
    *   **Process:** Calls `GOAPPlanner.evaluate_goal_importance(Character, Goal, GraphManager)`.
    *   **Data Out:** Prioritized `Goal` object(s).
    *   **Destination:** `GOAPPlanner`.

5.  **Action Planning (GOAP):**
    *   **Source:** `StrategyManager` (invoking `GOAPPlanner`)
    *   **Action:** `GOAPPlanner.goap_planner(CharacterState, Goal, Actions)`
    *   **Data In:** `CharacterState`, `Goal`, list of `Action` objects.
    *   **Process:** Searches for action sequence. Checks `Action.preconditions_met()`, simulates `Action.apply_effects()`, uses `GraphManager.calculate_goal_difficulty()`.
    *   **Data Out:** `Plan` object (sequence of `Action` instances).
    *   **Destination:** `StrategyManager`.

6.  **Prompt Generation for LLM:**
    *   **Source:** `StrategyManager` or `Character`
    *   **Action:** `PromptBuilder.generate_daily_routine_prompt()`
    *   **Data In:** Character details, needs priorities, context, action options/plan.
    *   **Data Out:** Formatted prompt string.
    *   **Destination:** `TinyBrainIO.input_to_model()`.

7.  **LLM Interaction:**
    *   **Source:** `TinyBrainIO`
    *   **Data In:** Prompt string.
    *   **Data Out:** LLM response string.
    *   **Destination:** `OutputInterpreter` (Conceptual).

8.  **Output Interpretation (Conceptual):**
    *   **Source:** `OutputInterpreter`
    *   **Action:** Parses LLM response, maps to `Action` object(s).
    *   **Data In:** LLM response string.
    *   **Data Out:** Instantiated `Action` object(s).
    *   **Destination:** `GameplayController`.

9.  **Action Execution:**
    *   **Source:** `GameplayController`
    *   **Action:** `Action.execute(target, initiator)`.
    *   **Data In:** `Action` object, `CharacterState`.
    *   **Process:** Checks preconditions, applies effects.
    *   **Data Change:** Modifies `CharacterState`.
    *   **Graph Update:** Changes reflected in `GraphManager`.
    *   **Destination (of updates):** `GraphManager`, `Character` object.

10. **Memory Update:**
    *   **Source:** `Character`
    *   **Action:** Creates `SpecificMemory`.
    *   **Data In:** Description, timestamp, importance.
    *   **Destination:** `MemoryManager.add_memory()`.
    *   **Process:** NLP processing, embedding, FAISS indexing.

## Mermaid Sequence Diagram Structure

```mermaid
sequenceDiagram
    participant EH as EventHandler
    participant App as MainApplication
    participant SM as StrategyManager
    participant GM as GraphManager
    participant Char as CharacterObject
    participant GOAP as GOAPPlanner
    participant PB as PromptBuilder
    participant TBI as TinyBrainIO
    participant LLM as LLM_External
    participant OI as OutputInterpreter
    participant AS as ActionSystem
    participant MM as MemoryManager

    EH->>App: Event("New Day")
    App->>SM: update_strategy(CharacterID, Event)
    SM->>GM: get_character_state(CharID)
    GM-->>SM: CharacterState
    SM->>GOAP: evaluate_goal_importance(CharState, Goals, GM)
    GOAP-->>SM: PrioritizedGoal
    SM->>GOAP: goap_planner(CharState, PrioritizedGoal, Actions)
    GOAP->>GM: calculate_goal_difficulty(Goal, GM)
    GM-->>GOAP: DifficultyInfo
    GOAP-->>SM: Plan (ActionSequence)
    SM->>PB: generate_routine_prompt(Context, CharState, Plan/Options)
    PB->>Char: GetDetails()
    Char-->>PB: Details
    PB->>TBI: input_to_model(Prompt)
    TBI->>LLM: Prompt
    LLM-->>TBI: LLM_Response_Text
    TBI->>OI: LLM_Response_Text
    OI-->>App: ExecutableAction(s)
    App->>Char: command_to_execute(Action)
    Char->>AS: Action.execute()
    AS->>Char: Update_Internal_State_Attributes
    Char->>GM: update_graph_node_attributes()
    Char->>MM: store_memory(Action_Outcome_Description)
    MM->>MM: NLP_Process_Memory()
    MM->>MM: Index_Memory_Embedding_FAISS() 

end
```
