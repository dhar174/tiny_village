# High-Level Architecture

The TinyVillage system is designed as a modular, agent-based architecture. The primary components and their interactions are as follows:

1.  **`MainApplication` (Conceptual - likely `main.py` + `tiny_gameplay_controller.py`)**
    *   **Responsibilities:** Initializes all systems, runs the main game loop, manages game time.
    *   **Interacts with:** `EventHandler`, `StrategyManager`, `GraphManager`, `MapController`.

2.  **`EventHandler` (`tiny_event_handler.py`)**
    *   **Responsibilities:** Detects and queues game events.
    *   **Interacts with:** `MainApplication`, `GraphManager`.

3.  **`StrategyManager` (`tiny_strategy_manager.py`)**
    *   **Responsibilities:** Orchestrates high-level decision-making for characters.
    *   **Interacts with:** `MainApplication`, `GraphManager`, `GOAPPlanner`, `PromptBuilder` (indirectly).

4.  **`GraphManager` (`tiny_graph_manager.py`)**
    *   **Responsibilities:** Central repository of game world state (entities, relationships). Calculates derived data (motives, relationship strengths).
    *   **Interacts with:** Nearly all other modules.

5.  **`Character` (Object in `tiny_characters.py`)**
    *   **Responsibilities:** Represents an individual AI agent, holding personal state and making decisions.
    *   **Interacts with:** `GraphManager`, `ActionSystem`, `MemoryManager`, `GOAPPlanner`, `PromptBuilder`.

6.  **`ActionSystem` (`actions.py`)**
    *   **Responsibilities:** Defines rules for actions (preconditions, effects, costs).
    *   **Interacts with:** `Character` objects, `GOAPPlanner`, `GraphManager`.

7.  **`GOAPPlanner` (`tiny_goap_system.py`)**
    *   **Responsibilities:** Generates sequences of actions (plans) to achieve character goals.
    *   **Interacts with:** `StrategyManager`, `GraphManager`, `Character` objects, `ActionSystem`.

8.  **`LLM_Interface` (Conceptual: `PromptBuilder` + `TinyBrainIO` + `OutputInterpreter`)**
    *   `PromptBuilder`: Constructs text prompts for the LLM.
    *   `TinyBrainIO`: Handles communication with the LLM.
    *   `OutputInterpreter` (Conceptual): Parses LLM responses into game actions.
    *   **Interacts with:** `Character`, `GraphManager`, `LLM`.

9.  **`MemoryManager` (`tiny_memories.py`)**
    *   **Responsibilities:** Manages character memories, including NLP processing and retrieval.
    *   **Interacts with:** `Character` objects, `GraphManager`.

10. **`MapController` (`tiny_map_controller.py`)**
    *   **Responsibilities:** Manages visual representation and map-based interactions.
    *   **Interacts with:** `MainApplication`, `Character` objects.

## Mermaid Diagram Structure

```mermaid
graph TD
    A[MainApplication] -->|Triggers| B(EventHandler);
    B -->|Events| A;
    A -->|Updates Strategy| C(StrategyManager);
    C -->|Gets State/Actions| D(GraphManager);
    C -->|Requests Plan| E(GOAPPlanner);
    E -->|World State, Action Costs| D;
    C -->|Needs LLM Decision| F(PromptBuilder);
    F -->|Character/World Context| D;
    F -->|Character State| G(Character);
    F -->|Sends Prompt| H(TinyBrainIO);
    H -->|LLM Interaction| I((LLM));
    I -->|LLM Text Response| J(OutputInterpreter);
    J -->|Executable Actions| A;
    A -->|Executes Action via| K(ActionSystem);
    G -->|Executes Action via| K;
    K -->|Updates State| D;
    G -->|Updates Own State in| D;
    G -->|Records Memory| L(MemoryManager);
    D -->|Provides Data for Memory| L;
    A -->|Render Updates| M(MapController);
    G -->|Location/Movement| M;

    subgraph LLM_Interface
        F
        H
        J
    end
end
```
