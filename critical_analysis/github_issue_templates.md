# GitHub Issue Templates for TinyVillage Development

This file contains detailed templates for submitting GitHub issues based on the codebase analysis.

---

## Issue 1: Implement `tiny_output_interpreter.py`

**Title:** Core: Implement `tiny_output_interpreter.py` for LLM Action Translation

**Description:**
The `tiny_output_interpreter.py` file is currently empty. This component is critical for enabling the Large Language Model (LLM) to have a functional role in character decision-making. It's responsible for translating the LLM's natural language responses into concrete, executable game actions or commands that the game engine can understand and process. Without this, the LLM interaction loop is incomplete.

**Suggested Next Steps / Implementation Details:**
1.  **Define LLM Response Contract:** Establish a clear and consistent format for how the LLM should structure its responses when choosing actions or providing parameters. Examples:
    *   Keyword-based: "ACTION: Eat ITEM: Apple"
    *   JSON-like string: `{"action": "Talk", "target": "John", "topic": "weather"}`
    *   Function-call like: `choose_action(action_name="EatFood", item="Apple")`
2.  **Develop Parsing Logic:** Implement robust parsing functions within `OutputInterpreter` to handle the defined LLM response formats. This should include error handling for malformed or unexpected LLM outputs.
3.  **Action Mapping:** Create a mechanism to map parsed LLM responses to:
    *   Specific `Action` classes/templates defined in `actions.py`.
    *   Instantiation of these `Action` objects with appropriate parameters (e.g., target entities, items involved) derived from the LLM response and current game context.
4.  **Parameter Resolution:** If the LLM provides general parameters (e.g., "eat some food"), the interpreter might need to query `GraphManager` or the character's state to resolve specifics (e.g., select a food item from inventory).
5.  **Integration:**
    *   Connect `TinyBrainIO` to pass LLM responses to the `OutputInterpreter`.
    *   The `OutputInterpreter` should then provide the executable `Action` object(s) to the `GameplayController` or the relevant `Character` object for execution.
6.  **Testing:** Create test cases with various LLM response examples (valid and invalid) to ensure correct parsing and action instantiation.

**Labels:**
`enhancement`, `core-system`, `llm-interface`, `critical`

**Acceptance Criteria (Optional):**
*   LLM can choose a simple action (e.g., "go to park") and the character executes it.
*   LLM can choose an action with a target (e.g., "talk to Emma") and the character executes it.
*   Interpreter handles basic variations or minor errors in LLM response format gracefully.

---

## Issue 2: Flesh out `StrategyManager.get_daily_actions()`

**Title:** AI: Implement Dynamic Action Generation in `StrategyManager.get_daily_actions()`

**Description:**
The `StrategyManager.get_daily_actions()` method currently returns a static, placeholder list of actions. For characters to exhibit intelligent and contextually relevant behavior, the actions available to them must be dynamically determined based on their current situation, environment, and internal state.

**Suggested Next Steps / Implementation Details:**
1.  **Contextual Analysis:**
    *   Modify `get_daily_actions()` to take the `Character` object or `CharacterID` as input.
    *   Query `GraphManager` to retrieve relevant contextual information for the character:
        *   Current location and available interactions/items/services at that location.
        *   Nearby characters and their relationship status.
        *   Character's current inventory.
        *   Character's active goals and needs (from `Character.motives` or `NeedsPriorities`).
        *   Time of day, weather, active events.
2.  **Action Filtering/Generation:**
    *   Define a comprehensive global list of `ActionTemplate`s available in the game.
    *   Filter these templates based on the retrieved context. An action is available if:
        *   Its general preconditions (e.g., required location type, item type in inventory) seem satisfiable in the current context.
        *   Character possesses necessary skills (if applicable).
    *   Alternatively, or in combination, develop context-specific `ActionGenerator` logic that directly produces relevant `Action` instances.
3.  **Parameterization (Partial):** For generated actions, partially fill in parameters where obvious from context (e.g., if at a "Cafe", an "OrderDrink" action might have the Cafe as the implicit target for some effects). Full parameterization might occur during LLM choice or final execution.
4.  **Integration with `PromptBuilder`:** Ensure the dynamically generated list of relevant actions (or their descriptions) can be effectively used by `PromptBuilder` to present choices to the LLM.

**Labels:**
`enhancement`, `ai-behavior`, `strategy-manager`, `graph-manager`

**Acceptance Criteria (Optional):**
*   If a character is at a "Library", "ReadBook" is a suggested action, but "GoFishing" is not (unless there's a pond in the library).
*   If a character has "Food" in inventory, "EatFood" is an option.
*   If a character is near a "Friend", "TalkToFriend" is an option.

---

## Issue 3: Enhance GOAP Replanning Capabilities

**Title:** AI: Implement Robust Replanning in `Plan.replan()`

**Description:**
The `Plan.replan()` method in `tiny_goap_system.py` is a placeholder. GOAP plans are generated based on a snapshot of the world state. If the world changes significantly during plan execution or an action fails unexpectedly, the existing plan can become invalid or suboptimal. Robust replanning is essential for adaptive character behavior.

**Suggested Next Steps / Implementation Details:**
1.  **Plan Monitoring:**
    *   Within `Plan.execute()`, before executing each action, re-validate its preconditions against the current (potentially changed) world state from `GraphManager`.
    *   Optionally, periodically check if the overall goal of the plan is still relevant or if a much better way to achieve it has emerged.
2.  **Replanning Triggers:**
    *   If an action's preconditions are no longer met.
    *   If an action execution fails.
    *   If a significantly higher priority goal emerges.
    *   (Advanced) If a more efficient path to the current goal is detected.
3.  **Replanning Logic in `Plan.replan()`:**
    *   When triggered, `replan()` should typically invoke `GOAPPlanner.goap_planner()` again with the character's *current* state and the original (or a potentially revised) goal.
    *   The list of available actions might also need to be updated.
4.  **Partial Replanning (Advanced):**
    *   Instead of discarding the entire plan, investigate techniques to repair or replan only the necessary portions. This can be more efficient. For example, if only a sub-goal action fails, try to find an alternative for that sub-goal.
5.  **State Management:** Ensure that the state passed to the replanner is the most up-to-date.
6.  **Avoid Thrashing:** Implement mechanisms to prevent characters from getting stuck in replanning loops (e.g., by adding a cost to replanning or temporarily deprioritizing a goal if it causes frequent replans).

**Labels:**
`enhancement`, `ai-behavior`, `goap`, `planning`

**Acceptance Criteria (Optional):**
*   If an item required for a planned action is lost, the character attempts to replan to reacquire the item or find an alternative action.
*   If a character's path to a location is blocked, they replan to find a new path.

---

## Issue 4: Performance Profiling and Optimization

**Title:** Performance: Conduct Profiling and Optimization for Core Systems

**Description:**
Several core components of TinyVillage are potentially computationally intensive, which could lead to performance bottlenecks as the game scales in complexity (more characters, memories, interactions, larger graph). These include `GraphManager` queries and updates, the `MemoryManager` NLP pipeline, and `GOAPPlanner` state-space search.

**Suggested Next Steps / Implementation Details:**
1.  **Profiling:**
    *   Integrate Python profiling tools (e.g., `cProfile`, `line_profiler`, `memory_profiler`).
    *   Develop representative test scenarios (e.g., a busy day with many characters interacting, extensive memory recall, complex GOAP planning).
    *   Run profiling under these scenarios to identify:
        *   CPU hotspots (functions consuming the most time).
        *   Memory bottlenecks (high memory usage, memory leaks).
2.  **Optimization - `GraphManager`:**
    *   Analyze common query patterns and optimize them (e.g., caching results of expensive queries if data doesn't change frequently).
    *   Consider if more specialized graph data structures or indexing within `networkx` (or alternative graph libraries if absolutely necessary) could benefit critical queries.
    *   Batch graph updates where possible.
3.  **Optimization - `MemoryManager`:**
    *   **NLP Pipeline:**
        *   Evaluate the speed of embedding models. Consider smaller/faster alternatives if feasible for the required accuracy.
        *   Optimize keyword extraction and fact extraction logic.
        *   Cache intermediate NLP results (e.g., tokenized text, POS tags) if they are reused.
    *   **FAISS:** Ensure FAISS indices are appropriately chosen and configured for the data size and query types (e.g., `IndexFlatL2` vs. HNSW for approximate search).
4.  **Optimization - `GOAPPlanner`:**
    *   If `goap_planner` search becomes too slow, implement or improve heuristics for A*.
    *   Consider hierarchical GOAP (planning at different levels of abstraction) to reduce search space.
    *   Limit plan depth or search time if necessary for real-time constraints.
5.  **General Python Optimizations:** Look for standard Python bottlenecks (e.g., inefficient loops, excessive object creation).

**Labels:**
`performance`, `core-system`, `graph-manager`, `memory-manager`, `goap`

**Acceptance Criteria (Optional):**
*   Identify top 3-5 performance bottlenecks.
*   Achieve measurable improvement (e.g., X% reduction in processing time for a specific scenario) for at least one major bottleneck.

---

## Issue 5: Develop `tiny_utility_functions.py`

**Title:** AI: Implement Utility Functions in `tiny_utility_functions.py`

**Description:**
The `StrategyManager` and `GOAPPlanner` reference the concept of utility evaluation (e.g., `evaluate_utility()`, action sorting by utility) for decision-making. However, the actual utility calculation functions and their definitions seem to be missing or are placeholders (likely intended for `tiny_utility_functions.py`). These functions are crucial for characters to rank the desirability of different actions, plans, or states.

**Suggested Next Steps / Implementation Details:**
1.  **Define Utility Metrics:**
    *   Determine what factors contribute to "utility" for a character. This will likely be a weighted sum of various considerations, such as:
        *   Predicted change in needs (e.g., hunger reduction, energy increase).
        *   Progress towards active goals.
        *   Predicted emotional outcome (e.g., happiness increase).
        *   Resource costs (time, money, energy).
        *   Alignment with personality traits and motives.
        *   Social consequences (impact on relationships).
2.  **Implement Core Utility Functions in `tiny_utility_functions.py`:**
    *   `calculate_action_utility(character_state, action, current_goal)`: Estimates the utility of performing a single action.
    *   `calculate_plan_utility(character_state, plan)`: Estimates the overall utility of executing a sequence of actions.
    *   `calculate_state_utility(character_state)`: (Optional) Estimates the general desirability of a given character state.
3.  **Weighting and Balancing:** Design a system for weighting different utility factors. These weights might be character-specific (e.g., a materialistic character values wealth gain more) or context-dependent.
4.  **Integration:**
    *   Modify `StrategyManager` to use these utility functions when selecting strategies or presenting options.
    *   Modify `GOAPPlanner` to use these functions if it needs to choose between multiple valid plans or to guide its search heuristic.
    *   The `PromptBuilder` might also use utility scores to highlight more "sensible" options to the LLM.

**Labels:**
`enhancement`, `ai-behavior`, `decision-making`, `core-system`

**Acceptance Criteria (Optional):**
*   A basic `calculate_action_utility` function is implemented that considers at least need fulfillment and resource cost.
*   `StrategyManager` uses this function to sort or select potential actions.

---

## Issue 6: Refine and Test `GraphManager.calculate_goal_difficulty()`

**Title:** Core: Refine and Thoroughly Test `GraphManager.calculate_goal_difficulty()`

**Description:**
The `GraphManager.calculate_goal_difficulty()` function is highly complex and central to the GOAP planner's ability to find efficient plans. It involves interpreting goal criteria, finding relevant nodes, evaluating action viability, and potentially complex pathfinding or combinatorial analysis. Ensuring its accuracy, robustness, and performance is critical. The current snippet shows a sophisticated approach that needs careful validation.

**Suggested Next Steps / Implementation Details:**
1.  **Code Review & Simplification:**
    *   Review the existing logic for clarity and correctness.
    *   Identify any overly complex sections that could be simplified or refactored.
    *   Ensure clear handling of how different costs (edge traversal, action execution, effects on goal progress) are aggregated.
2.  **Test Case Development:** Create a comprehensive suite of test cases covering:
    *   Simple goals with direct paths/actions.
    *   Goals requiring multi-step action sequences.
    *   Goals with complex criteria involving multiple node types or attributes.
    *   Scenarios where no solution exists or is very costly.
    *   Goals where different characters might have different difficulty scores due to skills or resources.
3.  **Validate `action_viability_cost` Interaction:** Ensure that the costs and viability information derived from `calculate_action_viability_cost` are correctly used within `calculate_goal_difficulty`.
4.  **Pathfinding and Combination Logic:**
    *   If the "Greedy initial solution" vs. "A* Search logic" paths are both intended, clarify their conditions and test them independently.
    *   The use of `ProcessPoolExecutor` for `evaluate_combination` suggests parallel processing; ensure this is robust and doesn't introduce race conditions or excessive overhead for simpler goals.
5.  **Heuristic Validation (if A* is used):** If a heuristic is used in the A* search part, ensure it is admissible and consistent to guarantee optimality (or understand the trade-offs if it's not).
6.  **Performance Benchmarking:** Test the function's performance with graphs and goals of varying complexity.

**Labels:**
`bug`, `core-system`, `graph-manager`, `goap`, `testing`

**Acceptance Criteria (Optional):**
*   The function returns reasonable difficulty scores for a defined set of benchmark goal scenarios.
*   Identified edge cases (e.g., impossible goals) are handled gracefully (e.g., returning `float('inf')`).
*   The logic for choosing between different solution paths (e.g., shortest vs. lowest goal cost) is clear and intended.

---

## Issue 7: Memory Coherence and Belief Revision

**Title:** AI: Design and Implement Basic Memory Coherence / Belief Revision

**Description:**
The `MemoryManager` currently extracts facts from memories. However, as characters accumulate more memories, they may encounter information that is contradictory, outdated, or of varying reliability. The current system does not explicitly address how characters manage these inconsistencies or update their "beliefs" based on new evidence.

**Suggested Next Steps / Implementation Details (Advanced/Long-term):**
1.  **Confidence Scoring for Facts:**
    *   When extracting facts, assign a confidence score based on factors like:
        *   Source of information (e.g., direct experience vs. hearsay).
        *   Clarity or ambiguity of the memory text.
        *   Recency of the memory.
2.  **Contradiction Detection:**
    *   Implement mechanisms to detect direct contradictions between newly acquired facts and existing highly-weighted facts in memory. This could involve:
        *   Comparing embeddings of facts.
        *   Logical comparison if facts are structured (e.g., SVO triples).
3.  **Belief Update Strategy:**
    *   **Recency Bias:** Newer information might be weighted more heavily.
    *   **Confidence-Weighted Endorsement:** When retrieving information for decision-making, aggregate support for a proposition, weighting by fact confidence. The proposition with the highest weighted support is considered the current "belief."
    *   **Explicit Invalidation (Simple):** If a new, very high-confidence memory directly contradicts an old one, the old one could be flagged as "outdated" or have its importance/confidence significantly reduced.
4.  **Source Reliability (Very Advanced):** If characters can learn who told them what, they could develop trust scores for information sources, influencing fact confidence.
5.  **Integration with Decision Making:** Ensure that decision-making processes (GOAP, LLM prompts) can query or be influenced by these refined beliefs rather than just raw, potentially contradictory memories.

**Labels:**
`enhancement`, `ai-behavior`, `memory-system`, `long-term`

**Acceptance Criteria (Optional):**
*   A basic confidence score is associated with extracted facts.
*   A simple mechanism for preferring newer or higher-confidence facts in cases of direct contradiction is implemented.

---

## Issue 8: Action and Goal Definition Expansion

**Title:** Content: Expand Set of Actions and Goals

**Description:**
The richness and depth of the TinyVillage simulation are directly tied to the variety and complexity of actions characters can perform and goals they can pursue. The current set of actions and goals, while foundational, will need significant expansion to create a truly engaging and emergent world.

**Suggested Next Steps / Implementation Details (Ongoing Task):**
1.  **Identify Key Gameplay Loops:** Determine the core activities and progression paths for characters (e.g., career advancement, relationship building, skill development, exploration, problem-solving).
2.  **Brainstorm Actions & Goals:** For each gameplay loop, brainstorm a comprehensive list of:
    *   **Actions:** Individual steps characters can take.
    *   **Goals:** Short-term and long-term objectives characters might have.
3.  **Define Action Details (for `actions.py` and `ActionTemplate`s):**
    *   Clear `name`.
    *   Specific `preconditions` (what character/world state is required).
    *   Detailed `effects` (how character/world state changes).
    *   Appropriate `cost` (time, energy, resources).
    *   Relevant `related_skills`.
    *   Impact ratings.
4.  **Define Goal Details (for `Goal` class in `tiny_characters.py`):**
    *   Clear `name` and `description`.
    *   `completion_conditions` (how to know the goal is met).
    *   Logic for `evaluate_utility_function` (how important is this goal *now*?).
    *   Logic for `difficulty` (how hard is it to achieve *now*? often links to `GraphManager.calculate_goal_difficulty`).
    *   `completion_reward` and `failure_penalty`.
    *   `criteria` (nodes/edges in `GraphManager` relevant to this goal).
5.  **Categorization:** Organize new actions and goals into logical categories for easier management and potential use in contextual action generation.
6.  **Iterative Implementation:** Add actions and goals incrementally, testing their impact on character behavior and game balance.

**Labels:**
`content`, `enhancement`, `gameplay`, `ai-behavior`

**Acceptance Criteria (Optional):**
*   N new actions related to a specific gameplay loop (e.g., "Job Performance") are implemented and tested.
*   M new goals related to a character aspiration (e.g., "Become Master Chef") are implemented and characters can pursue them.

---

## Issue 9: Testing and Validation of AI Behavior

**Title:** Testing: Implement Comprehensive Testing and Validation for AI Systems

**Description:**
The interaction of multiple complex AI systems (GOAP, LLM, Memory, Graph-based reasoning) can lead to unpredictable, unintended, or buggy emergent behaviors. A robust testing strategy is needed to ensure characters act coherently, make reasonable decisions, and that the underlying AI systems function as expected.

**Suggested Next Steps / Implementation Details:**
1.  **Unit Tests for AI Components:**
    *   **Actions:** Test `preconditions_met()` and `apply_effects()` for various actions with mock states.
    *   **Conditions:** Test `check_condition()` logic.
    *   **MemoryManager:** Test NLP functions (keyword extraction, sentiment, fact extraction) with sample texts. Test memory storage and retrieval logic.
    *   **GOAPPlanner:** Test `evaluate_goal_importance` with different character states/motives. Test the core `goap_planner` search with small, controlled sets of actions and states.
    *   **GraphManager:** Test specific query functions and relationship update logic.
2.  **Integration Tests for Decision Loops:**
    *   Create test scenarios that trigger a full decision cycle (e.g., from event to action execution).
    *   Verify that characters select appropriate goals and plans based on the scenario.
    *   If using LLM, mock LLM responses to test the `OutputInterpreter` (once implemented) and downstream effects.
3.  **Scenario-Based Validation:**
    *   Design specific game scenarios (e.g., "character is hungry and has money", "character is lonely and an event is happening", "character faces a threat").
    *   Observe or log character behavior in these scenarios to ensure it's plausible and aligns with their personality, motives, and goals.
    *   This might involve creating tools to set up specific game states easily.
4.  **Metrics and Logging:**
    *   Implement logging for key decision points, chosen actions, goal statuses, and character need levels.
    *   Consider defining metrics to track overall AI "health" (e.g., average character happiness, goal completion rates, instances of idle or repetitive behavior).
5.  **AI "Debugging" Tools:**
    *   (Advanced) Consider tools to visualize a character's current goals, active plan, key memories, or the reasons behind a specific decision. This can be invaluable for understanding and debugging complex AI.

**Labels:**
`testing`, `ai-behavior`, `core-system`, `bug-prevention`

**Acceptance Criteria (Optional):**
*   Unit tests cover X% of critical functions in AI modules.
*   A set of Y integration scenarios pass, demonstrating correct end-to-end decision-making for common situations.
*   Basic logging of character decisions and goal progress is implemented.

---
