## `tiny_characters.py` Analysis

**Character Attributes:**
A `Character` is defined by a multitude of attributes, including:
-   **Basic Info**: `name`, `age`, `pronouns`.
-   **Core Stats**: `health_status`, `hunger_level`, `wealth_money` (bank account implied), `mental_health`, `social_wellbeing`, `energy`.
-   **Job & Performance**: `job` (can be a `JobRoles` object or string), `job_performance`.
-   **Social**: `community` score, `friendship_grid` (list of dicts, seems to represent relationships), `romantic_relationships`, `romanceable`, `exclusive_relationship`, `base_libido`, `monogamy`.
-   **Goals & History**: `recent_event`, `long_term_goal`, `career_goals`, `short_term_goals`, `goals` (list of `Goal` objects).
-   **Location & Home**: `home` (a `House` object), `location` (a `Location` object), `coordinates_location`, `destination`, `path`, `move_speed`.
-   **Inventory & Possessions**: `inventory` (an `ItemInventory` object), `investment_portfolio` (list of `Stock` objects), `material_goods`.
-   **Personality & Motives**: `personality_traits` (a `PersonalityTraits` object: openness, conscientiousness, extraversion, agreeableness, neuroticism), `motives` (a `PersonalMotives` object, which is a collection of `Motive` objects like hunger, wealth, health, etc.).
-   **Skills**: `skills` (a `CharacterSkills` object).
-   **State & Activity**: `current_satisfaction`, `current_mood`, `current_activity`, `stamina`.
-   **System Identifiers**: `uuid`.
-   **Other**: `possible_interactions` (list of `Action` objects), `needed_items`.
-   **Calculated Attributes**: `happiness`, `stability`, `luxury`, `hope`, `success`, `control`, `beauty` (these are calculated based on other attributes).

**Implemented Methods for Behavior/State Changes:**
-   **Initialization (`__init__`)**: Complex, sets up many attributes, calculates some derived ones, and can add the character to the `GraphManager`.
-   **Setters/Getters**: Numerous for most attributes (e.g., `set_name`, `get_age`).
-   **Calculations**: `calculate_motives`, `calculate_happiness`, `calculate_stability`, `calculate_success`, `calculate_control`, `calculate_hope`, `calculate_base_libido`, `calculate_monogamy`, `calculate_material_goods`.
-   **Goal Management**: `generate_goals` (uses `GoalGenerator`), `evaluate_goals` (sorts goals by utility from `GOAPPlanner`), `add_new_goal`.
-   **Memory**: `create_memory` (adds to `MemoryManager`), `recall_recent_memories`, `make_decision_based_on_memories` (rudimentary influence), `retrieve_specific_memories`, `update_memory_importance`.
-   **Movement**: `set_destination` (uses `a_star_search`), `move_towards_next_waypoint` (uses `SteeringBehaviors`).
-   **Interaction/Animation**: `play_animation` (placeholder).
-   **State Representation**: `get_state` (returns a `State` object, likely for `ActionSystem`).
-   **Descriptive**: `describe` (prints personality traits), `define_descriptors` (creates a rich dictionary of character descriptors).
-   **Decision Helpers (Personality-based)**: `decide_to_join_event`, `decide_to_explore`, `decide_to_take_challenge`, `respond_to_conflict`.
-   **Update**: `update_character` (bulk attribute updater).

**Completeness of Character Definition:**
-   **History**: `recent_event` is present. A more structured "history" (list of significant past events or memories) is not explicitly a top-level attribute but is managed by `MemoryManager`.
-   **Likes/Dislikes**: Not explicitly defined as attributes. Preferences would likely emerge from personality traits, motives, and possibly learned from memories, but there isn't a direct `likes = []` or `dislikes = []`.
-   **Relationships**: `friendship_grid` and `romantic_relationships` along with `exclusive_relationship` address this. The `generate_friendship_grid` method suggests integration with `GraphManager` for relationship data.
-   **Careers**: `job` attribute (can be a `JobRoles` object from `tiny_jobs.py`), `job_performance`, and `career_goals` are present.
-   **Bank Accounts**: `wealth_money` serves this purpose. `investment_portfolio` for stocks suggests more advanced financial activities.

**Overall**: The character definition is quite comprehensive and detailed, covering most aspects mentioned in the project description. Many attributes are complex objects themselves (e.g., `PersonalityTraits`, `PersonalMotives`, `ItemInventory`). The interaction with other systems like `GraphManager`, `MemoryManager`, `ActionSystem`, and `GOAPPlanner` is evident in the methods.

**Comparison with `documentation_summary.txt`:**
- Aligns well with the description of `Character` as an AI agent holding personal state and making decisions.
- The interaction points with `GraphManager`, `ActionSystem`, `MemoryManager`, `GOAPPlanner`, and `PromptBuilder` are present in the `Character` class methods or through objects it holds (like `MemoryManager` instance).

---
## `tiny_goap_system.py` Analysis

**Current State of GOAP Implementation:**
-   **`Plan` Class**:
    -   Represents a sequence of actions to achieve goals.
    -   Attributes: `name`, `goals` (list of `Goal` objects), `action_queue` (a `heapq` priority queue for actions), `completed_actions`.
    -   Methods: `add_goal`, `add_action` (adds to priority queue), `evaluate` (checks if all goals are complete - seems simplistic as it doesn't check if actions *led* to goal completion), `replan` (basic re-prioritization of actions in queue, doesn't generate new actions), `execute` (iterates goals, executes actions from queue if dependencies and preconditions met), `handle_failure` (calls `replan`, has a placeholder for finding alternative actions).
    -   The `execute` method's logic seems to assume actions directly contribute to goal completion conditions, which might be a simplification.
-   **`GOAPPlanner` Class**:
    -   `__init__`: Takes a `GraphManager`.
    -   `plan_actions(state, actions)`: This method's current implementation is very basic: `return sorted(actions, key=lambda x: -x.utility - goal_difficulty["difficulty"])`. This is NOT a GOAP planning algorithm; it's just sorting pre-defined actions by a utility score. The parameters `state` and the structure of `actions` it expects are not fully aligned with a typical GOAP planner that searches a state space.
    -   `goap_planner(character, goal: Goal, char_state: State, actions: list)`: This is a static method.
        -   It attempts a state-space search (BFS-like with `open_list.pop(0)`).
        -   It expects `actions` to be a list of dictionaries, each with "conditions_met" (a callable) and "effects" (a dict for state changes).
        -   It aims to find a sequence of action *names*.
        -   This method seems more aligned with a classic GOAP planner but is disconnected from the class instance's `graph_manager` and other methods. Its `actions` parameter format differs from the `Action` objects used elsewhere.
    -   `calculate_goal_difficulty(character, goal: Goal)`: Delegates to `GraphManager`.
    -   `evaluate_goal_importance(character: Character, goal: Goal, graph_manager: GraphManager, **kwargs)`: A detailed heuristic method to score goal importance based on character stats, motives, relationships (from `GraphManager`), and goal type. It calls specific helper methods for different goal types (e.g., `calculate_basic_needs_importance`).
    -   Helper methods for calculating importance for different goal types (e.g., `calculate_basic_needs_importance`, `calculate_social_goal_importance`).
    -   `calculate_utility(action, character)`: Calculates utility for a single action based on satisfaction, energy cost, urgency, and character state.
    -   `evaluate_utility(plan, character)`: Finds the action with the highest utility in a given plan (list of actions).
    -   `evaluate_feasibility_of_goal(goal, state)`: Checks if goal completion conditions are met in the given state.

**Actions, Goals, and Planner Definition:**
-   **Goals**: The `Goal` class is defined in `tiny_characters.py`. `GOAPPlanner` uses these `Goal` objects, especially in `evaluate_goal_importance`.
-   **Actions**: The primary `Action` class is defined in `actions.py`.
    -   The `Plan` class seems to work with these `Action` objects.
    -   However, the static `goap_planner` method expects actions as dictionaries with specific keys.
    -   The `StrategyManager` also defines its own simplified `Action` subclasses (EatAction, SleepAction, etc.). This indicates a potential inconsistency or multiple ways actions are represented/handled.
-   **Planner**:
    -   The static `GOAPPlanner.goap_planner` method is a search-based planner.
    -   The instance method `GOAPPlanner.plan_actions` is currently just a sort, not a planner.
    -   The `Plan.execute` method drives the execution of a pre-determined sequence of actions.

**Integration with Characters and Decision-Making:**
-   `evaluate_goal_importance` is heavily reliant on the `Character` object's attributes and `GraphManager` for contextual data.
-   The `StrategyManager` is intended to use `GOAPPlanner` to formulate plans (as per `strategy_management_architecture.md`). The `Character` class has a `goap_planner` attribute (an instance of `GOAPPlanner`) and methods like `evaluate_goals` that use it.
-   The actual planning call from `StrategyManager` to `GOAPPlanner` (and which `plan_actions` or `goap_planner` method gets used) is not entirely clear from this file alone and seems to be a point of confusion or incomplete integration. The `StrategyManager`'s `plan_daily_activities` calls `self.goap_planner(character, goal, current_state, actions)`, which doesn't match the signature of the static `goap_planner` method or the instance method `plan_actions`.

**Comparison with `documentation_summary.txt`:**
-   The documentation states `GOAPPlanner` "Generates sequences of actions (plans) to achieve character goals."
    -   The static `goap_planner` method aligns with this.
    -   The `Plan` class is designed to hold and execute these sequences.
-   It interacts with `StrategyManager`, `GraphManager`, `Character` objects, and `ActionSystem`. This is partially true:
    -   It uses `GraphManager` (passed in constructor).
    -   It uses `Character` and `Goal` objects extensively in `evaluate_goal_importance`.
    -   Its direct interaction with `ActionSystem` is less clear; `Plan.execute()` calls `action.execute()`, which implies `Action` objects from `actions.py`.
-   The `Plan.replan()` is noted as a placeholder in the documentation, and its current implementation is basic (re-sorting).
-   The discrepancy between the static `goap_planner` and the instance method `plan_actions` suggests an ongoing or incomplete implementation of the planning core.

---
## `tiny_memories.py` Analysis

**Memory Storage and Retrieval:**
-   **Storage**:
    -   `Memory` (base class): Basic description, creation/access times.
    -   `GeneralMemory(Memory)`: Represents broad categories. Stores an embedding of its description. It *used* to have a FAISS index and `MemoryBST` instances for its specific memories, but the provided code for `GeneralMemory` shows these attributes (faiss_index, timestamp_tree, importance_tree, key_tree) are initialized but not clearly populated or used in the `add_specific_memory` method for local indexing within the `GeneralMemory` object itself. Instead, `add_specific_memory` seems to rely on the global `MemoryManager.flat_access.faiss_index`.
    -   `SpecificMemory(Memory)`: Represents individual experiences. Stores description, parent_memory link, embedding, keywords, tags, importance, sentiment, emotion, extracted facts, SVO. Undergoes NLP analysis via `analyze_description` (called by `MemoryManager`).
    -   `MemoryBST`: Defined with AVL tree logic (rotations, balancing) but its instances in `GeneralMemory` (`timestamp_tree`, `importance_tree`, `key_tree`) are not clearly used for inserting/retrieving specific memories *within* a `GeneralMemory` object. The `add_specific_memory` method in `GeneralMemory` primarily focuses on adding to these BSTs.
    -   `FlatMemoryAccess`: This class, managed by `MemoryManager`, holds a global FAISS index (`self.faiss_index`) for *all* `SpecificMemory` embeddings and their fact embeddings. It uses `index_id_to_node_id` for mapping. This seems to be the primary mechanism for large-scale indexing.
-   **Retrieval**:
    -   `MemoryManager.retrieve_memories(query)`: Main entry point.
        -   Analyzes the query using `analyze_query_context`.
        -   Calls `FlatMemoryAccess.find_memories_by_query()` which searches the global FAISS index using the query embedding.
    -   `GeneralMemory.find_specific_memory(key, tree)` and `search_by_key`: These methods suggest BST-based search within a `GeneralMemory` but seem underutilized in the main retrieval flow described.
    -   `MemoryQuery` class: Encapsulates query text, embedding, tags, and has methods for various filter functions (by tags, time, importance, etc.), but these filter functions don't seem to be directly integrated into the main FAISS-based retrieval flow of `FlatMemoryAccess.find_memories_by_query`.

**Information Stored in Memories:**
-   `SpecificMemory` stores:
    -   Textual `description`.
    -   `parent_memory` (link to `GeneralMemory`).
    -   Sentence `embedding` of the description.
    -   `keywords` (extracted via RAKE, TF-IDF, NER).
    -   `tags` (manual or derived).
    -   `importance_score`.
    -   `sentiment_score` (polarity, subjectivity from TextBlob).
    -   `emotion_classification` (from Hugging Face model).
    -   Extracted `facts` (SVO triples/clauses via spaCy/tsm).
    -   `facts_embeddings`.
    -   `main_subject/verb/object`.
    -   Linguistic features like `temporal_expressions`, `verb_aspects`.

**Alignment with `memory_manager_deep_dive.md`:**
-   **Overall Structure**: Largely aligns. The classes `Memory`, `GeneralMemory`, `SpecificMemory`, `MemoryBST`, `EmbeddingModel`, `SentimentAnalysis`, `FlatMemoryAccess`, `MemoryQuery`, and `MemoryManager` are all present as described.
-   **NLP Pipeline**: The `MemoryManager.analyze_query_context` method implements a rich NLP pipeline very similar to the one described (spaCy, embeddings, sentiment/emotion, keyword, fact extraction, linguistic features).
-   **Memory Retrieval Process**: The described process (query analysis -> `FlatMemoryAccess.find_memories_by_query` -> FAISS search -> mapping to `SpecificMemory`) matches the implementation.
-   **FAISS Indexing**: `FlatMemoryAccess` manages a global FAISS index as described. `GeneralMemory.add_specific_memory` also adds to this global index.
-   **BSTs in GeneralMemory**: The deep dive mentions `MemoryBST` instances in `GeneralMemory` for sorting specific memories by timestamp, importance, etc. The code defines these BSTs and `GeneralMemory.add_specific_memory` *does* insert into them. However, the primary *retrieval* path highlighted in both the deep dive and the code (`MemoryManager.retrieve_memories`) uses the global `FlatMemoryAccess` FAISS index, not these BSTs for primary searching. The BSTs might be intended for more structured, non-semantic filtering within a `GeneralMemory` context after initial retrieval, or this part of the integration is less emphasized in the current retrieval flow.
-   **Potential Challenges**: The identified challenges (complexity, NLP accuracy, fact extraction robustness, computational cost, memory coherence) are relevant to this implementation.

---
## `tiny_prompt_builder.py` Analysis

**Prompt Construction:**
-   The `PromptBuilder` class's main method for construction is `generate_daily_routine_prompt(time, weather)`.
-   It uses an f-string approach, embedding various pieces of information.
-   It leverages the `DescriptorMatrices` class heavily to get varied descriptive phrases for character job, status, and context, making prompts less static.
-   The prompt structure includes system, user, and assistant tags (`<|system|>`, `<|user|>`, `<|assistant|>`).
-   It sets up a scenario where the character needs to choose their next action.

**Inputs Used:**
-   `self.character`: The `Character` object, from which it pulls:
    -   `name`, `job` (and its associated descriptors from `DescriptorMatrices`).
    -   `health_status`, `hunger_level` (values are passed to `DescriptorMatrices` to get phrases).
    -   `recent_event`, `wealth_money` (values passed to `DescriptorMatrices`).
    -   `long_term_goal`.
-   `time`: Current game time (presumably a string like "morning", "afternoon").
-   `weather`: Current weather conditions (a string to be used with `DescriptorMatrices`).
-   `DescriptorMatrices`: An instance of this class is used to fetch context-dependent phrases.
-   **Action Options**: The prompt includes a hardcoded list of 5 action options (e.g., "1. Go to the market to Buy_Food."). It does *not* dynamically use the `prioritized_actions` generated by the `ActionOptions` class within `PromptBuilder`.

**Sophistication of Prompt Generation:**
-   **Strengths**:
    -   Uses `DescriptorMatrices` to add significant textual variety and context based on character job, status, and environment. This makes the prompts feel more personalized and less robotic.
    -   Includes character's internal state (health, hunger), recent events, financial status, and long-term goals, providing good context for the LLM.
    -   Follows a standard chat-like format which LLMs are typically fine-tuned for.
-   **Weaknesses/Areas for Improvement**:
    -   **Action Choices**: The action choices presented to the LLM are static and hardcoded in the prompt string. The `ActionOptions.prioritize_actions` method, which is supposed to generate a dynamic list of relevant actions, is not currently used to populate these choices in the prompt. This is a major limitation.
    -   **Needs Priorities**: The `NeedsPriorities` class calculates detailed need scores for the character. While `calculate_needs_priorities` is called in `PromptBuilder`, the resulting priorities are not explicitly included in the prompt string sent to the LLM. Their influence is indirect, possibly through `ActionOptions` if it were used.
    -   **Depth of Context**: While good, it could be expanded. For example, including summaries of relevant memories, more details about current relationships, or active quests could lead to more nuanced LLM decisions.
    -   **Flexibility**: Primarily generates one type of prompt ("daily routine"). The `generate_crisis_response_prompt` is a stub. A more robust system would have templates for various situations (social interactions, problem-solving, etc.).
    -   **Output Formatting Expectation**: The prompt ends with "... I choose ", implying the LLM should complete this sentence. This is a common technique but relies on the `OutputInterpreter` to parse this specific format.

**Comparison with `documentation_summary.txt`:**
-   Aligns with the role of `PromptBuilder` in constructing text prompts using character and world context.
-   The "LLM Prompting (Optional/Implicit)" step in the data flow mentions prompts based on character context, needs, and action options/plans. The current prompt includes context and (hardcoded) action options. "Needs" are calculated but not directly inserted. "Plans" from GOAP are not included.
-   The sophistication is moderate. Good contextual detail, but lacks dynamic action choices and deeper integration of internal character state (like explicit needs priorities or memory summaries) into the prompt text itself.

---
## `tiny_strategy_manager.py` Analysis

**Strategies Adoptable:**
-   The file implies strategies for:
    -   **Daily Activities**: `plan_daily_activities(character)` method. The goal is defined as maximizing "satisfaction" and minimizing "energy_usage".
    -   **Responding to Job Offers**: `respond_to_job_offer(character, job_details, graph)` method. The goal is to maximize "career_progress".
    -   **Generic Event Response**: `update_strategy(events, subject)` seems to be a general entry point, which then calls specific planning methods based on event type.

**Strategy Formulation and Management:**
-   **`StrategyManager` Class**:
    -   Initializes with a `GOAPPlanner` and `GraphManager`.
    -   `get_character_state_dict(character)`: Extracts a simplified character state (hunger, energy, money, social, mental health) for utility calculations.
    -   `get_daily_actions(character: Character, current_goal: Goal = None)`:
        -   Generates a list of potential actions (e.g., `NoOpAction`, `WanderAction`, `EatAction` from inventory, `SleepAction` if at home and low energy, `WorkAction` if employed).
        -   Crucially, it uses `calculate_action_utility` (from `tiny_utility_functions.py`) to score each potential action based on the character's current state and the (optional) `current_goal`.
        -   Returns actions sorted by utility. This method acts more like a dynamic action generator and ranker rather than a GOAP planner itself.
    -   `update_strategy(events, subject="Emma")`:
        -   Currently, it only reacts to "new_day" events by calling `plan_daily_activities`.
        -   For other events (not implemented), it fetches character state and possible actions from `GraphManager` and calls `goap_planner.plan_actions`. This `plan_actions` call seems to be to the `GOAPPlanner`'s own method, which is currently just a sort.
    -   `plan_daily_activities(character)`:
        -   Defines a goal (max satisfaction, min energy).
        -   Calls `self.get_daily_actions(character)` to get a list of utility-sorted actions.
        -   The line `current_state = self.graph_analysis(character_graph, character, "daily")` is commented out or refers to a non-existent method.
        -   The line `plan = self.goap_planner(character, goal, current_state, actions)` is problematic:
            -   `self.goap_planner` is an instance of `GOAPPlanner`. It doesn't have a method that directly matches this signature. The static `GOAPPlanner.goap_planner` requires `char_state` and `actions` in a different format. The instance method `plan_actions` just sorts.
        -   `final_decision = evaluate_utility(plan, character)`: This seems to call a function from `tiny_utility_functions` that might be intended to evaluate a whole plan, or it might be a miscall to `GOAPPlanner.evaluate_utility` which evaluates a single action from a plan.
    -   `get_career_actions(character, job_details)`: Returns a static list of mock career-related actions.
    -   `respond_to_job_offer(character, job_details, graph)`: Similar structure to `plan_daily_activities`, including the problematic `goap_planner` call.

**Alignment with `strategy_management_architecture.md` and `documentation_summary.txt`:**
-   **`strategy_management_architecture.md`**:
    -   Describes `StrategyManager` as a "Strategic Decision Orchestrator." The current implementation attempts this by having methods for different strategic contexts (daily, career).
    -   Lists dependencies on `GOAPPlanner` and `GraphManager`, which are present.
    -   Core functions like `update_strategy` and `plan_daily_activities` exist.
    -   The Mermaid diagrams in this document show a flow where `StrategyManager` requests planning from `GOAPSystem`. This interaction point seems to be the most problematic/incomplete in the current code, as the method calls don't align correctly with `GOAPPlanner`'s methods.
-   **`documentation_summary.txt`**:
    -   States `StrategyManager` orchestrates high-level decision-making using GOAP and LLM prompts.
        -   GOAP usage is intended but implemented confusingly (as noted above).
        -   LLM prompt usage is not evident in `StrategyManager`.
    -   It "involve selecting and prioritizing goals using `GOAPPlanner.evaluate_goal_importance()`." This evaluation happens within `GOAPPlanner`, but `StrategyManager` needs to provide the goals to be evaluated. The current `plan_daily_activities` defines a very high-level goal dictionary, not a specific `Goal` object that `evaluate_goal_importance` would expect.
    -   It "uses `GOAPPlanner` to generate action sequences (plans)." This is the key area of misalignment. The `get_daily_actions` method in `StrategyManager` itself generates and ranks actions based on immediate utility, rather than relying on `GOAPPlanner` to search for a sequence of actions to achieve a specific `Goal` object.
-   **Overall**: The `StrategyManager` has the basic structure for strategic orchestration but its core logic, especially its interaction with `GOAPPlanner` for actual multi-step planning, is either incomplete, misaligned with `GOAPPlanner`'s current methods, or relies on a simplified utility-based action selection (`get_daily_actions`) instead of true GOAP planning for daily activities. The documented data flow (points 2-5) where `StrategyManager` gets state, evaluates goals, and then uses GOAP for planning is not fully realized in a coherent way.
