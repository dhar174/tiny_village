# GraphManager Analysis

## Core Concepts

The `GraphManager` class is a central component in the Tiny Village system, serving as the primary model for the game world's state and the relationships between its entities. It utilizes a graph data structure to represent these complex interactions.

### Core Data Structure

The `GraphManager` uses a `networkx.MultiDiGraph`. This choice is significant because:
- **MultiGraph:** It allows multiple edges between the same two nodes. This is useful for representing different types of relationships or interactions simultaneously (e.g., two characters can be colleagues and also friends).
- **DiGraph (Directed Graph):** Edges have a direction, which can be important for representing asymmetrical relationships or flows (e.g., character A influences character B, an item is owned *by* a character).

### Entity Types as Nodes

The graph stores various game entities as nodes. Each node type has a set of key attributes that define its state and properties:

*   **Character:**
    *   Key Attributes: `name`, `age`, `job` (instance of Job class), `happiness`, `energy_level` (likely a float/int), `relationships` (dict storing details), `emotional_state` (dict), `coordinates_location` (tuple/Point object), `inventory` (ItemInventory instance), `needed_resources` (list of item requirements), `mood`, `wealth_money`, `health_status`, `hunger_level`, `mental_health`, `social_wellbeing`, `shelter`, `has_investment` (boolean).
*   **Location:**
    *   Key Attributes: `name`, `type` (e.g., cafe, park), `popularity` (float/int), `activities_available` (list of Activity instances/names), `accessible` (boolean), `security` (float/int), `coordinates_location`, `threat_level`, `visit_count`.
*   **Event:**
    *   Key Attributes: `name`, `event_type`, `date` (datetime object), `importance` (float/int), `impact` (dict describing effects), `required_items` (list), `coordinates_location`.
*   **Object (Item):**
    *   Key Attributes: `name`, `item_type`, `item_subtype`, `value` (float/int), `usability` (boolean/float), `coordinates_location`, `ownership_history` (list of character names).
*   **Activity:**
    *   Key Attributes: `name`, `related_skills` (list of skill names/ids), `required_items` (list), `preconditions` (list of functions/conditions), `effects` (list of state changes).
*   **Job:**
    *   Key Attributes: `job_name` (or `name`), `job_title`, `required_skills` (list), `location` (Location instance/name), `salary` (float/int), `available` (boolean).

### Relationship Types as Edges

Edges connect these nodes, representing various forms of relationships and interactions. Key attributes define the nature and strength of these connections:

*   **`character_character`**:
    *   Key Attributes: `relationship_type` (e.g., "friend", "family", "colleague", "antagonist"), `strength` (float, 0-1), `historical` (float, representing duration/intensity of history), `emotional_impact` (float, -1 to 1), `trust` (float, 0-1), `interaction_frequency` (float), `interaction_log` (dict of timestamps and interaction details), `romance_compatibility` (float), `romanceable` (boolean), `romance_interest` (boolean), `romance_value` (float).
*   **`character_location`**:
    *   Key Attributes: `frequency_of_visits` (int), `last_visit` (datetime), `favorite_activities` (dict), `ownership_status` (e.g., "full", "partial", None), `distance`.
*   **`character_object` (Item Interaction)**:
    *   Key Attributes: `ownership_status` (boolean), `usage_frequency` (int), `sentimental_value` (float), `last_used_time` (datetime), `distance`.
*   **`character_event`**:
    *   Key Attributes: `participation_status` (boolean), `role` (e.g., "organizer", "attendee"), `impact_on_character` (float/dict), `emotional_outcome` (float), `distance`.
*   **`character_activity`**:
    *   Key Attributes: `engagement_level` (float), `skill_improvement` (float), `activity_frequency` (float), `motivation` (float), `distance`.
*   **`character_job`**:
    *   Key Attributes: `role`, `job_status`, `job_performance`, `qualifies_for_job` (boolean), `distance`.
*   **Other Edge Types**:
    *   `location_location`: `proximity`, `connectivity`, `rivalry`, `is_overlapping`.
    *   `location_item`: `item_presence`, `item_relevance`, `item_at_location`.
    *   `location_event`: `event_occurrence`, `location_role`, `capacity`, `event_at_location`.
    *   `location_activity`: `activity_suitability`, `activity_popularity`, `exclusivity`, `activity_at_location`.
    *   `item_item`: `compatibility`, `combinability`, `conflict`.
    *   `item_activity`: `necessity`, `enhancement`, `obstruction`.
    *   `event_activity`: `activities_involved`, `activity_impact`.
    *   `job_location`: `essential_for_job`, `location_dependence`.
    *   `job_activity`: `activity_necessity`, `performance_enhancement`.

These core concepts establish the `GraphManager` as a detailed and dynamic representation of the game world, enabling complex queries and simulations.

## Key Responsibilities

The `GraphManager` is tasked with several critical responsibilities within the Tiny Village ecosystem:

### 1. World State Representation
The `GraphManager` acts as the single source of truth for the current state of all game entities and their attributes.
- **Dynamic Updates:** It handles the addition, removal, and modification of nodes (characters, locations, items, etc.) and edges (relationships, interactions) as the game progresses. For instance, when a character moves, their `coordinates_location` attribute is updated. When a new item is created, an item node is added.

### 2. Relationship Management
A core function is the detailed modeling and dynamic updating of relationships between entities, especially characters.
- **Complex Attributes:** It calculates and stores nuanced relationship attributes such as `trust`, `emotional_impact`, `historical` (duration/intensity of past interactions), `strength`, and `romance_compatibility`.
- **Dynamic Evolution:** Methods like `update_emotional_impact`, `update_trust`, `calculate_romance_compatibility`, and `calculate_romance_interest` ensure that relationships evolve based on interactions.
- **Scaling and Decay:** It employs mathematical functions (sigmoid, tanh) for normalizing relationship metrics and decay functions (`decay_effect`, `linear_decay_effect`) to simulate the fading of emotional impacts or the strengthening/weakening of ties over time and based on interaction history.
- **Analysis:** Provides methods like `analyze_character_relationships` and `check_friendship_status` to query and understand these relationships.

### 3. Data Provision for AI Systems
The `GraphManager` is a primary data source for various AI components, enabling them to make informed decisions.
- **Character State:** `get_character_state(character_name)` retrieves a comprehensive dictionary of a character's current attributes.
- **World Perception:** `get_filtered_nodes(**kwargs)` is a powerful and flexible querying method. AI systems can use it to find entities (characters, items, locations) that match complex criteria, such as finding all characters with a "friendly" relationship, all "food" items within a certain distance, or locations suitable for a specific activity. This is crucial for context-aware behavior.
- **Possible Actions:** `get_possible_actions(character_name)` likely analyzes the character's current situation in the graph (e.g., nearby entities, available items) to suggest potential interactions.
- **Social Context:** `calculate_social_influence(...)` helps AI understand how a character's decisions might be swayed by their social network and past memories.

### 4. Goal & Action Feasibility/Cost Calculation
The `GraphManager` plays a vital role in planning systems like GOAP by assessing goals and actions.
- **`calculate_goal_difficulty(goal, character)`:** This sophisticated method determines how challenging a given `Goal` is for a `Character`. It considers:
    - Identifying nodes in the graph that match the goal's criteria.
    - The viability and cost of actions that can be performed on/with these nodes.
    - Finding optimal sequences or combinations of actions to fulfill all goal requirements, potentially using search algorithms like A* and heuristics.
- **`calculate_action_viability_cost(node, goal, character)`:** Evaluates the cost and whether an action performed on/with a specific `node` is viable in the context of achieving a `goal`.
- **`calculate_action_effect_cost(action, character, goal)`:** Assesses how the effects of an `action` contribute to or detract from `goal` progress, effectively costing the impact of an action on the goal.
- **`will_action_fulfill_goal(action, goal, current_state, character)`:** Checks if the effects of a single `action` are sufficient to meet one or more of the `goal`'s completion conditions.
- **`is_goal_achieved(character, goal)`:** Determines if a character has successfully met all completion criteria for a given goal.
- **Edge and Action Costs:** Numerous helper methods (e.g., `calculate_edge_cost`, `calculate_char_char_edge_cost`) compute the "cost" of traversing edges or performing actions, which likely feeds into pathfinding and decision-making utilities.

### 5. Character Motive Calculation
- **`calculate_motives(character)`:** This method derives a character's intrinsic motivations (e.g., hunger, wealth, social connection, stability) based on their personality traits (openness, extraversion, etc.) and current status in the game world. These motives are scaled using cached sigmoid functions.

### 6. Spatial Awareness Support
The `GraphManager` helps with understanding the spatial layout of the game world.
- **Location Tracking:** It stores `coordinates_location` for all relevant entities.
- **Distance Calculations:** It provides utilities to calculate distances between entities (e.g., `dist()`, `get_distance_between_nodes`), which is fundamental for pathfinding, proximity checks, and some cost calculations.
- **Resource Proximity:** Methods like `get_nearest_resource` help characters find items or locations they need.

## Interactions with Other System Components

The `GraphManager` does not operate in isolation; it is deeply integrated with various other components of the Tiny Village system, serving as a central hub for data exchange and world state queries.

*   **`Character` Objects (e.g., from `tiny_characters.py`)**
    *   **Interaction:** `GraphManager` stores `Character` instances themselves as nodes (or references to them) and mirrors their attributes (like `name`, `age`, `inventory`, `mood`, `motives`, `personality_traits`) within the graph node data. When character states change (e.g., wealth increases, mood shifts), these changes are ideally reflected in the graph. `GraphManager` calls methods on `Character` objects (e.g., `to_dict()`, `get_wealth_money()`, `get_motives()`, `qualifies_for_job()`).
    *   **Role:** Characters are the primary actors. They would query `GraphManager` to perceive their environment (e.g., "who is nearby?", "what items are at this location?"), understand their relationships, and get context for their decisions. After performing actions, their state changes would be updated in the `GraphManager`. `GraphManager`'s `calculate_motives` directly influences character drives.

*   **`Location`, `Event`, `ItemObject`, `Action`, `Job` Objects (from respective `tiny_*.py` files)**
    *   **Interaction:** Similar to characters, instances of these classes (or their data) are added as nodes in the graph. `GraphManager` calls their methods (e.g., `to_dict()`, `distance_to_point_from_nearest_edge()` for locations) to populate node attributes or get information.
    *   **Role:** These entities form the static and dynamic elements of the game world. Their properties and relationships, managed by `GraphManager`, define the environment characters interact with.

*   **`GOAPPlanner` (Goal-Oriented Action Planning system, e.g., `tiny_goap_system.py`)**
    *   **Interaction:** The `GOAPPlanner` is a major client of `GraphManager`. It relies on:
        *   `get_character_state()` and `get_filtered_nodes()` for current world state.
        *   `calculate_goal_difficulty()` to understand the complexity of achieving objectives.
        *   `calculate_action_viability_cost()` and `calculate_action_effect_cost()` to evaluate the costs and benefits of potential actions.
        *   Pathfinding and connectivity data to plan sequences of actions that might involve movement or interaction with multiple entities.
        *   `will_action_fulfill_goal()` and `is_goal_achieved()` to check progress towards goals.
    *   **Role:** `GraphManager` provides the necessary world knowledge for the GOAP system to construct and evaluate plans for characters.

*   **`StrategyManager` (e.g., `tiny_strategy_manager.py`)**
    *   **Interaction:** This higher-level decision-making component would query `GraphManager` for a broader understanding of the game state, such as:
        *   Character relationships and social networks (`analyze_character_relationships`, `calculate_social_influence`).
        *   Overall status of character needs and motives.
        *   Availability of strategic resources or opportunities (e.g., `explore_career_opportunities`).
    *   **Role:** `GraphManager` provides the contextual data that `StrategyManager` uses to set long-term goals or guide character behavior at a more abstract level than GOAP.

*   **`ActionSystem` (e.g., `actions.py`)**
    *   **Interaction:**
        *   **Precondition Checking:** The `ActionSystem` (or individual `Action` objects) would check action preconditions against the current world state maintained in `GraphManager`.
        *   **Effect Application:** When an action is executed, its effects (e.g., changes to character attributes, item transfers, relationship modifications) are recorded by updating the relevant nodes and edges in `GraphManager`.
    *   **Role:** `GraphManager` serves as the database for validating actions and the canvas on which action outcomes are painted.

*   **`EventHandler` (e.g., `tiny_event_handler.py`)**
    *   **Interaction:** The `EventHandler` is responsible for introducing new events into the game. It would:
        *   Call `GraphManager.add_event_node()` to add the event to the graph.
        *   Update character-event edges (`add_character_event_edge()`) to reflect participation or impact.
    *   **Role:** `GraphManager` logs events, allowing their impact to be considered in character decision-making and historical tracking.

*   **`MemoryManager` (e.g., `tiny_memories.py`)**
    *   **Interaction:** `GraphManager.calculate_social_influence()` includes a mechanism (though partly placeholder in the provided code) to query a `MemoryManager` (e.g., `tiny_memories.MemoryManager().search_memories(topic)`).
    *   **Role:** Memories of past interactions or events, managed by `MemoryManager`, can be retrieved and factored into `GraphManager`'s calculations (like social influence), adding depth to character behavior.

*   **`TimeManager` (e.g., `tiny_time_manager.py`)**
    *   **Interaction:** `GraphManager` uses time information (e.g., `datetime.now().timestamp()`, `GameTimeManager`) for:
        *   Timestamping interaction logs in character-character relationships.
        *   Calculating `days_known` for relationship strength (`historical` attribute).
        *   Powering decay functions for emotional impact over time.
    *   **Role:** Provides the temporal context necessary for features that evolve or depend on the passage of time.

*   **Utility Functions (e.g., `tiny_utility_functions.py`)**
    *   **Interaction:** `GraphManager` imports `is_goal_achieved` from `tiny_utility_functions`, suggesting that some utility calculations or checks might directly use or be used by `GraphManager`.
    *   **Role:** Utility functions likely leverage data from `GraphManager` to assess the value or outcome of states or actions.

## `GraphManager`'s Place in the Tiny Village System

The `GraphManager` is the backbone of the Tiny Village simulation, acting as a dynamic and comprehensive model of the game world. Its central role can be understood through several key aspects:

*   **Foundation for Perception:** For AI characters, the `GraphManager` *is* the world they perceive. They query it to understand their surroundings, identify other characters and objects, learn about relationships, assess potential threats, and find necessary resources. Methods like `get_filtered_nodes()` and `get_character_state()` are crucial for this "sensory" input.

*   **Engine for Decision-Making:**
    *   **Strategic Layer (via `StrategyManager`):** The `GraphManager` provides the high-level context (social dynamics, long-term character needs, world state) that allows the `StrategyManager` to guide characters' overarching goals and behaviors.
    *   **Tactical Layer (via `GOAPPlanner`):** It supplies the detailed information needed for fine-grained planning. The `GOAPPlanner` heavily relies on the graph to find sequences of actions to achieve immediate goals, using `GraphManager`'s methods to evaluate action costs, feasibility, and their impact on goal fulfillment (`calculate_goal_difficulty`, `calculate_action_viability_cost`).

*   **Core of World Simulation:** As characters act and events unfold, the `GraphManager` is updated to reflect these changes. This makes it the canonical representation of the evolving game state. An action by one character (e.g., giving a gift) can alter relationship attributes stored in the graph, which then influences future interactions between those characters, creating a persistent and reactive world.

*   **Enabler of Emergent Behavior:** The intricate web of nodes and edges, each with multiple dynamic attributes (emotions, trust, needs, motives, item states, location properties), forms a complex system. Simple rules governing individual interactions, when played out across this graph, can lead to emergent social dynamics, unforeseen consequences, and complex character behaviors that are not explicitly scripted. For example, a series of minor negative interactions could gradually turn friends into rivals, or a character's persistent need for a resource could drive them to explore new locations or interact with new characters.

In essence, the `GraphManager` transforms the game from a collection of separate entities into a cohesive, interconnected ecosystem. It provides the intelligence layer that allows characters to behave believably and for the world to feel alive and responsive.

## Key Data Structures and Algorithms

The `GraphManager` leverages several important data structures and algorithms to perform its functions effectively:

*   **`networkx.MultiDiGraph`:**
    *   **Structure:** The fundamental data structure from the NetworkX library. It's a directed graph that allows multiple edges between any pair of nodes.
    *   **Use:** Represents the game world, with entities as nodes and relationships/interactions as directed edges. The multi-edge capability is crucial for modeling complex relationships (e.g., characters can be friends and coworkers simultaneously, with distinct edges for each).

*   **Python Dictionaries:**
    *   **Structure:** Standard Python hash maps.
    *   **Use:** Extensively used for:
        *   Storing collections of game entities (e.g., `self.characters`, `self.locations`) for quick lookup by name or ID, mapping entity names to their Python object instances.
        *   Representing node and edge attributes within the NetworkX graph.
        *   Caching results (e.g., `self.dp_cache`).
        *   Passing structured data (e.g., criteria for `get_filtered_nodes`).

*   **Pathfinding Algorithms (via NetworkX):**
    *   **Algorithms:** Dijkstra's algorithm is implicitly used by NetworkX functions like `nx.shortest_path()` and `nx.all_pairs_dijkstra_path()`.
    *   **Use:** Finding the shortest or all paths between nodes, essential for character movement, determining reachability, and some cost calculations (e.g., distance-based costs).

*   **Community Detection (via NetworkX):**
    *   **Algorithm:** The Louvain method, implemented as `networkx.algorithms.community.louvain_communities()`.
    *   **Use:** To identify clusters or communities of nodes within the graph, which can be useful for analyzing social structures or grouping related entities.

*   **Centrality Measures (via NetworkX):**
    *   **Algorithm:** Degree centrality (`nx.degree_centrality()`).
    *   **Use:** To identify the most connected or influential nodes in the graph, particularly for analyzing character influence.

*   **Caching Mechanisms:**
    *   **`@functools.lru_cache`:**
        *   **Algorithm:** Least Recently Used (LRU) caching decorator.
        *   **Use:** Applied to frequently called functions with potentially expensive computations that have deterministic outputs for given inputs. This is seen on scaling functions (like `cached_sigmoid_relationship_scale_approx_optimized`) and `calculate_action_viability_cost` (though the latter uses a custom `self.dp_cache`).
    *   **`StdCache` Class:**
        *   **Algorithm:** A custom time-based cache for standard deviation values.
        *   **Use:** `self.std_cache` is used to cache standard deviation calculations for attribute values, refreshing periodically to avoid stale data while reducing redundant computations.
    *   **`self.dp_cache` (Dynamic Programming Cache):**
        *   **Algorithm:** A dictionary used for memoization in the `calculate_action_viability_cost` method.
        *   **Use:** Stores results of previous calculations for specific (node, goal, character) tuples to avoid re-computing complex action viability and costs.

*   **Heuristic Search / A* Principles:**
    *   **Algorithm:** Evident in `calculate_goal_difficulty`. While not a direct A* implementation, it uses concepts like:
        *   A priority queue (`heapq`) for exploring solutions.
        *   A heuristic function (`heuristic(remaining_conditions)`) to estimate the cost to reach the goal.
        *   Evaluation of combinations of actions to fulfill goal requirements.
    *   **Use:** To find an optimal or near-optimal sequence of actions to achieve a complex goal by intelligently exploring the solution space.

*   **Greedy Algorithms:**
    *   **Algorithm:** Making locally optimal choices at each step.
    *   **Use:** Mentioned as an initial approach within `calculate_goal_difficulty` if the problem space seems less complex (high viability ratio of actions).

*   **Combinatorics (`itertools.combinations`):**
    *   **Algorithm:** Generates all possible combinations of a certain length from a set of items.
    *   **Use:** In `calculate_goal_difficulty` to evaluate different combinations of actions or nodes that could satisfy goal requirements.

*   **Mathematical Scaling/Normalization Functions:**
    *   **Algorithms:** Sigmoid functions (various approximations like `cached_sigmoid_relationship_scale_approx_optimized`, `cached_sigmoid_motive_scale_approx_optimized`) and hyperbolic tangent (`tanh_scaling`).
    *   **Use:** To normalize or scale values (e.g., relationship strengths, motive scores) into specific ranges (often 0-1 or -1 to 1), making them comparable and suitable for use in other calculations.

These structures and algorithms enable the `GraphManager` to model a complex, dynamic world and provide powerful analytical capabilities for the game's AI systems.

## Concluding Analysis Summary

The `GraphManager` stands out as a highly sophisticated and pivotal component of the Tiny Village system. It transcends a simple data store, acting as a dynamic computational engine that models the intricate web of entities, relationships, and their evolving states within the game world.

**Key Takeaways:**

*   **Centralized Intelligence:** It serves as the primary knowledge base, providing perception, context, and analytical capabilities that drive intelligent agent behavior through systems like GOAP and `StrategyManager`.
*   **Richly Detailed World Model:** The use of a `MultiDiGraph` combined with extensive attributes for nodes and edges allows for a remarkably detailed and nuanced representation of game entities and their multifaceted interactions, including complex emotional and social dynamics.
*   **Dynamic and Reactive:** The `GraphManager` is not static. It incorporates mechanisms for real-time updates, relationship evolution (including decay and growth), and the calculation of dynamic attributes like character motives, making the world feel alive and responsive.
*   **Foundation for Complex AI:** Its advanced functionalities, such as `calculate_goal_difficulty` (which involves heuristic search and combinatorial analysis), `get_filtered_nodes` (for complex querying), and social influence calculations, are fundamental for enabling complex, goal-driven, and socially aware AI.
*   **Performance Considerations:** The extensive use of caching (`@lru_cache`, custom caches) indicates an awareness of the potential performance demands of such a complex system and efforts to mitigate them.

In conclusion, the `GraphManager` is the heart of Tiny Village's simulation, providing the structural foundation and the analytical power necessary for creating a believable, interactive, and emergent game experience. Its design reflects a deep understanding of the requirements for simulating complex social systems and intelligent agent behavior.
