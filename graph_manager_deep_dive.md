# Component Deep Dive: GraphManager

The `GraphManager` is a crucial component in TinyVillage, serving as the central nervous system for world state and entity relationships. It uses a `networkx.MultiDiGraph` to model the game world.

## Internal Structure

*   **Core Graph (`self.G`):** A `networkx.MultiDiGraph` allowing multiple directed edges between nodes, representing different types of relationships.
*   **Node Dictionaries:** Separate dictionaries (`self.characters`, `self.locations`, etc.) hold Python instances of game entities, which are also used as nodes in the graph.
*   **Node Types & Key Attributes:**
    *   **Character:** `name`, `age`, `job`, `happiness`, `energy_level`, `wealth_money`, `health_status`, `hunger_level`, `mental_health`, `social_wellbeing`, `shelter`, `coordinates_location`, `inventory`, `needed_resources`, `mood`.
    *   **Location:** `name`, `popularity`, `activities_available`, `accessible`, `security`, `coordinates_location`, `threat_level`, `visit_count`.
    *   **Event:** `name`, `event_type`, `date`, `importance`, `impact`, `required_items`, `coordinates_location`.
    *   **Object (Item):** `name`, `item_type`, `value`, `usability`, `coordinates_location`, `ownership_history`.
    *   **Activity:** `name`, `related_skills`, `required_items`, `preconditions`, `effects`.
    *   **Job:** `name`, `job_title`, `required_skills`, `location`, `salary`, `available`.
*   **Edge Types & Key Attributes:**
    *   **`character_character`**: `relationship_type` (friend, family), `strength`, `historical`, `emotional_impact`, `trust`, `interaction_frequency`, `romance_compatibility`.
    *   **`character_location`**: `frequency_of_visits`, `last_visit`, `favorite_activities`, `ownership_status`.
    *   Many others, including interactions between characters and items, events, activities, jobs, and inter-location/item relationships.

## Main Responsibilities

1.  **World State Representation:**
    *   Acts as the single source of truth for all game entities and their current states.
    *   Dynamically adds/removes/updates nodes and edges.

2.  **Relationship Management:**
    *   Stores and calculates complex, evolving relationship attributes (e.g., trust, emotional impact).
    *   Provides functions to analyze relationships (e.g., `analyze_character_relationships`).

3.  **Data Provision for AI:**
    *   Supplies world context to `StrategyManager` and `GOAPPlanner`.
    *   `get_character_state()`: Retrieves character attributes.
    *   `get_filtered_nodes()`: Powerful querying to find entities matching complex criteria, essential for planning and decision-making.

4.  **Goal & Action Feasibility/Cost Calculation:**
    *   `calculate_goal_difficulty()`: Determines how hard a goal is to achieve, factoring in paths, resources, and action viability.
    *   `calculate_action_effect_cost()`: Assesses the impact of an action on goal progress.
    *   `will_action_fulfill_goal()`: Checks if an action's effects satisfy goal conditions.

5.  **Character Motive Calculation:**
    *   `calculate_motives(character)`: Derives character motive scores (hunger, wealth, etc.) from personality traits and current game state.

6.  **Spatial Awareness Support:**
    *   Manages entity locations and calculates distances, supporting pathfinding.

## Key Interactions

*   **`Character` Objects:** Characters are nodes; their states are mirrored in the graph. They query `GraphManager` for perception and update it after actions. `GraphManager` calculates their motives.
*   **`GOAPPlanner`:** Relies heavily on `GraphManager` for world state, action details, and goal/action cost/difficulty calculations.
*   **`StrategyManager`:** Queries `GraphManager` for character state and world context to inform strategic decisions.
*   **`ActionSystem`:** Action preconditions are checked against `GraphManager` state; action effects update the `GraphManager`.
*   **`EventHandler`:** Adds event nodes to the graph.

## Strengths

*   **Centralized & Comprehensive World Model:** Enables consistent and complex world state.
*   **Rich Relationship Modeling:** Captures nuanced interactions.
*   **Powerful Querying:** Facilitates complex AI decision-making.

## Potential Challenges

*   **Complexity & Performance:** The sheer volume of data and calculations can be demanding.
*   **Data Synchronization:** Keeping Python object states and graph attributes perfectly in sync is crucial.
*   **Scalability:** Performance may degrade as the game world grows.

In essence, `GraphManager` is the knowledge foundation of TinyVillage, enabling sophisticated AI by providing a detailed, queryable, and dynamic representation of the game world and its inhabitants.
