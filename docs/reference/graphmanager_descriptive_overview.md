# GraphManager Analysis

## Overview
GraphManager, defined in `tiny_graph_manager.py`, acts as the central hub for Tiny Village's world state. The class is implemented as a singleton and maintains a `networkx.MultiDiGraph` that stores characters, locations, objects, events, activities, and jobs as nodes. Edges capture relationships and interactions between these entities. Besides raw storage, GraphManager computes higher-level data such as relationship metrics, motive evaluations, and goal difficulties.

GraphManager interacts with nearly all other modules: `tiny_characters` for character objects and motives, `actions` for possible interactions, `tiny_goap_system` for planning, and others. It is referenced by `StrategyManager`, `EventHandler`, and `PromptBuilder` according to the architecture documents.

## Initialization

`tiny_globals.initialize_global_graph_manager()` imports the graph module before
acquiring the global lifecycle lock. `get_global_graph_manager()` delegates its
cold path to that initializer. A private construction gate serializes the whole
`GraphManager()` allocation/initialization call, including direct constructor
users; public calls return a fully initialized singleton backed by WorldState.

The lock order is global lifecycle lock then class construction lock. Thread-local
guards reject recursive initialization before reacquiring those locks. Construction
failure restores the previous singleton reference and permits retry. A constructing
thread can query publication status without waiting on its own lock.

```python
from tiny_globals import initialize_global_graph_manager, get_global_graph_manager
from actions import ActionSystem
from tiny_strategy_manager import StrategyManager

graph = initialize_global_graph_manager()
assert get_global_graph_manager() is graph
actions = ActionSystem(graph_manager=graph)
strategy = StrategyManager(graph_manager=graph)
```

Explicit graph/action-system injection takes precedence over legacy lazy globals.
Controller composition, recovery, and strategy-generated actions forward that
owner. `tiny_characters.build_example_world_criteria(graph)` replaces the two
object-valued sample globals; importing runtime definitions constructs no sample
world.

`reset_global_graph_manager()` clears both singleton references without importing
the graph module. Reset/replacement is for quiescent test/setup boundaries; discard
old consumers afterwards. It is not a concurrent live-world hot-swap feature.
See [issue #719 validation](../testing/issue-719-graph-initialization-validation.md)
for required coverage and remaining feature/dependency limitations.

## Node Management
GraphManager provides dedicated methods to add different node types. For example `add_character_node` imports `Character` dynamically, stores the object, and records numerous attributes into the graph node:

```python
def add_character_node(self, char: Character):
    Character = importlib.import_module("tiny_characters").Character
    if len(self.characters) == 0:
        self.character_attributes = char.to_dict().keys()
    self.characters[char.name] = char
    wealth_money = char.get_wealth_money()
    has_investment = char.has_investment()
    self.G.add_node(
        char,
        type="character",
        age=char.age,
        job=char.job,
        happiness=char.happiness,
        ...
    )
```
【F:tiny_graph_manager.py†L746-L781】
Similar functions exist for locations, events, objects, activities, jobs, and stocks. These nodes capture domain-specific data such as popularity for locations or salary for jobs.

## Edge Creation
Relationships and interactions are encoded as edges with rich attribute sets. The `add_character_character_edge` method demonstrates this complexity. It maintains an interaction log, calculates romance factors, and stores measures like `strength`, `trust`, and `interaction_frequency`:

```python
def add_character_character_edge(
        self,
        char1,
        char2,
        relationship_type=0,
        ...):
    ...
    self.G.add_edge(
        char1,
        char2,
        type=edge_type,
        relationship_type=relationship_type,
        strength=strength,
        historical=historical,
        emotional=emotional_impact_after,
        interaction_frequency=interaction_frequency,
        interaction_count=len(interaction_log),
        key=edge_type,
        trust=trust,
        distance=dist(char1.coordinates_location, char2.coordinates_location),
        interaction_cost=self.calculate_char_char_edge_cost(...),
        dist_cost=self.calc_distance_cost(...),
        interaction_log=interaction_log,
    )
```
【F:tiny_graph_manager.py†L972-L1034】
Similar edge methods exist for other entity pairs (character-location, item-item, job-activity, etc.), each computing cost metrics relevant for GOAP planning and utility evaluation.

## Retrieval Utilities
GraphManager exposes getters to retrieve objects or node data by name. Functions such as `get_character`, `get_item`, `get_location`, `get_event`, and `get_job` iterate over node attributes to locate the requested entity, logging helpful messages on failures. Example snippet:

```python
def get_character(self, character_str):
    try:
        for node, data in self.G.nodes(data="name"):
            if node == character_str:
                ...
                return node
            elif data == character_str:
                ...
                return self.characters[character_str]
    except Exception as e:
        logging.error(f"Error retrieving character {character_str}: {e}")
    logging.warning(f"No character found with name '{character_str}'.")
    return None
```
【F:tiny_graph_manager.py†L4488-L4518】
Other methods compute aggregate statistics across the graph, e.g., `get_average_attribute_value`, `get_maximum_attribute_value`, and `get_stddev_attribute_value`.

## Goal and Action Evaluation
A significant portion of GraphManager focuses on evaluating actions and goals for GOAP planning. Methods such as `calculate_goal_difficulty`, `calculate_action_difficulty`, and `calculate_action_viability_cost` analyze preconditions, possible interactions, and how actions affect goal completion. These routines often import `tiny_characters`, `tiny_goals`, or `actions` dynamically and use caches (`dp_cache` or `@lru_cache`) for performance. An excerpt from `calculate_action_viability_cost` shows the general approach:

```python
@lru_cache(maxsize=1000)
def calculate_action_viability_cost(self, node, goal: Goal, character: Character):
    cache_key = (node, goal, character)
    if cache_key in self.dp_cache:
        return self.dp_cache[cache_key]
    ...
    possible_interactions = self.node_type_resolver(node).get_possible_interactions()
    for interaction in possible_interactions:
        if interaction.preconditions_met():
            fulfilled_conditions = self.will_action_fulfill_goal(...)
            ...
```
【F:tiny_graph_manager.py†L5960-L6015】
These calculations feed into higher-level path and plan evaluation routines, which compute viable action sequences to meet goal conditions.

## Graph Queries and Analysis
Beyond immediate gameplay usage, GraphManager includes a variety of graph algorithms. For example, `find_shortest_path` wraps NetworkX's shortest path search, `detect_communities` uses Louvain community detection, and `calculate_centrality` measures node influence. Utility functions like `analyze_location_popularity` or `item_ownership_network` traverse the graph to produce analytics about the world.

## Integration in Tiny Village
Design documents describe GraphManager as the central repository for world data, servicing nearly every other subsystem. Characters query it to update state, the GOAP planner consults edge costs and motive calculations, StrategyManager relies on it to formulate plans, and memory operations access it for contextual data. The `TinyVillageGraph` class at the end of `tiny_graph_manager.py` demonstrates how GraphManager is instantiated and provides methods used by the gameplay controller, like `plan_daily_activities` and `analyze_event_impact`.

Overall, GraphManager maintains a rich, interconnected representation of the game world. It encapsulates domain knowledge about relationships, locations, items, and goals. Through numerous helper methods it supports pathfinding, utility assessment, and high-level analytics, forming a backbone for decision making across the Tiny Village project.
