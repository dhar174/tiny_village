# GraphManager Analysis


## Purpose and Overview

The `GraphManager` class serves as the central data hub for Tiny Village. It encapsulates the entire world state using `networkx.MultiDiGraph` and exposes rich methods for adding entities (characters, locations, items, events, etc.), defining relationships, and querying or updating this knowledge graph.

The class is implemented as a singleton (lines 634‑647) so all systems share the same graph instance.

```python
class GraphManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
```


Once instantiated, it creates data structures such as `self.characters`, `self.locations`, etc., and initializes the underlying graph (lines 648‑699). Nodes store references back to their Python objects, which keeps the graph in sync with object state.

## Major Responsibilities

1. **World State Representation**
   - `add_*_node` methods insert entities into the graph with extensive attributes (e.g., `add_character_node` lines 738‑783).
   - Edges represent interactions such as social relationships or resource ownership. Functions like `add_character_character_edge` (lines 920‑1063) record trust, emotional impact and other metrics.

2. **Relationship and Motive Calculations**
   - Functions calculate romance compatibility (`calculate_romance_compatibility` starting at line 1117) or romance interest (`calculate_romance_interest` lines 1208‑1276).
   - `calculate_motives` (lines 3094‑3294) generates a `PersonalMotives` object for characters using their traits and current status.

3. **Graph Querying & Analysis**
   - `get_filtered_nodes` (lines 4653‑4980) performs complex multi-criteria searches across node and edge attributes, supporting GOAP planning and strategic decisions.
   - Utility methods like `get_nearest_resource`, `get_distance_between_nodes`, and `calculate_social_influence` help AI modules reason about proximity and social context.

4. **Goal & Action Evaluation**
   - `calculate_goal_difficulty` (lines 5232‑5406) and `calculate_action_viability_cost` (lines 5964‑6056) evaluate how hard a goal or action is by inspecting the graph for required conditions and costs.
   - `will_action_fulfill_goal` (lines 6063‑6171) simulates applying action effects to see if a goal’s completion conditions are met.

5. **Integration with Other Systems**
   - Characters, the StrategyManager, GameplayController and ActionSystem all rely on GraphManager for up‑to‑date world info. For example, `StrategyManager` initializes a GraphManager instance and queries it for character state and possible actions (see `tiny_strategy_manager.py` lines 18 and 85‑356).
   - Memory queries (`query_memories` at lines 4048‑4114) integrate with `tiny_memories.py` to factor past events into social influence.
   - The bottom of `tiny_graph_manager.py` defines a helper class `TinyVillageGraph` which simply wraps GraphManager for demo purposes (lines 6223‑6350).

## Example: Adding a Character Relationship

The `add_character_character_edge` method illustrates how rich the relationship model is. It logs interactions, computes romance metrics and stores many attributes on the edge:

```python
self.G.add_edge(
    char1,
    char2,
    type=edge_type,
    relationship_type=relationship_type,
    strength=strength,
    historical=historical,
    emotional=emotional_impact_after,
    interaction_frequency=interaction_frequency,
    trust=trust,
    distance=dist(char1.coordinates_location, char2.coordinates_location),
    interaction_log=interaction_log,
)
```

These values are later used when calculating edge costs (`calculate_edge_cost` starting at line 1481), influencing pathfinding and goal evaluation.

## Role in Tiny Village Architecture

Design documents describe GraphManager as the **knowledge foundation** of the game. It feeds data to:
- **GOAPPlanner** for plan generation.
- **StrategyManager** for higher-level decision making.
- **ActionSystem** for verifying preconditions and applying effects.
- **MemoryManager** for retrieving context-aware memories.

All updates from actions or events ultimately propagate back into GraphManager, keeping a single authoritative model of the world.

## Conclusion

`GraphManager` is a massive module (~6.4k lines) that centralizes entity data and complex relationship logic. By exposing detailed node/edge operations, query utilities and cost calculations, it enables the rest of Tiny Village to reason about the evolving world. Maintaining synchronization between Python objects and the graph is critical for accurate AI behaviour.

