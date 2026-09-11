# GraphManager Overview

`tiny_graph_manager.py` contains a massive `GraphManager` class that represents the central world model for Tiny Village. It acts as a singleton so that every system manipulates a single graph instance. Nodes represent characters, locations, items, jobs, events and activities while edges encode relationships and interactions between these entities.

## Construction and Core Data
- The class ensures a single instance via `__new__` and sets up dictionaries for each node type along with an empty `networkx.MultiDiGraph` (`self.G`).【F:tiny_graph_manager.py†L634-L696】
- Node addition methods (`add_character_node`, `add_location_node`, etc.) populate the graph with detailed attributes for each entity. For example `add_character_node` stores personal data, inventory and coordinates when a `Character` object is inserted.【F:tiny_graph_manager.py†L750-L782】

## Relationship Edges
`GraphManager` maintains complex character-to-character edges. When a new relationship is created, emotional impact, interaction log and computed romance data are stored in the edge attributes. Edge updates adjust trust and historical metrics:
【F:tiny_graph_manager.py†L961-L1041】

## Analysis Utilities
Several methods provide graph analysis:
- `find_shortest_path` uses NetworkX to compute a route between nodes.
- `detect_communities` applies Louvain community detection.
- `calculate_centrality` returns degree centrality values.
- `shortest_path_between_characters` wraps the generic path function for convenience.
These appear around lines 2522‑2592 of the file.【F:tiny_graph_manager.py†L2522-L2592】

## Social Influence and Memories
The class includes a long `calculate_social_influence` function which combines relationship attributes and memory data to score how peers influence a character. It queries memory topics (via `tiny_memories`) and applies decay functions to weight past interactions.【F:tiny_graph_manager.py†L3962-L4045】

## Goal and Action Evaluation
GraphManager also computes GOAP-related values. `calculate_action_difficulty`, `calculate_action_viability_cost` and `calculate_goal_difficulty` analyze action preconditions, effects and path costs using graph data. There are helpers like `calculate_how_goal_impacts_character` that inspect completion conditions in relation to a character state.【F:tiny_graph_manager.py†L5908-L5950】

## Integration with Other Modules
Other systems—`EventHandler`, `StrategyManager`, `tiny_goap_system`, etc.—pass or retrieve the singleton GraphManager for world queries. Characters are stored in `self.characters`, locations in `self.locations`, etc. Many tests instantiate a `GraphManager` and pass it to `ActionSystem` or other managers.

## Summary
GraphManager serves as the global knowledge base and reasoning hub. It stores the entire world graph and exposes a large API for adding nodes, updating edges, measuring relationships, finding paths and calculating utilities for GOAP. Other Tiny Village modules depend on it to query current world state or to persist changes after actions and events.
