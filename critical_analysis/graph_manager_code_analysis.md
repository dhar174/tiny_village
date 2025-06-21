# GraphManager Code Analysis

The `GraphManager` class in `tiny_graph_manager.py` is the central repository for
all entities and relationships in Tiny Village.  It is implemented as a
singleton so that other modules share a single world graph instance.  The class
stores a `networkx.MultiDiGraph` and maintains dictionaries mapping entity names
to their Python objects.

## Initialization
`GraphManager` overrides `__new__` to enforce the singleton pattern.  During
`__init__` it creates caches, the dictionaries for characters, locations, events,
objects, activities and jobs, and initializes the graph:
```python
class GraphManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
    def __init__(self):
        if self.__initialized:
            return
        self.std_cache = StdCache(refresh_interval_seconds=60)
        self.unique_graph_id = uuid.uuid4()
        ...
        self.G = self.initialize_graph()
        self.__initialized = True
```
【F:tiny_graph_manager.py†L634-L705】

The call to `initialize_graph` creates a `nx.MultiDiGraph` which allows
bidirectional edges and multiple edge types between the same nodes.

## Node Storage
For each entity type there is a dictionary.  The mapping `type_to_dict_map` is
used by helper methods so the graph can return Python objects when needed.  When
adding the first instance of a type the available attributes are recorded so
queries can later refer to them.  Example for characters:
```python
def add_character_node(self, char: Character):
    if len(self.characters) == 0:
        self.character_attributes = char.to_dict().keys()
    self.characters[char.name] = char
    self.G.add_node(
        char,
        type="character",
        age=char.age,
        job=char.job,
        happiness=char.happiness,
        ...
    )
```
【F:tiny_graph_manager.py†L752-L792】
Similar `add_*_node` functions exist for locations, events, items, activities and
jobs.

## Edge Creation
Edges store rich attributes describing relationships.  The method
`add_character_character_edge` handles interactions between two characters.  It
updates trust, emotional impact, interaction frequency and other metrics, then
creates or updates the edge in the graph:
```python
def add_character_character_edge(self, char1, char2, ...):
    trust = update_trust()
    emotional_impact_after = update_emotional_impact(...)
    ...
    if self.G.has_edge(char1, char2):
        self.update_character_character_edge(...)
        return
    self.G.add_edge(
        char1,
        char2,
        type=edge_type,
        relationship_type=relationship_type,
        strength=strength,
        historical=historical,
        emotional=emotional_impact_after,
        interaction_frequency=interaction_frequency,
        ...
    )
```
【F:tiny_graph_manager.py†L920-L1033】
Edges exist for many combinations (character–location, character–item, etc.) with
corresponding cost calculation methods.  Costs are used by GOAP planning and
utility evaluation.  Example of the distance based cost:
```python
def calc_distance_cost(self, distance, char1, char2):
    if not self.G.has_edge(char1, char2):
        return distance * 1.0
    return (
        distance * (1 - char1.energy)
        * (1 - char2.energy)
        * (1 - self.G[char1][char2]["strength"])
        * (1 - self.G[char1][char2]["trust"])
        ...
    )
```
【F:tiny_graph_manager.py†L1399-L1472】

## Relationship and Community Analysis
GraphManager exposes analysis helpers such as `find_shortest_path`,
`detect_communities`, `calculate_centrality`, `most_influential_character` and
others.  These functions rely on NetworkX algorithms to compute shortest paths
or community clusters.

## Goal and Action Evaluation
A large portion of the class is dedicated to evaluating goals.  The method
`calculate_goal_difficulty` analyzes goal criteria, computes possible actions at
matching nodes and uses a mix of greedy search and A* to find viable paths.  It
returns a difficulty score and diagnostic information:
```python
def calculate_goal_difficulty(self, goal: Goal, character: Character):
    ...
    nodes_per_requirement = {}
    for requirement in goal_requirements:
        nodes_per_requirement[requirement] = self.get_filtered_nodes(**requirement)
    ...
    valid_combinations = [best_solution]
    for combo in combinations(best_solution, len(best_solution)):
        result = self.evaluate_combination(...)
        if result:
            valid_combinations.append(result)
    ...
    return {
        "difficulty": difficulty,
        "viable_paths": viable_paths,
        ...
    }
```
【F:tiny_graph_manager.py†L5120-L5567】

Related methods such as `calculate_action_viability_cost` and
`calculate_action_effect_cost` are used during this computation. They check
preconditions and effects of each `Action` object, referencing the
`ActionSystem`.

## Motive Calculation
`calculate_motives` computes a `PersonalMotives` object for a character derived
from personality traits and current state.  It applies several cached sigmoid and
`tanh_scaling` helper functions for normalization:
```python
def calculate_motives(self, character: Character):
    social_wellbeing_motive = cached_sigmoid_motive_scale_approx_optimized(
        character.personality_traits.get_openness()
        + (character.personality_traits.get_extraversion() * 2)
        + character.personality_traits.get_agreeableness()
        - character.personality_traits.get_neuroticism(),
        10.0,
    )
    ...
    return PersonalMotives(
        hunger_motive=Motive("hunger", "bias toward satisfying hunger", hunger_motive),
        ...
    )
```
【F:tiny_graph_manager.py†L2889-L3240】
These motive scores feed back into action evaluation and the StrategyManager.

## Filtering and Querying
`get_filtered_nodes` is a flexible query method used by planners to locate
entities meeting complex conditions (node/edge attributes, distances,
relationship status, trade opportunities etc.)  It combines several helper
checks and interacts with the inventory system and trade evaluations.

## Integration with Other Components
- **Characters (`tiny_characters.py`)** are created with a reference to the
  GraphManager.  They call it to update location, add edges, and request current
  motives or relationships.
- **ActionSystem** relies on GraphManager for checking and updating state when
  actions are executed.
- **StrategyManager** and **GOAPPlanner** request world state, analyze goals and
  compute plans using GraphManager APIs like `calculate_goal_difficulty` and
  `get_character_state`.
- **EventHandler** adds event nodes and edges when world events occur.

In short, GraphManager is the authoritative world model.  It coordinates data
across many modules and provides analytical functions that enable the AI systems
to reason about the world.
