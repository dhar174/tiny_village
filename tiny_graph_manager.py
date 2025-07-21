from ast import Not
from calendar import c
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, time
import dis
from functools import lru_cache
import heapq
from hmac import new
from itertools import chain, combinations, product
import logging
from math import dist, inf, log
import math
import operator
import random
import re
import uuid
import networkx as nx

# from tiny_characters import Motive
from tiny_types import Character, Goal, Action, State, ActionSystem
from tiny_types import Location, Event

from tiny_jobs import Job

# from actions import Action, State, ActionSystem
from tiny_items import ItemInventory, ItemObject, InvestmentPortfolio, Stock
import tiny_memories  # Temporarily commented out for testing
from tiny_utility_functions import is_goal_achieved
import numpy as np
from tiny_time_manager import GameTimeManager as tiny_time_manager
import importlib
from social_model import SocialModel

""" Graph Construction
Defining Nodes:
Characters: Each character in the game will be a node. This includes not only playable characters but also non-playable characters (NPCs).
Locations: Places in the game such as homes, workplaces, public spaces, and event locations.
Events: Significant events that can affect character decisions and relationships, like festivals, job openings, etc.
Interests and Activities: Specific hobbies or tasks that characters can engage in, impacting their skills and relationships.
Defining Edges:
Relationships Between Characters: These edges represent different types of relationships (friends, family, colleagues, antagonists) and their strength or status.
Character-Location Relationships: Connections between characters and locations (e.g., owner, frequent visitor, employee).
Character-Event Relationships: How characters are involved or affected by events (participant, organizer, bystander).
Character-Interest Relationships: Links between characters and their interests or activities, which could influence their skills and social connections.
Edge Attributes:
Strength of Relationship: Quantitative measure of how strong a relationship is, which can affect decision-making.
Nature of Interaction: Attributes that describe the type of interaction (positive, negative, neutral).
Frequency of Interaction: How often characters interact at a location or through an event, impacting the character's routine and decisions. 


1. Nodes Categorization
Nodes represent entities within the game world. Each type of node will have specific attributes based on its role and interactions.

Character Nodes
Attributes: Name, age, job, current status (e.g., happiness, energy level), long-term aspirations, skills.
Purpose: Represent the player and NPC dynamics within the game.
Location Nodes
Attributes: Name, type (e.g., cafe, park, workplace), popularity, activities available.
Purpose: Represent places characters can visit or interact with.
Event Nodes
Attributes: Name, type (e.g., festival, job fair), date, significance.
Purpose: Represent scheduled or random occurrences that characters can participate in.
Object Nodes
Attributes: Name, type (e.g., book, tool), value, usability.
Purpose: Represent items that characters can own, use, or interact with.
Activity Nodes
Attributes: Name, type (e.g., exercise, study), related skill, satisfaction level.
Purpose: Represent actions characters can undertake for personal development, leisure, or job-related tasks.
2. Edges Categorization
Edges represent relationships or interactions between nodes, with attributes that describe the nature and dynamics of these relationships.

Character-Character Edges
Attributes: Relationship type (e.g., friend, family, colleague), strength, historical.
Purpose: Represent social and professional dynamics between characters.
Character-Location Edges
Attributes: Frequency of visits, last visit, favorite activities.
Purpose: Indicate how often and why characters visit certain locations.
Character-Event Edges
Attributes: Participation status, role (e.g., organizer, attendee), impact.
Purpose: Reflect characters' involvement in events and their effects.
Character-Object Edges
Attributes: Ownership status, usage frequency, sentimental value.
Purpose: Represent characters' possessions and how they use or value them.
Character-Activity Edges
Attributes: Engagement level, skill improvement, recentness.
Purpose: Show how characters engage in various activities and their effects.
3. Attributes Definition
Each node and edge will have attributes that need to be quantitatively or qualitatively defined to enable effective graph analyses.

Quantitative Attributes: Numeric values (e.g., strength of relationships, frequency of visits) that can be easily measured and calculated.
Qualitative Attributes: Descriptive characteristics (e.g., type of relationship, role in an event) that provide context for decision-making processes.
4. Graph Dynamics
The graph should be dynamic, allowing for updates based on characters' actions and game events. This involves:

Object-Object Edges
Attributes: Compatibility (e.g., items that are parts of a set), combinability (e.g., items that can be used together like ingredients in a recipe), conflict (e.g., items that negate each other’s effects).
Purpose: These edges can represent how different objects can interact or be used together, enhancing gameplay depth.
Object-Activity Edges
Attributes: Necessity (e.g., tools required for an activity), enhancement (e.g., items that improve the effectiveness or outcomes of an activity), obstruction (e.g., items that hinder an activity).
Purpose: Reflect how objects are utilized in activities, influencing choices in gameplay related to skill development or task completion.
Activity-Activity Edges
Attributes: Synergy (e.g., activities that enhance each other, like studying then practicing a skill), conflict (e.g., activities that interfere with each other, like noisy and quiet activities happening concurrently), dependency (e.g., prerequisite activities).
Purpose: Indicate relationships between activities that might affect scheduling, planning, or the strategic development of skills and character growth.
Location-Activity Edges
Attributes: Suitability (e.g., appropriateness of a location for certain activities, such as a park for jogging), popularity (e.g., how popular an activity is at a location, which could affect social interactions or congestion), exclusivity (e.g., activities only available at specific locations).
Purpose: Show how locations support or are associated with different activities, influencing character decisions about where to go for various tasks.
Location-Location Edges
Attributes: Proximity (e.g., how close locations are to each other), connectivity (e.g., ease of traveling between locations, such as direct bus routes), rivalry (e.g., locations competing for the same visitors or resources).
Purpose: These edges can help model travel decisions, area popularity, and strategic choices regarding where characters spend their time.
Implementing These Interactions in the Graph
To effectively use these interactions within the game, you would set up your graph database to store and update these relationships dynamically. Here’s how you might approach this:

Dynamic Interaction Updates: As characters use objects in various activities or visit different locations, update the object-activity and location-activity edges to reflect changes in necessity, enhancement, or suitability.
Real-Time Feedback: Use feedback from these interactions to adjust the attributes dynamically. For example, if an activity at a location becomes extremely popular, increase the congestion attribute, which might affect future decisions by characters to visit.
Complex Decision-Making: Integrate these detailed relationships into the decision-making algorithms for GOAP and utility evaluations. For example, consider object necessity and activity synergy when planning a character’s daily schedule or career path.

Adding/Removing Nodes and Edges: As new characters, locations, or items are introduced or removed.
Updating Attributes: As characters develop or relationships evolve.
By systematically defining these elements, we can ensure that the graph not only represents the complexity of the game world accurately but also supports the AI’s ability to make nuanced decisions. This structure will serve as the foundation for implementing the graph-based analyses and decision-making processes

3. GOAP System and Graph Analysis
Where it happens: goap_system.py and graph_manager.py
What happens: The GOAP planner uses the graph to analyze relationships and preferences, and formulates a plan consisting of a sequence of actions that maximize the character’s utility for the day.
"""

import networkx as nx


import networkx as nx
from networkx.algorithms import community


# Improved Piecewise Linear Approximation with Caching
@lru_cache(maxsize=1024)
def sigmoid_relationship_scale_piecewise_cached(
    days_known, max_days=1000, steepness=0.01
):
    days_known = max(0, days_known)
    mid_point = max_days / 2
    if days_known < mid_point * 0.1:
        return 0  # Approximated as 0 for very early days
    elif days_known < mid_point * 0.25:
        return (days_known / (mid_point * 0.25)) * 25  # Linear increase to 25
    elif days_known < mid_point:
        return (
            25 + ((days_known - mid_point * 0.25) / (mid_point * 0.75)) * 25
        )  # Linear increase to 50
    elif days_known < mid_point * 1.5:
        return (
            50 + ((days_known - mid_point) / (mid_point * 0.5)) * 50
        )  # Linear increase to 100
    else:
        return 100  # Approximated as 100 for late days


# Define the optimized original sigmoid function with caching
@lru_cache(maxsize=1024)
def cached_sigmoid_relationship_scale_optimized(
    days_known, max_days=1000, steepness=0.01
):
    days_known = max(0, days_known)
    mid_point = max_days / 2
    exponent = -steepness * (days_known - mid_point)
    exp_value = np.exp(exponent)
    sigmoid_value = 1 / (1 + exp_value)
    return min(sigmoid_value * 100, 100)


def cached_sigmoid_relationship_scale_optimized_b(
    days_known, max_days=1000, steepness=0.01, b=0.666
):
    days_known = max(0, days_known)
    mid_point = max_days / 2
    exponent1 = -steepness * (days_known - mid_point)
    exponent2 = -steepness * (days_known - mid_point) - b
    exp_value1 = np.exp(exponent1)
    exp_value2 = np.exp(exponent2)
    sigmoid_value1 = 1 / (1 + exp_value1)
    sigmoid_value2 = 1 / (1 + exp_value2)
    average_sigmoid = 0.5 * (sigmoid_value1 + sigmoid_value2)
    return min(average_sigmoid * 100, 100)


# Define the optimized approximate sigmoid function with caching
@lru_cache(maxsize=1024)
def cached_sigmoid_relationship_scale_approx_optimized(
    days_known, max_days=1000, steepness=0.01
):
    days_known = max(0, days_known)
    mid_point = max_days / 2
    x = steepness * (days_known - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return min(sigmoid_value * 100, 100)


@lru_cache(maxsize=1024)
def cached_sigmoid_raw_approx_optimized(motive_value, max_value=10.0, steepness=0.01):
    motive_value = max(0, motive_value)
    mid_point = max_value / 2
    x = steepness * (motive_value - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return sigmoid_value


# Define the optimized approximate sigmoid function with caching
@lru_cache(maxsize=1024)
def cached_sigmoid_motive_scale_approx_optimized(
    motive_value, max_value=10.0, steepness=0.01
):
    motive_value = max(0, motive_value)
    mid_point = max_value / 2
    x = steepness * (motive_value - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return min(sigmoid_value * max_value, max_value)


@lru_cache(maxsize=1024)
def tanh_scaling(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return scaled_value * (data_max - data_min) / 2.0 + (data_max + data_min) / 2.0


@lru_cache(maxsize=1024)
def tanh_scaling_b(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return (data_max - data_min) / 2.0 * scaled_value


@lru_cache(maxsize=1024)
def tanh_scaling_raw(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return scaled_value


# version of tanh scaling coded manually without using the math.tanh function
def tanh_scaling_manual(x, data_max, data_min, data_avg):
    data_range = data_max - data_min
    a = data_range / 2.0
    centered_value = x - data_avg
    scaled_value = (math.exp(centered_value / a) - math.exp(-centered_value / a)) / (
        math.exp(centered_value / a) + math.exp(-centered_value / a)
    )
    return scaled_value


# Example interaction types and their weights


class StdCache:
    def __init__(self, refresh_interval_seconds):
        self.refresh_interval_seconds = refresh_interval_seconds
        self.cached_std = None
        self.last_calculation_time = None

    def cache_std(self, std):
        # Recalculate the standard deviation
        self.cached_std = std
        self.last_calculation_time = datetime.now().timestamp()

    def get_std(self):
        current_time = datetime.now().timestamp()

        # Check if we need to recalculate the standard deviation
        if (
            self.cached_std is None
            or self.last_calculation_time is None
            or current_time - self.last_calculation_time > self.refresh_interval_seconds
        ):
            return None
        else:

            return self.cached_std

    def update_data(self):
        # Invalidate the cache
        self.cached_std = None
        self.last_calculation_time = None


# Function to calculate interaction frequency
def calculate_interaction_frequency_combined(
    interaction_log, current_time, window_size=100, decay_factor=0.01
):
    # Interaction log format: {timestamp: (interaction_weight, emotional_impact_before, emotional_impact_after, timestamp_end)}
    # Example: {1627680000: ('significant', 0.5215, 0.5915), 1627681000: ('casual', -0.2, -0.15)}

    # Initialize interaction counts
    interaction_counts = defaultdict(int)

    # Calculate the weighted sum and moving average within the window
    total_weighted_interactions = 0
    recent_interactions_count = 0
    time_diffs = []

    for interaction_time, (
        interaction_weight,
        emotional_impact_before,
        emotional_impact_after,
        timestamp_end,
    ) in interaction_log.items():
        # convert time_diff to days
        time_diff = abs((current_time - interaction_time) / (24 * 3600))
        time_length = abs((timestamp_end - interaction_time) / (24 * 3600))
        # use a modulus to fit the time_length into into a weight between 0 and 1 representing 4 tiers of interaction length ("casual", "short", "medium", "long")
        time_length = time_length % 4
        # use the time_length to calculate the weight of the interaction
        time_weight = time_length * 0.25
        if time_diff <= window_size:
            recent_interactions_count += 1
        time_diffs.append(time_diff)
        decay_weight = np.exp(-decay_factor * time_diff)
        total_weighted_interactions += decay_weight * time_weight

    # Calculate moving average interaction frequency
    moving_average_frequency = recent_interactions_count / window_size
    # Normalize total_weighted_interactions
    max_time_weight = 1  # Since time_weight ranges from 0 to 1
    max_possible_weighted_interactions = window_size * max_time_weight
    normalized_weighted_interactions = (
        total_weighted_interactions / max_possible_weighted_interactions
    )
    # Combine the weighted sum and moving average
    combined_frequency = (total_weighted_interactions + moving_average_frequency) / 2

    return combined_frequency


def fast_decay_effect(target_value, input_value, decay_rate, elapsed_time):
    """Apply linear decay to an input value towards a target value."""
    for i in range(int(elapsed_time)):
        decay_rate = decay_rate + (i * 0.01)
        input_value = input_value * (1 - decay_rate)
        if (input_value - target_value) > 0 and input_value <= target_value:
            break
        elif (input_value - target_value) < 0 and input_value >= target_value:
            break
    return input_value


def decay_effect(target_value, input_value, decay_rate, elapsed_time):
    """Apply decay to a value over time, decaying towards a target value.
    Args:
        target_value (float): The target value to decay towards.
        input_value (float): The current value to decay.
        decay_rate (float): The rate of decay.
        elapsed_time (float): The elapsed time since the last interaction in days.

    Returns:
        float: The decayed value.

    """
    if input_value > target_value:
        # Calculate the decay multiplier
        decay_multiplier = (1 - decay_rate) ** elapsed_time

        # Decay the value towards the target value
        decayed_value = target_value + (input_value - target_value) * decay_multiplier
    elif input_value < target_value:
        # Calculate the decay multiplier
        decay_multiplier = (1 + decay_rate) ** elapsed_time

        # Decay the value towards the target value
        decayed_value = target_value + (input_value - target_value) * decay_multiplier
    else:
        decayed_value = input_value

    return decayed_value


def linear_decay_effect(target_value, input_value, decay_rate, elapsed_time):
    """Apply linear decay to a value over time towards a target value."""
    decay_amount = decay_rate * elapsed_time
    if input_value > target_value:
        new_value = input_value - decay_amount
        return max(target_value, new_value)
    elif input_value < target_value:
        new_value = input_value + decay_amount
        return min(target_value, new_value)


def logarithmic_decay_effect(target_value, input_value, decay_rate, elapsed_time):
    """Apply logarithmic decay to a value over time towards a target value."""
    decay_multiplier = math.log1p(elapsed_time) * decay_rate
    if input_value > target_value:
        new_value = max(target_value, input_value - decay_multiplier)
    elif input_value < target_value:
        new_value = min(target_value, input_value + decay_multiplier)
    return new_value


def calculate_relationship_type(
    char1, char2, emotional_impact, interaction_frequency, strength, trust, historical
):
    """
    Determine the relationship type using a more complex model with weighted and interacting factors.
    """
    # Calculate the base relationship type based on emotional impact
    if emotional_impact > 0.5:
        base_type = "positive"
    elif emotional_impact < -0.5:
        base_type = "negative"
    else:
        base_type = "neutral"

    # Adjust the relationship type based on interaction frequency
    if interaction_frequency > 0.5:
        if base_type == "positive":
            final_type = "significant"
        elif base_type == "negative":
            final_type = "strained"
        else:
            final_type = "casual"

    # Adjust the relationship type based on strength and trust
    if strength > 0.5:
        if base_type == "positive":
            final_type = "close"
        elif base_type == "negative":
            final_type = "hostile"
        else:
            final_type = "professional"

    if trust > 0.5:
        if base_type == "positive":
            final_type = "trusted"
        elif base_type == "negative":
            final_type = "distrusted"
        else:
            final_type = "acquaintance"

    # Adjust the relationship type based on historical interactions
    if historical > 0.5:
        if base_type == "positive":
            final_type = "loyal"
        elif base_type == "negative":
            final_type = "grudge"
        else:
            final_type = "familiar"

    if char1.job.location == char2.job.location:
        if base_type == "positive":
            final_type = "colleague"
        elif base_type == "negative":
            final_type = "professional rival"
        else:
            final_type = "associate"

    if char1.home == char2.home:
        if base_type == "positive":
            final_type = "roommate"
        elif base_type == "negative":
            final_type = "adversary"
        else:
            final_type = "live-in neighbor"

    return final_type


def decay_emotional_impact(
    self,
    impact_value,
    interaction_type,
    historical,
    emotional_impact,
    base_decay_rate=0.01,
    elapsed_time=1,
    interaction_log={},
):
    # Interaction log format: {timestamp: (interaction_weight, emotional_impact_before, emotional_impact_after, timestamp_end)
    # Example: {1627680000: ('significant', 0.5215, 0.5915), 1627681000: ('casual', -0.2, -0.15)}
    stability_factor = (
        historical / 100
    )  # As historical approaches 100, stability_factor approaches 1
    decay_rate = base_decay_rate * (
        1 - stability_factor
    )  # Adjust decay rate based on historical interactions
    current_impact = emotional_impact
    if interaction_log != {}:
        # sort the interaction log by time, earliest to most recent
        interaction_log = dict(
            sorted(interaction_log.items(), key=lambda item: item[0])
        )
        for interaction_time, (
            interaction_weight,
            emotional_impact_before,
            emotional_impact_after,
            timestamp_end,
        ) in interaction_log.items():
            elapsed_time = (time.time() - interaction_time) / (
                24 * 3600
            )  # Convert elapsed time to days
            interaction_impact = emotional_impact_after - emotional_impact_before
            target = (
                emotional_impact - interaction_impact
                if interaction_impact > 0
                else emotional_impact + interaction_impact
            )
            if interaction_weight <= 1:
                current_impact = fast_decay_effect(
                    target, current_impact, decay_rate, elapsed_time
                )
            elif interaction_weight >= 1 and interaction_weight <= 2:

                current_impact = linear_decay_effect(
                    target, current_impact, decay_rate, elapsed_time
                )
            elif interaction_weight >= 2 and interaction_weight <= 3:
                current_impact = decay_effect(
                    target, current_impact, decay_rate, elapsed_time
                )
            elif interaction_weight >= 3:
                current_impact = logarithmic_decay_effect(
                    target, current_impact, decay_rate, elapsed_time
                )
    return current_impact


def update_emotional_impact(
    impact_value,
    interaction_type,
    historical,
    emotional_impact,
    trust,
    impact_rating,
    base_decay_rate=0.01,
):
    """Update emotional impact with decay over time."""
    #     High Trust Amplifies Positive Impact:

    # When trust is high, positive interactions tend to have a more significant positive emotional impact, as characters are more likely to value and be affected by the actions of trusted companions.
    # High Trust Mitigates Negative Impact:

    # Conversely, when trust is high, negative interactions may have a reduced negative impact. Characters might be more forgiving or understanding, assuming good intentions or past positive history as reasons for the negative behavior.
    # Low Trust Amplifies Negative Impact:

    # When trust is low, negative interactions can have a more pronounced negative impact. The characters may be more sensitive to slights or perceived betrayals, leading to a deeper emotional response.
    # Low Trust Reduces Positive Impact:

    # Positive interactions may also have a less significant positive impact when trust is low, as characters may be skeptical of the motives behind the actions. They might interpret positive gestures with suspicion or doubt, diminishing the emotional effect.

    current_impact = emotional_impact
    stability_factor = historical / 100
    impact_value *= (
        1 - stability_factor
    )  # Adjust impact value based on historical interactions

    if interaction_type == "positive":
        current_impact += impact_value * (1 + trust)
    elif interaction_type == "negative":
        current_impact -= impact_value * (1 - trust)
    else:  # Neutral interaction
        current_impact *= 0.95  # Slight decay for neutral interaction

    emotional_impact = max(-1, min(1, current_impact))
    last_interaction_time = datetime.now().timestamp()

    return emotional_impact


def update_trust(
    trust_increment=0.01,
    interaction_type="proximity",
    betrayal=False,
    impact_rating=1,
    historical=50,
    personality_traits={},
    trust=0.5,
):
    stability_factor = historical / 100
    if betrayal:
        trust -= trust_increment * (1 - stability_factor)
    elif interaction_type == "cooperative":
        trust += (
            trust_increment
            * (1 + personality_traits.get("agreeableness", 0.5))
            * stability_factor
        )
    elif interaction_type == "proximity":
        trust += trust_increment * 0.5 * stability_factor
    trust = max(0, min(1, trust))
    return trust


def cooperative_activity(self, benefit_shared):
    """Update trust based on cooperative activity success."""
    increment = benefit_shared * 0.2  # More weight to cooperative success
    self.update_trust(increment, "cooperative")


def betrayal_event(self, severity):
    """Handle a betrayal event with severity consideration."""
    decrement = severity * 0.3
    self.update_trust(decrement, "betrayal", betrayal=True)


character_attributes = [
    "name",
    "age",
    "pronouns",
    "job",
    "health_status",
    "hunger_level",
    "wealth_money",
    "mental_health",
    "social_wellbeing",
    "happiness",
    "shelter",
    "stability",
    "luxury",
    "hope",
    "success",
    "control",
    "job_performance",
    "beauty",
    "community",
    "material_goods",
    "friendship_grid",
    "recent_event",
    "long_term_goal",
    "inventory",
    "home",
    "personality_traits",
    "motives",
]


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
        self.std_cache = StdCache(
            refresh_interval_seconds=60
        )  # Refresh every 60 seconds
        self.unique_graph_id = uuid.uuid4()

        self.dp_cache = {}
        self.characters = {}
        self.locations = {}
        self.objects = {}
        self.events = {}
        self.activities = {}
        self.jobs = {}
        self.type_to_dict_map = {
            "character": self.characters,
            "location": self.locations,
            "object": self.objects,
            "event": self.events,
            "activity": self.activities,
            "job": self.jobs,
        }
        self.character_attributes = None
        self.location_attributes = None
        self.event_attributes = None
        self.object_attributes = None
        self.activity_attributes = None
        self.job_attributes = None
        self.ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "eq": operator.eq,
            "ge": operator.ge,
            "le": operator.le,
            "ne": operator.ne,
        }
        self.symb_map = {
            ">": "gt",
            "<": "lt",
            "==": "eq",
            ">=": "ge",
            "<=": "le",
            "!=": "ne",
        }

        self.G = self.initialize_graph()
        
        # Initialize social model with this graph manager as world state
        self.social_model = SocialModel(world_state=self)
        
        self.__initialized = True

    def initialize_graph(self):
        self.G = (
            nx.MultiDiGraph()
        )  # Using MultiDiGraph for directional and multiple edges
        return self.G

    def get_location(self, name):
        return self.locations.get(name, (0, 0))

    def update_location(self, name, location):
        self.characters[name]["coordinates_location"] = location

    def add_obstacle(self, obstacle_location):
        self.obstacles.append(obstacle_location)

    def get_obstacles(self):
        return self.obstacles

    def directional(self, current):
        x, y = current
        # 8-directional movement (up, down, left, right, and diagonals)
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
        ]

    def cost(self, current, next):
        return 1  # Uniform cost for simplicity

    def node_type_resolver(self, node):
        """
        Returns the root class instance of a node based on its attributes.
        Args:
            node (Node): The node to resolve.
        Returns:
            str: The resolved type of the node.
        """
        if "type" in node:
            if node["type"] == "character":
                return self.characters[node["name"]]
            elif node["type"] == "location":
                return self.locations[node["name"]]
            elif node["type"] == "event":
                return self.events[node["name"]]
            elif node["type"] == "item" or node["type"] == "object":
                return self.objects[node["name"]]
            elif node["type"] == "activity":
                return self.activities[node["name"]]
            elif node["type"] == "job":
                return self.jobs[node["name"]]
        return None

    # Node Addition Methods
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
            energy_level=char.energy,
            relationships={},  # Stores additional details about relationships
            emotional_state={},
            coordinates_location=char.coordinates_location,
            resources=char.inventory,
            needed_resources=char.needed_items,  # This is a list of tuples, each tuple is (dict of item requirements, quantity needed).
            # The dict of item requirements is composed of various keys like item_type, value, usability, sentimental_value, trade_value, scarcity, coordinates_location, name, etc. from either the node or the root class instance.
            name=char.name,
            mood=char.current_mood,
            wealth_money=wealth_money,
            health_status=char.health_status,
            hunger_level=char.hunger_level,
            mental_health=char.mental_health,
            social_wellbeing=char.social_wellbeing,
            shelter=char.shelter,
            has_investment=has_investment,
        )
        logging.debug(
            f"Added character node: \n {char} \n\n with attributes:\n {self.G.nodes[char]}\n"
        )

    def add_location_node(self, loc: Location):
        from tiny_locations import Location

        if len(self.locations) == 0:
            self.location_attributes = loc.to_dict().keys()
        self.locations[loc.name] = loc
        self.G.add_node(
            loc,
            type="location",
            popularity=loc.popularity,
            activities_available=loc.activities_available,
            accessible=loc.accessible,
            security=loc.security,
            coordinates_location=loc.coordinates_location,
            name=loc.name,
            threat_level=loc.threat_level,
            visit_count=loc.visit_count,
        )

    def add_event_node(self, event: Event):
        from tiny_event_handler import Event

        if len(self.events) == 0:
            self.event_attributes = event.to_dict().keys()
        self.events[event.name] = event
        self.G.add_node(
            event,
            type="event",
            event_type=event.type,
            date=event.date,
            importance=event.importance,
            impact=event.impact,
            required_items=event.required_items,
            coordinates_location=event.coordinates_location,
            name=event.name,
        )

    def add_object_node(self, obj: ItemObject):
        if len(self.objects) == 0:
            self.object_attributes = obj.to_dict().keys()
        self.objects[obj.name] = obj
        self.G.add_node(
            obj,
            type="object",
            item_type=obj.item_type,
            item_subtype=obj.item_subtype,
            value=obj.value,
            usability=obj.usability,
            # scarcity=obj.scarcity,
            coordinates_location=obj.coordinates_location,
            name=obj.name,
            ownership_history=obj.ownership_history,
            type_specific_attributes=(
                obj.get_type_specific_attributes()
                if obj.type_specific_attributes
                else {}
            ),
        )

    def add_stock_node(self, stock: Stock):

        if len(self.stocks) == 0:
            self.stock_attributes = stock.to_dict().keys()
        self.stocks[stock.name] = stock
        self.G.add_node(
            stock,
            type="stock",
            stock_type=stock.stock_type,
            stock_description=stock.stock_description,
            value=stock.value,
            scarcity=stock.scarcity,
            name=stock.name,
            ownership_history=stock.ownership_history,
            type_specific_attributes=(
                stock.get_type_specific_attributes()
                if stock.type_specific_attributes
                else {}
            ),
        )

    def add_activity_node(self, act: Action):
        Action = importlib.import_module("actions").Action

        if len(self.activities) == 0:
            self.activity_attributes = act.to_dict().keys()
        self.activities[act.name] = act
        self.G.add_node(
            act,
            type="activity",
            related_skills=act.related_skills,
            name=act.name,
            required_items=act.required_items,
            preconditions=act.preconditions,
            effects=act.effects,
        )

    def add_job_node(self, job: Job):
        if len(self.jobs) == 0:
            self.job_attributes = job.to_dict().keys()
        self.jobs[job.job_name] = job
        self.G.add_node(
            job,
            type="job",
            required_skills=job.job_skills,
            location=job.location,
            salary=job.job_salary,
            job_title=job.job_title,
            available=job.available,
            name=job.job_name,
        )

    def add_dict_of_nodes(self, nodes_dict):
        for node_type, nodes in nodes_dict.items():
            if node_type == "characters":
                for char in nodes:
                    self.add_character_node(char)
            elif node_type == "locations":
                for loc in nodes:
                    self.add_location_node(loc)
            elif node_type == "events":
                for event in nodes:
                    self.add_event_node(event)
            elif node_type == "objects":
                for obj in nodes:
                    self.add_object_node(obj)
            elif node_type == "activities":
                for act in nodes:
                    self.add_activity_node(act)
            elif node_type == "jobs":
                for job in nodes:
                    self.add_job_node(job)

    # Edge Addition Methods with Detailed Attributes
    # Character-Character
    def add_character_character_edge(
        self,
        char1,
        char2,
        relationship_type=0,
        strength=0,
        emotional_impact=0,
        trust=0,
        impact_factor=1,
        impact_value=0.01,
        edge_type="character_character",
        interaction_log={},
        romance_compatibility=0,
        romanceable=True,
        romance_interest=False,
        romance_value=0,
    ):
        # Interaction log format: {timestamp: (interaction_weight, emotional_impact_before, emotional_impact_after, timestamp_end)}
        # Example: {1627680000: (3.1, 0.5215, 0.5915), 1627681000: (0.6, -0.2, -0.15)}

        trust = update_trust()
        emotional_impact_before = emotional_impact
        emotional_impact_after = update_emotional_impact(
            impact_value,
            "positive" if (impact_value > 0) else "negative",
            0,
            emotional_impact_before,
            trust,
            impact_factor,
        )
        historical = 0
        if interaction_log == {}:
            interaction_frequency = 0
            interaction_log = {
                datetime.now().timestamp(): (
                    impact_factor,
                    emotional_impact_before,
                    emotional_impact_after,
                    datetime.now().timestamp(),
                )
            }
            relationship_type = calculate_relationship_type(
                char1,
                char2,
                emotional_impact_after,
                interaction_frequency,
                strength,
                trust,
                historical,
            )
        else:
            interaction_frequency = calculate_interaction_frequency_combined(
                interaction_log, time.time()
            )
            # find earliest timestamp in interaction_log
            earliest_timestamp = min(interaction_log.keys())
            # calculate how many days have passed since the earliest interaction
            days_known = (time.time() - earliest_timestamp) / (24 * 3600)
            historical = cached_sigmoid_relationship_scale_approx_optimized(days_known)

        if self.G.has_edge(char1, char2):

            self.update_character_character_edge(
                char1,
                char2,
                impact_factor,
                impact_value,
                relationship_type,
                strength,
                historical,
                emotional_impact_after,
                interaction_frequency,
                trust,
                edge_type,
                interaction_log,
            )
            return
        romance_compatibility = self.calculate_romance_compatibility(
            char1, char2, historical
        )
        romance_interest = self.calculate_romance_interest(
            char1,
            char2,
            romance_compatibility,
            romance_value,
            relationship_type,
            strength,
            historical,
            trust,
            interaction_frequency,
            emotional_impact_after,
        )
        self.G.add_edge(
            char1,
            char2,
            type=edge_type,
            relationship_type=relationship_type,  # This is a string representing the type of relationship (e.g., friend, family, colleague, antagonist, rival, enemy).
            strength=strength,  # This is a float between 0 and 1 representing the strength of the relationship. 0 is weak, 1 is strong.
            historical=historical,  # This is a value representing the length of the relationship in time. It
            emotional=emotional_impact_after,  # This is a float between -1 and 1 representing the emotional impact of the relationship over time. -1 is negative, 0 is neutral, 1 is positive.
            interaction_frequency=interaction_frequency,  # This is a float representing how often the characters interact with each other.
            interaction_count=len(interaction_log),
            key=edge_type,
            trust=trust,  # This is a float between 0 and 1 representing the level of trust between the characters. 0 is no trust, 1 is complete trust.
            distance=dist(char1.coordinates_location, char2.coordinates_location),
            interaction_cost=self.calculate_char_char_edge_cost(
                char1,
                char2,
                relationship_type,
                strength,
                historical,
                trust,
                emotional_impact,
                interaction_frequency,
            ),
            dist_cost=self.calc_distance_cost(
                dist(char1.coordinates_location, char2.coordinates_location),
                char1,
                char2,
            ),
            interaction_log=interaction_log,
        )

    def update_character_character_edge(
        self,
        char1,
        char2,
        impact_factor,
        impact_value,
        relationship_type=None,
        strength=None,
        historical=None,
        emotional_impact=None,
        trust=None,
        edge_type="character_character",
        interaction_log=None,  # dict of interactions like: {day: (interaction_weight, emotional_impact)}
    ):

        interaction_frequency = calculate_interaction_frequency_combined(
            interaction_log, time.time()
        )  # find earliest timestamp in interaction_log
        earliest_timestamp = min(interaction_log.keys())
        # calculate how many days have passed since the earliest interaction
        days_known = (time.time() - earliest_timestamp) / (24 * 3600)
        historical = cached_sigmoid_relationship_scale_approx_optimized(days_known)
        if not self.G.has_edge(char1, char2):

            self.add_character_character_edge(
                char1,
                char2,
                relationship_type,
                strength,
                historical,
                emotional_impact,
                trust,
                impact_factor=impact_factor,
                impact_value=impact_value,
                interaction_log=interaction_log,
            )
            return
        emotional_impact_before = emotional_impact
        emotional_impact_after = update_emotional_impact(
            impact_value,
            "positive" if (impact_value > 0) else "negative",
            emotional_impact,
            trust,
            impact_factor,
        )

        if relationship_type:
            self.G[char1][char2]["relationship_type"] = relationship_type
        if strength:
            self.G[char1][char2]["strength"] = strength
        if historical:
            self.G[char1][char2]["historical"] = historical
        if emotional_impact_after:
            self.G[char1][char2]["emotional_impact"] = emotional_impact
        if interaction_frequency:
            self.G[char1][char2]["interaction_frequency"] = interaction_frequency
        if trust:
            self.G[char1][char2]["trust"] = trust
        if interaction_log:
            self.G[char1][char2]["interaction_log"] = interaction_log

    def calculate_dynamic_weights(self, historical):
        """
        Calculate dynamic weights for personality traits.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.calculate_dynamic_weights(historical)

    def calculate_romance_compatibility(self, char1, char2, historical):
        """
        Calculate romantic compatibility between two characters.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.calculate_romance_compatibility(char1, char2, historical)

    def calculate_romance_interest(
        self,
        char1,
        char2,
        romance_compat,
        romance_value,
        relationship_type,
        strength,
        historical,
        trust,
        interaction_frequency,
        emotional_impact,
    ):
        """
        Calculate romance interest between two characters.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.calculate_romance_interest(
            char1, char2, romance_compat, romance_value, relationship_type,
            strength, historical, trust, interaction_frequency, emotional_impact
        )

    def calc_distance_cost(self, distance, char1, char2):
        # Distance cost calculation
        # Distance cost is calculated as a function of the distance between two characters.
        # The cost is higher for characters that are further apart.
        # The cost is lower for characters that are closer together.
        # The cost is also influenced by the characters' energy levels, with higher energy levels resulting in lower costs.
        # The cost is also influenced by the characters' relationship strength, with stronger relationships resulting in lower costs.
        # The cost is also influenced by the characters' trust levels, with higher trust levels resulting in lower costs.
        # The cost is also influenced by the characters' emotional states, with more positive emotional states resulting in lower costs.
        # The cost is also influenced by the characters' interaction frequencies, with higher interaction frequencies resulting in lower costs.
        # The cost is also influenced by the characters' historical relationship lengths, with longer relationships resulting in lower costs.
        # The cost is also influenced by the characters' relationship types, with more positive relationship types resulting in lower costs.
        # The cost is also influenced by the characters' relationship impact factors, with higher impact factors resulting in lower costs.
        # The cost is also influenced by the characters' relationship emotional impacts, with more positive emotional impacts resulting in lower costs.
        # The cost is also influenced by the characters' relationship trust levels, with higher trust levels resulting in lower costs.
        # The cost is also influenced by the characters' relationship strengths, with stronger relationships resulting in lower costs.
        # The cost is also influenced by the characters' relationship historical lengths, with longer relationships resulting in lower costs.
        # The cost is also influenced by the characters' relationship interaction frequencies, with higher interaction frequencies resulting in lower costs.
        # The cost is also influenced by the characters' relationship interaction logs, with more positive interactions resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance compatibilities, with higher compatibilities resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance interests, with more positive interests resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance values, with higher values resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance compatibilities, with higher compatibilities resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance interests, with more positive interests resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance values, with higher values resulting in lower costs.
        # The cost is also influenced by the characters' relationship romance compatibilities, with higher compatibilities resulting in lower costs.
        assert self.G.has_node(char1) and self.G.has_node(char2)
        if not self.G.has_edge(char1, char2):
            return distance * 1.0
        return (
            distance
            * (1 - char1.energy)
            * (1 - char2.energy)
            * (1 - self.G[char1][char2]["strength"])
            * (1 - self.G[char1][char2]["trust"])
            * (1 - self.G[char1][char2]["emotional"])
            * (1 - self.G[char1][char2]["interaction_frequency"])
            * (1 - self.G[char1][char2]["historical"])
            * (1 - self.G[char1][char2]["relationship_type"])
            * (1 - self.G[char1][char2]["impact_factor"])
            * (1 - self.G[char1][char2]["emotional_impact"])
            * (1 - self.G[char1][char2]["trust"])
            * (1 - self.G[char1][char2]["strength"])
            * (1 - self.G[char1][char2]["historical"])
            * (1 - self.G[char1][char2]["interaction_frequency"])
            * (1 - self.G[char1][char2]["interaction_log"])
            * (1 - self.G[char1][char2]["romance_compatibility"])
            * (1 - self.G[char1][char2]["romance_interest"])
            * (1 - self.G[char1][char2]["romance_value"])
            * (1 - self.G[char1][char2]["romance_compatibility"])
            * (1 - self.G[char1][char2]["romance_interest"])
            * (1 - self.G[char1][char2]["romance_value"])
            * (1 - self.G[char1][char2]["romance_compatibility"])
            * (1 - self.G[char1][char2]["romance_interest"])
            * (1 - self.G[char1][char2]["romance_value"])
        )

    def calculate_locations_between_nodes(self, node1, node2):
        locs = [n for n in self.G.nodes if self.G.nodes[n]["type"] == "location"]
        locs_between = []
        # for loc in locs:
        #     if loc not in self.locations:
        #         self.add_location_node(loc)
        #     if loc != node1 and loc != node2:
        #         if self.G.has_edge(node1, loc) and self.G.has_edge(loc, node2):
        #             locs_between.append(loc)
        return locs_between

    def calc_char_char_edge_cost(self, node, other_node, attribute_dict):
        # Find edge in question
        return self.calculate_char_char_edge_cost(
            node,
            other_node,
            attribute_dict["relationship_type"],
            attribute_dict["strength"],
            attribute_dict["historical"],
            attribute_dict["emotional_impact"],
            attribute_dict["interaction_frequency"],
            attribute_dict["trust"],
        )

    def calculate_edge_cost(self, node, other_node, attribute_dict=None):
        if attribute_dict is None:
            attribute_dict = self.G[node][other_node]
        if attribute_dict["edge_type"] == "character_character":
            return self.calc_char_char_edge_cost(
                node,
                other_node,
                attribute_dict["relationship_type"],
                attribute_dict["strength"],
                attribute_dict["historical"],
                attribute_dict["emotional_impact"],
                attribute_dict["interaction_frequency"],
                attribute_dict["trust"],
            )
        if attribute_dict["edge_type"] == "character_location":
            return self.calculate_char_loc_edge_cost(
                node,
                other_node,
                attribute_dict["frequency_of_visits"],
                attribute_dict["last_visit"],
                attribute_dict["favorite_activities"],
                attribute_dict["ownership_status"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "character_object":
            return self.calculate_char_obj_edge_cost(
                node,
                other_node,
                attribute_dict["ownership_status"],
                attribute_dict["usage_frequency"],
                attribute_dict["sentimental_value"],
                attribute_dict["last_used_time"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "character_event":
            return self.calculate_char_event_edge_cost(
                node,
                other_node,
                attribute_dict["participation_status"],
                attribute_dict["role"],
                attribute_dict["impact_on_character"],
                attribute_dict["emotional_outcome"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "character_activity":
            return self.calculate_char_act_edge_cost(
                node,
                other_node,
                attribute_dict["engagement_level"],
                attribute_dict["skill_improvement"],
                attribute_dict["activity_frequency"],
                attribute_dict["motivation"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "location_location":
            return self.calculate_loc_loc_edge_cost(
                node,
                other_node,
                attribute_dict["proximity"],
                attribute_dict["connectivity"],
                attribute_dict["rivalry"],
                attribute_dict["trade_relations"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "location_item":
            return self.calculate_loc_item_edge_cost(
                node,
                other_node,
                attribute_dict["item_presence"],
                attribute_dict["item_relevance"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "location_event":
            return self.calculate_loc_event_edge_cost(
                node,
                other_node,
                attribute_dict["event_occurrence"],
                attribute_dict["location_role"],
                attribute_dict["capacity"],
                attribute_dict["preparation_level"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "location_activity":
            return self.calculate_loc_act_edge_cost(
                node,
                other_node,
                attribute_dict["suitability"],
                attribute_dict["popularity"],
                attribute_dict["exclusivity"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "item_item":
            return self.calculate_item_item_edge_cost(
                node,
                other_node,
                attribute_dict["compatibility"],
                attribute_dict["combinability"],
                attribute_dict["conflict"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "item_activity":
            return self.calculate_item_act_edge_cost(
                node,
                other_node,
                attribute_dict["necessity"],
                attribute_dict["enhancement"],
                attribute_dict["obstruction"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "event_activity":
            return self.calculate_event_act_edge_cost(
                node,
                other_node,
                attribute_dict["synergy"],
                attribute_dict["conflict"],
                attribute_dict["dependency"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "event_item":
            return self.calculate_event_item_edge_cost(
                node,
                other_node,
                attribute_dict["necessity"],
                attribute_dict["relevance"],
                attribute_dict["distance"],
            )
        if attribute_dict["edge_type"] == "job_activity":
            return self.calculate_job_act_edge_cost(
                node,
                other_node,
                attribute_dict["skill_match"],
                attribute_dict["risk_level"],
                attribute_dict["reward_level"],
                attribute_dict["distance"],
            )

    def calculate_reward(self, node, other_node, attribute_dict=None):

        if attribute_dict is None:
            attribute_dict = self.G[node][other_node]
        if attribute_dict["edge_type"] == "job_activity":
            if self.G.get_edge_data(node, other_node)["reward_level"]:
                return self.G.get_edge_data(node, other_node)["reward_level"]
            else:
                if self.get_node(other_node)["type"] == "activity":
                    return self.get_node(other_node).apply_effects()
        elif attribute_dict["edge_type"] == "event_activity":
            if self.G.get_edge_data(node, other_node)["reward_level"]:
                return self.G.get_edge_data(node, other_node)["reward_level"]
            else:
                if self.get_node(other_node)["type"] == "activity":
                    return self.get_node(other_node).apply_effects()
        elif isinstance(self.get_node(node), Action):
            Action = importlib.import_module("tiny_actions").Action
            return self.get_node(node).apply_effects()
        elif isinstance(self.get_node(other_node), Action):
            Action = importlib.import_module("tiny_actions").Action
            return self.get_node(other_node).apply_effects()
        else:
            return 0

    def calculate_penalty(self, node, other_node, attribute_dict=None):
        return self.calculate_edge_cost(node, other_node, attribute_dict)

    def calculate_char_char_edge_cost(
        self,
        char1,
        char2,
        relationship_type,
        strength,
        historical,
        emotional_impact,
        interaction_frequency,
        trust,
    ):
        cost = 0
        char2_traits = char2.personality_traits.to_dict()
        logging.info(f"char2_traits: {char2_traits}")
        char2_motives = char2.motives.to_dict()
        logging.info(f"char2_motives: {char2_motives}")
        for char1_trait, char1val in char1.personality_traits.to_dict().items():
            cost += abs(char1val - char2_traits[char1_trait])
            logging.info(f"cost: {cost}")
        for char1_motive, char1val in char1.motives.to_dict().items():
            if (
                relationship_type == "friend"
                or relationship_type == "family"
                or relationship_type == "colleague"
            ):
                # The closer the motives are to each other, the lower the cost
                logging.info(f"char1val.get_score(): {char1val.get_score()}")
                logging.info(f"char1_motive: {char1_motive}")
                logging.info(
                    f"char2_motive[char1_motive]: {char2_motives[char1_motive].get_score()}"
                )
                cost += (
                    abs(char1val.get_score() - char2_motives[char1_motive].get_score())
                    / strength
                )
                # The more frequent the interaction, the lower the cost
                cost += (1 / (interaction_frequency + 1)) * strength
                # The longer the historical, the lower the cost
                cost += (1 / (historical + 1)) * strength
                # The higher the emotional impact, the lower the cost
                cost += 1 / (emotional_impact + 1)
            if (
                relationship_type == "antagonist"
                or relationship_type == "rival"
                or relationship_type == "enemy"
            ):
                # The closer the motives are to each other, the higher the cost
                cost += (
                    abs(char1val.get_score() + char2_motives[char1_motive].get_score())
                    - abs(
                        char1val.get_score() - char2_motives[char1_motive].get_score()
                    )
                ) * strength
                # The longer the historical, the higher the cost
                cost += (1 / (len(historical) + 1)) * strength
                # The higher the emotional impact, the higher the cost
                cost += 1 / (emotional_impact + 1)
            #

        return cost

    def calculate_char_loc_edge_cost(
        self,
        char,
        loc,
        frequency_of_visits,
        last_visit,
        favorite_activities,
        ownership_status,
        distance,
    ):
        cost = 0
        # More frequent visits reduce cost, with diminishing returns
        cost += 1 / (frequency_of_visits + 1) ** 0.5
        # Recent visits reduce cost, with more weight on regular visits
        cost += 1 / (last_visit + 1)
        # Ownership status reduces cost, more for full ownership
        if ownership_status == "full":
            cost -= 10
        elif ownership_status == "partial":
            cost -= 5
        # Closer distance reduces cost
        cost += distance / 10
        # Favorite activities alignments reduce cost, weighted by importance
        for activity in favorite_activities:
            if activity in loc.activities_available:
                cost -= favorite_activities[activity]

        return max(cost, 0)

    def calculate_char_obj_edge_cost(
        self,
        char,
        obj,
        ownership_status,
        usage_frequency,
        sentimental_value,
        last_used_time,
        distance,
    ):
        cost = 0
        # Ownership reduces cost significantly
        if ownership_status:
            cost -= 5
        # More frequent usage reduces cost, with critical use consideration
        cost += 1 / (usage_frequency + 1) ** 0.5
        # Higher sentimental value (positive or negative) impacts cost
        cost += sentimental_value / 10 if sentimental_value >= 0 else sentimental_value
        # Recent usage reduces cost
        cost += 1 / (last_used_time + 1)
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_char_event_edge_cost(
        self,
        char,
        event,
        participation_status,
        role,
        impact_on_character,
        emotional_outcome,
        distance,
    ):
        cost = 0
        # Participation reduces cost
        if participation_status:
            cost -= 5
        # Important role reduces cost, weighted by activity level
        cost += 1 / (role + 1) ** 0.5
        # Positive impact reduces cost, consider short-term vs long-term
        cost += 1 / (impact_on_character["short_term"] + 1)
        cost += 1 / (impact_on_character["long_term"] + 1) / 2
        # Positive emotional outcome reduces cost
        cost += 1 / (emotional_outcome + 1)
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_char_act_edge_cost(
        self,
        char,
        act,
        engagement_level,
        skill_improvement,
        activity_frequency,
        motivation,
        distance,
    ):
        cost = 0
        # Higher engagement reduces cost, considering both motivations
        cost += 1 / (engagement_level + motivation + 1)
        # Skill improvement reduces cost, with diminishing returns
        cost += 1 / (skill_improvement + 1) ** 0.5
        # More frequent activity reduces cost, with diminishing returns
        cost += 1 / (activity_frequency + 1) ** 0.5
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_loc_loc_edge_cost(
        self, loc1, loc2, proximity, connectivity, rivalry, trade_relations, distance
    ):
        cost = 0
        # Closer proximity reduces cost
        cost += 1 / (proximity + 1)
        # Better connectivity reduces cost
        cost += 1 / (connectivity + 1)
        # Rivalry increases cost
        cost += rivalry * 10
        # Trade relations reduce cost
        cost -= trade_relations["volume"] / 10
        cost -= trade_relations["value"] / 10
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_loc_item_edge_cost(
        self, loc, item, item_presence, item_relevance, distance
    ):
        cost = 0
        # Presence of item reduces cost, considering abundance
        cost -= item_presence / 10
        # Higher relevance reduces cost, considering primary function
        cost += 1 / (item_relevance["primary"] + 1)
        cost += 1 / (item_relevance["secondary"] + 1) / 2
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_loc_event_edge_cost(
        self,
        loc,
        event,
        event_occurrence,
        location_role,
        capacity,
        preparation_level,
        distance,
    ):
        cost = 0
        # Frequent event occurrence reduces cost, with predictability
        cost += 1 / (event_occurrence["frequency"] + 1)
        cost += 1 / (event_occurrence["predictability"] + 1) / 2
        # Significant role of location reduces cost
        cost += 1 / (location_role + 1)
        # Higher capacity reduces cost
        cost += 1 / (capacity + 1)
        # Better preparation level reduces cost
        cost += 1 / (preparation_level + 1)
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_loc_act_edge_cost(
        self, loc, act, suitability, popularity, exclusivity, distance
    ):
        cost = 0
        # Higher suitability reduces cost
        cost += 1 / (suitability + 1)
        # Higher popularity reduces cost
        cost += 1 / (popularity + 1)
        # Higher exclusivity increases cost
        cost += exclusivity * 10
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_item_item_edge_cost(
        self, item1, item2, compatibility, combinability, conflict, distance
    ):
        cost = 0
        # Higher compatibility reduces cost, considering multiple aspects
        cost += 1 / (compatibility["functional"] + 1)
        cost += 1 / (compatibility["aesthetic"] + 1)
        # Higher combinability reduces cost, with ease and value
        cost += 1 / (combinability["ease"] + 1)
        cost += 1 / (combinability["value"] + 1) / 2
        # Higher conflict increases cost
        cost += conflict * 10
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_item_act_edge_cost(
        self, item, act, necessity, enhancement, obstruction, distance
    ):
        cost = 0
        # Higher necessity reduces cost
        cost += 1 / (necessity + 1)
        # Higher enhancement reduces cost
        cost += 1 / (enhancement + 1)
        # Higher obstruction increases cost
        cost += obstruction * 10
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_event_act_edge_cost(
        self, event, act, synergy, conflict, dependency, distance
    ):
        cost = 0
        # Higher synergy reduces cost, considering immediate and long-term
        cost += 1 / (synergy["immediate"] + 1)
        cost += 1 / (synergy["long_term"] + 1) / 2
        # Higher conflict increases cost, considering direct and indirect
        cost += conflict["direct"] * 10
        cost += conflict["indirect"] * 5
        # Higher dependency reduces cost
        cost += 1 / (dependency + 1)
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_event_item_edge_cost(
        self, event, item, necessity, relevance, distance
    ):
        cost = 0
        # Higher necessity reduces cost
        cost += 1 / (necessity + 1)
        # Higher relevance reduces cost
        cost += 1 / (relevance + 1)
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    def calculate_job_act_edge_cost(
        self, job, act, skill_match, risk_level, reward_level, distance
    ):
        cost = 0
        # Higher skill match reduces cost, considering required and beneficial
        cost += 1 / (skill_match["required"] + 1)
        cost += 1 / (skill_match["beneficial"] + 1) / 2
        # Higher risk increases cost significantly
        cost += risk_level * 15
        # Higher reward reduces cost significantly
        cost += 1 / (reward_level + 1) * 2
        # Closer distance reduces cost
        cost += distance / 10

        return max(cost, 0)

    # Character-Location
    def add_character_location_edge(
        self,
        char,
        loc,
        frequency_of_visits,
        last_visit,
        favorite_activities,
        ownership_status,
        edge_type="character_location",
    ):
        # check if the character has visited the location before
        if (frequency_of_visits > 0 or last_visit) and self.G.has_edge(char, loc):
            self.G[char][loc]["frequency_of_visits"] += 1
            self.G[char][loc]["last_visit"] = datetime.now().strftime("%Y-%m-%d")

        else:
            self.G.add_edge(
                char,
                loc,
                type=edge_type,
                frequency_of_visits=frequency_of_visits if frequency_of_visits else 1,
                last_visit=datetime.now().strftime("%Y-%m-%d"),
                favorite_activities=favorite_activities,
                ownership_status=ownership_status,
                key=edge_type,
                distance=loc.distance_to_point_from_nearest_edge(
                    char.coordinates_location
                ),
            )
        loc.visit_count += 1

    # Character-Item
    def add_character_object_edge(
        self,
        char,
        obj,
        ownership_status,
        usage_frequency,
        sentimental_value,
        last_used_time,
        edge_type="character_object",
    ):
        if ownership_status == True:
            if (
                len(self.get_item(obj.name).ownership_history) == 0
                or self.get_item(obj.name).ownership_history[-1] != char.name
            ):
                self.get_item(obj.name).ownership_history.append(char.name)
        self.G.add_edge(
            char,
            obj,
            type=edge_type,
            ownership_status=ownership_status,
            usage_frequency=usage_frequency,
            sentimental_value=sentimental_value,
            last_used_time=last_used_time.strftime("%Y-%m-%d"),
            key=edge_type,
            distance=dist(char.coordinates_location, obj.coordinates_location),
        )

    # Character-Event
    def add_character_event_edge(
        self,
        char,
        event,
        participation_status,
        role,
        impact_on_character=0,
        emotional_outcome=0,
        edge_type="character_event",
    ):
        self.G.add_edge(
            char,
            event,
            type=edge_type,
            participation_status=participation_status,
            role=role,
            impact_on_character=impact_on_character,
            emotional_outcome=emotional_outcome,
            key=edge_type,
            distance=event.location.distance_to_point_from_center(
                *event.coordinates_location
            ),
        )

    # Character-Activity
    def add_character_activity_edge(
        self,
        char,
        act,
        engagement_level,
        skill_improvement,
        activity_frequency,
        motivation,
        edge_type="character_activity",
    ):
        self.G.add_edge(
            char,
            act,
            type=edge_type,
            engagement_level=engagement_level,
            skill_improvement=skill_improvement,
            activity_frequency=activity_frequency,
            motivation=motivation,
            key=edge_type,
            distance=dist(
                char.coordinates_location,
                act.location.distance_to_point_from_center(*act.coordinates_location),
            ),
        )

        # Location-Location Edges

    def add_location_location_edge(
        self,
        loc1,
        loc2,
        proximity,
        connectivity,
        rivalry,
        trade_relations,
        edge_type="location_location",
    ):
        self.G.add_edge(
            loc1,
            loc2,
            type=edge_type,
            proximity=proximity,
            connectivity=connectivity,
            rivalry=rivalry,
            trade_relations=trade_relations,
            key=edge_type,
            is_overlapping=loc1.overlaps(loc2),
        )

    # Location-Item Edges
    def add_location_item_edge(
        self, loc, obj, item_presence, item_relevance, edge_type="location_item"
    ):
        self.G.add_edge(
            loc,
            obj,
            type=edge_type,
            item_presence=item_presence,
            item_relevance=item_relevance,
            key=edge_type,
            item_at_location=loc.contains_point(*obj.coordinates_location),
        )

    # Location-Event Edges
    def add_location_event_edge(
        self,
        loc,
        event,
        event_occurrence,
        location_role,
        capacity,
        preparation_level,
        edge_type="location_event",
    ):
        self.G.add_edge(
            loc,
            event,
            type=edge_type,
            event_occurrence=event_occurrence,
            location_role=location_role,
            capacity=capacity,
            preparation_level=preparation_level,
            key=edge_type,
            event_at_location=loc.contains_point(*event.coordinates_location),
        )

    # Location-Activity Edges
    def add_location_activity_edge(
        self,
        loc,
        act,
        activity_suitability,
        activity_popularity,
        exclusivity,
        edge_type="location_activity",
    ):
        self.G.add_edge(
            loc,
            act,
            type=edge_type,
            activity_suitability=activity_suitability,
            activity_popularity=activity_popularity,
            exclusivity=exclusivity,
            key=edge_type,
            activity_at_location=loc.contains_point(*act.coordinates_location),
        )

    # Item-Item Edges
    def add_item_item_edge(
        self, obj1, obj2, compatibility, conflict, combinability, edge_type="item_item"
    ):
        self.G.add_edge(
            obj1,
            obj2,
            type=edge_type,
            compatibility=compatibility,
            conflict=conflict,
            combinability=combinability,
            key=edge_type,
            distance=dist(obj1.coordinates_location, obj2.coordinates_location),
        )

    # Item-Activity Edges
    def add_item_activity_edge(
        self, obj, act, necessity, enhancement, obstruction, edge_type="item_activity"
    ):
        self.G.add_edge(
            obj,
            act,
            type=edge_type,
            necessity=necessity,
            enhancement=enhancement,
            obstruction=obstruction,
            key=edge_type,
        )

    # Event-Activity Edges
    def add_event_activity_edge(
        self,
        event,
        act,
        activities_involved,
        activity_impact,
        activity_requirements,
        edge_type="event_activity",
    ):
        self.G.add_edge(
            event,
            act,
            type=edge_type,
            activities_involved=activities_involved,
            activity_impact=activity_impact,
            activity_requirements=activity_requirements,
            key=edge_type,
        )

    # Event-Item Edges
    def add_event_item_edge(
        self,
        event,
        obj,
        required_for_event,
        item_usage,
        item_impact,
        edge_type="event_item",
    ):
        self.G.add_edge(
            event,
            obj,
            type=edge_type,
            required_for_event=required_for_event,
            item_usage=item_usage,
            item_impact=item_impact,
            key=edge_type,
        )

    # Activity-Activity Edges
    def add_activity_activity_edge(
        self, act1, act2, synergy, conflict, dependency, edge_type="activity_activity"
    ):
        self.G.add_edge(
            act1,
            act2,
            type=edge_type,
            synergy=synergy,
            conflict=conflict,
            dependency=dependency,
            key=edge_type,
        )

    # Additional Job-Related Edges
    # Character-Job Edges
    def add_character_job_edge(
        self, char, job, role, job_status, job_performance, edge_type="character_job"
    ):
        self.G.add_edge(
            char,
            job,
            type=edge_type,
            role=role,
            job_status=job_status,
            job_performance=job_performance,
            key=edge_type,
            qualifies_for_job=char.qualifies_for_job(job),
            distance=dist(char.coordinates_location, job.location.coordinates_location),
        )

    # Job-Location Edges
    def add_job_location_edge(
        self, job, loc, essential_for_job, location_dependence, edge_type="job_location"
    ):
        self.G.add_edge(
            job,
            loc,
            type=edge_type,
            essential_for_job=essential_for_job,
            location_dependence=location_dependence,
            key=edge_type,
        )

    # Job-Activity Edges
    def add_job_activity_edge(
        self,
        job,
        act,
        activity_necessity,
        performance_enhancement,
        edge_type="job_activity",
    ):
        self.G.add_edge(
            job,
            act,
            type=edge_type,
            activity_necessity=activity_necessity,
            performance_enhancement=performance_enhancement,
            key=edge_type,
        )

    # Adding temporal, emotional, economic, historical, and security attributes dynamically
    def add_temporal_edge_attribute(self, node1, node2, temporal_data):
        self.G[node1][node2]["temporal"] = temporal_data

    def add_emotional_edge_attribute(self, node1, node2, emotional_data):
        self.G[node1][node2]["emotional"] = emotional_data

    def add_economic_edge_attribute(self, node1, node2, economic_data):
        self.G[node1][node2]["economic"] = economic_data

    def add_historical_edge_attribute(self, node1, node2, historical_data):
        self.G[node1][node2]["historical"] = historical_data

    def add_security_edge_attribute(self, node1, node2, security_data):
        self.G[node1][node2]["security"] = security_data

        # Enhanced Character-Character Edges

    def add_enhanced_character_character_edge(
        self, edge_type, char1, char2, shared_experiences, mutual_relations
    ):
        self.G.add_edge(
            char1,
            char2,
            type=edge_type,
            shared_experiences=shared_experiences,
            mutual_relations=mutual_relations,
        )

    # Enhanced Character-Location Edges
    def add_enhanced_character_location_edge(
        self, edge_type, char, loc, emotional_attachment, significant_events
    ):
        self.G.add_edge(
            char,
            loc,
            type=edge_type,
            emotional_attachment=emotional_attachment,
            significant_events=significant_events,
        )

    # Enhanced Character-Item Edges
    def add_enhanced_character_item_edge(
        self, edge_type, char, obj, items_exchanged, items_lost_found
    ):
        self.G.add_edge(
            char,
            obj,
            type=edge_type,
            items_exchanged=items_exchanged,
            items_lost_found=items_lost_found,
        )

    # Enhanced Character-Event Edges
    def add_enhanced_character_event_edge(
        self, edge_type, char, event, anticipations, memories
    ):
        self.G.add_edge(
            char,
            event,
            type=edge_type,
            anticipations=anticipations,
            memories=memories,
        )

    # Enhanced Character-Activity Edges
    def add_enhanced_character_activity_edge(
        self, edge_type, char, act, aversions, aspirations
    ):
        self.G.add_edge(
            char,
            act,
            type=edge_type,
            aversions=aversions,
            aspirations=aspirations,
        )

    # Enhanced Location-Location Edges
    def add_enhanced_location_location_edge(
        self, edge_type, loc1, loc2, historical_links, environmental_factors
    ):
        self.G.add_edge(
            loc1,
            loc2,
            type=edge_type,
            historical_links=historical_links,
            environmental_factors=environmental_factors,
        )

    # Enhanced Location-Item Edges
    def add_enhanced_location_item_edge(
        self, edge_type, loc, obj, items_history, symbolic_items
    ):
        self.G.add_edge(
            loc,
            obj,
            type=edge_type,
            items_history=items_history,
            symbolic_items=symbolic_items,
        )

    # Enhanced Location-Event Edges
    def add_enhanced_location_event_edge(
        self, edge_type, loc, event, recurring_events, historic_impact
    ):
        self.G.add_edge(
            loc,
            event,
            type=edge_type,
            recurring_events=recurring_events,
            historic_impact=historic_impact,
        )

    # Enhanced Location-Activity Edges
    def add_enhanced_location_activity_edge(
        self, edge_type, loc, act, prohibitions, historical_activities
    ):
        self.G.add_edge(
            loc,
            act,
            type=edge_type,
            prohibitions=prohibitions,
            historical_activities=historical_activities,
        )

    # Enhanced Item-Item Edges
    def add_enhanced_item_item_edge(
        self, edge_type, obj1, obj2, part_of_set, usage_combinations
    ):
        self.G.add_edge(
            obj1,
            obj2,
            type=edge_type,
            part_of_set=part_of_set,
            usage_combinations=usage_combinations,
        )

    # Enhanced Item-Activity Edges
    def add_enhanced_item_activity_edge(
        self, edge_type, obj, act, damage_risks, repair_opportunities
    ):
        self.G.add_edge(
            obj,
            act,
            type=edge_type,
            damage_risks=damage_risks,
            repair_opportunities=repair_opportunities,
        )

    # Enhanced Event-Activity Edges
    def add_enhanced_event_activity_edge(
        self, edge_type, event, act, preventions_triggers, traditional_activities
    ):
        self.G.add_edge(
            event,
            act,
            type=edge_type,
            preventions_triggers=preventions_triggers,
            traditional_activities=traditional_activities,
        )

    # Enhanced Event-Item Edges
    def add_enhanced_event_item_edge(
        self, edge_type, event, obj, event_triggers, traditional_uses
    ):
        self.G.add_edge(
            event,
            obj,
            type=edge_type,
            event_triggers=event_triggers,
            traditional_uses=traditional_uses,
        )

    # Enhanced Activity-Activity Edges
    def add_enhanced_activity_activity_edge(
        self, edge_type, act1, act2, exclusivity, sequences
    ):
        self.G.add_edge(
            act1,
            act2,
            type=edge_type,
            exclusivity=exclusivity,
            sequences=sequences,
        )

    def add_dict_of_edges(self, edges_dict):
        for edge_type, edges in edges_dict.items():
            if edge_type == "character_character":
                for edge in edges:
                    self.add_character_character_edge(*edge)
            elif edge_type == "character_location":
                for edge in edges:
                    self.add_character_location_edge(*edge)
            elif edge_type == "character_object":
                for edge in edges:
                    self.add_character_object_edge(*edge)
            elif edge_type == "character_event":
                for edge in edges:
                    self.add_character_event_edge(*edge)
            elif edge_type == "character_activity":
                for edge in edges:
                    self.add_character_activity_edge(*edge)
            elif edge_type == "location_location":
                for edge in edges:
                    self.add_location_location_edge(*edge)
            elif edge_type == "location_item":
                for edge in edges:
                    self.add_location_item_edge(*edge)
            elif edge_type == "location_event":
                for edge in edges:
                    self.add_location_event_edge(*edge)
            elif edge_type == "location_activity":
                for edge in edges:
                    self.add_location_activity_edge(*edge)
            elif edge_type == "item_item":
                for edge in edges:
                    self.add_item_item_edge(*edge)
            elif edge_type == "item_activity":
                for edge in edges:
                    self.add_item_activity_edge(*edge)
            elif edge_type == "event_activity":
                for edge in edges:
                    self.add_event_activity_edge(*edge)
            elif edge_type == "event_item":
                for edge in edges:
                    self.add_event_item_edge(*edge)
            elif edge_type == "activity_activity":
                for edge in edges:
                    self.add_activity_activity_edge(*edge)
            elif edge_type == "character_job":
                for edge in edges:
                    self.add_character_job_edge(*edge)
            elif edge_type == "job_location":
                for edge in edges:
                    self.add_job_location_edge(*edge)
            elif edge_type == "job_activity":
                for edge in edges:
                    self.add_job_activity_edge(*edge)

    def find_shortest_path(self, source, target):
        """
        Returns the shortest path between source and target nodes using Dijkstra's algorithm.

        Parameters:
            source (str): Node identifier for the source node.
            target (str): Node identifier for the target node.

        Returns:
            list or None: List of nodes representing the shortest path or None if no path exists.

        Usage example:
            path = graph_manager.find_shortest_path('char1', 'char2')
            if path:
                print("Path found:", path)
            else:
                print("No path exists between the characters.")
        """
        try:
            path = nx.shortest_path(self.G, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return None

    def detect_communities(self):
        """
        Detects communities within the graph using the Louvain method for community detection.

        Returns:
            list of sets: A list where each set contains the nodes that form a community.

        Usage example:
            communities = graph_manager.detect_communities()
            print("Detected communities:", communities)
        """
        communities = community.louvain_communities(self.G, weight="weight")
        return communities

    def calculate_centrality(self):
        """
        Calculates and returns centrality measures for nodes in the graph, useful for identifying
        key influencers or central nodes within the network.

        Returns:
            dict: A dictionary where keys are node identifiers and values are centrality scores.

        Usage example:
            centrality = graph_manager.calculate_centrality()
            print("Centrality scores:", centrality)
        """
        centrality = nx.degree_centrality(self.G)
        return centrality

    def shortest_path_between_characters(self, char1, char2):
        """
        Find the most direct connection or interaction chain between two characters, which can be useful
        for understanding potential influences or conflicts.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.

        Returns:
            list or None: List of characters forming the path or None if no path exists.

        Usage example:
            path = graph_manager.shortest_path_between_characters('char1', 'char3')
            print("Direct interaction chain:", path)
        """
        return self.find_shortest_path(char1, char2)

    def common_interests_cluster(self):
        """
        Identify clusters of characters that share common interests, which can be used to form groups
        or communities within the game.

        Returns:
            list of sets: Each set contains characters that share common interests.

        Usage example:
            interest_clusters = graph_manager.common_interests_cluster()
            print("Clusters based on common interests:", interest_clusters)
        """

        # Assuming 'interests' is a node attribute containing a set of interests for each character
        def shared_interests(node1, node2):
            return len(
                set(self.G.nodes[node1]["interests"])
                & set(self.G.nodes[node2]["interests"])
            )

        clusters = community.greedy_modularity_communities(
            self.G, weight=shared_interests
        )
        return clusters

    def most_influential_character(self):
        """
        Identify the character who has the most connections or the highest influence scores with others,
        which could be used to simulate social dynamics.

        Returns:
            str: Node identifier of the most influential character.

        Usage example:
            influencer = graph_manager.most_influential_character()
            print("Most influential character:", influencer)
        """
        centrality = nx.degree_centrality(self.G)
        most_influential = max(centrality, key=centrality.get)
        return most_influential

    def expanded_event_impact_analysis(self, event_node):
        """
        Analyze the broader range of impacts from an event, including long-term changes in relationships
        and character development.

        Parameters:
            event_node (str): Node identifier for the event.

        Returns:
            dict: A dictionary detailing the impacts on each character linked to the event.

        Usage example:
            impacts = graph_manager.expanded_event_impact_analysis('event1')
            print("Event impacts:", impacts)
        """
        impacts = {}
        for node in self.G.nodes:
            if self.G.has_edge(node, event_node):
                impacts[node] = {
                    "emotional": self.G[node][event_node].get("emotional", 0),
                    "historical": self.G[node][event_node].get("historical", 0),
                }
        return impacts

    def retrieve_characters_relationships(self, character):
        """
        Retrieve all relationships of a given character.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.retrieve_characters_relationships(character)

    def update_relationship_status(self, char1, char2, update_info):
        """
        Update relationship status between two characters.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.update_relationship_status(char1, char2, update_info)

    def analyze_location_popularity(self):
        """
        Analyzes and ranks locations based on visitation frequency.

        Returns:
            dict: A dictionary where each location node is mapped to its visit count.

        Usage example:
            location_popularity = graph_manager.analyze_location_popularity()
            print("Location popularity:", location_popularity)
        """
        location_visits = {
            node: data["visit_count"]
            for node, data in self.G.nodes(data=True)
            if "visit_count" in data
        }
        popular_locations = dict(
            sorted(location_visits.items(), key=lambda item: item[1], reverse=True)
        )
        return popular_locations

    def item_ownership_network(self):
        """
        Analyzes the network of item ownership and transactions between characters.

        Returns:
            dict: A dictionary representing the flow of items between characters.

        Usage example:
            ownership_network = graph_manager.item_ownership_network()
            print("Item ownership network:", ownership_network)
        """
        ownership_network = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("type") == "ownership":
                if u not in ownership_network:
                    ownership_network[u] = {}
                ownership_network[u][v] = data
        return ownership_network

    def transfer_item_ownership(self, item, from_char, to_char):
        """
        Transfers ownership of an item from one character to another.

        Parameters:
            item (str): Node identifier for the item.
            from_char (str): Node identifier for the character transferring the item.
            to_char (str): Node identifier for the recipient character.

        Usage example:
            graph_manager.transfer_item_ownership('item1', 'char1', 'char2')
            print(f"{item} has been transferred from {from_char} to {to_char}.")
        """
        if self.G.has_edge(from_char, item):
            self.G.remove_edge(from_char, item)
            self.add_character_object_edge(
                to_char,
                item,
                ownership_status=True,
                usage_frequency=0,
                sentimental_value=0,
                last_used_time=datetime.now(),
            )
        else:
            raise ValueError(f"The item {item} is not owned by {from_char}.")

    def analyze_character_relationships(self, character_id):
        """
        Analyze relationships for a specific character.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.analyze_character_relationships(character_id)

    def calculate_motives(self, character: Character):
        Character = importlib.import_module("tiny_characters").Character
        PersonalMotives = importlib.import_module("tiny_characters").PersonalMotives
        Motive = importlib.import_module("tiny_characters").Motive

        social_wellbeing_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + (character.personality_traits.get_extraversion() * 2)
            + character.personality_traits.get_agreeableness()
            - character.personality_traits.get_neuroticism(),
            10.0,
        )
        beauty_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + character.personality_traits.get_extraversion()
            + character.personality_traits.get_agreeableness()
            + character.personality_traits.get_neuroticism()
            + social_wellbeing_motive,
            10.0,
        )
        hunger_motive = cached_sigmoid_motive_scale_approx_optimized(
            (10 - character.get_mental_health())
            + character.personality_traits.get_neuroticism()
            - character.personality_traits.get_conscientiousness()
            - beauty_motive,
            10.0,
        )
        community_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + character.personality_traits.get_extraversion()
            + character.personality_traits.get_agreeableness()
            + character.personality_traits.get_neuroticism()
            + social_wellbeing_motive,
            10.0,
        )
        health_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + character.personality_traits.get_extraversion()
            + character.personality_traits.get_agreeableness()
            + character.personality_traits.get_neuroticism()
            - hunger_motive
            + beauty_motive
            + character.personality_traits.get_conscientiousness(),
            10.0,
        )
        mental_health_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + character.personality_traits.get_extraversion()
            + character.personality_traits.get_agreeableness()
            + character.personality_traits.get_neuroticism()
            - hunger_motive
            + beauty_motive
            + character.personality_traits.get_conscientiousness()
            + health_motive,
            10.0,
        )
        stability_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_openness()
            + character.personality_traits.get_extraversion()
            + character.personality_traits.get_agreeableness()
            + character.personality_traits.get_neuroticism()
            + health_motive
            + community_motive,
            10.0,
        )
        shelter_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_neuroticism()
            + character.personality_traits.get_conscientiousness()
            + health_motive
            + community_motive
            + beauty_motive
            + stability_motive,
            10.0,
        )
        control_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_conscientiousness()
            + character.personality_traits.get_neuroticism()
            + shelter_motive
            + stability_motive,
            10.0,
        )
        success_motive = cached_sigmoid_motive_scale_approx_optimized(
            character.personality_traits.get_conscientiousness()
            + character.personality_traits.get_neuroticism()
            + shelter_motive
            + stability_motive
            + control_motive,
            10.0,
        )
        material_goods_motive = random.gauss(
            cached_sigmoid_motive_scale_approx_optimized(
                cached_sigmoid_motive_scale_approx_optimized(
                    (
                        cached_sigmoid_motive_scale_approx_optimized(
                            shelter_motive, 25.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            stability_motive, 25.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            success_motive, 25.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            control_motive, 25.0
                        )
                        if control_motive > 0.0
                        else 1.0
                    ),
                    10.0,
                )
                + tanh_scaling(
                    character.personality_traits.get_conscientiousness()
                    + character.personality_traits.get_neuroticism() * 10.0,
                    10.0,
                    -10.0,
                    0.0,
                    1.0,
                ),
                10.0,
            ),
            1.0,
        )

        luxury_motive = cached_sigmoid_motive_scale_approx_optimized(
            tanh_scaling(
                character.personality_traits.get_openness()
                + character.personality_traits.get_extraversion()
                + character.personality_traits.get_agreeableness()
                + character.personality_traits.get_neuroticism() * 10.0,
                10.0,
                -10.0,
                0.0,
                10.0,
            )
            + cached_sigmoid_motive_scale_approx_optimized(
                cached_sigmoid_motive_scale_approx_optimized(
                    material_goods_motive, 50.0
                )
                + cached_sigmoid_motive_scale_approx_optimized(beauty_motive, 50.0),
                10.0,
            ),
            10.0,
        )

        wealth_motive = random.gauss(
            cached_sigmoid_motive_scale_approx_optimized(
                cached_sigmoid_motive_scale_approx_optimized(luxury_motive, 25.0)
                + cached_sigmoid_motive_scale_approx_optimized(shelter_motive, 10.0)
                + cached_sigmoid_motive_scale_approx_optimized(stability_motive, 15.0)
                + cached_sigmoid_motive_scale_approx_optimized(success_motive, 15.0)
                + cached_sigmoid_motive_scale_approx_optimized(control_motive, 10.0)
                + cached_sigmoid_motive_scale_approx_optimized(
                    material_goods_motive, 25.0
                )
                + tanh_scaling(
                    character.personality_traits.get_conscientiousness()
                    + abs(character.personality_traits.get_neuroticism()) * 10,
                    10.0,
                    -10.0,
                    0.0,
                    1.0,
                ),
                10.0,
            ),
            1.0,
        )

        job_performance_motive = abs(
            round(
                random.gauss(
                    abs(
                        cached_sigmoid_motive_scale_approx_optimized(
                            success_motive
                            + material_goods_motive
                            + wealth_motive * 2.0 / 3.0,
                            10.0,
                        )
                    ),
                    cached_sigmoid_motive_scale_approx_optimized(
                        character.personality_traits.get_conscientiousness()
                        + character.personality_traits.get_extraversion()
                        + character.personality_traits.get_agreeableness()
                        + character.personality_traits.get_neuroticism() * 10.0,
                        1.0,
                    ),
                )
            )
        )
        happiness_motive = abs(
            round(
                random.gauss(
                    abs(
                        cached_sigmoid_motive_scale_approx_optimized(
                            (
                                cached_sigmoid_motive_scale_approx_optimized(
                                    success_motive, 25.0
                                )
                                + cached_sigmoid_motive_scale_approx_optimized(
                                    material_goods_motive, 15.0
                                )
                                + cached_sigmoid_motive_scale_approx_optimized(
                                    wealth_motive, 10.0
                                )
                                + cached_sigmoid_motive_scale_approx_optimized(
                                    job_performance_motive, 25.0
                                )
                                + cached_sigmoid_motive_scale_approx_optimized(
                                    social_wellbeing_motive, 25.0
                                )
                            ),
                            10.0,
                        )
                    ),
                    cached_sigmoid_motive_scale_approx_optimized(
                        character.personality_traits.get_openness()
                        + character.personality_traits.get_extraversion()
                        + character.personality_traits.get_agreeableness()
                        - character.personality_traits.get_neuroticism() * 10,
                        1.0,
                    ),
                )
            )
        )
        hope_motive = abs(
            round(
                random.gauss(
                    cached_sigmoid_motive_scale_approx_optimized(
                        cached_sigmoid_motive_scale_approx_optimized(
                            mental_health_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            social_wellbeing_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            happiness_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            health_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            shelter_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            stability_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            luxury_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            success_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            control_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            job_performance_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            beauty_motive, 100.0
                        )
                        + cached_sigmoid_motive_scale_approx_optimized(
                            community_motive, 100.0
                        )
                        / 12,
                        10.0,
                    ),
                    cached_sigmoid_motive_scale_approx_optimized(
                        character.personality_traits.get_openness()
                        + character.personality_traits.get_extraversion()
                        + character.personality_traits.get_agreeableness()
                        + character.personality_traits.get_neuroticism() * 10,
                        1.0,
                    ),
                )
            )
        )
        family_motive = random.gauss(
            abs(
                cached_sigmoid_motive_scale_approx_optimized(
                    cached_sigmoid_motive_scale_approx_optimized(
                        mental_health_motive, 5.0
                    )
                    + cached_sigmoid_motive_scale_approx_optimized(
                        social_wellbeing_motive, 10.0
                    )
                    + cached_sigmoid_motive_scale_approx_optimized(
                        happiness_motive, 20.0
                    )
                    + cached_sigmoid_motive_scale_approx_optimized(
                        stability_motive, 10.0
                    )
                    - cached_sigmoid_motive_scale_approx_optimized(luxury_motive, 20.0)
                    + cached_sigmoid_motive_scale_approx_optimized(success_motive, 5.0)
                    + cached_sigmoid_motive_scale_approx_optimized(control_motive, 5.0)
                    + cached_sigmoid_motive_scale_approx_optimized(
                        community_motive, 15.0
                    )
                    + cached_sigmoid_motive_scale_approx_optimized(hope_motive, 15.0),
                    10.0,
                )
            ),
            cached_sigmoid_motive_scale_approx_optimized(
                (
                    character.personality_traits.get_openness()
                    + character.personality_traits.get_extraversion()
                    + character.personality_traits.get_agreeableness()
                    + character.personality_traits.get_conscientiousness() * 10
                ),
                1.0,
            ),
        )

        return PersonalMotives(
            hunger_motive=Motive(
                "hunger", "bias toward satisfying hunger", hunger_motive
            ),
            wealth_motive=Motive(
                "wealth", "bias toward accumulating wealth", wealth_motive
            ),
            mental_health_motive=Motive(
                "mental health",
                "bias toward maintaining mental health",
                mental_health_motive,
            ),
            social_wellbeing_motive=Motive(
                "social wellbeing",
                "bias toward maintaining social wellbeing",
                social_wellbeing_motive,
            ),
            happiness_motive=Motive(
                "happiness", "bias toward maintaining happiness", happiness_motive
            ),
            health_motive=Motive(
                "health", "bias toward maintaining health", health_motive
            ),
            shelter_motive=Motive(
                "shelter", "bias toward maintaining shelter", shelter_motive
            ),
            stability_motive=Motive(
                "stability", "bias toward maintaining stability", stability_motive
            ),
            luxury_motive=Motive(
                "luxury", "bias toward maintaining luxury", luxury_motive
            ),
            hope_motive=Motive("hope", "bias toward maintaining hope", hope_motive),
            success_motive=Motive(
                "success", "bias toward maintaining success", success_motive
            ),
            control_motive=Motive(
                "control", "bias toward maintaining control", control_motive
            ),
            job_performance_motive=Motive(
                "job performance",
                "bias toward maintaining job performance",
                job_performance_motive,
            ),
            beauty_motive=Motive(
                "beauty", "bias toward maintaining beauty", beauty_motive
            ),
            community_motive=Motive(
                "community", "bias toward maintaining community", community_motive
            ),
            material_goods_motive=Motive(
                "material goods",
                "bias toward maintaining material goods",
                material_goods_motive,
            ),
            family_motive=Motive(
                "family", "bias toward maintaining family", family_motive
            ),
        )

    def location_popularity_analysis(self):
        """
        Determines the popularity of locations based on visitation data.

        Returns:
            dict: A dictionary where keys are location IDs and values are the number of visits.

        Usage example:
            location_popularity = graph_manager.location_popularity_analysis()
            print("Location popularity:", location_popularity)
        """
        popularity = {}
        for node in self.G.nodes(data=True):
            if node[1].get("type") == "location":
                popularity[node[0]] = sum(1 for _ in self.G.edges(node[0]))
        return popularity

    def track_item_ownership(self):
        """
        Tracks ownership and transaction historical of items.

        Returns:
            dict: A dictionary where keys are item IDs and values are details of ownership and transaction historical.

        Usage example:
            item_ownership = graph_manager.track_item_ownership()
            print("Item ownership details:", item_ownership)
        """
        ownership = {}
        for node in self.G.nodes(data=True):
            if node[1].get("type") == "item":
                ownership[node[0]] = {
                    "current_owner": self.G.nodes[node[0]].get("owner"),
                    "transaction_history": self.G.nodes[node[0]].get(
                        "transaction_history", []
                    ),
                }
        return ownership

    def determine_owner(self, item):
        """
        Determines the current owner of an item.

        Parameters:
            item (str): Node identifier for the item.

        Returns:
            Any: Node of the current owner.

        Usage example:
            owner = graph_manager.determine_owner('item1')
            print("Current owner:", owner)
        """
        # Filter edges that are character_object and have ownership_status True
        owner_edges = [
            edge
            for edge in self.G.edges(item)
            if self.G.edges[edge].get("type") == "character_object"
            and self.G.edges[edge].get("ownership_status") == True
        ]

        # Return the character node from the filtered edges
        for edge in owner_edges:
            if edge[0] != item:
                return edge[0]
            return edge[1]

    def predict_future_relationships(self, character_id):
        """
        Predicts future interactions and relationships based on historical data.

        Parameters:
            character_id (str): Node identifier for the character.

        Returns:
            dict: Predictions of future relationships.

        Usage example:
            future_relationships = graph_manager.predict_future_relationships('char2')
            print("Predicted future relationships:", future_relationships)
        """
        # Placeholder for a more complex predictive model
        return {
            "info": "This feature requires additional predictive modeling capabilities."
        }

    def update_node_attribute(self, node, attribute, value):
        """
        Updates an attribute for a node with a new value.

        Parameters:
            node (str): Node identifier.
            attribute (str): Attribute to update.
            value (any): New value for the attribute.

        Usage example:
            graph_manager.update_node_attribute('char1', 'mood', 'happy')
        """
        if node in self.G:
            self.G.nodes[node][attribute] = value
        else:
            raise ValueError("Node does not exist in the graph.")

    def find_all_paths(self, source, target, max_length=None):
        """
        Finds all paths between source and target nodes up to a maximum length.

        Parameters:
            source (str): Node identifier for the source.
            target (str): Node identifier for the target.
            max_length (int): Maximum number of edges in the path (optional).

        Returns:
            list of lists: A list containing all paths, each path is a list of nodes.

        Usage example:
            paths = graph_manager.find_all_paths('char1', 'char2', max_length=5)
            print("All paths up to length 5:", paths)
        """
        paths = list(
            nx.all_simple_paths(self.G, source=source, target=target, cutoff=max_length)
        )
        return paths

    def node_influence_spread(self, node, decay_function=lambda x: 1 / x):
        """
        Calculates the influence spread of a node over others in the graph, with influence decreasing
        over distance based on a decay function.

        Parameters:
            node (str): Node identifier.
            decay_function (function): A function that defines how influence decays with distance.

        Returns:
            dict: A dictionary with node identifiers as keys and influence scores as values.

        Usage example:
            influence = graph_manager.node_influence_spread('char1')
            print("Influence spread of char1:", influence)
        """
        influence = {}
        for target in self.G.nodes:
            if target != node:
                paths = self.find_all_paths(node, target)
                for path in paths:
                    distance = len(path) - 1  # edges are one less than nodes in path
                    if target not in influence:
                        influence[target] = 0
                    influence[target] += decay_function(distance)
        return influence

    def analyze_relationship_health(self, char1, char2):
        """
        Analyze the health of a relationship between two characters.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.analyze_relationship_health(char1, char2)

    def update_edge_attribute(self, node1, node2, attribute, value):
        """
        Updates or adds an attribute to an edge between two nodes.

        Parameters:
            node1 (str): The first node identifier.
            node2 (str): The second node identifier.
            attribute (str): The attribute to update or add.
            value (any): The new value for the attribute.

        Usage example:
            graph_manager.update_edge_attribute('char1', 'char2', 'trust', 75)
        """
        # Access the mutable dictionary of the edge attributes
        edge_data = self.G.edges[node1, node2]
        # Update the attribute
        edge_data[attribute] = value

    def evaluate_relationship_strength(self, char1, char2):
        """
        Evaluate the overall strength of a relationship.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.evaluate_relationship_strength(char1, char2)

    def check_friendship_status(self, char1, char2):
        """
        Determines the friendship status between two characters based on their historical interactions and emotional connections.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.

        Returns:
            str: Describes the friendship status ('friends', 'neutral', or 'enemies').

        Usage example:
            status = graph_manager.check_friendship_status('char1', 'char2')
            print("Friendship status:", status)
        """
        if self.G.has_edge(char1, char2):
            emotional = self.G[char1][char2].get("emotional_impact", 0)
            historical = self.G[char1][char2].get("historical", 0)
            trust = self.G[char1][char2].get("trust", 0)
            strength = self.G[char1][char2].get("strength", 0)
            # Example logic to determine friendship status
            if emotional > 50 and historical > 50 and trust > 50 and strength > 50:
                return "friends"
            elif emotional < 25 and historical < 25 and trust < 25 and strength < 25:
                return "adversaries"
            elif emotional > 35 or historical > 35 or trust > 35 or strength > 35:
                return "acquaintances"
        return "neutral"

    def character_location_frequency(self, char):
        """
        Returns a dictionary detailing the frequency of visits a character has made to various locations.

        Parameters:
            char (str): Node identifier for the character.

        Returns:
            dict: Keys are locations and values are counts of visits.

        Usage example:
            frequency = graph_manager.character_location_frequency('char1')
            print("Location visit frequency:", frequency)
        """
        frequency = {}
        for edge in self.G.edges(char, data=True):
            if edge[2]["type"] == "character_location":
                location = edge[1]
                frequency[location] = edge[2]["frequency_of_visits"]
        return frequency

    def location_popularity(self, location):
        """
        Determines the popularity of a location based on visit frequency by characters.

        Parameters:
            location (str): Location node identifier.

        Returns:
            int: Number of visits to the location.

        Usage example:
            popularity = graph_manager.location_popularity('park')
            print("Popularity of the location:", popularity)
        """
        if location in self.G.nodes:
            return self.G.nodes[location].get("frequency_of_visits", 0)
        return 0

    def item_ownership_history(self, item):
        """
        Tracks the historical of ownership for a specific item.

        Parameters:
            item (str): Item node identifier.

        Returns:
            list: History of characters who have owned the item.

        Usage example:
            historical = graph_manager.item_ownership_history('sword')
            print("Ownership historical of the item:", historical)
        """
        if item in self.G:
            return self.G.nodes[item].get("ownership_history", [])
        else:
            logging.warning(f"Item '{item}' not found in the graph.")
            return []

    def can_interact_directly(self, char1, char2, conditions=None):
        """
        Determines if two characters can interact directly based on a set of specified conditions.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.
            conditions (dict): A dictionary where keys are attributes or conditions to check and
                            values are the required values for these attributes or conditions.
                            Conditions can specify node or edge attributes.

        Returns:
            bool: True if all conditions are met for interaction, False otherwise.

        Usage example:
            conditions = {
                'proximity': True,
                'relationship_status': 'friendly',
                'mood': ['happy', 'neutral']  # Character must be either happy or neutral
            }
            if graph_manager.can_interact_directly('char1', 'char2', conditions):
                print("Characters can interact.")
            else:
                print("Characters cannot interact.")
        """
        if not self.G.has_edge(char1, char2):
            return False

        # Check edge-specific conditions
        for key, required_value in conditions.items():
            if key in self.G[char1][char2]:
                value = self.G[char1][char2][key]
                if isinstance(required_value, list) and value not in required_value:
                    return False
                elif not isinstance(required_value, list) and value != required_value:
                    return False

        # Check node-specific conditions for both characters
        for char in [char1, char2]:
            for key, required_value in conditions.items():
                if key in self.G.nodes[char]:
                    value = self.G.nodes[char][key]
                    if isinstance(required_value, list) and value not in required_value:
                        return False
                    elif (
                        not isinstance(required_value, list) and value != required_value
                    ):
                        return False

        return True

    def get_nearest_resource(
        self,
        character,
        resource_filter,
        attribute_name="distance",
        max_search_depth=None,
        default_attribute_value=float("inf"),
    ):
        """
        Enhanced function to find the nearest resource to a character using a custom filter function or attribute criteria,
        considering a configurable edge attribute and optional max search depth for optimized performance.

        Parameters:
            character (str): Node identifier for the character.
            resource_filter (function or dict): Custom filter function or dictionary of attribute-value pairs defining the resource.
            attribute_name (str): Name of the edge attribute to use for path calculations (default is 'weight').
            max_search_depth (int): Optional maximum depth to search in the graph.
            default_attribute_value (float or int): Default value for the edge attribute if not present (default is infinity).

        Returns:
            tuple: Identifier of the nearest resource and the distance, or (None, None) if no resource is found.
        """
        is_filter_function = callable(resource_filter)
        to_visit = [(0, character)]
        visited = set()
        nearest_resource = None
        shortest_distance = float("inf")

        while to_visit:
            distance, current_node = heapq.heappop(to_visit)
            if current_node in visited:
                continue
            visited.add(current_node)
            node_data = self.G.nodes[current_node]

            # Check resource criteria or filter
            if (is_filter_function and resource_filter(current_node, node_data)) or (
                not is_filter_function
                and all(node_data.get(k) == v for k, v in resource_filter.items())
            ):
                if distance < shortest_distance:
                    shortest_distance = distance
                    nearest_resource = (current_node, distance)

            # Process neighbors considering the specified edge attribute
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    # Select the edge with the minimum attribute value
                    if attribute_name == "distance":
                        edge_attribute_value = self.get_distance_between_nodes(
                            current_node, neighbor
                        )
                    else:
                        edge_data = min(
                            self.G[current_node][neighbor].values(),
                            key=lambda e: e.get(
                                attribute_name, default_attribute_value
                            ),
                        )
                        edge_attribute_value = edge_data.get(
                            attribute_name, default_attribute_value
                        )

                    new_dist = distance + edge_attribute_value
                    if max_search_depth is None or new_dist <= max_search_depth:
                        heapq.heappush(to_visit, (new_dist, neighbor))

        return nearest_resource if nearest_resource else (None, None)

    def get_distance_between_nodes(self, node1, node2):
        """
        Returns the distance between two nodes in the graph.

        Parameters:
            node1 (str): Node identifier for the first node.
            node2 (str): Node identifier for the second node.

        Returns:
            float: The distance between the two nodes.

        Usage example:
            distance = graph_manager.get_distance_between_nodes('char1', 'char2')
            print("Distance between nodes:", distance)
        """
        # Get the location coordinates for each node
        loc1 = self.G.nodes[node1].get("coordinate_location")
        loc2 = self.G.nodes[node2].get("coordinate_location")
        if loc1 and loc2:
            if isinstance(loc1, tuple) and isinstance(loc2, tuple):
                # Calculate the Euclidean distance between the locations
                return dist(loc1, loc2)

            # Calculate the Euclidean distance between the locations
            # return math.sqrt(sum((a - b) ** 2 for a, b in zip(loc1, loc2)))

    def track_event_participation(self, char, event):
        """
        Tracks a character's participation in an event, updating the graph to reflect this.

        Parameters:
            char (str): Node identifier for the character.
            event (str): Node identifier for the event.

        Usage example:
            graph_manager.track_event_participation('char1', 'event1')
        """
        self.add_character_event_edge(char, event, True, role="participant")

    def check_safety_of_locations(self, loc):
        """
        Checks the safety of a specific location based on nearby threats and its security attributes.

        Parameters:
            loc (str): Location node identifier.

        Returns:
            float: Safety score of the location.

        Usage example:
            safety_score = graph_manager.check_safety_of_locations('location1')
            print("Safety score for location1:", safety_score)
        """
        threats = sum(data.get("threat_level", 0) for node, data in self.G[loc].items())
        security = self.G.nodes[loc].get("security", 1)
        return security / (1 + threats)

    def evaluate_trade_opportunities_by_char_surplus(self, char):
        """
        Evaluates trade opportunities for a character based on the surplus and demand across the graph.

        Parameters:
            char (str): Character node identifier.

        Returns:
            dict: A dictionary of potential trades, where keys are resource types and values are lists of potential trade partners.

        Usage example:
            trade_opportunities = graph_manager.evaluate_trade_opportunities('char1')
            print("Trade opportunities for char1:", trade_opportunities)
        """
        surplus = {
            res: qty
            for res, qty in self.G.nodes[char]["resources"].report_inventory().items()
            if qty > 10
        }
        # Remove any resources from surplus if they are in char's needed resources
        for res, _ in surplus:
            for req_dict in self.G.nodes[char].get("needed_resources", {}).items():
                for attr, value in res.to_dict().items():
                    if (
                        isinstance(value, int) or isinstance(value, float)
                    ) and value > 0:
                        if req_dict[0] >= value:
                            surplus.pop(res)
                            break
                    elif isinstance(value, str) and value in req_dict[0].get(attr, []):
                        surplus.pop(res)
                        break

        # Remove any resources from surplus if they are in char's needed resources
        needed_resources = self.G.nodes[char].get("needed_resources", {})
        surplus = {
            res: qty for res, qty in surplus.items() if res not in needed_resources
        }

        opportunities = {}
        for res, qty in surplus.items():
            for node in self.G.nodes:
                if node == char:
                    continue
                node_needed_resources = self.G.nodes[node].get("needed_resources", {})
                if res in node_needed_resources:
                    if node not in opportunities:
                        opportunities[node] = {}
                    opportunities[node][res] = qty

        return opportunities

    def evaluate_trade_opportunities_for_item(self, item):
        """
        Evaluates trade opportunities for a specific item based on the demand and availability across the graph.

        Parameters:
            item (str): Item node identifier.

        Returns:
            dict: A dictionary of potential trades, where keys are characters and values are potential trade quantities.

        Usage example:
            trade_opportunities = graph_manager.evaluate_trade_opportunities_for_item('item1')
            print("Trade opportunities for item1:", trade_opportunities)
        """
        opportunities = {}
        qty = item.quantity
        for node in self.G.nodes:
            for attr, value in item.to_dict().items():
                if (isinstance(value, int) or isinstance(value, float)) and value > 0:
                    for req_dict in self.G.nodes[node].get("needed_resources", {}):
                        if req_dict[0].get(attr, 0) >= value:
                            if node not in opportunities:
                                opportunities[node] = {}
                            opportunities[node][item] = qty
                            break
                elif isinstance(value, str):
                    for req_dict in self.G.nodes[node].get("needed_resources", {}):
                        if isinstance(req_dict[0], dict) and attr in req_dict[0]:
                            if value in req_dict[0]:
                                if node not in opportunities:
                                    opportunities[node] = {}
                                opportunities[node][item] = qty
                                break
                        else:
                            logging.warning(
                                f"Attribute {attr} not found in {req_dict[0]} of node {node}, with value {value}. Skipping. req_dict: {req_dict} of type {type(req_dict)}"
                            )
                        # if value in req_dict[0]:
                        #     if node not in opportunities:
                        #         opportunities[node] = {}
                        #     opportunities[node][item] = qty
                        #     break

        return opportunities

    def evaluate_trade_opportunities_for_wanted_item(self, item):
        """
        Evaluates trade opportunities for a specific item based on the demand and availability across the graph. Opposite of evaluate_trade_opportunities_for_item.

        Parameters:
            item (str): Item node identifier.

        Returns:
            dict: A dictionary of potential trades, where keys are characters and values are potential trade quantities.

        Usage example:
            trade_opportunities = graph_manager.evaluate_trade_opportunities_for_wanted_item('item1')
            print("Trade opportunities for item1:", trade_opportunities)
        """
        opportunities = {}
        qty = item.quantity
        for node in self.G.nodes:
            for attr, value in item.to_dict().items():
                if (isinstance(value, int) or isinstance(value, float)) and value > 0:
                    for surplus_res, surplus_qty in (
                        self.G.nodes[node]["resources"].report_inventory().items()
                    ):
                        if surplus_res == item:
                            if node not in opportunities:
                                opportunities[node] = {}
                            opportunities[node][item] = min(qty, surplus_qty)
                            break
                elif isinstance(value, str) and value in item.to_dict().get(attr, []):
                    if node not in opportunities:
                        opportunities[node] = {}
                    opportunities[node][item] = qty
                    break

        return opportunities

    def evaluate_trade_opportunities_for_desired_items(self, char):
        """
        Opposite of evaluate_trade_opportunities_by_char_surplus, this function evaluates trade opportunities for a character based on their desired resources and the surplus of other characters. This can be used to identify potential trade partners.

        Parameters:
            char (str): Character node identifier.

        Returns:
            dict: A dictionary of potential trades, where keys are resource types and values are lists of potential trade partners.

        Usage example:
            trade_opportunities = graph_manager.evaluate_trade_opportunities_for_desired_items('char1')
            print("Trade opportunities for char1:", trade_opportunities)
        """

        opportunities = {}
        for res, qty in self.G.nodes[char].get("needed_resources", {}).items():
            for node in self.G.nodes:
                for attr, value in res.to_dict().items():
                    if (
                        isinstance(value, int) or isinstance(value, float)
                    ) and value > 0:
                        for surplus_res, surplus_qty in (
                            self.G.nodes[node]["resources"].report_inventory().items()
                        ):
                            if surplus_res == res:
                                if node not in opportunities:
                                    opportunities[node] = {}
                                opportunities[node][res] = min(qty, surplus_qty)
                                break
                    elif isinstance(value, str) and value in res.to_dict().get(
                        attr, []
                    ):
                        if node not in opportunities:
                            opportunities[node] = {}
                        opportunities[node][res] = qty
                        break

        return opportunities

    def find_most_accessible_resources(self, char, resource_type):
        """
        Finds the most accessible resource nodes of a specific type for a given character, based on proximity and quantity.

        Parameters:
            char (str): Character node identifier.
            resource_type (str): Type of resource to find.

        Returns:
            list: Sorted list of resource nodes, starting with the most accessible.

        Usage example:
            resources = graph_manager.find_most_accessible_resources('char1', 'water')
            print("Accessible water resources for char1:", resources)
        """
        resources = [
            (res, data["quantity"] / nx.shortest_path_length(self.G, char, res))
            for res, data in self.G.nodes(data=True)
            if data.get("type") == resource_type
        ]
        return sorted(resources, key=lambda x: x[1], reverse=True)

    def get_neighbors_with_attributes(self, node, **attributes):
        """
        Retrieves neighbors of a given node that meet specified attribute criteria.

        Parameters:
            node (str): Node identifier.
            attributes (dict): Keyword arguments specifying the attribute values to match.

        Returns:
            list: A list of nodes (neighbors) that match the specified attributes.

        Usage example:
            allies = graph_manager.get_neighbors_with_attributes('char1', alignment='friendly')
            print("Friendly characters near char1:", allies)
        """
        matches = []
        for neighbor in self.G.neighbors(node):
            if all(
                self.G.nodes[neighbor].get(key) == value
                for key, value in attributes.items()
            ):
                matches.append(neighbor)
        return matches

    def calculate_social_influence(
        self,
        character,
        influence_attributes=None,
        memory_topic=None,
        memory_weight=0.5,
        decision_context=None,
        influence_factors=None,
    ):
        """
        Calculate social influence on a character's decisions.
        Delegates to SocialModel for calculation.
        """
        return self.social_model.calculate_social_influence(
            character, influence_attributes, memory_topic, memory_weight,
            decision_context, influence_factors
        )

    # Including the provided decay functions here for completeness
    def time_based_decay(self, time_since):
        return 1 / (1 + time_since / 365)

    def frequency_based_decay(self, frequency):
        return 1 + (frequency / 12)

    def advanced_decay(self, distance, attributes):
        base_decay = 1 / (1 + distance)
        emotional_factor = (
            attributes.get("trust", 1) * 0.5 + attributes.get("friendship", 1) * 0.5
        )
        emotional_decay = base_decay * emotional_factor
        if "professional" in attributes:
            professional_factor = 1 + attributes["professional"] * 0.1
            emotional_decay *= professional_factor
        if "family" in attributes:
            family_factor = 1 + attributes["family"] * 0.2
            emotional_decay *= family_factor
        historical_factor = 1 + attributes.get("shared_history", 0) * 0.3
        emotional_decay *= historical_factor
        proximity_factor = 1 / (1 + 0.1 * (10 - attributes.get("proximity", 10)))
        emotional_decay *= proximity_factor
        return emotional_decay

    def query_memories(self, character, topic):
        """
        Queries memories related to a specific topic to determine their influence on the character's current decision.

        This is a placeholder and assumes integration with tiny_memories.py where actual memory querying would be implemented.

        Parameters:
            character (str): The character whose memories to query.
            topic (str): The topic to query about in the memories.

        Returns:
            float: Influence score from memories.
        """
        if topic:
            topic = tiny_memories.MemoryManager().search_memories(
                topic
            )  # Temporarily commented out
        # else:
        #     topic = tiny_memories.MemoryManager().get_all_memories(character)
        return 0.5  # Example fixed return value

    # More edge methods for other types (Location-Location, Item-Item, etc.), and edge methods from previous parts

    # Additional edge methods with comprehensive attributes to cover all detailed aspects
    # Continued in the following sections for remaining edge types and additional attributes (Temporal, Emotional, Economic, Historical, Security)

    # Other helper methods as required for managing the graph

    def create_tiny_village_graph(
        self, characters, locations, events, objects, activities
    ):
        # Create a new graph
        G = self.G

        # Add character nodes
        # characters = ["Emma", "John", "Alice", "Bob"]
        for char in characters:
            self.add_character_node(char)

        # Adding location nodes
        # locations = ["Cafe", "Park", "Library"]
        for loc in locations:
            G.add_node(
                loc,
                type="location",
                popularity=5,
                activities_available=["Read", "Socialize"],
            )

        # Add event nodes
        # events = ["Festival", "Concert"]
        for event in events:
            G.add_node(event, type="event", date="2024-10-05")

        # Adding object nodes
        # objects = ["Book", "Laptop", "Coffee"]
        for obj in objects:
            G.add_node(obj, type="object", value=20, usability=True)

        # Adding activity nodes
        # activities = ["Read", "Write", "Jog"]
        for act in activities:
            G.add_node(
                act, type="activity", related_skills="Literature", satisfaction_level=7
            )

        return G

    def graph_analysis(character, event, graph, context):
        """
        Analyzes the implications of an event on a character's network using graph theory.

        Args:
            character (str): Name of the character whose situation is being analyzed.
            event (dict): Details of the current event or decision point.
            graph (Graph): A NetworkX graph representing relationships, influences, etc.
            context (str): Context of the analysis (e.g., 'daily', 'career', 'relationship').

        Returns:
            impacted_nodes (list): List of nodes (characters, locations, etc.) impacted by the event.
            action_recommendations (list): Recommended actions based on the analysis.
        """
        # Initialize a list to store nodes impacted by the event
        impacted_nodes = []
        action_recommendations = []

        # Identify nodes directly connected to the event
        if context == "career":
            # Career context might focus on professional connections and opportunities
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "professional":
                    impacted_nodes.append(node)
                    if "opportunity" in graph.edges[character, node]:
                        action_recommendations.append(
                            {"action": "pursue_opportunity", "node": node}
                        )
        elif context == "relationship":
            # Relationship context focuses on personal connections
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "personal":
                    impacted_nodes.append(node)
                    if "conflict" in graph.edges[character, node]:
                        action_recommendations.append(
                            {"action": "resolve_conflict", "node": node}
                        )
        elif context == "daily":
            # Daily context might consider locations and routines
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "location":
                    impacted_nodes.append(node)
                    if "favorite" in graph.edges[character, node]:
                        action_recommendations.append({"action": "visit", "node": node})

        # Evaluate the broader impact using a breadth-first search or similar method
        # to explore secondary impacts
        for node in impacted_nodes:
            for secondary_node in graph.neighbors(node):
                if secondary_node not in impacted_nodes:  # Avoid cycles
                    impacted_nodes.append(secondary_node)
                    # Further analysis to add more sophisticated recommendations

        return impacted_nodes, action_recommendations

    def update_graph(graph, action, character, target=None):
        """
        Update the graph based on an action taken by a character.
        Args:
            graph (Graph): The networkx graph.
            action (str): The action taken, which will determine the update logic.
            character (str): The character who is performing the action.
            target (str, optional): The target of the action, if applicable.
        """
        if action == "Make Friend":
            if target and not graph.has_edge(character, target):
                graph.add_edge(character, target, type="friend", strength=0.1)
            elif target:
                # Strengthen existing friendship
                graph.edges[character, target]["strength"] += 0.1

    def get_character_state(self, character_name):
        """
        Fetches the current state of the character from the graph.

        Args:
            character_name (str): The name of the character.

        Returns:
            dict: The state of the character.
        """
        if character_name in self.G.nodes:
            return self.characters[self.G.nodes[character_name].get("name")].to_dict()
        else:
            raise ValueError(f"No character named {character_name} in the graph.")

    def calculate_utility(self, action, character_state):
        """
        Calculates the utility of an action for a character based on the current state.

        Args:
            action (Action): The action to evaluate.
            character_state (dict): The current state of the character.

        Returns:
            float: The utility value of the action.
        """

        utility = 0
        for effect in action.effects:
            if effect.attribute in character_state:
                # Calculate the change in the attribute value
                delta = effect.value - character_state[effect.attribute]
                # Apply a weight based on the importance of the attribute
                utility += delta * effect.weight
        return utility

    def get_possible_actions(self, character_name):
        """
        Analyzes the graph to determine possible actions for the character.

        Args:
            character_name (str): The name of the character.

        Returns:
            list: A list of possible actions and their utilities.
        """
        if character_name not in self.G.nodes:
            raise ValueError(f"No character named {character_name} in the graph.")

        possible_actions = []
        character_state = self.get_character_state(character_name)

        for neighbor in self.G.neighbors(character_name):
            neighbor_data = self.G.nodes[neighbor]
            neighbor_type = neighbor_data.get("type")

            if neighbor_type == "Character":
                character_actions = [
                    action
                    for action in self.characters[
                        neighbor.get("name")
                    ].get_possible_interactions()
                ]
                for action in character_actions:
                    # does the approaching character meet the preconditions for the action
                    if all(
                        precondition(character_state)
                        for precondition in action.preconditions
                    ):
                        possible_actions.append(
                            {
                                "name": action.name,
                                "utility": self.calculate_utility(
                                    action, character_state
                                ),
                            }
                        )

            elif neighbor_type == "Location":
                location_actions = [
                    action
                    for action in self.locations[
                        neighbor.get("name")
                    ].get_possible_interactions()
                ]
                for action in location_actions:
                    if all(
                        precondition(character_state)
                        for precondition in action.preconditions
                    ):
                        possible_actions.append(
                            {
                                "name": action.name,
                                "utility": self.calculate_utility(
                                    action, character_state
                                ),
                            }
                        )

            elif neighbor_type == "Item":
                item_actions = [
                    action
                    for action in self.items[
                        neighbor.get("name")
                    ].get_possible_interactions()
                ]
                for action in item_actions:
                    if all(
                        precondition(character_state)
                        for precondition in action.preconditions
                    ):
                        possible_actions.append(
                            {
                                "name": action.name,
                                "utility": self.calculate_utility(
                                    action, character_state
                                ),
                            }
                        )

            elif neighbor_type == "Event":
                event_actions = [
                    action
                    for action in self.events[
                        neighbor.get("name")
                    ].get_possible_interactions()
                ]
                for action in event_actions:
                    if all(
                        precondition(character_state)
                        for precondition in action.preconditions
                    ):
                        possible_actions.append(
                            {
                                "name": action.name,
                                "utility": self.calculate_utility(
                                    action, character_state
                                ),
                            }
                        )

        return possible_actions

    def get_strongest_relationships(graph, character):
        # Filter edges to find strongest relationships for a given character
        relationships = [
            (n, attrs["strength"])
            for n, attrs in graph[character].items()
            if attrs["type"] in ["friends", "family", "colleagues"]
        ]
        # Sort relationships by strength
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships[:5]  # Return top 5 strongest relationships

    def get_favorite_locations(graph, character):
        # Get locations with highest frequency of visits
        locations = [
            (n, attrs["frequency"])
            for n, attrs in graph[character].items()
            if attrs["type"] == "frequent_visitor"
        ]
        locations.sort(key=lambda x: x[1], reverse=True)
        return locations[:3]  # Return top 3 locations

    def analyze_event_impact(graph, event_node):
        # Find all characters connected to an event and evaluate impact
        participants = [(n, attrs["type"]) for n, attrs in graph[event_node].items()]
        impact_analysis = {}
        for participant, relation_type in participants:
            impact = (
                "high"
                if relation_type == "organizer"
                else "medium" if relation_type == "participant" else "low"
            )
            impact_analysis[participant] = impact
        return impact_analysis

    def explore_career_opportunities(self, character):
        # Examine professional connections and opportunities
        opportunities = []
        for node, attrs in self.G.nodes[character].items():
            if attrs["type"] == "job" and "available" in attrs and attrs["available"]:
                opportunities.append((node, attrs["opportunity"]))
        return opportunities

    def analyze_daily_preferences(graph, character):
        # Fetch locations and activities with positive outcomes
        locations = []
        activities = []
        for node, attr in graph.nodes(data=True):
            if attr["type"] == "location":
                # Check historical data for positive experiences
                if (
                    graph[character][node]["experience"] > 7
                ):  # Assuming experience is rated out of 10
                    locations.append(node)
            elif attr["type"] == "activity":
                if graph[character][node]["satisfaction"] > 7:
                    activities.append(node)

        # Consider social factors, e.g., friends going to the same location
        for friend in [
            n
            for n, attrs in graph[character].items()
            if attrs["type"] == "friend" and attrs["strength"] > 0.5
        ]:
            for location in locations:
                if (
                    graph.has_edge(friend, location)
                    and graph[friend][location]["frequency"] > 3
                ):
                    activities.append((location, "social"))

        return locations, activities

    def analyze_career_impact(graph, character, job_offer):
        # Evaluate how the new job aligns with the character's career aspirations
        current_job_node = next(
            (n for n, attrs in graph.nodes(data=True) if attrs.get("current_job")), None
        )
        new_job_node = job_offer["job_node"]

        # Assess potential benefits based on professional network expansion
        potential_connections = set(graph.neighbors(new_job_node)) - set(
            graph.neighbors(character)
        )
        career_benefit = len(potential_connections)

        # Analyze past job roles for similar positions
        past_experience_benefit = 0
        for node, attrs in graph.nodes(data=True):
            if (
                attrs["type"] == "job"
                and graph[character][node]["satisfaction"]
                and attrs["industry"] == job_offer["industry"]
            ):
                past_experience_benefit += attrs["satisfaction"]

        return career_benefit, past_experience_benefit

    def get_character(self, character_str):
        """
        Retrieve a character object from the graph based on the character's name.

        Parameters:
            character_str (str): The name of the character to retrieve.

        Returns:
            Character: The character object if found, or None if the character does not exist.
        """
        try:
            for node, data in self.G.nodes(data="name"):
                if node == character_str:
                    logging.info(
                        f"Exact character object found: '{character_str}' with data: {data}"
                    )
                    return node
                elif data == character_str:
                    logging.info(
                        f"Character found with name '{character_str}' with data: {data}"
                    )
                    return self.characters[character_str]
        except Exception as e:
            logging.error(f"Error retrieving character {character_str}: {e}")
        logging.warning(f"No character found with name '{character_str}'.")
        return None

    def get_average_attribute_value(self, attribute):
        """
        Retrieve the average value of a specific attribute across all nodes in the graph.

        Parameters:
            attribute (str): The attribute to find the average value for.

        Returns:
            float: The average value of the specified attribute.
        """
        values = [data.get(attribute, 0) for node, data in self.G.nodes(data=True)]
        return sum(values) / len(values) if values else 0

    def get_maximum_attribute_value(self, attribute):
        """
        Retrieve the maximum value of a specific attribute across all nodes in the graph.

        Parameters:
            attribute (str): The attribute to find the maximum value for.

        Returns:
            float: The maximum value of the specified attribute.
        """
        return max([data.get(attribute, 0) for node, data in self.G.nodes(data=True)])

    def get_stddev_attribute_value(self, attribute: str):
        try:
            if self.std_cache.get_std() is not None:
                return self.std_cache.get_std()
            else:
                values = [
                    data.get(attribute, 0) for node, data in self.G.nodes(data=True)
                ]
                self.std_cache.cache_std(np.std(values))
                return self.std_cache.get_std()
        except Exception as e:
            logging.error(f"Error calculating standard deviation: {e}")
            return False
        # return np.std([self.G.nodes[n][attribute] for n in self.G.nodes])

    def get_item(self, item_str):
        """
        Retrieve an item object from the graph based on the item's name.

        Parameters:
            item_str (str): The name of the item to retrieve.

        Returns:
            Item: The item object if found, or None if the item does not exist.
        """
        for node, data in self.G.nodes(data=True):
            if (data.get("type") == "object" or data.get("type") == "item") and (
                data.get("name") == item_str
            ):
                return self.objects[item_str]

        return None

    def get_location(self, location_str):
        """
        Retrieve a location object from the graph based on the location's name.

        Parameters:
            location_str (str): The name of the location to retrieve.

        Returns:
            Location: The location object if found, or None if the location does not exist.
        """
        for node, data in self.G.nodes(data=True):
            if node == location_str or data.get("name") == location_str:
                return self.locations[location_str]
        return None

    def get_event(self, event_str):
        """
        Retrieve an event object from the graph based on the event's name.

        Parameters:
            event_str (str): The name of the event to retrieve.

        Returns:
            Event: The event object if found, or None if the event does not exist.
        """
        for node, data in self.G.nodes(data=True):
            if node == event_str or data.get("name") == event_str:
                return self.events[event_str]
        return None

    def get_action(self, action_str):
        """
        Retrieve an action object from the graph based on the action's name.

        Parameters:
            action_str (str): The name of the action to retrieve.

        Returns:
            Action: The action object if found, or None if the action does not exist.
        """
        for node, data in self.G.nodes(data=True):
            if node == action_str or data.get("name") == action_str:
                return self.actions[action_str]
        return None

    def get_job(self, job_str):
        """
        Retrieve a job object from the graph based on the job's name.

        Parameters:
            job_str (str): The name of the job to retrieve.

        Returns:
            Job: The job object if found, or None if the job does not exist.
        """
        for node, data in self.G.nodes(data=True):
            if node == job_str or data.get("name") == job_str:
                return self.jobs[job_str]
        return None

    def get_node(self, node_id):
        """
        Retrieve a node from the graph based on node_id, which can be a string, a character object, an item object, etc. If node_id is a dict with a string key, it will be treated as a node data dictionary.

        Parameters:
            node_id: The identifier of the node to retrieve. This can be a string, a character object, an item object, or a dictionary.

        Returns:
            dict: The node data if found, or an empty dictionary if the node does not exist.
        """
        if isinstance(node_id, str):
            for node, data in self.G.nodes(data=True):
                if node.name == node_id or data.get("name") == node_id:
                    return data
        elif isinstance(node_id, dict):
            for node, data in self.G.nodes(data=True):
                if data == node_id or data.get("name") == node_id.get("name"):
                    return data
        return {}

    def get_filtered_nodes(self, **kwargs):
        filtered_nodes = set(self.graph.nodes)

        action_effects = kwargs.get("action_effects")
        if action_effects:
            oper = action_effects.get("operator", "ge")
            if oper not in self.ops:
                oper = self.symb_map[oper]

            for attr, value in action_effects["effects"].items():

                filtered_nodes.intersection_update(
                    {
                        n
                        for n in filtered_nodes
                        if any(
                            attr
                            in [
                                action.effects["attribute"]
                                for action in self.node_type_resolver(
                                    n
                                ).get_possible_interactions()
                                if (
                                    attr in action.effects["attribute"]
                                    and action_effects["target"]
                                    in action.effects["targets"]
                                    and self.ops[oper](
                                        action.effects["change_value"], value
                                    )
                                )
                            ]
                        )
                    }
                )
            if action_effects.get("early_quit"):
                return filtered_nodes if filtered_nodes else None

        # Filter based on node attributes
        node_attributes = kwargs.get("node_attributes", {})
        for attr, value in node_attributes.items():
            filtered_nodes.intersection_update(
                {
                    n
                    for n, attrs in self.graph.nodes(data=True)
                    if attrs.get(attr) == value
                }
            )

        # Filter based on edge attributes
        edge_attributes = kwargs.get("edge_attributes", {})
        for attr, value in edge_attributes.items():
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if any(
                        self.graph.get_edge_data(n, neighbor).get(attr) == value
                        for neighbor in self.graph.neighbors(n)
                    )
                }
            )

        # Filter based on distance
        source_node = kwargs.get("source_node")
        max_distance = kwargs.get("max_distance")
        if source_node is not None and max_distance is not None:
            lengths = nx.single_source_shortest_path_length(
                self.graph, source=source_node, cutoff=max_distance
            )
            filtered_nodes.intersection_update(lengths.keys())

        # Further filter by node type
        node_type = kwargs.get("node_type")
        if node_type is not None:
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.graph.nodes[n].get("type") == node_type
                }
            )

        # Filter by character relationships
        relationship = kwargs.get("relationship")
        if relationship is not None:
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.check_friendship_status(relationship, n) == "friends"
                }
            )

        # Filter by location safety
        safety_threshold = kwargs.get("safety_threshold")
        if safety_threshold is not None:
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.check_safety_of_locations(n) > safety_threshold
                }
            )

        # Filter by item ownership
        item = kwargs.get("item_ownership")
        if item is not None:
            filtered_nodes.intersection_update(
                {n for n in filtered_nodes if item in self.item_ownership_history(n)}
            )

        # Filter by event participation
        event = kwargs.get("event_participation")
        if event is not None:
            filtered_nodes.intersection_update(
                {n for n in filtered_nodes if self.G.has_edge(n, event)}
            )
            # check participation_status in edge attributes
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.G[n][event].get("participation_status") == True
                }
            )

        # Filter by trade opportunities
        trade_resource = kwargs.get("want_item_trade")
        if trade_resource is not None:
            if isinstance(trade_resource, str):
                trade_resource = self.items[trade_resource]
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if trade_resource
                    in self.evaluate_trade_opportunities_for_wanted_item(trade_resource)
                }
            )

        trade_resource = kwargs.get("offer_item_trade")
        if trade_resource is not None:
            if isinstance(trade_resource, ItemInventory):
                for item in trade_resource.all_items():
                    filtered_nodes.intersection_update(
                        {
                            n
                            for n in filtered_nodes
                            if item in self.evaluate_trade_opportunities_for_item(item)
                        }
                    )
            if isinstance(trade_resource, str):
                trade_resource = self.items[trade_resource]
            if isinstance(trade_resource, ItemObject):
                filtered_nodes.intersection_update(
                    {
                        n
                        for n in filtered_nodes
                        if trade_resource
                        in self.evaluate_trade_opportunities_for_item(trade_resource)
                    }
                )

        # Filter by trade opportunities based on character surplus. Argument must be a character name or instance of Character class
        trade_opportunity = kwargs.get("trade_opportunity")
        if trade_opportunity is not None:
            if isinstance(trade_opportunity, str):
                trade_opportunity = self.characters[trade_opportunity]
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if n
                    in self.evaluate_trade_opportunities_by_char_surplus(
                        trade_opportunity
                    ).keys()
                }
            )

        # Filter by desired resources of a character. Argument must be a character name or instance of Character class
        desired_resource = kwargs.get("desired_resource")
        if desired_resource is not None:
            if isinstance(desired_resource, str):
                desired_resource = self.characters[desired_resource]
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if n
                    in self.evaluate_trade_opportunities_for_desired_items(
                        desired_resource
                    ).keys()
                }
            )

        # Filter by career opportunities
        career_opportunity = kwargs.get("career_opportunity")

        if career_opportunity is not None:
            available_jobs = self.get_available_jobs()

            filtered_nodes.intersection_update(
                {
                    n
                    for n in available_jobs
                    if any(
                        n.get("job_title") in career_opportunity
                        or n.get("name") in career_opportunity
                        or n.get("required_skills") in career_opportunity
                    )
                }
            )

        # Filter by social influence
        social_influence = kwargs.get("social_influence")
        if social_influence is not None:
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.calculate_social_influence(n) > social_influence
                }
            )

        # Filter by memory influence
        memory_topic = kwargs.get("memory_topic")
        memory_influence = kwargs.get("memory_influence")
        if memory_topic is not None and memory_influence is not None:
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.query_memories(n, memory_topic) > memory_influence
                }
            )

        # Filter by attributes of the Character class. First check if there is a kwarg that is also in the character_attributes list
        if any(attr in kwargs for attr in self.character_attributes):
            character_attribute = next(
                attr for attr in character_attributes if attr in kwargs
            )
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.characters[self.G.nodes[n].get("name")]
                    .to_dict()
                    .get(character_attribute)
                }
            )

        # Filter by attributes of the Location class
        if any(attr in kwargs for attr in self.location_attributes):
            location_attribute = next(
                attr for attr in self.location_attributes if attr in kwargs
            )
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.locations[self.G.nodes[n].get("name")]
                    .to_dict()
                    .get(location_attribute)
                }
            )

        # Filter by attributes of the Event class
        if any(attr in kwargs for attr in self.event_attributes):
            event_attribute = next(
                attr for attr in self.event_attributes if attr in kwargs
            )
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.events[self.G.nodes[n].get("name")]
                    .to_dict()
                    .get(event_attribute)
                }
            )

        # Filter by attributes of the Item class
        if any(attr in kwargs for attr in self.item_attributes):
            item_attribute = next(
                attr for attr in self.item_attributes if attr in kwargs
            )
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.items[self.G.nodes[n].get("name")]
                    .to_dict()
                    .get(item_attribute)
                }
            )

        # Filter by attributes of the Activity class
        if any(attr in kwargs for attr in self.activity_attributes):
            activity_attribute = next(
                attr for attr in self.activity_attributes if attr in kwargs
            )
            filtered_nodes.intersection_update(
                {
                    n
                    for n in filtered_nodes
                    if self.activities[self.G.nodes[n].get("name")]
                    .to_dict()
                    .get(activity_attribute)
                }
            )

        return {
            n: self.graph.nodes[n] for n in filtered_nodes
        }  # return dict will look like: {"node1": {"type": "item", "item_type": "food"}, "node2": {"type": "item", "item_type": "food"}}

    def get_available_jobs(self):
        """
        Retrieve a list of available jobs from the graph.

        Returns:
            list: A list of available job nodes.
        """
        return [
            node
            for node, data in self.G.nodes(data=True)
            if data.get("type") == "job" and data.get("available") == True
        ]

    def determine_job_eligibility(self, character, job):
        """
        Determine if a character is eligible for a specific job based on their skills and attributes.
        Args:
            character (Character): The character applying for the job.
            job (Job): The job being applied for.
        Returns:
            bool: True if the character is eligible for the job, False otherwise.
        """
        # Check if the character has the required skills
        if (
            job.available
            and all(
                skill in character.get_skills_as_list_of_strings()
                for skill in job.required_skills
            )
            and character.experience >= job.req_job_experience
        ):
            return True
        return False

    def calculate_potential_utility_of_plan(self, character, plan):
        """
        Calculate the range possible utility of pursuing a Goal for a character based on their current state and preferences, given the current state of the graph.
        Uses the graph to analyze relationships, locations, and other factors that may influence the utility of the goal.
        Also considers the character's past experiences and preferences to determine the potential satisfaction level.
        Using a combination of graph analysis and character-specific attributes, the utility of the goal is evaluated.
        The get_filtered_nodes method can be used to retrieve relevant nodes based on specific criteria.
           Args:
               character (str): The character for whom the utility is being evaluated.
               goal (Goal): The goal to evaluate.
           Returns:
               float: The estimated utility value of pursuing the goal.
        """
        utility = 0

        # # Analyze the goal and its requirements
        # # goal_requirements = goal.criteria
        # # Filter nodes based on goal requirements
        # filtered_nodes = self.get_filtered_nodes(**goal_requirements)
        # # Analyze the filtered nodes to determine potential utility
        # for node_id, node_data in filtered_nodes.items():
        #     # Evaluate the utility of each node based on its attributes and relationships
        #     node_utility = self.evaluate_node_utility(character, node_data)
        #     utility += node_utility
        return utility

    def evaluate_combination(self, combo, action_viability_cost, goal_requirements):
        """
        Evaluate a combination of actions to determine if they fulfill all goal requirements.

        Args:
            combo (tuple of tuples): The combination of actions to evaluate.
            action_viability_cost (dict): The viability and cost data for actions.
            goal_requirements (dict): The goal requirements to fulfill.

        Returns:
            tuple: (total_cost, combo) if valid, else None.
        """
        fulfilled_conditions = set()
        redundant_fullfilled_conditions = []
        total_cost = 0
        necessary_actions = []
        num_actions_per_condition = {}

        for i, (node, action) in enumerate(combo):
            num_actions_per_condition.update(
                {
                    condition: len(actions)
                    for condition, actions in action_viability_cost[node][
                        "actions_that_fulfill_condition"
                    ].items()
                }
            )
            for condition in action_viability_cost[node][
                "actions_that_fulfill_condition"
            ]:
                if (
                    action
                    in action_viability_cost[node]["actions_that_fulfill_condition"][
                        condition
                    ]
                ):
                    if condition in fulfilled_conditions:
                        redundant_fullfilled_conditions.append(condition)
            fulfilled_conditions.update(
                action_viability_cost[node]["conditions_fulfilled_by_action"][action]
            )
            total_cost += sum(action_viability_cost[node]["goal_cost"][action])
            if all(cond in fulfilled_conditions for cond in goal_requirements):
                necessary_actions = combo[: i + 1]
                break

        if all(cond in fulfilled_conditions for cond in goal_requirements):
            if len(necessary_actions) == 0:
                necessary_actions = combo
            # Calculate the number of actions that can fulfill each condition

            # Calculate the total cost
            for condition in redundant_fullfilled_conditions:
                # Get the number of actions that can fulfill this condition
                num_actions = num_actions_per_condition.get(condition, 1)

                # Add a penalty proportional to the number of redundant conditions and inversely proportional to both the number of goal requirements and the number of actions that can fulfill the condition
                total_cost += (
                    math.log(len(redundant_fullfilled_conditions) + 1)
                    * 10
                    / (len(goal_requirements) * num_actions)
                )
            return total_cost, tuple(necessary_actions)
        return None

    def initialize_cache(self, action_viability_cost):
        action_cache = {}
        for node in action_viability_cost:
            for action in action_viability_cost[node]["conditions_fulfilled_by_action"]:
                new_conditions = set(
                    action_viability_cost[node]["conditions_fulfilled_by_action"][
                        action
                    ]
                )
                new_cost = action_viability_cost[node]["goal_cost"][action]
                action_cache[(node, action)] = (new_conditions, new_cost)
        return action_cache

    def calculate_goal_difficulty(self, goal: Goal, character: Character):
        """
        Calculate the difficulty of a goal based on its complexity and requirements.
        Args:
            goal (Goal): The goal to evaluate.
            character (Character): The character attempting the goal.
        Returns:
            dict: A dictionary containing the difficulty score and additional metrics.
        """
        try:
            Character = importlib.import_module("tiny_characters").Character

        except ImportError:
            logging.error(
                "Failed to import Character class from tiny_characters module."
            )
            difficulty = 0
            # Analyze the goal requirements and complexity
            if not hasattr(goal, "criteria") or not goal.criteria:
                return {"difficulty": 0, "error": "Goal has no criteria"}

            goal_requirements = goal.criteria  # goal_requirements will look like: {
        #     "node_attributes": {"type": "item", "item_type": "food"},
        #     "max_distance": 20,
        # }
        # Analyze graph to identify nodes that match the goal criteria
        nodes_per_requirement = {}
        for requirement in goal_requirements:
            nodes_per_requirement[requirement] = self.get_filtered_nodes(**requirement)

        # nodes_per_requirement will look like this:
        # {
        #    "node_attributes": {"type": "item", "item_type": "food"}: {
        #        "item1": {"type": "item", "item_type": "food"},
        # Check if any requirement has no matching nodes

        for requirement, nodes in nodes_per_requirement.items():
            if not nodes:
                return float("inf")

        # Check for nodes that fulfill multiple requirements
        fulfill_multiple = {}
        action_viability_cost = {}
        ## self.calculate_action_viability_cost returns as below
        # {
        #     "action_cost": action_cost, # A dict of costs for each action
        #     "viable": viable, # A dict of booleans indicating viability of each action
        #     "goal_cost": goal_cost, # A dict of goal costs for each action based on goal requirements
        # "conditions_fulfilled_by_action": conditions_fulfilled_by_action, # A dict of goal conditions fulfilled by each action
        #   "actions_that_fulfill_condition": actions_that_fulfill_condition, # A dict of actions that fulfill each condition
        # }

        for nodes in nodes_per_requirement.values():
            for node in nodes:
                action_viability_cost[node] = self.calculate_action_viability_cost(
                    node, goal, character
                )
                if sum(node in nodes for nodes in nodes_per_requirement.values()) > 1:
                    # Node fulfills multiple requirements
                    # Add your code here to handle this case
                    # For example, you can store the node in a list or perform some other operation
                    fulfill_multiple[node] = {
                        "num_reqs": sum(
                            node in nodes for nodes in nodes_per_requirement.values()
                        )
                    }
                    # fulfill_multiple[node]["action_viability_cost"] = self.calculate_action_viability_cost(
        # Build condition_actions mapping for A* heuristic
        condition_actions = {}
        goal_conditions = set()
        for node in action_viability_cost:
            for condition, actions in action_viability_cost[node][
                "actions_that_fulfill_condition"
            ].items():
                goal_conditions.add(condition)
                if condition not in condition_actions:
                    condition_actions[condition] = []
                for action in actions:
                    condition_actions[condition].append((action, node))

        remaining_conditions = goal_conditions.copy()
        # Determine the number of viable actions and nodes and compare that to the total number of actions and nodes
        # Estimate counts
        total_nodes = len(action_viability_cost)
        total_actions_per_node = (
            sum(
                len(node["actions_that_fulfill_condition"])
                for node in action_viability_cost.values()
            )
            / total_nodes
        )

        viable_nodes = sum(
            1
            for node in action_viability_cost
            if any(action_viability_cost[node]["viable"].values())
        )
        viable_actions_per_node = (
            sum(sum(node["viable"].values()) for node in action_viability_cost.values())
            / viable_nodes
            if viable_nodes > 0
            else 0
        )
        initial_solution = []
        initial_cost = 0

        # Use greedy approach if we have good viable action coverage
        # Switch to A* if the problem is complex (low viability ratio)
        viability_ratio = viable_nodes / total_nodes if total_nodes > 0 else 0
        complexity_threshold = 0.5  # Use greedy if >50% of nodes are viable

        if viability_ratio >= complexity_threshold and viable_actions_per_node > 0:
            # Greedy initial solution
            for condition in goal_conditions:
                if condition not in condition_actions:
                    continue

                viable_actions = [
                    (action, node)
                    for action, node in condition_actions[condition]
                    if action_viability_cost[node]["viable"].get(action, False)
                ]

                if not viable_actions:
                    # No viable actions for this condition, goal is impossible
                    return float("inf")

                # Find the best action for this condition
                best_action, best_node = None, None
                min_cost = float("inf")

                for action, node in viable_actions:
                    action_cost_data = action_viability_cost[node]
                    total_cost = action_cost_data["action_cost"].get(
                        action, 0
                    ) + action_cost_data["goal_cost"].get(action, 0)
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_action, best_node = action, node

                if best_action is not None:
                    initial_solution.append((best_node, best_action))
                    initial_cost += min_cost
        else:
            # Use more sophisticated search for complex scenarios
            selected_nodes = []
            selected_actions = []
            total_cost = 0
            action_cache = {}
            max_iterations = len(goal_conditions) * 2  # Prevent infinite loops

            iteration_count = 0
            while remaining_conditions and iteration_count < max_iterations:
                iteration_count += 1
                best_node_action = None
                best_new_conditions = set()
                best_cost = float("inf")

                for node in action_viability_cost:
                    if not any(action_viability_cost[node]["viable"].values()):
                        continue

                    for action in action_viability_cost[node][
                        "conditions_fulfilled_by_action"
                    ]:
                        if not action_viability_cost[node]["viable"].get(action, False):
                            continue

                        if (node, action) in action_cache:
                            new_conditions, new_cost = action_cache[(node, action)]
                        else:
                            new_conditions = {
                                condition
                                for condition in action_viability_cost[node][
                                    "conditions_fulfilled_by_action"
                                ][action]
                                if condition in remaining_conditions
                            }
                            new_cost = action_viability_cost[node]["goal_cost"].get(
                                action, 0
                            )
                            action_cache[(node, action)] = (new_conditions, new_cost)

                        if len(new_conditions) > len(best_new_conditions) or (
                            len(new_conditions) == len(best_new_conditions)
                            and new_cost < best_cost
                        ):
                            best_node_action = (node, action)
                            best_new_conditions = new_conditions
                            best_cost = new_cost

                if best_node_action and best_new_conditions:
                    selected_nodes.append(best_node_action[0])
                    selected_actions.append(best_node_action[1])
                    remaining_conditions.difference_update(best_new_conditions)
                    total_cost += best_cost
                else:
                    # No progress possible, goal might be impossible
                    break

            if remaining_conditions:
                # Some conditions couldn't be satisfied
                return float("inf")

            initial_solution = list(zip(selected_nodes, selected_actions))
            initial_cost = total_cost

        # A* Search logic with priority queue
        def heuristic(remaining_conditions):
            if not remaining_conditions:
                return 0
            total = 0
            for condition in remaining_conditions:
                if condition in condition_actions:
                    try:
                        min_cost = min(
                            action_viability_cost[node]["goal_cost"].get(action, 0)
                            for action, node in condition_actions[condition]
                            if node in action_viability_cost
                            and action in action_viability_cost[node]["goal_cost"]
                        )
                        total += min_cost
                    except (ValueError, KeyError):
                        total += 1  # Default cost if calculation fails
            return total

        best_cost = initial_cost
        best_solution = initial_solution

        priority_queue = [
            (
                initial_cost + heuristic(set(goal_conditions)),
                initial_cost,
                initial_solution,
                set(goal_conditions),
            )
        ]

        # Add limits to prevent infinite loops
        max_iterations = 1000
        iterations = 0

        while priority_queue and iterations < max_iterations:
            iterations += 1
            _, current_cost, current_solution, remaining_conditions = heapq.heappop(
                priority_queue
            )

            if not remaining_conditions:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_solution
                continue

            for condition in remaining_conditions:
                if condition not in condition_actions:
                    continue

                for action, node in condition_actions[condition]:
                    if any(node == n for n, _ in current_solution):
                        continue
                    new_solution = current_solution + [(node, action)]

                    try:
                        new_cost = current_cost + action_viability_cost[node][
                            "goal_cost"
                        ].get(action, 0)
                    except (KeyError, TypeError):
                        continue

                    new_remaining_conditions = remaining_conditions - {condition}

                    if new_cost >= best_cost:
                        continue

                    heuristic_cost = new_cost + heuristic(new_remaining_conditions)
                    if heuristic_cost >= best_cost:
                        continue
                    heapq.heappush(
                        priority_queue,
                        (
                            heuristic_cost,
                            new_cost,
                            new_solution,
                            new_remaining_conditions,
                        ),
                    )

        valid_combinations = [best_solution]

        # Use regular iteration instead of ProcessPoolExecutor for better compatibility
        try:
            for combo in combinations(best_solution, len(best_solution)):
                try:
                    result = self.evaluate_combination(
                        combo,
                        action_viability_cost,
                        goal_conditions,
                    )
                    if result:
                        valid_combinations.append(result)
                except Exception as e:
                    print(f"Error evaluating combination {combo}: {e}")
        except Exception as e:
            print(f"Error in combination evaluation: {e}")

        valid_combinations.sort(key=lambda x: x[0])  # Sort by total cost

        valid_paths = []
        # Now calculate paths that would fulfill each requirement in goal_requirements, using only one node per condition
        # Generate all combinations of nodes fulfilling the requirements
        all_pairs_paths = dict(
            nx.all_pairs_dijkstra_path(
                nx.subgraph(
                    self.G,
                    [n for nodes in nodes_per_requirement.values() for n in nodes],
                ),
                weight=self.calculate_edge_cost,
            )
        )

        def find_combined_path(combo):
            combined_path = []
            for i in range(len(combo) - 1):
                source, target = combo[i], combo[i + 1]
                if source in all_pairs_paths and target in all_pairs_paths[source]:
                    combined_path.extend(all_pairs_paths[source][target])
                else:
                    return None
            return combined_path

        for combo in valid_combinations:
            path = find_combined_path(combo)
            if path:
                valid_paths.append(path)

        # Incorporate action costs into path cost calculation
        def calc_path_cost(path):
            if not path or len(path) < 2:
                return 0
            total_cost = 0
            for i in range(len(path) - 1):
                # Add edge cost
                try:
                    edge_cost = self.calculate_edge_cost(path[i], path[i + 1])
                    total_cost += edge_cost
                except Exception:
                    total_cost += 1  # Default edge cost

                # Add action cost for the current node
                node = path[i]
                if node in action_viability_cost:
                    action_costs = action_viability_cost[node].get("action_cost", {})
                    if action_costs:
                        # Take the minimum action cost for this node
                        total_cost += min(action_costs.values())
            return total_cost

        # Filter out non-viable paths
        viable_paths = []
        for path in valid_paths:
            is_viable = True
            for node in path:
                if node in action_viability_cost:
                    if not any(action_viability_cost[node]["viable"].values()):
                        is_viable = False
                        break
            if is_viable:
                viable_paths.append(path)

        # Handle case where no viable paths exist
        if not viable_paths:
            return {
                "difficulty": float("inf"),
                "viable_paths": [],
                "error": "No viable paths found",
            }

        # Use goal costs to prioritize paths
        def calc_goal_cost(path):
            # Sum goal_cost values for each node in the path
            total = 0
            for node in path:
                if node in action_viability_cost:
                    # Sum all goal_costs for the node
                    total += sum(
                        action_viability_cost[node].get("goal_cost", {}).values()
                    )
            return total

        # Find the shortest path among all viable paths based on the weighted sum of edge and action costs
        try:
            shortest_path = min(viable_paths, key=calc_path_cost)
            # Calculate difficulty based on the sum of edge and action costs
            difficulty = calc_path_cost(shortest_path)

            # Prioritize paths with lower goal costs
            lowest_goal_cost_path = min(viable_paths, key=calc_goal_cost)
            # If the shortest path is not the same as the path with the lowest goal cost, increase the difficulty
            # Calculate the goal cost of the shortest path
            shortest_path_goal_cost = calc_goal_cost(shortest_path)

            # Calculate the path cost of the path with the lowest goal cost
            lowest_goal_cost_path_cost = calc_path_cost(lowest_goal_cost_path)

            # If the shortest path is not the same as the path with the lowest goal cost, increase the difficulty
            if shortest_path != lowest_goal_cost_path:
                difficulty += shortest_path_goal_cost + lowest_goal_cost_path_cost / 2
        except (ValueError, TypeError) as e:
            return {
                "difficulty": float("inf"),
                "viable_paths": viable_paths,
                "error": f"Path calculation error: {e}",
            }
        return {
            "difficulty": difficulty,
            "calc_path_cost": calc_path_cost,
            "calc_goal_cost": calc_goal_cost,
            "action_viability_cost": action_viability_cost,
            "viable_paths": viable_paths,
            "shortest_path": shortest_path,
            "lowest_goal_cost_path": lowest_goal_cost_path,
            "shortest_path_goal_cost": shortest_path_goal_cost,
            "lowest_goal_cost_path_cost": lowest_goal_cost_path_cost,
        }

    def is_goal_achieved(self, character, goal: Goal):
        """
        Check if a goal is achieved by evaluating its completion conditions.

        Args:
            character: The character for whom to check goal achievement
            goal (Goal): The goal to check for completion

        Returns:
            bool: True if goal is achieved, False otherwise
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
            Goal = importlib.import_module("tiny_characters").Goal
            State = importlib.import_module("actions").State

            # Get character's current state
            character_state = State(character)

            # Check if all completion conditions are met
            if hasattr(goal, "completion_conditions") and goal.completion_conditions:
                for condition in goal.completion_conditions:
                    if not condition.check_condition(character_state):
                        return False
                return True

            # Fallback: check based on goal priority and description
            if hasattr(goal, "priority") and goal.priority >= 10:
                # High priority goals might be considered "achieved" if certain thresholds are met
                if "wealth" in goal.description.lower():
                    return character.wealth_money >= 500
                elif "social" in goal.description.lower():
                    return character.social_wellbeing >= 80
                elif "health" in goal.description.lower():
                    return character.health_status >= 90
                elif "food" in goal.description.lower():
                    return character.hunger_level <= 20
                elif (
                    "job" in goal.description.lower()
                    or "career" in goal.description.lower()
                ):
                    return character.job_performance >= 80

            # Default: goal not achieved
            return False

        except Exception as e:
            logging.error(f"Error checking goal achievement for {character.name}: {e}")
            return False

    def calculate_preconditions_difficulty(self, preconditions, initiator: Character):
        """
        Calculate the difficulty of a set of preconditions based on the actions required to satisfy them.
        Args:
            preconditions (list): The preconditions to evaluate.
            initiator (Character): The character initiating the actions.
        Returns:
            float: The difficulty score of the preconditions.
        """
        Character = importlib.import_module("tiny_characters").Character

        action_cost = {}
        char_possible_actions = self.get_possible_actions(initiator)
        for precondition in preconditions:
            if not precondition().check_condition():
                filters = {}
                # use get_filtered_nodes to get the nodes that satisfy the preconditions
                if "inventory.check_has_item_by_type" in precondition["attribute"]:
                    args = (
                        re.search(r"\[.*\]", precondition["attribute"])
                        .group()
                        .strip("[]")
                    )
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["item_type"] for item in filters]:
                        filters.append(({"item_type": args[0]}))
                elif "inventory.check_has_item_by_name" in precondition["attribute"]:
                    args = (
                        re.search(r"\[.*\]", precondition["attribute"])
                        .group()
                        .strip("[]")
                    )
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["name"] for item in filters]:
                        filters.append(({"name": args[0]}))
                elif "inventory.check_has_item_by_value" in precondition["attribute"]:
                    args = (
                        re.search(r"\[.*\]", precondition["attribute"])
                        .group()
                        .strip("[]")
                    )
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["value"] for item in filters]:
                        filters.append(({"value": args[0]}))
                elif (
                    "inventory.check_has_item_by_quantity" in precondition["attribute"]
                ):
                    args = (
                        re.search(r"\[.*\]", precondition["attribute"])
                        .group()
                        .strip("[]")
                    )
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["quantity"] for item in filters]:
                        filters.append(({"quantity": args[0]}))
                elif precondition["attribute"] in character_attributes:
                    # Find nodes with possible_interactions that have Actions.effects with the same attribute and a change_value that satisfies the precondition
                    filters = {
                        "action_effects": {
                            "effects": {
                                precondition["attribute"]: precondition["satisfy_value"]
                            },
                            "operator": precondition.operator,
                            "early_quit": True,
                            "target": (
                                "initiator"
                                if precondition.target == initiator
                                else "target"
                            ),
                        }
                    }
                nodes = None
                if "action_effects" in filters.keys():
                    nodes = self.get_filtered_nodes(filters)
                elif "action_effects" not in filters.keys() or not nodes:
                    nodes = self.get_filtered_nodes({"node_attributes": filters})
                if not nodes:
                    return float("inf")

                for node in nodes:
                    action_costs = {}
                    for interaction in self.node_type_resolver(
                        node
                    ).get_possible_interactions():
                        if interaction in char_possible_actions:
                            action_costs[interaction] = (
                                interaction.cost
                                + self.calculate_action_effect_cost(
                                    interaction, initiator, precondition
                                )
                                + self.calculate_char_act_edge_cost(
                                    initiator, node, interaction
                                )
                            )

                    action_cost[node] = min(action_costs.values())

        return sum(action_cost.values())

    def calculate_action_effect_cost(
        self, action: Action, character: Character, goal: Goal
    ):
        """
        Calculate the cost of an action based on the effects it will have on the character.
        Args:
            action (Action): The action to evaluate.
        Returns:
            float: The cost of the action based on its effects.
        """
        Character = importlib.import_module("tiny_characters").Character
        Action = importlib.import_module("tiny_actions").Action
        Goal = importlib.import_module("tiny_goals").Goal

        goal_cost = 0
        conditions = goal.completion_conditions
        weights = {}
        for effect in action.effects:
            # In a creative way, we calculate the weight of the effect based on the character's current state and the goal conditions. We do this by checking if the effect will help or hinder the character in achieving the goal.
            weights[effect["attribute"]] = 1
            for condition in conditions:
                if (
                    effect["attribute"] in character.get_state()
                    and effect["targets"] == "initiator"
                ):
                    if (
                        condition.attribute == effect["attribute"]
                        and condition.target == character
                    ):
                        if (
                            condition.operator == "ge" or condition.operator == "gt"
                        ) and effect["change_value"] > 0:
                            delta = (
                                character.get_state()[effect["attribute"]]
                                + effect["change_value"]
                            ) - condition.satisfy_value

                            # Consider the magnitude of effect["change_value"]
                            delta = abs(effect["change_value"]) * delta

                            # Use a different scaling factor
                            scaling_factor = condition.weight / sum(
                                condition.weight for condition in conditions
                            )

                            # Use a non-linear function for the difference
                            delta = 1 / (1 + math.exp(-delta))

                            weights[effect["attribute"]] += delta * scaling_factor
                        elif (
                            condition.operator == "le" or condition.operator == "lt"
                        ) and effect["change_value"] < 0:
                            delta = condition.satisfy_value - (
                                character.get_state()[effect["attribute"]]
                                + effect["change_value"]
                            )
                            # Consider the magnitude of effect["change_value"]
                            delta = abs(effect["change_value"]) * delta

                            # Use a different scaling factor
                            scaling_factor = condition.weight / sum(
                                condition.weight for condition in conditions
                            )

                            # Use a non-linear function for the difference
                            delta = 1 / (1 + math.exp(-delta))

                            weights[effect["attribute"]] += delta * scaling_factor
                elif (
                    effect["attribute"] in condition.target.get_state()
                    and effect["targets"] == "target"
                ):
                    if (
                        condition.operator == "ge" or condition.operator == "gt"
                    ) and effect["change_value"] > 0:
                        delta = (
                            condition.target.get_state()[effect["attribute"]]
                            + effect["change_value"]
                        ) - condition.satisfy_value
                        # Consider the magnitude of effect["change_value"]
                        delta = abs(effect["change_value"]) * delta

                        # Use a different scaling factor
                        scaling_factor = condition.weight / sum(
                            condition.weight for condition in conditions
                        )

                        # Use a non-linear function for the difference
                        delta = 1 / (1 + math.exp(-delta))

                        weights[effect["attribute"]] += delta * scaling_factor

                    elif (
                        condition.operator == "le" or condition.operator == "lt"
                    ) and effect["change_value"] < 0:
                        delta = condition.satisfy_value - (
                            condition.target.get_state()[effect["attribute"]]
                            + effect["change_value"]
                        )
                        # Consider the magnitude of effect["change_value"]
                        delta = abs(effect["change_value"]) * delta

                        # Use a different scaling factor
                        scaling_factor = condition.weight / sum(
                            condition.weight for condition in conditions
                        )

                        # Use a non-linear function for the difference
                        delta = 1 / (1 + math.exp(-delta))

                        weights[effect["attribute"]] += delta * scaling_factor

        # After calculating all the weights
        weights_array = np.array(list(weights.values()))
        softmax_weights = np.exp(weights_array) / np.sum(np.exp(weights_array))

        for i, attribute in enumerate(weights):
            weights[attribute] = softmax_weights[i]

        for effect in action.effects:
            # In a creative way, we calculate the weight of the effect based on the character's current state and the goal conditions. We do this by checking if the effect will help or hinder the character in achieving the goal.
            for condition in conditions:
                if (
                    effect["attribute"] in character.get_state()
                    and effect["targets"] == "initiator"
                ):
                    if (
                        condition.attribute == effect["attribute"]
                        and condition.target == character
                    ):
                        if (
                            condition.operator == "ge" or condition.operator == "gt"
                        ) and effect["change_value"] < 0:
                            delta = (
                                condition.satisfy_value
                                - character.get_state()[effect["attribute"]]
                                + abs(effect["change_value"])
                            )
                            # Consider the magnitude of effect["change_value"]
                            delta = abs(effect["change_value"]) * delta

                            # Use a different scaling factor
                            try:
                                scaling_factor = 1 / weights[effect["attribute"]]
                            except ZeroDivisionError:
                                scaling_factor = float("inf")  # or some large number

                            # The rest of the code remains the same

                            # Use a non-linear function for the difference
                            delta = 1 / (1 + math.exp(-delta))

                            goal_cost += delta * scaling_factor

                        elif (
                            condition.operator == "le" or condition.operator == "lt"
                        ) and effect["change_value"] > 0:
                            delta = (
                                character.get_state()[effect["attribute"]]
                                + effect["change_value"]
                                - condition.satisfy_value
                            )
                            # Consider the magnitude of effect["change_value"]
                            delta = abs(effect["change_value"]) * delta

                            # Use a different scaling factor
                            try:
                                scaling_factor = 1 / weights[effect["attribute"]]
                            except ZeroDivisionError:
                                scaling_factor = float("inf")  # or some large number
                            # The rest of the code remains the same
                            # Use a non-linear function for the difference
                            delta = 1 / (1 + math.exp(-delta))

                            goal_cost += delta * scaling_factor
                elif (
                    effect["attribute"] in condition.target.get_state()
                    and effect["targets"] == "target"
                ):
                    if (
                        condition.operator == "ge" or condition.operator == "gt"
                    ) and effect["change_value"] < 0:
                        delta = (
                            condition.satisfy_value
                            - condition.target.get_state()[effect["attribute"]]
                            + abs(effect["change_value"])
                        )
                        # Consider the magnitude of effect["change_value"]
                        delta = abs(effect["change_value"]) * delta

                        # Use a different scaling factor
                        try:
                            scaling_factor = 1 / weights[effect["attribute"]]
                        except ZeroDivisionError:
                            scaling_factor = float("inf")  # or some large number

                        # The rest of the code remains the same

                        # Use a non-linear function for the difference
                        delta = 1 / (1 + math.exp(-delta))

                        goal_cost += delta * scaling_factor
                    elif (
                        condition.operator == "le" or condition.operator == "lt"
                    ) and effect["change_value"] > 0:
                        delta = (
                            condition.target.get_state()[effect["attribute"]]
                            + effect["change_value"]
                            - condition.satisfy_value
                        )
                        # Consider the magnitude of effect["change_value"]
                        delta = abs(effect["change_value"]) * delta

                        # Use a different scaling factor
                        try:
                            scaling_factor = 1 / weights[effect["attribute"]]
                        except ZeroDivisionError:
                            scaling_factor = float("inf")  # or some large number

                        # The rest of the code remains the same

                        # Use a non-linear function for the difference
                        delta = 1 / (1 + math.exp(-delta))

                        goal_cost += delta * scaling_factor

        # At the end of the function
        goal_cost = 1 / (1 + math.exp(-goal_cost))

        return goal_cost

    def calculate_how_goal_impacts_character(self, goal: Goal, character: Character):
        """
        Calculate the impact of a goal on a character based on the goal's completion conditions and the character's current state.
        Args:
            goal (Goal): The goal to evaluate.
            character (Character): The character to evaluate.
        Returns:
            float: The impact of the goal on the character.
        """
        Character = importlib.import_module("tiny_characters").Character

        impact = 0
        for condition in goal.completion_conditions:
            if condition.attribute in character.get_state():
                if condition.operator == "ge" or condition.operator == "gt":
                    impact += max(
                        0,
                        character.get_state()[condition.attribute]
                        - condition.satisfy_value,
                    )
                elif condition.operator == "le" or condition.operator == "lt":
                    impact += max(
                        0,
                        condition.satisfy_value
                        - character.get_state()[condition.attribute],
                    )
        return impact

    def calculate_action_difficulty(self, action: Action, character: Character):
        """
        Calculate the difficulty of an action based on its complexity and requirements.
        Args:
            action (Action): The action to evaluate.
        Returns:
            float: The difficulty score of the action.
        """
        Character = importlib.import_module("tiny_characters").Character
        Action = importlib.import_module("tiny_actions").Action

        difficulty = action.cost
        difficulty += self.calculate_preconditions_difficulty(
            action.preconditions, character
        )
        difficulty += self.calculate_char_act_edge_cost(
            character, action, self.node_type_resolver(action.target)
        )

        return difficulty

    @lru_cache(maxsize=1000)
    def calculate_action_viability_cost(self, node, goal: Goal, character: Character):
        """
        Calculate the cost of an action based on the viability of the node.
        Args:
            node (Node): The node to evaluate.
        Returns:
            dict(cost: float, viable: bool, goal_cost: float): The cost of the action, whether it is viable, and the cost of the action on the goal.
        """
        Character = importlib.import_module("tiny_characters").Character

        cache_key = (node, goal, character)
        if cache_key in self.dp_cache:
            return self.dp_cache[cache_key]
        if isinstance(node, str):
            node = self.G.nodes[node]

        if not isinstance(node, list):
            node = [node]

        viable = {}
        action_cost = {}  # Cost of action (Action.cost) for each action in the node
        goal_cost = (
            {}
        )  # goal_cost represents the cost the action levies on progress toward the goal, ie the effect "costs" progress toward the goal,
        possible_interactions = []
        try:
            possible_interactions = self.node_type_resolver(
                node
            ).get_possible_interactions()
        except:
            possible_interactions = self.node_type_resolver(node).possible_interactions
        actions_that_fulfill_condition = {}  # {condition: [actions]}
        conditions_fulfilled_by_action = {}  # {action: [conditions]}
        for interaction in possible_interactions:
            if interaction.preconditions_met():
                fulfilled_conditions = self.will_action_fulfill_goal(
                    interaction,
                    goal,
                    (
                        self.node_type_resolver(node).get_state()
                        if interaction.target == None
                        else self.node_type_resolver(interaction.target).get_state()
                    ),
                    character,
                )
                for condition, fulfilled in fulfilled_conditions.items():
                    if fulfilled:
                        conditions_fulfilled_by_action.setdefault(
                            tuple([interaction, node]), []
                        ).append(condition)
                        actions_that_fulfill_condition.setdefault(condition, []).append(
                            tuple([interaction, node])
                        )

                action_cost[interaction] = interaction.cost
                goal_cost[interaction] += self.calculate_action_effect_cost(
                    interaction, character, goal
                )
                if any(fulfilled_conditions.values()):
                    viable[interaction] = True
                else:
                    viable[interaction] = False
                if all(fulfilled_conditions.values()):
                    result = {
                        "action_cost": action_cost,
                        "viable": viable,
                        "goal_cost": goal_cost,
                        "conditions_fulfilled_by_action": conditions_fulfilled_by_action,
                        "actions_that_fulfill_condition": actions_that_fulfill_condition,
                    }
                    self.dp_cache[cache_key] = result
                    return result
            else:
                action_cost[interaction] += self.calculate_action_difficulty(
                    interaction, character
                )
                goal_cost[interaction] += self.calculate_action_effect_cost(
                    interaction, character, goal
                )
                viable[interaction] = False

        result = {
            "action_cost": action_cost,
            "viable": viable,
            "goal_cost": goal_cost,
            "conditions_fulfilled_by_action": conditions_fulfilled_by_action,
            "actions_that_fulfill_condition": actions_that_fulfill_condition,
        }

        self.dp_cache[cache_key] = result
        return result

    def will_action_fulfill_goal(
        self, action: Action, goal: Goal, current_state: State, character: Character
    ):
        """
        Determine if an action fulfills the specified goal by checking the completion conditions.
        Remember, Goals can have multiple completion conditions, so we will return a dictionary of booleans indicating whether each completion condition is met.
        Args:
            action (Action): The action to evaluate.
            goal (Goal): The goal to achieve.
            current_state (State): The current state of the target of the action.
        Returns:
            dict: A dictionary of booleans indicating whether each completion condition is met.
        """
        Character = importlib.import_module("tiny_characters").Character
        Action = importlib.import_module("tiny_actions").Action
        Goal = importlib.import_module("tiny_goals").Goal
        State = importlib.import_module("tiny_states").State
        goal_copy = copy.deepcopy(goal)
        completion_conditions = (
            goal_copy.completion_conditions
        )  # Example: {False: Condition(name="has_food", attribute="inventory.check_has_item_by_type(['food'])", satisfy_value=True, op="==")}
        action_effects = action.effects  # Example [
        #     {"targets": ["initiator"], "attribute": "inventory", "change_value": "add_item('food')"},
        #     {
        #         "targets": ["initiator"],
        #         "method": "play_animation",
        #         "method_args": ["taking_item"],
        #     },
        # ]
        goal_target = goal_copy.target
        # make a copy of the State so the original is not modified
        current_state_ = copy.deepcopy(current_state)
        for effect in action_effects:
            if (
                "target" in effect["targets"]
                and goal_target.name == current_state_.name
            ):
                current_state_ = action.apply_single_effect(effect, current_state_)
                for condition in completion_conditions[False]:
                    check = condition.check_condition(current_state_)
                    # if check is true, change k (the key) to True in the completion_conditions dictionary
                    if check == True:
                        # remove the condition from the completion_conditions dictionary
                        completion_conditions[False] = None
                        completion_conditions[True] = condition

            elif (
                "initiator" in effect["targets"] and goal_target.name == character.name
            ):
                current_state_ = action.apply_single_effect(effect, current_state_)
                for condition in completion_conditions[False]:
                    check = condition.check_condition(current_state_)
                    # if check is true, change k (the key) to True in the completion_conditions dictionary
                    if check == True:
                        # remove the condition from the completion_conditions dictionary
                        completion_conditions[False] = None
                        completion_conditions[True] = condition

        # return the reversed completion_conditions dictionary
        return {v: k for k, v in completion_conditions.items()}

    ##TODO: Figure out how to check if the effects of all actions the action list fulfill the goal completion_conditions together after being applied to the character State

    def will_path_achieve_goal(self, path, goal: Goal):
        """
        Determine if a given path through the graph will achieve the specified goal by checking the preconditions and effects of the Action objects in the possible_interactions attribute of each node.
        Args:
            path (list): The path through the graph.
            goal (Goal): The goal to achieve.
        Returns:
            bool: True if the path fulfills the goal, False otherwise.
        """
        try:
            import copy
            from actions import State

            # Initialize current state from the goal's initial state
            current_state = (
                copy.deepcopy(goal.current_state)
                if hasattr(goal, "current_state")
                else {}
            )

            # Track which goal conditions have been fulfilled
            goal_conditions = set()
            if hasattr(goal, "completion_conditions") and goal.completion_conditions:
                for condition in goal.completion_conditions:
                    condition_key = f"{condition.attribute}_{condition.satisfy_value}_{condition.op}"
                    goal_conditions.add(condition_key)

            fulfilled_conditions = set()

            # Simulate actions along the path
            for node in path:
                if node not in self.G.nodes:
                    logging.warning(f"Node {node} not found in graph")
                    continue

                node_data = self.G.nodes[node]
                possible_interactions = node_data.get("possible_interactions", [])

                # Process each possible action at this node
                for action in possible_interactions:
                    if not hasattr(action, "effects") or not action.effects:
                        continue

                    # Check if action preconditions are met
                    if hasattr(action, "preconditions") and action.preconditions:
                        preconditions_met = True
                        for precondition in action.preconditions.values():
                            if hasattr(precondition, "check_condition"):
                                state_obj = State(current_state)
                                if not precondition.check_condition(state_obj):
                                    preconditions_met = False
                                    break

                        if not preconditions_met:
                            continue

                    # Apply action effects to current state
                    for effect in action.effects:
                        if "attribute" in effect and "change_value" in effect:
                            attr = effect["attribute"]
                            change_val = effect["change_value"]

                            # Update state
                            if attr in current_state:
                                if isinstance(change_val, (int, float)):
                                    current_state[attr] += change_val
                                else:
                                    current_state[attr] = change_val
                            else:
                                current_state[attr] = change_val

                            # Check if this effect fulfills any goal conditions
                            for condition in (
                                goal.completion_conditions
                                if hasattr(goal, "completion_conditions")
                                else []
                            ):
                                if condition.attribute == attr:
                                    state_obj = State(current_state)
                                    if condition.check_condition(state_obj):
                                        condition_key = f"{condition.attribute}_{condition.satisfy_value}_{condition.op}"
                                        fulfilled_conditions.add(condition_key)

            # Check if all goal conditions are fulfilled
            return fulfilled_conditions >= goal_conditions

        except Exception as e:
            logging.error(f"Error evaluating path for goal achievement: {e}")
            return False

    ###TODO: Calculate the difficulty of a goal using the graph and goal criteria, using get_possible_actions on the filtered_nodes. Also figure out how to initialize the criteria.
    ### Also finish the calculate_utility function above get_possible_actions.


""" 
Graph-Based Decision Making:
Use the graph to inform decisions dynamically. For example, if a character is planning their social interactions, the graph can provide current friends and their relationship strengths, which in turn informs the utility evaluation of different social actions.
Real-Time Adaptation
Feedback Loops:
Establish feedback loops where the outcome of actions affects future planning cycles by updating both the state and the graph.
After executing an action, the system should assess the actual outcomes versus expected and adjust the planning parameters if necessary.
Learning and Adaptation:
Integrate simple learning mechanisms where characters can adjust their preferences based on past outcomes (e.g., if attending certain events consistently leads to positive outcomes, increase the preference for similar future events). """


# Create the graph
class TinyVillageGraph:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.G = self.graph_manager.create_tiny_village_graph()

    def update_strategy(self, event):
        if event["event"] == "new_day":
            character_state = self.graph_manager.get_character_state("Emma")
            actions = self.graph_manager.get_possible_actions("Emma")
            plan = self.goap_planner.goap_planner("Emma", character_state, actions)
            return plan

    def goap_planner(self, character, state, actions):
        # Placeholder for GOAP algorithm logic
        # Sorts actions based on utility and state requirements
        return sorted(actions, key=lambda x: -x["utility"])

    def plan_daily_activities(self, character):
        # Define the goal for daily activities
        goal = {"satisfaction": max, "energy_usage": min}

        # Get potential actions from a dynamic or context-specific action generator
        actions = self.graph_manager.get_possible_actions(character)

        # Use the graph to analyze current relationships and preferences
        current_state = self.graph_manager.graph_analysis(self.G, character, "daily")

        # Plan the career steps using GOAP
        plan = self.goap_planner(character, goal, current_state, actions)

        # Evaluate the utility of each step in the plan
        utility_scores = self.graph_manager.evaluate_utility(actions, character)

        return plan, utility_scores

    def update_graph(self, action, character, target=None):
        self.graph_manager.update_graph(self.G, action, character, target)
        return self.G

    def get_strongest_relationships(self, character):
        return self.graph_manager.get_strongest_relationships(self.G, character)

    def get_favorite_locations(self, character):
        return self.graph_manager.get_favorite_locations(self.G, character)

    def analyze_event_impact(self, event_node):
        return self.graph_manager.analyze_event_impact(self.G, event_node)

    def explore_career_opportunities(self, character):
        return self.graph_manager.explore_career_opportunities(self.G, character)

    def analyze_daily_preferences(self, character):
        return self.graph_manager.analyze_daily_preferences(self.G, character)

    def analyze_career_impact(self, character, job_offer):
        return self.graph_manager.analyze_career_impact(self.G, character, job_offer)
