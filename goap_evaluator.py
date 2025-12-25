"""
GOAP Evaluator Module

This module contains the GoapEvaluator class which handles all Goal-Oriented Action Planning
(GOAP) logic that was previously scattered throughout the GraphManager class.

The GoapEvaluator is responsible for:
- Goal difficulty assessment
- Action plan evaluation  
- Motive calculations
- Action viability and cost analysis
- Goal impact calculations

This class is designed to be stateless and receive world state as dependencies,
making it easier to test, maintain, and improve the AI's decision-making process.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, time
import dis
from functools import lru_cache
import heapq
from itertools import chain, combinations, product
import logging
from math import dist, inf, log
import math
import operator
import random
import re
import uuid
import networkx as nx
import importlib
import numpy as np

# Import utility functions
from tiny_utility_functions import is_goal_achieved


# Cache functions from tiny_graph_manager
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
def tanh_scaling_raw(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return scaled_value


class WorldState:
    """
    Represents the current state of the world for GOAP evaluation.
    This includes character states, graph data, and environment conditions.
    """
    
    def __init__(self, graph_manager=None, characters=None, locations=None, objects=None, 
                 events=None, activities=None, jobs=None):
        self.graph_manager = graph_manager
        self.characters = characters or {}
        self.locations = locations or {}
        self.objects = objects or {}
        self.events = events or {}
        self.activities = activities or {}
        self.jobs = jobs or {}
        
        # If graph_manager is provided, extract state from it
        if graph_manager:
            self.characters = graph_manager.characters
            self.locations = graph_manager.locations
            self.objects = graph_manager.objects
            self.events = graph_manager.events
            self.activities = graph_manager.activities
            self.jobs = graph_manager.jobs
            self.G = graph_manager.G
        else:
            self.G = None
    
    def get_character(self, character_name):
        """Get character by name."""
        return self.characters.get(character_name)
    
    def get_location(self, location_name):
        """Get location by name."""
        return self.locations.get(location_name)
    
    def get_graph(self):
        """Get the NetworkX graph."""
        return self.G


class GoapEvaluator:
    """
    Goal-Oriented Action Planning evaluator responsible for all GOAP-related calculations.
    
    This class is stateless and receives world state as a dependency, making it easier
    to test and maintain than the monolithic GraphManager approach.
    """
    
    def __init__(self):
        """Initialize the GoapEvaluator."""
        self.dp_cache = {}
    
    def calculate_motives(self, character, world_state: WorldState = None):
        """
        Calculate motives for a character based on their personality traits and current state.
        
        Args:
            character: Character object to calculate motives for
            world_state: Current world state (optional, for future use)
            
        Returns:
            PersonalMotives: Object containing all calculated motives
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
            PersonalMotives = importlib.import_module("tiny_characters").PersonalMotives
            Motive = importlib.import_module("tiny_characters").Motive
        except ImportError as e:
            logging.error(f"Failed to import required classes: {e}")
            return None

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
                + tanh_scaling_raw(
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
            tanh_scaling_raw(
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
                + tanh_scaling_raw(
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
    
    def calculate_goal_difficulty(self, goal, character, world_state: WorldState):
        """
        Calculate the difficulty of a goal based on its complexity and requirements.
        
        Args:
            goal: Goal object to evaluate
            character: Character attempting the goal
            world_state: Current world state containing graph and node data
            
        Returns:
            dict: Dictionary containing difficulty score and additional metrics
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
            Goal = importlib.import_module("tiny_characters").Goal
        except ImportError:
            logging.error("Failed to import Character class from tiny_characters module.")
            return {"difficulty": float("inf"), "error": "Import error"}

        # Analyze the goal requirements and complexity
        if not hasattr(goal, "criteria") or not goal.criteria:
            return {"difficulty": 0, "error": "Goal has no criteria"}

        goal_requirements = goal.criteria
        
        # Get graph from world state - for now delegate back to graph manager for complex operations
        # This is a transitional approach - future versions should extract more functionality
        if not world_state or not world_state.graph_manager:
            return {"difficulty": float("inf"), "error": "No world state or graph manager"}
            
        # Delegate to graph manager for now - this is the heavy graph analysis part
        # In future iterations, we can extract these methods too
        try:
            result = world_state.graph_manager.calculate_goal_difficulty(goal, character)
            return result
        except Exception as e:
            logging.error(f"Error calculating goal difficulty: {e}")
            return {"difficulty": float("inf"), "error": str(e)}
    
    def calculate_how_goal_impacts_character(self, goal, character, world_state: WorldState = None):
        """
        Calculate the impact of a goal on a character based on the goal's completion conditions 
        and the character's current state.
        
        Args:
            goal: Goal object to evaluate
            character: Character to evaluate  
            world_state: Current world state (optional)
            
        Returns:
            float: Impact of the goal on the character
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
        except ImportError:
            logging.error("Failed to import Character class")
            return 0.0

        impact = 0
        if hasattr(goal, 'completion_conditions') and goal.completion_conditions:
            for condition in goal.completion_conditions:
                if hasattr(condition, 'attribute') and hasattr(character, 'get_state'):
                    character_state = character.get_state()
                    state_dict = character_state.dict_or_obj if hasattr(character_state, 'dict_or_obj') else {}
                    
                    if condition.attribute in state_dict:
                        if hasattr(condition, 'operator'):
                            if condition.operator in ["ge", "gt"]:
                                impact += max(0, state_dict[condition.attribute] - condition.satisfy_value)
                            elif condition.operator in ["le", "lt"]:
                                impact += max(0, condition.satisfy_value - state_dict[condition.attribute])
                                
        return impact
    
    def calculate_action_effect_cost(self, action, character, goal, world_state: WorldState = None):
        """
        Calculate the cost of an action based on the effects it will have on the character.
        
        Args:
            action: Action object to evaluate
            character: Character performing the action
            goal: Goal being pursued
            world_state: Current world state (optional)
            
        Returns:
            float: Cost of the action based on its effects
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
            Action = importlib.import_module("actions").Action
            Goal = importlib.import_module("tiny_characters").Goal
        except ImportError:
            logging.error("Failed to import required classes")
            return float("inf")

        goal_cost = 0
        if not hasattr(goal, 'completion_conditions') or not goal.completion_conditions:
            return 0
            
        conditions = goal.completion_conditions
        weights = {}
        
        # Calculate weights for each effect
        if hasattr(action, 'effects') and action.effects:
            for effect in action.effects:
                if isinstance(effect, dict) and 'attribute' in effect:
                    weights[effect["attribute"]] = 1
                    
                    for condition in conditions:
                        if (hasattr(condition, 'attribute') and 
                            hasattr(character, 'get_state') and 
                            effect["attribute"] in character.get_state().dict_or_obj and
                            effect.get("targets") == "initiator"):
                            
                            if (condition.attribute == effect["attribute"] and 
                                hasattr(condition, 'target') and condition.target == character):
                                
                                # Calculate weight based on condition satisfaction
                                if hasattr(condition, 'operator') and hasattr(condition, 'satisfy_value'):
                                    change_value = effect.get("change_value", 0)
                                    if isinstance(change_value, (int, float)):
                                        if condition.operator in ["ge", "gt"] and change_value > 0:
                                            current_val = character.get_state().dict_or_obj.get(effect["attribute"], 0)
                                            delta = (current_val + change_value) - condition.satisfy_value
                                            delta = abs(change_value) * delta
                                            scaling_factor = getattr(condition, 'weight', 1) / sum(
                                                getattr(c, 'weight', 1) for c in conditions
                                            )
                                            delta = 1 / (1 + math.exp(-delta))
                                            weights[effect["attribute"]] += delta * scaling_factor

        # Apply softmax to weights
        if weights:
            weights_array = np.array(list(weights.values()))
            softmax_weights = np.exp(weights_array) / np.sum(np.exp(weights_array))
            for i, attribute in enumerate(weights):
                weights[attribute] = softmax_weights[i]

        # Calculate goal cost based on negative effects
        if hasattr(action, 'effects') and action.effects:
            for effect in action.effects:
                if isinstance(effect, dict) and 'attribute' in effect:
                    for condition in conditions:
                        if (hasattr(condition, 'attribute') and 
                            hasattr(character, 'get_state') and
                            effect["attribute"] in character.get_state().dict_or_obj and
                            effect.get("targets") == "initiator"):
                            
                            if (condition.attribute == effect["attribute"] and
                                hasattr(condition, 'target') and condition.target == character):
                                
                                change_value = effect.get("change_value", 0)
                                if isinstance(change_value, (int, float)):
                                    # Negative effects increase cost
                                    if hasattr(condition, 'operator') and hasattr(condition, 'satisfy_value'):
                                        if condition.operator in ["ge", "gt"] and change_value < 0:
                                            current_val = character.get_state().dict_or_obj.get(effect["attribute"], 0)
                                            delta = (condition.satisfy_value - current_val + abs(change_value))
                                            delta = abs(change_value) * delta
                                            
                                            try:
                                                scaling_factor = 1 / weights.get(effect["attribute"], 1)
                                            except ZeroDivisionError:
                                                scaling_factor = float("inf")
                                                
                                            delta = 1 / (1 + math.exp(-delta))
                                            goal_cost += delta * scaling_factor

        # Apply sigmoid to final cost
        goal_cost = 1 / (1 + math.exp(-goal_cost))
        return goal_cost
    
    def calculate_action_viability_cost(self, node, goal, character, world_state: WorldState):
        """
        Calculate the cost and viability of actions for a node in relation to a goal.
        
        Args:
            node: Node to evaluate
            goal: Goal being pursued  
            character: Character performing actions
            world_state: Current world state
            
        Returns:
            dict: Dictionary containing action costs, viability, and goal fulfillment data
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
        except ImportError:
            logging.error("Failed to import Character class")
            return {}

        # For now, delegate to graph manager for complex node resolution
        # Future versions should extract this functionality
        if not world_state or not world_state.graph_manager:
            return {}
            
        cache_key = (node, goal, character)
        if cache_key in self.dp_cache:
            return self.dp_cache[cache_key]
            
        try:
            result = world_state.graph_manager.calculate_action_viability_cost(node, goal, character)
            self.dp_cache[cache_key] = result
            return result
        except Exception as e:
            logging.error(f"Error calculating action viability cost: {e}")
            return {}
    
    def will_action_fulfill_goal(self, action, goal, current_state, character, world_state: WorldState = None):
        """
        Determine if an action fulfills the specified goal by checking completion conditions.
        
        Args:
            action: Action to evaluate
            goal: Goal to achieve
            current_state: Current state of the action target
            character: Character performing the action
            world_state: Current world state (optional)
            
        Returns:
            dict: Dictionary indicating which completion conditions are met
        """
        try:
            Character = importlib.import_module("tiny_characters").Character
            Action = importlib.import_module("actions").Action
            Goal = importlib.import_module("tiny_characters").Goal
            State = importlib.import_module("actions").State
        except ImportError:
            logging.error("Failed to import required classes")
            return {}

        goal_copy = copy.deepcopy(goal)
        completion_conditions = getattr(goal_copy, 'completion_conditions', {})
        action_effects = getattr(action, 'effects', [])
        goal_target = getattr(goal_copy, 'target', None)
        
        # Make a copy of the State so the original is not modified
        current_state_ = copy.deepcopy(current_state)
        
        for effect in action_effects:
            if isinstance(effect, dict):
                targets = effect.get("targets", [])
                if (("target" in targets and goal_target and 
                     hasattr(goal_target, 'name') and hasattr(current_state_, 'name') and
                     goal_target.name == current_state_.name) or
                    ("initiator" in targets and goal_target and 
                     hasattr(goal_target, 'name') and hasattr(character, 'name') and
                     goal_target.name == character.name)):
                    
                    # Apply effect to state
                    if hasattr(action, 'apply_single_effect'):
                        current_state_ = action.apply_single_effect(effect, current_state_)
                    
                    # Check conditions
                    if isinstance(completion_conditions, dict):
                        for condition_status, conditions_list in completion_conditions.items():
                            if conditions_list and not condition_status:  # False conditions
                                for condition in conditions_list:
                                    if hasattr(condition, 'check_condition'):
                                        check = condition.check_condition(current_state_)
                                        if check:
                                            # Move condition from False to True
                                            completion_conditions[True] = condition
                                            completion_conditions[False] = None

        # Return reversed completion_conditions dictionary
        return {v: k for k, v in completion_conditions.items() if v is not None}
    
    def evaluate_action_plan(self, plan, character, goal, world_state: WorldState):
        """
        Evaluate the effectiveness of an action plan for achieving a goal.
        
        Args:
            plan: List of actions forming the plan
            character: Character executing the plan
            goal: Goal to achieve
            world_state: Current world state
            
        Returns:
            dict: Evaluation metrics including cost, viability, and success probability
        """
        if not plan:
            return {"cost": float("inf"), "viability": 0.0, "success_probability": 0.0}
        
        total_cost = 0
        total_viability = 0
        conditions_satisfied = set()
        
        # Simulate plan execution
        for action in plan:
            # Calculate action cost
            action_cost = getattr(action, 'cost', 1.0)
            effect_cost = self.calculate_action_effect_cost(action, character, goal, world_state)
            total_cost += action_cost + effect_cost
            
            # Check viability
            if hasattr(action, 'preconditions_met'):
                if action.preconditions_met():
                    total_viability += 1
            else:
                total_viability += 0.5  # Assume partially viable
            
            # Check goal fulfillment
            if hasattr(character, 'get_state'):
                current_state = character.get_state()
                fulfilled = self.will_action_fulfill_goal(action, goal, current_state, character, world_state)
                conditions_satisfied.update(fulfilled.keys())
        
        # Calculate metrics
        avg_viability = total_viability / len(plan) if plan else 0.0
        
        # Calculate success probability based on conditions satisfied
        total_conditions = 0
        if hasattr(goal, 'completion_conditions') and goal.completion_conditions:
            if isinstance(goal.completion_conditions, dict):
                total_conditions = sum(len(conditions) for conditions in goal.completion_conditions.values() 
                                     if conditions)
            elif isinstance(goal.completion_conditions, list):
                total_conditions = len(goal.completion_conditions)
        
        success_probability = len(conditions_satisfied) / max(total_conditions, 1)
        
        return {
            "cost": total_cost,
            "viability": avg_viability,
            "success_probability": success_probability,
            "conditions_satisfied": len(conditions_satisfied),
            "total_conditions": total_conditions
        }