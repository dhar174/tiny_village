"""
Utility Evaluation Module

This module provides comprehensive utility evaluation for goals and actions in the tiny village simulation.
It includes:
- Basic utility calculations for actions and plans
- Advanced utility evaluation with context and history
- Caching and optimization for performance
- Proper data structure validation and documentation

Expected Data Structures:
- character_state (dict): Keys are need/resource names with numeric values
  - Needs (lower is better): "hunger" (0.0=full, 1.0=very hungry)
  - Resources (higher is better): "energy", "money", "health", "social_needs"
- action.effects (list): [{"attribute": str, "change_value": float}, ...]
- goal.target_effects (dict): {"attribute": target_value, ...}
- goal.priority (float): Importance weight for goal completion
"""

import time
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from actions import (
    Action,
    State,
)  # Assuming Action class is available and has cost & effects

# Try to import the real Goal class first, fall back to simple implementation
try:
    from tiny_characters import Goal
    print("✅ Using real Goal class from tiny_characters")
except ImportError:
    # Fallback Goal class for compatibility when real Goal is not available
    class Goal:
        """Simple Goal class for utility calculations - fallback implementation."""

        def __init__(self, name=None, target_effects=None, priority=0.5, score=None, **kwargs):
            self.name = name or "UnnamedGoal"
            self.target_effects = target_effects if target_effects else {}
            self.priority = priority
            self.score = score if score is not None else priority
            # Additional attributes that may be expected by utility functions
            self.urgency = kwargs.get('urgency', priority)
            self.attributes = kwargs.get('attributes', {})
    
    print("⚠️  Using fallback Goal implementation in tiny_utility_functions")


# === CONSTANTS ===
# Utility calculation scalers - adjust these to balance different factors
HUNGER_SCALER = 20.0  # Weight for hunger need fulfillment
ENERGY_SCALER = 15.0  # Weight for energy need fulfillment
MONEY_SCALER = 0.5  # Weight for money resource gains
HEALTH_SCALER = 25.0  # Weight for health improvements
SOCIAL_SCALER = 10.0  # Weight for social need fulfillment

# Advanced evaluation constants
HISTORY_DECAY_FACTOR = 0.9  # How much historical data decays over time
CONTEXT_WEIGHT = 0.3  # Weight of environmental context in decisions
URGENCY_MULTIPLIER = 2.0  # Multiplier for urgent goals
DIMINISHING_RETURNS_FACTOR = 0.8  # Factor for diminishing returns on repeated actions

# Caching configuration
CACHE_TTL_SECONDS = 300  # Time-to-live for cached utility calculations
MAX_CACHE_SIZE = 1000  # Maximum number of cached entries


# === CACHING AND OPTIMIZATION ===


def timed_lru_cache(maxsize=MAX_CACHE_SIZE, ttl_seconds=CACHE_TTL_SECONDS):
    """
    Decorator that provides LRU cache with time-to-live functionality.
    """

    def decorator(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.cache_info = func.cache_info
        func.cache_clear = func.cache_clear
        func._cache_time = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            cache_key = str(args) + str(sorted(kwargs.items()))

            # Check if cache entry has expired
            if cache_key in func._cache_time:
                if current_time - func._cache_time[cache_key] > ttl_seconds:
                    # Clear expired entry
                    func.cache_clear()
                    func._cache_time.clear()

            # Update cache time
            func._cache_time[cache_key] = current_time
            return func(*args, **kwargs)

        return wrapper

    return decorator


class UtilityEvaluator:
    """
    Advanced utility evaluator with context awareness, history tracking, and optimization.

    This class provides comprehensive utility evaluation that considers:
    - Character state and needs
    - Historical action patterns
    - Environmental context
    - Goal priorities and progress
    - Performance optimizations through caching
    """

    def __init__(self):
        self.action_history: Dict[str, List[Tuple[float, str]]] = defaultdict(
            list
        )  # character_id -> [(timestamp, action_name), ...]
        self.utility_cache: Dict[str, Tuple[float, float]] = (
            {}
        )  # cache_key -> (utility, timestamp)
        self.context_weights: Dict[str, float] = {
            "time_of_day": 0.1,
            "season": 0.05,
            "weather": 0.08,
            "social_events": 0.15,
            "resource_availability": 0.2,
        }

    def clear_cache(self):
        """Clear all cached utility calculations."""
        self.utility_cache.clear()

    def update_action_history(self, character_id: str, action_name: str):
        """Update the action history for a character."""
        current_time = time.time()
        self.action_history[character_id].append((current_time, action_name))

        # Keep only recent history (last 24 hours in game time)
        cutoff_time = current_time - (24 * 3600)  # 24 hours
        self.action_history[character_id] = [
            (t, a) for t, a in self.action_history[character_id] if t > cutoff_time
        ]

    def get_action_frequency(
        self, character_id: str, action_name: str, time_window: float = 3600
    ) -> int:
        """Get how many times an action was performed in the given time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window

        return sum(
            1
            for t, a in self.action_history.get(character_id, [])
            if t > cutoff_time and a == action_name
        )

    def calculate_context_modifier(self, environment: Dict[str, any]) -> float:
        """
        Calculate a context-based modifier for utility calculations.

        Args:
            environment: Dictionary containing environmental factors

        Returns:
            Float modifier (typically 0.8-1.2) to apply to base utility
        """
        if not environment:
            return 1.0

        modifier = 1.0

        # Time of day effects
        time_of_day = environment.get("time_of_day", 12)  # 0-23 hours
        if 22 <= time_of_day or time_of_day <= 6:  # Night time
            modifier += (
                self.context_weights["time_of_day"] * 0.5
            )  # Slight bonus for rest actions

        # Weather effects
        weather = environment.get("weather", "clear")
        if weather in ["rain", "storm"]:
            modifier -= self.context_weights["weather"]  # Penalty for outdoor actions
        elif weather == "sunny":
            modifier += self.context_weights["weather"] * 0.5  # Small bonus

        # Social events
        if environment.get("social_event_active", False):
            modifier += self.context_weights[
                "social_events"
            ]  # Bonus for social actions

        # Resource scarcity
        resource_scarcity = environment.get("resource_scarcity", 0.0)  # 0.0-1.0
        modifier -= self.context_weights["resource_availability"] * resource_scarcity

        return max(0.5, min(1.5, modifier))  # Clamp between 0.5 and 1.5

    @timed_lru_cache(maxsize=500)
    def evaluate_action_utility_advanced(
        self,
        character_id: str,
        character_state_hash: str,  # Hash of character state for caching
        action_name: str,
        action_cost: float,
        action_effects_hash: str,  # Hash of action effects for caching
        goal_hash: str = "",  # Hash of current goal for caching
        environment_hash: str = "",  # Hash of environment for caching
    ) -> float:
        """
        Advanced utility calculation with caching, history, and context awareness.

        This method is cached and uses hashed inputs for efficient comparison.
        """
        # Note: This is a cached version that works with hashed inputs
        # The actual implementation details would be called from the public method
        return 0.0  # Placeholder - actual logic implemented in public method

    def evaluate_action_utility(
        self,
        character_id: str,
        character_state: Dict[str, float],
        action: Action,
        current_goal: Optional[Goal] = None,
        environment: Optional[Dict[str, any]] = None,
    ) -> float:
        """
        Comprehensive utility evaluation for an action.

        Args:
            character_id: Unique identifier for the character
            character_state: Current state of the character
            action: Action to evaluate
            current_goal: Current goal being pursued (optional)
            environment: Environmental context (optional)

        Returns:
            Calculated utility value for the action
        """
        # Calculate base utility using existing function
        base_utility = calculate_action_utility(character_state, action, current_goal)

        # Apply context modifier
        context_modifier = 1.0
        if environment:
            context_modifier = self.calculate_context_modifier(environment)

        # Apply history-based adjustments
        history_modifier = self._calculate_history_modifier(character_id, action.name)

        # Apply urgency multiplier if goal is urgent
        urgency_modifier = 1.0
        if (
            current_goal
            and hasattr(current_goal, "urgency")
            and current_goal.urgency > 0.8
        ):
            urgency_modifier = URGENCY_MULTIPLIER

        # Calculate final utility
        final_utility = (
            base_utility * context_modifier * history_modifier * urgency_modifier
        )

        return final_utility

    def _calculate_history_modifier(self, character_id: str, action_name: str) -> float:
        """
        Calculate a modifier based on action history to handle diminishing returns.
        """
        recent_frequency = self.get_action_frequency(
            character_id, action_name, 3600
        )  # Last hour

        if recent_frequency == 0:
            return 1.0  # No penalty for first time

        # Apply diminishing returns
        modifier = DIMINISHING_RETURNS_FACTOR ** (recent_frequency - 1)
        return max(0.2, modifier)  # Don't go below 20% utility

    def evaluate_plan_utility_advanced(
        self,
        character_id: str,
        character_state: Dict[str, float],
        plan: List[Action],
        current_goal: Optional[Goal] = None,
        environment: Optional[Dict[str, any]] = None,
        simulate_effects: bool = True,
    ) -> Tuple[float, Dict[str, any]]:
        """
        Advanced plan utility evaluation with detailed analysis.

        Returns:
            Tuple of (total_utility, analysis_details)
        """
        if not plan:
            return 0.0, {"error": "Empty plan"}

        total_utility = 0.0
        simulated_state = character_state.copy()
        action_utilities = []

        for i, action in enumerate(plan):
            # Evaluate action utility with current state
            action_utility = self.evaluate_action_utility(
                character_id, simulated_state, action, current_goal, environment
            )

            action_utilities.append(
                {"action": action.name, "utility": action_utility, "step": i + 1}
            )

            total_utility += action_utility

            # Update simulated state if requested
            if simulate_effects and action.effects:
                for effect in action.effects:
                    attribute = effect.get("attribute")
                    change = effect.get("change_value", 0.0)

                    if attribute in simulated_state:
                        simulated_state[attribute] += change

                        # Apply clamping based on attribute type
                        if attribute in ["hunger", "energy", "health"]:
                            simulated_state[attribute] = max(
                                0.0, min(1.0, simulated_state[attribute])
                            )
                        elif attribute == "money":
                            simulated_state[attribute] = max(
                                0.0, simulated_state[attribute]
                            )

        analysis = {
            "total_utility": total_utility,
            "average_utility": total_utility / len(plan),
            "action_breakdown": action_utilities,
            "final_simulated_state": simulated_state,
            "plan_length": len(plan),
        }

        return total_utility, analysis


# Global instance for convenience
utility_evaluator = UtilityEvaluator()


# === ENHANCED EXISTING FUNCTIONS ===


# function that evaluates the current importance of the goal based on the character's state and the environment.
# tiny_utility_functions.py


# --- Enhanced Functions ---
def calculate_importance(
    health: float,
    hunger: float,
    social_needs: float,
    current_activity: str,
    social_factor: float,
    event_participation_factor: float,
    goal_importance: float,
    goal_urgency: float = 0.5,
    goal_benefit: float = 0.5,
    goal_consequence: float = 0.5,
    location_factor: float = 0.5,
    event_factor: float = 0.5,
    resource_factor: float = 0.5,
    social_influence: float = 0.5,
    potential_utility: float = 0.5,
    path_achieves_goal: bool = True,
    safety_factor: float = 0.5,
    nearest_resource: float = 0.5,
    character_history: Optional[Dict[str, any]] = None,
    environmental_context: Optional[Dict[str, any]] = None,
) -> float:
    """
    Enhanced importance calculation that supports context, history, and advanced evaluation.

    Args:
        health: Character's health level (0.0-1.0, higher is better)
        hunger: Character's hunger level (0.0-1.0, lower is better)
        social_needs: Character's social needs level (0.0-1.0, lower is better)
        current_activity: Character's current activity name
        social_factor: Influence of relationships (0.0-1.0)
        event_participation_factor: Character's participation in relevant events (0.0-1.0)
        goal_importance: Specific importance score based on goal type (0.0-1.0)
        goal_urgency: Urgency of the goal (0.0-1.0)
        goal_benefit: Benefit of achieving the goal (0.0-1.0)
        goal_consequence: Consequence of not achieving the goal (0.0-1.0)
        location_factor: Impact of location relevance (0.0-1.0)
        event_factor: Impact of current events (0.0-1.0)
        resource_factor: Availability of needed resources (0.0-1.0)
        social_influence: Social influence on the character (0.0-1.0)
        potential_utility: Utility of achieving the goal (0.0-1.0)
        path_achieves_goal: Whether the path will achieve the goal
        safety_factor: Safety of the goal location (0.0-1.0)
        nearest_resource: Proximity to the nearest resource (0.0-1.0)
        character_history: Historical data about character's actions and preferences
        environmental_context: Current environmental factors affecting decisions

    Returns:
        A float representing the importance score (typically 0.0-10.0)
    """
    # Enhanced weighting factors with better balance
    weights = {
        "health": 0.08,
        "hunger": 0.12,
        "social_needs": 0.06,
        "urgency": 0.15,
        "benefit": 0.10,
        "consequence": 0.12,
        "social": 0.08,
        "location": 0.04,
        "event": 0.06,
        "resource": 0.07,
        "social_influence": 0.05,
        "potential_utility": 0.10,
        "path_achieves_goal": 0.08,
        "safety": 0.04,
        "event_participation": 0.03,
        "nearest_resource": 0.03,
        "goal_importance": 0.20,
    }

    # Base importance calculation
    base_score = (
        weights["health"] * health
        + weights["hunger"]
        * (1.0 - hunger)  # Invert hunger (lower hunger = higher importance)
        + weights["social_needs"] * (1.0 - social_needs)  # Invert social needs
        + weights["urgency"] * goal_urgency
        + weights["benefit"] * goal_benefit
        + weights["consequence"] * goal_consequence
        + weights["social"] * social_factor
        + weights["location"] * location_factor
        + weights["event"] * event_factor
        + weights["resource"] * resource_factor
        + weights["social_influence"] * social_influence
        + weights["potential_utility"] * potential_utility
        + weights["path_achieves_goal"] * (1.0 if path_achieves_goal else 0.0)
        + weights["safety"] * safety_factor
        + weights["event_participation"] * event_participation_factor
        + weights["nearest_resource"] * nearest_resource
        + weights["goal_importance"] * goal_importance
    )

    # Apply historical context if available
    history_modifier = 1.0
    if character_history:
        # Check if this type of goal was recently completed
        recent_completions = character_history.get("recent_goal_completions", [])
        goal_type = character_history.get("current_goal_type", "")

        if goal_type in recent_completions:
            # Apply diminishing returns for recently completed goal types
            completion_count = recent_completions.count(goal_type)
            history_modifier *= DIMINISHING_RETURNS_FACTOR**completion_count

        # Consider character preferences
        preferences = character_history.get("goal_preferences", {})
        if goal_type in preferences:
            preference_modifier = preferences[goal_type]  # -1.0 to 1.0
            history_modifier *= 1.0 + preference_modifier * 0.2  # Max 20% bonus/penalty

    # Apply environmental context if available
    context_modifier = 1.0
    if environmental_context:
        context_modifier = utility_evaluator.calculate_context_modifier(
            environmental_context
        )

    # Calculate final importance with modifiers
    final_importance = (
        base_score * history_modifier * context_modifier * 10.0
    )  # Scale to 0-10 range

    return max(0.0, min(10.0, final_importance))  # Clamp to reasonable range


def evaluate_goal_importance(
    goal, character_state: State, environment: dict, difficulty: float, criteria: dict
):
    """
    Evaluates the importance of a goal based on the character's current state and the environment.
    Args:
        goal (Goal): The goal to evaluate.
        character_state (State): The character's current state.
        environment (dict): The environment's state.
    Returns:
        importance (int): The importance of the goal.
    """
    importance = 0
    if hasattr(goal, "attributes") and isinstance(goal.attributes, dict):
        for attribute, value in goal.attributes.items():
            # Assuming character_state can be accessed like a dict or has a get method
            importance += value * character_state.get(attribute, 0)

    if (
        hasattr(goal, "attributes")
        and isinstance(goal.attributes, dict)
        and isinstance(environment, dict)
    ):

        for attribute, value in environment.items():
            if attribute in goal.attributes:
                importance += value * goal.attributes[attribute]
    return importance


def is_goal_achieved(goal, character_state: State):
    if hasattr(goal, "target_effects") and isinstance(goal.target_effects, dict):

        for effect, target_value in goal.target_effects.items():
            if character_state.get(effect, 0) < target_value:
                return False
        return True
    return False  # Default if goal has no target_effects to check


# --- New Utility Calculation Functions ---


def calculate_action_utility(
    character_state: dict, action: Action, current_goal: Goal = None
) -> float:
    """
    Calculates the utility of a single action for a character.

    Assumptions:
    - character_state (dict): Keys are need/resource names (e.g., "hunger", "energy", "money").
        - For needs like "hunger": Higher value means more need (0 is full, 1+ is very hungry).
        - For resources like "energy", "money": Higher value means more resource.
    - action.cost (float): Represents a generic cost (e.g., time, effort).
    - action.effects (list of dicts): Each dict is `{'attribute': str, 'change_value': float}`.
        - For "hunger", a negative change_value is beneficial (reduces hunger).
        - For "energy", a positive change_value is beneficial (increases energy).
    - current_goal (Goal object, optional): Assumed to have `name` and `target_effects` (dict).
    """
    utility = 0.0
    need_fulfillment_score = 0.0
    goal_progress_score = 0.0


    # 1. Need Fulfillment
    if action.effects:
        for effect in action.effects:
            attribute = effect.get("attribute")
            change = effect.get("change_value", 0.0)

            if attribute == "hunger":
                # Higher current hunger + action reduces hunger = good
                current_hunger = character_state.get("hunger", 0.0)
                if change < 0:  # Action reduces hunger
                    need_fulfillment_score += (
                        current_hunger * abs(change) * HUNGER_SCALER
                    )
            elif attribute == "energy":
                # Lower current energy + action increases energy = good
                current_energy = character_state.get("energy", 0.0)
                if change > 0:  # Action increases energy
                    need_fulfillment_score += (
                        (1.0 - current_energy) * change * ENERGY_SCALER
                    )
            elif attribute == "health":
                # Lower current health + action increases health = good
                current_health = character_state.get("health", 1.0)
                if change > 0:  # Action increases health
                    need_fulfillment_score += (
                        (1.0 - current_health) * change * HEALTH_SCALER
                    )
            elif attribute == "social_needs":
                # Higher social needs + action reduces social needs = good
                current_social_needs = character_state.get("social_needs", 0.0)
                if change < 0:  # Action reduces social needs
                    need_fulfillment_score += (
                        current_social_needs * abs(change) * SOCIAL_SCALER
                    )

            elif attribute == "money":
                # Money gained is generally positive utility
                if change > 0:
                    need_fulfillment_score += change * MONEY_SCALER


    utility += need_fulfillment_score

    # 2. Goal Progress
    if (
        current_goal
        and hasattr(current_goal, "target_effects")
        and current_goal.target_effects
    ):

        if action.effects:
            for effect in action.effects:
                attr = effect.get("attribute")
                change = effect.get("change_value", 0.0)
                if attr in current_goal.target_effects:
                    goal_target_change = current_goal.target_effects[attr]
                    # If action moves attribute towards goal (e.g., hunger goal -0.5, action does -0.2)
                    if (goal_target_change < 0 and change < 0) or (
                        goal_target_change > 0 and change > 0
                    ):
                        # Add goal progress bonus
                        goal_priority = getattr(current_goal, "priority", 0.5)
                        goal_progress_score += goal_priority * 25.0
                        # Break if one effect contributes, to avoid over-counting for multi-effect actions

                        break

    utility += goal_progress_score

    # 3. Action Cost
    action_cost_score = 0.0
    if hasattr(action, "cost"):
        action_cost_score = float(action.cost) * 10.0  # Scaler for cost impact

    utility -= action_cost_score



    # Consider inherent utility/disutility of certain actions if not captured by needs/goals/cost
    # Example: "Rest" might have a small positive base utility if not costly and energy is low.
    # "Argue" might have a small negative base utility.
    # For now, this is not explicitly added.

    return utility


def calculate_plan_utility(
    character_state: dict,
    plan: list[Action],
    current_goal: Goal = None,
    simulate_effects=False,
) -> float:
    """
    Calculates the utility of a sequence of actions (a plan).

    If simulate_effects is True, the character_state is updated after each action
    to calculate the utility of subsequent actions more accurately.
    """
    total_utility = 0.0

    if not simulate_effects:
        for action in plan:
            total_utility += calculate_action_utility(
                character_state, action, current_goal
            )
    else:
        # Create a deep copy for simulation if character_state can contain nested dicts/lists
        # For simple dict of floats, .copy() is fine.
        simulated_state = character_state.copy()
        for action in plan:
            action_utility = calculate_action_utility(
                simulated_state, action, current_goal
            )
            total_utility += action_utility


            # Update simulated_state based on action's effects
            if action.effects:
                for effect in action.effects:
                    attribute = effect.get("attribute")
                    change = effect.get("change_value", 0.0)

                    if attribute:  # Ensure attribute is not None

                        current_value = simulated_state.get(attribute, 0.0)
                        # Special handling for needs like hunger (lower is better) vs resources (higher is better)
                        # This simplistic update assumes all attributes are numeric and additive/subtractive.
                        simulated_state[attribute] = current_value + change
                        # Clamp values if necessary (e.g., hunger 0-1, energy 0-1)
                        if attribute == "hunger":
                            simulated_state[attribute] = max(
                                0.0, min(simulated_state[attribute], 1.0)
                            )
                        elif attribute == "energy":
                            simulated_state[attribute] = max(
                                0.0, min(simulated_state[attribute], 1.0)
                            )
                        # Money can be unbounded (or have a floor of 0)
                        elif attribute == "money":
                            simulated_state[attribute] = max(
                                0.0, simulated_state[attribute]
                            )
    return total_utility


# === DATA VALIDATION AND HELPER FUNCTIONS ===


def validate_character_state(character_state: Dict[str, any]) -> Tuple[bool, str]:
    """
    Validate that character_state conforms to expected structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(character_state, dict):
        return False, "character_state must be a dictionary"

    required_attributes = ["hunger", "energy", "health"]
    for attr in required_attributes:
        if attr not in character_state:
            return False, f"Missing required attribute: {attr}"

        value = character_state[attr]
        if not isinstance(value, (int, float)):
            return False, f"Attribute {attr} must be numeric, got {type(value)}"

        if attr in ["hunger", "energy", "health"] and not (0.0 <= value <= 1.0):
            return False, f"Attribute {attr} must be between 0.0 and 1.0, got {value}"

    return True, ""


def validate_action(action: Action) -> Tuple[bool, str]:
    """
    Validate that action conforms to expected structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not hasattr(action, "name"):
        return False, "Action must have a 'name' attribute"

    if not hasattr(action, "cost"):
        return False, "Action must have a 'cost' attribute"

    if not isinstance(action.cost, (int, float)) or action.cost < 0:
        return False, f"Action cost must be non-negative number, got {action.cost}"

    if hasattr(action, "effects") and action.effects:
        if not isinstance(action.effects, list):
            return False, "Action effects must be a list"

        for i, effect in enumerate(action.effects):
            if not isinstance(effect, dict):
                return False, f"Effect {i} must be a dictionary"

            if "attribute" not in effect or "change_value" not in effect:
                return (
                    False,
                    f"Effect {i} must have 'attribute' and 'change_value' keys",
                )

            if not isinstance(effect["change_value"], (int, float)):
                return False, f"Effect {i} change_value must be numeric"

    return True, ""


def validate_goal(goal: Goal) -> Tuple[bool, str]:
    """
    Validate that goal conforms to expected structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not hasattr(goal, "name"):
        return False, "Goal must have a 'name' attribute"

    if hasattr(goal, "priority"):
        if not isinstance(goal.priority, (int, float)) or not (
            0.0 <= goal.priority <= 1.0
        ):
            return (
                False,
                f"Goal priority must be between 0.0 and 1.0, got {goal.priority}",
            )

    if hasattr(goal, "target_effects") and goal.target_effects:
        if not isinstance(goal.target_effects, dict):
            return False, "Goal target_effects must be a dictionary"

        for attr, target in goal.target_effects.items():
            if not isinstance(target, (int, float)):
                return False, f"Goal target_effect for {attr} must be numeric"

    return True, ""


def safe_calculate_action_utility(
    character_state: dict,
    action: Action,
    current_goal: Goal = None,
    validate_inputs: bool = True,
) -> Tuple[float, str]:
    """
    Safe wrapper for calculate_action_utility with input validation.

    Returns:
        Tuple of (utility_value, error_message)
    """
    if validate_inputs:
        # Validate character state
        is_valid, error = validate_character_state(character_state)
        if not is_valid:
            return 0.0, f"Invalid character_state: {error}"

        # Validate action
        is_valid, error = validate_action(action)
        if not is_valid:
            return 0.0, f"Invalid action: {error}"

        # Validate goal if provided
        if current_goal:
            is_valid, error = validate_goal(current_goal)
            if not is_valid:
                return 0.0, f"Invalid goal: {error}"

    try:
        utility = calculate_action_utility(character_state, action, current_goal)
        return utility, ""
    except Exception as e:
        return 0.0, f"Error calculating utility: {str(e)}"


def get_utility_statistics(utilities: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of utility values.

    Returns:
        Dictionary with statistics (mean, median, std, min, max)
    """
    if not utilities:
        return {"count": 0}

    import statistics

    return {
        "count": len(utilities),
        "mean": statistics.mean(utilities),
        "median": statistics.median(utilities),
        "std": statistics.stdev(utilities) if len(utilities) > 1 else 0.0,
        "min": min(utilities),
        "max": max(utilities),
    }


# === DOCUMENTATION AND EXAMPLES ===


def get_utility_system_info() -> str:
    """
    Return comprehensive documentation about the utility system.
    """
    return f"""
    Tiny Village Utility System Documentation
    =========================================
    
    This module provides comprehensive utility evaluation for actions and goals in the simulation.
    
    Key Components:
    ---------------
    1. UtilityEvaluator: Advanced evaluator with caching, history, and context awareness
    2. calculate_action_utility: Basic utility calculation for individual actions
    3. calculate_plan_utility: Utility calculation for sequences of actions
    4. calculate_importance: Enhanced goal importance evaluation
    
    Data Structure Requirements:
    ---------------------------
    
    character_state (dict):
        - "hunger": float (0.0=full, 1.0=very hungry)
        - "energy": float (0.0=exhausted, 1.0=fully energized)
        - "health": float (0.0=dead, 1.0=perfect health)
        - "money": float (0.0+, unbounded)
        - "social_needs": float (0.0=fulfilled, 1.0=very lonely)
    
    action.effects (list of dicts):
        [{{"attribute": "hunger", "change_value": -0.5}}, ...]
        - attribute: string matching character_state keys
        - change_value: float (positive=increase, negative=decrease)
    
    goal.target_effects (dict):
        {{"hunger": -0.8, "energy": 0.6}}
        - Keys match character_state attributes
        - Values are target changes or absolute targets
    
    Example Usage:
    --------------
    
    # Basic usage
    char_state = {{"hunger": 0.8, "energy": 0.3, "health": 0.9}}
    action = MockAction("EatMeal", cost=0.1, effects=[{{"attribute": "hunger", "change_value": -0.6}}])
    utility = calculate_action_utility(char_state, action)
    
    # Advanced usage with evaluator
    evaluator = UtilityEvaluator()
    environment = {{"time_of_day": 18, "weather": "sunny"}}
    advanced_utility = evaluator.evaluate_action_utility("char1", char_state, action, environment=environment)
    
    Constants:
    ----------
    - HUNGER_SCALER: {HUNGER_SCALER} (importance weight for hunger needs)
    - ENERGY_SCALER: {ENERGY_SCALER} (importance weight for energy needs)
    - HEALTH_SCALER: {HEALTH_SCALER} (importance weight for health needs)
    - SOCIAL_SCALER: {SOCIAL_SCALER} (importance weight for social needs)
    - MONEY_SCALER: {MONEY_SCALER} (importance weight for money gains)
    - HISTORY_DECAY_FACTOR: {HISTORY_DECAY_FACTOR} (how much historical data decays)
    - DIMINISHING_RETURNS_FACTOR: {DIMINISHING_RETURNS_FACTOR} (factor for repeated actions)
    """
