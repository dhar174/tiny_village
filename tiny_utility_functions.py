""" 
4. Utility Evaluation
Where it happens: Part of goap_system.py or a dedicated utility module
What happens: Actions are evaluated based on their expected outcomes, like improving mood or health, with adjustments based on current needs and previous experiences. """

from actions import Action, State # Assuming Action class is available and has cost & effects

# function that evaluates the current importance of the goal based on the character's state and the environment.
# tiny_utility_functions.py

# --- Existing Functions ---
def calculate_importance(
    health,
    hunger,
    social_needs,
    current_activity,
    social_factor,
    event_participation_factor,
    goal_importance,
):
    """
    Calculate the importance of a goal based on character stats, goal attributes, and additional factors.
    :param health: Character's health level.
    :param hunger: Character's hunger level.
    :param social_needs: Character's social needs level.
    :param goal_urgency: Urgency of the goal.
    :param goal_benefit: Benefit of achieving the goal.
    :param goal_consequence: Consequence of not achieving the goal.
    :param current_activity: Character's current activity.
    :param social_factor: Influence of relationships.
    :param location_factor: Impact of location relevance.
    :param event_factor: Impact of current events.
    :param resource_factor: Availability of needed resources.
    :param social_influence: Social influence on the character.
    :param potential_utility: Utility of achieving the goal.
    :param path_achieves_goal: Whether the path will achieve the goal.
    :param safety_factor: Safety of the goal location.
    :param event_participation_factor: Character's participation in relevant events.
    :param nearest_resource: Proximity to the nearest resource.
    :param goal_importance: Specific importance score based on goal type.
    :return: A float representing the importance score.
    """
    # Example weighting factors
    health_weight = 0.05
    hunger_weight = 0.1
    social_needs_weight = 0.05
    urgency_weight = 0.15
    benefit_weight = 0.1
    consequence_weight = 0.1
    social_weight = 0.1
    location_weight = 0.05
    event_weight = 0.05
    resource_weight = 0.05
    social_influence_weight = 0.1
    potential_utility_weight = 0.1
    path_achieves_goal_weight = 0.1
    safety_weight = 0.05
    event_participation_weight = 0.05
    nearest_resource_weight = 0.05
    goal_importance_weight = 0.2  # Added to incorporate specific goal importance
    social_weight = 0.1
    event_participation_weight = 0.05
    goal_importance_weight = 0.2

    importance_score = (
        health_weight * health
        + hunger_weight * hunger
        + social_needs_weight * social_needs
        + social_weight * social_factor
        + event_participation_weight * event_participation_factor
        + goal_importance_weight * goal_importance
    )
    return importance_score

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
    if hasattr(goal, 'attributes') and isinstance(goal.attributes, dict):
        for attribute, value in goal.attributes.items():
            # Assuming character_state can be accessed like a dict or has a get method
            importance += value * character_state.get(attribute, 0) 

    if hasattr(goal, 'attributes') and isinstance(goal.attributes, dict) and isinstance(environment, dict):
        for attribute, value in environment.items():
            if attribute in goal.attributes:
                importance += value * goal.attributes[attribute]
    return importance

def is_goal_achieved(goal, character_state: State):
    if hasattr(goal, 'target_effects') and isinstance(goal.target_effects, dict):
        for effect, target_value in goal.target_effects.items():
            if character_state.get(effect, 0) < target_value:
                return False
        return True
    return False # Default if goal has no target_effects to check

# --- New Utility Calculation Functions ---

def calculate_action_utility(character_state: dict, action: Action, current_goal: Goal = None) -> float:
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
                if change < 0: # Action reduces hunger
                    need_fulfillment_score += current_hunger * abs(change) * HUNGER_SCALER # Scaler for impact
            elif attribute == "energy":
                # Lower current energy + action increases energy = good
                current_energy = character_state.get("energy", 0.0) # Assume 0 is empty, 1 is full
                if change > 0: # Action increases energy
                    need_fulfillment_score += (1.0 - current_energy) * change * ENERGY_SCALER # Scaler for impact
            elif attribute == "money":
                # This is a resource change, not directly "need" fulfillment in the same way.
                # Could be handled by specific actions like "Work" having positive utility here,
                # or costs handling money decrease.
                # For now, let's say getting money is a direct utility gain if not a primary need.
                if change > 0:
                    need_fulfillment_score += change * MONEY_SCALER # Money gained is somewhat good
    
    utility += need_fulfillment_score

    # 2. Goal Progress
    if current_goal and hasattr(current_goal, 'target_effects') and current_goal.target_effects:
        if action.effects:
            for effect in action.effects:
                attr = effect.get("attribute")
                change = effect.get("change_value", 0.0)
                if attr in current_goal.target_effects:
                    goal_target_change = current_goal.target_effects[attr]
                    # If action moves attribute towards goal (e.g., hunger goal -0.5, action does -0.2)
                    if (goal_target_change < 0 and change < 0) or \
                       (goal_target_change > 0 and change > 0):
                        # Simple progress: add priority * scale. More sophisticated: % of goal achieved.
                        goal_progress_score += current_goal.priority * 25.0 
                        # Break if one effect contributes, to avoid over-counting for multi-effect actions
                        # This is a simplification.
                        break 
    
    utility += goal_progress_score

    # 3. Action Cost
    # Assuming action.cost is a non-negative value representing effort/time/direct_resource_depletion
    # This cost is separate from specific resource changes in 'effects' (e.g. an Eat action might have low cost but eating an item removes it)
    action_cost_score = 0.0
    if hasattr(action, 'cost'):
        action_cost_score = float(action.cost) * 10.0 # Scaler for cost impact

    utility -= action_cost_score
    
    # Consider inherent utility/disutility of certain actions if not captured by needs/goals/cost
    # Example: "Rest" might have a small positive base utility if not costly and energy is low.
    # "Argue" might have a small negative base utility.
    # For now, this is not explicitly added.

    return utility


def calculate_plan_utility(character_state: dict, plan: list[Action], current_goal: Goal = None, simulate_effects=False) -> float:
    """
    Calculates the utility of a sequence of actions (a plan).

    If simulate_effects is True, the character_state is updated after each action
    to calculate the utility of subsequent actions more accurately.
    """
    total_utility = 0.0
    
    if not simulate_effects:
        for action in plan:
            total_utility += calculate_action_utility(character_state, action, current_goal)
    else:
        # Create a deep copy for simulation if character_state can contain nested dicts/lists
        # For simple dict of floats, .copy() is fine.
        simulated_state = character_state.copy() 
        for action in plan:
            action_utility = calculate_action_utility(simulated_state, action, current_goal)
            total_utility += action_utility
            
            # Update simulated_state based on action's effects
            if action.effects:
                for effect in action.effects:
                    attribute = effect.get("attribute")
                    change = effect.get("change_value", 0.0)
                    
                    if attribute: # Ensure attribute is not None
                        current_value = simulated_state.get(attribute, 0.0)
                        # Special handling for needs like hunger (lower is better) vs resources (higher is better)
                        # This simplistic update assumes all attributes are numeric and additive/subtractive.
                        simulated_state[attribute] = current_value + change
                        
                        # Clamp values if necessary (e.g., hunger 0-1, energy 0-1)
                        if attribute == "hunger":
                            simulated_state[attribute] = max(0.0, min(simulated_state[attribute], 1.0)) 
                        elif attribute == "energy":
                            simulated_state[attribute] = max(0.0, min(simulated_state[attribute], 1.0))
                        # Money can be unbounded (or have a floor of 0)
                        elif attribute == "money":
                             simulated_state[attribute] = max(0.0, simulated_state[attribute])
    return total_utility
