""" 
4. Utility Evaluation
Where it happens: Part of goap_system.py or a dedicated utility module
What happens: Actions are evaluated based on their expected outcomes, like improving mood or health, with adjustments based on current needs and previous experiences. """

# from tiny_characters import Goal
from actions import Action, State


# function that evaluates the current importance of the goal based on the character's state and the environment.
# tiny_utility_functions.py


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

    # Calculate the weighted importance score
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
    # Evaluate the importance of the goal based on the character's state
    for attribute, value in goal.attributes.items():
        importance += value * character_state[attribute]

    # Adjust the importance based on the environment
    for attribute, value in environment.items():
        if attribute in goal.attributes:
            importance += value * goal.attributes[attribute]

    return importance


def is_goal_achieved(goal, character_state: State):
    """
    Checks if the goal has been achieved based on the character's current state.

    Args:
        goal (Goal): The goal to check.
        character_state (State): The character's current state.

    Returns:
        achieved (bool): True if the goal is achieved, False otherwise.
    """
    for attribute, value in goal.attributes.items():
        if character_state.get(attribute, 0) < value:
            return False
    return True


# class UtilityEvaluator:
#     """
#     The UtilityEvaluator class is responsible for evaluating the utility of each possible action based on the character's current state and the planning context.
#     """

#     def __init__(self, actions, character_state, planning_context):
#         self.actions = actions
#         self.character_state = character_state
#         self.planning_context = planning_context

#     def evaluate_utility(self):
#         """
#         Evaluates the utility of each possible action.

#         Returns:
#             ranked_actions (list): Actions sorted by their utility scores.
#         """
#         utilities = {}
#         for action in self.actions:
#             utility_score = 0
#             # Evaluate base on planning context specifics
#             if self.planning_context == "daily":
#                 utility_score += (
#                     action["time_efficiency"] * self.character_state["time_preference"]
#                 )
#                 utility_score += (
#                     action["satisfaction"]
#                     * self.character_state["happiness_importance"]
#                 )
#             elif self.planning_context == "career":
#                 utility_score += (
#                     action["career_growth"] * self.character_state["career_ambition"]
#                 )
#                 utility_score += (
#                     action["risk"] * -1 * self.character_state["risk_aversion"]
#                 )
#             elif self.planning_context == "relationship":
#                 utility_score += (
#                     action["relationship_improvement"]
#                     * self.character_state["social_importance"]
#                 )
#                 utility_score += (
#                     action["emotional_cost"]
#                     * -1
#                     * self.character_state["emotional_stability"]
#                 )

#             utilities[action["name"]] = utility_score

#         # Sort actions based on utility scores in descending order
#         ranked_actions = sorted(
#             utilities.items(), key=lambda item: item[1], reverse=True
#         )
#         return [action[0] for action in ranked_actions]
