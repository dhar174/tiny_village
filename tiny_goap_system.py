"""GOAP System Structure
Define Goals:
Goals are end states that the character aims to achieve. These could range from personal (e.g., improve a relationship, learn a skill) to professional (e.g., get a promotion, change careers).
Define Actions:
Each action available to the character should have conditions (requirements to be able to perform the action) and effects (results of performing the action). Actions are the building blocks of the plans that GOAP will create.
Planning Algorithm:
The planner takes the current state of the world (including the character's state) and a goal, and it generates a plan that outlines the sequence of actions needed to achieve that goal.
This involves searching through the space of possible actions to find a sequence that transitions the character from their current state to the goal state with minimal cost.
Integrating Graph Analysis
The graph analysis provides a dynamic view of the character’s world, including relationships, interests, and historical interactions, which are crucial for setting realistic goals and conditions for actions.

Example Steps for Integration:
Graph-Based Condition Setting:
Use the output from the graph analysis to set conditions for actions. For example, if the goal is to improve a relationship, the condition might be that the relationship strength is below a certain threshold.
Graph-Based Effect Prediction:
Predict the potential effects of actions on the character’s network by simulating changes in the graph. For instance, attending social events might strengthen certain relationships.
Feedback Loop:
Post-action, update the graph based on the outcomes, which then informs future GOAP planning.

Further Development
Action Definitions: Further detail actions with specific conditions and effects based on the character’s attributes and the graph analysis.
Cost Functions: Integrate a cost function to select the optimal path among several possibilities.
Real-Time Adjustments: Allow the GOAP system to adjust plans in real-time based on new information or changes in the game state, using continuous feedback from the graph analysis.

3. GOAP System and Graph Analysis
Where it happens: goap_system.py and graph_manager.py
What happens: The GOAP planner uses the graph to analyze relationships and preferences, and formulates a plan consisting of a sequence of actions that maximize the character’s utility for the day.
"""

from tiny_types import Goal, Character, GraphManager
from actions import Action, State

# from tiny_graph_manager import GraphManager

from heapq import heappush, heappop

import tiny_utility_functions as util_funcs

import logging
import importlib

logging.basicConfig(level=logging.DEBUG)


class Plan:
    """
    Represents a plan for a character in the game.

    Attributes:
        name (str): The name of the plan.
        goals (list): A list of goals that are part of the plan.
        actions (list): A list of actions that are part of the plan.
        current_goal_index (int): Index of the current goal being evaluated.
        current_action_index (int): Index of the current action being evaluated.
        completed_actions (set): Set of completed actions.
    """

    def __init__(self, name, goals=None, actions=None):
        self.name = name
        self.goals = goals if goals is not None else []
        self.action_queue = []
        self.current_goal_index = 0
        self.completed_actions = set()
        self.current_action_index = 0

        if actions is not None:
            for action in actions:
                self.add_action(
                    action["action"], action["priority"], action["dependencies"]
                )

    def add_goal(self, goal):
        self.goals.append(goal)

    def add_action(self, action, priority=0, dependencies=None):
        if dependencies is None:
            dependencies = []
        heappush(self.action_queue, (priority, action, dependencies))

    def evaluate(self):
        """
        Evaluates the plan by checking the completion status of goals and actions.
        """
        for goal in self.goals:
            if not goal.check_completion():
                return False
        return True

    def replan(self):
        """
        Replans the sequence of actions based on the current game state.
        """
        print("Replanning based on new game state...")

        # Clear current action queue to rebuild with updated priorities
        old_queue = list(self.action_queue)
        self.action_queue = []

        # Re-evaluate and re-prioritize actions based on current goals
        for priority, action, dependencies in old_queue:
            # Skip actions that are already completed
            if action.name in self.completed_actions:
                continue

            # Re-calculate priority based on current state
            # Higher priority (lower number) for more urgent actions
            new_priority = priority

            # Adjust priority based on action urgency and current goal progress
            if hasattr(action, "cost") and hasattr(action, "urgency"):
                new_priority = action.cost / max(action.urgency, 0.1)

            # Re-add action with updated priority
            heappush(self.action_queue, (new_priority, action, dependencies))

    def execute(self):
        while self.current_goal_index < len(self.goals):
            self.replan()
            current_goal = self.goals[self.current_goal_index]
            if not current_goal.check_completion():
                while self.action_queue:
                    priority, current_action, dependencies = heappop(self.action_queue)
                    if all(dep in self.completed_actions for dep in dependencies):
                        if current_action.preconditions_met():
                            success = current_action.execute(
                                target=current_action.target,
                                initiator=current_action.initiator,
                            )
                            if success:
                                self.completed_actions.add(current_action.name)
                                break
                            else:
                                self.handle_failure(current_action)
                                return False
                self.current_goal_index += 1
            else:
                self.current_goal_index += 1
        self.handle_success()
        return True

    def handle_failure(self, action):
        """
        Handles the failure of an action within the plan.
        """
        print(f"Action {action.name} failed. Re-evaluating plan.")
        self.replan()

    def handle_success(self):
        """
        Handles the successful completion of the plan.
        """
        print(f"Plan {self.name} successfully completed.")

    def handle_failure(self, action):
        print(f"Action {action.name} failed. Re-evaluating plan.")
        # Enhanced error handling logic
        try:
            self.replan()
        except Exception as e:
            print(f"Replanning failed due to: {e}")
            # Retry logic or switch to an alternative plan
            alternative_action = self.find_alternative_action(action)
            if alternative_action:
                self.add_action(
                    alternative_action["action"],
                    alternative_action["priority"],
                    alternative_action["dependencies"],
                )
                self.execute()

    def find_alternative_action(self, failed_action):
        """Logic to find an alternative action for the failed action"""
        print(f"Finding alternative action for {failed_action.name}")

        # Look for actions with similar effects but different preconditions
        alternative_name = f"alt_{failed_action.name}"

        # Create a simplified alternative with relaxed preconditions
        try:
            # Copy basic properties from failed action
            alt_preconditions = {}
            alt_effects = getattr(failed_action, "effects", {})
            alt_cost = getattr(failed_action, "cost", 5) + 2  # Slightly higher cost

            # Create alternative action with more flexible requirements
            alternative_action = Action(
                name=alternative_name,
                preconditions=alt_preconditions,  # Empty preconditions for flexibility
                effects=alt_effects,
                cost=alt_cost,
                target=getattr(failed_action, "target", None),
                initiator=getattr(failed_action, "initiator", None),
                related_skills=getattr(failed_action, "related_skills", []),
            )

            return {
                "action": alternative_action,
                "priority": 2,  # Lower priority than original
                "dependencies": [],
            }
        except Exception as e:
            print(f"Could not create alternative action: {e}")
            return None

    def __str__(self):
        return f"Plan: {self.name} - Goals: {self.goals} - Actions: {self.action_queue} - Completed Actions: {self.completed_actions} - Current Goal Index: {self.current_goal_index} - Current Action Index: {self.current_action_index} - Completed: {self.evaluate()}"

    def __repr__(self):
        return f"Plan: {self.name} - Goals: {self.goals} - Actions: {self.action_queue} - Completed Actions: {self.completed_actions} - Current Goal Index: {self.current_goal_index} - Current Action Index: {self.current_action_index} - Completed: {self.evaluate()}"

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.goals == other.goals
            and self.action_queue == other.action_queue
            and self.completed_actions == other.completed_actions
            and self.current_goal_index == other.current_goal_index
            and self.current_action_index == other.current_action_index
            and self.evaluate() == other.evaluate()
        )

    def __hash__(self):
        return hash(
            tuple(
                self.name,
                self.goals,
                self.action_queue,
                self.completed_actions,
                self.current_goal_index,
                self.current_action_index,
                self.evaluate(),
            )
        )


# # Example of using the Plan class with Goal and Action classes
# goal1 = Goal(
#     description="Find food",
#     character=character,
#     target=food_location,
#     score=10,
#     name="find_food",
#     completion_conditions=[...],
#     evaluate_utility=...,
#     difficulty=...,
#     completion_reward=...,
#     failure_penalty=...,
#     completion_message=...,
#     failure_message=...,
#     criteria=[...],
# )

# action1 = Action(
#     name="search_for_food",
#     preconditions={...},
#     effects={...},
#     cost=5,
#     target=food_location,
#     initiator=character,
# )

# action2 = Action(
#     name="hunt_for_animals",
#     preconditions={...},
#     effects={...},
#     cost=8,
#     target=forest_location,
#     initiator=character,
# )

# plan = Plan(name="Survival Plan")
# plan.add_goal(goal1)
# plan.add_action(action1, priority=2)
# plan.add_action(action2, priority=1, dependencies=["search_for_food"])

# plan.execute()


class GOAPPlanner:
    def __init__(self, graph_manager: "GraphManager"):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
class GOAPPlanner:
    def __init__(self, graph_manager: "GraphManager"):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        Goal = importlib.import_module("tiny_characters").Goal # Add Goal import
        Condition = importlib.import_module("actions").Condition # Add Condition import
        # Action import needed for type hinting if used in goap_planner signature strictly
        # from actions import Action
        self.graph_manager = graph_manager
        self.plans = {}

    # def plan_actions(self, state, actions): # This seems to be an old or alternative planning method stub
    #     goal_difficulty = self.calculate_goal_difficulty(character, goal)
    #     return sorted(actions, key=lambda x: -x.utility - goal_difficulty["difficulty"])

    def _hashable_state(self, state_dict: dict) -> tuple:
        """Converts a dictionary (including nested ones) to a hashable tuple of sorted items."""
        if isinstance(state_dict, dict):
            return tuple(sorted((k, self._hashable_state(v)) for k, v in state_dict.items()))
        elif isinstance(state_dict, list):
            return tuple(self._hashable_state(i) for i in state_dict)
        return state_dict

    def heuristic(self, current_state_dict: dict, target_goal: Goal) -> float:
        """Estimates the cost to reach the target_goal from current_state_dict."""
        h_score = 0
        # Assuming target_goal.completion_conditions is a dict of lists of Condition objects
        if hasattr(target_goal, 'completion_conditions') and isinstance(target_goal.completion_conditions, dict):
            for condition_list in target_goal.completion_conditions.values():
                for condition in condition_list:
                    if isinstance(condition, importlib.import_module("actions").Condition): # Check type
                        if not condition.is_met(current_state_dict):
                            h_score += 1 # Each unmet condition adds 1 to the heuristic
                    else:
                        logging.warning(f"Item in completion_conditions is not a Condition object: {condition}")
        else:
            logging.warning(f"Goal '{target_goal.name}' has no valid completion_conditions attribute for heuristic calculation.")
        return h_score

    def _is_goal_achieved(self, current_state_dict: dict, target_goal: Goal) -> bool:
        """Checks if all completion conditions of the goal are met in the current state."""
        if hasattr(target_goal, 'completion_conditions') and isinstance(target_goal.completion_conditions, dict):
            if not target_goal.completion_conditions: # No conditions means goal is trivially achieved
                return True
            all_conditions_met = True
            for condition_list in target_goal.completion_conditions.values():
                if not condition_list: # Empty list of conditions for a key
                    continue
                for condition in condition_list:
                    if isinstance(condition, importlib.import_module("actions").Condition):
                        if not condition.is_met(current_state_dict):
                            all_conditions_met = False
                            break # One condition not met is enough
                    else:
                        # logging.warning(f"Non-Condition object in goal completion_conditions: {condition}")
                        all_conditions_met = False # Treat non-Condition as problematic / not met
                        break
                if not all_conditions_met:
                    break
            return all_conditions_met
        # If using target_effects as primary (or fallback)
        # elif hasattr(target_goal, 'target_effects') and isinstance(target_goal.target_effects, dict):
        #     for attr, val in target_goal.target_effects.items():
        #         if Condition._get_nested_attribute(current_state_dict, attr) != val: # Using static method
        #             return False
        #     return True
        logging.warning(f"Goal '{target_goal.name}' has no completion_conditions or suitable target_effects for achievement check.")
        return False # Default if no clear conditions

    def goap_planner(self, character_state: State, target_goal: Goal, available_actions: list[Action]) -> list[Action] | None:
        """
        Generates a plan of actions to achieve a target_goal from the character_state using A*.
        """
        character_obj = character_state.dict_or_obj # Get the actual character object from State
        char_name = getattr(character_obj, 'name', 'UnknownCharacter')
        goal_name = getattr(target_goal, 'name', 'UnknownGoal')
        logging.debug(f"GOAPPlanner: Starting planning for {char_name} to achieve goal '{goal_name}'.")

        if not hasattr(character_obj, 'to_dict'):
            logging.error(f"Character object {char_name} within State does not have a to_dict() method.")
            return None
        initial_state_dict = character_obj.to_dict()

        open_list = []  # Priority queue (min-heap)
        # (f_score, g_score, state_dict, plan_actions_list)

        initial_h_score = self.heuristic(initial_state_dict, target_goal)
        initial_g_score = 0
        initial_f_score = initial_g_score + initial_h_score

        heappush(open_list, (initial_f_score, initial_g_score, initial_state_dict, []))

        # came_from stores (previous_hashed_state, action_taken_to_reach_current_state)
        came_from = {self._hashable_state(initial_state_dict): (None, None)}
        cost_so_far = {self._hashable_state(initial_state_dict): 0} # Stores g_score

        max_iterations = 1000 # Prevent infinite loops in complex scenarios
        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1
            f_score, g_score, current_state_dict, current_plan_actions = heappop(open_list)

            if self._is_goal_achieved(current_state_dict, target_goal):
                # Reconstruct path
                plan = []
                hashed_curr = self._hashable_state(current_state_dict)
                while hashed_curr in came_from and came_from[hashed_curr][1] is not None:
                    prev_hashed, action = came_from[hashed_curr]
                    plan.append(action)
                    hashed_curr = prev_hashed
                plan.reverse()
                logging.info(f"GOAPPlanner: Plan found for {char_name}, goal '{goal_name}'. Plan length: {len(plan)} steps.")
                return plan

            for action in available_actions:
                if not isinstance(action, Action):
                    logging.warning(f"Item in available_actions is not an Action object: {action}")
                    continue

                if action.are_preconditions_met(current_state_dict):
                    next_state_dict = action.apply_effects(current_state_dict)
                    new_g_score = g_score + action.cost

                    hashed_next_state = self._hashable_state(next_state_dict)

                    if hashed_next_state not in cost_so_far or new_g_score < cost_so_far[hashed_next_state]:
                        cost_so_far[hashed_next_state] = new_g_score
                        h_val = self.heuristic(next_state_dict, target_goal)
                        new_f_score = new_g_score + h_val

                        heappush(open_list, (new_f_score, new_g_score, next_state_dict, current_plan_actions + [action]))
                        came_from[hashed_next_state] = (self._hashable_state(current_state_dict), action)

        if iterations >= max_iterations:
            logging.warning(f"GOAPPlanner: Reached max iterations ({max_iterations}) for {char_name}, goal '{goal_name}'.")

        logging.info(f"GOAPPlanner: No plan found for {char_name} to achieve goal '{goal_name}'.")
        return None

    def calculate_goal_difficulty(self, character, goal: Goal):
        """
        Calculates the difficulty of achieving a given goal for a character.

        Args:
            character: The character for whom the goal difficulty is being calculated.
            goal (Goal): The goal for which the difficulty is being calculated.

        Returns:
            Dict: {
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
        """
        Character = importlib.import_module("tiny_characters").Character
        Goal = importlib.import_module("tiny_characters").Goal
        return self.graph_manager.calculate_goal_difficulty(goal, character)

    def evaluate_goal_importance(
        self, character: Character, goal: Goal, graph_manager: GraphManager, **kwargs
    ) -> float:
        """
        Evaluates the importance of a goal to a character based on the current game state, character's stats, and personal motives.

        :param character: Character object containing all attributes and current state.
        :param goal: Goal object containing the details of the goal.
        :param graph_manager: GraphManager object to access the game graph.
        :return: A float value representing the importance of the goal.
        """
        Character = importlib.import_module("tiny_characters").Character
        Goal = importlib.import_module("tiny_characters").Goal
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        logging.info(
            f"Evaluating importance of goal {goal.name} for character {character.name}"
        )
        # logging.debug(f"Character:\n {character}\n")
        # logging.debug(f"Goal:\n {goal}\n")
        for arg in kwargs:
            logging.debug(
                f"\nAdditional argument:\n {arg} \nof type {type(kwargs[arg])}"
            )
        # TODO: We need Machine Learning here
        # For now, implement a comprehensive heuristic-based approach

        # Fetch character attributes
        health = character.health_status
        hunger = character.hunger_level
        social_needs = character.social_wellbeing
        current_activity = (
            character.current_activity
        )  # Action class object of the current activity
        financial_status = character.wealth_money

        # Fetch goal attributes
        goal_benefit = goal.completion_reward
        goal_consequence = goal.failure_penalty
        goal_type = goal.goal_type  # Added to differentiate goal types

        # Fetch personal motives
        motives = character.motives
        social_factor = 0
        event_participation_factor = 0

        # Analyze character relationships if the character has been fully initialized
        if character._initialized:
            relationships = graph_manager.analyze_character_relationships(character)
            social_factor = sum(
                graph_manager.evaluate_relationship_strength(character, rel)
                for rel in relationships.keys()
            )

            # Implement the relationship analysis
            try:
                for rel_name, rel_data in relationships.items():
                    # Calculate how this goal impacts each relationship
                    if hasattr(graph_manager, "calculate_how_goal_impacts_character"):
                        impact = graph_manager.calculate_how_goal_impacts_character(
                            character, rel_name, goal
                        )

                        # Weight the impact by relationship strength
                        rel_strength = graph_manager.evaluate_relationship_strength(
                            character, rel_name
                        )
                        weighted_impact = impact * rel_strength

                        # Add to social factor (positive or negative)
                        social_factor += weighted_impact

                    # Consider relationship type and goals
                    if hasattr(rel_data, "relationship_type"):
                        rel_type = rel_data.relationship_type
                        if rel_type in ["family", "romantic"]:
                            # Family and romantic relationships have higher impact
                            social_factor *= 1.5
                        elif rel_type in ["friend", "colleague"]:
                            social_factor *= 1.2
                        elif rel_type in ["enemy", "rival"]:
                            # Negative relationships might actually motivate certain goals
                            if goal_type in ["competitive", "achievement"]:
                                social_factor += 10  # Boost competitive goals
            except Exception as e:
                logging.warning(
                    f"Error in relationship analysis for goal evaluation: {e}"
                )
                # Continue with basic social factor calculation
            #         goal, character
            #     )

            # Analyze location relevance and safety
            # location = graph_manager.G.nodes[character.name].get(
            #     "coordinates_location", None
            # )
            # location_factor = 0
            # safety_factor = 0
            # if location and goal.target_location:
            #     path = graph_manager.find_shortest_path(location, goal.target_location)
            #     location_factor = (
            #         1 / (len(path) + 1) if path else 0
            #     )  # Simplified proximity impact
            #     safety_factor = (
            #         1
            #         if graph_manager.check_safety_of_locations(goal.target_location)
            #         else 0
            #     )

            # Analyze event impact and participation
            current_events = [
                event for event in graph_manager.events.values() if event.is_active()
            ]
            event_participation_factor = sum(
                event.importance
                for event in current_events
                if self.graph_manager.G[character.name][event.name][
                    "participation_status"
                ]
                == True
            )

        # # Check if the path will achieve the goal
        # path_achieves_goal = graph_manager.will_path_achieve_goal(character.name, goal)

        # Add specific criteria based on goal type and personal motives
        goal_importance = 0
        if goal_type == "Basic Needs":
            goal_importance = self.calculate_basic_needs_importance(
                health, hunger, motives
            )
        elif goal_type == "Social":
            goal_importance = self.calculate_social_goal_importance(
                social_needs, social_factor, motives
            )
        elif goal_type == "Career":
            goal_importance = self.calculate_career_goal_importance(
                character, goal, graph_manager, motives
            )
        elif goal_type == "Personal Development":
            goal_importance = self.calculate_personal_development_importance(
                character, goal, graph_manager, motives
            )
        elif goal_type == "Economic":
            goal_importance = self.calculate_economic_goal_importance(
                financial_status, goal, graph_manager, motives, character # Pass character
            )
        elif goal_type == "Health":
            goal_importance = self.calculate_health_goal_importance(
                health, goal, graph_manager, motives, character # Pass character
            )
        elif goal_type == "Safety":
            goal_importance = self.calculate_safety_goal_importance(goal, motives, graph_manager, character)
        elif goal_type == "Long-term Aspiration":
            goal_importance = self.calculate_long_term_goal_importance(
                character, goal, graph_manager, motives
            )

        # Combine general and specific criteria
        importance_score = util_funcs.calculate_importance(
            health=health,
            hunger=hunger,
            social_needs=social_needs,
            current_activity=current_activity,
            social_factor=social_factor,
            event_participation_factor=event_participation_factor,
            goal_importance=goal_importance,  # Specific to the goal type
        )

        return importance_score

    def calculate_basic_needs_importance(self, health, hunger, motives):
        # Calculate the importance of basic needs goals
        return (
            0.3 * health
            + 0.4 * hunger
            + 0.2 * motives.get_hunger_motive().get_score()
            + 0.1 * motives.get_health_motive().get_score()
        )

    def calculate_social_goal_importance(self, social_needs, social_factor, motives):
        # Calculate the importance of social goals
        return (
            0.5 * social_needs
            + 0.3 * social_factor
            + 0.2 * motives.get_social_wellbeing_motive().get_score()
        )

    def calculate_career_goal_importance(self, character, goal, graph_manager, motives):
        # Calculate the importance of career goals
        job_opportunities = graph_manager.explore_career_opportunities(character.name)
        # career_impact = graph_manager.analyze_career_impact(character.name, goal) # Ensure this method exists and is used if needed
        # For now, focusing on motive score
        return (
            0.4 * character.job_performance # Assuming job_performance is a relevant character attribute for career goals
            + 0.4 * job_opportunities # Assuming this returns a quantifiable metric
            + 0.2 * motives.get_job_performance_motive().get_score()
        )

    def calculate_personal_development_importance(
        self, character: Character, goal: Goal, graph_manager: GraphManager, motives: PersonalMotives
    ):
        # Calculate the importance of personal development goals
        difficulty_score = 0
        if callable(goal.difficulty):
            # Assuming goal.difficulty returns a dict with a 'difficulty' key or just a score
            difficulty_result = goal.difficulty(character, graph_manager)
            if isinstance(difficulty_result, dict):
                difficulty_score = difficulty_result.get('difficulty', 0.5) # Default if key missing
            elif isinstance(difficulty_result, (int, float)):
                difficulty_score = difficulty_result
            else: # Default if unexpected type
                difficulty_score = 0.5
        else: # if goal.difficulty is not callable, use a default
            difficulty_score = getattr(goal, 'difficulty_value', 0.5) # Check for a direct value

        # Inverse of difficulty (lower difficulty = higher importance for development)
        # Normalize difficulty: assume it's 0-1, if not, it needs scaling.
        # For simplicity, if difficulty is 0-100, divide by 100. Assuming it's already normalized or a small value.
        personal_dev_factor = (1 - min(1, max(0, difficulty_score))) * 0.5 # Max contribution of 0.5 from difficulty

        # Consider if the goal has specific skills to improve
        # related_skill_improvement_factor = 0 # Default
        # if hasattr(goal, 'target_skills') and goal.target_skills:
        #    # Logic to see how much this goal helps with target skills based on character's current skills
        #    pass # Placeholder for more complex skill interaction logic

        return personal_dev_factor + 0.5 * motives.get_success_motive().get_score() # Success motive for personal dev

    def calculate_economic_goal_importance(
        self, financial_status: float, goal: Goal, graph_manager: GraphManager, motives: PersonalMotives, character: Character
    ):
        # Calculate the importance of economic goals
        trade_opportunity_factor = 0.0 # Default
        if hasattr(goal, 'required_items') and goal.required_items:
            # This is a list of tuples: (dict of item requirements, quantity needed)
            # For simplicity, let's consider the first required item if it exists
            # A more complex version would iterate or check graph_manager for general opportunities
            first_item_req = goal.required_items[0][0] if goal.required_items else None
            if first_item_req and graph_manager.evaluate_trade_opportunities_for_item:
                 # evaluate_trade_opportunities_for_item might need character or just item_name/type
                 # Assuming it takes item properties dict
                opportunities = graph_manager.evaluate_trade_opportunities_for_item(character, first_item_req)
                trade_opportunity_factor = min(1, opportunities * 0.1) # Normalize, e.g. if opportunities is count

        return 0.4 * financial_status + 0.4 * trade_opportunity_factor + 0.2 * motives.get_wealth_motive().get_score()

    def calculate_health_goal_importance(self, health: float, goal: Goal, graph_manager: GraphManager, motives: PersonalMotives, character: Character):
        # Calculate the importance of health goals
        health_facility_factor = 0.0 # Default
        if hasattr(goal, 'target_location_type') and goal.target_location_type == "health_facility":
            if graph_manager.get_nearest_resource:
                # Assuming get_nearest_resource needs character's current location and type of resource
                # And character has a 'location' attribute
                nearest_facility_info = graph_manager.get_nearest_resource(character.location, "health_facility")
                if nearest_facility_info and isinstance(nearest_facility_info, dict):
                    distance = nearest_facility_info.get('distance', float('inf'))
                    health_facility_factor = max(0, 1 - distance / 100.0) # Closer is better, normalized by 100 units

        return 0.6 * health + 0.2 * health_facility_factor + 0.2 * motives.get_health_motive().get_score()

    def calculate_safety_goal_importance(self, goal: Goal, motives: PersonalMotives, graph_manager: GraphManager, character: Character):
        # Calculate the importance of safety goals
        # Assuming safety_factor is derived from graph_manager or environment
        # Example: graph_manager.get_location_safety(character.location)
        safety_factor = graph_manager.get_node_attribute(character.location, "safety_score") if character.location else 0.5
        # Ensure goal has an urgency attribute
        goal_urgency = getattr(goal, 'urgency', 0.5)
        return safety_factor * goal_urgency + motives.get_stability_motive().get_score()

    def calculate_long_term_goal_importance(
        self, character, goal, graph_manager, motives
    ):
        # Calculate the importance of long-term goals
        progress = graph_manager.calculate_goal_progress(character.name, goal)
        goal_urgency = getattr(goal, 'urgency', 0.5) # Ensure goal has an urgency attribute
        return progress * goal_urgency + motives.get_hope_motive().get_score()

    def calculate_utility(self, action, character):
        """Calculate action utility based on multiple factors"""
        if isinstance(action, dict):
            # Handle dictionary format
            satisfaction = action.get("satisfaction", 0)
            energy_cost = action.get("energy_cost", 0)
            urgency = action.get("urgency", 1)
            base_utility = satisfaction - energy_cost
        else:
            # Handle Action object format
            satisfaction = getattr(action, "satisfaction", 0)
            energy_cost = getattr(action, "cost", 0)
            urgency = getattr(action, "urgency", 1)
            base_utility = satisfaction - energy_cost

        # Factor in character state if available
        if hasattr(character, "get_state"):
            char_state = character.get_state()
            # Adjust utility based on character's current needs
            if "energy" in char_state and energy_cost > 0:
                energy_factor = char_state["energy"] / 100.0  # Normalize to 0-1
                base_utility *= energy_factor

        # Apply urgency multiplier
        final_utility = base_utility * urgency

        return final_utility

    def evaluate_utility(self, plan, character):
        """Evaluate the utility of a plan by finding the action with highest utility"""
        if not plan:
            return None

        try:
            return max(plan, key=lambda x: self.calculate_utility(x, character))
        except (TypeError, ValueError) as e:
            print(f"Error evaluating plan utility: {e}")
            return plan[0] if plan else None

    def evaluate_feasibility_of_goal(self, goal, state):
        """Evaluate if a goal is feasible given the current state"""
        if not goal or not state:
            return False

        # Handle different goal formats
        if hasattr(goal, "completion_conditions"):
            # Goal object with completion conditions
            conditions = goal.completion_conditions
            if isinstance(conditions, dict):
                return all(state.get(k, 0) >= v for k, v in conditions.items())
            elif isinstance(conditions, list):
                # Assume conditions are callable functions
                return all(
                    condition(state) if callable(condition) else True
                    for condition in conditions
                )
        elif isinstance(goal, dict):
            # Dictionary format goal
            return all(
                state.get(k, 0) >= v
                for k, v in goal.items()
                if isinstance(v, (int, float))
            )

        # Default to feasible if we can't determine otherwise
        return True
