""" GOAP System Structure
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
        from tiny_characters import Goal

        print("Replanning based on new game state...")
        # # Example logic for dynamic replanning
        # if self.character.get_state().get("hunger") > 80:
        #     new_goal = Goal(
        #         description="Find water",
        #         character=self.character,
        #         target=water_source,
        #         score=8,
        #         name="find_water",
        #         completion_conditions=[...],
        #         evaluate_utility=...,
        #         difficulty=...,
        #         completion_reward=...,
        #         failure_penalty=...,
        #         completion_message=...,
        #         failure_message=...,
        #         criteria=[...],
        #     )
        #     self.add_goal(new_goal)
        #     new_action = Action(
        #         name="search_for_water",
        #         preconditions={...},
        #         effects={...},
        #         cost=4,
        #         target=water_source,
        #         initiator=self.character,
        #     )
        #     self.add_action(new_action, priority=1)

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
        # Logic to find an alternative action for the failed action
        print(f"Finding alternative action for {failed_action.name}")
        # Example alternative action (this should be dynamically determined)
        return {
            "action": Action(
                name="alternative_action",
                preconditions={...},
                effects={...},
                cost=5,
                target=failed_action.target,
                initiator=failed_action.initiator,
                related_skills=failed_action.related_skills,
            ),
            "priority": 1,
            "dependencies": [],
        }

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
        self.graph_manager = graph_manager
        self.plans = {}

    def plan_actions(self, state, actions):
        goal_difficulty = self.calculate_goal_difficulty(character, goal)
        return sorted(actions, key=lambda x: -x.utility - goal_difficulty["difficulty"])

    def goap_planner(character, goal: Goal, char_state: State, actions: list):
        """
        Generates a plan of actions to achieve a goal from the current state.

        Args:
            character (str): The character for whom the plan is being generated.
            goal (dict): The desired state to achieve.
            state (dict): The current state of the character.
            actions (list): A list of possible actions.

        Returns:
            plan (list): A list of actions that form a plan to achieve the goal.
        """
        Goal = importlib.import_module("tiny_characters").Goal
        Character = importlib.import_module("tiny_characters").Character
        State = importlib.import_module("tiny_characters").State

        # Initialize the open list with the initial state
        open_list = [(char_state, [])]
        visited_states = set()

        while open_list:
            current_state, current_plan = open_list.pop(0)

            # Check if the current state satisfies the goal
            if all(goal.check_completion(current_state)):
                return current_plan

            for action in actions:
                if action["conditions_met"](current_state):
                    new_state = current_state.copy()
                    # Apply the effects of the action
                    for effect, change in action["effects"].items():
                        new_state[effect] = new_state.get(effect, 0) + change

                    if new_state not in visited_states:
                        visited_states.add(new_state)
                        # Add the new state and updated plan to the open list
                        open_list.append((new_state, current_plan + [action["name"]]))

        return None  # If no plan is found

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

            # TODO: Implement the relationship analysis
            # for rel in relationships.keys():
            #     impact = self.graph_manager.calculate_how_goal_impacts_character(
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
                financial_status, goal, graph_manager, motives
            )
        elif goal_type == "Health":
            goal_importance = self.calculate_health_goal_importance(
                health, goal, graph_manager, motives
            )
        elif goal_type == "Safety":
            goal_importance = self.calculate_safety_goal_importance(goal, motives)
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
            + 0.2 * motives.hunger_motive.value
            + 0.1 * motives.health_motive.value
        )

    def calculate_social_goal_importance(self, social_needs, social_factor, motives):
        # Calculate the importance of social goals
        return (
            0.5 * social_needs
            + 0.3 * social_factor
            + 0.2 * motives.social_wellbeing_motive.value
        )

    def calculate_career_goal_importance(self, character, goal, graph_manager, motives):
        # Calculate the importance of career goals
        job_opportunities = graph_manager.explore_career_opportunities(character.name)
        career_impact = graph_manager.analyze_career_impact(character.name, goal)
        return (
            0.4 * character.skills
            + 0.4 * job_opportunities
            + 0.2 * motives.job_performance_motive.value
        )

    def calculate_personal_development_importance(
        self, character, goal, graph_manager, motives
    ):
        # Calculate the importance of personal development goals
        current_skill_level = character.skills.get(goal.skill, 0)
        potential_benefits = graph_manager.calculate_potential_utility_of_plan(
            character.name, goal
        )
        return 0.5 * current_skill_level + 0.5 * motives.happiness_motive.value

    def calculate_economic_goal_importance(
        self, financial_status, goal, graph_manager, motives
    ):
        # Calculate the importance of economic goals
        trade_opportunities = graph_manager.evaluate_trade_opportunities_for_item(
            goal.required_resource
        )
        return 0.5 * financial_status + 0.5 * motives.wealth_motive.value

    def calculate_health_goal_importance(self, health, goal, graph_manager, motives):
        # Calculate the importance of health goals
        nearest_health_facility = graph_manager.get_nearest_resource(
            goal.target_location, "health"
        )
        return 0.7 * health + 0.3 * motives.health_motive.value

    def calculate_safety_goal_importance(self, goal, motives):
        # Calculate the importance of safety goals
        return safety_factor * goal.urgency + motives.stability_motive.value

    def calculate_long_term_goal_importance(
        self, character, goal, graph_manager, motives
    ):
        # Calculate the importance of long-term goals
        progress = graph_manager.calculate_goal_progress(character.name, goal)
        return progress * goal.urgency + motives.hope_motive.value

    def calculate_utility(self, action, character):
        # Placeholder for calculating action utility
        return action["satisfaction"] - action["energy_cost"]

    def evaluate_utility(self, plan, character):
        # Placeholder for evaluating the utility of a plan
        return max(plan, key=lambda x: self.calculate_utility(x, character))

    def evaluate_feasibility_of_goal(self, goal, state):
        # Placeholder for evaluating the feasibility of a goal
        return all(goal[k] <= state.get(k, 0) for k in goal)
