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

from tiny_characters import Goal
from actions import Action, State
from tiny_graph_manager import GraphManager

from heapq import heappush, heappop


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
            ),
            "priority": 1,
            "dependencies": [],
        }


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
    def __init__(self, graph_manager: GraphManager):
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
        return self.graph_manager.calculate_goal_difficulty(goal, character)

    def calculate_utility(self, action, character):
        # Placeholder for calculating action utility
        return action["satisfaction"] - action["energy_cost"]

    def evaluate_utility(self, plan, character):
        # Placeholder for evaluating the utility of a plan
        return max(plan, key=lambda x: self.calculate_utility(x, character))

    def evaluate_feasibility_of_goal(self, goal, state):
        # Placeholder for evaluating the feasibility of a goal
        return all(goal[k] <= state.get(k, 0) for k in goal)
