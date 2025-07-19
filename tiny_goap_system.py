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

    def __init__(self, name, goals=None, actions=None, graph_manager=None):
        self.name = name
        self.goals = goals if goals is not None else []
        self.action_queue = []
        self.graph_manager = graph_manager
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
        # Add counter to ensure unique comparison for items with same priority
        heappush(self.action_queue, (priority, id(action), action, dependencies))

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
        Replans the sequence of actions based on the current game state with cost-aware prioritization.
        """
        print("Replanning based on new game state...")

        # Clear current action queue to rebuild with updated priorities
        old_queue = list(self.action_queue)
        self.action_queue = []

        # Re-evaluate and re-prioritize actions based on current goals and costs
        for priority, action_id, action, dependencies in old_queue:
            # Skip actions that are already completed
            if action.name in self.completed_actions:
                continue

            # Calculate new priority using utility functions
            new_priority = self._calculate_action_priority(action, priority)

            # Re-add action with updated priority
            heappush(self.action_queue, (new_priority, id(action), action, dependencies))

    def _calculate_action_priority(self, action, old_priority):
        """
        Calculate action priority using cost and utility functions.
        Lower priority number = higher actual priority in heapq.
        
        Args:
            action: Action to prioritize
            old_priority: Previous priority value
            
        Returns:
            float: New priority value (lower is better)
        """
        try:
            # Import utility functions
            import tiny_utility_functions as util_funcs
            
            # Get character state if available
            character_state = {}
            current_goal = None
            
            if self.goals and self.current_goal_index < len(self.goals):
                current_goal = self.goals[self.current_goal_index]
                
            # Use utility evaluation to determine priority
            if hasattr(action, 'cost'):
                base_cost = action.cost
            else:
                base_cost = 1.0
                
            # Calculate utility for this action
            utility = 0.0
            if character_state:  # Only if we have state information
                utility = util_funcs.calculate_action_utility(character_state, action, current_goal)
            
            # Convert to priority (lower utility = higher priority cost)
            # Combine base cost with utility consideration
            priority = base_cost - (utility * 0.05)  # Small utility influence
            
            # Adjust priority based on action urgency and current goal progress
            if hasattr(action, "urgency"):
                priority = priority / max(action.urgency, 0.1)
                
            return max(0.1, priority)  # Ensure positive priority
            
        except Exception as e:
            print(f"Warning: Error calculating action priority: {e}")
            # Fallback to basic cost-based priority
            if hasattr(action, "cost") and hasattr(action, "urgency"):
                return action.cost / max(action.urgency, 0.1)
            else:
                return old_priority

    def execute(self):
        while self.current_goal_index < len(self.goals):
            self.replan()
            current_goal = self.goals[self.current_goal_index]
            if not current_goal.check_completion():
                while self.action_queue:
                    priority, action_id, current_action, dependencies = heappop(self.action_queue)
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
                graph_manager=self.graph_manager,
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
        try:
            GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        except ImportError:
            # Graph manager is optional for core GOAP functionality
            GraphManager = None
            
        self.graph_manager = graph_manager
        self.plans = {}

    def plan_actions(self, character, goal, current_state, actions):
        """
        Generates a plan of actions to achieve a goal from the current state using cost-aware GOAP algorithm.

        Args:
            character: The character for whom the plan is being generated.
            goal (Goal): The goal object with completion conditions.
            current_state (State): The current state of the character.
            actions (list): A list of Action objects.

        Returns:
            list: A list of Action objects that form a plan to achieve the goal, or None if no plan found.
        """
        import heapq
        
        # Initialize the open list as a priority queue with (cost, counter, state, plan)
        # Counter ensures unique comparison for items with same cost
        open_list = [(0.0, 0, current_state, [])]
        visited_states = set()
        counter = 1  # Unique counter for each entry
        
        while open_list:
            current_cost, _, state, current_plan = heapq.heappop(open_list)

            # Check if the current state satisfies the goal
            if self._goal_satisfied(goal, state):
                return current_plan

            # Convert state to hashable format for visited check
            state_hash = self._hash_state(state)
            if state_hash in visited_states:
                continue
            visited_states.add(state_hash)

            for action in actions:
                # Check if action preconditions are met
                if self._action_applicable(action, state):
                    # Create new state by applying action effects
                    new_state = self._apply_action_effects(action, state)
                    new_plan = current_plan + [action]
                    
                    # Calculate cost using utility functions
                    action_cost = self._calculate_action_cost(action, state, character, goal)
                    total_cost = current_cost + action_cost
                    
                    # Add heuristic cost estimate to goal
                    heuristic_cost = self._estimate_cost_to_goal(new_state, goal, character)
                    priority_cost = total_cost + heuristic_cost
                    
                    heapq.heappush(open_list, (priority_cost, counter, new_state, new_plan))
                    counter += 1

        return None  # If no plan is found

    def _goal_satisfied(self, goal, state):
        """Check if the goal is satisfied in the given state."""
        try:
            if hasattr(goal, 'check_completion'):
                return goal.check_completion(state)
            elif hasattr(goal, 'completion_conditions'):
                # Handle different types of completion conditions
                conditions = goal.completion_conditions
                if isinstance(conditions, dict):
                    return all(state.get(k, 0) >= v for k, v in conditions.items())
                elif isinstance(conditions, list):
                    return all(condition(state) if callable(condition) else True for condition in conditions)
            elif hasattr(goal, 'target_effects'):
                # Handle target_effects format - goal is satisfied when state reaches target values
                target_effects = goal.target_effects
                if isinstance(target_effects, dict):
                    for attribute, target_value in target_effects.items():
                        current_value = state.get(attribute, 0)
                        # Check if current value has reached the target
                        # The goal specifies the desired final value, not a change
                        if abs(current_value - target_value) > 0.1:  # Allow small tolerance
                            return False
                    return True
            return False
        except Exception as e:
            print(f"Warning: Error checking goal completion: {e}")
            return False

    def _action_applicable(self, action, state):
        """Check if an action's preconditions are met in the given state."""
        try:
            if hasattr(action, 'preconditions_met'):
                return action.preconditions_met()
            elif hasattr(action, 'preconditions'):
                # Handle different precondition formats
                preconditions = action.preconditions
                if isinstance(preconditions, list):
                    return all(
                        precond.check_condition(state) if hasattr(precond, 'check_condition')
                        else True for precond in preconditions
                    )
                elif isinstance(preconditions, dict):
                    return all(state.get(k, 0) >= v for k, v in preconditions.items())
            return True  # No preconditions means always applicable
        except Exception as e:
            print(f"Warning: Error checking action preconditions for {action.name}: {e}")
            return False

    def _apply_action_effects(self, action, state):
        """Apply action effects to create a new state."""
        try:
            # Create a copy of the current state
            new_state = State(state.dict_or_obj.copy() if hasattr(state, 'dict_or_obj') else state.copy())
            
            if hasattr(action, 'effects'):
                for effect in action.effects:
                    if isinstance(effect, dict):
                        attribute = effect.get('attribute')
                        change_value = effect.get('change_value', 0)
                        
                        if attribute:
                            current_value = new_state.get(attribute, 0)
                            if isinstance(change_value, (int, float)):
                                new_state[attribute] = current_value + change_value
                            elif isinstance(change_value, str) and change_value.startswith('set:'):
                                new_state[attribute] = change_value[4:]  # Remove 'set:' prefix
                            else:
                                new_state[attribute] = change_value
            
            return new_state
        except Exception as e:
            print(f"Warning: Error applying action effects for {action.name}: {e}")
            return state  # Return original state if effects can't be applied

    def _calculate_action_cost(self, action, state, character, goal):
        """
        Calculate the cost of an action using utility functions from tiny_utility_functions.
        Lower utility = higher cost (inverse relationship).
        
        Args:
            action: Action to evaluate
            state: Current state
            character: Character performing the action
            goal: Current goal (optional)
            
        Returns:
            float: Action cost (higher is worse)
        """
        try:
            # Import utility functions
            import tiny_utility_functions as util_funcs
            
            # Convert state to dictionary format if needed
            if hasattr(state, 'dict_or_obj'):
                state_dict = state.dict_or_obj if isinstance(state.dict_or_obj, dict) else {}
            elif hasattr(state, '__dict__'):
                state_dict = state.__dict__
            else:
                state_dict = {}
                
            # Create a goal object compatible with utility functions
            goal_obj = None
            if goal:
                if hasattr(goal, 'target_effects'):
                    goal_obj = goal
                else:
                    # Create a minimal goal object
                    goal_obj = util_funcs.Goal(
                        name=getattr(goal, 'name', 'UnknownGoal'),
                        target_effects=getattr(goal, 'target_effects', {}),
                        priority=getattr(goal, 'priority', 0.5)
                    )
            
            # Calculate utility using existing function
            utility = util_funcs.calculate_action_utility(state_dict, action, goal_obj)
            
            # Convert utility to cost (inverse relationship)
            # Add base cost to prevent negative costs
            base_cost = getattr(action, 'cost', 1.0)
            cost = base_cost - (utility * 0.1)  # Scale utility impact
            
            return max(0.1, cost)  # Ensure positive cost
            
        except Exception as e:
            # Fallback to simple cost if utility calculation fails
            return getattr(action, 'cost', 1.0)
    
    def _estimate_cost_to_goal(self, state, goal, character):
        """
        Estimate remaining cost to reach goal (heuristic for A* search).
        
        Args:
            state: Current state
            goal: Target goal
            character: Character
            
        Returns:
            float: Estimated cost to goal
        """
        try:
            if not goal or not hasattr(goal, 'target_effects') or not goal.target_effects:
                return 0.0
                
            # Convert state to dictionary format
            if hasattr(state, 'dict_or_obj'):
                state_dict = state.dict_or_obj if isinstance(state.dict_or_obj, dict) else {}
            elif hasattr(state, '__dict__'):
                state_dict = state.__dict__
            else:
                state_dict = {}
            
            # Calculate distance to goal based on target effects
            total_distance = 0.0
            for attribute, target_value in goal.target_effects.items():
                current_value = state_dict.get(attribute, 0.0)
                distance = abs(target_value - current_value)
                total_distance += distance
                
            # Simple heuristic: assume average action cost * rough number of actions needed
            if total_distance <= 0.1:  # Very close to goal
                return 0.0
                
            estimated_actions = max(1.0, total_distance / 0.5)  # Assume actions change ~0.5 per attribute
            average_action_cost = 1.0  # Conservative estimate
            
            return estimated_actions * average_action_cost
            
        except Exception:
            return 0.0  # Conservative fallback

    def _hash_state(self, state):
        """Convert state to a hashable format for visited state tracking."""
        try:
            if hasattr(state, '__hash__'):
                return hash(state)
            elif hasattr(state, 'dict_or_obj'):
                # For State objects, hash the underlying dict/object
                state_data = state.dict_or_obj
                if isinstance(state_data, dict):
                    return hash(tuple(sorted(state_data.items())))
                else:
                    return hash(str(state_data))  # Fallback to string representation
            else:
                return hash(str(state))  # Fallback to string representation
        except Exception as e:
            print(f"Warning: Error hashing state: {e}")
            return hash(str(state))  # Ultimate fallback

    @staticmethod
    def goap_planner(character, goal: Goal, char_state: State, actions: list):
        """
        Static method that delegates to instance method for backwards compatibility.
        
        Args:
            character: The character for whom the plan is being generated.
            goal (Goal): The goal object with completion conditions.
            char_state (State): The current state of the character.
            actions (list): A list of Action objects.

        Returns:
            list: A list of Action objects that form a plan to achieve the goal.
        """
        # Create a temporary planner instance with None graph_manager
        planner = GOAPPlanner(None)
        return planner.plan_actions(character, goal, char_state, actions)

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
        """Calculate action utility based on multiple factors using integrated utility functions"""
        try:
            # Import utility functions
            import tiny_utility_functions as util_funcs
            
            # Get character state
            character_state = {}
            if hasattr(character, "get_state"):
                char_state_obj = character.get_state()
                if hasattr(char_state_obj, 'dict_or_obj'):
                    character_state = char_state_obj.dict_or_obj if isinstance(char_state_obj.dict_or_obj, dict) else {}
                elif hasattr(char_state_obj, '__dict__'):
                    character_state = char_state_obj.__dict__
            elif hasattr(character, '__dict__'):
                character_state = character.__dict__
            
            # Get current goal if available
            current_goal = None
            if self.goals and self.current_goal_index < len(self.goals):
                current_goal = self.goals[self.current_goal_index]
                
            # Use the comprehensive utility calculation
            if character_state:
                utility = util_funcs.calculate_action_utility(character_state, action, current_goal)
                return utility
            
        except Exception as e:
            print(f"Warning: Error in utility calculation, falling back to basic method: {e}")
        
        # Fallback to basic utility calculation
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
            try:
                char_state = character.get_state()
                # Adjust utility based on character's current needs
                if hasattr(char_state, 'dict_or_obj') and isinstance(char_state.dict_or_obj, dict):
                    state_dict = char_state.dict_or_obj
                    if "energy" in state_dict and energy_cost > 0:
                        energy_factor = state_dict["energy"] / 100.0  # Normalize to 0-1
                        base_utility *= energy_factor
            except Exception:
                pass  # Continue with base utility if state access fails

        # Apply urgency multiplier
        final_utility = base_utility * urgency

        return final_utility

    def evaluate_utility(self, plan, character):
        """Evaluate the utility of a plan using integrated utility functions"""
        if not plan:
            return None

        try:
            # Import utility functions
            import tiny_utility_functions as util_funcs
            
            # Get character state
            character_state = {}
            if hasattr(character, "get_state"):
                char_state_obj = character.get_state()
                if hasattr(char_state_obj, 'dict_or_obj'):
                    character_state = char_state_obj.dict_or_obj if isinstance(char_state_obj.dict_or_obj, dict) else {}
                elif hasattr(char_state_obj, '__dict__'):
                    character_state = char_state_obj.__dict__
            elif hasattr(character, '__dict__'):
                character_state = character.__dict__
                
            # Get current goal if available
            current_goal = None
            if self.goals and self.current_goal_index < len(self.goals):
                current_goal = self.goals[self.current_goal_index]
            
            # Use comprehensive plan utility calculation if we have character state
            if character_state:
                plan_utility = util_funcs.calculate_plan_utility(
                    character_state, plan, current_goal, simulate_effects=True
                )
                
                # Return the action with highest individual utility within the plan
                best_action = None
                best_utility = float('-inf')
                
                for action in plan:
                    action_utility = util_funcs.calculate_action_utility(character_state, action, current_goal)
                    if action_utility > best_utility:
                        best_utility = action_utility
                        best_action = action
                        
                return best_action if best_action else plan[0]
                
        except Exception as e:
            print(f"Warning: Error in plan utility evaluation, falling back to basic method: {e}")

        # Fallback to basic utility evaluation
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
