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

# Constants for utility calculations
UTILITY_SCALING_FACTOR = 0.1
UTILITY_INFLUENCE_FACTOR = 0.05
HEURISTIC_SCALING_FACTOR = 0.1


class ActionWrapper:
    """Wrapper class to convert dictionary action data to Action-like objects."""
    def __init__(self, name, cost=1.0, effects=None, preconditions=None, utility=0.0):
        self.name = name
        self.cost = cost
        self.effects = effects if effects is not None else []
        self.preconditions = preconditions if preconditions is not None else {}
        self.utility = utility
        
    def preconditions_met(self):
        """Simple precondition check - can be enhanced based on actual requirements."""
        return True
        
    def execute(self, target=None, initiator=None):
        """Simple execution - returns True for basic compatibility."""
        return True


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
    
    # Constants for utility calculations
    UTILITY_INFLUENCE_FACTOR = 0.05

    def __init__(self, name, goals=None, actions=None, graph_manager=None):
        self.name = name
        self.goals = goals if goals is not None else []
        self.action_queue = []
        self.graph_manager = graph_manager
        self.current_goal_index = 0
        self.completed_actions = set()
        self.current_action_index = 0
        self._action_counter = 0
        self._failure_count = {}
        self._max_retries = 3
        self._replan_count = 0
        self._max_replans = 5

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

        # Add counter to ensure unique ordering when priorities are equal
        counter = getattr(self, '_action_counter', 0)
        self._action_counter = counter + 1
        heappush(self.action_queue, (priority, counter, action, dependencies))
 

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

        Replans the sequence of actions based on the current game state.
        Now generates alternative action sequences instead of just re-sorting.
 
        """
        self._replan_count += 1
        if self._replan_count > self._max_replans:
            print(f"Maximum replanning attempts ({self._max_replans}) reached for plan {self.name}")
            return False
            
        print(f"Replanning based on new game state... (attempt {self._replan_count})")

        # Clear current action queue to rebuild with updated priorities
        old_queue = list(self.action_queue)
        self.action_queue = []
        self._action_counter = 0

        # If we have a graph manager and goals, try to generate new action sequences
        if self.graph_manager and self.goals:
            try:
                current_goal = self.goals[self.current_goal_index] if self.current_goal_index < len(self.goals) else None
                if current_goal and hasattr(self.graph_manager, 'plan_actions'):
                    # Try to get new action plan from GOAP planner
                    from actions import State
                    # Construct a meaningful current state based on game data
                    current_state_data = {
                        "character_health": self.graph_manager.character.health,
                        "character_energy": self.graph_manager.character.energy,
                        "environment_conditions": self.graph_manager.get_environment_conditions(),
                        "goal_progress": {goal.name: goal.progress for goal in self.goals}
                    }
                    current_state = State(current_state_data)
                    available_actions = [action for _, _, action, _ in old_queue if action.name not in self.completed_actions]
                    
                    new_plan = self.graph_manager.plan_actions(None, current_goal, current_state, available_actions)
                    if new_plan:
                        print("Generated new action sequence from GOAP planner")
                        for i, action in enumerate(new_plan):
                            # Higher priority (lower number) for earlier actions in plan
                            self.add_action(action, priority=i+1)
                        return True
            except Exception as e:
                print(f"Failed to generate new action sequence: {e}")


        # Fallback: Re-evaluate and re-prioritize existing actions
        failed_actions = set()
        for priority, counter, action, dependencies in old_queue:
 
            # Skip actions that are already completed
            if action.name in self.completed_actions:
                continue
                
            # Track failed actions to deprioritize them
            failure_count = self._failure_count.get(action.name, 0)
            if failure_count >= self._max_retries:
                failed_actions.add(action.name)
                continue

 
            # Calculate new priority using utility functions
            new_priority = self._calculate_action_priority(action, priority)

            # Re-calculate priority based on current state
           # new_priority = priority
            
            # Penalize actions that have failed before
            if failure_count > 0:
                new_priority += failure_count * 2  # Increase priority number (lower priority)
 

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
            priority = base_cost - (utility * self.UTILITY_INFLUENCE_FACTOR)  # Small utility influence
            
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

                new_priority = (action.cost / max(action.urgency, 0.1)) + (failure_count * 2)

            # Re-add action with updated priority
            self.add_action(action, new_priority, dependencies)
            
        if failed_actions:
            print(f"Skipped permanently failed actions: {failed_actions}")
            
        return True
 

    def execute(self):
        """
        Execute the plan with enhanced robustness and retry mechanisms.
        """
        plan_start_time = getattr(self, '_start_time', 0)
        max_execution_time = 300  # 5 minutes max
        
        while self.current_goal_index < len(self.goals):
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

            
            # Check if goal is already completed
            if current_goal.check_completion():
                print(f"Goal {current_goal.name if hasattr(current_goal, 'name') else 'unnamed'} already completed")
 
                self.current_goal_index += 1
                continue
            
            # Only replan if we have no actions or after failures
            if not self.action_queue or self._replan_count == 0:
                replan_success = self.replan()
                if not replan_success:
                    print(f"Failed to replan for goal {current_goal.name if hasattr(current_goal, 'name') else 'unnamed'}")
                    return False
            
            # Try to execute actions for current goal
            goal_achieved = False
            actions_attempted = 0
            max_actions_per_goal = 10
            
            while self.action_queue and not goal_achieved and actions_attempted < max_actions_per_goal:
                priority, counter, current_action, dependencies = heappop(self.action_queue)
                actions_attempted += 1
                
                # Check dependencies
                if not all(dep in self.completed_actions for dep in dependencies):
                    print(f"Action {current_action.name} dependencies not met: {dependencies}")
                    continue
                
                # Check preconditions
                if not current_action.preconditions_met():
                    print(f"Action {current_action.name} preconditions not met")
                    continue
                
                # Execute the action with retry logic
                success = self._execute_action_with_retry(current_action)
                
                if success:
                    self.completed_actions.add(current_action.name)
                    print(f"Action {current_action.name} completed successfully")
                    
                    # Check if goal is now completed
                    if current_goal.check_completion():
                        goal_achieved = True
                        print(f"Goal achieved after completing action {current_action.name}")
                        break
                else:
                    # Handle action failure
                    if not self._handle_action_failure(current_action):
                        print(f"Failed to handle failure of action {current_action.name}")
                        return False
            
            if goal_achieved or current_goal.check_completion():
                self.current_goal_index += 1
                self._replan_count = 0  # Reset replan count for next goal
            else:
                print(f"Could not achieve goal {current_goal.name if hasattr(current_goal, 'name') else 'unnamed'}")
                return False
                
        self.handle_success()
        return True
        
    def _execute_action_with_retry(self, action):
        """
        Execute an action with retry logic and exponential backoff.
        """
        max_retries = 3
        base_delay = 0.1  # Base delay in seconds
        
        for attempt in range(max_retries + 1):
            try:
                print(f"Executing action {action.name} (attempt {attempt + 1})")
                # Try different execute method signatures
                try:
                    success = action.execute()
                except TypeError:
                    # Fallback to execute with parameters
                    success = action.execute(
                        target=getattr(action, 'target', None),
                        initiator=getattr(action, 'initiator', None),
                    )
                
                if success:
                    # Reset failure count on success
                    if action.name in self._failure_count:
                        del self._failure_count[action.name]
                    return True
                else:
                    print(f"Action {action.name} returned failure")
                    
            except Exception as e:
                print(f"Action {action.name} raised exception: {e}")
            
            # Track failure
            self._failure_count[action.name] = self._failure_count.get(action.name, 0) + 1
            
            # If not the last attempt, wait before retrying
            if attempt < max_retries:
                import time
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying action {action.name} in {delay:.2f} seconds...")
                time.sleep(delay)
        
        print(f"Action {action.name} failed after {max_retries + 1} attempts")
        return False
        
    def _handle_action_failure(self, action):
        """
        Handle action failure with intelligent recovery strategies.
        """
        failure_count = self._failure_count.get(action.name, 0)
        print(f"Handling failure of action {action.name} (failure count: {failure_count})")
        
        # If action has failed too many times, try to find alternative
        if failure_count >= self._max_retries:
            print(f"Action {action.name} has exceeded maximum retries, seeking alternative")
            alternative = self.find_alternative_action(action)
            if alternative:
                print(f"Found alternative action: {alternative['action'].name}")
                self.add_action(
                    alternative["action"],
                    alternative["priority"],
                    alternative["dependencies"],
                )
                return True
            else:
                print(f"No alternative found for action {action.name}")
                return False
        
        # Try replanning if we haven't exceeded replan limit
        if self._replan_count < self._max_replans:
            return self.replan()
        else:
            print(f"Maximum replanning attempts reached")
            return False

    def handle_failure(self, action):
        """
        Handles the failure of an action within the plan.
        This method is now mainly used for backward compatibility.
        The main failure handling is done in _handle_action_failure.
        """
        print(f"Action {action.name} failed. Re-evaluating plan.")
        return self._handle_action_failure(action)

    def handle_success(self):
        """
        Handles the successful completion of the plan.
        """
        print(f"Plan {self.name} successfully completed.")
        print(f"Total completed actions: {len(self.completed_actions)}")
        print(f"Total replanning attempts: {self._replan_count}")

    def find_alternative_action(self, failed_action):
        """
        Logic to find an alternative action for the failed action.
        Enhanced to use graph manager when available and prevent infinite loops.
        """
        print(f"Finding alternative action for {failed_action.name}")

        # Prevent infinite alternative creation
        if failed_action.name.count('alt_') >= 3:
            print(f"Too many alternative attempts for {failed_action.name}, stopping")
            return None

        # First try to use graph manager for intelligent alternatives
        if self.graph_manager and hasattr(self.graph_manager, 'find_alternative_actions'):
            try:
                alternatives = self.graph_manager.find_alternative_actions(failed_action)
                if alternatives:
                    print(f"Graph manager found {len(alternatives)} alternatives")
                    return alternatives[0]  # Return the best alternative
            except Exception as e:
                print(f"Graph manager alternative search failed: {e}")

        # Fallback: Create a mock alternative that just succeeds
        # This is a simple fallback for testing - in real implementation,
        # this would use more sophisticated alternative action generation
        alternative_name = f"alt_{failed_action.name}"

        try:
            # Create a simple mock alternative that succeeds
            class MockAlternativeAction:
                def __init__(self, name, original_action):
                    self.name = name
                    self.target = getattr(original_action, "target", None)
                    self.initiator = getattr(original_action, "initiator", None)
                    self.cost = getattr(original_action, "cost", 5) + 2
                    self.urgency = getattr(original_action, "urgency", 1)
                    self.effects = getattr(original_action, "effects", {})
                    self.related_skills = getattr(original_action, "related_skills", [])
                
                def preconditions_met(self):
                    return True
                
                def execute(self, target=None, initiator=None):
                    import random
                    success_probability = 0.8  # 80% chance of success
                    if random.random() < success_probability:
                        print(f"Mock alternative action {self.name} executed successfully")
                        return True
                    else:
                        print(f"Mock alternative action {self.name} failed")
                        return False

            alternative_action = MockAlternativeAction(alternative_name, failed_action)

            return {
                "action": alternative_action,
                "priority": 10,  # Lower priority than original (higher number)
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
    # Maximum length of plans to prevent overly complex planning
    MAX_PLAN_LENGTH = 10
    
    def __init__(self, graph_manager: "GraphManager"):
        try:
            GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        except ImportError:
            # Graph manager is optional for core GOAP functionality
            GraphManager = None
            
        self.graph_manager = graph_manager
        self.plans = {}

    def get_current_world_state(self, character):
        """
        Dynamically retrieves the current world state for a character.
        
        Args:
            character: The character object to get state for
            
        Returns:
            State: Current world state including character state and environment
        """
        try:
            # Get the character's current state
            character_state = character.get_state() if hasattr(character, 'get_state') else State({})
            
            # If we have a graph manager, enrich the state with world information
            if self.graph_manager:
                try:
                    # Get additional world context from graph manager
                    world_context = self.graph_manager.get_character_state(character.name)
                    
                    # Merge character state with world context
                    if hasattr(character_state, 'dict_or_obj'):
                        if isinstance(character_state.dict_or_obj, dict):
                            combined_state = character_state.dict_or_obj.copy()
                        else:
                            combined_state = {'character': character_state.dict_or_obj}
                    else:
                        combined_state = {'character': character}
                    
                    # Add world context
                    if isinstance(world_context, dict):
                        combined_state.update(world_context)
                    
                    return State(combined_state)
                except Exception as e:
                    logging.warning(f"Could not get world context from graph manager: {e}")
                    
            return character_state
            
        except Exception as e:
            logging.warning(f"Error getting current world state: {e}")
            # Return a minimal state as fallback
            return State({'character': character})

    def get_available_actions(self, character):
        """
        Dynamically retrieves available actions for a character based on current state.
        
        Args:
            character: The character object to get actions for
            
        Returns:
            list: List of Action objects available to the character
        """
        available_actions = []
        
        try:
            # Try to get actions from graph manager first (most comprehensive)
            if self.graph_manager and hasattr(self.graph_manager, 'get_possible_actions'):
                try:
                    graph_actions = self.graph_manager.get_possible_actions(character.name)
                    # Convert graph manager action format to Action objects if needed
                    for action_data in graph_actions:
                        if isinstance(action_data, dict):
                            # Convert dict format to action-like object
                            action_obj = ActionWrapper(
                                name=action_data.get('name', 'Unknown'),
                                cost=action_data.get('cost', 1),
                                effects=action_data.get('effects', []),
                                preconditions=action_data.get('preconditions', {}),
                                utility=action_data.get('utility', 0)
                            )
                            available_actions.append(action_obj)
                        else:
                            available_actions.append(action_data)
                except Exception as e:
                    logging.warning(f"Could not get actions from graph manager: {e}")
            
            # Try to get actions from strategy manager if available via graph manager
            try:
                # Strategy manager should be accessed externally, not directly from GOAP planner
                # For now, skip strategy manager actions since GOAP planner should be independent
                pass
            except Exception as e:
                logging.warning(f"Could not get actions from strategy manager: {e}")
            
            # Add basic fallback actions if no actions were found
            if not available_actions:
                # Import Action class for fallback actions
                try:
                    Action = importlib.import_module("actions").Action
                    
                    # Create basic actions that any character can perform
                    rest_action = Action(
                        name="Rest",
                        preconditions={},
                        effects=[{"attribute": "energy", "change_value": 10}],
                        cost=0
                    )
                    available_actions.append(rest_action)
                    
                    idle_action = Action(
                        name="Idle", 
                        preconditions={},
                        effects=[],
                        cost=1
                    )
                    available_actions.append(idle_action)
                    
                except Exception as e:
                    logging.warning(f"Could not create fallback actions: {e}")
                    
        except Exception as e:
            logging.error(f"Error getting available actions: {e}")
            
        return available_actions

    def plan_actions(self, character, goal, current_state=None, actions=None):
        """

        Generates a plan of actions to achieve a goal from the current state using GOAP algorithm.
        Now dynamically retrieves current state and available actions if not provided.
 

        Args:
            character: The character for whom the plan is being generated.
            goal (Goal): The goal object with completion conditions.
            current_state (State, optional): The current state. If None, will be retrieved dynamically.
            actions (list, optional): A list of Action objects. If None, will be retrieved dynamically.

        Returns:
            list: A list of Action objects that form a plan to achieve the goal, or None if no plan found.
        """
 
        import heapq
        
        # Dynamically retrieve current world state if not provided
        if current_state is None:
            current_state = self.get_current_world_state(character)
            logging.info(f"Dynamically retrieved world state for {getattr(character, 'name', 'character')}")
        
        # Dynamically retrieve available actions if not provided
        if actions is None:
            actions = self.get_available_actions(character)
            logging.info(f"Dynamically retrieved {len(actions)} available actions for {getattr(character, 'name', 'character')}")
        
        # Initialize the open list as a priority queue with (cost, counter, state, plan)
        # Counter ensures unique comparison for items with same cost
        open_list = [(0.0, 0, current_state, [])]
        visited_states = set()
        counter = 1  # Unique counter for each entry
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0

        while open_list and iteration_count < max_iterations:
            iteration_count += 1
            current_cost, _, state, current_plan = heapq.heappop(open_list)
 

            # Check if the current state satisfies the goal
            if self._goal_satisfied(goal, state):
                logging.info(f"Goal achieved with plan of {len(current_plan)} actions")
                return current_plan


        
            # Convert state to hashable format for visited check
            state_hash = self._hash_state(state)
            if state_hash in visited_states:
                continue
            visited_states.add(state_hash)

            # Limit plan length to prevent overly complex plans
            if len(current_plan) >= self.MAX_PLAN_LENGTH:
                continue
 

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

 
        logging.warning(f"No plan found to achieve goal for {getattr(character, 'name', 'character')}")

        # If we exceeded max iterations or couldn't find a plan
        if iteration_count >= max_iterations:
            print(f"Warning: GOAP planning exceeded max iterations ({max_iterations})")
        
 
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
                        # Check if current value has reached or exceeded the target
                        # Goals are satisfied when we reach at least the target value
                        if current_value < target_value - 0.1:  # Allow small tolerance for floating point
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
                        targets = effect.get('targets', [])
                        
                        # Apply all effects to the current state during planning
                        # The "targets" field is ignored during planning and is handled during actual execution
                        # This ensures that all potential effects are considered for state transitions
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
            cost = base_cost - (utility * UTILITY_SCALING_FACTOR)  # Scale utility impact
            
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

    def plan_for_character(self, character, goal):
        """
        Convenience method that automatically retrieves current world state and available actions
        for a character and generates a plan to achieve the specified goal.
        
        Args:
            character: The character for whom the plan is being generated.
            goal (Goal): The goal object with completion conditions.
            
        Returns:
            list: A list of Action objects that form a plan to achieve the goal, or None if no plan found.
        """
        return self.plan_actions(character, goal)

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
            
            # Get current goal if available - GOAPPlanner doesn't manage goals directly
            current_goal = None
            # Goals should be passed in as parameters, not stored in planner
                
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

        # Get character state properly instead of using empty dictionary
        character_state = {}
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

            char_state_obj = character.get_state()
            if hasattr(char_state_obj, 'dict_or_obj'):
                character_state = char_state_obj.dict_or_obj if isinstance(char_state_obj.dict_or_obj, dict) else {}
            elif hasattr(char_state_obj, '__getitem__'):
                # Handle State object with dictionary-like access
                try:
                    character_state = {"energy": char_state_obj.get("energy", 100)}
                except:
                    character_state = {}
        
        # Adjust utility based on character's current needs using the populated character_state
        if character_state and energy_cost > 0:
            energy = character_state.get("energy", 100)
            if energy > 0:
                energy_factor = energy / 100.0  # Normalize to 0-1
                base_utility *= energy_factor
            
            # Consider other character state factors
            health = character_state.get("health", 100)
            if health < 50:  # Low health reduces utility for high-cost actions
                health_factor = health / 100.0
                base_utility *= health_factor
 


        # Apply urgency multiplier
        final_utility = base_utility * urgency

        return final_utility

    def evaluate_utility(self, plan, character):

        """
        Evaluate the utility of a plan by finding the action with highest utility.
        Fixed to properly use character state instead of empty dictionary.
        """
 
        if not plan:
            return None

        # Get character state properly instead of using empty dictionary
        character_state = {}
        if hasattr(character, "get_state"):
            char_state_obj = character.get_state()
            if hasattr(char_state_obj, 'dict_or_obj'):
                character_state = char_state_obj.dict_or_obj if isinstance(char_state_obj.dict_or_obj, dict) else {}

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
                
            # Get current goal if available - should be passed as parameter
            current_goal = None
            # Goals should be passed in as parameters, not stored in planner
            
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



    def _calculate_action_priority(self, action, character):
        """
        Calculate action priority based on cost and utility.
        Replaces magic number 0.05 with UTILITY_INFLUENCE_FACTOR constant.
        Uses character state properly instead of empty dictionary.
        """
        # Get character state instead of using empty dictionary
        character_state = {}
        if hasattr(character, 'get_state'):
            char_state_obj = character.get_state()
            if hasattr(char_state_obj, 'dict_or_obj'):
                character_state = char_state_obj.dict_or_obj if isinstance(char_state_obj.dict_or_obj, dict) else {}
        
        cost = self._calculate_action_cost(action, character)
        utility = self.calculate_utility(action, character)
        
        # Calculate priority with utility influence
        # Lower priority number means higher priority
        base_priority = cost
        utility_adjustment = utility * self.UTILITY_INFLUENCE_FACTOR
        
        # Higher utility should result in lower priority number (higher actual priority)
        priority = base_priority - utility_adjustment
        
        return max(priority, 0.1)  # Ensure priority is not negative


