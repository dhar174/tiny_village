"""
Depending on the type of planning (e.g., career, daily activities, relationship management), different nodes and edges are highlighted or prioritized during the analysis.
GOAP (Goal-Oriented Action Planning)
Purpose: To create a series of actions that lead to achieving a specific goal.
Implementation Strategy:
Design a GOAP system that operates on a set of possible actions, each with conditions and effects that are defined generically enough to apply to various goals.
Implement action selection based on the current state of the world and the desired end state, using a cost function that might include time, resource expenditure, emotional cost, etc., which can adapt to the type of goal.
Utility Evaluation
Purpose: To assess and rank possible actions based on their expected utility.
Implementation Strategy:
Define a utility function that inputs various factors like character preferences, potential rewards, risks, and long-term impacts.
This function should dynamically weigh these factors according to the current planning focus—whether it's deciding on a daily activity or a life-changing decision like a job change.

This module integrates the GOAP system and the graph manager to formulate comprehensive strategies based on events.
"""

from tiny_goap_system import GOAPPlanner
from tiny_utility_functions import (
    calculate_action_utility,
    Goal,
)  # evaluate_utility seems to be for plans
from actions import Action, State  # Use the modified Action from actions.py
import logging

# Optional imports to avoid dependency issues
try:
    from tiny_characters import Character
except ImportError:
    Character = None
    logging.warning("Character class not available - using simplified character handling")

try:
    from tiny_graph_manager import GraphManager
except ImportError:
    GraphManager = None
    logging.warning("GraphManager not available - graph functionality will be limited")

try:
    from tiny_prompt_builder import PromptBuilder
    from tiny_brain_io import TinyBrainIO
    from tiny_output_interpreter import OutputInterpreter
except ImportError:
    PromptBuilder = None
    TinyBrainIO = None
    OutputInterpreter = None
    logging.warning("LLM components not available - LLM functionality will be disabled")

logger = logging.getLogger(__name__)

# Constants for goal targets
SATISFACTION_TARGET = 70
ENERGY_TARGET = 65


# Define placeholder/simplified Action classes for use within StrategyManager if not using complex ones from actions.py
# These are structured to be compatible with calculate_action_utility


class EatAction(Action):
    def __init__(self, name="Eat", cost=0.1, effects=None, item_name="food"):
        super().__init__(
            name=f"{name} {item_name}",
            preconditions={},
            effects=effects if effects else [],
            cost=cost,
        )
        self.item_name = item_name


class SleepAction(Action):
    def __init__(
        self, name="Sleep", cost=0.0, effects=None
    ):  # Sleeping might have time cost, but 0 for this model
        super().__init__(
            name=name, preconditions={}, effects=effects if effects else [], cost=cost
        )


class WorkAction(Action):
    def __init__(self, name="Work", cost=0.3, effects=None):
        super().__init__(
            name=name, preconditions={}, effects=effects if effects else [], cost=cost
        )


class NoOpAction(Action):
    def __init__(self, name="NoOp", cost=0.0, effects=None):
        super().__init__(
            name=name, preconditions={}, effects=effects if effects else [], cost=cost
        )


class WanderAction(Action):  # Example generic action
    def __init__(self, name="Wander", cost=0.1, effects=None):
        super().__init__(
            name=name, preconditions={}, effects=effects if effects else [], cost=cost
        )


class StrategyManager:
    """
    The StrategyManager class is responsible for managing the strategies used in goal-oriented action planning and utility evaluation.
    """


    def __init__(self, use_llm=False, model_name=None):
        self.goap_planner = GOAPPlanner(GraphManager() if GraphManager else None)
        # Initialize graph_manager if available
        if GraphManager:
            self.graph_manager = GraphManager()
        else:
            self.graph_manager = None
 
        self.use_llm = use_llm

        # Initialize LLM components if needed and available
        if self.use_llm and TinyBrainIO and PromptBuilder and OutputInterpreter:
            self.brain_io = TinyBrainIO(
                model_name or "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"
            )
            self.output_interpreter = OutputInterpreter()
        else:
            self.brain_io = None
            self.output_interpreter = None
            if self.use_llm:
                logging.warning("LLM components not available - disabling LLM functionality")

    def get_character_state_dict(self, character) -> dict:
        """
        Extracts a simplified dictionary representation of the character's state
        relevant for utility calculations.
        
        Args:
            character: Character object or dictionary with character state
        """
        state = {}
        
        # Handle different character input types
        if isinstance(character, dict):
            # If character is already a dictionary, use it directly
            return character
        elif hasattr(character, '__dict__'):
            # If character is an object, extract attributes
            # Basic needs - assuming direct attribute access or simple getters
            # Normalize hunger/energy to 0-1 range if they aren't already.
            # For utility function: higher hunger = more need; lower energy = more need.
            state["hunger"] = (
                getattr(character, "hunger_level", 0.0) / 10.0
                if hasattr(character, "hunger_level")
                else 0.5
            )  # Assuming hunger 0-10
            state["energy"] = (
                getattr(character, "energy", 0.0) / 10.0
                if hasattr(character, "energy")
                else 0.5
            )  # Assuming energy 0-10
            state["money"] = float(getattr(character, "wealth_money", 0))

            # Add other relevant states if needed by utility function's need fulfillment logic
            # e.g., social_wellbeing, mental_health
            state["social_wellbeing"] = (
                getattr(character, "social_wellbeing", 5.0) / 10.0
                if hasattr(character, "social_wellbeing")
                else 0.5
            )
            state["mental_health"] = (
                getattr(character, "mental_health", 5.0) / 10.0
                if hasattr(character, "mental_health")
                else 0.5
            )
        else:
            # Fallback for simple character representation
            state = {
                "hunger": 0.5,
                "energy": 0.5,
                "money": 0.0,
                "social_wellbeing": 0.5,
                "mental_health": 0.5
            }

        return state

    def get_daily_actions(
        self, character, current_goal=None
    ) -> list[Action]:
        """
        Generates a list of potential daily actions for a character,
        calculates their utility, and returns them sorted by utility.
        
        Args:
            character: Character object, dictionary, or string name
            current_goal: Optional goal object for utility calculation
        """
        potential_actions = []

        # --- Action Generation ---
        # 1. Generic Actions
        potential_actions.append(NoOpAction())
        potential_actions.append(
            WanderAction(effects=[{"attribute": "energy", "change_value": -0.05}])
        )  # Wandering costs a little energy

        # 2. Contextual Actions
        # Handle different character input types
        char_energy_normalized = 0.5  # Default value
        char_location_name = "Unknown"  # Default value
        char_job = "unemployed"  # Default value
        
        if hasattr(character, 'energy'):
            char_energy_normalized = getattr(character, "energy", 5.0) / 10.0
        elif isinstance(character, dict):
            char_energy_normalized = character.get("energy", 5.0) / 10.0
            
        if hasattr(character, 'location'):
            char_location_name = getattr(character.location, "name", "Unknown") if character.location else "Unknown"
        elif isinstance(character, dict):
            char_location_name = character.get("location", "Unknown")
            
        if hasattr(character, 'job'):
            char_job = character.job
        elif isinstance(character, dict):
            char_job = character.get("job", "unemployed")

        LOW_ENERGY_THRESHOLD = 0.3  # Assuming energy is normalized 0-1 for this check
        HIGH_HUNGER_THRESHOLD = 0.6  # Assuming hunger is normalized 0-1 for this check

        # Eat Actions (from inventory) - simplified for now
        if hasattr(character, "inventory") and hasattr(character.inventory, "get_food_items"):
            food_items = character.inventory.get_food_items()
            if food_items:
                # Consider eating the first available food item for simplicity
                for food_item in food_items[:2]:  # Limit to checking first 2 food items
                    hunger_reduction_effect = -(
                        getattr(food_item, "calories", 20) * 0.1
                    )  # Scale calories to hunger effect
                    eat_effects = [
                        {"attribute": "hunger", "change_value": hunger_reduction_effect}
                    ]
                    potential_actions.append(
                        EatAction(
                            item_name=food_item.name, effects=eat_effects, cost=0.05
                        )
                    )

        # Sleep Action
        if (char_energy_normalized < LOW_ENERGY_THRESHOLD and char_location_name == "Home"):
            sleep_effects = [
                {"attribute": "energy", "change_value": 0.7}
            ]  # Restore 70% energy
            potential_actions.append(
                SleepAction(effects=sleep_effects, cost=0)
            )  # Sleep itself has no direct cost other than time

        # Work Action
        if char_job and char_job != "unemployed":
            job_name = char_job
            if hasattr(char_job, "job_title"):
                job_name = char_job.job_title
            elif not isinstance(char_job, str):
                job_name = str(char_job)
                
            work_effects = [
                {"attribute": "money", "change_value": 20.0},  # Example money gain
                {"attribute": "energy", "change_value": -0.3},  # Example energy cost
            ]
            potential_actions.append(
                WorkAction(name=f"Work as {job_name}", effects=work_effects, cost=0.2)
            )  # Base cost for work action

        # --- Utility Calculation & Sorting ---
        action_utilities = []
        character_state_dict = self.get_character_state_dict(character)
        if not character_state_dict:  # Basic error handling
            # Return some default if state can't be determined, or raise error
            return sorted(
                potential_actions, key=lambda x: x.name
            )  # Sort by name as fallback

        for action in potential_actions:
            utility = calculate_action_utility(
                character_state_dict, action, current_goal
            )
            action_utilities.append((action, utility))

        # Sort actions by utility in descending order
        sorted_actions = sorted(action_utilities, key=lambda x: x[1], reverse=True)

        return [action_tuple[0] for action_tuple in sorted_actions]

    def decide_action_with_llm(
        self, character, time="morning", weather="clear"
    ) -> list[Action]:
        """
        Use LLM to make an intelligent decision about character actions.
        This integrates the full decision-making pipeline:
        Character Context → PromptBuilder → BrainIO → OutputInterpreter → Actions
        """
        if not self.use_llm or not self.brain_io or not self.output_interpreter:
            logger.warning(
                "LLM decision-making not available, falling back to utility-based actions"
            )
            return self.get_daily_actions(character)

        try:
            # Step 1: Generate potential actions using utility-based system
            potential_actions = self.get_daily_actions(character)

            # Step 2: Create PromptBuilder for this character
            if not PromptBuilder:
                logger.warning("PromptBuilder not available, falling back to utility-based actions")
                return potential_actions[:1] if potential_actions else []
                
            prompt_builder = PromptBuilder(character)

            # Step 3: Generate action choices with utility scores from potential actions
            action_choices = []
            character_state_dict = self.get_character_state_dict(character)

            for i, action in enumerate(potential_actions[:5]):  # Limit to top 5 actions
                action_name = getattr(action, "name", str(action))
                # Calculate utility for this action to show reasoning
                utility_score = calculate_action_utility(
                    character_state_dict,
                    action,
                    (
                        character.get_current_goal()
                        if hasattr(character, "get_current_goal")
                        else None
                    ),
                )

                # Create detailed action choice with utility reasoning
                action_choice = f"{i+1}. {action_name} (Utility: {utility_score:.1f})"

                # Add action effects if available
                if hasattr(action, "effects") and action.effects:
                    effects_str = ", ".join(
                        [
                            f"{eff.get('attribute', '')}: {eff.get('change_value', 0):+.1f}"
                            for eff in action.effects
                            if eff.get("attribute")
                        ]
                    )
                    if effects_str:
                        action_choice += f" - Effects: {effects_str}"

                action_choices.append(action_choice)

            # Step 4: Generate LLM prompt with enhanced action choices and character context
            character_name = getattr(character, 'name', 'Character') if hasattr(character, 'name') else 'Character'
            prompt = prompt_builder.generate_decision_prompt(
                time, weather, action_choices, character_state_dict
            )

            # Step 5: Query LLM
            logger.debug(
                f"Sending prompt to LLM for {character_name}: {prompt[:100]}..."
            )
            llm_responses = self.brain_io.input_to_model([prompt])

            if not llm_responses or len(llm_responses) == 0:
                logger.warning(f"No LLM response received for {character_name}")
                return potential_actions[:1]  # Return top utility action as fallback

            llm_response_text = (
                llm_responses[0][0]
                if isinstance(llm_responses[0], tuple)
                else llm_responses[0]
            )
            logger.debug(f"LLM response for {character_name}: {llm_response_text}")

            # Step 6: Interpret LLM response
            try:
                selected_actions = self.output_interpreter.interpret_response(
                    llm_response_text, character, potential_actions
                )

                if selected_actions and len(selected_actions) > 0:
                    logger.info(
                        f"LLM selected action for {character_name}: {[a.name for a in selected_actions]}"
                    )
                    return selected_actions
                else:
                    logger.warning(
                        f"LLM response could not be interpreted for {character_name}"
                    )
                    return potential_actions[:1]  # Fallback to top utility action

            except Exception as interpretation_error:
                logger.error(
                    f"Error interpreting LLM response for {character_name}: {interpretation_error}"
                )
                return potential_actions[:1]  # Fallback to top utility action

        except Exception as e:
            character_name = getattr(character, 'name', 'Character') if hasattr(character, 'name') else 'Character'
            logger.error(f"Error in LLM decision-making for {character_name}: {e}")
            # Always fallback to utility-based decision making
            return self.get_daily_actions(character)[:1]

    # --- Other methods from the original file (potentially needing updates) ---
    def update_strategy(self, events, subject="Emma"):
        """
        Updates strategy based on events using GOAP planning.
        Enhanced to handle various event types and respond meaningfully.
        """
        if not events:
            return self.plan_daily_activities(subject)
            
        for event in events:
            # Handle different event types with specific responses
            if hasattr(event, 'type'):
                event_type = event.type
            elif isinstance(event, dict):
                event_type = event.get('type', 'unknown')
            else:
                event_type = str(type(event).__name__).lower()
                
            logger.debug(f"Processing event of type '{event_type}' for {subject}")
            
            # Route to specific event handlers
            if event_type == "new_day":
                return self.plan_daily_activities(subject)
            elif event_type in ["social", "celebration", "festival"]:
                return self._handle_social_event(event, subject)
            elif event_type in ["economic", "trade", "market"]:
                return self._handle_economic_event(event, subject)
            elif event_type in ["crisis", "disaster", "emergency"]:
                return self._handle_crisis_event(event, subject)
            elif event_type in ["work", "task", "project"]:
                return self._handle_work_event(event, subject)
            elif event_type in ["weather", "environmental"]:
                return self._handle_environmental_event(event, subject)
            else:
                # Default event handling
                return self._handle_generic_event(event, subject)

    def _handle_social_event(self, event, character):
        """Handle social events by prioritizing social actions."""
        try:
            # Create goal focused on social interaction and happiness
            goal = Goal(
                name="social_engagement",
                target_effects={"social_wellbeing": 80, "happiness": 75},
                priority=0.8
            )
            
            # Get social-focused actions
            actions = self.get_daily_actions(character)
            social_actions = [a for a in actions if 'social' in a.name.lower() or 'talk' in a.name.lower() or 'help' in a.name.lower()]
            
            if not social_actions:
                social_actions = actions[:3]  # Take top 3 if no social actions found
                
            return self._plan_with_goal_and_actions(character, goal, social_actions)
            
        except Exception as e:
            logger.warning(f"Error handling social event: {e}")
            return self.plan_daily_activities(character)

    def _handle_economic_event(self, event, character):
        """Handle economic events by prioritizing wealth and trade actions."""
        try:
            goal = Goal(
                name="economic_opportunity",
                target_effects={"wealth": 100, "satisfaction": 70},
                priority=0.9
            )
            
            actions = self.get_daily_actions(character)
            economic_actions = [a for a in actions if 'work' in a.name.lower() or 'trade' in a.name.lower() or 'craft' in a.name.lower()]
            
            if not economic_actions:
                economic_actions = actions[:3]
                
            return self._plan_with_goal_and_actions(character, goal, economic_actions)
            
        except Exception as e:
            logger.warning(f"Error handling economic event: {e}")
            return self.plan_daily_activities(character)

    def _handle_crisis_event(self, event, character):
        """Handle crisis events by prioritizing safety and recovery actions."""
        try:
            goal = Goal(
                name="crisis_response",
                target_effects={"safety": 90, "energy": 60},
                priority=1.0  # Highest priority
            )
            
            # In crisis, prioritize safety and basic needs
            crisis_actions = []
            
            # Add safety actions
            crisis_actions.append(self.create_action(
                name="Seek Safety",
                preconditions={},
                effects=[
                    {"targets": ["initiator"], "attribute": "safety", "change_value": 20},
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -10}
                ],
                cost=0.2
            ))
            
            # Add help others action
            crisis_actions.append(Action(
                name="Help Others",
                preconditions={},
                effects=[
                    {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 15},
                    {"targets": ["initiator"], "attribute": "satisfaction", "change_value": 10}
                ],
                cost=0.3
            ))
            
            return self._plan_with_goal_and_actions(character, goal, crisis_actions)
            
        except Exception as e:
            logger.warning(f"Error handling crisis event: {e}")
            return self.plan_daily_activities(character)

    def _handle_work_event(self, event, character):
        """Handle work events by prioritizing productivity and skill development."""
        try:
            goal = Goal(
                name="work_productivity",
                target_effects={"job_performance": 85, "satisfaction": 65},
                priority=0.7
            )
            
            actions = self.get_daily_actions(character)
            work_actions = [a for a in actions if 'work' in a.name.lower() or 'craft' in a.name.lower() or 'build' in a.name.lower()]
            
            if not work_actions:
                # Create a basic work action if none exist
                work_actions = [WorkAction(
                    name="Contribute to Project",
                    effects=[
                        {"targets": ["initiator"], "attribute": "job_performance", "change_value": 10},
                        {"targets": ["initiator"], "attribute": "energy", "change_value": -15}
                    ]
                )]
                
            return self._plan_with_goal_and_actions(character, goal, work_actions)
            
        except Exception as e:
            logger.warning(f"Error handling work event: {e}")
            return self.plan_daily_activities(character)

    def _handle_environmental_event(self, event, character):
        """Handle environmental/weather events by adapting to conditions."""
        try:
            # Determine appropriate response based on event severity
            event_impact = getattr(event, 'impact', 0)
            
            if event_impact < 0:  # Negative weather (storm, etc.)
                goal = Goal(
                    name="weather_adaptation",
                    target_effects={"safety": 80, "energy": 70},
                    priority=0.8
                )
                
                # Stay indoors, rest, prepare
                weather_actions = [
                    SleepAction(name="Take Shelter", effects=[
                        {"targets": ["initiator"], "attribute": "safety", "change_value": 15},
                        {"targets": ["initiator"], "attribute": "energy", "change_value": 10}
                    ]),
                ]
            else:  # Positive weather
                goal = Goal(
                    name="enjoy_weather",
                    target_effects={"happiness": 80, "energy": 75},
                    priority=0.6
                )
                
                # Outdoor activities, work, socialize
                weather_actions = [
                    WanderAction(name="Enjoy Outdoors", effects=[
                        {"targets": ["initiator"], "attribute": "happiness", "change_value": 10},
                        {"targets": ["initiator"], "attribute": "energy", "change_value": 5}
                    ])
                ]
                
            return self._plan_with_goal_and_actions(character, goal, weather_actions)
            
        except Exception as e:
            logger.warning(f"Error handling environmental event: {e}")
            return self.plan_daily_activities(character)

    def _handle_generic_event(self, event, character):
        """Handle generic events with balanced response."""
        try:
            # Default balanced goal
            goal = Goal(
                name="balanced_response",
                target_effects={"satisfaction": 70, "energy": 65},
                priority=0.6
            )
            
            # Use standard daily actions
            actions = self.get_daily_actions(character)
            
            return self._plan_with_goal_and_actions(character, goal, actions[:5])
            
        except Exception as e:
            logger.warning(f"Error handling generic event: {e}")
            return self.plan_daily_activities(character)

    def _plan_with_goal_and_actions(self, character, goal, actions):
        """Helper method to plan actions with a specific goal."""
        try:
            # Get character state
            if isinstance(character, str):
                current_state = State({"satisfaction": 50, "energy": 50, "happiness": 50})
            else:
                character_state_dict = self.get_character_state_dict(character)
                current_state = State(character_state_dict)
            
            # Plan using GOAP with the specific goal and actions
            if self.goap_planner:
                plan = self.goap_planner.plan_actions(character, goal, current_state, actions)
                
                if plan:
                    # Evaluate the utility of the plan
                    final_decision = self.goap_planner.evaluate_utility(plan, character)
                    return final_decision
                    
            # Fallback to highest utility action
            if actions:
                return actions[0]  # Actions are already sorted by utility
                
            return None
            
        except Exception as e:
            logger.warning(f"Error in planning with goal and actions: {e}")
            return None

    def plan_daily_activities(self, character):
        """
        Plans the daily activities for the given character.
        This method properly interfaces with GOAPPlanner.
        """
        # Define a proper Goal object for daily activities
        goal = Goal(
            name="daily_wellbeing",
            target_effects={"satisfaction": SATISFACTION_TARGET, "energy": ENERGY_TARGET},
            priority=0.8
        )

        # Get potential actions from the utility-based action generator
        actions = self.get_daily_actions(character)

        # Create current state - handle both Character objects and string names
        if isinstance(character, str):
            # If character is a string, create a simple state
            current_state = State({"satisfaction": 50, "energy": 50, "hunger": 50})
        else:
            # If character is a Character object, get its state
            character_state_dict = self.get_character_state_dict(character)
            current_state = State(character_state_dict)

        # Plan using GOAP with correct interface
        plan = self.goap_planner.plan_actions(character, goal, current_state, actions)

        # Evaluate the utility of the plan using the planner's evaluate_utility method
        if plan:
            final_decision = self.goap_planner.evaluate_utility(plan, character)
            return final_decision
        else:
            # If no plan found, return the highest utility action as fallback
            if actions:
                return max(actions, key=lambda a: self.goap_planner.calculate_utility(a, character))
            return None

    def get_career_actions(self, character, job_details):
        # This is a placeholder and would need similar utility-based ranking
        return [
            {
                "name": "Accept Offer",
                "career_progress": 15,
                "cost": 0,
                "effects": [],
            },  # Mocking for Action structure
            {
                "name": "Negotiate Salary",
                "career_progress": 5,
                "cost": 0,
                "effects": [],
            },
            {"name": "Decline Offer", "career_progress": 0, "cost": 0, "effects": []},
        ]

    def respond_to_job_offer(self, character, job_details, graph=None):
        """
        Plans a response to a job offer using GOAP planning.
        This method properly interfaces with GOAPPlanner.
        """
        # Create a proper Goal object for career decisions
        goal = Goal(
            name="career_advancement",
            target_effects={"career_progress": 80, "satisfaction": 70},
            priority=0.9
        )
        
        # Create current state
        current_state = State({"satisfaction": 70, "career_progress": 50})  # Assuming current state
        
        # Get career actions
        actions = self.get_career_actions(character, job_details)
        
        # Convert dictionary actions to Action objects if needed
        action_objects = []
        for action_dict in actions:
            if isinstance(action_dict, dict):
                action_obj = Action(
                    name=action_dict["name"],
                    preconditions=[],
                    effects=[
                        {"attribute": "career_progress", "change_value": action_dict.get("career_progress", 0)},
                        {"attribute": "satisfaction", "change_value": action_dict.get("cost", 0) * -1}  # Cost reduces satisfaction
                    ],
                    cost=action_dict.get("cost", 0)
                )
                action_objects.append(action_obj)
            else:
                action_objects.append(action_dict)

        # Use GOAP to plan career moves with correct interface
        plan = self.goap_planner.plan_actions(character, goal, current_state, action_objects)

        # Evaluate the utility of the plan
        if plan:
            final_decision = self.goap_planner.evaluate_utility(plan, character)
            return final_decision
        else:
            # Fallback to first action if no plan found
            return action_objects[0] if action_objects else None
