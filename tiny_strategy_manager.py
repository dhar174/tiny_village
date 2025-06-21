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
from tiny_graph_manager import GraphManager
from tiny_utility_functions import (
    calculate_action_utility,
    Goal,
)  # evaluate_utility seems to be for plans
from tiny_characters import Character  # Assuming Character class is imported
from actions import Action  # Use the modified Action from actions.py
from tiny_prompt_builder import PromptBuilder
from tiny_brain_io import TinyBrainIO
from tiny_output_interpreter import OutputInterpreter
import logging

logger = logging.getLogger(__name__)


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
        self.goap_planner = GOAPPlanner()
        self.graph_manager = GraphManager()
        self.use_llm = use_llm

        # Initialize LLM components if needed
        if self.use_llm:
            self.brain_io = TinyBrainIO(
                model_name or "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"
            )
            self.output_interpreter = OutputInterpreter()
        else:
            self.brain_io = None
            self.output_interpreter = None

    def get_character_state_dict(self, character: Character) -> dict:
        """
        Extracts a simplified dictionary representation of the character's state
        relevant for utility calculations.
        """
        state = {}
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

        return state

    def get_daily_actions(
        self, character: Character, current_goal: Goal = None
    ) -> list[Action]:
        """
        Generates a list of potential daily actions for a character,
        calculates their utility, and returns them sorted by utility.
        """
        potential_actions = []

        # --- Action Generation ---
        # 1. Generic Actions
        potential_actions.append(NoOpAction())
        potential_actions.append(
            WanderAction(effects=[{"attribute": "energy", "change_value": -0.05}])
        )  # Wandering costs a little energy

        # 2. Contextual Actions
        # Assumed Character object structure:
        # - character.inventory: an object with get_food_items() -> list of FoodItem objects
        #   - FoodItem has .name and .calories (or similar attribute for hunger satisfaction)
        # - character.energy: float/int, current energy level (e.g., 0-10, or 0.0-1.0)
        # - character.hunger_level: float/int, current hunger (e.g., 0-10, 0 is not hungry, 10 is very hungry)
        # - character.location: an object with .name (e.g., "Home", "Cafe")
        # - character.job: a string or an object indicating job (e.g., "Farmer", None)

        LOW_ENERGY_THRESHOLD = 0.3  # Assuming energy is normalized 0-1 for this check
        HIGH_HUNGER_THRESHOLD = 0.6  # Assuming hunger is normalized 0-1 for this check

        char_energy_normalized = (
            getattr(character, "energy", 5.0) / 10.0
        )  # Normalize for threshold checks

        # Eat Actions (from inventory)
        if hasattr(character, "inventory") and hasattr(
            character.inventory, "get_food_items"
        ):

            food_items = character.inventory.get_food_items()
            if food_items:
                # Consider eating the first available food item for simplicity
                # More advanced: choose based on hunger level or food properties
                for food_item in food_items[:2]:  # Limit to checking first 2 food items
                    # Assuming food_item.calories is a positive value indicating hunger reduction potential
                    # The effect's change_value for hunger should be negative.
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
        if hasattr(character, "location") and hasattr(character.location, "name"):
            if (
                char_energy_normalized < LOW_ENERGY_THRESHOLD
                and character.location.name == "Home"
            ):
                sleep_effects = [
                    {"attribute": "energy", "change_value": 0.7}
                ]  # Restore 70% energy
                potential_actions.append(
                    SleepAction(effects=sleep_effects, cost=0)
                )  # Sleep itself has no direct cost other than time

        # Work Action
        if (
            hasattr(character, "job")
            and character.job
            and character.job != "unemployed"
        ):
            job_name = (
                character.job.job_title
                if hasattr(character.job, "job_title")
                else str(character.job)
            )
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
        self, character: Character, time="morning", weather="clear"
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
            prompt = prompt_builder.generate_decision_prompt(
                time, weather, action_choices, character_state_dict
            )

            # Step 5: Query LLM
            logger.debug(
                f"Sending prompt to LLM for {character.name}: {prompt[:100]}..."
            )
            llm_responses = self.brain_io.input_to_model([prompt])

            if not llm_responses or len(llm_responses) == 0:
                logger.warning(f"No LLM response received for {character.name}")
                return potential_actions[:1]  # Return top utility action as fallback

            llm_response_text = (
                llm_responses[0][0]
                if isinstance(llm_responses[0], tuple)
                else llm_responses[0]
            )
            logger.debug(f"LLM response for {character.name}: {llm_response_text}")

            # Step 6: Interpret LLM response
            try:
                selected_actions = self.output_interpreter.interpret_response(
                    llm_response_text, character, potential_actions
                )

                if selected_actions and len(selected_actions) > 0:
                    logger.info(
                        f"LLM selected action for {character.name}: {[a.name for a in selected_actions]}"
                    )
                    return selected_actions
                else:
                    logger.warning(
                        f"LLM response could not be interpreted for {character.name}"
                    )
                    return potential_actions[:1]  # Fallback to top utility action

            except Exception as interpretation_error:
                logger.error(
                    f"Error interpreting LLM response for {character.name}: {interpretation_error}"
                )
                return potential_actions[:1]  # Fallback to top utility action

        except Exception as e:
            logger.error(f"Error in LLM decision-making for {character.name}: {e}")
            # Always fallback to utility-based decision making
            return self.get_daily_actions(character)[:1]

    # --- Other methods from the original file (potentially needing updates) ---
    def update_strategy(self, events, subject="Emma"):
        # This method likely needs significant updates to use Character objects
        # and the new get_daily_actions with utility.
        # For now, focusing on get_daily_actions as per subtask.
        for event in events:
            if event.type == "new_day":
                return self.plan_daily_activities("Emma")
            character_state = self.graph_manager.get_character_state("Emma")
            actions = self.graph_manager.get_possible_actions("Emma")
            plan = self.goap_planner.plan_actions(character_state, actions)
            return plan

    def plan_daily_activities(self, character):
        """
        Plans the daily activities for the given character.
        It defines the goal for daily activities, gets potential actions, and uses the graph to analyze current relationships and preferences.
        """
        # Define the goal for daily activities
        goal = {"satisfaction": max, "energy_usage": min}

        # Get potential actions from a dynamic or context-specific action generator
        actions = self.get_daily_actions(character)

        # Use the graph to analyze current relationships and preferences
        current_state = self.graph_analysis(character_graph, character, "daily")

        # Plan the career steps using GOAP
        plan = self.goap_planner(character, goal, current_state, actions)

        # Evaluate the utility of each step in the plan
        final_decision = evaluate_utility(plan, character)

        return final_decision

    def get_daily_actions(self, character):
        """Get daily actions for a character."""
        # This is a placeholder for daily action logic
        return []

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

    def respond_to_job_offer(self, character, job_details, graph):
        # FIX: This method also needs significant updates
        goal = {"career_progress": "max"}
        current_state = {"satisfaction": 100}  # Assuming current job satisfaction
        actions = self.get_career_actions(character, job_details)

        # Use GOAP to plan career moves
        plan = self.goap_planner(character, goal, current_state, actions)

        # Evaluate the utility of the plan
        final_decision = evaluate_utility(plan, character)

        return final_decision
