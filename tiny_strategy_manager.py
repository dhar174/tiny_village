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
from tiny_utility_functions import evaluate_utility


class StrategyManager:
    """
    The StrategyManager class is responsible for managing the strategies used in goal-oriented action planning and utility evaluation.
    It uses the GOAPPlanner and GraphManager to plan actions and manage graphs respectively.
    """

    def __init__(self):
        self.goap_planner = GOAPPlanner()
        self.graph_manager = GraphManager()

    def update_strategy(self, events, subject="Emma"):
        """
        Updates the strategy based on the given event.
        If the event is "new_day", it gets the character state and possible actions for "Emma" and plans the actions.
        """
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
        # Placeholder function to fetch or generate possible actions for a character's day
        return [
            {"name": "Go to Gym", "energy_cost": 15, "satisfaction": 10},
            {"name": "Visit Cafe", "energy_cost": 5, "satisfaction": 6},
            {"name": "Work on Project", "energy_cost": 10, "satisfaction": 8},
            {"name": "Socialize at Park", "energy_cost": 8, "satisfaction": 9},
        ]

    def get_career_actions(self, character, job_details):
        # Example actions based on job details
        return [
            {"name": "Accept Offer", "career_progress": 15},
            {"name": "Negotiate Salary", "career_progress": 5},
            {"name": "Decline Offer", "career_progress": 0},
        ]

    def respond_to_job_offer(self, character, job_details, graph):
        goal = {"career_progress": "max"}
        current_state = {"satisfaction": 100}  # Assuming current job satisfaction
        actions = self.get_career_actions(character, job_details)

        # Use GOAP to plan career moves
        plan = self.goap_planner(character, goal, current_state, actions)

        # Evaluate the utility of the plan
        final_decision = evaluate_utility(plan, character)

        return final_decision
