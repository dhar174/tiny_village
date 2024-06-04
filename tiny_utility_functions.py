""" 
4. Utility Evaluation
Where it happens: Part of goap_system.py or a dedicated utility module
What happens: Actions are evaluated based on their expected outcomes, like improving mood or health, with adjustments based on current needs and previous experiences. """


class UtilityEvaluator:
    """
    The UtilityEvaluator class is responsible for evaluating the utility of each possible action based on the character's current state and the planning context.
    """

    def __init__(self, actions, character_state, planning_context):
        self.actions = actions
        self.character_state = character_state
        self.planning_context = planning_context

    def evaluate_utility(self):
        """
        Evaluates the utility of each possible action.

        Returns:
            ranked_actions (list): Actions sorted by their utility scores.
        """
        utilities = {}
        for action in self.actions:
            utility_score = 0
            # Evaluate base on planning context specifics
            if self.planning_context == "daily":
                utility_score += (
                    action["time_efficiency"] * self.character_state["time_preference"]
                )
                utility_score += (
                    action["satisfaction"]
                    * self.character_state["happiness_importance"]
                )
            elif self.planning_context == "career":
                utility_score += (
                    action["career_growth"] * self.character_state["career_ambition"]
                )
                utility_score += (
                    action["risk"] * -1 * self.character_state["risk_aversion"]
                )
            elif self.planning_context == "relationship":
                utility_score += (
                    action["relationship_improvement"]
                    * self.character_state["social_importance"]
                )
                utility_score += (
                    action["emotional_cost"]
                    * -1
                    * self.character_state["emotional_stability"]
                )

            utilities[action["name"]] = utility_score

        # Sort actions based on utility scores in descending order
        ranked_actions = sorted(
            utilities.items(), key=lambda item: item[1], reverse=True
        )
        return [action[0] for action in ranked_actions]
