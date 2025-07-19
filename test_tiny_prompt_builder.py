import unittest
from unittest.mock import MagicMock, patch
import sys, types

# Provide a minimal stub for tiny_characters to avoid heavy dependencies
tc_stub = types.ModuleType('tiny_characters')
class DummyCharacter:
    pass
tc_stub.Character = DummyCharacter
sys.modules['tiny_characters'] = tc_stub

from tiny_prompt_builder import PromptBuilder, descriptors

class SimpleCharacter:
    def __init__(self):
        self.name = "Emily"
        self.job = "Engineer"
        self.health_status = 10
        self.hunger_level = 5
        self.wealth_money = 10
        self.mental_health = 8
        self.social_wellbeing = 8
        self.job_performance = "average"
        self.recent_event = "nothing"
        self.long_term_goal = "excel at testing"

class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.character = SimpleCharacter()
        self.prompt_builder = PromptBuilder(self.character)
        self.mock_needs = MagicMock()
        self.mock_actions = MagicMock()
        self.prompt_builder.needs_priorities_func = self.mock_needs
        self.prompt_builder.action_options = self.mock_actions
        self.prompt_builder.long_term_goal = "achieve greatness"
        # Ensure descriptor defaults exist to avoid KeyError
        descriptors.job_currently_working_on.setdefault("default", ["a project"])
        descriptors.job_planning_to_attend.setdefault("default", ["an event"])
        descriptors.job_hoping_to_there.setdefault("default", ["participate"])
        descriptors.feeling_health.setdefault("default", ["healthy"])
        descriptors.feeling_hunger.setdefault("default", ["hungry"])
        descriptors.event_recent.setdefault("default", ["Recently"])
        descriptors.financial_situation.setdefault("default", ["you have some money"])

    def test_calculate_needs_priorities(self):
        self.mock_needs.calculate_needs_priorities.return_value = {"need1": 1}
        self.prompt_builder.calculate_needs_priorities()
        self.mock_needs.calculate_needs_priorities.assert_called_once_with(self.character)
        self.assertEqual(self.prompt_builder.needs_priorities, {"need1": 1})

    def test_generate_daily_routine_prompt(self):
        self.mock_actions.prioritize_actions.return_value = ["buy_food", "social_visit"]
        with patch('tiny_prompt_builder.descriptors.get_action_descriptors') as mock_desc:
            mock_desc.side_effect = ["Go shopping", "Meet friend"]
            prompt = self.prompt_builder.generate_daily_routine_prompt("morning", "sunny")
        self.mock_actions.prioritize_actions.assert_called_once_with(self.character)
        self.assertIn("1. Go shopping to Buy_Food.", prompt)
        self.assertIn("2. Meet friend to Social_Visit.", prompt)
        self.assertIn("Emily, I choose", prompt)

if __name__ == '__main__':
    unittest.main()
