import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tiny_strategy_manager import StrategyManager, ActionWrapper


class TestStrategyManagerEventPlanning(unittest.TestCase):
    def test_update_strategy_processes_multiple_events_and_characters(self):
        manager = StrategyManager.__new__(StrategyManager)
        manager.graph_manager = Mock()
        manager.goap_planner = Mock()
        manager.use_llm = False
        manager.brain_io = None
        manager.output_interpreter = None
        manager._characters_using_llm = set()

        manager.get_daily_actions = Mock(return_value=[Mock(name="FallbackAction")])
        manager.plan_daily_activities = Mock(return_value=["fallback_plan"])

        manager.graph_manager.get_character_state.side_effect = [
            {"energy": 50},
            {"energy": 40},
        ]
        manager.graph_manager.get_possible_actions.side_effect = [
            [{"name": "Talk", "effects": [], "preconditions": {}, "cost": 1}],
            [{"name": "Work", "effects": [], "preconditions": {}, "cost": 1}],
        ]
        manager.goap_planner.plan_actions.return_value = ["planned"]

        social_event = SimpleNamespace(type="social", participants=["Alice"])
        work_event = SimpleNamespace(type="work", participants=["Bob"])

        plans = manager.update_strategy([social_event, work_event])

        self.assertEqual(set(plans.keys()), {"Alice", "Bob"})
        self.assertEqual(manager.graph_manager.get_character_state.call_count, 2)
        self.assertEqual(manager.graph_manager.get_possible_actions.call_count, 2)
        self.assertEqual(manager.goap_planner.plan_actions.call_count, 2)

        used_actions = manager.goap_planner.plan_actions.call_args_list[0][0][3]
        self.assertTrue(all(isinstance(action, ActionWrapper) for action in used_actions))


if __name__ == "__main__":
    unittest.main()
