import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tiny_goap_system import ActionWrapper
from tiny_strategy_manager import StrategyManager


class TestStrategyManagerEventPlanning(unittest.TestCase):
    def _build_manager(self):
        manager = StrategyManager.__new__(StrategyManager)
        manager.graph_manager = Mock()
        manager.graph_manager.get_character = Mock(return_value=None)
        manager.goap_planner = Mock()
        manager.use_llm = False
        manager.brain_io = None
        manager.output_interpreter = None
        manager._characters_using_llm = set()
        manager.get_daily_actions = Mock(return_value=[ActionWrapper(name="FallbackAction")])
        manager.plan_daily_activities = Mock(return_value=["fallback_plan"])
        return manager

    def test_update_strategy_processes_multiple_events_and_characters(self):
        manager = self._build_manager()

        manager.graph_manager.get_character_state.side_effect = [
            {"energy": 50},
            {"energy": 40},
        ]
        manager.graph_manager.get_possible_actions.side_effect = [
            [
                {"name": "Talk", "effects": [], "preconditions": {}, "cost": 1},
                {"name": "Craft", "effects": [], "preconditions": {}, "cost": 1},
            ],
            [
                {"name": "Work", "effects": [], "preconditions": {}, "cost": 1},
                {"name": "Chat", "effects": [], "preconditions": {}, "cost": 1},
            ],
        ]

        manager.goap_planner.plan_actions.side_effect = lambda character, goal, state, actions: actions

        social_event = SimpleNamespace(type="social", participants=["Alice"])
        work_event = SimpleNamespace(type="work", participants=["Bob"])

        plans = manager.update_strategy([social_event, work_event])

        self.assertEqual(set(plans.keys()), {"alice", "bob"})
        self.assertEqual(manager.graph_manager.get_character_state.call_count, 2)
        self.assertEqual(manager.graph_manager.get_possible_actions.call_count, 2)

        social_actions = manager.goap_planner.plan_actions.call_args_list[0][0][3]
        self.assertTrue(
            all(
                "talk" in a.name.lower()
                or "chat" in a.name.lower()
                or "social" in a.name.lower()
                or "help" in a.name.lower()
            )
            for a in social_actions
        )

        work_actions = manager.goap_planner.plan_actions.call_args_list[1][0][3]
        self.assertTrue(
            all(
                "work" in a.name.lower()
                or "trade" in a.name.lower()
                or "craft" in a.name.lower()
                or "build" in a.name.lower()
                or "sell" in a.name.lower()
                or "buy" in a.name.lower()
                or "earn" in a.name.lower()
                or "job" in a.name.lower()
            )
            for a in work_actions
        )

    def test_plan_fallback_when_goap_returns_none(self):
        manager = self._build_manager()

        manager.graph_manager.get_character_state.return_value = {"energy": 50}
        manager.graph_manager.get_possible_actions.return_value = [{"name": "Talk"}]
        manager.goap_planner.plan_actions.return_value = None

        social_event = SimpleNamespace(type="social", participants=["Alice"])
        plans = manager.update_strategy([social_event])

        self.assertEqual(plans.get("alice"), ["fallback_plan"])
        manager.plan_daily_activities.assert_called_once()

    def test_graph_errors_fallback_to_defaults(self):
        manager = self._build_manager()

        manager.graph_manager.get_character_state.side_effect = Exception("state error")
        manager.graph_manager.get_possible_actions.side_effect = Exception("actions error")
        manager.goap_planner.plan_actions.return_value = [ActionWrapper(name="planned")]

        event = SimpleNamespace(type="social", participants=["Alice"])

        plans = manager.update_strategy([event])

        self.assertIn("alice", plans)
        self.assertEqual(len(plans["alice"]), 1)
        self.assertEqual(getattr(plans["alice"][0], "name", None), "planned")


if __name__ == "__main__":
    unittest.main()
