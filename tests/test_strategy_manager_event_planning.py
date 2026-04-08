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

        manager.goap_planner.plan_actions.side_effect = lambda character, goal, state, actions: actions

        social_event = SimpleNamespace(type="social", participants=["Alice"])
        work_event = SimpleNamespace(type="work", participants=["Bob"])

        plans = manager.update_strategy([social_event, work_event])

        self.assertEqual(set(plans.keys()), {"alice", "bob"})
        self.assertEqual(manager.goap_planner.plan_actions.call_count, 2)

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

        unknown_event = SimpleNamespace(type="mystery", participants=["Alice"])
        plans = manager.update_strategy([unknown_event])

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

    def test_get_character_state_dict_aliases_wealth_money_for_dict_input(self):
        manager = self._build_manager()

        state = manager.get_character_state_dict({"name": "Alice", "wealth_money": 75})

        self.assertEqual(state["money"], 75.0)
        self.assertEqual(state["wealth_money"], 75)

    def test_get_character_state_dict_only_falls_back_to_satisfaction_when_happiness_missing(self):
        manager = self._build_manager()

        character = SimpleNamespace(
            hunger_level=5,
            energy=5,
            wealth_money=10,
            social_wellbeing=5,
            mental_health=5,
            health_status=100,
            satisfaction=80,
            happiness="not-a-number",
            job_performance=50,
        )

        state = manager.get_character_state_dict(character)

        self.assertEqual(state["satisfaction"], 0.8)
        self.assertEqual(state["happiness"], 0.5)

    def test_get_character_state_dict_uses_satisfaction_when_happiness_attribute_is_missing(self):
        manager = self._build_manager()

        character = SimpleNamespace(
            hunger_level=5,
            energy=5,
            wealth_money=10,
            social_wellbeing=5,
            mental_health=5,
            health_status=100,
            satisfaction=80,
            job_performance=50,
        )

        state = manager.get_character_state_dict(character)

        self.assertEqual(state["satisfaction"], 0.8)
        self.assertEqual(state["happiness"], 0.8)

    def test_plan_daily_activities_treats_empty_plan_as_already_satisfied(self):
        manager = self._build_manager()
        manager.plan_daily_activities = StrategyManager.plan_daily_activities.__get__(
            manager, StrategyManager
        )
        manager.goap_planner.plan_actions.return_value = []
        manager.goap_planner.evaluate_utility.return_value = ActionWrapper(name="ShouldNotBeUsed")
        manager.graph_manager.get_possible_actions.return_value = [
            {"name": "LowUtility", "utility": 1.0, "effects": [], "preconditions": {}, "cost": 1},
            {"name": "HighUtility", "utility": 5.0, "effects": [], "preconditions": {}, "cost": 1},
        ]

        result = manager.plan_daily_activities("alice")

        self.assertEqual(result, [])
        manager.goap_planner.evaluate_utility.assert_not_called()

    def test_plan_daily_activities_uses_utility_based_fallback_when_planner_missing(self):
        manager = self._build_manager()
        manager.plan_daily_activities = StrategyManager.plan_daily_activities.__get__(
            manager, StrategyManager
        )
        manager.goap_planner = None
        manager.graph_manager.get_possible_actions.return_value = [
            {
                "name": "LowUtility",
                "utility": 1.0,
                "effects": [{"attribute": "energy", "change_value": 0.05}],
                "preconditions": {},
                "cost": 1,
            },
            {
                "name": "HighUtility",
                "utility": 9.0,
                "effects": [{"attribute": "energy", "change_value": 0.25}],
                "preconditions": {},
                "cost": 1,
            },
        ]

        result = manager.plan_daily_activities("alice")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "HighUtility")


if __name__ == "__main__":
    unittest.main()
