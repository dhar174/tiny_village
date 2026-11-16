import unittest
from types import SimpleNamespace
from tiny_gameplay_controller import GameplayController
from tests.test_graph_initialization import FreshPythonTestCase
from tests.test_goap_graph_contracts import WORLD_SETUP


class TestGameplayControllerEventProcessing(FreshPythonTestCase):
    def setUp(self):
        self.controller = SimpleNamespace(
            game_statistics={
                "actions_executed": 0,
                "actions_failed": 0,
                "characters_created": 0,
                "errors_recovered": 0,
            },
            game_state={},
        )

    def test_normalize_and_prioritize_events_deterministic_sort(self):
        social_event = SimpleNamespace(
            name="social-object",
            type="social",
            importance=3,
            impact=-7,
        )
        economic_event = {
            "name": "economic-dict",
            "type": "economic",
            "importance": 8,
            "impact": 2,
        }
        career_event = {
            "name": "career-dict",
            "type": "career",
            "importance": 8,
            "impact": 2,
        }

        update_errors = []
        normalized_events = GameplayController._normalize_and_prioritize_events(
            self.controller,
            [social_event, economic_event, career_event],
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(
            [event["name"] if isinstance(event, dict) else event.name for event in normalized_events],
            ["career-dict", "economic-dict", "social-object"],
        )
        self.assertIsInstance(normalized_events[0], dict)
        self.assertEqual(normalized_events[0]["_normalized_event_type"], "career")
        self.assertIs(normalized_events[2], social_event)
        self.assertEqual(social_event._normalized_event_type, "social")
        self.assertEqual(social_event._normalized_importance, 3)
        self.assertEqual(social_event._normalized_impact, -7)

    def test_normalize_and_prioritize_events_normalizes_event_type_casing(self):
        update_errors = []
        normalized_events = GameplayController._normalize_and_prioritize_events(
            self.controller,
            [{"name": "market-shift", "type": " Economic ", "importance": 2, "impact": -4}],
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(normalized_events[0]["_normalized_event_type"], "economic")

    def test_apply_event_consequences_uses_normalized_metadata_for_dispatch(self):
        normalized_events = [
            {
                "name": "market-shift",
                "type": "ignored",
                "impact": 999,
                "_normalized_event_type": "economic",
                "_normalized_impact": 3,
            },
            SimpleNamespace(
                name="festival",
                type="ignored",
                impact=999,
                _normalized_event_type="social",
                _normalized_impact=2,
            ),
            SimpleNamespace(
                name="storm",
                type="ignored",
                impact=999,
                _normalized_event_type="weather",
                _normalized_impact=-4,
            ),
            {
                "name": "job-fair",
                "_normalized_event_type": "career",
                "_normalized_impact": 5,
            },
        ]

        update_errors = []
        GameplayController._apply_event_consequences(
            self.controller,
            normalized_events,
            {"processed_events": ["market-shift", "festival", "storm", "job-fair"]},
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(self.controller.game_statistics["events_processed"], 4)
        self.assertEqual(self.controller.game_statistics["events_economic"], 1)
        self.assertEqual(self.controller.game_statistics["events_social"], 1)
        self.assertEqual(self.controller.game_statistics["events_weather"], 1)
        self.assertEqual(self.controller.game_statistics["events_career"], 1)
        self.assertEqual(self.controller.game_state["economy_stability"], 53)
        self.assertEqual(self.controller.game_state["social_cohesion"], 52)
        self.assertEqual(self.controller.game_state["environment_pressure"], 4)
        self.assertEqual(self.controller.game_state["job_market_activity"], 55)

    def test_apply_event_consequences_ignores_failed_events(self):
        update_errors = []
        normalized_events = [
            {"name": "market-shift", "_normalized_event_type": "economic", "_normalized_impact": 3},
            {"name": "storm", "_normalized_event_type": "weather", "_normalized_impact": -4},
        ]

        GameplayController._apply_event_consequences(
            self.controller,
            normalized_events,
            {"processed_events": ["market-shift"], "failed_events": ["storm"]},
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(self.controller.game_statistics["events_processed"], 1)
        self.assertEqual(self.controller.game_statistics["events_economic"], 1)
        self.assertNotIn("events_weather", self.controller.game_statistics)
        self.assertEqual(self.controller.game_state["economy_stability"], 53)
        self.assertNotIn("environment_pressure", self.controller.game_state)

    def test_apply_event_consequences_tracks_new_event_type_without_preseeded_counter(self):
        update_errors = []

        GameplayController._apply_event_consequences(
            self.controller,
            [{"name": "harvest-fair", "_normalized_event_type": "festival", "_normalized_impact": 1}],
            {"processed_events": ["harvest-fair"]},
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(self.controller.game_statistics["events_festival"], 1)
        self.assertNotIn("festival", self.controller.game_state)

    def test_live_events_stage_requests_without_eager_planning(self):
        self.run_python(WORLD_SETUP + """
try:
 from unittest.mock import patch
 actor = next(iter(controller.characters.values()))
 strategy = controller.strategy_manager
 controller.events.append({'name':'Tick day','type':'new_day','importance':5,'participants':[actor]})
 with patch.object(strategy,'plan_request',wraps=strategy.plan_request) as planning:
  controller.update_game_state(.5)
  assert planning.call_count == 0
  assert controller.game_statistics['actions_executed'] == 0
  controller.update_game_state(.5)
  assert planning.call_count == 1
  assert controller.game_statistics['actions_executed'] == 1
  status = controller.get_plan_execution_state(actor)
  assert status['source'] == 'event:new_day' and status['cursor'] == 1
  controller.update_game_state(.25)
  assert planning.call_count == 1
  assert controller.get_plan_execution_state(actor)['cursor'] == 1
finally:
 controller.close()
""")
