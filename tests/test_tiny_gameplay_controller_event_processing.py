import importlib
import sys
import types
import unittest
from types import SimpleNamespace


_ORIGINAL_MODULE_SNAPSHOTS = {}
GameplayController = None


def setUpModule():
    global GameplayController

    stub_modules = {
        "tiny_strategy_manager": types.ModuleType("tiny_strategy_manager"),
        "tiny_event_handler": types.ModuleType("tiny_event_handler"),
        "tiny_types": types.ModuleType("tiny_types"),
        "tiny_map_controller": types.ModuleType("tiny_map_controller"),
    }

    stub_modules["tiny_strategy_manager"].StrategyManager = type("StrategyManager", (), {})
    stub_modules["tiny_event_handler"].EventHandler = type("EventHandler", (), {})
    stub_modules["tiny_event_handler"].Event = type("Event", (), {})
    stub_modules["tiny_types"].GraphManager = type("GraphManager", (), {})
    stub_modules["tiny_map_controller"].MapController = type("MapController", (), {})

    for module_name, module in stub_modules.items():
        _ORIGINAL_MODULE_SNAPSHOTS[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module

    _ORIGINAL_MODULE_SNAPSHOTS["tiny_gameplay_controller"] = sys.modules.get(
        "tiny_gameplay_controller"
    )
    sys.modules.pop("tiny_gameplay_controller", None)

    GameplayController = importlib.import_module("tiny_gameplay_controller").GameplayController


def tearDownModule():
    for module_name, original_module in _ORIGINAL_MODULE_SNAPSHOTS.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class TestGameplayControllerEventProcessing(unittest.TestCase):
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
            {"processed_events": normalized_events},
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

    def test_apply_event_consequences_tracks_new_event_type_without_preseeded_counter(self):
        update_errors = []

        GameplayController._apply_event_consequences(
            self.controller,
            [{"name": "harvest-fair", "_normalized_event_type": "festival", "_normalized_impact": 1}],
            {"processed_events": []},
            update_errors,
        )

        self.assertEqual(update_errors, [])
        self.assertEqual(self.controller.game_statistics["events_festival"], 1)
        self.assertNotIn("festival", self.controller.game_state)
