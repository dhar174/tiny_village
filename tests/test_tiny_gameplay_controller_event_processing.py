import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


_ORIGINAL_MODULE_SNAPSHOTS = {}
GameplayController = None


def setUpModule():
    global GameplayController

    stub_modules = {
        "pygame": types.ModuleType("pygame"),
        "tiny_strategy_manager": types.ModuleType("tiny_strategy_manager"),
        "tiny_event_handler": types.ModuleType("tiny_event_handler"),
        "tiny_types": types.ModuleType("tiny_types"),
        "tiny_map_controller": types.ModuleType("tiny_map_controller"),
    }

    stub_modules["pygame"].Surface = type("Surface", (), {})
    stub_modules["pygame"].font = types.SimpleNamespace(Font=type("Font", (), {}))
    stub_modules["pygame"].math = types.SimpleNamespace(Vector2=type("Vector2", (), {}))
    stub_modules["pygame"].Rect = type("Rect", (), {})
    stub_modules["pygame"].SRCALPHA = 0
    stub_modules["tiny_strategy_manager"].StrategyManager = type("StrategyManager", (), {})
    stub_modules["tiny_event_handler"].EventHandler = type("EventHandler", (), {})
    stub_modules["tiny_event_handler"].Event = type("Event", (), {})
    stub_modules["tiny_types"].GraphManager = type("GraphManager", (), {})
    stub_modules["tiny_map_controller"].MapController = type("MapController", (), {})

    _ORIGINAL_MODULE_SNAPSHOTS["tiny_gameplay_controller"] = sys.modules.get(
        "tiny_gameplay_controller"
    )
    for module_name, module in stub_modules.items():
        _ORIGINAL_MODULE_SNAPSHOTS[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module
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

    def test_process_events_and_drive_strategy_passes_highest_priority_last(self):
        update_errors = []
        high_priority_event = {
            "name": "Market Crash",
            "type": " Economic ",
            "importance": 9,
            "impact": -8,
        }
        low_priority_event = {
            "name": "Town Gossip",
            "type": "social",
            "importance": 2,
            "impact": 1,
        }
        strategy_manager = SimpleNamespace(update_strategy=MagicMock(return_value=None))
        event_handler = SimpleNamespace(
            check_events=MagicMock(return_value=[low_priority_event, high_priority_event]),
            process_events=MagicMock(
                return_value={
                    "processed_events": ["Market Crash", "Town Gossip"],
                    "failed_events": [],
                }
            ),
        )
        apply_strategy_result = MagicMock()
        handle_cascading_events = MagicMock()
        controller = SimpleNamespace(
            event_handler=event_handler,
            events=[],
            strategy_manager=strategy_manager,
            game_statistics={
                "actions_executed": 0,
                "actions_failed": 0,
                "characters_created": 0,
                "errors_recovered": 0,
            },
            game_state={},
            _process_basic_events_fallback=MagicMock(),
            _normalize_and_prioritize_events=lambda events, errors: GameplayController._normalize_and_prioritize_events(
                controller, events, errors
            ),
            _apply_event_consequences=lambda events, event_results, errors: GameplayController._apply_event_consequences(
                controller, events, event_results, errors
            ),
            _apply_strategy_result=apply_strategy_result,
            _handle_cascading_and_dynamic_events=handle_cascading_events,
        )

        GameplayController._process_events_and_drive_strategy(controller, update_errors)

        strategy_events = strategy_manager.update_strategy.call_args.args[0]
        self.assertEqual(
            [event["name"] for event in strategy_events],
            ["Town Gossip", "Market Crash"],
        )
        apply_strategy_result.assert_called_once_with(None, update_errors)
        handle_cascading_events.assert_called_once_with(
            strategy_manager.update_strategy.call_args.args[0][::-1],
            update_errors,
        )
