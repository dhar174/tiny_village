import importlib
import sys
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


GameplayController = None
_MODULE_PATCHER = None
_ORIGINAL_MODULES = {}


class _DummySurface:
    pass


class _DummyFont:
    def __init__(self, *args, **kwargs):
        pass


def _build_stub_modules():
    fake_pygame = types.ModuleType("pygame")
    fake_pygame.Surface = _DummySurface
    fake_pygame.font = types.SimpleNamespace(Font=_DummyFont)
    fake_pygame.time = types.SimpleNamespace(get_ticks=lambda: 0)
    fake_pygame.display = types.SimpleNamespace(set_mode=lambda *args, **kwargs: None)
    fake_pygame.init = lambda: None
    fake_pygame.error = RuntimeError

    stub_strategy_manager = types.ModuleType("tiny_strategy_manager")
    stub_strategy_manager.StrategyManager = object

    stub_event_handler = types.ModuleType("tiny_event_handler")
    stub_event_handler.EventHandler = object
    stub_event_handler.Event = object

    stub_types = types.ModuleType("tiny_types")
    stub_types.GraphManager = object

    stub_map_controller = types.ModuleType("tiny_map_controller")
    stub_map_controller.MapController = object

    return {
        "pygame": fake_pygame,
        "tiny_strategy_manager": stub_strategy_manager,
        "tiny_event_handler": stub_event_handler,
        "tiny_types": stub_types,
        "tiny_map_controller": stub_map_controller,
    }


def setUpModule():
    global GameplayController, _MODULE_PATCHER, _ORIGINAL_MODULES

    module_names = (
        "pygame",
        "tiny_strategy_manager",
        "tiny_event_handler",
        "tiny_types",
        "tiny_map_controller",
        "tiny_gameplay_controller",
    )
    _ORIGINAL_MODULES = {name: sys.modules.get(name) for name in module_names}
    sys.modules.pop("tiny_gameplay_controller", None)

    _MODULE_PATCHER = patch.dict(sys.modules, _build_stub_modules())
    _MODULE_PATCHER.start()
    GameplayController = importlib.import_module("tiny_gameplay_controller").GameplayController


def tearDownModule():
    if _MODULE_PATCHER is not None:
        _MODULE_PATCHER.stop()

    for module_name, original_module in _ORIGINAL_MODULES.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class RecordingStrategyManager:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def update_strategy(self, events, subject=None):
        self.calls.append({"events": list(events), "subject": subject})
        return self.result


class CharacterBoundAction:
    def __init__(self, name, should_raise=False):
        self.name = name
        self.should_raise = should_raise
        self.calls = []

    def execute(self, *, character=None, graph_manager=None):
        self.calls.append((character, graph_manager))
        if self.should_raise:
            raise RuntimeError(f"{self.name} failed")
        return True


class TestEventHandlerLoopIntegration(unittest.TestCase):
    def _build_character(self, name):
        character = Mock()
        character.name = name
        character.uuid = f"{name.lower()}-uuid"
        character.wealth_money = 10
        character.health_status = 8
        character.add_memory = MagicMock()
        return character

    def _build_controller(self):
        controller = GameplayController.__new__(GameplayController)
        controller.paused = False
        controller.events = []
        controller.characters = {}
        controller.graph_manager = Mock(name="graph_manager")
        controller.graph_manager.get_social_networks.return_value = {"relationships": {}}
        controller.map_controller = Mock()
        controller.map_controller.update = Mock()
        controller.gametime_manager = None
        controller.animation_system = None
        controller.recovery_manager = Mock()
        controller.recovery_manager.attempt_recovery = Mock(return_value=False)
        controller.game_statistics = defaultdict(int)
        controller.action_resolver = None
        controller._update_feature_systems = Mock()
        controller._update_character = Mock(return_value=True)
        controller.event_handler = Mock()
        controller.event_handler.add_event = Mock()
        controller.event_handler.check_events = Mock(return_value=[])
        controller.event_handler.process_events = Mock(
            return_value={"processed_events": [], "failed_events": []}
        )
        controller.event_handler.process_cascading_queue = Mock(return_value=[])
        controller.event_handler.generate_dynamic_events = Mock(return_value=[])
        return controller

    def test_update_game_state_passes_explicit_subject_and_executes_action(self):
        controller = self._build_controller()
        alice = self._build_character("Alice")
        controller.characters = {"alice": alice}

        queued_event = Mock(name="queued_event")
        triggered_event = SimpleNamespace(name="triggered_event")
        controller.events = [queued_event]
        controller.event_handler.check_events.return_value = [triggered_event]

        strategy_action = CharacterBoundAction("EventAction")
        controller.strategy_manager = RecordingStrategyManager([strategy_action])

        controller.update_game_state(0.1)

        self.assertEqual(len(controller.strategy_manager.calls), 1)
        self.assertIs(controller.strategy_manager.calls[0]["subject"], alice)
        self.assertEqual(
            controller.strategy_manager.calls[0]["events"],
            [triggered_event],
        )
        controller.event_handler.add_event.assert_called_once_with(queued_event)
        self.assertEqual(strategy_action.calls, [(alice, controller.graph_manager)])
        alice.add_memory.assert_called_once_with("Performed action: EventAction")
        self.assertEqual(controller.game_statistics["actions_executed"], 1)

    def test_apply_strategy_result_executes_character_mapping_and_continues_after_failure(self):
        controller = self._build_controller()
        alice = self._build_character("Alice")
        bob = self._build_character("Bob")
        controller.characters = {"alice": alice, "bob": bob}

        failing_action = CharacterBoundAction("FailingAction", should_raise=True)
        success_action = CharacterBoundAction("SuccessAction")
        update_errors = []

        controller._apply_strategy_result(
            {"alice": [failing_action], "bob": [success_action]},
            update_errors,
        )

        self.assertEqual(failing_action.calls, [(alice, controller.graph_manager)])
        self.assertEqual(success_action.calls, [(bob, controller.graph_manager)])
        self.assertEqual(controller.game_statistics["actions_failed"], 1)
        self.assertEqual(controller.game_statistics["actions_executed"], 1)
        self.assertTrue(
            any("Decision application failed" in error for error in update_errors)
        )
        bob.add_memory.assert_called_once_with("Performed action: SuccessAction")
        alice.add_memory.assert_not_called()

    def test_apply_decision_without_action_resolver_uses_character_context(self):
        controller = self._build_controller()
        alice = self._build_character("Alice")
        controller.characters = {"alice": alice}

        resolved_action = CharacterBoundAction("ResolvedAction")

        controller.apply_decision(
            {
                "type": "character_action",
                "character_id": "alice",
                "action": resolved_action,
            },
            None,
        )

        self.assertEqual(resolved_action.calls, [(alice, controller.graph_manager)])
        alice.add_memory.assert_called_once_with("Performed action: ResolvedAction")
        self.assertEqual(controller.game_statistics["actions_executed"], 1)
