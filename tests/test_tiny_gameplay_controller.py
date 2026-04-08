import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pygame

from tiny_gameplay_controller import GameplayController, MAX_SPEED, MIN_SPEED, SPEED_STEP
from tiny_map_controller import MapController


def make_config():
    return {
        "screen_width": 800,
        "screen_height": 600,
        "map": {
            "image_path": "assets/default_map.png",
            "width": 100,
            "height": 100,
            "buildings_file": None,
        },
        "characters": {"count": 0},
        "key_bindings": {
            "increase_speed": [pygame.K_PAGEUP],
            "decrease_speed": [pygame.K_PAGEDOWN],
            "minimap": [pygame.K_m],
            "overview": [pygame.K_o],
        },
    }


class MockCharacter:
    def __init__(self, name="Test Char"):
        self.name = name
        self.uuid = f"{name}_uuid"
        self.energy = 100
        self.add_memory = MagicMock()


class LegacyUIController(GameplayController):
    def _init_ui_system(self):
        self.ui_panels = {}
        self.ui_fonts = {
            "normal": pygame.font.Font(None, 24),
            "small": pygame.font.Font(None, 18),
            "tiny": pygame.font.Font(None, 16),
        }


class TestGameplayController(unittest.TestCase):
    def setUp(self):
        pygame.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error:
            pass

        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_graph_manager = MagicMock()
        self.controller = GameplayController(
            graph_manager=self.mock_graph_manager,
            config=make_config(),
        )
        self.controller.action_resolver = MagicMock()
        self.mock_action = MagicMock()
        self.mock_action.name = "TestAction"
        self.mock_character = MockCharacter()

        if not self.controller.screen:
            self.controller.screen = pygame.Surface((800, 600))

    def tearDown(self):
        self.temp_dir.cleanup()
        pygame.quit()

    def create_map_controller(self, *, width=100, height=100, buildings=None, fill=(0, 128, 0)):
        map_path = os.path.join(self.temp_dir.name, "test_map.png")
        surface = pygame.Surface((width, height))
        surface.fill(fill)
        pygame.image.save(surface, map_path)
        return MapController(
            map_path,
            {
                "width": width,
                "height": height,
                "buildings": buildings or [],
            },
        )

    def test_execute_single_action_resolves_and_executes_action(self):
        self.controller.action_resolver.resolve_action.return_value = self.mock_action
        self.mock_action.execute = MagicMock(return_value=True)

        with patch.object(
            self.controller,
            "_update_character_state_after_action",
        ) as update_state:
            result = self.controller._execute_single_action(
                self.mock_character,
                {"name": "TestActionData"},
            )

        self.assertTrue(result)
        self.controller.action_resolver.resolve_action.assert_called_once_with(
            {"name": "TestActionData"},
            self.mock_character,
        )
        self.mock_action.execute.assert_called_once_with(
            character=self.mock_character,
            graph_manager=self.controller.graph_manager,
        )
        update_state.assert_called_once_with(self.mock_character, self.mock_action)

    def test_execute_single_action_uses_legacy_execute_signature_when_needed(self):
        class LegacyAction:
            def __init__(self):
                self.name = "LegacyAction"
                self.calls = []

            def execute(self, target=None, initiator=None):
                self.calls.append((target, initiator))
                return True

        legacy_action = LegacyAction()
        self.controller.action_resolver.resolve_action.return_value = legacy_action

        with patch.object(
            self.controller,
            "_update_character_state_after_action",
        ) as update_state:
            result = self.controller._execute_single_action(
                self.mock_character,
                {"name": "LegacyActionData"},
            )

        self.assertTrue(result)
        self.assertEqual(legacy_action.calls, [(self.mock_character, self.mock_character)])
        update_state.assert_called_once_with(self.mock_character, legacy_action)

    def test_execute_single_action_does_not_retry_internal_type_errors(self):
        execute_calls = []

        def raising_execute(*, character=None, graph_manager=None):
            execute_calls.append((character, graph_manager))
            raise TypeError("action body failure")

        self.controller.action_resolver.resolve_action.return_value = self.mock_action
        self.mock_action.execute = MagicMock(side_effect=raising_execute)

        with self.assertLogs("tiny_gameplay_controller", level="WARNING") as captured_logs, patch.object(
            self.controller,
            "_update_character_state_after_action",
        ) as update_state:
            result = self.controller._execute_single_action(
                self.mock_character,
                {"name": "BrokenActionData"},
            )

        self.assertFalse(result)
        self.assertEqual(
            execute_calls,
            [(self.mock_character, self.controller.graph_manager)],
        )
        self.mock_action.execute.assert_called_once_with(
            character=self.mock_character,
            graph_manager=self.controller.graph_manager,
        )
        update_state.assert_not_called()
        self.assertTrue(
            any("action body failure" in message for message in captured_logs.output)
        )

    def test_update_character_state_records_memory_without_graph_update(self):
        specific_graph_manager = MagicMock()
        self.controller.graph_manager = specific_graph_manager

        result = self.controller._update_character_state_after_action(
            self.mock_character,
            self.mock_action,
        )

        self.assertTrue(result)
        self.mock_character.add_memory.assert_called_once_with(
            "Performed action: TestAction"
        )
        specific_graph_manager.update_character_state.assert_not_called()

    def test_speed_text_caching(self):
        self.controller._render_ui()
        initial_cached_surface = self.controller._cached_speed_text
        self.assertIsNotNone(initial_cached_surface)

        self.controller._render_ui()
        self.assertIs(initial_cached_surface, self.controller._cached_speed_text)

        original_speed = self.controller.time_scale_factor
        new_speed = original_speed + SPEED_STEP
        if new_speed > MAX_SPEED:
            new_speed = max(MIN_SPEED, original_speed - SPEED_STEP)

        self.controller.time_scale_factor = new_speed
        self.controller._render_ui()
        self.assertIsNot(initial_cached_surface, self.controller._cached_speed_text)

    def test_modular_ui_system_initialization(self):
        self.assertTrue(hasattr(self.controller, "ui_panels"))
        self.assertTrue(hasattr(self.controller, "ui_fonts"))
        self.assertIn("character_info", self.controller.ui_panels)
        self.assertIn("village_overview", self.controller.ui_panels)
        self.assertIn("normal", self.controller.ui_fonts)
        self.assertIn("small", self.controller.ui_fonts)
        self.assertIn("tiny", self.controller.ui_fonts)

    def test_render_ui_with_modular_system_draws_content(self):
        self.assertTrue(self.controller.initialize_modular_ui_system())
        character_info_panel = self.controller.ui_panels["character_info"]
        weather_panel = self.controller.ui_panels["weather"]

        render_surface = pygame.Surface((120, 120))
        before_panel_render = pygame.image.tostring(render_surface, "RGB")
        height = character_info_panel.render(
            render_surface,
            self.controller,
            self.controller.ui_fonts,
        )
        after_panel_render = pygame.image.tostring(render_surface, "RGB")

        self.assertIsInstance(height, int)
        self.assertGreaterEqual(height, 0)
        self.assertNotEqual(before_panel_render, after_panel_render)

        original_visibility = {
            name: panel.visible for name, panel in self.controller.ui_panels.items()
        }
        try:
            for panel in self.controller.ui_panels.values():
                panel.visible = False

            character_info_panel.visible = True
            weather_panel.visible = False
            self.controller.screen.fill((0, 0, 0))
            before_visible_ui = pygame.image.tostring(self.controller.screen, "RGB")
            self.controller._render_ui()
            after_visible_ui = pygame.image.tostring(self.controller.screen, "RGB")
            self.assertNotEqual(before_visible_ui, after_visible_ui)
        finally:
            for name, visible in original_visibility.items():
                self.controller.ui_panels[name].visible = visible

    def test_render_ui_uses_legacy_fallback_when_panels_are_unavailable(self):
        legacy_controller = LegacyUIController(
            graph_manager=self.mock_graph_manager,
            config=make_config(),
        )
        if not legacy_controller.screen:
            legacy_controller.screen = pygame.Surface((800, 600))

        with patch.object(legacy_controller, "_render_legacy_ui") as render_legacy:
            legacy_controller._render_ui()

        render_legacy.assert_called_once()

    def test_render_ui_uses_minimal_fallback_when_modular_render_raises(self):
        with patch.object(
            self.controller,
            "_render_modular_ui",
            side_effect=RuntimeError("boom"),
        ), patch.object(self.controller, "_render_minimal_ui") as render_minimal:
            self.controller._render_ui()

        render_minimal.assert_called_once()

    def test_speed_text_cache_invalidation_via_handle_keydown(self):
        self.controller._render_ui()
        initial_cached_surface = self.controller._cached_speed_text
        increase_key = self.controller.config["key_bindings"]["increase_speed"][0]
        event = pygame.event.Event(pygame.KEYDOWN, key=increase_key)
        original_speed = self.controller.time_scale_factor

        self.controller._handle_keydown(event)

        self.assertNotEqual(original_speed, self.controller.time_scale_factor)
        self.controller._render_ui()
        self.assertIsNot(initial_cached_surface, self.controller._cached_speed_text)

    def test_minimap_toggle(self):
        self.assertFalse(getattr(self.controller, "_minimap_mode", False))
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m)

        self.controller._handle_keydown(event)
        self.assertTrue(getattr(self.controller, "_minimap_mode", False))

        self.controller._handle_keydown(event)
        self.assertFalse(getattr(self.controller, "_minimap_mode", False))

    def test_overview_mode_toggle(self):
        self.assertFalse(getattr(self.controller, "_overview_mode", False))
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o)

        self.controller._handle_keydown(event)
        self.assertTrue(getattr(self.controller, "_overview_mode", False))

        self.controller._handle_keydown(event)
        self.assertFalse(getattr(self.controller, "_overview_mode", False))

    def test_render_minimap_draws_overlay_with_real_map_controller(self):
        real_map_controller = self.create_map_controller(
            buildings=[
                {"name": "Test Building", "type": "shop", "rect": pygame.Rect(25, 25, 20, 20)}
            ]
        )
        self.controller.map_controller = real_map_controller

        self.controller.screen.fill((0, 0, 0))
        before = pygame.image.tostring(self.controller.screen, "RGB")
        self.controller._render_minimap()
        after = pygame.image.tostring(self.controller.screen, "RGB")

        self.assertNotEqual(before, after)

    def test_render_minimap_caches_scaled_map(self):
        real_map_controller = self.create_map_controller(
            buildings=[
                {"name": "Town Hall", "type": "government", "rect": pygame.Rect(20, 20, 20, 20)},
            ]
        )
        self.controller.map_controller = real_map_controller

        with patch(
            "tiny_gameplay_controller.pygame.transform.smoothscale",
            wraps=pygame.transform.smoothscale,
        ) as smoothscale:
            self.controller._render_minimap()
            self.controller._render_minimap()

        self.assertEqual(smoothscale.call_count, 1)

    def test_render_minimap_handles_building_objects(self):
        class ObjectBackedBuilding:
            def __init__(self, rect):
                self._rect = rect

            def get_location(self):
                return self._rect

        real_map_controller = self.create_map_controller(
            buildings=[
                {"name": "Town Hall", "type": "government", "rect": pygame.Rect(20, 20, 20, 20)},
            ]
        )
        real_map_controller.map_data["buildings"] = [
            {"name": "Town Hall", "type": "government", "rect": pygame.Rect(20, 20, 20, 20)},
            ObjectBackedBuilding(pygame.Rect(60, 60, 15, 15)),
            object(),
        ]
        self.controller.map_controller = real_map_controller

        self.controller.screen.fill((0, 0, 0))
        before = pygame.image.tostring(self.controller.screen, "RGB")
        self.controller._render_minimap()
        after = pygame.image.tostring(self.controller.screen, "RGB")

        self.assertNotEqual(before, after)

    def test_render_overview_draws_summary_with_real_map_controller(self):
        real_map_controller = self.create_map_controller(
            buildings=[
                {"name": "Town Hall", "type": "government", "rect": pygame.Rect(30, 30, 25, 25)},
                {"name": "Market", "type": "shop", "rect": pygame.Rect(60, 60, 15, 15)},
            ],
            fill=(0, 100, 200),
        )
        self.controller.map_controller = real_map_controller

        self.controller.screen.fill((0, 0, 0))
        before = pygame.image.tostring(self.controller.screen, "RGB")
        self.controller._render_overview()
        after = pygame.image.tostring(self.controller.screen, "RGB")

        self.assertNotEqual(before, after)

    def test_render_overview_caches_scaled_map_between_frames(self):
        real_map_controller = self.create_map_controller(fill=(0, 100, 200))
        self.controller.map_controller = real_map_controller

        with patch.object(self.controller, "_render_minimap"), patch(
            "tiny_gameplay_controller.pygame.transform.smoothscale",
            wraps=pygame.transform.smoothscale,
        ) as smoothscale:
            self.controller._render_overview()
            self.controller._render_overview()

        self.assertEqual(smoothscale.call_count, 1)

    def test_render_dispatches_to_overview_mode(self):
        with patch.object(self.controller, "_render_overview") as render_overview:
            self.controller._overview_mode = True
            self.controller.render()

        render_overview.assert_called_once()

    def test_render_dispatches_to_minimap_overlay_in_normal_mode(self):
        self.controller.map_controller = self.create_map_controller()
        self.controller._minimap_mode = True

        with patch.object(self.controller, "_render_minimap") as render_minimap:
            self.controller.render()

        render_minimap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
