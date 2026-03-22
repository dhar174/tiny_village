#!/usr/bin/env python3
"""Focused tests for map interactivity using real pygame primitives."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pygame

from tiny_map_controller import ContextMenu, InfoPanel, MapController


class BaseMapControllerTestCase(unittest.TestCase):
    def setUp(self):
        pygame.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error:
            pass

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()
        pygame.quit()

    def create_controller(self, *, width=800, height=600, buildings=None, fill=(34, 139, 34)):
        map_path = os.path.join(self.temp_dir.name, "map.png")
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

    def building_name(self, building):
        if hasattr(building, "name"):
            return building.name
        return building.get("name", "Unknown Building")


class TestInfoPanel(unittest.TestCase):
    def setUp(self):
        self.panel = InfoPanel(10, 10, 200, 150)

    def test_initial_state(self):
        self.assertFalse(self.panel.visible)
        self.assertEqual(self.panel.content, {})

    def test_show_panel(self):
        content = {"name": "Test Building", "type": "House", "size": "50x30"}
        self.panel.show(content, (100, 100))

        self.assertTrue(self.panel.visible)
        self.assertEqual(self.panel.content, content)
        self.assertEqual(self.panel.x, 110)
        self.assertEqual(self.panel.y, 110)

    def test_hide_panel(self):
        self.panel.show({"name": "Test Building"}, (100, 100))
        self.panel.hide()

        self.assertFalse(self.panel.visible)
        self.assertEqual(self.panel.content, {})

    def test_position_boundary_checking(self):
        self.panel.show({"name": "Edge"}, (750, 100))
        self.assertEqual(self.panel.x, 600)

        self.panel.show({"name": "Edge"}, (100, 550))
        self.assertEqual(self.panel.y, 450)


class TestContextMenu(unittest.TestCase):
    def setUp(self):
        self.menu = ContextMenu()

    def test_initial_state(self):
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.options, [])
        self.assertEqual(self.menu.selected_option, -1)

    def test_show_and_hide_menu(self):
        options = [
            {"label": "Enter Building", "action": "enter"},
            {"label": "View Details", "action": "details"},
        ]
        self.menu.show(options, (100, 100), Mock())
        self.assertTrue(self.menu.visible)
        self.assertEqual(self.menu.height, 60)

        self.menu.hide()
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.options, [])
        self.assertIsNone(self.menu.target_object)

    def test_mouse_motion_and_click_handling(self):
        options = [
            {"label": "Option 1", "action": "one"},
            {"label": "Option 2", "action": "two"},
        ]
        self.menu.show(options, (100, 100), Mock())

        self.menu.handle_mouse_motion((125, 110))
        self.assertEqual(self.menu.selected_option, 0)

        self.menu.handle_mouse_motion((125, 135))
        self.assertEqual(self.menu.selected_option, 1)

        selected = self.menu.handle_click((125, 110))
        self.assertEqual(selected, options[0])
        self.assertFalse(self.menu.visible)


class TestMapControllerInteractivity(BaseMapControllerTestCase):
    def setUp(self):
        super().setUp()
        self.map_data = {
            "width": 800,
            "height": 600,
            "buildings": [
                {
                    "name": "Town Hall",
                    "type": "government",
                    "rect": pygame.Rect(100, 100, 50, 50),
                },
                {
                    "name": "General Store",
                    "type": "shop",
                    "rect": pygame.Rect(200, 150, 40, 30),
                },
            ],
        }
        self.controller = self.create_controller(
            buildings=self.map_data["buildings"],
        )

    def test_building_detection_prefers_real_building_objects(self):
        building = self.controller.is_building((125, 125))

        self.assertIsNotNone(building)
        self.assertEqual(self.building_name(building), "Town Hall")
        self.assertTrue(hasattr(building, "get_location"))

        self.assertIsNone(self.controller.is_building((500, 500)))

    def test_building_info_generation_uses_real_building_geometry(self):
        building = self.controller.is_building((125, 125))
        info = self.controller.get_building_info(building)

        self.assertEqual(info["name"], "Town Hall")
        self.assertEqual(info["type"], "Government")
        self.assertEqual(info["position"], "(100, 100)")
        self.assertEqual(info["size"], "50 x 50")
        self.assertEqual(info["area"], 2500)

    def test_character_info_generation(self):
        mock_character = Mock()
        mock_character.name = "John Doe"
        mock_character.position = pygame.math.Vector2(200, 300)
        mock_character.energy = 75
        mock_character.health = 90
        mock_character.mood = "Happy"

        info = self.controller.get_character_info(mock_character)

        self.assertEqual(info["name"], "John Doe")
        self.assertEqual(info["type"], "Character")
        self.assertEqual(info["position"], "(200, 300)")
        self.assertEqual(info["energy"], 75)
        self.assertEqual(info["health"], 90)
        self.assertEqual(info["mood"], "Happy")

    def test_building_context_menu_options_follow_real_building_type(self):
        general_building = self.controller.is_building((125, 125))
        with patch.object(self.controller.context_menu, "show") as show_menu:
            self.controller.show_building_context_menu(general_building, (100, 100))
        option_labels = [opt["label"] for opt in show_menu.call_args.args[0]]
        self.assertIn("Enter Building", option_labels)
        self.assertIn("View Details", option_labels)
        self.assertIn("Get Directions", option_labels)

        shop_building = self.controller.is_building((210, 160))
        with patch.object(self.controller.context_menu, "show") as show_menu:
            self.controller.show_building_context_menu(shop_building, (210, 160))
        option_labels = [opt["label"] for opt in show_menu.call_args.args[0]]
        self.assertIn("Browse Items", option_labels)

    def test_ui_element_hiding(self):
        self.controller.info_panel.show({"name": "Test"}, (100, 100))
        self.controller.context_menu.show(
            [{"label": "Test", "action": "test"}],
            (100, 100),
            Mock(),
        )

        self.controller.hide_ui_elements()

        self.assertFalse(self.controller.info_panel.visible)
        self.assertFalse(self.controller.context_menu.visible)

    def test_selection_clearing_resets_all_state(self):
        self.controller.selected_character = Mock()
        self.controller.selected_building = Mock()
        self.controller.selected_location = Mock()
        self.controller.selected_poi = Mock()
        self.controller.show_location_info = True

        self.controller.clear_selections()

        self.assertIsNone(self.controller.selected_character)
        self.assertIsNone(self.controller.selected_building)
        self.assertIsNone(self.controller.selected_location)
        self.assertIsNone(self.controller.selected_poi)
        self.assertFalse(self.controller.show_location_info)


class TestActionExecution(BaseMapControllerTestCase):
    def setUp(self):
        super().setUp()
        self.controller = self.create_controller(
            buildings=[
                {
                    "name": "Test Building",
                    "type": "house",
                    "rect": pygame.Rect(100, 100, 50, 50),
                }
            ],
        )

    def test_enter_building_action_accepts_context_menu_target(self):
        building = self.controller.is_building((125, 125))
        option = {"action": "enter", "target": building}

        with patch("builtins.print") as mock_print:
            self.controller.execute_context_action(option)

        printed_messages = [call.args[0] for call in mock_print.call_args_list]
        self.assertTrue(any("Entering Test Building" in message for message in printed_messages))

    def test_details_action(self):
        building = self.controller.is_building((125, 125))
        option = {"action": "details", "target": building}

        with patch.object(self.controller.info_panel, "show") as show_panel:
            self.controller.execute_context_action(option)

        show_panel.assert_called_once()

    def test_unknown_action(self):
        option = {"action": "unknown_action", "target": Mock()}

        with patch("builtins.print") as mock_print:
            self.controller.execute_context_action(option)

        mock_print.assert_called_with("Unknown action: unknown_action")


if __name__ == "__main__":
    unittest.main(verbosity=2)
