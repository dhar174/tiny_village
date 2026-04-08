#!/usr/bin/env python3
"""Integration tests for map interactivity using real pygame behavior."""

from unittest.mock import Mock, patch

import pygame

from tiny_map_controller import MapController


def ensure_pygame():
    pygame.init()
    try:
        pygame.display.set_mode((1, 1))
    except pygame.error:
        pass


def create_controller(tmp_path, *, width=800, height=600, buildings=None, fill=(34, 139, 34)):
    ensure_pygame()
    map_path = tmp_path / "integration-map.png"
    surface = pygame.Surface((width, height))
    surface.fill(fill)
    pygame.image.save(surface, str(map_path))
    controller = MapController(
        str(map_path),
        {
            "width": width,
            "height": height,
            "buildings": buildings or [],
        },
    )
    return controller


def building_name(building):
    if hasattr(building, "name"):
        return building.name
    return building.get("name", "Unknown Building")


def test_complete_interaction_flow(tmp_path):
    map_data = {
        "width": 800,
        "height": 600,
        "buildings": [
            {
                "name": "Village Inn",
                "type": "social",
                "rect": pygame.Rect(100, 100, 60, 40),
                "owner": "Martha",
                "capacity": 25,
                "description": "A cozy inn with warm fires",
            },
            {
                "name": "Weapon Shop",
                "type": "shop",
                "rect": pygame.Rect(300, 200, 50, 50),
                "owner": "Blacksmith Joe",
            },
        ],
    }
    controller = create_controller(tmp_path, buildings=map_data["buildings"])

    mock_character = Mock()
    mock_character.name = "Adventurer Alice"
    mock_character.position = pygame.math.Vector2(250, 150)
    mock_character.energy = 80
    mock_character.health = 95
    mock_character.mood = "Excited"
    mock_character.job = "Explorer"
    mock_character.color = (0, 255, 0)
    controller.characters = {"alice": mock_character}

    inn = controller.is_building((130, 120))
    weapon_shop = controller.is_building((325, 225))

    controller.select_building(inn, (130, 120))
    assert controller.selected_building == inn
    assert controller.info_panel.visible is True
    assert controller.info_panel.content["name"] == "Village Inn"

    controller.show_building_context_menu(weapon_shop, (325, 225))
    option_labels = [opt["label"] for opt in controller.context_menu.options]
    assert "Browse Items" in option_labels
    assert "Enter Building" in option_labels

    char_info = controller.get_character_info(mock_character)
    for field in ["name", "position", "energy", "health", "mood", "job"]:
        assert field in char_info

    with patch("builtins.print") as mock_print:
        controller.execute_context_action({"action": "enter", "target": inn})
        controller.execute_context_action({"action": "browse", "target": weapon_shop})
        controller.execute_context_action({"action": "talk", "target": mock_character})

    printed = [call.args[0] for call in mock_print.call_args_list]
    assert any("Entering Village Inn" in line for line in printed)
    assert any("Browsing items in Weapon Shop" in line for line in printed)
    assert any("Starting conversation with Adventurer Alice" in line for line in printed)

    controller.info_panel.show({"name": "Edge Test"}, (750, 550))
    assert controller.info_panel.x <= 800 - controller.info_panel.width
    assert controller.info_panel.y <= 600 - controller.info_panel.height

    controller.hide_ui_elements()
    assert controller.info_panel.visible is False
    assert controller.context_menu.visible is False


def test_backwards_compatibility_supports_legacy_building_targets(tmp_path):
    legacy_building = {
        "name": "Test Building",
        "type": "house",
        "rect": pygame.Rect(100, 100, 50, 50),
    }
    controller = create_controller(tmp_path, buildings=[legacy_building])

    detected_building = controller.is_building((125, 125))
    assert detected_building is not None
    assert building_name(detected_building) == "Test Building"

    info = controller.get_building_info(legacy_building)
    assert info["name"] == "Test Building"
    assert info["position"] == "(100, 100)"

    with patch("builtins.print") as mock_print:
        controller.execute_context_action({"action": "enter", "target": legacy_building})

    printed = [call.args[0] for call in mock_print.call_args_list]
    assert any("Entering Test Building" in line for line in printed)


def test_error_handling(tmp_path):
    controller = create_controller(tmp_path, buildings=[])

    info = controller.get_building_info({})
    assert info["name"] == "Unknown Building"

    minimal_char = Mock()
    minimal_char.name = "Test"
    minimal_char.position = pygame.math.Vector2(0, 0)
    char_info = controller.get_character_info(minimal_char)
    assert char_info["name"] == "Test"

    with patch("builtins.print") as mock_print:
        controller.execute_context_action({"action": "unknown", "target": None})

    mock_print.assert_called_with("Unknown action: unknown")
