import importlib
import sys
import types
import unittest
from unittest.mock import patch


class MockRect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left = x
        self.top = y
        self.right = x + width
        self.bottom = y + height
        self.centerx = x + width // 2
        self.centery = y + height // 2

    def collidepoint(self, pos):
        px, py = pos
        return self.x <= px <= self.right and self.y <= py <= self.bottom


def build_pygame_stub():
    pygame_stub = types.ModuleType("pygame")
    pygame_stub.Rect = MockRect
    pygame_stub.error = Exception
    pygame_stub.image = types.SimpleNamespace(load=lambda path: object())
    pygame_stub.font = types.SimpleNamespace(
        Font=lambda *args, **kwargs: object(),
        SysFont=lambda *args, **kwargs: object(),
    )
    pygame_stub.mouse = types.SimpleNamespace(get_pos=lambda: (25, 35))
    pygame_stub.draw = types.SimpleNamespace(rect=lambda *args, **kwargs: None)
    pygame_stub.Surface = lambda *args, **kwargs: object()
    return pygame_stub


def build_tiny_locations_stub():
    tiny_locations_stub = types.ModuleType("tiny_locations")
    tiny_locations_stub.LocationManager = lambda: None
    tiny_locations_stub.PointOfInterest = object
    return tiny_locations_stub


class FakeLocation:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class FakeBuilding:
    def __init__(self, name, building_type, x, y, width, height, owner=None):
        self.name = name
        self.building_type = building_type
        self.coordinates_location = (x, y)
        self.width = width
        self.height = height
        self.length = height
        self.owner = owner
        self.location = FakeLocation(x, y, width, height)

    def get_location(self):
        return self.location


class FakeCharacter:
    def __init__(self):
        self.name = "Jordan"
        self.position = types.SimpleNamespace(x=10, y=20)
        self.energy = 80
        self.coordinates_location = (10, 20)

    def get_location(self):
        return FakeLocation(10, 20, 1, 1)


class TestMapControllerBuildingInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_patcher = patch.dict(
            sys.modules,
            {
                "pygame": build_pygame_stub(),
                "tiny_locations": build_tiny_locations_stub(),
            },
        )
        cls._module_patcher.start()
        cls._original_map_controller_module = sys.modules.pop("tiny_map_controller", None)
        cls._map_module = importlib.import_module("tiny_map_controller")
        cls.MapController = cls._map_module.MapController

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("tiny_map_controller", None)
        if cls._original_map_controller_module is not None:
            sys.modules["tiny_map_controller"] = cls._original_map_controller_module
        cls._module_patcher.stop()

    def setUp(self):
        self.controller = self.MapController.__new__(self.MapController)
        self.controller.map_data = {
            "buildings": [
                {
                    "name": "Town Hall",
                    "type": "civic",
                    "rect": MockRect(100, 150, 50, 60),
                    "capacity": 50,
                    "owner": "City Council",
                    "value": 1200,
                    "description": "Village administration building",
                }
            ]
        }

    def test_get_building_info_uses_original_map_data_for_building_objects(self):
        building = FakeBuilding(
            name="Town Hall",
            building_type="civic",
            x=100,
            y=150,
            width=50,
            height=60,
            owner="City Council",
        )

        info = self.controller.get_building_info(building)

        self.assertEqual(info["name"], "Town Hall")
        self.assertEqual(info["type"], "civic")
        self.assertEqual(info["position"], "(100, 150)")
        self.assertEqual(info["size"], "50 x 60")
        self.assertEqual(info["area"], 3000)
        self.assertEqual(info["capacity"], 50)
        self.assertEqual(info["owner"], "City Council")
        self.assertEqual(info["value"], 1200)
        self.assertEqual(info["description"], "Village administration building")

    def test_show_target_details_with_character_object(self):
        character = FakeCharacter()

        def mock_show(content, _pos):
            self.shown_content = content

        self.controller.info_panel = types.SimpleNamespace(show=mock_show)

        self.controller.show_target_details(character)

        self.assertEqual(self.shown_content["name"], "Jordan")
        self.assertEqual(self.shown_content["type"], "Character")
        self.assertEqual(self.shown_content["position"], "(10, 20)")
        self.assertEqual(self.shown_content["energy"], 80)

    def test_execute_context_action_uses_building_target_entry_path(self):
        building = FakeBuilding(
            name="Town Hall",
            building_type="civic",
            x=100,
            y=150,
            width=50,
            height=60,
            owner="City Council",
        )
        entry_call_record = {}

        def record_entry(target):
            entry_call_record["target"] = target

        self.controller.enter_building_target = record_entry
        self.controller.enter_building = lambda position: self.fail(
            "execute_context_action should not call the position-based enter_building path"
        )

        self.controller.execute_context_action({"action": "enter", "target": building})

        self.assertIs(entry_call_record["target"], building)

    def test_enter_building_prints_entry_message_once(self):
        building = FakeBuilding(
            name="Town Hall",
            building_type="civic",
            x=100,
            y=150,
            width=50,
            height=60,
            owner="City Council",
        )

        with patch("builtins.print") as mock_print:
            self.controller.enter_building(building)

        mock_print.assert_called_once_with("Entering Town Hall")


if __name__ == "__main__":
    unittest.main()
