import sys
import types
import unittest


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

tiny_locations_stub = types.ModuleType("tiny_locations")
tiny_locations_stub.LocationManager = lambda: None
tiny_locations_stub.PointOfInterest = object

sys.modules["pygame"] = pygame_stub
sys.modules["tiny_locations"] = tiny_locations_stub

from tiny_map_controller import MapController


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


class TestMapControllerBuildingInfo(unittest.TestCase):
    def setUp(self):
        self.controller = MapController.__new__(MapController)
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

    def test_show_target_details_treats_building_objects_as_buildings(self):
        building = FakeBuilding(
            name="Town Hall",
            building_type="civic",
            x=100,
            y=150,
            width=50,
            height=60,
            owner="City Council",
        )
        self.controller.info_panel = types.SimpleNamespace(show=lambda content, pos: setattr(self, "shown_content", content))

        self.controller.show_target_details(building)

        self.assertEqual(self.shown_content["name"], "Town Hall")
        self.assertEqual(self.shown_content["description"], "Village administration building")


if __name__ == "__main__":
    unittest.main()
