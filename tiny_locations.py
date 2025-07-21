import importlib
import logging
from math import cos
import uu
import uuid


# from actions import Action, State, ActionSystem
from tiny_types import Action, State, ActionSystem

effect_dict = {
    "Enter Location Boundary": [
        {
            "targets": ["initiator"],
            "method": "play_animation",
            "method_args": ["walking"],
        },
        {
            "targets": ["initiator"],
            "method": "walk_to",
            "method_args": ["walking"],
        },
    ],
}

preconditions_dict = {
    "Enter Location Boundary": [
        {
            "name": "energy",
            "attribute": "energy",
            "target": "initiator",
            "satisfy_value": 10,
            "operator": "gt",
        },
        {
            "name": "extraversion",
            "attribute": "personality_traits.extraversion",
            "target": "initiator",
            "satisfy_value": 50,
            "operator": "gt",
        },
    ]
}


class Location:
    def __init__(
        self,
        name,
        x,
        y,
        width,
        height,
        action_system: ActionSystem,
        security=0,
        threat_level=0,
        popularity=0,
    ):
        ActionSystem = importlib.import_module("actions").ActionSystem
        Action = importlib.import_module("actions").Action
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.coordinates_location = (x, y)
        self.effect_dict = {
            "Enter Location Boundary": [
                {
                    "targets": ["initiator"],
                    "method": "play_animation",
                    "method_args": ["walking"],
                },
            ],
        }
        self.possible_interactions = [
            Action(
                "Enter Location Boundary",
                action_system.instantiate_conditions(
                    preconditions_dict["Enter Location Boundary"]
                ),
                effect_dict["Enter Location Boundary"],
                cost=0,
            )
        ]
        self.security = 0
        self.threat_level = 0
        self.popularity = 0
        self.security = security
        self.threat_level = threat_level
        self.popularity = popularity
        self.activities_available = []
        self.accessible = True
        self.current_visitors = []
        self.uuid = uuid.uuid4()
        self.visit_count = 0

    def add_activity(self, activity):
        self.activities_available.append(activity)

    def get_possible_interactions(self, requester):
        self.effect_dict["Enter Location Boundary"].append(
            {
                "targets": ["initiator"],
                "method": "walk_to",
                "method_args": [
                    self.point_of_edge_nearest_to_point(
                        *requester.location.coordinates_location
                    )
                ],
            }
        )
        return self.possible_interactions

    def get_coordinates(self):
        return self.coordinates_location

    def get_dimensions(self):
        return self.width, self.height

    def get_area(self):
        if self.width == 0 or self.height == 0:
            return 0
        return self.width * self.height

    def get_center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    def get_diagonal(self):
        if self.width == 0 or self.height == 0:
            return 0
        return (self.width**2 + self.height**2) ** 0.5

    def get_aspect_ratio(self):
        if self.height == 0:
            return 0
        return self.width / self.height

    def get_perimeter(self):
        if self.width == 0 or self.height == 0:
            return 0
        return 2 * (self.width + self.height)

    def get_bounding_box(self):
        if self.width == 0 or self.height == 0:
            return self.x, self.y, self.x, self.y
        return self.x, self.y, self.x + self.width, self.y + self.height

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_x(self, x):
        self.x = x
        self.coordinates_location = (x, self.y)

    def set_y(self, y):
        self.y = y
        self.coordinates_location = (self.x, y)

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_coordinates(self, x, y):
        self.x = x
        self.y = y
        self.coordinates_location = (x, y)

    def contains_point(self, point_x, point_y):
        return (self.x <= point_x < self.x + self.width) and (
            self.y <= point_y < self.y + self.height
        )

    def overlaps(self, other: "Location"):
        return (
            self.x < other.x + other.width
            and self.x + self.width > other.x
            and self.y < other.y + other.height
            and self.y + self.height > other.y
        )

    def character_within_location(self, character):
        self.current_visitors.append(character)
        return self.contains_point(*character.location.coordinates_location)

    def character_leaves_location(self, character):
        self.current_visitors.remove(character)

    def character_within_location_boundary(self, character):
        return (
            self.distance_to_point_from_nearest_edge(
                *character.location.coordinates_location
            )
            <= 1
        )

    def check_for_missing_visitors(self):
        for visitor in self.current_visitors:
            if not self.contains_point(*visitor.location.coordinates_location):
                self.current_visitors.remove(visitor)

    def move(self, delta_x, delta_y):
        self.x += delta_x
        self.y += delta_y

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height

    def __repr__(self):
        return f"Location(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def center(self):
        if self.width == 0 or self.height == 0:
            return self.x, self.y
        return self.x + self.width / 2, self.y + self.height / 2

    def distance_to_point_from_center(self, point_x, point_y):
        center_x, center_y = self.center()
        return ((center_x - point_x) ** 2 + (center_y - point_y) ** 2) ** 0.5

    def distance_to_point_from_nearest_edge(self, *args):
        if len(args) == 1:
            point_x, point_y = args[0]
        elif len(args) == 2:
            point_x, point_y = args
        else:
            raise TypeError(
                "distance_to_point_from_nearest_edge() takes 1 or 2 positional arguments but {} were given".format(
                    len(args)
                )
            )

        if self.contains_point(point_x, point_y):
            return 0
        nearest_x = max(self.x, min(point_x, self.x + self.width))
        nearest_y = max(self.y, min(point_y, self.y + self.height))
        return ((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2) ** 0.5

    def point_of_edge_nearest_to_point(self, point_x, point_y):
        nearest_x = max(self.x, min(point_x, self.x + self.width))
        nearest_y = max(self.y, min(point_y, self.y + self.height))
        return nearest_x, nearest_y

    def distance_to_location_from_center(self, other: "Location"):
        return self.distance_to_point_from_center(*other.center())

    def __eq__(self, other):
        if not isinstance(other, Location):
            if isinstance(other, tuple):
                return self.x == other[0] and self.y == other[1]
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
            and self.coordinates_location == other.coordinates_location
            and self.security == other.security
            and self.threat_level == other.threat_level
            and self.popularity == other.popularity
            and self.activities_available == other.activities_available
            and self.accessible == other.accessible
            and self.current_visitors == other.current_visitors
            and self.effect_dict == other.effect_dict
            and self.name == other.name
        )

    def hash_nested_list(self, obj):
        try:
            if isinstance(obj, list):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(
                    (key, self.hash_nested_list(value)) for key, value in obj.items()
                )
            elif isinstance(obj, set):
                return frozenset(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, tuple):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif hasattr(obj, "__hash__") and callable(getattr(obj, "__hash__")):
                # Test if the object can be hashed without raising an error
                try:
                    hash(obj)
                    return obj
                except TypeError:
                    if hasattr(obj, "__dict__"):
                        return tuple(
                            (key, self.hash_nested_list(value))
                            for key, value in obj.__dict__.items()
                        )
                    else:
                        # If the object is not hashable and has no __dict__, return its id or a string representation
                        return id(obj)
            elif hasattr(obj, "__dict__"):  # For custom objects without __hash__ method
                return tuple(
                    (key, self.hash_nested_list(value))
                    for key, value in obj.__dict__.items()
                )
            else:
                return obj
        except Exception as e:
            logging.error(f"Error hashing object: {e}")
            return None

    def __hash__(self):
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(e) for e in obj)
            elif isinstance(obj, set):
                return frozenset(make_hashable(e) for e in obj)
            elif isinstance(obj, tuple):
                return tuple(make_hashable(e) for e in obj)
            elif type(self.dict_or_obj).__name__ == "Character":
                Character = importlib.import_module("tiny_characters").Character

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )
            elif type(self.dict_or_obj).__name__ == "Location":
                Location = importlib.import_module("tiny_locations").Location

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )

            return obj

        return hash(
            tuple(
                [
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    self.coordinates_location,
                    self.security,
                    self.threat_level,
                    self.popularity,
                    make_hashable(self.activities_available),
                    self.accessible,
                    make_hashable(self.current_visitors),
                    self.name,
                ]
            )
        )

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __le__(self, other):
        return self.x <= other.x and self.y <= other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __ge__(self, other):
        return self.x >= other.x and self.y >= other.y

    def __ne__(self, other):
        return not self == other

    def __contains__(self, point):
        return self.contains_point(*point)

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "coordinates_location": self.coordinates_location,
            "center": self.center(),
            "area": self.get_area(),
            "aspect_ratio": self.get_aspect_ratio(),
            "perimeter": self.get_perimeter(),
            "bounding_box": self.get_bounding_box(),
            "diagonal": self.get_diagonal(),
            "security": self.security,
            "threat_level": self.threat_level,
            "popularity": self.popularity,
            "activities_available": self.activities_available,
            "accessible": self.accessible,
            "current_visitors": self.current_visitors,
            "effect_dict": self.effect_dict,
            "name": self.name,
        }


class LocationManager:
    def __init__(self):
        self.locations = []

    def add_location(self, location):
        self.locations.append(location)

    def find_locations_containing_point(self, point_x, point_y):
        return [loc for loc in self.locations if loc.contains_point(point_x, point_y)]

    def find_overlapping_locations(self, location):
        return [loc for loc in self.locations if loc.overlaps(location)]

    def move_location(self, location, delta_x, delta_y):
        if location in self.locations:
            location.move(delta_x, delta_y)

    def resize_location(self, location, new_width, new_height):
        if location in self.locations:
            location.resize(new_width, new_height)
