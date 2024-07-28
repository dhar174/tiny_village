from math import cos
from actions import Action, State, ActionSystem


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
    def __init__(self, name, x, y, width, height, action_system: ActionSystem):
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
        return self.width * self.height

    def get_center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    def get_diagonal(self):
        return (self.width**2 + self.height**2) ** 0.5

    def get_aspect_ratio(self):
        return self.width / self.height

    def get_perimeter(self):
        return 2 * (self.width + self.height)

    def get_bounding_box(self):
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
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )

    def __hash__(self):
        return hash((self.x, self.y, self.width, self.height))

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
