import importlib
import logging
from math import cos
import uu
import uuid

# Import actual implementations instead of placeholders
try:
    from actions import Action, State, ActionSystem
except ImportError:
    # Fallback to tiny_types if actions module is not available
    from tiny_types import Action, State, ActionSystem
    logging.warning("Failed to import tiny modules; defaulting to tiny_types. Functions may not work!")


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

    def get_safety_score(self):
        """Calculate overall safety score for AI decision making"""
        # Higher security, lower threat = safer
        base_safety = max(0, self.security - self.threat_level)
        # Crowded places might be safer (but not too crowded)
        visitor_factor = min(len(self.current_visitors) * 0.1, 1.0)
        return base_safety + visitor_factor

    def get_attractiveness_score(self, character_preferences=None):
        """Calculate how attractive this location is for a character"""
        base_attractiveness = self.popularity
        
        # Consider activities available
        activity_bonus = len(self.activities_available) * 0.5
        
        # Less crowded is generally more attractive (personal space)
        crowding_penalty = len(self.current_visitors) * 0.2
        
        # Consider character preferences if provided
        preference_bonus = 0
        if character_preferences:
            # Example: character might prefer certain activities
            preferred_activities = character_preferences.get('preferred_activities', [])
            for activity in self.activities_available:
                if activity in preferred_activities:
                    preference_bonus += 2
        
        return max(0, base_attractiveness + activity_bonus - crowding_penalty + preference_bonus)

    def is_suitable_for_character(self, character, purpose=None):
        """Determine if location is suitable for a character's needs"""
        # Check accessibility
        if not self.accessible:
            return False
        
        # Check safety requirements - some characters might avoid dangerous areas
        safety_threshold = getattr(character, 'safety_threshold', 0)
        if self.get_safety_score() < safety_threshold:
            return False
        
        # Check if location supports the intended purpose
        if purpose and purpose not in self.activities_available:
            return False
        
        # Check capacity - don't overcrowd
        max_comfortable_visitors = max(1, self.get_area() // 50)  # rough estimate
        if len(self.current_visitors) >= max_comfortable_visitors:
            return False
        
        return True

    def get_travel_appeal(self, from_location, character_preferences=None):
        """Calculate appeal of traveling to this location from another location"""
        # Base attractiveness
        appeal = self.get_attractiveness_score(character_preferences)
        
        # Distance penalty - closer is generally better
        if from_location:
            distance = self.distance_to_location_from_center(from_location)
            distance_penalty = distance * 0.01  # Adjust scaling as needed
            appeal -= distance_penalty
        
        # Variety bonus - if character hasn't been here recently
        if character_preferences and 'recent_locations' in character_preferences:
            recent_locations = character_preferences['recent_locations']
            if self.uuid not in recent_locations:
                appeal += 1.0  # Bonus for new experiences
        
        return max(0, appeal)

    def update_from_interactions(self, interaction_type, character):
        """Update location properties based on character interactions"""
        # Increase visit count
        self.visit_count += 1
        
        # Popular characters might increase location popularity
        character_influence = getattr(character, 'social_influence', 1)
        if interaction_type == "positive_interaction":
            self.popularity += character_influence * 0.1
        
        # Security might change based on incidents
        if interaction_type == "security_incident":
            self.security = max(0, self.security - 1)
            self.threat_level += 1
        elif interaction_type == "security_improvement":
            self.security += 1
            self.threat_level = max(0, self.threat_level - 0.5)

    def get_recommended_activities_for_character(self, character):
        """Get activities recommended for a specific character"""
        suitable_activities = []
        
        # Filter activities based on character attributes
        character_energy = getattr(character, 'energy', 50)
        character_social_preference = getattr(character, 'social_preference', 50)
        
        for activity in self.activities_available:
            activity_lower = activity.lower()
            
            # Energy-based filtering
            if character_energy < 30 and 'rest' in activity_lower:
                suitable_activities.append(activity)
            elif character_energy > 70 and ('exercise' in activity_lower or 'active' in activity_lower):
                suitable_activities.append(activity)
            
            # Social preference filtering
            if character_social_preference > 60 and 'social' in activity_lower:
                suitable_activities.append(activity)
            elif character_social_preference < 40 and ('quiet' in activity_lower or 'solitary' in activity_lower):
                suitable_activities.append(activity)
            
            # General activities available to all
            if activity_lower in ['explore', 'visit', 'observe']:
                suitable_activities.append(activity)
        
        return list(set(suitable_activities))  # Remove duplicates


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


class PointOfInterest:
    """Points of Interest system for specific interactive spots that aren't buildings"""
    
    def __init__(self, name, x, y, poi_type="generic", interaction_radius=5, 
                 action_system=None, description=""):
        self.name = name
        self.x = x
        self.y = y
        self.poi_type = poi_type  # e.g., "bench", "well", "statue", "garden"
        self.interaction_radius = interaction_radius
        self.description = description
        self.uuid = uuid.uuid4()
        self.coordinates = (x, y)
        self.current_users = []  # Characters currently interacting with this POI
        self.max_users = self._get_max_users_by_type(poi_type)
        
        # Initialize actions based on POI type
        if action_system:
            self.possible_interactions = self._create_type_specific_actions(action_system)
        else:
            self.possible_interactions = []
    
    def _get_max_users_by_type(self, poi_type):
        """Get maximum users based on POI type"""
        type_capacities = {
            "bench": 2,
            "well": 3,
            "statue": 5,
            "garden": 8,
            "fountain": 6,
            "tree": 4,
            "generic": 2
        }
        return type_capacities.get(poi_type, 2)
    
    def _create_type_specific_actions(self, action_system):
        """Create actions specific to POI type"""
        actions = []
        
        # Common interaction for all POIs
        actions.append(Action(
            f"Interact with {self.name}",
            action_system.instantiate_conditions([
                {
                    "name": "energy",
                    "attribute": "energy", 
                    "target": "initiator",
                    "satisfy_value": 5,
                    "operator": "gt"
                }
            ]),
            [
                {"targets": ["initiator"], "attribute": "mood", "change_value": 2},
                {"targets": ["initiator"], "attribute": "energy", "change_value": -1}
            ],
            cost=1
        ))
        
        # Type-specific actions
        if self.poi_type == "bench":
            actions.append(Action(
                "Rest on bench",
                action_system.instantiate_conditions([]),
                [
                    {"targets": ["initiator"], "attribute": "energy", "change_value": 5},
                    {"targets": ["initiator"], "attribute": "comfort", "change_value": 3}
                ],
                cost=0
            ))
        elif self.poi_type == "well":
            actions.append(Action(
                "Draw water",
                action_system.instantiate_conditions([]),
                [
                    {"targets": ["initiator"], "attribute": "thirst", "change_value": -10},
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -2}
                ],
                cost=1
            ))
        elif self.poi_type == "garden":
            actions.append(Action(
                "Admire flowers",
                action_system.instantiate_conditions([]),
                [
                    {"targets": ["initiator"], "attribute": "mood", "change_value": 5},
                    {"targets": ["initiator"], "attribute": "beauty_appreciation", "change_value": 3}
                ],
                cost=0
            ))
        
        return actions
    
    def can_interact(self, character):
        """Check if character can interact with this POI"""
        if len(self.current_users) >= self.max_users:
            return False
        
        # Check distance
        distance = self.distance_to_point(character.location.coordinates_location[0], 
                                        character.location.coordinates_location[1])
        return distance <= self.interaction_radius
    
    def distance_to_point(self, x, y):
        """Calculate distance from POI to a point"""
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
    
    def add_user(self, character):
        """Add a character as current user of this POI"""
        if self.can_interact(character) and character not in self.current_users:
            self.current_users.append(character)
            return True
        return False
    
    def remove_user(self, character):
        """Remove a character from current users"""
        if character in self.current_users:
            self.current_users.remove(character)
    
    def get_possible_interactions(self, requester):
        """Get available interactions for a character"""
        if self.can_interact(requester):
            return self.possible_interactions
        return []
    
    def get_info(self):
        """Get information about this POI"""
        return {
            "name": self.name,
            "type": self.poi_type,
            "description": self.description,
            "coordinates": self.coordinates,
            "current_users": len(self.current_users),
            "max_users": self.max_users,
            "available": len(self.current_users) < self.max_users
        }
