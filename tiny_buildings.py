import logging
import uuid
import random
from actions import Action, ActionSystem
from tiny_locations import Location, LocationManager
from tiny_types import Character, GraphManager

effect_dict = {
    "Enter Building": [
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 5},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -2},
        {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 5},
        {
            "targets": ["initiator"],
            "method": "play_animation",
            "method_args": ["talking"],
        },
    ],
}

preconditions_dict = {
    "Enter Building": [
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


class Building:
    def __init__(
        self,
        name,
        x,
        y,
        height,
        width,
        length,
        stories=1,
        num_rooms: int = 1,
        address="123 Main St",
        action_system: ActionSystem = ActionSystem(),
        door=None,
        building_type="building",
        owner=None,
    ):
        self.name = name
        self.height = height
        self.width = width
        self.length = length
        self.address = address
        self.volume_val = self.volume()
        self.stories = stories
        self.door = door
        self.owner = owner
        self.building_type = building_type
        self.num_rooms = num_rooms

        self.area_val = self.area()
        self.uuid = uuid.uuid4()
        self.coordinates_location = (x, y)
        
        # Create Location instance for this building
        self.location = Location(
            name=f"{name} Location",
            x=x,
            y=y,
            width=width,
            height=length,  # Use length as height for location
            action_system=action_system,
            security=self._calculate_security(),
            popularity=self._calculate_popularity(),
        )
        
        # Add building-specific activities to location
        self._setup_location_activities()
        
        self.possible_interactions = [
            Action(
                "Enter Building",
                action_system.instantiate_conditions(
                    preconditions_dict["Enter Building"]
                ),
                effect_dict["Enter Building"],
            )
        ]

    def get_possible_interactions(self, requester):
        return [
            action
            for action in self.possible_interactions
            if action.can_execute(requester)
        ]

    def set_owner(self, owner):
        self.owner = owner

    def volume(self):
        return self.length * self.width * self.height

    def area(self):
        return int((self.length * self.width) * self.stories)

    def calculate_area_per_floor(self):
        return self.area() / self.stories

    def __str__(self):
        return (
            f"{self.name} is {self.height} high, {self.width} wide, {self.length} long, "
            f"and is located at {self.address} and has {self.stories} stories. "
            f"It has a building type of {self.building_type} and coordinates {self.x}, {self.y} "
            f"and total area of  {self.area_val} with {self.calculate_area_per_floor()} area per floor. "
            f"It has a current price value of {self.price_value}. It has {self.num_rooms} rooms and {self.stories} "
            f"floors and is owned by {self.owner} \n "
            f"             It has the following possible interactions: {self.possible_interactions}. "
            f"It has a door at {self.door}. It has the following coordinates: {self.coordinates_location}. "
            f"It has the following ID: {self.uuid}. It has the following volume: {self.volume_val}."
        )

    def __repr__(self):
        return (
            f"Building({self.name}, {self.height}, {self.width}, {self.length}, "
            f"{self.address}, {self.stories}, {self.coordinates_location}, "
            f"{self.possible_interactions}, {self.door}, {self.uuid}, "
            f"{self.volume_val}, {self.area_val})"
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
                        # If the object is not hashable and has no __dict__, return its uuid or a string representation
                        return uuid(obj)
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
        return hash(
            tuple(
                [
                    self.name,
                    self.height,
                    self.width,
                    self.length,
                    self.address,
                    self.stories,
                    self.coordinates_location,
                    self.door,
                    self.uuid,
                    self.volume_val,
                    self.area_val,
                    self.num_rooms,
                ]
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Building):
            if isinstance(other, Location):
                return self.coordinates_location == other.get_coordinates()
            return False
        return (
            self.name == other.name
            and self.height == other.height
            and self.width == other.width
            and self.length == other.length
            and self.address == other.address
            and self.stories == other.stories
            and self.coordinates_location == other.coordinates_location
            and self.door == other.door
            and self.uuid == other.uuid
            and self.volume_val == other.volume_val
            and self.area_val == other.area_val
            and self.num_rooms == other.num_rooms
        )

    def get_coordinates(self):
        return self.coordinates_location
    
    def _calculate_security(self):
        """Calculate security level based on building properties"""
        # Base security value
        security = 5
        
        # Larger buildings tend to be more secure
        if self.area_val > 1000:
            security += 2
        elif self.area_val > 500:
            security += 1
            
        # Multi-story buildings are more secure
        if self.stories > 1:
            security += 1
            
        # Building type affects security
        if self.building_type == "house":
            security += 2  # Houses are generally safe
        elif self.building_type == "commercial":
            security -= 1  # Commercial areas may be less secure
            
        return max(0, min(10, security))  # Clamp between 0-10
    
    def _calculate_popularity(self):
        """Calculate popularity based on building properties"""
        # Base popularity
        popularity = 3
        
        # Larger buildings attract more people
        if self.area_val > 2000:
            popularity += 3
        elif self.area_val > 1000:
            popularity += 2
        elif self.area_val > 500:
            popularity += 1
            
        # Building type affects popularity
        if self.building_type == "commercial":
            popularity += 3  # Commercial buildings are popular
        elif self.building_type == "house":
            popularity -= 1  # Houses are more private
            
        return max(0, min(10, popularity))  # Clamp between 0-10
    
    def _setup_location_activities(self):
        """Setup activities available at this building location"""
        # Add basic building activities
        self.location.add_activity("enter_building")
        self.location.add_activity("observe_building")
        
        # Add activities based on building type
        if self.building_type == "house":
            self.location.add_activity("visit_residents")
            self.location.add_activity("rest")
        elif self.building_type == "commercial":
            self.location.add_activity("shop")
            self.location.add_activity("conduct_business")
        elif self.building_type == "office":
            self.location.add_activity("work")
            self.location.add_activity("meet")
    
    def get_location(self):
        """Get the Location instance associated with this building"""
        return self.location
    
    def get_security_level(self):
        """Get the security level of this building's location"""
        return self.location.security
    
    def get_popularity_level(self):
        """Get the popularity level of this building's location"""
        return self.location.popularity
    
    def get_available_activities(self):
        """Get list of activities available at this building"""
        return self.location.activities_available
    
    def is_within_building(self, character_location):
        """Check if a character location is within this building"""
        return self.location.contains_point(*character_location)


class House(Building):
    def __init__(
        self,
        name,
        x,
        y,
        height,
        width,
        length,
        address,
        stories=1,
        bedrooms=1,
        bathrooms=1,
        initial_beauty_value=10,
        price_value=0,
        door=None,
        owner=None,
    ):
        num_rooms = int(bedrooms + bathrooms)
        super().__init__(
            name=name,
            x=x,
            y=y,
            width=width,
            height=height,
            length=length,
            stories=stories,
            address=address,
            action_system=ActionSystem(),
            door=door,
            building_type="house",
            owner=owner,
            num_rooms=num_rooms,
        )
        self.name = name
        self.height = height
        self.width = width
        self.length = length
        self.address = address
        self.stories = stories
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.shelter_value = self.calculate_shelter_value()
        self.beauty_value = self.set_beauty_value(initial_beauty_value)
        self.price = self.calculate_price(price_value)
        self.x = x
        self.y = y
        
        # Override location activities for houses
        self._setup_house_activities()

    def __str__(self):
        return (
            f"{self.name} is {self.height} high, {self.width} wide, {self.length} long, "
            f"and is located at {self.address}. It has {self.stories} stories, {self.bedrooms} bedrooms, "
            f"and {self.bathrooms} bathrooms and is worth ${self.price} and has a beauty value of {self.beauty_value}. "
            f"It has a shelter value of {self.shelter_value} and is {self.area_val} square feet. "
            f"It is located at {self.x}, {self.y} on the map. It has the following possible interactions: {self.possible_interactions}. "
            f"It has a door at {self.door}. It has the following coordinates: {self.coordinates_location}. "
            f"It has the following ID: {self.uuid}. It has the following volume: {self.volume_val}."
        )

    def __repr__(self):
        return (
            f"House({self.name}, {self.height}, {self.width}, {self.length}, {self.address}, {self.stories}, "
            f"{self.bedrooms}, {self.bathrooms}, {self.beauty_value}, {self.price}, {self.x}, {self.y}, "
            f"{self.area_val}, {self.volume_val}, {self.shelter_value}, {self.coordinates_location}, "
            f"{self.possible_interactions}, {self.door}, {self.uuid})"
        )

    def calculate_shelter_value(self):
        score = 1
        score += min(round(self.area_val / 1000), 5)
        if self.bedrooms > 1:
            score += 1
        if self.bathrooms > 1:
            score += 1
        if self.stories > 1:
            score += 1
        if self.bedrooms > 2:
            score += 1
        if self.bathrooms > 2:
            score += 1
        if self.stories > 2:
            score += 1

        return score

    def set_beauty_value(self, beauty_value):
        if beauty_value < 1:
            self.beauty_value = 1
        elif beauty_value > 100:
            self.beauty_value = 100
        else:
            self.beauty_value = beauty_value
        return self.beauty_value

    def calculate_price(self, price_value):
        if price_value <= 0:
            return 10 * (((self.shelter_value * self.beauty_value) / 100) * self.area_val)
        else:
            return price_value

    def get_price(self):
        return self.price

    def get_beauty_value(self):
        return self.beauty_value

    def get_shelter_value(self):
        return self.shelter_value

    def get_bedrooms(self):
        return self.bedrooms

    def get_bathrooms(self):
        return self.bathrooms

    def get_stories(self):
        return self.stories

    def get_area(self):
        return self.area_val

    def get_volume(self):
        return self.volume_val

    def get_address(self):
        return self.address

    def get_name(self):
        return self.name

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_length(self):
        return self.length
    
    def _setup_house_activities(self):
        """Setup house-specific activities"""
        # Clear existing activities and add house-specific ones
        self.location.activities_available.clear()
        
        # Basic house activities
        self.location.add_activity("enter_house")
        self.location.add_activity("rest")
        self.location.add_activity("sleep")
        self.location.add_activity("relax")
        
        # Activities based on number of bedrooms and bathrooms
        if self.bedrooms > 1:
            self.location.add_activity("visit_family")
        if self.bathrooms > 0:
            self.location.add_activity("use_facilities")
        if self.stories > 1:
            self.location.add_activity("explore_floors")
            
        # Activities based on shelter value and beauty
        if self.shelter_value > 5:
            self.location.add_activity("secure_shelter")
        if self.beauty_value > 10:
            self.location.add_activity("enjoy_aesthetics")

    def to_dict(self):
        return {
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "length": self.length,
            "address": self.address,
            "stories": self.stories,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "shelter_value": self.shelter_value,
            "beauty_value": self.beauty_value,
            "price": self.price,
            "x": self.x,
            "y": self.y,
            "area": self.area_val,
            "volume": self.volume_val,
            "coordinates_location": self.coordinates_location,
            "possible_interactions": self.possible_interactions,
            "door": self.door,
            "uuid": self.uuid,
        }

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
                        # If the object is not hashable and has no __dict__, return its uuid or a string representation
                        return uuid(obj)
            elif hasattr(obj, "__dict__"):  # For custom objects without __hash__ method
                return tuple(
                    (key, self.hash_nested_list(value))
                    for key, value in obj.__dict__.items()
                )
            else:
                return obj
        except Exception as e:
            logging.error(f"Error hashing object {obj}: {e}")
            return None

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.name,
                    self.uuid,
                    self.coordinates_location,
                    self.area_val,
                    self.volume_val,
                    self.shelter_value,
                    self.beauty_value,
                    self.price,
                    self.door,
                    self.x,
                    self.y,
                    self.address,
                    self.stories,
                    self.bedrooms,
                    self.bathrooms,
                    self.height,
                    self.width,
                    self.length,
                    self.num_rooms,
                ]
            )
        )

    def __eq__(self, other):
        if not isinstance(other, House) and not isinstance(other, Building):
            if isinstance(other, Location):
                return self.coordinates_location == other.get_coordinates()
            return False
        return (
            self.name == other.name
            and self.uuid == other.uuid
            and self.coordinates_location == other.coordinates_location
            and self.area_val == other.area_val
            and self.volume_val == other.volume_val
            and self.shelter_value == other.shelter_value
            and self.beauty_value == other.beauty_value
            and self.price == other.price
            and self.door == other.door
            and self.x == other.x
            and self.y == other.y
            and self.address == other.address
            and self.stories == other.stories
            and self.bedrooms == other.bedrooms
            and self.bathrooms == other.bathrooms
            and self.height == other.height
            and self.width == other.width
            and self.length == other.length
            and self.num_rooms == other.num_rooms
        )


class CreateBuilding:
    def __init__(self, map_data=None):
        self.description = "This is a class to create a building."
        self.map_data = map_data or {"width": 100, "height": 100, "buildings": []}
        self.occupied_areas = set()  # Track occupied grid positions

        # Initialize occupied areas with existing buildings
        for building in self.map_data.get("buildings", []):
            rect = building.get("rect")
            if rect:
                for x in range(rect.left, rect.right):
                    for y in range(rect.top, rect.bottom):
                        self.occupied_areas.add((x, y))

    def find_valid_coordinates(self, width, length, max_attempts=100):
        """
        Find valid x, y coordinates for a building placement.

        Args:
            width: Building width in grid units
            length: Building length in grid units
            max_attempts: Maximum placement attempts before giving up

        Returns:
            tuple: (x, y) coordinates or (0, 0) if placement fails
        """
        for attempt in range(max_attempts):
            # Generate random coordinates with buffer from edges
            buffer = 5
            max_x = max(buffer, self.map_data["width"] - width - buffer)
            max_y = max(buffer, self.map_data["height"] - length - buffer)

            if max_x <= buffer or max_y <= buffer:
                logging.warning(f"Building too large for map: {width}x{length}")
                return (0, 0)

            x = random.randint(buffer, max_x)
            y = random.randint(buffer, max_y)

            # Check if placement area is free
            placement_valid = True
            for bx in range(x, x + width):
                for by in range(y, y + length):
                    if (bx, by) in self.occupied_areas:
                        placement_valid = False
                        break
                if not placement_valid:
                    break

            if placement_valid:
                # Reserve the area
                for bx in range(x, x + width):
                    for by in range(y, y + length):
                        self.occupied_areas.add((bx, by))
                return (x, y)

        # If we can't find a spot, try systematic placement
        return self._systematic_placement(width, length)

    def _systematic_placement(self, width, length):
        """
        Systematically search for placement starting from corners.
        """
        buffer = 2
        for x in range(buffer, self.map_data["width"] - width - buffer, 5):
            for y in range(buffer, self.map_data["height"] - length - buffer, 5):
                placement_valid = True
                for bx in range(x, x + width):
                    for by in range(y, y + length):
                        if (bx, by) in self.occupied_areas:
                            placement_valid = False
                            break
                    if not placement_valid:
                        break

                if placement_valid:
                    # Reserve the area
                    for bx in range(x, x + width):
                        for by in range(y, y + length):
                            self.occupied_areas.add((bx, by))
                    return (x, y)

        logging.warning("Could not find valid placement for building")
        return (0, 0)

    def create_house_manually(self):
        self.name = input("What is the name of the building? ")
        self.height = int(input("How tall is the building? "))
        self.width = int(input("How wide is the building? "))
        self.length = int(input("How long is the building? "))
        self.address = input("What is the address of the building? ")
        self.stories = int(input("How many stories does the building have? "))
        self.bedrooms = int(input("How many bedrooms does the building have? "))
        self.bathrooms = int(input("How many bathrooms does the building have? "))
        return House(
            self.name,
            self.height,
            self.width,
            self.length,
            self.address,
            self.stories,
            self.bedrooms,
            self.bathrooms,
        )

    def create_house(
        self,
        name,
        height,
        width,
        length,
        address,
        stories,
        bedrooms,
        bathrooms,
        initial_beauty_value=10,
        price_value=0,
    ):
        # Find valid coordinates for building placement
        # Convert dimensions to grid units (assuming 1 unit per meter)
        grid_width = max(1, int(width / 10))  # Scale down to grid units
        grid_length = max(1, int(length / 10))

        x, y = self.find_valid_coordinates(grid_width, grid_length)

        return House(
            name,
            x,
            y,
            height,
            width,
            length,
            address,
            stories=stories,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            initial_beauty_value=initial_beauty_value,
            price_value=price_value,
            door=None,
            owner=None,
        )

    def create_building(self, name, height, width, length, stories, address):
        # Find valid coordinates for building placement
        grid_width = max(1, int(width / 10))
        grid_length = max(1, int(length / 10))

        x, y = self.find_valid_coordinates(grid_width, grid_length)

        return Building(name, x, y, height, width, length, stories, address=address)

    def create_house_by_type(self, structure: str = "hovel"):
        if "hovel" in structure.lower():
            return self.create_house("Hovel", 10, 10, 10, "123 Main St", 1, 1, 1, 1)
        elif "mansion" in structure.lower():
            return self.create_house(
                "Mansion", 100, 100, 100, "123 Main St", 3, 5, 5, 20
            )
        elif "apartment" in structure.lower():
            return self.create_house(
                "Apartment", 50, 50, 50, "123 Main St", 1, 1, 1, 10
            )
        elif "condo" in structure.lower():
            return self.create_house("Condo", 50, 50, 50, "123 Main St", 2, 2, 2, 10)
        elif "townhouse" in structure.lower():
            return self.create_house(
                "Townhouse", 50, 50, 50, "123 Main St", 2, 2, 2, 10
            )
        elif "duplex" in structure.lower():
            return self.create_house("Duplex", 50, 50, 50, "123 Main St", 2, 2, 2, 10)
        elif "trailer" in structure.lower():
            return self.create_house("Trailer", 50, 50, 50, "123 Main St", 1, 2, 1, 5)

        else:
            return self.create_house("Hovel", 10, 10, 10, "123 Main St", 1, 1, 1, 1)

    def generate_random_house(self):
        from random import randint

        name = "House"
        height = randint(10, 100)
        width = randint(10, 50)
        length = randint(10, 50)
        address = "123 Main St"
        stories = randint(1, 2)
        bedrooms = randint(1, 3)
        bathrooms = randint(1, 2)
        beauty = randint(1, 20)
        return self.create_house(
            name, height, width, length, address, stories, bedrooms, bathrooms, beauty
        )
