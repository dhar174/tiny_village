import uuid
from actions import Action, ActionSystem
from tiny_locations import Location, LocationManager

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


class Building(Location):
    def __init__(
        self,
        name,
        x,
        y,
        height,
        width,
        length,
        stories=1,
        address="123 Main St",
        action_system: ActionSystem = ActionSystem(),
        door=None,
    ):
        self.name = name
        self.height = height
        self.width = width
        self.length = length
        self.address = address
        self.volume = self.volume()
        self.stories = stories
        self.door = door

        self.area = self.area()
        self.id = uuid.uuid4()
        self.coordinates_location = (x, y)
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
        self.possible_interactions["Enter Building"].append(
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

    def volume(self):
        return self.length * self.width * self.height

    def area(self):
        return int((self.length * self.width) * self.stories)

    def calculate_area_per_floor(self):
        return self.area() / self.stories

    def __str__(self):
        return f"{self.name} is {self.height} high, {self.width} wide, {self.length} long, and is located at {self.address}."


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
    ):
        super().__init__(
            name,
            x,
            y,
            width,
            height,
            length,
            stories,
            address,
            action_system=ActionSystem(),
            door=door,
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

    def __str__(self):
        return f"{self.name} is {self.height} high, {self.width} wide, {self.length} long, and is located at {self.address}. It has {self.stories} stories, {self.bedrooms} bedrooms, and {self.bathrooms} bathrooms."

    def __repr__(self):
        return f"House({self.name}, {self.height}, {self.width}, {self.length}, {self.address}, {self.stories}, {self.bedrooms}, {self.bathrooms})"

    def calculate_shelter_value(self):
        score = 1
        score += min(round(self.area / 1000), 5)
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
            return 10 * (((self.shelter_value * self.beauty_value) / 100) * self.area)
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
        return self.area

    def get_volume(self):
        return self.volume

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
        }


class CreateBuilding:
    def __init__(self):
        self.description = "This is a class to create a building."
        # if structure == None:
        #     if name == None or height == None or width == None or length == None or address == None or stories == None:
        #         return self.create_house_manually()
        #     elif bedrooms == None or bathrooms == None:
        #         return self.create_building(name, height, width, length, stories, address)
        #     elif initial_beauty_value == None or price_value == None:
        #         return self.create_house(name, height, width, length, address, stories, bedrooms, bathrooms)
        #     else:
        #         return self.create_house(name, height, width, length, address, stories, bedrooms, bathrooms, initial_beauty_value, price_value)
        # else:
        #     return self.create_house_by_type(structure)

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
        return House(
            name,
            height,
            width,
            length,
            address,
            stories,
            bedrooms,
            bathrooms,
            initial_beauty_value,
            price_value,
        )

    def create_building(self, name, height, width, length, stories, address):
        return Building(name, height, width, length, stories, address)

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
