import uuid
from typing import List


class ItemObject:
    def __init__(self, name, description, value, weight, quantity):
        self.name = name
        self.description = description
        self.value = value
        self.weight = weight
        self.quantity = quantity
        self.id = uuid.uuid4()

    def __repr__(self):
        return f"Item({self.name}, {self.description}, {self.value}, {self.weight}, {self.quantity})"

    def __str__(self):
        return f"Item named {self.name} with description {self.description} and value {self.value}."

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.value == other.value
            and self.weight == other.weight
            and self.quantity == other.quantity
        )

    def __hash__(self):
        return hash(
            (self.name, self.description, self.value, self.weight, self.quantity)
        )

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return self.name

    def get_description(self):
        return self.description

    def set_description(self, description):
        self.description = description
        return self.description

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value
        return self.value

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
        return self.weight

    def get_quantity(self):
        return self.quantity

    def set_quantity(self, quantity):
        self.quantity = quantity
        return self.quantity

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "weight": self.weight,
            "quantity": self.quantity,
        }


class FoodItem(ItemObject):
    def __init__(self, name, description, value, weight, quantity, calories):
        super().__init__(name, description, value, weight, quantity)
        self.calories = calories

    def __repr__(self):
        return f"FoodItem({self.name}, {self.description}, {self.value}, {self.weight}, {self.quantity}, {self.calories})"

    def __str__(self):
        return f"FoodItem named {self.name} with description {self.description} and value {self.value}."

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.value == other.value
            and self.weight == other.weight
            and self.quantity == other.quantity
            and self.calories == other.calories
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                self.description,
                self.value,
                self.weight,
                self.quantity,
                self.calories,
            )
        )

    def get_calories(self):
        return self.calories

    def set_calories(self, calories):
        self.calories = calories
        return self.calories

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "weight": self.weight,
            "quantity": self.quantity,
            "calories": self.calories,
        }


class ItemInventory:
    def __init__(
        self,
        food_items: List[FoodItem] = [],
        clothing_items: List[ItemObject] = [],
        tools_items: List[ItemObject] = [],
        weapons_items: List[ItemObject] = [],
        medicine_items: List[ItemObject] = [],
        misc_items: List[ItemObject] = [],
    ):
        self.food_items = self.set_food_items(food_items)
        self.clothing_items = self.set_clothing_items(clothing_items)
        self.tools_items = self.set_tools_items(tools_items)
        self.weapons_items = self.set_weapons_items(weapons_items)
        self.medicine_items = self.set_medicine_items(medicine_items)
        self.misc_items = self.set_misc_items(misc_items)

    def __repr__(self):
        return f"ItemInventory({self.food_items}, {self.clothing_items}, {self.tools_items}, {self.weapons_items}, {self.medicine_items}, {self.misc_items})"

    def __str__(self):
        return f"ItemInventory with food items {self.food_items}, clothing items {self.clothing_items}, tools items {self.tools_items}, weapons items {self.weapons_items}, medicine items {self.medicine_items}, misc items {self.misc_items}."

    def __eq__(self, other):
        return (
            self.food_items == other.food_items
            and self.clothing_items == other.clothing_items
            and self.tools_items == other.tools_items
            and self.weapons_items == other.weapons_items
            and self.medicine_items == other.medicine_items
            and self.misc_items == other.misc_items
        )

    def __hash__(self):
        return hash(
            (
                self.food_items,
                self.clothing_items,
                self.tools_items,
                self.weapons_items,
                self.medicine_items,
                self.misc_items,
            )
        )

    def get_food_items(self):
        return self.food_items

    def set_food_items(self, food_items: List[FoodItem]):
        self.food_items = food_items
        return self.food_items

    def count_food_items_total(self):
        total = 0
        for item in self.food_items:
            total += item.get_quantity()
        return total

    def count_food_items_by_name(self, name):
        total = 0
        for item in self.food_items:
            if item.get_name() == name:
                total += item.get_quantity()
        return total

    def count_food_calories_total(self):
        total = 0
        for item in self.food_items:
            total += item.get_calories() * item.get_quantity()
        return total

    def get_clothing_items(self):
        return self.clothing_items

    def set_clothing_items(self, clothing_items: List[ItemObject]):
        self.clothing_items = clothing_items
        return self.clothing_items

    def get_tools_items(self):
        return self.tools_items

    def set_tools_items(self, tools_items: List[ItemObject]):
        self.tools_items = tools_items
        return self.tools_items

    def get_weapons_items(self):
        return self.weapons_items

    def set_weapons_items(self, weapons_items: List[ItemObject]):
        self.weapons_items = weapons_items
        return self.weapons_items

    def get_medicine_items(self):
        return self.medicine_items

    def set_medicine_items(self, medicine_items: List[ItemObject]):
        self.medicine_items = medicine_items
        return self.medicine_items

    def get_misc_items(self):
        return self.misc_items

    def set_misc_items(self, misc_items: List[ItemObject]):
        self.misc_items = misc_items
        return self.misc_items

    def get_total_value(self):
        total = 0
        if self.food_items:
            for item in self.food_items:
                total += item.get_value() * item.get_quantity()
        if self.clothing_items:
            for item in self.clothing_items:
                total += item.get_value() * item.get_quantity()
        if self.tools_items:
            for item in self.tools_items:
                total += item.get_value() * item.get_quantity()
        if self.weapons_items:
            for item in self.weapons_items:
                total += item.get_value() * item.get_quantity()
        if self.medicine_items:
            for item in self.medicine_items:
                total += item.get_value() * item.get_quantity()
        if self.misc_items:
            for item in self.misc_items:
                total += item.get_value() * item.get_quantity()
        return total

    def get_total_weight(self):
        total = 0
        for item in self.food_items:
            total += item.get_weight() * item.get_quantity()
        for item in self.clothing_items:
            total += item.get_weight() * item.get_quantity()
        for item in self.tools_items:
            total += item.get_weight() * item.get_quantity()
        for item in self.weapons_items:
            total += item.get_weight() * item.get_quantity()
        for item in self.medicine_items:
            total += item.get_weight() * item.get_quantity()
        for item in self.misc_items:
            total += item.get_weight() * item.get_quantity()
        return total

    def get_total_quantity(self):
        total = 0

        if self.food_items:
            for item in self.food_items:
                total += item.get_quantity()
        if self.clothing_items:
            for item in self.clothing_items:
                total += item.get_quantity()
        if self.tools_items:
            for item in self.tools_items:
                total += item.get_quantity()
        if self.weapons_items:
            for item in self.weapons_items:
                total += item.get_quantity()
        if self.medicine_items:
            for item in self.medicine_items:
                total += item.get_quantity()
        if self.misc_items:
            for item in self.misc_items:
                total += item.get_quantity()
        return total

    def count_total_items(self):
        total = 0
        if self.food_items:
            for item in self.food_items:
                total += item.get_quantity()
        if self.clothing_items:
            for item in self.clothing_items:
                total += item.get_quantity()
        if self.tools_items:
            for item in self.tools_items:
                total += item.get_quantity()
        if self.weapons_items:
            for item in self.weapons_items:
                total += item.get_quantity()
        if self.medicine_items:
            for item in self.medicine_items:
                total += item.get_quantity()
        if self.misc_items:
            for item in self.misc_items:
                total += item.get_quantity()
        return total

    def to_dict(self):
        return {
            "food": self.food_items,
            "clothing": self.clothing_items,
            "tools": self.tools_items,
            "weapons": self.weapons_items,
            "medicine": self.medicine_items,
            "misc": self.misc_items,
        }
