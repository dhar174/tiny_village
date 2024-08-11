from enum import unique
import importlib
import logging
import operator
from re import T
import uuid
from typing import List

from regex import F

# from actions import Action, ActionSystem
from tiny_locations import Location

from tiny_types import Action, ActionSystem, GraphManager


class Stock:
    def __init__(self, name, value, quantity, stock_description=None):
        self.name = name
        self.value = value
        self.quantity = quantity
        self.stock_description = stock_description
        self.uuid = uuid.uuid4()
        self.scarcity = None
        self.ownership_history = ["unowned"]

    def __repr__(self):
        return f"Stock({self.name}, {self.value}, {self.quantity})"

    def __eq__(self, other):
        if not isinstance(other, Stock):
            return False
        return (
            self.name == other.name
            and self.value == other.value
            and self.quantity == other.quantity
        )

    def __hash__(self):
        return hash(tuple([self.name, self.value, self.quantity]))

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return self.name

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value
        return self.value

    def get_quantity(self):
        return self.quantity

    def set_quantity(self, quantity):
        self.quantity = quantity
        return self.quantity

    def increase_quantity(self, quantity):
        self.quantity += quantity
        return self.quantity

    def decrease_quantity(self, quantity):
        self.quantity -= quantity
        return self.quantity

    def increase_value(self, value):
        self.value += value
        return self.value

    def decrease_value(self, value):
        self.value -= value
        return self.value

    def to_dict(self):
        return {"name": self.name, "value": self.value, "quantity": self.quantity}


class InvestmentPortfolio:
    def __init__(self, stocks: List[Stock]):
        self.stocks = stocks

    def __repr__(self):
        return f"InvestmentPortfolio({self.stocks})"

    def __eq__(self, other):
        return self.stocks == other.stocks

    def __hash__(self):
        return hash(tuple(self.stocks))

    def get_stocks(self):
        return self.stocks

    def set_stocks(self, stocks):
        self.stocks = stocks
        return self.stocks

    def add_stock(self, stock):
        self.stocks.append(stock)
        return self.stocks

    def remove_stock(self, stock):
        self.stocks.remove(stock)
        return self.stocks

    def update_stock(self, stock):
        if stock in self.stocks:
            self.stocks.remove(stock)
            self.stocks.append(stock)
        return self.stocks

    def get_stock_by_name(self, name):
        for stock in self.stocks:
            if stock.name == name:
                return stock
        return None

    def sell_stock(self, stock):
        if stock in self.stocks:
            self.stocks.remove(stock)
            return stock
        return None

    def buy_stock(self, stock):
        self.stocks.append(stock)
        return self.stocks

    def get_portfolio_value(self):
        total_value = 0
        for stock in self.stocks:
            total_value += stock.value
        return total_value


class ItemObject:

    def __init__(
        self,
        name,
        description,
        value,
        weight,
        quantity,
        item_type="misc",
        item_subtype=None,
        status="new",
        possible_interactions: List[Action] = [],
        action_system: ActionSystem = None,
        coordinates_location=(0, 0),
    ):
        ActionSystem = importlib.import_module("actions").ActionSystem
        Action = importlib.import_module("actions").Action
        self.name = name
        self.description = description
        self.value = value
        self.weight = weight
        self.quantity = quantity
        self.uuid = uuid.uuid4()
        self.item_type = item_type
        self.item_subtype = item_subtype
        self.location = Location(name, 0, 0, 0, 0, ActionSystem())
        self.coordinates_location = coordinates_location
        self.possible_interactions = possible_interactions
        self.usability = True
        self.ownership_history = ["unowned"]
        self.status = status
        self.type_specific_attributes = False
        # self.action_system = action_system

    def __repr__(self):
        return f"Item({self.name}, {self.description}, {self.value}, {self.weight}, {self.quantity})"

    def __str__(self):
        return f"Item named {self.name} with description {self.description} and value {self.value}."

    def __eq__(self, other):
        if not isinstance(other, ItemObject):
            if isinstance(other, dict):
                return self.to_dict() == other
            return False
        return (
            self.name == other.name
            and self.description == other.description
            and self.value == other.value
            and self.weight == other.weight
            and self.quantity == other.quantity
            and self.uuid == other.uuid
            and self.coordinates_location == other.coordinates_location
            and self.item_type == other.item_type
            and self.item_subtype == other.item_subtype
            and self.status == other.status
            and self.type_specific_attributes == other.type_specific_attributes
            and self.usability == other.usability
            and self.ownership_history == other.ownership_history
            and self.location == other.location
        )

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
            elif type(obj).__name__ == "Character":
                Character = importlib.import_module("tiny_characters").Character

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )
            elif type(obj).__name__ == "Location":
                Location = importlib.import_module("tiny_locations").Location

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )

            return obj

        return hash(
            tuple(
                [
                    self.name,
                    self.description,
                    self.value,
                    self.weight,
                    self.quantity,
                    self.uuid,
                    self.coordinates_location,
                    self.item_type,
                    self.item_subtype,
                    self.status,
                    make_hashable(self.type_specific_attributes),
                    self.usability,
                    make_hashable(self.ownership_history),
                    make_hashable(self.location),
                ]
            )
        )

    def get_name(self):
        return self.name

    def set_name(self, name):
        # Warning: Name MUST be unique! Check for duplicates before setting.
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

    def get_id(self):
        return self.uuid

    def set_id(self, id):
        self.uuid = id
        return self.uuid

        self.location = Location(0, 0)
        self.coordinates_location = self.location.get_coordinates()

    def get_location(self):
        return self.location

    def set_location(self, *location):
        if len(location) == 1:
            if isinstance(location[0], Location):
                self.location = location[0]
            elif isinstance(location[0], tuple):
                self.location = Location(location[0][0], location[0][1])
        elif len(location) == 2:
            self.location = Location(location[0], location[1])

        return self.location

    def get_coordinate_location(self):
        return self.location.get_coordinates()

    def set_coordinate_location(self, *coordinates):
        if len(coordinates) == 1:
            if isinstance(coordinates[0], tuple):
                self.location.set_coordinates(coordinates[0])
        elif len(coordinates) == 2:
            self.location.set_coordinates(coordinates)

        return self.location.coordinates_location

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "weight": self.weight,
            "quantity": self.quantity,
            "uuid": self.uuid,
            "coordinates_location": self.coordinates_location,
        }


effect_dict = {
    "Eat": [
        {"targets": ["initiator"], "attribute": "hunger"},
        {
            "targets": ["initiator"],
            "method": "play_animation",
            "method_args": ["eating"],
        },
    ]
}

preconditions_dict = {
    "Eat": [
        {
            "name": "energy",
            "attribute": "energy",
            "satisfy_value": 10,
            "target": "initiator",
            "operator": "gt",
        },
        {
            "name": "extraversion",
            "attribute": "personality_traits.extraversion",
            "satisfy_value": 50,
            "target": "initiator",
            "operator": "gt",
        },
    ],
    "Open": [
        {
            "name": "energy",
            "attribute": "energy",
            "satisfy_value": 10,
            "target": "initiator",
            "operator": "gt",
        }
    ],
}


class FoodItem(ItemObject):

    def __init__(
        self,
        name,
        description,
        value,
        perishable,
        weight,
        quantity,
        action_system: ActionSystem,
        calories=0,
        cooked=False,
        coordinates_location=(0, 0),
    ):
        ActionSystem = importlib.import_module("actions").ActionSystem
        Action = importlib.import_module("actions").Action
        effect = effect_dict["Eat"]
        effect[0]["change_value"] = calories
        possible_interactions = [
            Action(
                "Eat Food",
                action_system.instantiate_conditions(preconditions_dict["Eat"]),
                effects=effect_dict["Eat"],
                cost=1,
            ),
        ]
        super().__init__(
            name,
            description,
            value,
            weight,
            quantity,
            item_type="food",
            item_subtype=name,
            possible_interactions=possible_interactions,
            coordinates_location=coordinates_location,
        )
        self.calories = calories
        self.item_type = "food"
        self.possible_interactions = possible_interactions
        self.perishable = perishable
        self.cooked = cooked
        self.type_specific_attributes = True

    def __repr__(self):
        return f"FoodItem({self.name}, {self.description}, {self.value}, {self.weight}, {self.quantity}, {self.calories})"

    def __str__(self):
        return f"FoodItem named {self.name} with description {self.description} and value {self.value}."

    def __eq__(self, other):
        if not isinstance(other, FoodItem):
            if isinstance(other, dict):
                return self.to_dict() == other
            if isinstance(other, ItemObject):
                return (
                    self.name == other.name
                    and self.description == other.description
                    and self.value == other.value
                    and self.weight == other.weight
                    and self.quantity == other.quantity
                    and self.calories == other.calories
                    and self.perishable == other.perishable
                    and self.cooked == other.cooked
                    and self.type_specific_attributes == other.type_specific_attributes
                    and self.item_type == other.item_type
                    and self.status == other.status
                    and self.usability == other.usability
                    and self.ownership_history == other.ownership_history
                    and self.location == other.location
                    and self.coordinates_location == other.coordinates_location
                )
        return (
            self.name == other.name
            and self.description == other.description
            and self.value == other.value
            and self.weight == other.weight
            and self.quantity == other.quantity
            and self.calories == other.calories
            and self.perishable == other.perishable
            and self.cooked == other.cooked
            and self.type_specific_attributes == other.type_specific_attributes
            and self.item_type == other.item_type
            and self.status == other.status
            and self.usability == other.usability
            and self.ownership_history == other.ownership_history
            and self.location == other.location
            and self.coordinates_location == other.coordinates_location
        )

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
            elif type(obj).__name__ == "Character":
                Character = importlib.import_module("tiny_characters").Character

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )
            elif type(obj).__name__ == "Location":
                Location = importlib.import_module("tiny_locations").Location

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )

            return obj

        return hash(
            tuple(
                [
                    self.name,
                    self.description,
                    self.value,
                    self.weight,
                    self.quantity,
                    self.calories,
                    self.perishable,
                    self.cooked,
                    self.uuid,
                    make_hashable(self.type_specific_attributes),
                    self.item_type,
                    self.status,
                    self.usability,
                    make_hashable(self.ownership_history),
                    make_hashable(self.location),
                    self.coordinates_location,
                ]
            )
        )

    def get_possible_interactions(self):
        return self.possible_interactions

    def get_calories(self):
        return self.calories

    def set_calories(self, calories):
        self.calories = calories
        return self.calories

    def get_type_specific_attributes(self):
        return {
            "perishable": self.perishable,
            "cooked": self.cooked,
            "calories": self.calories,
        }

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "weight": self.weight,
            "quantity": self.quantity,
            "id": self.uuid,
            "coordinates_location": self.coordinates_location,
            "calories": self.calories,
            "perishable": self.perishable,
            "cooked": self.cooked,
            "type_specific_attributes": self.type_specific_attributes,
        }


class Door(ItemObject):
    def __init__(
        self,
        name,
        description,
        value,
        weight,
        quantity,
        action_system: ActionSystem,
    ):
        ActionSystem = importlib.import_module("actions").ActionSystem
        Action = importlib.import_module("actions").Action
        effect = effect_dict["Open"]
        possible_interactions = [
            Action(
                "Open Door",
                action_system.instantiate_conditions(preconditions_dict["Open"]),
                effects=effect_dict["Open"],
                cost=1,
            ),
        ]
        super().__init__(
            name,
            description,
            value,
            weight,
            quantity,
            item_type="door",
            possible_interactions=possible_interactions,
        )
        self.item_type = "door"
        self.possible_interactions = possible_interactions

    def __repr__(self):
        return f"Door({self.name}, {self.description}, {self.value}, {self.weight}, {self.quantity})"

    def __str__(self):
        return f"Door named {self.name} with description {self.description} and value {self.value}."

    def __eq__(self, other):
        if not isinstance(other, Door):
            if isinstance(other, dict):
                return self.to_dict() == other
            if isinstance(other, ItemObject):
                return (
                    self.name == other.name
                    and self.description == other.description
                    and self.value == other.value
                    and self.weight == other.weight
                    and self.quantity == other.quantity
                    and self.type_specific_attributes == other.type_specific_attributes
                    and self.item_type == other.item_type
                    and self.status == other.status
                    and self.usability == other.usability
                    and self.ownership_history == other.ownership_history
                    and self.location == other.location
                )
        return (
            self.name == other.name
            and self.description == other.description
            and self.value == other.value
            and self.weight == other.weight
            and self.quantity == other.quantity
            and self.type_specific_attributes == other.type_specific_attributes
            and self.item_type == other.item_type
            and self.status == other.status
            and self.usability == other.usability
            and self.ownership_history == other.ownership_history
            and self.location == other.location
        )

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.name,
                    self.description,
                    self.value,
                    self.weight,
                    self.quantity,
                    self.uuid,
                    self.coordinates_location,
                    self.item_type,
                    self.status,
                    tuple(self.type_specific_attributes),
                    self.usability,
                    tuple(self.ownership_history),
                    self.location,
                ]
            )
        )

    def get_possible_interactions(self):
        return self.possible_interactions

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "weight": self.weight,
            "quantity": self.quantity,
            "id": self.uuid,
            "coordinates_location": self.coordinates_location,
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
        self.all_items = []
        self.food_items = []
        self.clothing_items = []
        self.tools_items = []
        self.weapons_items = []
        self.medicine_items = []
        self.misc_items = []
        if food_items is not None:
            self.food_items = self.set_food_items(food_items)
        if clothing_items is not None:
            self.clothing_items = self.set_clothing_items(clothing_items)
        if tools_items is not None:
            self.tools_items = self.set_tools_items(tools_items)
        if weapons_items is not None:
            self.weapons_items = self.set_weapons_items(weapons_items)
        if medicine_items is not None:
            self.medicine_items = self.set_medicine_items(medicine_items)
        if misc_items is not None:
            self.misc_items = self.set_misc_items(misc_items)
        # make one list of all items
        if self.all_items == [] and (
            self.food_items
            or self.clothing_items
            or self.tools_items
            or self.weapons_items
            or self.medicine_items
            or self.misc_items
        ):
            self.all_items = (
                self.food_items
                + self.clothing_items
                + self.tools_items
                + self.weapons_items
                + self.medicine_items
                + self.misc_items
            )

        self.ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "eq": operator.eq,
            "ge": operator.ge,
            "le": operator.le,
            "ne": operator.ne,
        }
        self.symb_map = {
            ">": "gt",
            "<": "lt",
            "==": "eq",
            ">=": "ge",
            "<=": "le",
            "!=": "ne",
        }

    def report_inventory(self):
        report = {}
        self.get_all_items()
        for item in self.all_items:
            report[item] = item.get_quantity()
        return report

    def __repr__(self):
        return f"ItemInventory({self.food_items}, {self.clothing_items}, {self.tools_items}, {self.weapons_items}, {self.medicine_items}, {self.misc_items})"

    def __str__(self):
        return f"ItemInventory with food items {self.food_items}, clothing items {self.clothing_items}, tools items {self.tools_items}, weapons items {self.weapons_items}, medicine items {self.medicine_items}, misc items {self.misc_items}."

    def __eq__(self, other):
        if not isinstance(other, ItemInventory):
            return False
        return (
            self.food_items == other.food_items
            and self.clothing_items == other.clothing_items
            and self.tools_items == other.tools_items
            and self.weapons_items == other.weapons_items
            and self.medicine_items == other.medicine_items
            and self.misc_items == other.misc_items
        )

    def get_all_items(self):
        self.all_items = (
            self.food_items
            + self.clothing_items
            + self.tools_items
            + self.weapons_items
            + self.medicine_items
            + self.misc_items
        )
        return self.all_items

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
            elif type(obj).__name__ == "Character":
                Character = importlib.import_module("tiny_characters").Character

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )
            elif type(obj).__name__ == "Location":
                Location = importlib.import_module("tiny_locations").Location

                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
                )

            return obj

        return hash(
            tuple(
                [
                    make_hashable(self.food_items),
                    make_hashable(self.clothing_items),
                    make_hashable(self.tools_items),
                    make_hashable(self.weapons_items),
                    make_hashable(self.medicine_items),
                    make_hashable(self.misc_items),
                ]
            )
        )

    def add_item(self, item: ItemObject):
        item_lists = {
            "food": self.food_items,
            "clothing": self.clothing_items,
            "tools": self.tools_items,
            "weapons": self.weapons_items,
            "medicine": self.medicine_items,
            "misc": self.misc_items,
        }

        for existing_item in item_lists[item.item_type]:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity += item.quantity
                break
        else:
            item_lists[item.item_type].append(item)

        for existing_item in self.all_items:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity += item.quantity
                break
        else:
            self.all_items.append(item)

        return item

    def remove_item(self, item: ItemObject):
        item_lists = {
            "food": self.food_items,
            "clothing": self.clothing_items,
            "tools": self.tools_items,
            "weapons": self.weapons_items,
            "medicine": self.medicine_items,
            "misc": self.misc_items,
        }

        for existing_item in item_lists[item.item_type]:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity -= item.quantity
                if existing_item.quantity <= 0:
                    item_lists[item.item_type].remove(existing_item)
                break

        for existing_item in self.all_items:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity -= item.quantity
                if existing_item.quantity <= 0:
                    self.all_items.remove(existing_item)
                break

        return item

    def get_food_items(self):
        return self.food_items

    def set_food_items(self, food_items: List[FoodItem]):
        self.food_items = food_items
        self.all_items += food_items
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

    def count_clothing_items_total(self):
        total = 0
        for item in self.clothing_items:
            total += item.get_quantity()
        return total

    def set_clothing_items(self, clothing_items: List[ItemObject]):
        self.clothing_items = clothing_items
        self.all_items += clothing_items
        return self.clothing_items

    def get_tools_items(self):
        return self.tools_items

    def set_tools_items(self, tools_items: List[ItemObject]):
        self.tools_items = tools_items
        self.all_items += tools_items
        return self.tools_items

    def count_tools_items_total(self):
        total = 0
        for item in self.tools_items:
            total += item.get_quantity()
        return total

    def get_weapons_items(self):
        return self.weapons_items

    def count_weapons_items_total(self):
        total = 0
        for item in self.weapons_items:
            total += item.get_quantity()
        return total

    def set_weapons_items(self, weapons_items: List[ItemObject]):
        self.weapons_items = weapons_items
        self.all_items += weapons_items
        return self.weapons_items

    def get_medicine_items(self):
        return self.medicine_items

    def set_medicine_items(self, medicine_items: List[ItemObject]):
        self.medicine_items = medicine_items
        self.all_items += medicine_items
        return self.medicine_items

    def count_medicine_items_total(self):
        total = 0
        for item in self.medicine_items:
            total += item.get_quantity()
        return total

    def get_misc_items(self):
        return self.misc_items

    def set_misc_items(self, misc_items: List[ItemObject]):
        self.misc_items = misc_items
        self.all_items += misc_items
        return self.misc_items

    def count_misc_items_total(self):
        total = 0
        for item in self.misc_items:
            total += item.get_quantity()
        return total

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

    def count_total_items_by_type(self, item_type):
        total = 0
        if item_type == "food":
            for item in self.food_items:
                total += item.get_quantity()
        elif item_type == "clothing":
            for item in self.clothing_items:
                total += item.get_quantity()
        elif item_type == "tools":
            for item in self.tools_items:
                total += item.get_quantity()
        elif item_type == "weapons":
            for item in self.weapons_items:
                total += item.get_quantity()
        elif item_type == "medicine":
            for item in self.medicine_items:
                total += item.get_quantity()
        elif item_type == "misc":
            for item in self.misc_items:
                total += item.get_quantity()
        return total

    def count_total_items_by_name(self, name):
        total = 0
        for item in self.food_items:
            if item.get_name() == name:
                total += item.get_quantity()
        for item in self.clothing_items:
            if item.get_name() == name:
                total += item.get_quantity()
        for item in self.tools_items:
            if item.get_name() == name:
                total += item.get_quantity()
        for item in self.weapons_items:
            if item.get_name() == name:
                total += item.get_quantity()
        for item in self.medicine_items:
            if item.get_name() == name:
                total += item.get_quantity()
        for item in self.misc_items:
            if item.get_name() == name:
                total += item.get_quantity()
        return total

    def check_has_item(self, item, amount=1, oper="ge"):
        return bool(True if self.ops[oper](item.get_quantity(), amount) else False)

    def check_has_item_by_name(self, name, amount=1, oper="ge"):
        for item in self.all_items:
            if item.get_name() == name:
                return True if self.ops[oper](item.get_quantity(), amount) else False
        return False

    def check_has_item_by_type(self, item_type, amount=1, oper="ge"):
        if oper not in self.ops:
            oper = self.symb_map[oper]

        if item_type == "food":
            return bool(
                True
                if self.ops[oper](sum(self.count_food_items_total()), amount)
                else False
            )
        elif item_type == "clothing":
            return bool(
                True
                if self.ops[oper](sum(self.count_clothing_items_total()), amount)
                else False
            )
        elif item_type == "tools":
            return bool(
                True
                if self.ops[oper](sum(self.count_tools_items_total()), amount)
                else False
            )
        elif item_type == "weapons":
            return bool(
                True
                if self.ops[oper](sum(self.count_weapons_items_total()), amount)
                else False
            )
        elif item_type == "medicine":
            return bool(
                True
                if self.ops[oper](sum(self.count_medicine_items_total()), amount)
                else False
            )
        elif item_type == "misc":
            return bool(
                True
                if self.ops[oper](sum(self.count_misc_items_total()), amount)
                else False
            )
        else:
            return False

    def check_has_item_by_attribute_value(self, attribute, value, oper="ge"):
        if oper not in self.ops:
            oper = self.symb_map[oper]
        for item in self.all_items:
            return bool(
                True
                if self.ops[oper](item.get_attribute_value(attribute), value)
                else False
            )
        return False

    def to_dict(self):
        return {
            "food": self.food_items,
            "clothing": self.clothing_items,
            "tools": self.tools_items,
            "weapons": self.weapons_items,
            "medicine": self.medicine_items,
            "misc": self.misc_items,
        }
