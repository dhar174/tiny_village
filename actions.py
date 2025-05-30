""" Dynamic and Extensible Actions
Action Templates:
Design actions as templates that can be instantiated with specific parameters. This allows new actions to be defined or modified without hard-coding every possibility.
Actions should include not just conditions and effects but also metadata that defines how they integrate with the graph (e.g., which nodes or edges they affect). 

Dynamic Action Generation:
Implement a system where actions can be generated or modified based on game events, player inputs, or character development.
For instance, if a new technology is discovered in the game, related actions (like "Study Technology") can be dynamically added to the characters' possible actions.

"""

import importlib

import json
import operator

from pyparsing import Char
from torch import Graph


# from tiny_characters import Character


# from tiny_types import GraphManager as self.graph_manager
from tiny_types import Character, Location, ItemObject, Event
from tiny_util_funcs import is_numeric

# from tiny_graph_manager import GraphManager

# self.graph_manager = GraphManager()


class State:
    """State class to represent the current state of the character/object/etc. It stores attributes and their values. All game objects have a state, and actions can change these states."""

    def __init__(self, dict_or_obj):
        self.dict_or_obj = dict_or_obj
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

    def __getitem__(
        self, key
    ):  # if wanting to call a function, use key as a string with arguments in parentheses like: self['foo(["arg1", "arg2"])']
        if "(" in key and key.endswith(")"):
            key, arg_str = key[:-1].split("(", 1)
            args = json.loads(arg_str)
        else:
            args = []

        keys = key.split(".")
        val = self
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, 0)
            else:
                val = getattr(val, k, 0)

        if callable(val):
            return val(*args)
        else:
            return val

    def get(self, key, default=None):
        keys = key.split(".")
        val = self.dict_or_obj
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = getattr(val, k, None)
            if val is None:
                return default
        return val

    def __setitem__(self, key, value):
        self.dict_or_obj[key] = value

    def __str__(self):
        if self.dict_or_obj is None:
            return "State object with no data"
        if isinstance(self.dict_or_obj, dict):
            return ", ".join([f"{key}: {val}" for key, val in self.dict_or_obj.items()])
        elif type(self.dict_or_obj).__name__ == "Character":
            Character = importlib.import_module("tiny_characters").Character

            return f"State object of Character object {self.dict_or_obj.name}"
        elif type(self.dict_or_obj).__name__ == "Location":
            Location = importlib.import_module("tiny_locations").Location

            return f"State object of Location object {self.dict_or_obj.name} with coordinates ({self.dict_or_obj.x}, {self.dict_or_obj.y})"
        elif type(self.dict_or_obj).__name__ == "ItemObject":
            ItemObject = importlib.import_module("tiny_items").ItemObject

            return f"State object of Item object {self.dict_or_obj.name} with type {self.dict_or_obj.item_type} and value {self.dict_or_obj.value}, and weight {self.dict_or_obj.weight}, and quantity {self.dict_or_obj.quantity}"
        elif type(self.dict_or_obj).__name__ == "Event":
            Event = importlib.import_module("tiny_events").Event

            return f"State object of Event object {self.dict_or_obj.name} with date {self.dict_or_obj.date} and type {self.dict_or_obj.event_type} and importance {self.dict_or_obj.importance} and impact {self.dict_or_obj.impact} at coordinates {self.dict_or_obj.coordinates_location}"
        elif type(self.dict_or_obj).__name__ == "Goal":
            Goal = importlib.import_module("tiny_characters").Goal

            return f"State object of Goal object {self.dict_or_obj.name} with priority {self.dict_or_obj.priority} and deadline {self.dict_or_obj.deadline} and description {self.dict_or_obj.description}"
        elif type(self.dict_or_obj).__name__ == "Plan":
            Plan = importlib.import_module("tiny_characters").Plan

            return f"State object of Plan object {self.dict_or_obj.name} with actions {self.dict_or_obj.actions} and goal {self.dict_or_obj.goal}"
        elif type(self.dict_or_obj).__name__ == "Building":
            Building = importlib.import_module("tiny_buildings").Building

            return f"State object of Building object {self.dict_or_obj.name} with type {self.dict_or_obj.building_type} and coordinates {self.dict_or_obj.x}, {self.dict_or_obj.y} and size {self.dict_or_obj.area()} and value {self.dict_or_obj.price_value}. It has {self.dict_or_obj.num_rooms} rooms and {self.dict_or_obj.stories} floors and is owned by {self.dict_or_obj.owner}"
        elif isinstance(self.dict_or_obj, str):
            return self.dict_or_obj
        elif isinstance(self.dict_or_obj, list):
            return ", ".join([str(item) for item in self.dict_or_obj])
        elif isinstance(self.dict_or_obj, set):
            return ", ".join([str(item) for item in self.dict_or_obj])
        elif isinstance(self.dict_or_obj, tuple):
            return ", ".join([str(item) for item in self.dict_or_obj])

    def compare_to_condition(self, condition):

        if condition.operator not in self.ops:
            return self.ops[self.symb_map[condition.operator]](
                self[condition.attribute], condition.satisfy_value
            )
        return self.ops[condition.operator](
            self[condition.attribute], condition.satisfy_value
        )

    def __eq__(self, other):
        if not isinstance(other, State):
            if isinstance(other, dict):
                return self.dict_or_obj == other
            if isinstance(other, list):
                return self.dict_or_obj == other
            if isinstance(other, set):
                return self.dict_or_obj == other
            if isinstance(other, tuple):
                return self.dict_or_obj == other
            if isinstance(other, str):
                return self.dict_or_obj == other
            if isinstance(other, int):
                return self.dict_or_obj == other
            if isinstance(other, float):
                return self.dict_or_obj == other
            if isinstance(other, bool):
                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Character":
                Character = importlib.import_module("tiny_characters").Character

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Location":
                Location = importlib.import_module("tiny_locations").Location

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "ItemObject":
                ItemObject = importlib.import_module("tiny_items").ItemObject

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Event":
                Event = importlib.import_module("tiny_events").Event

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Goal":
                Goal = importlib.import_module("tiny_characters").Goal

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Plan":
                Plan = importlib.import_module("tiny_characters").Plan

                return self.dict_or_obj == other
            if type(self.dict_or_obj).__name__ == "Building":
                Building = importlib.import_module("tiny_buildings").Building

                return self.dict_or_obj == other
            return False

        return self.dict_or_obj == other.dict_or_obj

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

        return hash(make_hashable(self.dict_or_obj))


class Condition:
    def __init__(self, name, attribute, target, satisfy_value, op=">=", weight=1):
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
        # Check validitiy of operator
        if op not in self.ops:
            op = self.symb_map[op]
        if op not in self.ops:
            raise ValueError(f"Invalid operator: {op}")
            # Check type of satisfy_value and ensure correct operator is used
        if isinstance(satisfy_value, str) and op not in ["eq", "ne"]:
            raise ValueError(f"Invalid operator: {op} for string type satisfy_value")
        elif isinstance(satisfy_value, bool) and op not in ["eq", "ne"]:
            raise ValueError(f"Invalid operator: {op} for boolean type satisfy_value")

        self.name = name
        self.satisfy_value = satisfy_value
        self.attribute = attribute
        self.operator = op
        self.target = target
        self.target_id = target.uuid if hasattr(target, "uuid") else id(target)
        self.weight = weight

    def __str__(self):
        return f"{self.name}: {self.attribute} {self.operator} {self.satisfy_value}"

    def check_condition(self, state: State = None):
        if state is None:
            state = self.target.get_state()
        return state.compare_to_condition(self)

    def __call__(self, state: State = None):
        if state is None:
            state = self.target.get_state()
        return self.check_condition(state)

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.name,
                    self.attribute,
                    self.target_id,
                    self.satisfy_value,
                    self.operator,
                    self.weight,
                ]
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return False
        return (
            self.name == other.name
            and self.attribute == other.attribute
            and self.target_id == other.target_id
            and self.satisfy_value == other.satisfy_value
            and self.operator == other.operator
            and self.weight == other.weight
        )


class Action:
    def __init__(
        self,
        name,
        preconditions,
        effects,
        cost=0,
        target=None,
        initiator=None,
        related_skills=[],
        # change_value=None,
        default_target_is_initiator=False,
        impact_rating_on_target=None,  # number representing the weight of the impact on the target ranging from 0 to 3, with 0 being no impact and 3 being a high impact
        impact_rating_on_initiator=None,  # number representing the weight of the impact on the initiator ranging from 0 to 3
        impact_rating_on_other=None,  # Dict with keys like "proximity" and "relationship" and values ranging from 0 to 3 representing the weight of the impact on other characters as defined by the keys
        action_id=None, # Added for consistency with new actions
        created_at=None, # Added
        expires_at=None, # Added
        completed_at=None, # Added
        priority=None, # Added
        related_goal=None # Added
    ):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        # Warning: Name MUST be unique! Check for duplicates before setting.
        self.impact_rating_on_target = impact_rating_on_target
        self.impact_rating_on_initiator = impact_rating_on_initiator
        self.impact_rating_on_other = impact_rating_on_other
        self.action_id = action_id if action_id else id(self) # Use unique ID
        self.created_at = created_at
        self.expires_at = expires_at
        self.completed_at = completed_at
        self.priority = priority
        self.related_goal = related_goal
        self.name = name
        self.preconditions = (
            preconditions  # Dict of conditions needed to perform the action
        )
        self.effects = effects  # List of Dicts of state changes the action causes, like [{"targets": ["initiator","target"], "attribute": "social_wellbeing", "change_value": 8}]
        # calculate an emotional impact value per target based on the effects in self.effects

        self.cost = float(cost)  # Cost to perform the action, for planning optimality
        self.target = target  # Target of the action, if applicable
        self.initiator = initiator  # Initiator of the action, if applicable
        # self.change_value = change_value
        self.default_target_is_initiator = default_target_is_initiator
        self.target = None
        if default_target_is_initiator and target is None and initiator is not None:
            self.target = initiator
        elif target is not None:
            self.target = target
        self.change_value = 0
        self.target_id = target.uuid if hasattr(target, "uuid") else id(target)
        self.graph_manager = GraphManager()
        self.related_skills = related_skills

    def to_dict(self):
        return {
            "name": self.name,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "cost": self.cost,
        }

    def preconditions_met(self):

        if not self.preconditions: 
            return True
        # This assumes preconditions is a list of Condition objects
        # If it's a dict from old ActionSystem.create_precondition, this will fail.
        # For new actions, ensure preconditions are Condition objects or adapt this.
        try:
            return all(
                precondition.check_condition() 
                for precondition in self.preconditions # Assuming list of Condition objects
            )
        except AttributeError: # If precondition is not a Condition object
            print(f"Warning: Precondition for action {self.name} is not a Condition object or check_condition failed.")
            return False # Or handle differently

    def apply_single_effect(self, effect, state: State, change_value=None):
        if change_value is None:
            change_value = effect["change_value"]
        if (
            isinstance(effect["change_value"], str)
            and "(" in effect["change_value"]
            and effect["change_value"].endswith(")")
        ):
            state[effect["attribute"]] = state[effect["change_value"]]
        elif effect["change_value"] and is_numeric(effect["change_value"]):
            state[effect["attribute"]] = (
                state.get(effect["attribute"], 0) + effect["change_value"]
            )
        elif callable(change_value):
            state[effect["attribute"]] = change_value(state.get(effect["attribute"], 0))
        elif change_value:
            state[effect["attribute"]] = (
                state.get(effect["attribute"], 0) + change_value
            )
        else:
            raise ValueError(
                "Effect must have a change_value attribute or a change_value parameter must be provided."
            )
        return state

    def apply_effects(self, state: State):
        State = importlib.import_module("tiny_types").State
        try:

            for effect in self.effects:
                change_value = effect["change_value"]
                if (
                    isinstance(effect["change_value"], str)
                    and "(" in effect["change_value"]
                    and effect["change_value"].endswith(")")
                ):
                    state[effect["attribute"]] = state[effect["change_value"]]
                elif effect["change_value"] and is_numeric(effect["change_value"]):
                    state[effect["attribute"]] = (
                        state.get(effect["attribute"], 0) + effect["change_value"]
                    )
                elif callable(change_value):
                    state[effect["attribute"]] = change_value(
                        state.get(effect["attribute"], 0)
                    )
                elif change_value:
                    state[effect["attribute"]] = (
                        state.get(effect["attribute"], 0) + change_value
                    )
                else:
                    raise ValueError(
                        "Effect must have a change_value attribute or a change_value parameter must be provided."
                    )
                return state
        except:
            return self.force_apply_effects(state, change_value)

    def force_apply_effects(self, obj, change_value=None):
        if change_value:
            self.change_value = change_value
        if hasattr(obj, self.attribute):
            setattr(
                obj, self.attribute, getattr(obj, self.attribute) + self.change_value
            )
        elif "." in self.attribute:  # Handle nested attributes
            attrs = self.attribute.split(".")
            value = getattr(obj, attrs[0])
            for attr in attrs[1:-1]:
                value = getattr(value, attr)
            setattr(value, attrs[-1], getattr(value, attrs[-1]) + self.change_value)

    def invoke_method(self, obj, method, method_args=[]):
        if hasattr(obj, self.method):
            method = getattr(obj, method)
            method(*method_args)

    # def execute(self, target=None, initiator=None, extra_targets=[], change_value=None):
    #     Character = importlib.import_module("tiny_characters").Character
    #     if initiator is not None and self.initiator is None:
    #         self.initiator = initiator
    #     else:
    #         raise ValueError("Initiator must be provided to execute action.")
    #     if target is not None and self.target is None:
    #         self.target = target
    #     elif self.target is None:
    #         self.target = self.initiator if self.default_target_is_initiator else None
    #     else:
    #         raise ValueError("Target must be provided to execute action.")
    #     if self.preconditions_met():
    #         for d in self.effects:
    #             targets = d["targets"]
    #             if extra_targets:
    #                 targets.extend(extra_targets)
    #             for tgt in targets:
    #                 if tgt != "initiator":
    #                     if d["attribute"] in [
    #                         "social_wellbeing",
    #                         "happiness",
    #                         "wealth_money",
    #                         "luxury",
    #                         "health_status",
    #                         "mental_health",
    #                         "hunger_level",
    #                         "energy",
    #                         "hope",
    #                         "stability",
    #                         "control",
    #                         "success",
    #                         "shelter",
    #                         "current_mood",
    #                         "stamina",
    #                         "current_satisfaction",
    #                     ]:
    #                         emotional_impact_value = 0
    #                         change_count = 0
    #                         for d["attribute"] in [
    #                             "social_wellbeing",
    #                             "happiness",
    #                             "wealth_money",
    #                             "luxury",
    #                             "health_status",
    #                             "mental_health",
    #                             "hunger_level",
    #                             "energy",
    #                             "hope",
    #                             "stability",
    #                             "control",
    #                             "success",
    #                             "shelter",
    #                             "current_mood",
    #                             "stamina",
    #                             "current_satisfaction",
    #                         ]:
    #                             change_count += 1
    #                             if d["change_value"] > 0:
    #                                 emotional_impact_value += d["change_value"]
    #                             elif d["change_value"] < 0:
    #                                 emotional_impact_value -= d["change_value"]
    #                         emotional_impact_value = (
    #                             emotional_impact_value / change_count
    #                         )
    #                         emotional_impact_value *= 0.01

    #                         if self.graph_manager.get_character(tgt) or (
    #                             self.graph_manager.get_node(tgt)
    #                             and isinstance(
    #                                 self.graph_manager.get_node(tgt), Character
    #                             )
    #                         ):
    #                             if self.graph_manager.G.has_edge(self.initiator, tgt):
    #                                 self.graph_manager.update_character_character_edge(
    #                                     self.initiator,
    #                                     tgt,
    #                                     (
    #                                         self.impact_rating_on_target
    #                                         if tgt == "target"
    #                                         else self.impact_rating_on_other
    #                                     ),
    #                                     emotional_impact_value,
    #                                 )
    #                             self.graph_manager.add_character_character_edge(
    #                                 self.initiator,
    #                                 tgt,
    #                                 d["attribute"],
    #                                 d["change_value"],
    #                             )
    #                         self.graph_manager.add_character_character_edge(
    #                             self.initiator, tgt, d["attribute"], d["change_value"]
    #                         )

    #                 if tgt == "initiator":

    #                     self.apply_effects(self.initiator.get_state(), change_value)
    #                     if d["method"]:
    #                         self.invoke_method(
    #                             self.initiator, d["method"], d["method_args"]
    #                         )
    #                 elif tgt == "target":
    #                     self.apply_effects(self.target.get_state(), change_value)
    #                     if d["method"]:
    #                         self.invoke_method(
    #                             self.target, d["method"], d["method_args"]
    #                         )
    #                 # Next determine if tgt is has a class function get_state
    #                 elif hasattr(tgt, "get_state"):
    #                     self.apply_effects(tgt.get_state(), change_value)
    #                     if d["method"]:
    #                         self.invoke_method(tgt, d["method"], d["method_args"])
    #                 else:
    #                     if isinstance(tgt, str):
    #                         if self.graph_manager.get_node(tgt):
    #                             node = self.graph_manager.get_node(tgt)
    #                             if (
    #                                 tgt
    #                                 in self.graph_manager.type_to_dict_map[
    #                                     node.type
    #                                 ].keys()
    #                             ):
    #                                 self.apply_effects(
    #                                     self.graph_manager.type_to_dict_map[node.type][
    #                                         tgt
    #                                     ].get_state(),
    #                                     change_value,
    #                                 )
    #                             if d["method"]:
    #                                 self.invoke_method(
    #                                     self.graph_manager.type_to_dict_map[node.type][
    #                                         tgt
    #                                     ],
    #                                     d["method"],
    #                                     d["method_args"],
    #                                 )
    #                     else:
    #                         Character = importlib.import_module(
    #                             "tiny_characters"
    #                         ).Character
    #                         Location = importlib.import_module(
    #                             "tiny_locations"
    #                         ).Location
    #                         ItemObject = importlib.import_module(
    #                             "tiny_items"
    #                         ).ItemObject

    #                         try:
    #                             classes_list = [
    #                                 "Character",
    #                                 "Location",
    #                                 "Item",
    #                                 "Event",
    #                             ]
    #                             if tgt.__class__.__name__ in classes_list:
    #                                 self.apply_effects(tgt.get_state(), change_value)
    #                                 if d["method"]:
    #                                     self.invoke_method(
    #                                         tgt, d["method"], d["method_args"]
    #                                     )
    #                         except:
    #                             try:
    #                                 self.force_apply_effects(tgt, change_value)
    #                                 if d["method"]:
    #                                     self.invoke_method(
    #                                         tgt, d["method"], d["method_args"]
    #                                     )
    #                             except AttributeError:
    #                                 raise ValueError(
    #                                     "Target must have a get_state method or attribute to apply effects."
    #                                 )

    #         return True
    #     return False
    def execute(self, character=None, graph_manager=None): # Changed signature to match new actions
        # Placeholder execute for base, actual logic in subclasses
        print(f"Executing generic action: {self.name} by {character.name if character else self.initiator} on target {self.target}")
        # This method would apply effects to character.state or graph_manager
        # For now, as graph_manager is None due to workaround, true effects can't be applied here.
        if character and hasattr(character, 'get_state'):
            char_state_obj = character.get_state()
            # self.apply_effects(char_state_obj) # apply_effects needs a dict-like state
            # To use with State object, State class would need to allow direct item assignment
            # or effects must be translated.
            # For now, this part is illustrative of where effects would apply.
            # print(f"State of {character.name} potentially modified by {self.name}.")
            pass
        return True

    def __str__(self):
        preconditions_str = str(self.preconditions) if self.preconditions else "[]" # Changed from {}
        effects_str = str(self.effects) if self.effects else "[]"
        return f"{self.name}: Preconditions: {preconditions_str} -> Effects: {effects_str}, Cost: {self.cost}"
    def add_effect(self, effect):
        self.effects.append(effect)

    def add_precondition(self, precondition):
        self.preconditions.append(precondition)

    # def __hash__(self):
    #     def make_hashable(obj):
    #         if isinstance(obj, dict):
    #             return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    #         elif isinstance(obj, list):
    #             return tuple(make_hashable(e) for e in obj)
    #         elif isinstance(obj, set):
    #             return frozenset(make_hashable(e) for e in obj)
    #         elif isinstance(obj, tuple):
    #             return tuple(make_hashable(e) for e in obj)
    #         elif type(obj).__name__ == "Character":
    #             Character = importlib.import_module("tiny_characters").Character

    #             return tuple(
    #                 sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
    #             )
    #         elif type(obj).__name__ == "Location":
    #             Location = importlib.import_module("tiny_locations").Location

    #             return tuple(
    #                 sorted((k, make_hashable(v)) for k, v in obj.to_dict().items())
    #             )

    #         return obj

    #     return hash(
    #         tuple(
    #             [
    #                 self.name,
    #                 make_hashable(self.preconditions),
    #                 make_hashable(self.effects),
    #                 self.target_id,
    #                 self.cost,
    #                 self.default_target_is_initiator,
    #                 self.impact_rating_on_target,
    #                 self.impact_rating_on_initiator,
    #                 self.impact_rating_on_other,
    #                 self.change_value,
    #                 make_hashable(self.related_skills),
    #             ]
    #         )
    #     )
    def __hash__(self):
        # Simplified hash for workaround compatibility
        return hash((self.name, self.cost, tuple(sorted(self.effects.items()) if isinstance(self.effects, dict) else tuple(self.effects))))
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (
            # self.name == other.name
            # and self.preconditions == other.preconditions
            # and self.effects == other.effects
            # and self.cost == other.cost
            # and self.default_target_is_initiator == other.default_target_is_initiator
            # and self.impact_rating_on_target == other.impact_rating_on_target
            # and self.impact_rating_on_initiator == other.impact_rating_on_initiator
            # and self.impact_rating_on_other == other.impact_rating_on_other
            # and self.change_value == other.change_value
            self.name == other.name and
            self.cost == other.cost and
            self.effects == other.effects and # Simplistic comparison
            self.preconditions == other.preconditions # Simplistic comparison
        )

class TalkAction(Action):
    def __init__(self, initiator, target, name="Talk", preconditions=None, effects=None, cost=0.1, **kwargs):
        # Ensure preconditions and effects are lists if provided, or default to empty lists
        _preconditions = preconditions if preconditions is not None else []
        _effects = effects if effects is not None else [{"attribute": "social_wellbeing", "target_id": str(target), "change": 1, "operator": "add"}]
        super().__init__(name, _preconditions, _effects, cost, target=target, initiator=initiator, **kwargs)

    def execute(self, character=None, graph_manager=None): 
        initiator_obj = character if character else self.initiator
        target_obj = self.target

        initiator_name = getattr(initiator_obj, 'name', str(initiator_obj))
        target_name = getattr(target_obj, 'name', str(target_obj))

        print(f"{initiator_name} is talking to {target_name}")
        if hasattr(target_obj, 'respond_to_talk'):
            target_obj.respond_to_talk(initiator_obj)
        return True


class ExploreAction(Action):
    def execute(self): 
        if hasattr(self.initiator, 'name') and hasattr(self.target, 'location') and hasattr(self.target, 'discover'):
            print(f"{self.initiator.name} is exploring {self.target.location}")
            self.target.discover(self.initiator)
        else:
            print(f"Warning: ExploreAction executed with incomplete initiator/target for {self.name}")
        return True # Added return


class ActionTemplate:
    def __init__(self, name, preconditions, effects, cost, related_skills=[]):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.cost = cost
        self.related_skills = related_skills

    def instantiate(self, parameters):
        if "initiator" not in parameters or "target" not in parameters:
            raise ValueError(
                "Initiator and target must be provided to instantiate action."
            )
        return Action(
            self.name,
            self.preconditions,
            self.effects,
            self.cost,
            parameters["target"],
            parameters["initiator"],
        )

    def add_skill(self, skill):
        self.related_skills.append(skill)

    def create_action(self, action_type, initiator, target):
        if action_type == "talk":
            return TalkAction(initiator, target)
        elif action_type == "explore":
            return ExploreAction(initiator, target)
        else:
            raise ValueError("Unknown action type")


# In game loop or handler
""" action = character.talk_to(another_character)
action.execute()

action = character.explore(location)
action.execute() """


class CompositeAction(Action):
    def __init__(self):
        super().__init__(name="CompositeAction", preconditions=[], effects=[]) 
        self.actions = []

    def add_action(self, action):
        self.actions.append(action)

    def execute(self, character=None, graph_manager=None): # Added parameters
        for action in self.actions:
            action.execute(character, graph_manager) # Pass parameters


# # Example usage
# composite_action = CompositeAction()
# composite_action.add_action(character.talk_to(another_character))
# composite_action.add_action(character.explore(location))
# composite_action.execute()


class ActionGenerator:
    def __init__(self):
        self.templates = []

    def add_template(self, template):
        self.templates.append(template)

    def generate_actions(self, parameters):
        return [template.instantiate(parameters) for template in self.templates]


class Skill:
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def __str__(self):
        return f"{self.name} (Level {self.level})"

    def __hash__(self):
        return hash((self.name, self.level))

    def __eq__(self, other):
        if not isinstance(other, Skill):
            return False
        return self.name == other.name and self.level == other.level


class JobSkill(Skill):
    def __init__(self, name, level, job):
        super().__init__(name, level)
        self.job = job

    def __str__(self):
        return f"{self.name} (Level {self.level}) - {self.job}"


class ActionSkill(Skill):
    def __init__(self, name, level, action):
        super().__init__(name, level)
        self.action = action

    def __str__(self):
        return f"{self.name} (Level {self.level}) - {self.action}"


class ActionSystem:
    def __init__(self):
        self.action_generator = ActionGenerator()

    def setup_actions(self):
        # Define action templates
        # study_template = ActionTemplate(
        #     "Study",
        #     self.create_precondition("knowledge", "lt", 10),
        #     {"knowledge": 5, "energy": -10},
        #     1,
        # )
        # work_template = ActionTemplate(
        #     "Work",
        #     self.create_precondition("energy", "gt", 20),
        #     {"money": 50, "energy": -20},
        #     2,
        # )
        # socialize_template = ActionTemplate(
        #     "Socialize",
        #     self.instantiate_conditions(
        #         [
        #             {
        #                 "name": "Socialize Energy",
        #                 "attribute": "social",
        #                 "satisfy_value": 20,
        #                 "operator": ">=",
        #             },
        #             {
        #                 "name": "Socialize Happiness",
        #                 "attribute": "happiness",
        #                 "satisfy_value": 10,
        #                 "operator": ">=",
        #             },
        #         ]
        #     ),
        #     {"happiness": 10, "energy": -10},
        #     1,
        # )
        study_template = ActionTemplate("Study", [{"type": "knowledge_lt_10"}], [{"attribute": "knowledge", "change": 5}, {"attribute": "energy", "change": -10}], 1)
        work_template = ActionTemplate("Work", [{"type": "energy_gt_20"}], [{"attribute": "money", "change": 50}, {"attribute": "energy", "change": -20}], 2)
        socialize_template = ActionTemplate("Socialize", [{"type": "social_gt_20"}, {"type": "happiness_gt_10"}], [{"attribute": "happiness", "change": 10}, {"attribute": "energy", "change": -10}], 1)
        # Add templates to the action generator
        self.action_generator.add_template(study_template)
        self.action_generator.add_template(work_template)
        self.action_generator.add_template(socialize_template)

    def generate_actions(self, initiator, target=None):
        if initiator is not isinstance(initiator, str):
            # Get name attribute of the initiator
            try:
                initiator = initiator.name
            except AttributeError:
                try:
                    initiator = initiator["name"]
                except KeyError:
                    try:
                        # use python system methods to get the name attribute of the class instance
                        initiator = initiator.__name__
                    except AttributeError:
                        raise ValueError("Initiator must have a name attribute")

        # Generate actions based on character attributes
        if initiator is not None:
            return self.action_generator.generate_actions(
                {"initiator": initiator, "target": None}
            )
        # Generate generic actions
        return self.action_generator.generate_actions(
            {"initiator": None, "target": None}
        )

    def execute_action(self, action, state: State):
        if action.preconditions_met(state):
            state = action.apply_effects(state)
            state["energy"] -= action.cost
            return True
        return False

    def create_precondition(self, attribute, op, value):
        ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "eq": operator.eq,
            "ge": operator.ge,
            "le": operator.le,
            "ne": operator.ne,
        }
        if op not in ops:
            raise ValueError(f"Invalid operator: {op}")

        def precondition(state):
            return ops[op](state.get(attribute, 0), value)

        print(f"Precondition: {precondition}")

        return {f"{attribute}": precondition}

    def instantiate_condition(self, condition_dict):
        return Condition(
            condition_dict["name"],
            condition_dict["attribute"],
            condition_dict["target"],
            condition_dict["satisfy_value"],
            condition_dict["operator"],
        )

    def instantiate_conditions(self, conditions_list):
        # check if is not a list
        if not isinstance(conditions_list, list):
            if isinstance(conditions_list, Condition):
                return [conditions_list]
        # check the type of the entries in conditions_list
        if type(conditions_list[0]) == dict:
            return {
                cond["attribute"]: self.instantiate_condition(cond)
                for cond in conditions_list
            }
        elif type(conditions_list[0]) == Condition:
            return {cond.attribute: cond for cond in conditions_list}