"""Dynamic and Extensible Actions
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
# Removed unused torch import that was causing import errors


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
        val = self.dict_or_obj  # Start with the underlying data, not the State object itself
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
        related_goal=None, # Added
        graph_manager=None
    ):
        # Warning: Name MUST be unique! Check for duplicates before setting.
        self.impact_rating_on_target = impact_rating_on_target
        self.impact_rating_on_initiator = impact_rating_on_initiator
        self.impact_rating_on_other = impact_rating_on_other
        self.action_id = action_id if action_id else id(self)  # Use unique ID
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
        if graph_manager is not None:
            self.graph_manager = graph_manager
        else:
            # Fallback to the singleton instance if no graph_manager is explicitly passed.
            # This maintains some backward compatibility but explicit passing is preferred.
            try:
                GraphManager_module = importlib.import_module("tiny_graph_manager")
                self.graph_manager = GraphManager_module.GraphManager()
            except ImportError:
                # Graph manager is optional for core Action functionality
                self.graph_manager = None
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
                for precondition in self.preconditions  # Assuming list of Condition objects
            )
        except AttributeError:  # If precondition is not a Condition object
            print(
                f"Warning: Precondition for action {self.name} is not a Condition object or check_condition failed."
            )
            return False  # Or handle differently

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
    def execute(self, character=None, graph_manager=None):
        """
        Executes the action, applying its effects to the involved entities
        and updating the GraphManager.

        Args:
            character (Character, optional): The character initiating the action.
                                            Defaults to self.initiator.
            graph_manager (GraphManager, optional): The graph manager instance.
                                                 Defaults to self.graph_manager.

        Returns:
            bool: True if preconditions are met and effects applied, False otherwise.
        """
        if character:
            self.initiator = character # Ensure self.initiator is set if character is provided

        if graph_manager:
            self.graph_manager = graph_manager # Prefer graph_manager argument if provided

        if not self.graph_manager:
            try:
                GraphManager_module = importlib.import_module("tiny_graph_manager")
                self.graph_manager = GraphManager_module.GraphManager()
            except ImportError:
                # Graph manager is optional for core Action functionality
                self.graph_manager = None
            # print("Warning: Action.execute called without graph_manager, using fallback.")


        if self.preconditions_met():
            for effect in self.effects:
                targets_to_update = []

                # Resolve targets
                if "initiator" in effect.get("targets", []):
                    if self.initiator:
                        targets_to_update.append(self.initiator)
                    else:
                        print(f"Warning: Effect for action '{self.name}' specifies 'initiator' but action has no initiator.")

                if "target" in effect.get("targets", []):
                    if self.target:
                        targets_to_update.append(self.target)
                    else:
                        # If default_target_is_initiator is true and target is None, set target to initiator
                        if self.default_target_is_initiator and self.initiator:
                            self.target = self.initiator
                            targets_to_update.append(self.target)
                        # else:
                            # print(f"Warning: Effect for action '{self.name}' specifies 'target' but action has no target and default_target_is_initiator is false or initiator is None.")

                # TODO: Add logic for other named targets if effects can specify them (e.g. "target_character_in_location")

                for target_obj in targets_to_update:
                    if not target_obj: # Skip if a target resolved to None
                        continue

                    attribute_name = effect['attribute']
                    change_value = effect['change_value']

                    # 1. Apply effect to Python object
                    current_value = getattr(target_obj, attribute_name, 0)
                    new_value = None

                    if isinstance(change_value, (int, float)):
                        new_value = current_value + change_value
                    elif isinstance(change_value, str):
                        if change_value.startswith("add:") and is_numeric(change_value[4:]): # e.g. "add:5"
                             new_value = current_value + float(change_value[4:])
                        elif change_value.startswith("set:"): # e.g. "set:new_value_string" or "set:50"
                            set_val_str = change_value[4:]
                            if is_numeric(set_val_str):
                                new_value = float(set_val_str)
                            else:
                                new_value = set_val_str # Direct assignment for strings
                        elif hasattr(target_obj, change_value) and callable(getattr(target_obj, change_value)):
                            # If change_value is a method name of the target_obj
                            method_to_call = getattr(target_obj, change_value)
                            # This assumes methods for effects don't require arguments or handle them internally
                            # Or, we might need a way to specify arguments in the effect definition
                            method_to_call()
                            # After method call, re-fetch the attribute value if it's supposed to change it
                            new_value = getattr(target_obj, attribute_name, current_value)
                        else: # Direct assignment if not a special string command
                            new_value = change_value
                    elif callable(change_value): # If change_value is a function
                        new_value = change_value(current_value)
                    else: # Default to direct assignment if type is not recognized for operation
                        new_value = change_value

                    if new_value is not None:
                        setattr(target_obj, attribute_name, new_value)

                    # 2. Propagate to GraphManager
                    if self.graph_manager:
                        # Assuming target_obj.uuid or target_obj.name is the graph node ID
                        # Prefer uuid if available
                        graph_node_id = getattr(target_obj, 'uuid', None)
                        if not graph_node_id: # Fallback to name if uuid is not present
                            graph_node_id = getattr(target_obj, 'name', None)

                        if graph_node_id:
                            try:
                                # Ensure the node exists in the graph before updating
                                # This might be implicitly handled by graph_manager or require a check
                                # For now, assume update_node_attribute handles non-existent nodes gracefully or node exists
                                self.graph_manager.update_node_attribute(graph_node_id, attribute_name, new_value)
                                # print(f"GraphManager: Updated '{attribute_name}' for node '{graph_node_id}' to '{new_value}'")

                                # Example for relationship updates (conceptual, adjust based on actual effect structure)
                                if attribute_name.startswith("relationship_"): # e.g. "relationship_strength_with_Alice"
                                    related_char_name = attribute_name.split("_with_")[-1] # Extracts "Alice"
                                    # We need the ID of the related character. Assume it can be fetched or is known.
                                    # This part is highly dependent on how relationships are structured and identified.
                                    # For simplicity, let's assume related_char_name is a valid ID for now.
                                    # In a real scenario, you'd need to resolve related_char_name to its graph ID.
                                    # self.graph_manager.update_edge_attribute(graph_node_id, related_char_name, 'strength', new_value)

                            except Exception as e:
                                print(f"Error updating graph for node '{graph_node_id}', attribute '{attribute_name}': {e}")
                        # else:
                            # print(f"Could not determine graph node ID for target object: {target_obj}")
            return True
        return False

    def __str__(self):
        preconditions_str = (
            str(self.preconditions) if self.preconditions else "[]"
        )  # Changed from {}
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
        return hash(
            (
                self.name,
                self.cost,
                tuple(
                    sorted(self.effects.items())
                    if isinstance(self.effects, dict)
                    else tuple(self.effects)
                ),
            )
        )

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
            self.name == other.name
            and self.cost == other.cost
            and self.effects == other.effects  # Simplistic comparison
            and self.preconditions == other.preconditions  # Simplistic comparison
        )


class TalkAction(Action):
    def __init__(
        self,
        initiator,
        target,
        name="Talk",
        preconditions=None,
        effects=None,
        cost=0.1,
        **kwargs,
    ):
        # Ensure preconditions and effects are lists if provided, or default to empty lists
        _preconditions = preconditions if preconditions is not None else []
        _effects = (
            effects
            if effects is not None
            else [
                # No default social_wellbeing effects - let respond_to_talk method handle this
                # to avoid coupling between hardcoded values and actual implementation
            ]
        )

        # Filter kwargs to only include parameters that Action.__init__ accepts
        # Also ensure graph_manager is passed to super constructor if available in kwargs
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
            "graph_manager",  # Add graph_manager to allowed parameters
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name,
            _preconditions,
            _effects,
            cost,
            target=target,
            initiator=initiator,
            **filtered_kwargs, # This should already pass graph_manager if it was in original kwargs
        )

    def execute(self, character=None, graph_manager=None):
        # self.initiator and self.target are resolved by the base execute method
        # self.graph_manager is also set by the base class __init__ or execute

        # Call the base class execute to apply generic effects and update graph
        base_execution_successful = super().execute(character=character, graph_manager=graph_manager)

        if base_execution_successful:
            # Specific logic for TalkAction after generic effects are applied
            if self.initiator and self.target:
                initiator_name = getattr(self.initiator, "name", str(self.initiator))
                target_name = getattr(self.target, "name", str(self.target))
                print(f"{initiator_name} is talking to {target_name}")

                if hasattr(self.target, "respond_to_talk"):
                    # Assuming respond_to_talk might have its own effects or direct graph interactions
                    # if it has access to the graph_manager.
                    # If respond_to_talk purely modifies attributes that are NOT part of self.effects,
                    # and these need to be in the graph, manual graph updates would be needed here.
                    self.target.respond_to_talk(self.initiator)
            # else:
                # This case should ideally be handled by preconditions or base class logic if initiator/target are mandatory
                # print(f"Warning: Initiator or target not available for TalkAction {self.name}")
            return True
        return False


class ExploreAction(Action):
    def __init__(self, initiator, target, name="Explore", preconditions=None, effects=None, cost=0.2, **kwargs):
        _preconditions = preconditions if preconditions is not None else []
        # Define effects if 'discover' method is expected to change specific attributes
        _effects = effects if effects is not None else [
            # Example effects (uncomment and adjust if needed):
            # {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
            # {"targets": ["target"], "attribute": "is_explored", "change_value": "set:True"}, # Example for setting a flag
        ]
        # Ensure graph_manager is passed to super()__init__
        # Base Action.__init__ handles assigning self.graph_manager
        super().__init__(name, _preconditions, _effects, cost, target=target, initiator=initiator, **kwargs)

    def execute(self, character=None, graph_manager=None):
        # Call base execute first to handle preconditions and generic effects
        if not super().execute(character=character, graph_manager=graph_manager):
            return False # Preconditions not met or base effects application failed

        # Specific logic for ExploreAction after generic effects (if any) are applied
        # self.initiator and self.target should be set by the base class logic
        if self.initiator and self.target and hasattr(self.target, "discover"):
            initiator_name = getattr(self.initiator, "name", "Unknown Initiator")
            target_display_name = getattr(self.target, 'name', str(self.target)) # Use name for display

            print(f"{initiator_name} is exploring {target_display_name}")
            # The discover method might perform its own state changes and potentially graph updates
            # if it has access to a graph_manager instance or if its changes are captured by effects.
            self.target.discover(self.initiator)

            # If self.target.discover() changes attributes that are NOT covered by self.effects
            # and need to be reflected in the graph, manual updates would be needed here.
            # Example:
            # if self.graph_manager and hasattr(self.target, 'some_attribute_changed_by_discover'):
            #    target_node_id = getattr(self.target, 'uuid', getattr(self.target, 'name', None))
            #    if target_node_id:
            #        self.graph_manager.update_node_attribute(
            #            target_node_id,
            #            'some_attribute_changed_by_discover',
            #            self.target.some_attribute_changed_by_discover
            #        )
            return True
        else:
            missing = []
            if not self.initiator: missing.append("initiator")
            if not self.target: missing.append("target")
            if self.target and not hasattr(self.target, "discover"): missing.append("'discover' method on target")

            action_name = getattr(self, "name", "Unnamed ExploreAction")
            print(f"Warning: ExploreAction '{action_name}' did not execute fully due to missing parts: {', '.join(missing)}.")
            return False


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
            graph_manager=parameters.get("graph_manager") # Pass graph_manager
        )

    def add_skill(self, skill):
        self.related_skills.append(skill)

    def create_action(self, action_type, initiator, target, graph_manager=None):
        if action_type == "talk":
            return TalkAction(initiator, target, graph_manager=graph_manager)
        elif action_type == "explore":
            return ExploreAction(initiator, target, graph_manager=graph_manager)
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

    def execute(self, character=None, graph_manager=None):  # Added parameters
        for action in self.actions:
            action.execute(character, graph_manager)  # Pass parameters



# # Example usage
# composite_action = CompositeAction()
# composite_action.add_action(character.talk_to(another_character))
# composite_action.add_action(character.explore(location))
# composite_action.execute()


class ActionGenerator:
    def __init__(self, graph_manager=None):
        self.templates = []
        self.graph_manager = graph_manager

    def add_template(self, template):
        self.templates.append(template)

    def generate_actions(self, parameters):
        # Ensure graph_manager is in parameters for template.instantiate
        if "graph_manager" not in parameters and self.graph_manager is not None:
            parameters["graph_manager"] = self.graph_manager
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
    def __init__(self, graph_manager=None):
        self.action_generator = ActionGenerator(graph_manager=graph_manager)

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
        study_template = ActionTemplate(
            "Study",
            [{"type": "knowledge_lt_10"}],
            [
                {"attribute": "knowledge", "change": 5},
                {"attribute": "energy", "change": -10},
            ],
            1,
        )
        work_template = ActionTemplate(
            "Work",
            [{"type": "energy_gt_20"}],
            [
                {"attribute": "money", "change": 50},
                {"attribute": "energy", "change": -20},
            ],
            2,
        )
        socialize_template = ActionTemplate(
            "Socialize",
            [{"type": "social_gt_20"}, {"type": "happiness_gt_10"}],
            [
                {"attribute": "happiness", "change": 10},
                {"attribute": "energy", "change": -10},
            ],
            1,
        )
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


class EatAction(Action):
    def __init__(self, item_name, initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="Eat",
            initiator=initiator_id,
            target=item_name,
            cost=kwargs.get("cost", 0.1),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {"targets": ["initiator"], "attribute": "hunger", "change_value": -2},
                    {"targets": ["initiator"], "attribute": "energy", "change_value": 1},
                ],
            ),
            **filtered_kwargs,
        )
        self.item_name = item_name
    # No execute override needed if effects are correctly handled by base class


class GoToLocationAction(Action):
    def __init__(self, location_name, initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="GoToLocation",
            initiator=initiator_id,
            target=location_name,
            cost=kwargs.get("cost", 0.2),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {
                        "attribute": "location",
                        "attribute": "location",
                        "change_value": f"set:{location_name}", # Use "set:" for direct assignment
                        "targets": ["initiator"],
                    }
                ],
            ),
            **filtered_kwargs,
        )
        self.location_name = location_name
    # No execute override needed


class NoOpAction(Action):
    def __init__(self, initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="NoOp",
            initiator=initiator_id,
            cost=kwargs.get("cost", 0),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get("effects", []),
            **filtered_kwargs,
        )


class BuyFoodAction(Action):
    def __init__(self, food_type="food", initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="BuyFood",
            initiator=initiator_id,
            target=food_type,
            cost=kwargs.get("cost", 0.3),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {"targets": ["initiator"], "attribute": "money", "change_value": -5},
                    {
                        "targets": ["initiator"],
                        "attribute": "inventory",
                        "change_value": f"add:{food_type}", # Base execute handles "add:"
                    },
                ],
            ),
            **filtered_kwargs,
        )
        self.food_type = food_type
    # No execute override needed


class WorkAction(Action):
    def __init__(self, job_type="current_job", initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="Work",
            initiator=initiator_id,
            target=job_type,
            cost=kwargs.get("cost", 0.4),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {"targets": ["initiator"], "attribute": "money", "change_value": 20},
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -2},
                    {
                        "targets": ["initiator"],
                        "attribute": "job_performance",
                        "change_value": 1,
                    },
                ],
            ),
            **filtered_kwargs,
        )
        self.job_type = job_type
    # No execute override needed


class SleepAction(Action):
    def __init__(self, duration=8, initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="Sleep",
            initiator=initiator_id,
            cost=kwargs.get("cost", 0.0),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {"targets": ["initiator"], "attribute": "energy", "change_value": 5},
                    {"targets": ["initiator"], "attribute": "health", "change_value": 1},
                ],
            ),
            **filtered_kwargs,
        )
        self.duration = duration
    # No execute override needed


class SocialVisitAction(Action):
    def __init__(self, target_person, initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="SocialVisit",
            initiator=initiator_id,
            target=target_person,
            cost=kwargs.get("cost", 0.3),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {
                        "targets": ["initiator"],
                        "attribute": "social_wellbeing",
                        "change_value": 2,
                    },
                    {
                        "targets": ["initiator"],
                        "attribute": "happiness",
                        "change_value": 1,
                    },
                    {
                        "targets": ["target"], # self.target will be target_person object
                        "attribute": "friendship", # Assuming target Character object has 'friendship'
                        "change_value": 1,
                    },
                ],
            ),
            **filtered_kwargs,
        )
        self.target_person = target_person # Actual object is passed as target to super()__init__
    # No execute override needed


class ImproveJobPerformanceAction(Action):
    def __init__(self, method="study", initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="ImproveJobPerformance",
            initiator=initiator_id,
            cost=kwargs.get("cost", 0.5),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {
                        "targets": ["initiator"],
                        "attribute": "job_performance",
                        "change_value": 3,
                    },
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -1},
                ],
            ),
            **filtered_kwargs,
        )
        self.method = method
    # No execute override needed


class PursueHobbyAction(Action):
    def __init__(self, hobby_type="reading", initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="PursueHobby",
            initiator=initiator_id,
            target=hobby_type,
            cost=kwargs.get("cost", 0.2),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {
                        "targets": ["initiator"],
                        "attribute": "happiness",
                        "change_value": 2,
                    },
                    {
                        "targets": ["initiator"],
                        "attribute": "mental_health",
                        "change_value": 1,
                    },
                ],
            ),
            **filtered_kwargs,
        )
        self.hobby_type = hobby_type
    # No execute override needed


class VisitDoctorAction(Action):
    def __init__(self, reason="checkup", initiator_id=None, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        super().__init__(
            name="VisitDoctor",
            initiator=initiator_id,
            cost=kwargs.get("cost", 0.6),
            preconditions=kwargs.get("preconditions", []),
            effects=kwargs.get(
                "effects",
                [
                    {"targets": ["initiator"], "attribute": "health", "change_value": 3},
                    {"targets": ["initiator"], "attribute": "money", "change_value": -10},
                ],
            ),
            **filtered_kwargs,
        )
        self.reason = reason
    # No execute override needed


class GreetAction(Action):
    def __init__(self, initiator, target, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        _preconditions = kwargs.get("preconditions", [])
        _effects = kwargs.get(
            "effects",
            [
                {
                    "targets": ["target"],
                    "attribute": "social_wellbeing",
                    "change_value": 0.5,
                }
            ],
        )
        if "graph_manager" not in filtered_kwargs and "graph_manager" in kwargs: # Pass graph_manager if available
            filtered_kwargs["graph_manager"] = kwargs["graph_manager"]

        super().__init__(
            "Greet",
            _preconditions,
            _effects,
            kwargs.get("cost", 0.05),
            target=target,
            initiator=initiator,
            **filtered_kwargs,
        )


class ShareNewsAction(Action):
    def __init__(self, initiator, target, news_item, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        _preconditions = kwargs.get("preconditions", [])
        _effects = kwargs.get(
            "effects",
            [
                {
                    "targets": ["target"],
                    "attribute": "social_wellbeing",
                    "change_value": 1,
                },
                {
                    "targets": ["target"],
                    "attribute": "knowledge",
                    "change_value": f"set:{news_item}", # Assuming news_item is a value to be set
                },
            ],
        )
        if "graph_manager" not in filtered_kwargs and "graph_manager" in kwargs: # Pass graph_manager if available
            filtered_kwargs["graph_manager"] = kwargs["graph_manager"]

        super().__init__(
            "ShareNews",
            _preconditions,
            _effects,
            kwargs.get("cost", 0.1),
            target=target,
            initiator=initiator,
            **filtered_kwargs,
        )
        self.news_item = news_item


class OfferComplimentAction(Action):
    def __init__(self, initiator, target, compliment_topic, **kwargs):
        # Filter kwargs to only include parameters that Action.__init__ accepts
        base_action_params = {
            "related_skills",
            "default_target_is_initiator",
            "impact_rating_on_target",
            "impact_rating_on_initiator",
            "impact_rating_on_other",
            "action_id",
            "created_at",
            "expires_at",
            "completed_at",
            "priority",
            "related_goal",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_action_params}

        _preconditions = kwargs.get("preconditions", [])
        _effects = kwargs.get(
            "effects",
            [
                {
                    "targets": ["target"],
                    "attribute": "social_wellbeing",
                    "change_value": 1.5,
                },
                {
                    "targets": ["target"],
                    "attribute": "relationship_strength",
                    "change_value": 1,
                },
            ],
        )
        if "graph_manager" not in filtered_kwargs and "graph_manager" in kwargs: # Pass graph_manager if available
            filtered_kwargs["graph_manager"] = kwargs["graph_manager"]

        super().__init__(
            "OfferCompliment",
            _preconditions,
            _effects,
            kwargs.get("cost", 0.1),
            target=target,
            initiator=initiator,
            **filtered_kwargs,
        )
        self.compliment_topic = compliment_topic
