""" Dynamic and Extensible Actions
Action Templates:
Design actions as templates that can be instantiated with specific parameters. This allows new actions to be defined or modified without hard-coding every possibility.
Actions should include not just conditions and effects but also metadata that defines how they integrate with the graph (e.g., which nodes or edges they affect). 

Dynamic Action Generation:
Implement a system where actions can be generated or modified based on game events, player inputs, or character development.
For instance, if a new technology is discovered in the game, related actions (like "Study Technology") can be dynamically added to the characters' possible actions.

"""

from calendar import c
from mimetypes import init
import operator
import stat
from typing import Any
from venv import create

# from tiny_characters import Character

from tiny_graph_manager import GraphManager as graph_manager


class State:
    """State class to represent the current state of the character/object/etc. It stores attributes and their values."""

    def __init__(self, state_dict):
        self.state_dict = state_dict
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

    def __getitem__(self, key):
        return self.state_dict.get(key, 0)

    def get(self, key, default):
        return self.state_dict.get(key, default)

    def __setitem__(self, key, value):
        self.state_dict[key] = value

    def __str__(self):
        return ", ".join([f"{key}: {val}" for key, val in self.state_dict.items()])

    def compare_to_condition(self, condition):
        if condition.operator not in self.ops:
            return self.ops[self.symb_map[condition.operator]](
                self[condition.attribute], condition.satisfy_value
            )
        return self.ops[condition.operator](
            self[condition.attribute], condition.satisfy_value
        )


class Condition:
    def __init__(self, name, attribute, satisfy_value, op=">="):
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
        self.name = name
        self.satisfy_value = satisfy_value
        self.attribute = attribute
        self.operator = op

    def __str__(self):
        return f"{self.name}: {self.attribute} {self.operator} {self.satisfy_value}"

    def check_condition(self, state: State):
        return state.compare_to_condition(self)

    def __call__(self, state: State):
        return self.check_condition(state)


class Action:
    def __init__(self, name, preconditions, effects, cost, target=None, initiator=None):
        # Warning: Name MUST be unique! Check for duplicates before setting.

        self.name = name
        self.preconditions = (
            preconditions  # Dict of conditions needed to perform the action
        )
        self.effects = effects  # List of Dicts of state changes the action causes, like [{"targets": ["initiator","target"], "attribute": "social_wellbeing", "change_value": 8}]
        self.cost = cost  # Cost to perform the action, for planning optimality
        self.target = None  # Target of the action, if applicable
        self.initiator = None  # Initiator of the action, if applicable

    def to_dict(self):
        return {
            "name": self.name,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "cost": self.cost,
        }

    def conditions_met(self, state: State):
        print(f"Type of energy: {type(self.preconditions['energy'])}")
        print(f"Type of state: {type(state)}")
        print(f"State: {state}")
        print(f"Preconditions: {self.preconditions}")
        return all(precondition(state) for precondition in self.preconditions.values())

    def apply_effects(self, state: State):
        for effect in self.effects:
            state[effect["attribute"]] = (
                state.get(effect["attribute"], 0) + effect["change_value"]
            )

    def force_apply_effects(self, obj):
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

    def execute(self, target=None, initiator=None, extra_targets=[]):
        if initiator is not None:
            self.initiator = initiator
        else:
            raise ValueError("Initiator must be provided to execute action.")
        if target is not None:
            self.target = target
        if self.conditions_met(self.initiator.get_state()):
            for d in self.effects:
                targets = d["targets"]
                for tgt in targets:
                    if tgt == "initiator":
                        self.apply_effects(self.initiator.get_state())
                        if d["method"]:
                            self.invoke_method(
                                self.initiator, d["method"], d["method_args"]
                            )
                    elif tgt == "target":
                        self.apply_effects(self.target.get_state())
                        if d["method"]:
                            self.invoke_method(
                                self.target, d["method"], d["method_args"]
                            )
                    # Next determine if tgt is has a class function get_state
                    elif hasattr(tgt, "get_state"):
                        self.apply_effects(tgt.get_state())
                        if d["method"]:
                            self.invoke_method(tgt, d["method"], d["method_args"])
                    else:
                        if isinstance(tgt, str):
                            if graph_manager.get_node(tgt):
                                node = graph_manager.get_node(tgt)
                                if (
                                    tgt
                                    in graph_manager.type_to_dict_map[node.type].keys()
                                ):
                                    self.apply_effects(
                                        graph_manager.type_to_dict_map[node.type][
                                            tgt
                                        ].get_state()
                                    )
                                if d["method"]:
                                    self.invoke_method(
                                        graph_manager.type_to_dict_map[node.type][tgt],
                                        d["method"],
                                        d["method_args"],
                                    )
                        else:
                            try:
                                classes_list = [
                                    "Character",
                                    "Location",
                                    "Item",
                                    "Event",
                                ]
                                if tgt.__class__.__name__ in classes_list:
                                    self.apply_effects(tgt.get_state())
                                    if d["method"]:
                                        self.invoke_method(
                                            tgt, d["method"], d["method_args"]
                                        )
                            except:
                                try:
                                    self.force_apply_effects(tgt)
                                    if d["method"]:
                                        self.invoke_method(
                                            tgt, d["method"], d["method_args"]
                                        )
                                except AttributeError:
                                    raise ValueError(
                                        "Target must have a get_state method or attribute to apply effects."
                                    )

            return True
        return False


class TalkAction(Action):
    def execute(self):
        print(f"{self.initiator.name} is talking to {self.target.name}")
        self.target.respond_to_talk(self.initiator)


class ExploreAction(Action):
    def execute(self):
        print(f"{self.initiator.name} is exploring {self.target.location}")
        self.target.discover(self.initiator)


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
        self.actions = []

    def add_action(self, action):
        self.actions.append(action)

    def execute(self):
        for action in self.actions:
            action.execute()


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
        study_template = ActionTemplate(
            "Study",
            self.create_precondition("knowledge", "lt", 10),
            {"knowledge": 5, "energy": -10},
            1,
        )
        work_template = ActionTemplate(
            "Work",
            self.create_precondition("energy", "gt", 20),
            {"money": 50, "energy": -20},
            2,
        )
        socialize_template = ActionTemplate(
            "Socialize",
            self.instantiate_conditions(
                [
                    {
                        "name": "Socialize Energy",
                        "attribute": "social",
                        "satisfy_value": 20,
                        "operator": ">=",
                    },
                    {
                        "name": "Socialize Happiness",
                        "attribute": "happiness",
                        "satisfy_value": 10,
                        "operator": ">=",
                    },
                ]
            ),
            {"happiness": 10, "energy": -10},
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
        if action.conditions_met(state):
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
