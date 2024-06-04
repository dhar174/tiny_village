""" Dynamic and Extensible Actions
Action Templates:
Design actions as templates that can be instantiated with specific parameters. This allows new actions to be defined or modified without hard-coding every possibility.
Actions should include not just conditions and effects but also metadata that defines how they integrate with the graph (e.g., which nodes or edges they affect). 

Dynamic Action Generation:
Implement a system where actions can be generated or modified based on game events, player inputs, or character development.
For instance, if a new technology is discovered in the game, related actions (like "Study Technology") can be dynamically added to the characters' possible actions.

"""

import tiny_graph_manager


class State:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def __getitem__(self, key):
        return self.state_dict.get(key, 0)

    def get(self, key, default):
        return self.state_dict.get(key, default)

    def __str__(self):
        return ", ".join([f"{key}: {val}" for key, val in self.state_dict.items()])

    def compare_to_condition(self, condition):
        if condition.operator == ">=":
            return self[condition.attribute] >= condition.satisfy_value
        elif condition.operator == "<=":
            return self[condition.attribute] <= condition.satisfy_value
        elif condition.operator == "==":
            return self[condition.attribute] == condition.satisfy_value
        elif condition.operator == ">":
            return self[condition.attribute] > condition.satisfy_value
        elif condition.operator == "<":
            return self[condition.attribute] < condition.satisfy_value
        else:
            raise ValueError(f"Invalid operator: {condition.operator}")


class Condition:
    def __init__(self, name, attribute, satisfy_value, operator=">="):
        # Check validitiy of operator
        if operator not in [">=", "<=", "==", ">", "<"]:
            raise ValueError(f"Invalid operator: {operator}")
        self.name = name
        self.satisfy_value = satisfy_value
        self.attribute = attribute
        self.operator = operator

    def __str__(self):
        return f"{self.name}: {self.attribute} {self.operator} {self.satisfy_value}"

    def check_condition(self, state):
        return state.compare_to_condition(self)


class Action:
    def __init__(self, name, preconditions, effects, cost):
        self.name = name
        self.preconditions = (
            preconditions  # Dict of conditions needed to perform the action
        )
        self.effects = effects  # Dict of state changes the action causes
        self.cost = cost  # Cost to perform the action, for planning optimality
        self.target = None  # Target of the action, if applicable

    def conditions_met(self, state):
        return all(
            state.get(cond, False) == val for cond, val in self.preconditions.items()
        )

    def apply_effects(self, state):
        for effect, change in self.effects.items():
            state[effect] = state.get(effect, 0) + change
        return state

    def execute(self):
        raise NotImplementedError("Subclasses must implement this method")


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
        return Action(self.name, self.preconditions, self.effects, self.cost)

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
            {"energy": 10},
            {"knowledge": 5, "energy": -10},
            1,
        )
        work_template = ActionTemplate(
            "Work",
            {"energy": 20},
            {"money": 50, "energy": -20},
            2,
        )

        # Add templates to the action generator
        self.action_generator.add_template(study_template)
        self.action_generator.add_template(work_template)

    def generate_actions(self, character):
        # Generate actions based on character attributes
        actions = self.action_generator.generate_actions({"energy": character.energy})
        return actions

    def execute_action(self, action, character):
        if action.conditions_met(character.state):
            character.state = action.apply_effects(character.state)
            character.energy -= action.cost
            return True
        return False
