# This file contains the Character class, which is used to represent a character in the game.

from ast import arg
import heapq

# import imp  # Removed - deprecated in Python 3.12
import importlib
from math import e
import random
import re
from typing import List
import uuid
import attr
from tiny_types import PromptBuilder, GraphManager
# Removed incorrect torch import - Graph, eq, rand not used and Graph is not a torch function
import logging

logging.basicConfig(level=logging.DEBUG)
from tiny_types import House, CreateBuilding, Action, State, ActionSystem

from actions import (
    ActionGenerator,
    ActionTemplate,
    Condition,
    Skill,
    JobSkill,
    ActionSkill,
    State,
)
from tiny_event_handler import Event
import tiny_utility_functions as goap_planner
from tiny_goap_system import GOAPPlanner
from tiny_jobs import JobRoles, JobRules, Job


from tiny_util_funcs import ClampedIntScore, tweener

from tiny_items import ItemInventory, FoodItem, ItemObject, InvestmentPortfolio, Stock

# GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
from tiny_memories import Memory, MemoryManager  # Temporarily commented out for testing
from tiny_time_manager import GameTimeManager

from tiny_locations import Location, LocationManager


# def gaussian(input_value, mean, std):
#     return (1 / (std * (2 * 3.14159) ** 0.5)) * (2.71828 ** ((-1 / 2) * (((input_value - mean) / std) ** 2)))
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(graph, start, goal):
    GraphManager = importlib.import_module("tiny_graph_manager")
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in graph.directional(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    path = []
    current = goal
    while current and current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


class SteeringBehaviors:
    @staticmethod
    def seek(
        character,
        target,
        slow_down_radius=50,
        max_speed=10,
        max_acceleration=5,
        damping=0.9,
    ):
        desired_velocity = (
            target[0] - character.location[0],
            target[1] - character.location[1],
        )
        distance = (desired_velocity[0] ** 2 + desired_velocity[1] ** 2) ** 0.5

        # Normalize the desired velocity
        if distance > 0:
            desired_velocity = (
                desired_velocity[0] / distance,
                desired_velocity[1] / distance,
            )

        # Adjust speed based on distance for arrival behavior
        if distance < slow_down_radius:
            desired_velocity = (
                desired_velocity[0] * max_speed * (distance / slow_down_radius),
                desired_velocity[1] * max_speed * (distance / slow_down_radius),
            )
        else:
            desired_velocity = (
                desired_velocity[0] * max_speed,
                desired_velocity[1] * max_speed,
            )

        # Calculate the steering force
        steering = (
            desired_velocity[0] - character.velocity[0],
            desired_velocity[1] - character.velocity[1],
        )

        # Cap the steering force to max acceleration
        steering_magnitude = (steering[0] ** 2 + steering[1] ** 2) ** 0.5
        if steering_magnitude > max_acceleration:
            steering = (
                steering[0] / steering_magnitude * max_acceleration,
                steering[1] / steering_magnitude * max_acceleration,
            )

        # Apply damping for smooth movement
        character.velocity = (
            character.velocity[0] * damping + steering[0],
            character.velocity[1] * damping + steering[1],
        )

        # Cap the velocity to max speed
        velocity_magnitude = (
            character.velocity[0] ** 2 + character.velocity[1] ** 2
        ) ** 0.5
        if velocity_magnitude > max_speed:
            character.velocity = (
                character.velocity[0] / velocity_magnitude * max_speed,
                character.velocity[1] / velocity_magnitude * max_speed,
            )

        # Return the steering force for potential field combination
        return character.velocity

    @staticmethod
    def avoid(character, obstacles, avoidance_radius):
        avoid_force = (0, 0)
        for obs in obstacles:
            diff = (character.location[0] - obs[0], character.location[1] - obs[1])
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            if distance < avoidance_radius:
                force = (diff[0] / distance, diff[1] / distance)
                # Inverse square law for stronger avoidance force when closer
                avoid_force = (
                    avoid_force[0] + force[0] / (distance**2),
                    avoid_force[1] + force[1] / (distance**2),
                )
        return avoid_force

    @staticmethod
    def potential_field_force(character, potential_field):
        attractive_force = (0, 0)
        repulsive_force = (0, 0)

        # Attractive force towards the goal
        if "goal" in potential_field:
            goal = potential_field["goal"]
            diff = (goal[0] - character.location[0], goal[1] - character.location[1])
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            if distance > 0:
                attractive_force = (
                    diff[0] / distance,
                    diff[1] / distance,
                )

        # Repulsive forces from obstacles
        if "obstacles" in potential_field:
            for obs in potential_field["obstacles"]:
                diff = (character.location[0] - obs[0], character.location[1] - obs[1])
                distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
                if distance > 0:
                    force = (diff[0] / distance**2, diff[1] / distance**2)
                    repulsive_force = (
                        repulsive_force[0] + force[0],
                        repulsive_force[1] + force[1],
                    )

        # Combine forces
        combined_force = (
            attractive_force[0] - repulsive_force[0],
            attractive_force[1] - repulsive_force[1],
        )

        return combined_force


class RandomNameGenerator:
    def __init__(self):
        self.first_names_male = []
        self.first_names_female = []
        self.last_names = []
        self.load_names()

    def load_names(self):
        with open("first_names_she.txt", "r") as f:
            self.first_names_female = f.read().splitlines()
        with open("first_names_he.txt", "r") as f:
            self.first_names_male = f.read().splitlines()
        with open("last_names.txt", "r") as f:
            self.last_names = f.read().splitlines()

    def generate_name(self, pronouns: str = "they"):
        if "she" in pronouns or "her" in pronouns:
            return (
                random.choice(self.first_names_female)
                + " "
                + random.choice(self.last_names)
            )
        elif "he" in pronouns or "him" in pronouns:
            return (
                random.choice(self.first_names_male)
                + " "
                + random.choice(self.last_names)
            )
        else:
            rint = random.randint(0, 1)
            if rint == 0:
                return (
                    random.choice(self.first_names_female)
                    + " "
                    + random.choice(self.last_names)
                )
            else:
                return (
                    random.choice(self.first_names_male)
                    + " "
                    + random.choice(self.last_names)
                )

    # example_desired_results1 = [
    #     {
    #         "target": {"type": "character", "name": "Joe"},
    #         "end_state": {"relationship": "friend", "strength": 0.8},


class Goal:
    """
    Represents a goal for a character in the game.

    Attributes:
        name (str): The name of the goal.
        description (str): A description of the goal.
        score (int): The importance of the goal.
        character (Character): The character who has this goal.
        target (Character, Item, Location, etc): Represents the target of the goal (could be same as character, does not have to be)
        completion_conditions (dict): A dictionary of functions to check if the goal is completed, with the key being a bool representing whether the condition has been met. Example: {False: Condition(name="has_food", attribute="inventory.check_has_item_by_type(['food'])", satisfy_value=True, op="==")}
        evaluate_utility_function (function): A function that evaluates the current importance of the goal based on the character's state and the environment.
        difficulty (function): A function that calculates the difficulty of the goal based on the character's state and the environment.
        completion_reward (function): A function that calculates the reward for completing the goal based on the character's state and the environment.
        target_effects (dict): A dict of the target effects completing the goal will result in, ie, the desired effects of the goal.
        failure_penalty (function): A function that calculates the penalty for failing to complete the goal based on the character's state and the environment.
        completion_message (function): A function that generates a message when the goal is completed based on the character's state and the environment.
        failure_message (function): A function that generates a message when the goal is failed based on the character's state and the environment.
        criteria (list): A list of criteria (as dicts) that need to be met for the goal to be completed.
        required_items (list): A list of items required to complete the goal.

    """

    GraphManager = importlib.import_module("tiny_graph_manager")

    def __init__(
        self,
        description,
        character,
        target,  # Represents the target of the goal (could be same as character, does not have to be)
        score,  # Represents the importance of the goal
        name,
        completion_conditions,  # Dict of list of functions to check if the goal is completed, with the key being a bool representing whether the condition has been met. Example: {False: Condition(name="has_food", attribute="inventory.check_has_item_by_type(['food'])", satisfy_value=True, op="==")}
        evaluate_utility_function,  # function that evaluates the current importance of the goal based on the character's state and the environment.
        difficulty,  # function that calculates the difficulty of the goal based on the character's state and the environment.
        completion_reward,  # function that calculates the reward for completing the goal based on the character's state and the environment.
        failure_penalty,  # function that calculates the penalty for failing to complete the goal based on the character's state and the environment.
        completion_message,  # function that generates a message when the goal is completed based on the character's state and the environment.
        failure_message,  # function that generates a message when the goal is failed based on the character's state and the environment.
        criteria,  # list of criteria (as dicts) that need to be met for the goal to be completed
        graph_manager,
        goal_type,
        target_effects,# list of desired results (as dicts of dicts representing State attributes) when the goal is completed
    ):
        self.name = name
        self.completion_conditions = completion_conditions
        self.description = description
        self.score = score
        self.character = character
        self.evaluate_utility_function = evaluate_utility_function
        self.difficulty = difficulty
        self.completion_reward = completion_reward
        self.failure_penalty = failure_penalty
        self.completion_message = completion_message
        self.failure_message = failure_message
        self.criteria = criteria
        self.required_items = (
            self.extract_required_items()
        )  # list of tuples, each tuple is (dict of item requirements, quantity needed).
        # The dict of item requirements is composed of various keys like item_type, value, usability, sentimental_value, trade_value, scarcity, coordinates_location, name,
        self.target = target
        self.environment = graph_manager
        self.goal_type = goal_type
        self.target_effects = target_effects
        # self.desired_results = desired_results

    def extract_required_items(self):
        ##TODO: We will need multiple paths to adding items to the required items list, including with the agents decisions and with the completion conditions. Some of this will require the agent to remember which items fulfill goals
        required_items = (
            []
        )  # list of tuples, each tuple is (dict of item requirements, quantity needed).
        # The dict of item requirements is composed of various keys like item_type, value, usability, sentimental_value, trade_value, scarcity, coordinates_location, name,
        for criterion in self.criteria:
            if "node_attributes" in criterion:
                if (
                    "item_type" in criterion["node_attributes"]
                    or criterion["node_attributes"]["type"] == "item"
                ):
                    required_items.append(criterion["node_attributes"])
        # logging.debug(f"self.completion_conditions: {self.completion_conditions}")
        for condition_list in self.completion_conditions.values():
            for condition in condition_list:
                # logging.debug(
                #     f"Condition: {condition.attribute} of type {type(condition.attribute)}"
                # )
                # Skip processing if condition.attribute is not a string (e.g., during testing with Mock objects)
                if not isinstance(condition.attribute, str):
                    continue

                if "inventory.check_has_item_by_type" in condition.attribute:
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["item_type"] for item in required_items]:
                        required_items.append(
                            ({"item_type": args[0]}, condition.satisfy_value)
                        )
                elif "inventory.check_has_item_by_name" in condition.attribute:
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["name"] for item in required_items]:
                        required_items.append(
                            ({"name": args[0]}, condition.satisfy_value)
                        )
                elif "inventory.check_has_item_by_value" in condition.attribute:
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["value"] for item in required_items]:
                        required_items.append(
                            ({"value": args[0]}, condition.satisfy_value)
                        )
                elif "inventory.check_has_item_by_usability" in condition.attribute:
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["usability"] for item in required_items]:
                        required_items.append(
                            ({"usability": args[0]}, condition.satisfy_value)
                        )
                elif (
                    "inventory.check_has_item_by_sentimental_value"
                    in condition.attribute
                ):
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [
                        item["sentimental_value"] for item in required_items
                    ]:
                        required_items.append(
                            ({"sentimental_value": args[0]}, condition.satisfy_value)
                        )
                elif "inventory.check_has_item_by_trade_value" in condition.attribute:
                    args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                    args = [arg.strip("'") for arg in args.split(",")]
                    if args[0] not in [item["trade_value"] for item in required_items]:
                        required_items.append(
                            ({"trade_value": args[0]}, condition.satisfy_value)
                        )

        return required_items

    def __repr__(self):
        return f"Goal({self.name}, {self.description}, {self.score} {self.character.name}, {self.target.name}, {self.completion_conditions}, {self.evaluate_utility_function}, {self.difficulty}, {self.completion_reward}, {self.failure_penalty}, {self.completion_message}, {self.failure_message}, {self.criteria}, {self.required_items}, {self.environment}, {self.goal_type})"

    def __eq__(self, other):
        if not isinstance(other, Goal):
            return False
        return (
            self.name == other.name
            and self.description == other.description
            and self.score == other.score
            and self.target == other.target
            and self.completion_conditions == other.completion_conditions
            and self.evaluate_utility_function == other.evaluate_utility_function
            and self.difficulty == other.difficulty
            and self.completion_reward == other.completion_reward
            and self.failure_penalty == other.failure_penalty
            and self.completion_message == other.completion_message
            and self.failure_message == other.failure_message
            and self.criteria == other.criteria
            and self.required_items == other.required_items
            and self.environment == other.environment
            and self.goal_type == other
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
            (
                self.name,
                self.description,
                self.score,
                make_hashable(self.completion_conditions),
                str(self.evaluate_utility_function),
                self.difficulty,
                make_hashable(self.completion_reward),
                self.failure_penalty,
                self.completion_message,
                self.failure_message,
                make_hashable(self.criteria),
                make_hashable(self.required_items),
                self.goal_type,
            )
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

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score
        return self.score

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
            "character": self.character,
            "target": self.target,
            "completion_conditions": self.completion_conditions,
            "evaluate_utility_function": self.evaluate_utility_function,
            "difficulty": self.difficulty,
            "completion_reward": self.completion_reward,
            "failure_penalty": self.failure_penalty,
            "completion_message": self.completion_message,
            "failure_message": self.failure_message,
            "criteria": self.criteria,
            "required_items": self.required_items,
            "environment": self.environment,
            "goal_type": self.goal_type,
        }

    def check_completion(self):
        return all(
            [
                condition.check_condition()
                for condition in self.completion_conditions.values()
            ]
        )

    def evaluate_utility_function(self):
        return self.evaluate_utility_function(
            self.character,
            self.character.environment,
            self.difficulty,
            self.criteria,
        )

    def calculate_goal_difficulty(self):
        return self.difficulty(self.character, self.environment)


class Motive:
    def __init__(self, name: str, description: str, score: float):
        self.name = name
        self.description = description
        self.score = score  # Represents the strength of the motive

    def __repr__(self):
        return f"Motive({self.name}, {self.description}, {self.score})"

    def __eq__(self, other):
        if not isinstance(other, Motive):
            if isinstance(other, dict):
                return (
                    self.name == other["name"]
                    and self.description == other["description"]
                    and self.score == other["score"]
                )
            elif isinstance(other, tuple):
                return (
                    self.name == other[0]
                    and self.description == other[1]
                    and self.score == other[2]
                )
            elif isinstance(other, float) or isinstance(other, int):
                return self.score == float(other)
            return False
        return (
            self.name == other.name
            and self.description == other.description
            and self.score == other.score
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

        return hash(make_hashable(self.to_dict()))

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

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score
        return self.score

    def to_dict(self):
        return {"name": self.name, "description": self.description, "score": self.score}


class PersonalMotives:
    def __init__(
        self,
        hunger_motive: Motive,
        wealth_motive: Motive,
        mental_health_motive: Motive,
        social_wellbeing_motive: Motive,
        happiness_motive: Motive,
        health_motive: Motive,
        shelter_motive: Motive,
        stability_motive: Motive,
        luxury_motive: Motive,
        hope_motive: Motive,
        success_motive: Motive,
        control_motive: Motive,
        job_performance_motive: Motive,
        beauty_motive: Motive,
        community_motive: Motive,
        material_goods_motive: Motive,
        family_motive: Motive,
    ):
        self.hunger_motive = self.set_hunger_motive(hunger_motive)
        self.wealth_motive = self.set_wealth_motive(wealth_motive)
        self.mental_health_motive = self.set_mental_health_motive(mental_health_motive)
        self.social_wellbeing_motive = self.set_social_wellbeing_motive(
            social_wellbeing_motive
        )
        self.happiness_motive = self.set_happiness_motive(happiness_motive)
        self.health_motive = self.set_health_motive(health_motive)
        self.shelter_motive = self.set_shelter_motive(shelter_motive)
        self.stability_motive = self.set_stability_motive(stability_motive)
        self.luxury_motive = self.set_luxury_motive(luxury_motive)
        self.hope_motive = self.set_hope_motive(hope_motive)
        self.success_motive = self.set_success_motive(success_motive)
        self.control_motive = self.set_control_motive(control_motive)
        self.job_performance_motive = self.set_job_performance_motive(
            job_performance_motive
        )
        self.beauty_motive = self.set_beauty_motive(beauty_motive)
        self.community_motive = self.set_community_motive(community_motive)
        self.material_goods_motive = self.set_material_goods_motive(
            material_goods_motive
        )
        self.family_motive = self.set_family_motive(family_motive)
        self._attributes = [
            "hunger_motive",
            "wealth_motive",
            "mental_health_motive",
            "social_wellbeing_motive",
            "happiness_motive",
            "health_motive",
            "shelter_motive",
            "stability_motive",
            "luxury_motive",
            "hope_motive",
            "success_motive",
            "control_motive",
            "job_performance_motive",
            "beauty_motive",
            "community_motive",
            "material_goods_motive",
            "family_motive",
        ]
        self._index = 0

    def __repr__(self):
        return f"PersonalMotives({self.hunger_motive}, {self.wealth_motive}, {self.mental_health_motive}, {self.social_wellbeing_motive}, {self.happiness_motive}, {self.health_motive}, {self.shelter_motive}, {self.stability_motive}, {self.luxury_motive}, {self.hope_motive}, {self.success_motive}, {self.control_motive}, {self.job_performance_motive}, {self.beauty_motive}, {self.community_motive}, {self.material_goods_motive}, {self.family_motive})"

    def __eq__(self, other):
        if not isinstance(other, PersonalMotives):
            if isinstance(other, dict):
                return (
                    self.hunger_motive == other["hunger_motive"]
                    and self.wealth_motive == other["wealth_motive"]
                    and self.mental_health_motive == other["mental_health_motive"]
                    and self.social_wellbeing_motive == other["social_wellbeing_motive"]
                    and self.happiness_motive == other["happiness_motive"]
                    and self.health_motive == other["health_motive"]
                    and self.shelter_motive == other["shelter_motive"]
                    and self.stability_motive == other["stability_motive"]
                    and self.luxury_motive == other["luxury_motive"]
                    and self.hope_motive == other["hope_motive"]
                    and self.success_motive == other["success_motive"]
                    and self.control_motive == other["control_motive"]
                    and self.job_performance_motive == other["job_performance_motive"]
                    and self.beauty_motive == other["beauty_motive"]
                    and self.community_motive == other["community_motive"]
                    and self.material_goods_motive == other["material_goods_motive"]
                    and self.family_motive == other["family_motive"]
                )
            else:
                return False

        return (
            self.hunger_motive == other.hunger_motive
            and self.wealth_motive == other.wealth_motive
            and self.mental_health_motive == other.mental_health_motive
            and self.social_wellbeing_motive == other.social_wellbeing_motive
            and self.happiness_motive == other.happiness_motive
            and self.health_motive == other.health_motive
            and self.shelter_motive == other.shelter_motive
            and self.stability_motive == other.stability_motive
            and self.luxury_motive == other.luxury_motive
            and self.hope_motive == other.hope_motive
            and self.success_motive == other.success_motive
            and self.control_motive == other.control_motive
            and self.job_performance_motive == other.job_performance_motive
            and self.beauty_motive == other.beauty_motive
            and self.community_motive == other.community_motive
            and self.material_goods_motive == other.material_goods_motive
            and self.family_motive == other.family_motive
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
                    make_hashable(self.hunger_motive),
                    make_hashable(self.wealth_motive),
                    make_hashable(self.mental_health_motive),
                    make_hashable(self.social_wellbeing_motive),
                    make_hashable(self.happiness_motive),
                    make_hashable(self.health_motive),
                    make_hashable(self.shelter_motive),
                    make_hashable(self.stability_motive),
                    make_hashable(self.luxury_motive),
                    make_hashable(self.hope_motive),
                    make_hashable(self.success_motive),
                    make_hashable(self.control_motive),
                    make_hashable(self.job_performance_motive),
                    make_hashable(self.beauty_motive),
                    make_hashable(self.community_motive),
                    make_hashable(self.material_goods_motive),
                    make_hashable(self.family_motive),
                ]
            )
        )

    def __iter__(self):
        self._index = 0  # Reset index on new iteration
        return self

    def __next__(self):
        if self._index < len(self._attributes):
            attr_name = self._attributes[self._index]
            self._index += 1
            return getattr(self, attr_name)
        else:
            raise StopIteration

    def set_family_motive(self, family_motive):
        self.family_motive = family_motive
        return self.family_motive

    def get_family_motive(self):
        return self.family_motive

    def get_hunger_motive(self):
        return self.hunger_motive

    def set_hunger_motive(self, hunger_motive):
        self.hunger_motive = hunger_motive
        return self.hunger_motive

    def get_wealth_motive(self):
        return self.wealth_motive

    def set_wealth_motive(self, wealth_motive):
        self.wealth_motive = wealth_motive
        return self.wealth_motive

    def get_mental_health_motive(self):
        return self.mental_health_motive

    def set_mental_health_motive(self, mental_health_motive):
        self.mental_health_motive = mental_health_motive
        return self.mental_health_motive

    def get_social_wellbeing_motive(self):
        return self.social_wellbeing_motive

    def set_social_wellbeing_motive(self, social_wellbeing):
        self.social_wellbeing_motive = social_wellbeing
        return self.social_wellbeing_motive

    def get_happiness_motive(self):
        return self.happiness_motive

    def set_happiness_motive(self, happiness_motive):
        self.happiness_motive = happiness_motive
        return self.happiness_motive

    def get_health_motive(self):
        return self.health_motive

    def set_health_motive(self, health_motive):
        self.health_motive = health_motive
        return self.health_motive

    def get_shelter_motive(self):
        return self.shelter_motive

    def set_shelter_motive(self, shelter_motive):
        self.shelter_motive = shelter_motive
        return self.shelter_motive

    def get_stability_motive(self):
        return self.stability_motive

    def set_stability_motive(self, stability_motive):
        self.stability_motive = stability_motive
        return self.stability_motive

    def get_luxury_motive(self):
        return self.luxury_motive

    def set_luxury_motive(self, luxury_motive):
        self.luxury_motive = luxury_motive
        return self.luxury_motive

    def get_hope_motive(self):
        return self.hope_motive

    def set_hope_motive(self, hope_motive):
        self.hope_motive = hope_motive
        return self.hope_motive

    def get_success_motive(self):
        return self.success_motive

    def set_success_motive(self, success_motive):
        self.success_motive = success_motive
        return self.success_motive

    def get_control_motive(self):
        return self.control_motive

    def set_control_motive(self, control_motive):
        self.control_motive = control_motive
        return self.control_motive

    def get_job_performance_motive(self):
        return self.job_performance_motive

    def set_job_performance_motive(self, job_performance_motive):
        self.job_performance_motive = job_performance_motive
        return self.job_performance_motive

    def get_beauty_motive(self):
        return self.beauty_motive

    def set_beauty_motive(self, beauty_motive):
        self.beauty_motive = beauty_motive
        return self.beauty_motive

    def get_community_motive(self):
        return self.community_motive

    def set_community_motive(self, community_motive):
        self.community_motive = community_motive
        return self.community_motive

    def get_material_goods_motive(self):
        return self.material_goods_motive

    def set_material_goods_motive(self, material_goods_motive):
        self.material_goods_motive = material_goods_motive
        return self.material_goods_motive

    def get_family_motive(self):
        return self.family_motive

    def set_family_motive(self, family_motive):
        self.family_motive = family_motive
        return self.family_motive

    def to_dict(self):
        return {
            "hunger": self.hunger_motive,
            "wealth": self.wealth_motive,
            "mental health": self.mental_health_motive,
            "social wellbeing": self.social_wellbeing_motive,
            "happiness": self.happiness_motive,
            "health": self.health_motive,
            "shelter": self.shelter_motive,
            "stability": self.stability_motive,
            "luxury": self.luxury_motive,
            "hope": self.hope_motive,
            "success": self.success_motive,
            "control": self.control_motive,
            "job performance": self.job_performance_motive,
            "beauty": self.beauty_motive,
            "community": self.community_motive,
            "material goods": self.material_goods_motive,
            "family": self.family_motive,
        }


example_criteria__a = [
    {
        "node_attributes": {"type": "character"},
        "edge_attributes": {"relationship": "friend", "strength": 0.8},
        "max_distance": 20,
    }
]
example_criteria_b = [
    {
        "node_attributes": {"type": "character"},
        "relationship": "enemy",
        "max_distance": 20,
    }
]
example_criteria_c = [
    {
        "node_attributes": {"type": "character"},
        "relationship": "family",
        "max_distance": 100.0,
    },
    {
        "node_attributes": {"type": "location"},
        "safety_threshold": 0.8,
        "max_distance": 50,
    },
]
example_criteria_d = [
    {
        "node_attributes": {"type": "item", "item_type": "food"},
        "max_distance": 20,
    }
]
example_criteria_e = [
    {
        "node_attributes": {"type": "item", "item_type": "weapon", "usability": 0.8},
        "max_distance": 20,
    }
]
example_criteria_f = [
    {
        "node_attributes": {
            "type": "object",
            "item_type": "food",
            "type_specific_attributes": ("cooked", False),
        },
        "max_distance": 20,
    }
]
example_criteria_g = [
    {
        "node_attributes": {"type": "object", "trade_value": 100.0},
        "max_distance": 20,
    },
    {
        "node_attributes": {"type": "object", "scarcity": 0.2},
        "max_distance": 20,
    },
]
example_criteria_h = [
    {
        "node_attributes": {"type": "item", "item_type": "luxury", "value": 0.8},
        "max_distance": 20,
    }
]
example_criteria_i = [
    {
        "event_participation": Event(
            name="Festival",
            date="2022-07-04",
            event_type="annual",
            importance=0.8,
            impact=0.8,
            required_items=["food", "decorations"],
            coordinates_location=(100.0, 100.0),
        ),
    }
]

example_criteria_j = [
    {
        "offer_item_trade": ItemInventory(
            food_items=[
                FoodItem(
                    name="Apple",
                    description="A juicy red apple.",
                    value=1,
                    perishable=True,
                    weight=0.5,
                    quantity=1,
                    calories=2,
                    action_system=importlib.import_module("actions").ActionSystem(),
                )
            ]
        ),
        "max_distance": 20,
    }
]

example_criteria_k = [
    {
        "node_attributes": {
            "type": "job",
            "item_type": "food",
            "career_opportunities": ["farmer", "chef"],
        },
        "max_distance": 20,
    }
]

example_criteria_l = [
    {
        "node_attributes": {
            "type": "job",
            "item_type": "food",
            "career_opportunities": ["farming", "cooking"],
        },
        "max_distance": 20,
    }
]

# example_criteria_12 = [{
#     "trade_opportunity": Character(
#         name="Joe",
#         description="A traveling merchant.",
#         inventory=ItemInventory(
#             items=[
#                 FoodItem(
#                     name="Apple",
#                     description="A juicy red apple.",
#                     value=1,
#                     perishable=True,
#                     nutrition_value=2,
#                 )
#             ]
#         ),
#     ),

#     "max_distance": 20,
# }

# example_criteria_13 = [{
#     "desired_resource": Character(
#         name="Joe",
#         description="A traveling merchant.",
#         inventory=ItemInventory(
#             items=[
#                 FoodItem(
#                     name="Apple",
#                     description="A juicy red apple.",
#                     value=1,
#                     perishable=True,
#                     nutrition_value=2,
#                 )
#             ]
#         ),
#     ),
# }

# example_criteria_14 = [{
#     "want_item_trade": FoodItem(
#         name="Apple",
#         description="A juicy red apple.",
#         value=1,
#         perishable=True,
#         nutrition_value=2,
#     ),
# }


def motive_to_goals(
    motive,
    character,
    graph_manager: GraphManager,
    goap_planner: GOAPPlanner,
    prompt_builder: PromptBuilder,
):
    GraphManager = importlib.import_module("tiny_graph_manager")
    PromptBuilder = importlib.import_module("tiny_prompt_builder")

    goals = []
    if motive.name == "hunger":
        goals.append(
            (
                0,
                Goal(
                    name="Find Food",
                    description="Search for food to satisfy hunger.",
                    score=motive.score,
                    character=character,
                    target=character,
                    completion_conditions={
                        False: [
                            Condition(  # Remember the key is False because the condition is not met yet
                                name="has_food",
                                attribute="inventory.check_has_item_by_type(['food'])",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,  # This is the weight representing the importance of this condition toward the goal. This will be used in a division operation to calculate the overall importance of the goal.
                            )
                        ]
                    },
                    evaluate_utility_function=goap_planner.evaluate_goal_importance,
                    difficulty=graph_manager.calculate_goal_difficulty,
                    completion_reward=graph_manager.calculate_reward,
                    failure_penalty=graph_manager.calculate_penalty,
                    completion_message=prompt_builder.generate_completion_message,
                    failure_message=prompt_builder.generate_failure_message,
                    criteria=example_criteria_d,
                    graph_manager=graph_manager,
                    goal_type="basic",
                    target_effects={"hunger_level": -5},
                ),
            )
        )
        goals.append(
            (
                0,
                Goal(
                    name="Hunt",
                    description="Go hunting to gather food.",
                    target=character,
                    score=motive.score,
                    character=character,
                    completion_conditions={
                        False: [
                            Condition(
                                name="has_food",
                                attribute="inventory.check_has_item_by_type(['food'])",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,
                            )
                        ]
                    },
                    evaluate_utility_function=goap_planner.evaluate_goal_importance,
                    difficulty=graph_manager.calculate_goal_difficulty,
                    completion_reward=graph_manager.calculate_reward,
                    failure_penalty=graph_manager.calculate_penalty,
                    completion_message=prompt_builder.generate_completion_message,
                    failure_message=prompt_builder.generate_failure_message,
                    criteria=example_criteria_e,
                    graph_manager=graph_manager,
                    goal_type="basic",
                    target_effects={"hunger_level": -5},
                ),
            )
        )
        goals.append(
            (
                0,
                Goal(
                    name="Cook",
                    description="Cook a meal to eat.",
                    target=character,
                    score=motive.score,
                    character=character,
                    completion_conditions={
                        False: [
                            Condition(
                                name="has_food",
                                attribute="inventory.check_has_item_by_type(['food'])",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,
                            )
                        ],
                        False: [
                            Condition(
                                name="has_cooked_food",
                                attribute="inventory.check_has_item_by_attribute_value(['cooked'], ['True'])",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,
                            )
                        ],
                    },
                    evaluate_utility_function=goap_planner.evaluate_goal_importance,
                    difficulty=graph_manager.calculate_goal_difficulty,
                    completion_reward=graph_manager.calculate_reward,
                    failure_penalty=graph_manager.calculate_penalty,
                    completion_message=prompt_builder.generate_completion_message,
                    failure_message=prompt_builder.generate_failure_message,
                    criteria=example_criteria_f,
                    graph_manager=graph_manager,
                    goal_type="basic",
                    target_effects={"hunger_level": -7},
                ),
            )
        )
    elif motive.name == "wealth":
        goals.append(
            (
                0,
                Goal(
                    name="Earn Money",
                    description="Get a job to earn money.",
                    target=character,
                    score=motive.score,
                    character=character,
                    completion_conditions={
                        False: [
                            Condition(
                                name="has_job",
                                attribute="has_job",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,
                            )
                        ]
                    },
                    evaluate_utility_function=goap_planner.evaluate_goal_importance,
                    difficulty=graph_manager.calculate_goal_difficulty,
                    completion_reward=graph_manager.calculate_reward,
                    failure_penalty=graph_manager.calculate_penalty,
                    completion_message=prompt_builder.generate_completion_message,
                    failure_message=prompt_builder.generate_failure_message,
                    criteria=example_criteria_k,
                    graph_manager=graph_manager,
                    goal_type="economic",
                    target_effects={"wealth_level": 10},
                ),
            )
        )
        goals.append(
            (
                0,
                Goal(
                    name="Invest",
                    description="Invest money in stocks.",
                    target=character,
                    score=motive.score,
                    character=character,
                    completion_conditions={
                        False: [
                            Condition(
                                name="has_investment",
                                attribute="has_investment",
                                target=character,
                                satisfy_value=True,
                                op="==",
                                weight=1,
                            )
                        ]
                    },
                    evaluate_utility_function=goap_planner.evaluate_goal_importance,
                    difficulty=graph_manager.calculate_goal_difficulty,
                    completion_reward=graph_manager.calculate_reward,
                    failure_penalty=graph_manager.calculate_penalty,
                    completion_message=prompt_builder.generate_completion_message,
                    failure_message=prompt_builder.generate_failure_message,
                    criteria=example_criteria_g,
                    graph_manager=graph_manager,
                    goal_type="economic",
                    target_effects={"wealth_level": 15},
                ),
            )
        )
    # Add similar elif blocks for other motives
    return goals


class GoalGenerator:
    def __init__(
        self,
        personal_motives,
        graph_manager: GraphManager,
        goap_planner: GOAPPlanner,
        prompt_builder: PromptBuilder,
    ):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager

        self.personal_motives = personal_motives
        self.graph_manager = graph_manager
        self.goap_planner = goap_planner
        self.prompt_builder = prompt_builder

    def generate_goals(self, character):
        if isinstance(character, str):
            character = self.graph_manager.get_node(character)
        if not isinstance(character, Character):
            raise ValueError("Invalid character data type or character not found.")
        goals = []
        for motive in self.personal_motives:
            goals.extend(
                motive_to_goals(
                    motive,
                    character,
                    self.graph_manager,
                    self.goap_planner,
                    self.prompt_builder,
                )
            )
        for _, goal in goals:
            for item in goal.required_items:
                character.update_required_items(item)

        return goals


class PersonalityTraits:
    """1. Openness (to Experience)
    Description: This trait features characteristics such as imagination, curiosity, and openness to new experiences.
    Application: Characters with high openness might be more adventurous, willing to explore unknown parts of the village, or experiment with new skills and jobs. They could be more affected by novel events or changes in the environment. Conversely, characters with low openness might prefer routine, resist change, and stick to familiar activities and interactions.
    2. Conscientiousness
    Description: This trait encompasses high levels of thoughtfulness, with good impulse control and goal-directed behaviors.
    Application: Highly conscientious characters could have higher productivity in their careers, maintain their homes better, and be more reliable in relationships. They might also have routines they adhere to more strictly. Characters low in conscientiousness might miss work, have cluttered homes, and be more unpredictable.
    3. Extraversion
    Description: Extraversion is characterized by excitability, sociability, talkativeness, assertiveness, and high amounts of emotional expressiveness.
    Application: Extraverted characters would seek social interactions, be more active in community events, and have larger social networks. Introverted characters (low in extraversion) might prefer solitary activities, have a few close friends, and have lower energy levels during social events.
    4. Agreeableness
    Description: This trait includes attributes such as trust, altruism, kindness, and affection.
    Application: Characters high in agreeableness might be more likely to form friendships, help other characters, and have positive interactions. Those low in agreeableness might be more competitive, less likely to trust others, and could even engage in conflicts more readily.
    5. Neuroticism (Emotional Stability)
    Description: High levels of neuroticism are associated with emotional instability, anxiety, moodiness, irritability, and sadness.
    Application: Characters with high neuroticism might react more negatively to stress, have more fluctuating moods, and could require more support from friends or activities to maintain happiness. Those with low neuroticism (high emotional stability) tend to remain calmer in stressful situations and have a more consistent mood.
    Implementing Personality Traits in TinyVillage
    Quantitative Measures: Represent each personality trait with a numeric value (e.g., 0 to 100.0) for each character. This allows for nuanced differences between characters and can influence decision-making algorithms.
    Dynamic Interactions: Use personality traits to dynamically influence character interactions. For example, an extraverted character might initiate conversations more frequently, while a highly agreeable character might have more options to support others.
    Influence on Life Choices: Personality can affect career choice, hobbies, and life decisions within the game. For instance, an open and conscientious character might pursue a career in science or exploration.
    Character Development: Allow for personality development over time, influenced by game events, achievements, and relationships. This can add depth to the characters and reflect personal growth or change.
    """

    def __init__(
        self,
        openness: float = 0.0,
        conscientiousness: float = 0.0,
        extraversion: float = 0.0,
        agreeableness: float = 0.0,
        neuroticism: float = 0.0,
    ):
        self.openness = self.set_openness(ClampedIntScore().clamp_score(openness))
        self.conscientiousness = self.set_conscientiousness(
            ClampedIntScore().clamp_score(conscientiousness)
        )
        self.extraversion = self.set_extraversion(
            ClampedIntScore().clamp_score(extraversion)
        )
        self.agreeableness = self.set_agreeableness(
            ClampedIntScore().clamp_score(agreeableness)
        )
        self.neuroticism = self.set_neuroticism(
            ClampedIntScore().clamp_score(neuroticism)
        )
        self.motives = None

    def __repr__(self):
        return f"PersonalityTraits({self.openness}, {self.conscientiousness}, {self.extraversion}, {self.agreeableness}, {self.neuroticism})"

    def __eq__(self, other):
        return (
            self.openness == other.openness
            and self.conscientiousness == other.conscientiousness
            and self.extraversion == other.extraversion
            and self.agreeableness == other.agreeableness
            and self.neuroticism == other.neuroticism
        )

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.openness,
                    self.conscientiousness,
                    self.extraversion,
                    self.agreeableness,
                    self.neuroticism,
                ]
            )
        )

    def to_dict(self):
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    def get_openness(self):
        return self.openness

    def set_openness(self, openness):
        self.openness = openness
        return self.openness

    def get_conscientiousness(self):
        return self.conscientiousness

    def set_conscientiousness(self, conscientiousness):
        self.conscientiousness = conscientiousness
        return self.conscientiousness

    def get_extraversion(self):
        return self.extraversion

    def set_extraversion(self, extraversion):
        self.extraversion = extraversion
        return self.extraversion

    def get_agreeableness(self):
        return self.agreeableness

    def set_agreeableness(self, agreeableness):
        self.agreeableness = agreeableness
        return self.agreeableness

    def get_neuroticism(self):
        return self.neuroticism

    def set_neuroticism(self, neuroticism):
        self.neuroticism = neuroticism
        return self.neuroticism

    def get_motives(self):
        return self.motives

    def get_personality_trait(self, trait):
        if trait == "openness":
            return self.openness
        elif trait == "conscientiousness":
            return self.conscientiousness
        elif trait == "extraversion":
            return self.extraversion
        elif trait == "agreeableness":
            return self.agreeableness
        elif trait == "neuroticism":
            return self.neuroticism
        else:
            return None

    def set_motives(
        self,
        hunger_motive: float = 0.0,
        wealth_motive: float = 0.0,
        mental_health_motive: float = 0.0,
        social_wellbeing_motive: float = 0.0,
        happiness_motive: float = 0.0,
        health_motive: float = 0.0,
        shelter_motive: float = 0.0,
        stability_motive: float = 0.0,
        luxury_motive: float = 0.0,
        hope_motive: float = 0.0,
        success_motive: float = 0.0,
        control_motive: float = 0.0,
        job_performance_motive: float = 0.0,
        beauty_motive: float = 0.0,
        community_motive: float = 0.0,
        material_goods_motive: float = 0.0,
        family_motive: float = 0.0,
    ):
        self.motives = PersonalMotives(
            hunger_motive=Motive(
                "hunger", "bias toward satisfying hunger", hunger_motive
            ),
            wealth_motive=Motive(
                "wealth", "bias toward accumulating wealth", wealth_motive
            ),
            mental_health_motive=Motive(
                "mental health",
                "bias toward maintaining mental health",
                mental_health_motive,
            ),
            social_wellbeing_motive=Motive(
                "social wellbeing",
                "bias toward maintaining social wellbeing",
                social_wellbeing_motive,
            ),
            happiness_motive=Motive(
                "happiness", "bias toward maintaining happiness", happiness_motive
            ),
            health_motive=Motive(
                "health", "bias toward maintaining health", health_motive
            ),
            shelter_motive=Motive(
                "shelter", "bias toward maintaining shelter", shelter_motive
            ),
            stability_motive=Motive(
                "stability", "bias toward maintaining stability", stability_motive
            ),
            luxury_motive=Motive(
                "luxury", "bias toward maintaining luxury", luxury_motive
            ),
            hope_motive=Motive("hope", "bias toward maintaining hope", hope_motive),
            success_motive=Motive(
                "success", "bias toward maintaining success", success_motive
            ),
            control_motive=Motive(
                "control", "bias toward maintaining control", control_motive
            ),
            job_performance_motive=Motive(
                "job performance",
                "bias toward maintaining job performance",
                job_performance_motive,
            ),
            beauty_motive=Motive(
                "beauty", "bias toward maintaining beauty", beauty_motive
            ),
            community_motive=Motive(
                "community", "bias toward maintaining community", community_motive
            ),
            material_goods_motive=Motive(
                "material goods",
                "bias toward maintaining material goods",
                material_goods_motive,
            ),
            family_motive=Motive(
                "family", "bias toward maintaining family", family_motive
            ),
        )


class CharacterSkills:
    def __init__(self, skills: List[Skill]):
        self.action_skills = []
        self.job_skills = []
        self.other_skills = []
        self.skills = []
        self.set_skills(skills)

    def __repr__(self):
        return f"CharacterSkills({self.skills}, {self.job_skills}, {self.action_skills}, {self.other_skills})"

    def __eq__(self, other):
        return (
            self.skills == other.skills
            and self.job_skills == other.job_skills
            and self.action_skills == other.action_skills
            and self.other_skills == other.other_skills
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
                    make_hashable(self.skills),
                    make_hashable(self.job_skills),
                    make_hashable(self.action_skills),
                    make_hashable(self.other_skills),
                ]
            )
        )

    def get_skills(self):
        return self.skills

    def get_skills_as_list_of_strings(self):
        return [skill.name for skill in self.skills]

    def set_skills(self, skills):
        for skill in skills:
            self.skills.append(skill)
            if isinstance(skill, JobSkill):
                self.job_skills.append(skill)
            elif isinstance(skill, ActionSkill):
                self.action_skills.append(skill)
            else:
                self.other_skills.append(skill)

    def add_skill(self, skill):
        self.skills.append(skill)
        return self.skills


preconditions_dict = {
    "Talk": [
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
    ],
    "Trade": [
        {
            "name": "wealth_money",
            "attribute": "wealth_money",
            "target": "initiator",
            "satisfy_value": 5,
            "operator": "gt",
        },
        {
            "name": "conscientiousness",
            "attribute": "personality_traits.conscientiousness",
            "target": "target",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Help": [
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
            "target": "initiator",
            "satisfy_value": 20,
            "operator": "gt",
        },
        {
            "name": "agreeableness",
            "attribute": "personality_traits.agreeableness",
            "target": "initiator",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Attack": [
        {
            "name": "anger",
            "attribute": "current_mood",
            "target": "initiator",
            "satisfy_value": -10,
            "operator": "lt",
        },
        {
            "name": "strength",
            "attribute": "skills.strength",
            "target": "initiator",
            "satisfy_value": 20,
            "operator": "gt",
        },
    ],
    "Befriend": [
        {
            "name": "openness",
            "attribute": "personality_traits.openness",
            "target": "initiator",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Teach": [
        {
            "name": "knowledge",
            "attribute": "skills.knowledge",
            "target": "initiator",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "patience",
            "attribute": "personality_traits.agreeableness",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Learn": [
        {
            "name": "curiosity",
            "attribute": "personality_traits.openness",
            "target": "initiator",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "focus",
            "attribute": "mental_health",
            "target": "initiator",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Heal": [
        {
            "name": "medical_knowledge",
            "attribute": "skills.medical_knowledge",
            "target": "initiator",
            "satisfy_value": 40,
            "operator": "gt",
        },
        {
            "name": "compassion",
            "attribute": "personality_traits.agreeableness",
            "target": "initiator",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Gather": [
        {
            "name": "energy",
            "attribute": "energy",
            "target": "initiator",
            "satisfy_value": 20,
            "operator": "gt",
        },
        {
            "name": "curiosity",
            "attribute": "personality_traits.openness",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Build": [
        {
            "name": "construction_skill",
            "attribute": "skills.construction",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
        {
            "name": "conscientiousness",
            "attribute": "personality_traits.conscientiousness",
            "target": "initiator",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Give Item": [
        {
            "name": "item_in_inventory",
            "attribute": "inventory.item_count",
            "target": "initiator",
            "satisfy_value": 1,
            "operator": "gt",
        },
        {
            "name": "generosity",
            "attribute": "personality_traits.agreeableness",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Receive Item": [
        {
            "name": "need food",
            "attribute": "hunger_level",
            "target": "initiator",
            "satisfy_value": 5,
            "operator": "gt",
        },
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
            "target": "initiator",
            "satisfy_value": 10,
            "operator": "gt",
        },
    ],
}

effect_dict = {
    "Talk": [
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 5},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -2},
        {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 5},
        {
            "targets": ["initiator"],
            "method": "play_animation",
            "method_args": ["talking"],
        },
    ],
    "Eat": [
        {"targets": ["initiator"], "attribute": "hunger_level", "change_value": -5},
        {"targets": ["initiator"], "attribute": "energy", "change_value": 5},
        {
            "targets": ["initiator"],
            "method": "play_animation",
            "method_args": ["eating"],
        },
    ],
    "Trade": [
        {"targets": ["initiator"], "attribute": "wealth_money", "change_value": -5},
        {
            "targets": ["initiator"],
            "attribute": "inventory.item_count",
            "change_value": -1,
        },
        {"targets": ["target"], "attribute": "wealth_money", "change_value": 5},
        {"targets": ["target"], "attribute": "inventory.item_count", "change_value": 1},
    ],
    "Help": [
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 10},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
        {"targets": ["target"], "attribute": "health_status", "change_value": 10},
    ],
    "Attack": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
        {"targets": ["target"], "attribute": "health_status", "change_value": -10},
    ],
    "Befriend": [
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 8},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -3},
        {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 8},
    ],
    "Teach": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -4},
        {"targets": ["target"], "attribute": "skills.knowledge", "change_value": 5},
    ],
    "Learn": [
        {"targets": ["initiator"], "attribute": "skills.knowledge", "change_value": 7},
        {"targets": ["initiator"], "attribute": "mental_health", "change_value": 3},
        {"targets": ["target"], "attribute": "skills.teaching", "change_value": 1},
    ],
    "Heal": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -6},
        {"targets": ["target"], "attribute": "health_status", "change_value": 15},
    ],
    "Gather": [
        {"targets": ["initiator"], "attribute": "wealth_money", "change_value": 5},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -4},
    ],
    "Build": [
        {"targets": ["initiator"], "attribute": "material_goods", "change_value": 10},
        {"targets": ["initiator"], "attribute": "energy", "change_value": -8},
    ],
    "Give Item": [
        {
            "targets": ["initiator"],
            "attribute": "inventory.item_count",
            "change_value": -1,
        },
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 5},
        {"targets": ["target"], "attribute": "inventory.item_count", "change_value": 1},
        {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 5},
    ],
    "Receive Item": [
        {
            "targets": ["initiator"],
            "attribute": "inventory.item_count",
            "change_value": 1,
        },
        {"targets": ["initiator"], "attribute": "hunger_level", "change_value": -5},
        {
            "targets": ["target"],
            "attribute": "inventory.item_count",
            "change_value": -1,
        },
    ],
}


pronoun_mapper = {
    "he/him": ["he", "him", "his", "himself", "man"],
    "she/her": ["she", "her", "her", "herself", "woman"],
    "they/them": ["they", "them", "their", "themselves", "person"],
    "it/it": ["it", "it", "its", "itself", "thing"],
}


class Character:
    """
    Character Attributes
    Basic Attributes: These include name, age, gender identity, and appearance. Consider allowing for a diverse range of attributes to make each character unique and relatable to a wide audience.
    Personality Traits: Utilize established models like the Big Five Personality Traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) to create varied and predictable behavior patterns.
    Preferences: Likes, dislikes, hobbies, and interests can dictate how characters interact with the environment, objects, and other characters.
    Skills and Careers: Define a skill system and potential careers that characters can pursue, influencing their daily routines, income, and interactions with others.
    Relationships and Social Networks: Outline how characters form friendships, rivalities, and romantic relationships, affecting their social life and decisions.

    """

    def __init__(
        self,
        name,
        age,
        pronouns: str = "they/them",
        job: str = "unemployed",
        health_status: int = 10,
        hunger_level: int = 2,
        wealth_money: int = 10,
        mental_health: int = 8,
        social_wellbeing: int = 8,
        job_performance: int = 20,
        community: int = 5,
        friendship_grid=[],
        recent_event: str = "",
        long_term_goal: str = "",
        home: House = None,
        inventory: ItemInventory = None,
        motives: PersonalMotives = None,
        personality_traits: PersonalityTraits = None,
        career_goals: List[str] = [],
        possible_interactions: List[Action] = [],
        move_speed: int = 1,
        graph_manager: GraphManager = None,
        action_system: ActionSystem = None,
        gametime_manager: GameTimeManager = None,
        location: Location = None,
        energy: int = 10,
        romanceable: bool = True,
        physical_appearance: str = "",
        physical_beauty: int = random.randint(0, 100),
    ):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        self._updating = False  # Flag to prevent recursion
        self._initialized = False  # Flag to prevent recursion
        self.work_action_count = 0 # Added to track work actions


        ActionSystem = importlib.import_module("actions")
        Action = importlib.import_module("actions").Action
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        self.move_speed = move_speed
        if not action_system:
            action_system = ActionSystem()
        self.action_system = action_system
        self.name = self.set_name(name)
        self.age = self.set_age(age)
        self.destination = None
        self.path = []
        self.speed = 1.0  # Units per tick
        if graph_manager is None:
            raise ValueError("GraphManager instance required.")
        self.graph_manager = graph_manager
        self.energy = energy
        self.character_actions = [
            Action(
                "Talk",
                self.action_system.instantiate_conditions(preconditions_dict["Talk"]),
                effects=effect_dict["Talk"],
                cost=1,
            ),
            Action(
                "Trade",
                self.action_system.instantiate_conditions(preconditions_dict["Trade"]),
                effects=effect_dict["Trade"],
                cost=2,
            ),
            Action(
                "Help",
                self.action_system.instantiate_conditions(preconditions_dict["Help"]),
                effects=effect_dict["Help"],
                cost=1,
            ),
            Action(
                "Attack",
                self.action_system.instantiate_conditions(preconditions_dict["Attack"]),
                effects=effect_dict["Attack"],
                cost=3,
            ),
            Action(
                "Befriend",
                self.action_system.instantiate_conditions(
                    preconditions_dict["Befriend"]
                ),
                effects=effect_dict["Befriend"],
                cost=1,
            ),
            Action(
                "Teach",
                self.action_system.instantiate_conditions(preconditions_dict["Teach"]),
                effects=effect_dict["Teach"],
                cost=2,
            ),
            Action(
                "Learn",
                self.action_system.instantiate_conditions(preconditions_dict["Learn"]),
                effects=effect_dict["Learn"],
                cost=1,
            ),
            Action(
                "Heal",
                self.action_system.instantiate_conditions(preconditions_dict["Heal"]),
                effects=effect_dict["Heal"],
                cost=2,
            ),
            Action(
                "Gather",
                self.action_system.instantiate_conditions(preconditions_dict["Gather"]),
                effects=effect_dict["Gather"],
                cost=1,
            ),
            Action(
                "Build",
                self.action_system.instantiate_conditions(preconditions_dict["Build"]),
                effects=effect_dict["Build"],
                cost=2,
            ),
            Action(
                "Give Item",
                self.action_system.instantiate_conditions(
                    preconditions_dict["Give Item"]
                ),
                effects=effect_dict["Give Item"],
                cost=1,
            ),
            Action(
                "Receive Item",
                self.action_system.instantiate_conditions(
                    preconditions_dict["Receive Item"]
                ),
                effects=effect_dict["Receive Item"],
                cost=1,
            ),
        ]
        self.possible_interactions = possible_interactions + self.character_actions
        self.goap_planner = GOAPPlanner(self.graph_manager)
        from tiny_prompt_builder import PromptBuilder

        self.prompt_builder = PromptBuilder(self)
        self.pronouns = self.set_pronouns(pronouns)
        self.friendship_grid = friendship_grid if friendship_grid else [{}]
        self.job = None
        self.job = self.set_job(job)
        self.health_status = self.set_health_status(health_status)
        self.hunger_level = self.set_hunger_level(hunger_level)
        self.wealth_money = self.set_wealth_money(wealth_money)
        logging.info(
            f"Character {self.name} has been created with {self.wealth_money} money."
        )
        self.mental_health = self.set_mental_health(mental_health)
        self.social_wellbeing = self.set_social_wellbeing(social_wellbeing)

        self.beauty = 0  # fluctuates with environment
        self.community = self.set_community(community)

        self.friendship_grid = self.set_friendship_grid(friendship_grid)
        self.recent_event = self.set_recent_event(recent_event)
        self.long_term_goal = self.set_long_term_goal(long_term_goal)

        # Handle inventory parameter properly - can be None or ItemInventory object
        if inventory is None:
            self.inventory = self.set_inventory()  # Use defaults (empty lists)
        elif isinstance(inventory, ItemInventory):
            self.inventory = inventory  # Use the provided ItemInventory object directly
        else:
            # If it's something else, try to use set_inventory with defaults
            self.inventory = self.set_inventory()
        if home:
            self.home = self.set_home(home)
        else:
            self.home = self.set_home()

        self.personality_traits = self.set_personality_traits(personality_traits)

        if motives is not None:
            self.motives = self.set_motives(motives)
        elif motives is None or not self.motives:
            self.motives = self.calculate_motives()
        self.luxury = 0  # fluctuates with environment

        self.job_performance = self.set_job_performance(job_performance)
        self.material_goods = self.set_material_goods(self.calculate_material_goods())

        self.shelter = self.set_shelter(self.home.calculate_shelter_value())
        self.success = self.set_success(self.calculate_success())
        self.control = self.set_control(self.calculate_control())
        self.hope = self.set_hope(self.calculate_hope())
        self.stability = self.set_stability(self.calculate_stability())
        self.happiness = self.set_happiness(self.calculate_happiness())

        self.stamina = 0
        self.current_satisfaction = 0
        self.current_mood = 50
        self.current_activity = None
        self.skills = CharacterSkills([])
        self.career_goals = career_goals
        self.short_term_goals = []
        self.uuid = uuid.uuid4()
        # Check that character has a unique id

        if gametime_manager:
            self.gametime_manager = gametime_manager
        else:
            raise ValueError("GameTimeManager instance required.")
        self.memory_manager = MemoryManager(
            gametime_manager
        )  # Initialize MemoryManager with GameTimeManager instance
        if location is None:
            self.location = Location(name, 0, 0, 0, 0, action_system)
        else:
            self.location = location
        self.coordinates_location = self.location.get_coordinates()

        self.needed_items = (
            []
        )  # Items needed to complete goals. This is a list of tuples, each tuple is (dict of item requirements, quantity needed)
        self.goals = []

        self.state = self.get_state()
        self.romantic_relationships = []
        self.romanceable = romanceable
        self.exclusive_relationship = None
        self.base_libido = self.calculate_base_libido()
        self.monogamy = 0
        self.investment_portfolio = InvestmentPortfolio([])

        self.monogamy = self.calculate_monogamy()
        self.goals = self.evaluate_goals()
        # Check graph_manager to see if character has been added to the graph
        if not self.graph_manager.get_node(node_id=self):
            logging.info(
                f"Adding character {self.name} to graph manager with id {self.graph_manager.unique_graph_id}\n"
            )
            self.graph_manager.add_character_node(self)
            logging.info(
                f"Character {self.name} added to graph manager {self.graph_manager.unique_graph_id, self.graph_manager.G[self]} with node attributes \n {[self.graph_manager.G.nodes[self]]}\n"
            )

        self.post_init()

    # def __setattr__(self, name, value):
    #     super().__setattr__(name, value)
    #     if name != "uuid" and not self._updating:
    #         self.update_function()

    # def __getattribute__(self, name):
    #     if name != "uuid" and name != "_updating":
    #         self._updating = True
    #         self.update_function()
    #         self._updating = False
    #     return object.__getattribute__(self, name)

    # def update_function(self):
    #     self.dynamic_function = lambda: self._value

    # def set_value(self, value):
    #     self._value = value

    def get_base_libido(self):
        return self.base_libido

    def post_init(self):

        logging.info(f"Character {self.name} has been created\n")
        self._initialized = True

    def add_to_inventory(self, item: ItemObject):
        if not self.graph_manager.G.has_node(item):
            self.graph_manager.add_item_node(item)
        self.inventory.add_item(item)

    def generate_goals(self):

        goal_generator = GoalGenerator(
            self.motives, self.graph_manager, self.goap_planner, self.prompt_builder
        )
        return goal_generator.generate_goals(self)

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

    def get_coordinates_location(self):
        return self.location.get_coordinates()

    def set_coordinates_location(self, *coordinates):
        if len(coordinates) == 1:
            if isinstance(coordinates[0], tuple):
                self.location.set_coordinates(coordinates[0])
        elif len(coordinates) == 2:
            self.location.set_coordinates(coordinates)

        return self.location.coordinates_location

    def set_destination(self, destination):
        self.destination = destination
        self.path = a_star_search(self.graph_manager, self.location, destination)

    def move_towards_next_waypoint(self, potential_field):
        if not self.path:
            return

        next_waypoint = self.path[0]
        seek_force = SteeringBehaviors.seek(self, next_waypoint)
        avoid_force = SteeringBehaviors.avoid(
            self, self.graph_manager.get_obstacles(), avoidance_radius=2.0
        )
        potential_field_force = SteeringBehaviors.potential_field_force(
            self, potential_field
        )

        # Combine forces
        combined_force = (
            seek_force[0] + avoid_force[0] + potential_field_force[0],
            seek_force[1] + avoid_force[1] + potential_field_force[1],
        )

        # Apply combined force to location
        self.set_coordinates_location(
            self.location[0] + combined_force[0],
            self.location[1] + combined_force[1],
        )

        self.graph_manager.update_location(self.name, self.get_coordinates_location())

        if self.get_coordinates_location() == next_waypoint:
            self.path.pop(0)
            if not self.path:
                self.destination = None

    def add_new_goal(self, goal):
        self.goals.append((0, goal))

    def create_memory(self, description, timestamp, importance):
        memory = Memory(
            description, timestamp, importance
        )  # Assuming Memory class exists and is appropriately structured
        self.memory_manager.add_memory(memory)

    def recall_recent_memories(self):
        return self.memory_manager.flat_access.get_recent_memories()

    def make_decision_based_on_memories(self):
        """
        Makes decisions based on recent memories and their importance scores.
        Returns a decision context that can influence character behavior.
        """
        # Get recent memories
        important_memories = self.recall_recent_memories()
        decision_context = {
            "memory_influenced": False,
            "emotional_bias": 0,
            "risk_tolerance": 50,  # Default neutral
            "social_inclination": 50,  # Default neutral
            "preferred_actions": [],
            "avoided_actions": [],
        }

        if important_memories:
            total_emotional_impact = 0
            positive_memories = 0
            negative_memories = 0
            social_memories = 0

            for memory in important_memories:
                importance = getattr(
                    memory, "importance_score", 5
                )  # Default importance
                emotional_content = getattr(memory, "emotional_impact", 0)
                memory_description = getattr(memory, "description", "")

                # Analyze memory content for decision influence
                if importance > 7:  # High importance memories have more influence
                    decision_context["memory_influenced"] = True

                    # Emotional bias based on memory emotional content
                    total_emotional_impact += emotional_content * (importance / 10)

                    if emotional_content > 0:
                        positive_memories += 1
                    elif emotional_content < 0:
                        negative_memories += 1

                    # Check for social context in memories
                    social_keywords = [
                        "friend",
                        "talk",
                        "meet",
                        "party",
                        "social",
                        "relationship",
                    ]
                    if any(
                        keyword in memory_description.lower()
                        for keyword in social_keywords
                    ):
                        social_memories += 1

                    # Influence risk tolerance based on memory outcomes
                    if (
                        "success" in memory_description.lower()
                        or "achieve" in memory_description.lower()
                    ):
                        decision_context["risk_tolerance"] = min(
                            100, decision_context["risk_tolerance"] + 10
                        )
                    elif (
                        "fail" in memory_description.lower()
                        or "mistake" in memory_description.lower()
                    ):
                        decision_context["risk_tolerance"] = max(
                            0, decision_context["risk_tolerance"] - 15
                        )

                    # Influence action preferences
                    if "work" in memory_description.lower() and emotional_content > 0:
                        decision_context["preferred_actions"].append("Work")
                    elif "help" in memory_description.lower() and emotional_content > 0:
                        decision_context["preferred_actions"].append("Help")
                    elif (
                        "conflict" in memory_description.lower()
                        and emotional_content < 0
                    ):
                        decision_context["avoided_actions"].append("Attack")

            # Set emotional bias
            decision_context["emotional_bias"] = max(
                -50, min(50, total_emotional_impact)
            )

            # Adjust social inclination based on social memories
            if social_memories > 0:
                if positive_memories > negative_memories:
                    decision_context["social_inclination"] = min(
                        100,
                        decision_context["social_inclination"] + (social_memories * 10),
                    )
                else:
                    decision_context["social_inclination"] = max(
                        0,
                        decision_context["social_inclination"] - (social_memories * 5),
                    )

        # Store decision context for use by other systems
        self.memory_decision_context = decision_context
        return decision_context

    def retrieve_specific_memories(self, query):
        # Retrieve memories based on a specific query
        return self.memory_manager.retrieve_memories(query)

    def update_memory_importance(self, description, new_importance):
        # Example method to update the importance of a specific memory
        for memory in self.memory_manager.flat_access.get_all_memories():
            if memory.description == description:
                memory.importance_score = new_importance
                break

    def __repr__(self):
        return f"Character({self.name}, {self.age}, {self.pronouns}, {self.job}, {self.health_status}, {self.hunger_level}, {self.wealth_money}, {self.mental_health}, {self.social_wellbeing}, {self.job_performance}, {self.community}, {self.recent_event}, {self.long_term_goal}, {self.home}, {self.inventory}, {self.motives}, {self.personality_traits}, {self.skills}, {self.career_goals}, {self.possible_interactions}, {self.move_speed}, {self.location}, {self.energy}, {self.romanceable}, {self.romantic_relationships}, {self.exclusive_relationship}, {self.base_libido}, {self.monogamy}, {self.investment_portfolio}, {self.goals}, {self.needed_items}, {self.stamina}, {self.current_satisfaction}, {self.current_mood}, {self.current_activity}, {self.speed}, {self.path}, {self.destination})"

    def __str__(self):
        return f"\n{self.name} is a {self.age}-year-old {self.pronouns}. {pronoun_mapper[self.pronouns][0].capitalize()} is a {self.job.job_title if isinstance(self.job, JobRoles) else self.job} with {self.wealth_money} money. {pronoun_mapper[self.pronouns][0].capitalize()} is feeling {self.current_mood}. {pronoun_mapper[self.pronouns][0].capitalize()} is currently located at {self.location.name} with coordinates {self.location.get_coordinates()}. \n \
          {self.name} has the following investment portfolio: {self.investment_portfolio}. \n \
          {self.name} has the following motives: {self.motives}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following skills: {self.skills.get_skills_as_list_of_strings()}. \n \
          {self.name} has the following career goals: {[goal for goal in self.career_goals]}.\n \
          {self.name} has the following long-term goals: {self.long_term_goal}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following recent event: {self.recent_event}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following inventory: {self.inventory}. \n \
          {self.name} has the following personality traits: {self.personality_traits}. \n \
          {self.name} has the following romantic relationships: {self.romantic_relationships}. \n \
          {self.name} is currently in an exclusive relationship: {self.exclusive_relationship}. \n \
          {self.name} has the following investment portfolio: {self.investment_portfolio}. \n \
          {self.name} lives at {self.home.name} at {self.home.address} with a shelter level of {self.home.shelter_value} and a beauty level of {self.home.beauty_value}, with {self.home.bedrooms} bedrooms, {self.home.bathrooms} bathrooms, {self.home.stories} stories, and {self.home.area_val} square feet with a price of {self.home.price} and is located on the map at {self.home.get_coordinates()}.\n \
          {self.name} has the following energy: {self.energy}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following health status: {self.health_status}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following hunger level: {self.hunger_level}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following mental health: {self.mental_health}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following social wellbeing: {self.social_wellbeing}.\n \
          {self.name} has the following job performance: {self.job_performance}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following community: {self.community}. {pronoun_mapper[self.pronouns][0].capitalize()} has the current mood: {self.current_mood}{pronoun_mapper[self.pronouns][0].capitalize()} has the following stamina: {self.stamina}.\n \
          {self.name} has the following current satisfaction: {self.current_satisfaction}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following current activity: {self.current_activity}. \n \
          {self.name} has the following speed: {self.speed}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following path: {self.path}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following destination: {self.destination}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following move speed: {self.move_speed}. \n \
          {self.name} has the following romanceable status: {self.romanceable}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following monogamy level: {self.monogamy}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following base libido: {self.base_libido}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following exclusive relationship status: {[f'Exclusive with {self.exclusive_relationship}' if self.exclusive_relationship else 'Not exclusive']}.\n \
          {self.name} has the following luxury level: {self.luxury}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following success level: {self.success}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following control level: {self.control}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following hope level: {self.hope}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following happiness level: {self.happiness}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following stability level: {self.stability}.\n \
          {self.name} has the following shelter level: {self.shelter}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following beauty level: {self.beauty}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following state: {self.state}. \n \
          {self.name} has the following goals: {[goal[1].name for goal in self.goals]}.  \n \
          {pronoun_mapper[self.pronouns][0].capitalize()} has the following needed items: {self.needed_items}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following friendship grid: {self.friendship_grid}. \n \
          {self.name} has the following id: {self.uuid}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following character actions: {[action_name.name for action_name in self.character_actions]}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following possible interactions: {[action_name.name for action_name in self.possible_interactions]}.\n \
          {self.name} has the following romanceable status: {self.romanceable}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following romantic relationships: {self.romantic_relationships}. {pronoun_mapper[self.pronouns][0].capitalize()} has the following exclusive relationship status: {self.exclusive_relationship}.\n\n\n"

    def get_char_attribute(self, attribute):
        if attribute in self.__dict__:
            return self.__dict__[attribute]
        else:
            return None

    def __eq__(self, other):
        if not isinstance(other, Character):
            return False
        return (
            self.name == other.name
            and self.age == other.age
            and self.pronouns == other.pronouns
            and self.job == other.job
            and self.health_status == other.health_status
            and self.hunger_level == other.hunger_level
            and self.wealth_money == other.wealth_money
            and self.mental_health == other.mental_health
            and self.social_wellbeing == other.social_wellbeing
            and self.job_performance == other.job_performance
            and self.community == other.community
            and self.recent_event == other.recent_event
            and self.long_term_goal == other.long_term_goal
            and self.home == other.home
            and self.inventory == other.inventory
            and self.motives == other.motives
            and self.personality_traits == other.personality_traits
            and self.skills == other.skills
            and self.career_goals == other.career_goals
            and self.possible_interactions == other.possible_interactions
            and self.move_speed == other.move_speed
            and self.location == other.location
            and self.energy == other.energy
            and self.romanceable == other.romanceable
            and self.romantic_relationships == other.romantic_relationships
            and self.exclusive_relationship == other.exclusive_relationship
            and self.base_libido == other.base_libido
            and self.monogamy == other.monogamy
            and self.investment_portfolio == other.investment_portfolio
            and self.needed_items == other.needed_items
            and self.stamina == other.stamina
            and self.current_satisfaction == other.current_satisfaction
            and self.current_mood == other.current_mood
            and self.current_activity == other.current_activity
            and self.speed == other.speed
            and self.path == other.path
            and self.destination == other.destination
        )

    def get_possible_interactions(self):
        return self.possible_interactions

    def play_animation(self, animation):
        """Play an animation for this character using the animation system."""
        try:
            from tiny_animation_system import get_animation_system

            animation_system = get_animation_system()
            success = animation_system.play_animation(self.name, animation)

            if success:
                logging.info(f"{self.name} started playing animation: {animation}")
            else:
                logging.warning(f"Failed to play animation {animation} for {self.name}")

            return success
        except ImportError:
            # Fallback to simple print if animation system not available
            logging.info(f"{self.name} is playing animation: {animation}")
            return True
        except Exception as e:
            logging.error(f"Error playing animation {animation} for {self.name}: {e}")
            return False

    def describe(self):
        print(
            f"{self.name} is a {self.age}-year-old {self.gender_identity} with the following personality traits:"
        )
        print(
            f"Openness: {self.openness}, Conscientiousness: {self.conscientiousness}, Extraversion: {self.extraversion},"
        )
        print(f"Agreeableness: {self.agreeableness}, Neuroticism: {self.neuroticism}")

    def decide_to_join_event(self, event):
        if self.extraversion > 50:
            return True
        return False

    def decide_to_explore(self):
        if self.openness > 75:
            return True
        elif self.openness > 40 and self.conscientiousness < 50:
            return True
        return False

    def decide_to_take_challenge(self):
        if self.conscientiousness > 60 and self.neuroticism < 50:
            return "ready to tackle the challenge"
        elif self.agreeableness > 50 and self.neuroticism < 40:
            return "takes on the challenge to help others"
        return "too stressed to take on the challenge right now"

    def respond_to_conflict(self, conflict_level):
        if self.agreeableness > 65:
            return "seeks a peaceful resolution"
        elif self.neuroticism > 70:
            return "avoids the situation entirely"
        return "confronts the issue directly"

    def respond_to_talk(self, initiator):
        """
        Respond to a conversation initiated by another character.
        This method provides additional social interaction beyond the effects 
        handled by TalkAction.
        """
        # Give a small additional boost to social wellbeing when talked to
        self.social_wellbeing += 0.1
        
        # Boost based on personality compatibility
        if hasattr(initiator, 'personality_traits') and self.personality_traits:
            # Simple compatibility boost - agreeable characters get along better
            initiator_agreeable = getattr(initiator.personality_traits, 'agreeableness', 50)
            self_agreeable = getattr(self.personality_traits, 'agreeableness', 50)
            
            if initiator_agreeable > 60 and self_agreeable > 60:
                self.social_wellbeing += 0.1  # Bonus for agreeable personalities
                
        # Update friendship grid if it exists
        if hasattr(self, 'friendship_grid') and hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.1  # New acquaintance
            else:
                # Strengthen existing relationship slightly
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.05, 
                    1.0
                )
        
        # Return a response based on personality traits
        if hasattr(self.personality_traits, 'extraversion') and self.personality_traits.extraversion > 65:
            return f"{self.name} engages enthusiastically in conversation with {getattr(initiator, 'name', 'someone')}"
        elif hasattr(self.personality_traits, 'neuroticism') and self.personality_traits.neuroticism > 70:
            return f"{self.name} responds nervously but appreciates the attention from {getattr(initiator, 'name', 'someone')}"
        elif hasattr(self.personality_traits, 'openness') and self.personality_traits.openness > 70:
            return f"{self.name} shares interesting thoughts with {getattr(initiator, 'name', 'someone')}"
        else:
            return f"{self.name} listens and responds thoughtfully to {getattr(initiator, 'name', 'someone')}"

    def respond_to_greeting(self, initiator):
        """
        Respond to a greeting from another character.
        """
        # Small boost to social wellbeing from being greeted
        self.social_wellbeing += 0.05
        
        # Update friendship grid if it exists
        if hasattr(self, 'friendship_grid') and hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.05  # New acquaintance greeting
            else:
                # Strengthen existing relationship slightly
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.02, 
                    1.0
                )
        
        # Return greeting response based on personality
        initiator_name = getattr(initiator, 'name', 'someone')
        if hasattr(self.personality_traits, 'extraversion') and self.personality_traits.extraversion > 65:
            return f"{self.name} warmly greets {initiator_name} back"
        elif hasattr(self.personality_traits, 'neuroticism') and self.personality_traits.neuroticism > 70:
            return f"{self.name} shyly acknowledges {initiator_name}'s greeting"
        else:
            return f"{self.name} politely returns {initiator_name}'s greeting"

    def respond_to_compliment(self, initiator, compliment_topic):
        """
        Respond to a compliment from another character.
        """
        # Significant boost to social wellbeing from compliments
        self.social_wellbeing += 0.2
        
        # Compliments have a stronger effect on friendship
        if hasattr(self, 'friendship_grid') and hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.15  # Strong start for complimenting
            else:
                # Strengthen existing relationship more significantly
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.1, 
                    1.0
                )
        
        # Return compliment response based on personality
        initiator_name = getattr(initiator, 'name', 'someone')
        if hasattr(self.personality_traits, 'extraversion') and self.personality_traits.extraversion > 65:
            return f"{self.name} beams with joy at {initiator_name}'s compliment about {compliment_topic}"
        elif hasattr(self.personality_traits, 'neuroticism') and self.personality_traits.neuroticism > 70:
            return f"{self.name} blushes and thanks {initiator_name} for the kind words about {compliment_topic}"
        elif hasattr(self.personality_traits, 'agreeableness') and self.personality_traits.agreeableness > 70:
            return f"{self.name} graciously accepts {initiator_name}'s compliment about {compliment_topic}"
        else:
            return f"{self.name} appreciates {initiator_name}'s compliment about {compliment_topic}"

    def define_descriptors(self):
        """
        Creates comprehensive descriptors for the character based on their attributes,
        personality, relationships, and current state.
        """
        descriptors = {
            # Basic information
            "name": self.name,
            "age": self.age,
            "pronouns": self.pronouns,
            "job": self.job,
            # Physical and mental state
            "health_status": self.health_status,
            "mental_health": self.mental_health,
            "energy_level": self.energy,
            "hunger_level": self.hunger_level,
            # Social and economic status
            "wealth_money": self.wealth_money,
            "social_wellbeing": self.social_wellbeing,
            "job_performance": self.job_performance,
            "community_standing": self.community,
            # Personality descriptors
            "personality_summary": {
                "openness": (
                    self.personality_traits.get_openness()
                    if self.personality_traits
                    else 0
                ),
                "conscientiousness": (
                    self.personality_traits.get_conscientiousness()
                    if self.personality_traits
                    else 0
                ),
                "extraversion": (
                    self.personality_traits.get_extraversion()
                    if self.personality_traits
                    else 0
                ),
                "agreeableness": (
                    self.personality_traits.get_agreeableness()
                    if self.personality_traits
                    else 0
                ),
                "neuroticism": (
                    self.personality_traits.get_neuroticism()
                    if self.personality_traits
                    else 0
                ),
            },
            # Life status
            "recent_event": self.recent_event,
            "long_term_goal": self.long_term_goal,
            "current_mood": getattr(self, "current_mood", 50),
            "current_activity": getattr(self, "current_activity", "None"),
            # Living situation
            "home_status": "housed" if self.home else "homeless",
            "shelter_value": getattr(self, "shelter", 0),
            "stability": getattr(self, "stability", 0),
            # Calculated values
            "happiness": getattr(self, "happiness", 0),
            "success": getattr(self, "success", 0),
            "hope": getattr(self, "hope", 0),
            "luxury": getattr(self, "luxury", 0),
            "control": getattr(self, "control", 0),
            "beauty": getattr(self, "beauty", 0),
            # Relationships
            "friendship_count": (
                len(self.friendship_grid) if self.friendship_grid else 0
            ),
            "relationship_status": (
                "in_relationship"
                if getattr(self, "exclusive_relationship", None)
                else "single"
            ),
            "romanceable": getattr(self, "romanceable", True),
            # Goals and aspirations
            "goals_count": len(self.goals) if self.goals else 0,
            "has_career_goals": len(getattr(self, "career_goals", [])) > 0,
            # Inventory and possessions
            "item_count": self.inventory.count_total_items() if self.inventory else 0,
            "material_goods": getattr(self, "material_goods", 0),
            # Location
            "current_location": self.location.name if self.location else "Unknown",
            "coordinates": (
                self.coordinates_location
                if hasattr(self, "coordinates_location")
                else [0, 0]
            ),
        }

        # Generate text descriptors based on values
        text_descriptors = []

        # Health and wellbeing descriptors
        if descriptors["health_status"] > 8:
            text_descriptors.append("healthy and robust")
        elif descriptors["health_status"] < 4:
            text_descriptors.append("struggling with health issues")

        if descriptors["mental_health"] > 8:
            text_descriptors.append("mentally stable and positive")
        elif descriptors["mental_health"] < 4:
            text_descriptors.append("dealing with mental health challenges")

        # Personality descriptors
        personality = descriptors["personality_summary"]
        if personality["extraversion"] > 2:
            text_descriptors.append("outgoing and social")
        elif personality["extraversion"] < -2:
            text_descriptors.append("introverted and reserved")

        if personality["agreeableness"] > 2:
            text_descriptors.append("kind and cooperative")
        elif personality["agreeableness"] < -2:
            text_descriptors.append("competitive and skeptical")

        if personality["conscientiousness"] > 2:
            text_descriptors.append("organized and hardworking")
        elif personality["conscientiousness"] < -2:
            text_descriptors.append("spontaneous and flexible")

        # Economic descriptors
        if descriptors["wealth_money"] > 1000:
            text_descriptors.append("financially comfortable")
        elif descriptors["wealth_money"] < 50:
            text_descriptors.append("struggling financially")

        # Social descriptors
        if descriptors["friendship_count"] > 5:
            text_descriptors.append("well-connected socially")
        elif descriptors["friendship_count"] == 0:
            text_descriptors.append("socially isolated")

        # Professional descriptors
        if descriptors["job"] != "unemployed":
            if descriptors["job_performance"] > 80:
                text_descriptors.append(f"excelling as a {descriptors['job']}")
            elif descriptors["job_performance"] < 30:
                text_descriptors.append(
                    f"struggling in their role as a {descriptors['job']}"
                )
            else:
                text_descriptors.append(f"working as a {descriptors['job']}")
        else:
            text_descriptors.append("currently unemployed")

        descriptors["text_summary"] = ", ".join(text_descriptors)

        return descriptors

    def get_name(self):
        return self.name

    def set_name(self, name):
        # Warning: Name MUST be unique! Check for duplicates before setting.
        self.name = name
        return self.name

    def get_age(self):
        return self.age

    def set_age(self, age):
        self.age = age
        return self.age

    def get_pronouns(self):
        return self.pronouns

    def set_pronouns(self, pronouns):
        self.pronouns = pronouns
        return self.pronouns

    def get_job(self):
        return self.job

    def check_graph_uuid(self):
        return self.graph_manager.unique_graph_id

    def set_job(self, job):
        job_rules = JobRules()
        if isinstance(job, JobRoles):
            self.job = job
        elif isinstance(job, str):
            check = job_rules.check_job_name_validity(job)
            if check not in [None, False]:
                logging.info("Job valid")
                for job_role in job_rules.ValidJobRoles:
                    if (
                        job_role.get_job_name().lower() == job.lower()
                        or job_role.get_job_name().lower() in job.lower()
                    ) or (
                        job.lower() == job_role.get_job_name().lower()
                        or job.lower() in job_role.get_job_name().lower()
                    ):
                        logging.info("Job role found")
                        self.job = job_role
                    elif (
                        job_role.get_job_name().lower() == job.lower()
                        or job_role.get_job_name().lower() in job.lower()
                    ) or (
                        job.lower() == job_role.get_job_name().lower()
                        or job.lower() in job_role.get_job_name().lower()
                    ):
                        logging.info("Job role found")
                        self.job = job_role

            else:
                logging.info(f"Job role not found for job {job}")
                exit(12)
                self.job = random.choice(job_rules.ValidJobRoles)
        else:
            self.job = random.choice(job_rules.ValidJobRoles)
        return self.job

    def set_home(self, home=None):
        from tiny_buildings import House, CreateBuilding

        if isinstance(home, House):
            self.home = home
        elif isinstance(home, str):
            if home != "homeless":
                self.home = CreateBuilding().create_house_by_type(home)
            else:
                return None
        elif home is None:
            self.home = CreateBuilding().generate_random_house()
        else:
            raise TypeError("Invalid type for home")
        return self.home

    def get_home(self):
        return self.home

    def get_job_role(self):
        return self.job_role

    def set_job_role(self, job):
        job_rules = JobRules()
        if isinstance(job, JobRoles):
            self.job_role = job
        elif isinstance(job, str):
            if job_rules.check_job_name_validity(job):
                for job_role in job_rules.ValidJobRoles:
                    if job_role.get_job_name() == job:
                        self.job_role = job_role
            else:
                self.job_role = job_rules.ValidJobRoles[0]
        else:
            raise TypeError(f"Invalid type {type(job)} for job role")

    def get_health_status(self):
        return self.health_status

    def set_health_status(self, health_status):
        self.health_status = health_status
        return self.health_status

    def get_hunger_level(self):
        return self.hunger_level

    def set_hunger_level(self, hunger_level):
        self.hunger_level = hunger_level
        return self.hunger_level

    def get_wealth_money(self):
        return self.wealth_money

    def set_wealth_money(self, wealth_money):
        self.wealth_money = wealth_money
        return self.wealth_money

    def get_mental_health(self):
        return self.mental_health

    def set_mental_health(self, mental_health):
        self.mental_health = mental_health
        return self.mental_health

    def get_social_wellbeing(self):
        return self.social_wellbeing

    def set_social_wellbeing(self, social_wellbeing):
        self.social_wellbeing = social_wellbeing
        return self.social_wellbeing

    def get_happiness(self):
        return self.happiness

    def set_happiness(self, happiness):
        self.happiness = happiness
        return self.happiness

    def get_shelter(self):
        return self.shelter

    def set_shelter(self, shelter):
        self.shelter = shelter
        return self.shelter

    def get_stability(self):
        return self.stability

    def set_stability(self, stability):
        self.stability = stability
        return self.stability

    def get_luxury(self):
        return self.luxury

    def set_luxury(self, luxury):
        self.luxury = luxury
        return self.luxury

    def get_hope(self):
        return self.hope

    def set_hope(self, hope):
        self.hope = hope
        return self.hope

    def get_success(self):
        return self.success

    def set_success(self, success):
        self.success = success
        return self.success

    def get_control(self):
        return self.control

    def set_control(self, control):
        self.control = control
        return self.control

    def get_job_performance(self):
        return self.job_performance

    def set_job_performance(self, job_performance):
        self.job_performance = job_performance
        return self.job_performance

    def get_beauty(self):
        return self.beauty

    def set_beauty(self, beauty):
        self.beauty = beauty
        return self.beauty

    def get_community(self):
        return self.community

    def set_community(self, community):
        self.community = community
        return self.community

    def get_material_goods(self):
        return self.material_goods

    def set_material_goods(self, material_goods):
        self.material_goods = material_goods
        return self.material_goods

    def get_friendship_grid(self):
        return self.friendship_grid

    def generate_friendship_grid(self):
        """
        Generates a friendship grid based on character relationships in the graph manager.
        Returns a list of dictionaries containing relationship data.
        """
        friendship_grid = []

        try:
            # Get all character relationships from the graph manager
            relationships = self.graph_manager.analyze_character_relationships(self)

            for neighbor, relationship_data in relationships.items():
                if hasattr(neighbor, "name"):  # Ensure it's a character node
                    friendship_entry = {
                        "character_name": neighbor.name,
                        "character_id": neighbor,
                        "emotional_impact": relationship_data.get("emotional", 0),
                        "trust_level": relationship_data.get("trust", 0),
                        "relationship_strength": relationship_data.get("strength", 0),
                        "historical_bond": relationship_data.get("historical", 0),
                        "interaction_frequency": relationship_data.get(
                            "interaction_frequency", 0
                        ),
                        "friendship_status": self.graph_manager.check_friendship_status(
                            self, neighbor
                        ),
                    }

                    # Calculate overall friendship score
                    friendship_score = (
                        friendship_entry["emotional_impact"] * 0.3
                        + friendship_entry["trust_level"] * 0.25
                        + friendship_entry["relationship_strength"] * 0.25
                        + friendship_entry["historical_bond"] * 0.1
                        + friendship_entry["interaction_frequency"] * 0.1
                    )
                    friendship_entry["friendship_score"] = max(
                        0, min(100, friendship_score)
                    )

                    friendship_grid.append(friendship_entry)

        except Exception as e:
            logging.warning(f"Error generating friendship grid for {self.name}: {e}")
            # Return basic structure if graph analysis fails
            friendship_grid = [{}]

        return friendship_grid

    def set_friendship_grid(self, friendship_grid):
        if isinstance(friendship_grid, list) and len(friendship_grid) > 0:
            self.friendship_grid = friendship_grid
        else:
            self.friendship_grid = self.generate_friendship_grid()

    def get_recent_event(self):
        return self.recent_event

    def set_recent_event(self, recent_event):
        self.recent_event = recent_event
        return self.recent_event

    def get_long_term_goal(self):
        return self.long_term_goal

    def set_long_term_goal(self, long_term_goal):
        self.long_term_goal = long_term_goal
        return self.long_term_goal

    def get_inventory(self):
        return self.inventory

    def set_inventory(
        self,
        food_items: List[FoodItem] = [],
        clothing_items: List[ItemObject] = [],
        tools_items: List[ItemObject] = [],
        weapons_items: List[ItemObject] = [],
        medicine_items: List[ItemObject] = [],
        misc_items: List[ItemObject] = [],
    ):
        self.inventory = ItemInventory(
            food_items,
            clothing_items,
            tools_items,
            weapons_items,
            medicine_items,
            misc_items,
        )
        if food_items:
            self.inventory.set_food_items(food_items)
        if clothing_items:
            self.inventory.set_clothing_items(clothing_items)
        if tools_items:
            self.inventory.set_tools_items(tools_items)
        if weapons_items:
            self.inventory.set_weapons_items(weapons_items)
        if medicine_items:
            self.inventory.set_medicine_items(medicine_items)
        if misc_items:
            self.inventory.set_misc_items(misc_items)
        return self.inventory

    def set_goals(self, goals):
        self.goals = goals
        return self.goals

    # def calculate_goals(self):

    def evaluate_goals(self):

        goal_queue = []
        # if len(self.goals) > 1:
        if not self.goals or len(self.goals) == 0:
            goal_generator = GoalGenerator(
                self.motives, self.graph_manager, self.goap_planner, self.prompt_builder
            )
            self.goals = goal_generator.generate_goals(self)
        for _, goal in self.goals:
            utility = goal.evaluate_utility_function(self, goal, self.graph_manager)
            goal_queue.append((utility, goal))
        goal_queue.sort(reverse=True, key=lambda x: x[0])
        return goal_queue

    def calculate_material_goods(self):
        material_goods = round(
            tweener(self.inventory.count_total_items(), 1000, 0, 100.0, 2)
        )  # Tweening the wealth value
        return material_goods

    def calculate_stability(self):
        from tiny_graph_manager import cached_sigmoid_raw_approx_optimized

        stability = 0
        stability += self.get_shelter()
        stability += round(
            tweener(self.get_luxury(), 100.0, 0, 10, 2)
        )  # Tweening the luxury value
        stability += self.get_hope()
        stability += round(
            tweener(self.get_success(), 100.0, 0, 10, 2)
        )  # Tweening the success value
        stability += round(
            tweener(self.get_control(), 100.0, 0, 10, 2)
        )  # Tweening the control value
        stability += round(
            tweener(self.get_beauty(), 100.0, 0, 10, 2)
        )  # Tweening the job performance value
        stability += self.get_community()
        stability += round(
            tweener(self.get_material_goods(), 100.0, 0, 10, 2)
        )  # Tweening the material goods value
        stability += self.get_social_wellbeing()
        return cached_sigmoid_raw_approx_optimized(stability, 100.0)

    def calculate_happiness(self):
        from tiny_graph_manager import cached_sigmoid_raw_approx_optimized

        happiness = 0
        happiness += self.get_hope()
        happiness += cached_sigmoid_raw_approx_optimized(
            min(self.get_success(), self.motives.get_success_motive().get_score() * 10),
            100.0,
        )
        happiness += cached_sigmoid_raw_approx_optimized(
            min(self.get_control(), self.motives.get_control_motive().get_score() * 10),
            100.0,
        )
        happiness += cached_sigmoid_raw_approx_optimized(
            min(self.get_beauty(), self.motives.get_beauty_motive().get_score() * 10),
            100.0,
        )
        happiness += self.get_community()
        happiness += cached_sigmoid_raw_approx_optimized(
            min(
                self.get_job_performance(),
                self.motives.get_job_performance_motive().get_score() * 10,
            ),
            100.0,
        )
        # TODO: Add luxury calculation based on recent history of environmental luxury
        # happiness += cached_sigmoid_raw_approx_optimized(
        #     min(
        #         self.get_luxury(),
        #         self.motives.get_luxury_motive().get_score() * 10,
        #     ),
        #     100.0,
        # )
        happiness += cached_sigmoid_raw_approx_optimized(
            min(
                self.get_material_goods(),
                self.motives.get_material_goods_motive().get_score() * 10,
            ),
            100.0,
        )
        happiness += self.get_social_wellbeing()
        happiness *= cached_sigmoid_raw_approx_optimized(self.get_stability(), 100.0)
        happiness *= cached_sigmoid_raw_approx_optimized(self.get_hope(), 100.0)
        happiness *= cached_sigmoid_raw_approx_optimized(
            self.get_mental_health(), 100.0
        )

        # Add happiness calculation based on motives
        motive_satisfaction = 0
        if hasattr(self, "motives") and self.motives:
            try:
                # Average satisfaction across all motives
                basic_motives = [
                    self.motives.get_food_motive().get_score(),
                    self.motives.get_shelter_motive().get_score(),
                    self.motives.get_safety_motive().get_score(),
                    self.motives.get_social_motive().get_score(),
                ]
                motive_satisfaction = sum(basic_motives) / len(basic_motives)
                happiness += cached_sigmoid_raw_approx_optimized(
                    motive_satisfaction * 10, 100.0
                )
            except:
                pass

        # Add happiness calculation based on social relationships
        social_happiness = 0
        if hasattr(self, "relationships") and self.relationships:
            try:
                positive_relationships = sum(
                    1 for rel in self.relationships.values() if rel > 0.5
                )
                total_relationships = max(len(self.relationships), 1)
                social_score = (positive_relationships / total_relationships) * 100
                social_happiness = cached_sigmoid_raw_approx_optimized(
                    social_score, 100.0
                )
                happiness += social_happiness * 0.3  # Weight social relationships
            except:
                pass

        # Add happiness calculation based on romantic relationships
        romantic_happiness = 0
        if hasattr(self, "romantic_partner") and self.romantic_partner:
            try:
                # If has romantic partner, check relationship quality
                partner_relationship = self.relationships.get(self.romantic_partner, 0)
                if partner_relationship > 0.7:
                    romantic_happiness = cached_sigmoid_raw_approx_optimized(
                        partner_relationship * 100, 100.0
                    )
                    happiness += (
                        romantic_happiness * 0.2
                    )  # Weight romantic relationships
            except:
                pass

        # Add happiness calculation based on family relationships
        family_happiness = 0
        if hasattr(self, "family_members") and self.family_members:
            try:
                family_scores = []
                for family_member in self.family_members:
                    if family_member in self.relationships:
                        family_scores.append(self.relationships[family_member])

                if family_scores:
                    avg_family_score = sum(family_scores) / len(family_scores)
                    family_happiness = cached_sigmoid_raw_approx_optimized(
                        avg_family_score * 100, 100.0
                    )
                    happiness += family_happiness * 0.25  # Weight family relationships
            except:
                pass

        return cached_sigmoid_raw_approx_optimized(happiness, 100.0)

    def calculate_success(self):
        success = 0
        success += round(
            tweener(self.get_job_performance(), 100.0, 0, 50, 2)
        )  # Tweening the job performance value
        success += round(
            tweener(self.get_material_goods(), 100.0, 0, 20, 2)
        )  # Tweening the material goods value
        success += round(
            tweener(self.get_wealth_money(), 1000, 0, 20, 2)
        )  # Tweening the wealth value
        return success

    def calculate_control(self):
        control = 0
        control += self.get_shelter()
        control += round(tweener(self.get_success(), 100.0, 0, 10, 2))
        control += round(
            tweener(self.get_material_goods(), 100.0, 0, 20, 2)
        )  # Tweening the material goods value
        control += round(
            tweener(self.get_wealth_money(), 1000, 0, 20, 2)
        )  # Tweening the wealth value
        return control

    def calculate_monogamy(self):
        monogamy = 0
        monogamy -= self.personality_traits.get_openness()
        monogamy += self.personality_traits.get_conscientiousness()
        monogamy -= self.personality_traits.get_extraversion()
        monogamy += 1 if self.has_job() == False else 0
        monogamy += (
            1
            if self.get_wealth_money()
            <= self.graph_manager.get_average_attribute_value("wealth_money")
            else 0
        )
        logging.info(self.get_home())
        monogamy += 1 if self.get_home() is None else 0
        monogamy += max(0, (50 - self.base_libido) // 10)

        monogamy += self.age / 100.0
        monogamy += self.motives.get_control_motive().get_score()
        monogamy += self.motives.get_hope_motive().get_score()
        monogamy += self.motives.get_mental_health_motive().get_score()
        monogamy += self.motives.get_stability_motive().get_score()
        monogamy += self.motives.get_family_motive().get_score() * 2
        monogamy += self.stability / 100.0
        options_score = 0
        if self._initialized:
            romanceable_nodes = [
                node
                for node in list(self.graph_manager.G.nodes(data="romanceable"))
                if node[1] == True
            ]

            romanceable_nodes = sorted(
                romanceable_nodes,
                key=lambda x: self.graph_manager.G.edges(x, data=True)[0][2][
                    "romance_value"
                ],
                reverse=True,
            )

            for node in romanceable_nodes:
                options_score += (
                    self.graph_manager.G[self][node]["romance_value"] / 10
                ) * self.graph_manager.G[self][node]["romance_compatibility"]
            if len(romanceable_nodes) > 0:
                options_score = options_score / len(romanceable_nodes)
                options_score += max(10, max(0, (10 + len(romanceable_nodes)) // 10))
            interested_nodes = [
                node
                for node in self.graph_manager.G[self]
                if self.graph_manager.G[self][node]["romance_interest"] == True
            ]
            options_scoreb = 0
            for node in interested_nodes:
                options_scoreb += (
                    self.graph_manager.G[self][node]["romance_value"]
                    / 10
                    * self.graph_manager.G[self][node]["romance_compatibility"]
                )
            if len(interested_nodes) > 0:
                options_scoreb = options_scoreb / len(interested_nodes)
                options_scoreb += max(10, max(0, (10 + len(interested_nodes)) // 10))

        monogamy = max(1, (monogamy - options_score))
        # monogamy = max(1, (monogamy - options_scoreb))

        return monogamy

    def calculate_hope(self):
        from tiny_graph_manager import cached_sigmoid_raw_approx_optimized

        hope = 0
        hope += round(
            tweener(self.get_beauty(), 100.0, 0.0, 10.0, 2.0)
        )  # Tweening the luxury value
        hope += round(
            tweener(self.get_success(), 100.0, 0.0, 10.0, 2)
        )  # Tweening the success value
        hope += self.get_community()
        hope += round(
            tweener(self.get_material_goods(), 100.0, 0.0, 10, 2.0)
        )  # Tweening the material goods value
        hope += self.get_social_wellbeing()
        return cached_sigmoid_raw_approx_optimized(hope, 100.0)

    def calculate_base_libido(self):
        """Calculate the base libido based on the character's personality traits"""
        base_libido = 0
        base_libido += self.personality_traits.get_openness()
        base_libido += self.personality_traits.get_extraversion()
        base_libido += self.personality_traits.get_agreeableness()
        base_libido -= self.personality_traits.get_neuroticism()
        base_libido = max(0, base_libido)

        base_libido -= self.get_age() / 10
        base_libido = max(0, base_libido)

        base_libido += self.get_health_status() / 10
        base_libido += self.get_happiness() / 10
        base_libido += self.get_social_wellbeing() / 10
        base_libido += random.randint(-4, 4)
        base_libido += self.get_control() / 10
        base_libido = max(0, base_libido)
        # Scale and normalize the base libido to a range of 0-100.0
        base_libido = round(tweener(base_libido, 60, 0, 60, 2))

        return base_libido

    # def __repr__(self):
    #     try:
    #         return (
    #             f"Character(name={self.name}, job={self.job}, health_status={self.health_status}, "
    #             f"hunger_level={self.hunger_level}, wealth_money={self.wealth_money}, "
    #             f"long_term_goal={self.long_term_goal}, recent_event={self.recent_event}, "
    #             f"personality_traits={self.personality_traits}, motives={self.motives}, "
    #             f"inventory={self.inventory}, home={self.home}, community={self.community}, "
    #             f"beauty={self.beauty}, control={self.control}, success={self.success}, "
    #             f"hope={self.hope}, stability={self.stability}, luxury={self.luxury}, "
    #             f"romanceable={self.romanceable}, base_libido={self.base_libido}, "
    #             f"monogamy={self.monogamy}, romantic_relationships={self.romantic_relationships}, "
    #             f"exclusive_relationship={self.exclusive_relationship}, skills={self.skills}, "
    #             f"career_goals={self.career_goals}, short_term_goals={self.short_term_goals}, "
    #             f"id={self.uuid}, location={self.location}, coordinates_location={self.coordinates_location}, "
    #             f"goals={self.goals}, state={self.state}, character_actions={self.character_actions}, "
    #             f"speed={self.speed}, energy={self.energy}, pronouns={self.pronouns}, "
    #             f"age={self.age}, "
    #             f"destination={self.destination}, path={self.path}, needed_items={self.needed_items}, "
    #             f"possible_interactions={self.possible_interactions}, "
    #             f"job_performance={self.job_performance}, "
    #             f"friendship_grid={self.friendship_grid}, material_goods={self.material_goods})"
    #         )
    #     except Exception as e:
    #         return f"Character(An error occurred in __repr__: {e})"

    def has_job(self):
        if isinstance(self.job, str):
            if self.job == "unemployed":
                return False
        else:
            return self.job is not None and self.job.job_name != "unemployed"

    def has_investment(self):
        return len(self.investment_portfolio.get_stocks()) > 0

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
                    self.job,
                    self.health_status,
                    self.hunger_level,
                    self.wealth_money,
                    self.long_term_goal,
                    self.recent_event,
                    make_hashable(self.personality_traits.to_dict()),
                    self.motives,
                    self.inventory,
                    self.home,
                    self.community,
                    self.beauty,
                    self.control,
                    self.success,
                    self.hope,
                    self.stability,
                    self.luxury,
                    self.romanceable,
                    self.base_libido,
                    self.monogamy,
                    make_hashable(self.romantic_relationships),
                    self.exclusive_relationship,
                    self.skills,
                    make_hashable(self.career_goals),
                    make_hashable(self.short_term_goals),
                    self.uuid,
                    self.location,
                    tuple(self.coordinates_location),
                    make_hashable(self.goals),
                    self.speed,
                    self.energy,
                    make_hashable(self.pronouns),
                    self.age,
                    self.destination,
                    make_hashable(self.path),
                    self.job_performance,
                ]
            )
        )

    def update_character(
        self,
        job=None,
        health_status=None,
        hunger_level=None,
        wealth_money=None,
        long_term_goal=None,
        recent_event=None,
        mental_health=None,
        social_wellbeing=None,
        shelter=None,
        stability=None,
        luxury=None,
        hope=None,
        success=None,
        control=None,
        job_performance=None,
        beauty=None,
        community=None,
        material_goods=None,
        friendship_grid=None,
        food_items: List[FoodItem] = [],
        clothing_items: List[ItemObject] = [],
        tools_items: List[ItemObject] = [],
        weapons_items: List[ItemObject] = [],
        medicine_items: List[ItemObject] = [],
        misc_items: List[ItemObject] = [],
        personality_traits=None,
        motives=None,
    ):
        if job:
            self.job = self.set_job(job)
        if health_status:
            self.health_status = self.set_health_status(health_status)
        if hunger_level:
            self.hunger_level = self.set_hunger_level(hunger_level)
        if wealth_money:
            self.wealth_money = self.set_wealth_money(wealth_money)
        if mental_health:
            self.mental_health = self.set_mental_health(mental_health)
        if social_wellbeing:
            self.social_wellbeing = self.set_social_wellbeing(social_wellbeing)

        if shelter:
            self.shelter = self.set_shelter(shelter)
        if stability:
            self.stability = self.set_stability(stability)
        if luxury:
            self.luxury = self.set_luxury(luxury)
        if hope:
            self.hope = self.set_hope(hope)
        if success:
            self.success = self.set_success(success)
        if control:
            self.control = self.set_control(control)
        if job_performance:
            self.job_performance = self.set_job_performance(job_performance)
        if beauty:
            self.beauty = self.set_beauty(beauty)
        if community:
            self.community = self.set_community(community)
        if material_goods:
            self.material_goods = self.set_material_goods(material_goods)
        if friendship_grid:
            self.friendship_grid = self.set_friendship_grid(friendship_grid)
        if recent_event:
            self.recent_event = self.set_recent_event(recent_event)
        if long_term_goal:
            self.long_term_goal = self.set_long_term_goal(long_term_goal)
        if food_items:
            self.inventory.set_food_items(food_items)
        if clothing_items:
            self.inventory.set_clothing_items(clothing_items)
        if tools_items:
            self.inventory.set_tools_items(tools_items)
        if weapons_items:
            self.inventory.set_weapons_items(weapons_items)
        if medicine_items:
            self.inventory.set_medicine_items(medicine_items)
        if misc_items:
            self.inventory.set_misc_items(misc_items)
        if personality_traits:
            self.personality_traits = self.set_personality_traits(personality_traits)

        if motives:
            self.motives = self.set_motives(motives)

        return self

    def get_personality_traits(self):
        return self.personality_traits

    def set_personality_traits(self, personality_traits):
        logging.info(f"Setting personality traits: \n{personality_traits}")
        if isinstance(personality_traits, PersonalityTraits):
            self.personality_traits = personality_traits
            return self.personality_traits
        elif isinstance(personality_traits, dict):
            self.personality_traits = PersonalityTraits(
                personality_traits["openness"],
                personality_traits["conscientiousness"],
                personality_traits["extraversion"],
                personality_traits["agreeableness"],
                personality_traits["neuroticism"],
            )
            return self.personality_traits
        else:
            raise TypeError("Invalid type for personality traits")

    def get_motives(self):
        return self.motives

    def set_motives(self, motives):
        self.motives = motives
        return self.motives

    def get_character_data(self):
        response = self.to_dict()
        return response.json()

    def update_required_items(self, item_node_attrs, item_count=1):
        for item in item_node_attrs:
            self.needed_items.append((item, item_count))

    def calculate_motives(self):
        return self.graph_manager.calculate_motives(self)

    def to_dict(self):
        return {
            "name": self.name,
            "age": self.age,
            "pronouns": self.pronouns,
            "job": self.job,
            "health_status": self.health_status,
            "hunger_level": self.hunger_level,
            "wealth_money": self.wealth_money,
            "mental_health": self.mental_health,
            "social_wellbeing": self.social_wellbeing,
            "happiness": self.happiness,
            "shelter": self.shelter,
            "stability": self.stability,
            "luxury": self.luxury,
            "hope": self.hope,
            "success": self.success,
            "control": self.control,
            "job_performance": self.job_performance,
            "beauty": self.beauty,
            "community": self.community,
            "material_goods": self.material_goods,
            "friendship_grid": self.friendship_grid,
            "recent_event": self.recent_event,
            "long_term_goal": self.long_term_goal,
            "inventory": self.inventory,
            "home": self.home,
            "personality_traits": self.personality_traits,
            "motives": self.motives,
            "base_libido": self.base_libido,
            "monogamy": self.monogamy,
            "romantic_relationships": self.romantic_relationships,
            "exclusive_relationship": self.exclusive_relationship,
            "skills": self.skills,
            "career_goals": self.career_goals,
            "short_term_goals": self.short_term_goals,
            "location": self.location,
            "coordinates_location": self.coordinates_location,
            "goals": self.goals,
            "speed": self.speed,
            "energy": self.energy,
            "destination": self.destination,
            "path": self.path,
            "needed_items": self.needed_items,
            "possible_interactions": self.possible_interactions,
            "community": self.community,
        }

    def get_state(self):
        return State(self)


def default_long_term_goal_generator(character: Character):
    # goal:requirements

    goals = {
        "get a job": character.get_job().get_job_name() != "unemployed",
        "get a promotion": character.get_job_performance() > 50
        and character.get_job().get_job_name() != "unemployed",
        "get a raise": character.get_job_performance() > 50
        and character.get_job().get_job_name() != "unemployed",
        # "get married": character.get_relationship_status() == "married",
        # "have a child": character.get_children_count() > 0,
        "buy a house": character.get_shelter() > 0,
        "buy a mansion": character.get_luxury() > 0,
        "expanding shop": character.get_job().get_job_name() in ["merchant", "artisan"]
        and character.get_success() > 50,
        "continue innovating": character.get_job().get_job_name()
        in ["engineer", "doctor"]
        and character.get_success() > 20,
    }
    valid_goals = [goal for goal, requirement in goals.items() if requirement]
    return random.choice(valid_goals)


def recent_event_generator(character: Character):
    return random.choice(
        [
            "got a job",
            "movied into new home",
            "got a raise",
            "bought a house",
            "went on vacation",
        ]
    )


class CreateCharacter:
    def __init__(self):
        self.description = "This class is used to create a character."

    def __repr__(self):
        return f"CreateCharacter()"

    def create_new_character(
        self,
        mode: str = "auto",
        name: str = "John Doe",
        age: int = 18,
        pronouns: str = "they/them",
        job: str = "unemployed",
        health_status: float = 0.0,
        hunger_level: float = 0.0,
        wealth_money: float = 0.0,
        mental_health: float = 0.0,
        social_wellbeing: float = 0.0,
        job_performance: float = 0.0,
        community: float = 0.0,
        friendship_grid: dict = {},
        recent_event: str = "",
        long_term_goal: str = "",
        inventory: ItemInventory = None,
        personality_traits: PersonalityTraits = None,
        motives: PersonalMotives = None,
        home: str = "",
    ):
        from tiny_buildings import House, CreateBuilding

        if mode != "auto":
            if name == "John Doe":
                name = input("What is your character's name? ")
            if age == 18:
                age = int(input("What is your character's age? "))
                if age < 18:
                    age = 18
            if pronouns == "they/them":
                pronouns = input("What are your character's gender pronouns? ")
            if job == "unemployed":
                job = JobRules().check_job_name_validity(
                    input("What is your character's job? ")
                )
            if wealth_money == 0:
                wealth_money = int(input("How much money does your character have? "))
            if mental_health == 0:
                mental_health = int(
                    input(
                        "How mentally healthy is your character? A number between 0 and 10"
                    )
                )
            if social_wellbeing == 0:
                social_wellbeing = int(
                    input(
                        "How socially well is your character? A number between 0 and 10"
                    )
                )

            if job_performance == 0:
                job_performance = int(
                    input(
                        "How well does your character perform at their job? A number between 0 and 100.0"
                    )
                )
            if home == "":
                home = CreateBuilding().create_house_by_type(
                    input("Where does your character live? ")
                )
            if recent_event == "":
                recent_event = input(
                    "What is the most recent event that happened to your character?"
                )
            if long_term_goal == "":
                long_term_goal = input("What is your character's long term goal? ")
            if personality_traits is None:
                openness = int(
                    input("How open is your character? A number between -4 and 4.")
                )
                conscientiousness = int(
                    input(
                        "How conscientious is your character? A number between -4 and 4."
                    )
                )
                extraversion = int(
                    input(
                        "How extraverted is your character? A number between -4 and 4."
                    )
                )
                agreeableness = int(
                    input("How agreeable is your character? A number between -4 and 4.")
                )
                neuroticism = int(
                    input("How neurotic is your character? A number between -4 and 4.")
                )
                personality_traits = PersonalityTraits(
                    openness,
                    conscientiousness,
                    extraversion,
                    agreeableness,
                    neuroticism,
                )

            if motives is None:
                hunger_motive = int(
                    input("How hungry is your character? A number between 0 and 10.")
                )
                wealth_motive = int(
                    input(
                        "How much does your character want to accumulate wealth? A number between 0 and 10."
                    )
                )
                mental_health_motive = int(
                    input(
                        "How much does your character want to maintain their mental health? A number between 0 and 10."
                    )
                )
                social_wellbeing_motive = int(
                    input(
                        "How much does your character want to maintain their social wellbeing? A number between 0 and 10."
                    )
                )
                happiness_motive = int(
                    input(
                        "How much does your character want to maintain their happiness? A number between 0 and 10."
                    )
                )
                health_motive = int(
                    input(
                        "How much does your character want to maintain their health? A number between 0 and 10."
                    )
                )
                shelter_motive = int(
                    input(
                        "How much does your character want to maintain their shelter? A number between 0 and 10."
                    )
                )
                stability_motive = int(
                    input(
                        "How much does your character want to maintain their stability? A number between 0 and 10."
                    )
                )
                luxury_motive = int(
                    input(
                        "How much does your character want to maintain their luxury? A number between 0 and 10."
                    )
                )
                hope_motive = int(
                    input(
                        "How much does your character want to maintain their hope? A number between 0 and 10."
                    )
                )
                success_motive = int(
                    input(
                        "How much does your character want to maintain their success? A number between 0 and 10."
                    )
                )
                control_motive = int(
                    input(
                        "How much does your character want to maintain their control? A number between 0 and 10."
                    )
                )
                job_performance_motive = int(
                    input(
                        "How much does your character want to maintain their job performance? A number between 0 and 10."
                    )
                )
                beauty_motive = int(
                    input(
                        "How much does your character want to maintain their beauty? A number between 0 and 10."
                    )
                )
                community_motive = int(
                    input(
                        "How much does your character want to maintain their community? A number between 0 and 10."
                    )
                )
                material_goods_motive = int(
                    input(
                        "How much does your character want to maintain their material goods? A number between 0 and 10."
                    )
                )
                motives = PersonalMotives(
                    hunger_motive=Motive(
                        "hunger", "bias toward satisfying hunger", hunger_motive
                    ),
                    wealth_motive=Motive(
                        "wealth", "bias toward accumulating wealth", wealth_motive
                    ),
                    mental_health_motive=Motive(
                        "mental health",
                        "bias toward maintaining mental health",
                        mental_health_motive,
                    ),
                    social_wellbeing_motive=Motive(
                        "social wellbeing",
                        "bias toward maintaining social wellbeing",
                        social_wellbeing_motive,
                    ),
                    happiness_motive=Motive(
                        "happiness",
                        "bias toward maintaining happiness",
                        happiness_motive,
                    ),
                    health_motive=Motive(
                        "health", "bias toward maintaining health", health_motive
                    ),
                    shelter_motive=Motive(
                        "shelter", "bias toward maintaining shelter", shelter_motive
                    ),
                    stability_motive=Motive(
                        "stability",
                        "bias toward maintaining stability",
                        stability_motive,
                    ),
                    luxury_motive=Motive(
                        "luxury", "bias toward maintaining luxury", luxury_motive
                    ),
                    hope_motive=Motive(
                        "hope", "bias toward maintaining hope", hope_motive
                    ),
                    success_motive=Motive(
                        "success", "bias toward maintaining success", success_motive
                    ),
                    control_motive=Motive(
                        "control", "bias toward maintaining control", control_motive
                    ),
                    job_performance_motive=Motive(
                        "job performance",
                        "bias toward maintaining job performance",
                        job_performance_motive,
                    ),
                    beauty_motive=Motive(
                        "beauty", "bias toward maintaining beauty", beauty_motive
                    ),
                    community_motive=Motive(
                        "community",
                        "bias toward maintaining community",
                        community_motive,
                    ),
                    material_goods_motive=Motive(
                        "material goods",
                        "bias toward maintaining material goods",
                        material_goods_motive,
                    ),
                )

        elif mode == "auto":
            if pronouns == "they/them":
                r_val = random.random()
                if random.random() < 0.33:
                    pronouns = "he/him"
                elif random.random() < 0.66:
                    pronouns = "she/her"
                else:
                    pronouns = "they/them"
            if name == "John Doe":
                name = RandomNameGenerator().generate_name(pronouns=pronouns)

            if age == 18:
                age = round(random.gauss(21, 20))
                if age < 18:
                    age = 18

            if job == "unemployed":
                job = JobRules().ValidJobRoles[
                    random.randint(0, len(JobRules().ValidJobRoles) - 1)
                ]
            if wealth_money == 0:
                wealth_money = round(abs(random.triangular(10, 100000, 1000)))
            if mental_health == 0:
                mental_health = random.randint(8, 10)
            if social_wellbeing == 0:
                social_wellbeing = random.randint(8, 10)
            if job_performance == 0:
                job_performance = round(random.gauss(50, 20))
            if home == "":
                home = CreateBuilding().generate_random_house()

            if personality_traits is None:
                openness = max(-4, min(4, random.gauss(0, 2)))
                conscientiousness = max(-4, min(4, random.gauss(0, 2)))
                extraversion = max(-4, min(4, random.gauss(0, 2)))
                agreeableness = max(-4, min(4, random.gauss(0, 2)))
                neuroticism = max(-4, min(4, random.gauss(0, 2)))
                personality_traits = PersonalityTraits(
                    openness,
                    conscientiousness,
                    extraversion,
                    agreeableness,
                    neuroticism,
                )

        if health_status == 0:
            health_status = random.randint(8, 10)
        if hunger_level == 0:
            hunger_level = random.randint(0, 2)

        created_char = Character(
            name,
            age,
            pronouns,
            job,
            health_status,
            hunger_level,
            wealth_money,
            mental_health,
            social_wellbeing,
            job_performance,
            community,
            home=home,
            personality_traits=personality_traits,
        )
        created_char.set_happiness(created_char.calculate_happiness())
        created_char.set_stability(created_char.calculate_stability())
        created_char.set_control(created_char.calculate_control())
        created_char.set_success(created_char.calculate_success())
        created_char.set_hope(created_char.calculate_hope())
        created_char.set_material_goods(created_char.calculate_material_goods())
        created_char.set_shelter(created_char.home.calculate_shelter_value())

        if recent_event == "":
            recent_event = recent_event_generator(created_char)
        if long_term_goal == "":
            long_term_goal = default_long_term_goal_generator(created_char)

        if motives is None:

            motives = created_char.calculate_motives()

        if motives is not None:
            for key, val in motives.to_dict().items():
                print(key, val)
        return created_char.update_character(
            recent_event=recent_event, long_term_goal=long_term_goal, motives=motives
        )


if __name__ == "__main__":
    gametime_manager = GameTimeManager()
    action_system = ActionSystem()
