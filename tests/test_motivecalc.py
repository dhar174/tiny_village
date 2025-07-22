# Assuming the necessary classes and their dependencies are properly imported
import importlib
import json
import random
from numpy import sort
from torch import NoneType, ge, rand
from actions import Skill
from tiny_characters import Motive
from tiny_items import InvestmentPortfolio, ItemInventory, ItemObject, Stock
from tiny_time_manager import GameTimeManager as gametime_manager
# Import real classes for proper integration testing
from tiny_characters import Character
from tiny_locations import Location  
from actions import ActionSystem

from tiny_graph_manager import GraphManager
import logging

logging.basicConfig(level=logging.DEBUG)


class TestMotiveCalc:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.graph_uuid = self.graph_manager.unique_graph_id

    def test_calculate_motives(self):

        from tiny_characters import Motive
        from actions import Action

        action_system = ActionSystem()

        # Create an instance of PersonalityTraits with default or sample values
        # personality_traits = PersonalityTraits()  # Use appropriate arguments if needed

        # Create an instance of Character with necessary attributes
        character = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=20,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 4,
                "openness": 4,
                "conscientiousness": 4,
                "agreeableness": 4,
                "neuroticism": -4,
            },
            action_system=action_system,
            gametime_manager=gametime_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        # Call the calculate_motives method and store the result
        motives = character.get_motives()
        print(character.base_libido)
        print("\n\n")
        # Print the result to check the output
        for motive in motives.to_dict().values():
            print(motive.name, motive.score)
            print("\n")


def sanitize_input(input_str, also_remove=[]):
    # Replace new lines with spaces or any desired character
    if isinstance(input_str, list):
        for i in range(len(input_str)):
            input_str[i] = str(input_str[i].strip())
            for item in also_remove:
                input_str[i] = input_str[i].replace(item, "")
            input_str[i] = input_str[i].replace("\n", " ")
        return input_str
    input_str = str(input_str.strip())
    for item in also_remove:
        input_str = input_str.replace(item, "")
    return input_str.replace("\n", " ")


class CreateRandomizedCharacter:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.graph_uuid = self.graph_manager.unique_graph_id
        with open("first_names_she.txt", "r") as f:
            self.first_names_she = f.readlines()
        with open("first_names_he.txt", "r") as f:
            self.first_names_he = f.readlines()
        with open("last_names.txt", "r") as f:
            self.last_names = f.readlines()
        self.used_names = []
        self.families = {}  # {last_name: [first_name1, first_name2, ...]}
        self.jobs = None
        with open("job_roles.json", "r") as f:
            job_roles = json.load(f)
            self.jobs = [
                jobname
                for job_role in job_roles["jobs"]
                for jobname in job_roles["jobs"][job_role]
            ]
        # Remove all entries that say "salary" or "associated character motives"
        self.jobs = [
            job for job in self.jobs if "salary" not in job and "motives" not in job
        ]
        logging.debug(f"Job #3 for testing: {self.jobs[3]}")
        logging.debug(f"All jobs: {self.jobs}")
        # exit(1)
        self.job_occurrences = {j: 0 for j in self.jobs}
        self.ages = []
        self.recent_events = []
        with open("recent_event_list.txt", "r") as f:
            self.recent_events = f.readlines()
        self.used_recent_events = {u: 0 for u in self.recent_events}
        self.longterm_goals = []
        with open("longterm_goal_list.txt", "r") as f:
            self.longterm_goals = f.readlines()
        self.used_longterm_goals = {u: 0 for u in self.longterm_goals}

    def get_random_name(self):
        first_name = (
            random.choice(self.first_names_she).strip().replace("\n", "").capitalize()
        )
        last_name = (
            random.choice(self.last_names).strip().replace("\n", "").capitalize()
        )
        name = f"{first_name} {last_name}"
        if name in self.used_names:
            return self.get_random_name()
        self.used_names.append(name)
        if name.split(" ")[1] in self.families:
            self.families[name.split(" ")[1]].append(name.split(" ")[0])

        return name

    def get_random_job(self):
        # select a job with a weighted random distribution
        job = random.choices(
            self.jobs,
            weights=[1 / (2 ** self.job_occurrences[j]) for j in self.jobs],
            k=1,
        )
        self.job_occurrences[job[0]] += 1
        job = sanitize_input(job[0])

        return job.strip().replace("\n", "")

    def get_random_age(self):
        # select an age with a weighted random distribution that leans towards young adults
        age = random.choices(
            [i for i in range(18, 60)],
            weights=[1 / (2 ** (abs(25 - i))) for i in range(18, 60)],
            k=1,
        )
        self.ages.append(age[0])
        return age[0]

    def get_random_recent_event(self):
        # select a recent event with a weighted random distribution
        event = random.choices(
            self.recent_events,
            weights=[1 / (2 ** self.used_recent_events[e]) for e in self.recent_events],
            k=1,
        )

        self.used_recent_events[event[0]] += 1
        event = sanitize_input(event[0])
        return event

    def get_random_long_term_goal(self):
        # select a long term goal with a weighted random distribution
        goal = random.choices(
            self.longterm_goals,
            weights=[
                1 / (2 ** self.used_longterm_goals[g]) for g in self.longterm_goals
            ],
            k=1,
        )
        self.used_longterm_goals[goal[0]] += 1
        goal = sanitize_input(goal[0])

        return goal

    def get_random_personality_traits(self):
        # select a personality trait with a weighted random distribution
        return {
            "extraversion": random.randint(-4, 4),
            "openness": random.randint(-4, 4),
            "conscientiousness": random.randint(-4, 4),
            "agreeableness": random.randint(-4, 4),
            "neuroticism": random.randint(-4, 4),
        }

    def create_random_character(self):
        from tiny_characters import Character
        from tiny_locations import Location
        from actions import ActionSystem

        action_system = ActionSystem()
        name = self.get_random_name()

        age = self.get_random_age()
        job = self.get_random_job()
        pronouns = "she/her"

        character = Character(
            name=name,
            age=age,
            pronouns=pronouns,
            job=job,
            health_status=random.randint(0, 100),
            hunger_level=random.randint(0, 100),
            wealth_money=int(max(abs(random.gauss(100, 20000)), 20)),
            mental_health=random.randint(0, 100),
            social_wellbeing=random.randint(0, 100),
            job_performance=random.randint(0, 100),
            community=random.randint(0, 100),
            friendship_grid=[],
            recent_event=self.get_random_recent_event(),
            long_term_goal=self.get_random_long_term_goal(),
            personality_traits=self.get_random_personality_traits(),
            action_system=action_system,
            gametime_manager=gametime_manager,
            location=Location(
                name + " Home",
                random.randint(0, 100),
                random.randint(0, 100),
                1,
                1,
                action_system,
            ),
            graph_manager=self.graph_manager,
        )
        if character.check_graph_uuid() != self.graph_uuid:
            logging.error(
                f"Graph UUID does not match. Character graph is {character.check_graph_uuid} and expected graph is {self.graph_uuid}"
            )
            exit(1)
        return character


import json

if __name__ == "__main__":
    test = TestMotiveCalc()
    test.test_calculate_motives()
    create_random = CreateRandomizedCharacter()
    random_characters = [create_random.create_random_character() for _ in range(100)]
    uuids = []

    PersonalMotives = importlib.import_module("tiny_characters").PersonalMotives
    Goal = importlib.import_module("tiny_characters").Goal

    motives = {
        motive_name: [{}]
        for motive_name in [
            "hunger",
            "wealth",
            "mental health",
            "social wellbeing",
            "happiness",
            "health",
            "shelter",
            "stability",
            "luxury",
            "hope",
            "success",
            "control",
            "job performance",
            "beauty",
            "community",
            "material goods",
            "family",
        ]
    }
    # Check that each character has been added to the graph and can be found
    for character in random_characters:
        uuids.append(character.uuid)
        if not character.check_graph_uuid() == create_random.graph_uuid:
            logging.error(
                f"Character {character.name} has a different graph UUID than expected. Expected {create_random.graph_uuid}, got {character.check_graph_uuid()}"
            )
            exit(1)
        if not create_random.graph_manager.get_node(character.name):
            logging.error(
                f"Character {character.name} not found in graph. Graph UUID: {create_random.graph_uuid}"
            )
            exit(1)
        # Check the range of the character class attributes
        if not 0 <= character.health_status <= 100:
            logging.error(
                f"Character {character.name} has an invalid health status: {character.health_status}"
            )
            exit(1)
        if not 0 <= character.hunger_level <= 100:
            logging.error(
                f"Character {character.name} has an invalid hunger level: {character.hunger_level}"
            )
            exit(1)
        if not 0 <= character.wealth_money:
            logging.error(
                f"Character {character.name} has an invalid wealth: {character.wealth_money}"
            )
            exit(1)
        if not 0 <= character.mental_health <= 100:
            logging.error(
                f"Character {character.name} has an invalid mental health: {character.mental_health}"
            )
            exit(1)
        if not 0 <= character.social_wellbeing <= 100:
            logging.error(
                f"Character {character.name} has an invalid social wellbeing: {character.social_wellbeing}"
            )
            exit(1)
        if not 0 <= character.job_performance <= 100:
            logging.error(
                f"Character {character.name} has an invalid job performance: {character.job_performance}"
            )
            exit(1)
        if not 0 <= character.community <= 100:
            logging.error(
                f"Character {character.name} has an invalid community: {character.community}"
            )
            exit(1)
        if not character.personality_traits.get_extraversion() in range(-4, 5):
            logging.error(
                f"Character {character.name} has an invalid extraversion: {character.personality_traits.get_extraversion()}"
            )
            exit(1)
        if not character.personality_traits.get_openness() in range(-4, 5):
            logging.error(
                f"Character {character.name} has an invalid openness: {character.personality_traits.get_openness()}"
            )
            exit(1)
        if not character.personality_traits.get_conscientiousness() in range(-4, 5):
            logging.error(
                f"Character {character.name} has an invalid conscientiousness: {character.personality_traits.get_conscientiousness()}"
            )
            exit(1)
        if not character.personality_traits.get_agreeableness() in range(-4, 5):
            logging.error(
                f"Character {character.name} has an invalid agreeableness: {character.personality_traits.get_agreeableness()}"
            )
            exit(1)
        if not character.personality_traits.get_neuroticism() in range(-4, 5):
            logging.error(
                f"Character {character.name} has an invalid neuroticism: {character.personality_traits.get_neuroticism()}"
            )
            exit(1)
        if not character.recent_event in sanitize_input(create_random.recent_events):
            logging.error(
                f"Character {character.name} has an invalid recent event: {character.recent_event}. Here are the valid recent events: {create_random.recent_events}"
            )
            exit(1)
        if not character.long_term_goal in sanitize_input(create_random.longterm_goals):
            logging.error(
                f"Character {character.name} has an invalid long term goal: {character.long_term_goal}. Here are the valid long term goals: {create_random.longterm_goals}"
            )
            exit(1)
        if not character.job.job_name in sanitize_input(create_random.jobs):
            logging.error(
                f"Character {character.name} has an invalid job: {character.job.job_name}. Here are the valid jobs: {create_random.jobs}"
            )
            exit(1)
        if not character.age in create_random.ages:
            logging.error(
                f"Character {character.name} has an invalid age: {character.age}. Here are the valid ages: {create_random.ages}"
            )
            exit(1)
        for traitname, trait in character.personality_traits.to_dict().items():
            if not -4 <= trait <= 4:
                logging.error(
                    f"Character {character.name} has an invalid personality trait: {trait}"
                )
                exit(1)
            if trait != character.personality_traits.get_personality_trait(traitname):
                logging.error(
                    f"Character {character.name} has an invalid personality trait: {trait}"
                )
                exit(1)
        if not character.get_motives().to_dict():
            logging.error(
                f"Character {character.name} has an invalid motives: {character.get_motives().to_dict()}"
            )
            exit(1)
        if not character.goals:
            logging.error(f"Character {character.name} has no goals")
            exit(1)
        if not character.get_state():
            logging.error(f"Character {character.name} has no state")
            exit(1)
        # if not character.get_state().to_dict():
        #     logging.error(
        #         f"Character {character.name} has an invalid state: {character.get_state().to_dict()}"
        #     )
        #     exit(1)
        if character.name in create_random.families:
            if (
                not character.name.split(" ")[0]
                in create_random.families[character.name.split(" ")[1]]
            ):
                logging.error(f"Character {character.name} is not in the family list")
                exit(1)
        if character.get_home() is None:
            logging.error(f"Character {character.name} has no home")
            exit(1)
        if not character.get_home().get_coordinates():
            logging.error(f"Character {character.name} has an invalid home coordinates")
            exit(1)
        for attribute in [
            "health_status",
            "hunger_level",
            "wealth_money",
            "mental_health",
            "social_wellbeing",
            "job_performance",
            "community",
            "energy",
            "move_speed",
            "pronouns",
            "job",
            "health_status",
            "hunger_level",
            "wealth_money",
            "mental_health",
            "job_performance",
            "recent_event",
            "long_term_goal",
            "inventory",
            "personality_traits",
            "motives",
            "skills",
            "career_goals",
            "short_term_goals",
            "uuid",
            "location",
            "needed_items",
            "goals",
            "state",
            "romantic_relationships",
            "romanceable",
            "exclusive_relationship",
            "base_libido",
            "monogamy",
            "investment_portfolio",
            "luxury",
            "material_goods",
            "home",
            "shelter",
            "success",
            "control",
            "hope",
            "happiness",
            "stability",
            "stamina",
            "current_satisfaction",
            "current_mood",
            "coordinates_location",
            "current_activity",
            "beauty",
            "path",
            "destination",
        ]:
            if (
                not getattr(character, attribute)
                and attribute != "exclusive_relationship"
                and attribute != "current_activity"
                and attribute != "path"
                and attribute != "destination"
            ):
                if character.get_char_attribute(attribute) is None:
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}"
                    )
                    exit(1)

            elif attribute in [
                "health_status",
                "hunger_level",
                "mental_health",
                "social_wellbeing",
                "job_performance",
                "community",
                "energy",
                "base_libido",
                "monogamy",
                "luxury",
                "material_goods",
                "stamina",
                "current_satisfaction",
                "current_mood",
                "beauty",
                "stability",
                "hope",
                "happiness",
                "success",
                "control",
                "shelter",
                "physical_beauty",
            ]:
                if not 0 <= getattr(character, attribute) <= 100:
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
            elif attribute in ["wealth_money"]:
                if not 0 <= getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
            elif attribute == "motives":
                if not getattr(character, attribute).to_dict():
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute).to_dict()}"
                    )
                    exit(1)
                for motive in getattr(character, attribute).to_dict().values():
                    if motive.score <= 0 or motive.score > 10:
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute).to_dict()}"
                        )
                        exit(1)
                    motives[motive.name].append({character.name: motive.score})
                    logging.debug(
                        f"Character {character.name} has a valid {attribute}: {getattr(character, attribute).to_dict()} \n Motive: {motive.name} Score: {motive.score} {motives[motive.name]}\n"
                    )
            elif attribute == "personality_traits":
                traits = []
                for trait in [
                    "extraversion",
                    "openness",
                    "conscientiousness",
                    "agreeableness",
                    "neuroticism",
                ]:
                    traits.append(
                        getattr(character, attribute).get_personality_trait(trait)
                    )
                # sort traits
                traits = sort(traits)
                if all(
                    sorted(
                        [
                            value
                            for key, value in getattr(character, attribute)
                            .to_dict()
                            .items()
                        ]
                    )
                    != traits
                ):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {[value for key, value in getattr(character, attribute).to_dict().items()]}, traits: {traits}"
                    )
                    exit(1)
            elif attribute == "goals":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for goal in getattr(character, attribute):
                    if not goal[1]:
                        logging.error(
                            f"Character {character.name} has an invalid {attribute} No goal[1]: {goal}"
                        )
                        exit(1)
                    if not isinstance(goal[1], Goal):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute} goal[1] not Goals type: {goal}"
                        )
                        exit(1)
                    if not isinstance(goal[0], float) or not 0 <= goal[0] <= 100:
                        logging.error(
                            f"Character {character.name} has an invalid {attribute} goal[0] not a float between 0 and 10: {goal}"
                        )
                        exit(1)
            elif attribute == "location":
                if not getattr(character, attribute).get_coordinates():
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute).get_coordinates()}"
                    )
                    exit(1)
            elif attribute == "inventory":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                if not isinstance(getattr(character, attribute), ItemInventory):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for item in getattr(character, attribute).get_all_items():
                    if not isinstance(item, ItemObject):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )
                        exit(1)
            elif attribute == "romantic_relationships":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for relationship in getattr(character, attribute):
                    if not isinstance(relationship, Character):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )
                        exit(1)
            elif attribute == "romanceable":
                if not isinstance(getattr(character, attribute), bool):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
            elif attribute == "exclusive_relationship":
                if getattr(character, attribute) is not None:
                    pass
                    if (
                        not isinstance(getattr(character, attribute), bool)
                        and not isinstance(getattr(character, attribute), Character)
                        and not isinstance(getattr(character, attribute), NoneType)
                    ):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)} of type {type(getattr(character, attribute))}"
                        )
                        exit(1)
            elif attribute == "investment_portfolio":
                if not isinstance(getattr(character, attribute), InvestmentPortfolio):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for item in getattr(character, attribute).get_stocks():
                    if not isinstance(item, Stock):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute} item: {item} of type {type(item)}"
                        )
                        exit(1)
                    if not item.get_stock_value() or item.get_value() < 0:
                        logging.error(
                            f"Character {character.name} has an invalid {attribute} item: {item} of type {type(item)} with value {item.get_stock_value()}"
                        )
                        exit(1)
                if (
                    getattr(character, attribute).get_portfolio_value() is None
                    or getattr(character, attribute).get_portfolio_value() < 0
                ):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}: Collection value is {getattr(character, attribute).get_portfolio_value()}"
                    )
                    exit(1)
            elif attribute == "needed_items":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)

            elif attribute == "skills":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for skill in getattr(character, attribute).get_skills():
                    if not isinstance(skill, Skill):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )
                        exit(1)
            elif attribute == "career_goals":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for goal in getattr(character, attribute):
                    if not isinstance(goal, Goal) or not isinstance(goal, str):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )
                        exit(1)
            elif attribute == "short_term_goals":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                for goal in getattr(character, attribute):
                    if not isinstance(goal, Goal) or not isinstance(goal, str):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )
                        exit(1)

            elif attribute == "coordinates_location":
                if not getattr(character, attribute):
                    logging.error(
                        f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                    )
                    exit(1)
                if not isinstance(getattr(character, attribute), tuple):
                    if (
                        not isinstance(getattr(character, attribute)[0], int)
                        or not isinstance(getattr(character, attribute)[1], int)
                        or not isinstance(getattr(character, attribute)[0], float)
                        or not isinstance(getattr(character, attribute)[1], float)
                    ):
                        logging.error(
                            f"Character {character.name} has an invalid {attribute}: {getattr(character, attribute)}"
                        )

    # find average, max, and std deviation of motives
    outliers = {motive_name: [] for motive_name in motives}
    for motive_name, motive in motives.items():
        logging.debug(f"Calculating statistics for {motive_name}")
        scores = [
            [score for _, score in motive_object.items()] for motive_object in motive
        ]
        scores = [score for sublist in scores for score in sublist]
        logging.debug(f"\n Scores: {scores}")
        average = sum(scores) / len(scores)
        maximum = max(scores)
        minimum = min(scores)
        std_dev = (
            sum([(score - average) ** 2 for score in scores]) / len(scores)
        ) ** 0.5
        logging.debug(f"Average {motive_name} motive: {average}")
        logging.debug(f"Maximum {motive_name} motive: {maximum}")
        logging.debug(f"Minimum {motive_name} motive: {minimum}")
        logging.debug(f"Standard Deviation {motive_name}: {std_dev}")
        for score in scores:
            if score > average + 2 * std_dev or score < average - 2 * std_dev:
                outliers[motive_name].append(score)
    for outlier in outliers:
        if len(outliers[outlier]) > 0:
            logging.debug(
                f"Outliers for {outlier} motives: {outliers[outlier]} with average {sum(outliers[outlier]) / len(outliers[outlier])}"
            )

    # Find the average, max, and std deviation of the health status to compare to the graph manager functions
    health_scores = [character.health_status for character in random_characters]
    average_health = sum(health_scores) / len(health_scores)
    maximum_health = max(health_scores)
    minimum_health = min(health_scores)
    std_dev_health = (
        sum([(score - average_health) ** 2 for score in health_scores])
        / len(health_scores)
    ) ** 0.5

    # Test some graph manager functions
    if not isinstance(
        create_random.graph_manager.get_maximum_attribute_value("health_status"), int
    ) and not isinstance(
        create_random.graph_manager.get_maximum_attribute_value("health_status"), float
    ):
        logging.error(
            f"Graph manager get_maximum_attribute_value is invalid: {create_random.graph_manager.get_maximum_attribute_value('health_status')} of type {type(create_random.graph_manager.get_maximum_attribute_value('health_status'))}"
        )
        exit(1)
    if not isinstance(
        create_random.graph_manager.get_average_attribute_value("health_status"), int
    ) and not isinstance(
        create_random.graph_manager.get_average_attribute_value("health_status"), float
    ):
        logging.error(f"Graph manager get_average_attribute_value is invalid")
    if not isinstance(
        create_random.graph_manager.get_stddev_attribute_value("health_status"), int
    ) and not isinstance(
        create_random.graph_manager.get_stddev_attribute_value("health_status"), float
    ):
        logging.error(f"Graph manager get_stddev_attribute_value is invalid")

    if (
        create_random.graph_manager.get_maximum_attribute_value("health_status")
        != maximum_health
    ):
        logging.info(
            f"Graph manager get_maximum_attribute_value is different: {create_random.graph_manager.get_maximum_attribute_value('health_status')} {maximum_health}"
        )
    if (
        create_random.graph_manager.get_average_attribute_value("health_status")
        != average_health
    ):
        logging.info(
            f"Graph manager get_average_attribute_value is different: {create_random.graph_manager.get_average_attribute_value('health_status')} {average_health}"
        )

    if (
        create_random.graph_manager.get_stddev_attribute_value("health_status")
        != std_dev_health
    ):
        logging.info(
            f"Graph manager get_stddev_attribute_value is different: {create_random.graph_manager.get_stddev_attribute_value('health_status')} {std_dev_health}"
        )

    # Save the characters to a JSON file
    try:
        with open("/home/darf3/llm_game/random_characters.json", "w") as f:
            json.dump(
                [
                    {
                        "name": character.name,
                        "age": character.age,
                        "pronouns": character.pronouns,
                        "job": character.job.job_name,
                        "health_status": character.health_status,
                        "hunger_level": character.hunger_level,
                        "wealth_money": character.wealth_money,
                        "mental_health": character.mental_health,
                        "social_wellbeing": character.social_wellbeing,
                        "job_performance": character.job_performance,
                        "community": character.community,
                        "friendship_grid": character.friendship_grid,
                        "recent_event": character.recent_event,
                        "long_term_goal": character.long_term_goal,
                        "personality_traits": {
                            "extraversion": character.personality_traits.get_extraversion(),
                            "openness": character.personality_traits.get_openness(),
                            "conscientiousness": character.personality_traits.get_conscientiousness(),
                            "agreeableness": character.personality_traits.get_agreeableness(),
                            "neuroticism": character.personality_traits.get_neuroticism(),
                        },
                        "motives": {
                            motive_name: motive_object.score
                            for motive_name, motive_object in character.get_motives()
                            .to_dict()
                            .items()
                        },
                        "goals": {
                            goal_object.name: goal_object.description
                            for _, goal_object in character.goals
                        },
                    }
                    for character in random_characters
                ],
                f,
            )
    except Exception as e:
        logging.error(f"Error saving characters to JSON: {e}")

    with open("/home/darf3/llm_game/training_data_traits_to_motives.json", "w") as f:
        json.dump(
            [
                {
                    "input": {
                        trait: value
                        for trait, value in character.get_personality_traits()
                        .to_dict()
                        .items()
                    },
                    "output": {
                        motive_name: motive_object.score
                        for motive_name, motive_object in character.get_motives()
                        .to_dict()
                        .items()
                    },
                }
                for character in random_characters
            ],
            f,
        )

    with open("/home/darf3/llm_game/training_data_traits_to_motives.csv", "w") as f:
        f.write(
            "extraversion,openness,conscientiousness,agreeableness,neuroticism,libido,wealth,health,hunger,mental_health,social_wellbeing,job_performance,community\n"
        )
        for character in random_characters:
            f.write(
                f"{character.personality_traits.get_extraversion()},{character.personality_traits.get_openness()},{character.personality_traits.get_conscientiousness()},{character.personality_traits.get_agreeableness()},{character.personality_traits.get_neuroticism()},{character.base_libido},{character.wealth_money},{character.health_status},{character.hunger_level},{character.mental_health},{character.social_wellbeing},{character.job_performance},{character.community}\n"
            )
    logging.debug("Done")
