# This file contains the Character class, which is used to represent a character in the game.


from calendar import c
import random
import re
from typing import List
import uuid
from numpy import rint

from torch import eq, rand

import tiny_buildings as tb
from actions import (
    Action,
    ActionGenerator,
    ActionTemplate,
    Skill,
    JobSkill,
    ActionSkill,
    ActionSystem,
    State,
)

from tiny_jobs import JobRoles, JobRules, Job


from tiny_util_funcs import ClampedIntScore, tweener

from tiny_items import ItemInventory, FoodItem, ItemObject

from tiny_graph_manager import GraphManager
from tiny_memories import Memory, MemoryManager
from tiny_time_manager import GameTimeManager

# def gaussian(input_value, mean, std):
#     return (1 / (std * (2 * 3.14159) ** 0.5)) * (2.71828 ** ((-1 / 2) * (((input_value - mean) / std) ** 2)))


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


class Motive:
    def __init__(self, name, description, score):
        self.name = name
        self.description = description
        self.score = score

    def __repr__(self):
        return f"Motive({self.name}, {self.description}, {self.score})"

    def __str__(self):
        return f"Motive named {self.name} with description {self.description} and score {self.score}."

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.score == other.score
        )

    def __hash__(self):
        return hash((self.name, self.description, self.score))

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


class Goal:
    def __init__(self, description, character, score, name, completion_condition):
        # Example name, description, and completion: "get a job": character.get_job().get_job_name() != "unemployed",
        self.name = name
        self.completion_condition = completion_condition
        self.description = description
        self.score = score
        self.character = character

    def __repr__(self):
        return f"Goal({self.name}, {self.description}, {self.score})"

    def __str__(self):
        return f"Goal named {self.name} with description {self.description} and score {self.score}."

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.score == other.score
        )

    def __hash__(self):
        return hash((self.name, self.description, self.score))

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

    def check_completion(self):
        return self.completion_condition


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

    def __repr__(self):
        return f"PersonalMotives({self.hunger_motive}, {self.wealth_motive}, {self.mental_health_motive}, {self.social_wellbeing_motive}, {self.happiness_motive})"

    def __str__(self):
        return f"PersonalMotives with hunger {self.hunger_motive}, wealth {self.wealth_motive}, mental health {self.mental_health_motive}, social wellbeing {self.social_wellbeing_motive}, happiness {self.happiness_motive}."

    def __eq__(self, other):
        return (
            self.hunger_motive == other.hunger
            and self.wealth_motive == other.wealth_motive
            and self.mental_health_motive == other.mental_health_motive
            and self.social_wellbeing_motive == other.social_wellbeing_motive
            and self.happiness_motive == other.happiness_motive
        )

    def __hash__(self):
        return hash(
            (
                self.hunger_motive,
                self.wealth_motive,
                self.mental_health_motive,
                self.social_wellbeing_motive,
                self.happiness_motive,
            )
        )

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

    def get_friendship_grid_motive(self):
        return self.friendship_grid_motive

    def set_friendship_grid_motive(self, friendship_grid_motive):
        self.friendship_grid_motive = friendship_grid_motive
        return self.friendship_grid_motive

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
        }


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
    Quantitative Measures: Represent each personality trait with a numeric value (e.g., 0 to 100) for each character. This allows for nuanced differences between characters and can influence decision-making algorithms.
    Dynamic Interactions: Use personality traits to dynamically influence character interactions. For example, an extraverted character might initiate conversations more frequently, while a highly agreeable character might have more options to support others.
    Influence on Life Choices: Personality can affect career choice, hobbies, and life decisions within the game. For instance, an open and conscientious character might pursue a career in science or exploration.
    Character Development: Allow for personality development over time, influenced by game events, achievements, and relationships. This can add depth to the characters and reflect personal growth or change.
    """

    def __init__(
        self,
        openness: int = 0,
        conscientiousness: int = 0,
        extraversion: int = 0,
        agreeableness: int = 0,
        neuroticism: int = 0,
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

    def __str__(self):
        return f"PersonalityTraits with openness {self.openness}, conscientiousness {self.conscientiousness}, extraversion {self.extraversion}, agreeableness {self.agreeableness}, neuroticism {self.neuroticism}."

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
            (
                self.openness,
                self.conscientiousness,
                self.extraversion,
                self.agreeableness,
                self.neuroticism,
            )
        )

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

    def set_motives(
        self,
        hunger_motive: int = 0,
        wealth_motive: int = 0,
        mental_health_motive: int = 0,
        social_wellbeing_motive: int = 0,
        happiness_motive: int = 0,
        health_motive: int = 0,
        shelter_motive: int = 0,
        stability_motive: int = 0,
        luxury_motive: int = 0,
        hope_motive: int = 0,
        success_motive: int = 0,
        control_motive: int = 0,
        job_performance_motive: int = 0,
        beauty_motive: int = 0,
        community_motive: int = 0,
        material_goods_motive: int = 0,
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
        )


class CharacterSkills:
    def __init__(self, skills: List[Skill]):
        self.action_skills = []
        self.job_skills = []
        self.other_skills = []
        self.set_skills(skills)

    def __repr__(self):
        return f"CharacterSkills({self.skills})"

    def __str__(self):
        return f"CharacterSkills with skills {self.skills}."

    def __eq__(self, other):
        return self.skills == other.skills

    def __hash__(self):
        return hash(self.skills)

    def get_skills(self):
        return self.skills

    def set_skills(self, skills):
        for skill in skills:
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
            "satisfy_value": 10,
            "operator": "gt",
        },
        {
            "name": "extraversion",
            "attribute": "personality_traits.extraversion",
            "satisfy_value": 50,
            "operator": "gt",
        },
    ],
    "Trade": [
        {
            "name": "wealth_money",
            "attribute": "wealth_money",
            "satisfy_value": 5,
            "operator": "gt",
        },
        {
            "name": "conscientiousness",
            "attribute": "personality_traits.conscientiousness",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Help": [
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
            "satisfy_value": 20,
            "operator": "gt",
        },
        {
            "name": "agreeableness",
            "attribute": "personality_traits.agreeableness",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Attack": [
        {
            "name": "anger",
            "attribute": "current_mood",
            "satisfy_value": -10,
            "operator": "lt",
        },
        {
            "name": "strength",
            "attribute": "skills.strength",
            "satisfy_value": 20,
            "operator": "gt",
        },
    ],
    "Befriend": [
        {
            "name": "openness",
            "attribute": "personality_traits.openness",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Teach": [
        {
            "name": "knowledge",
            "attribute": "skills.knowledge",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "patience",
            "attribute": "personality_traits.agreeableness",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Learn": [
        {
            "name": "curiosity",
            "attribute": "personality_traits.openness",
            "satisfy_value": 50,
            "operator": "gt",
        },
        {
            "name": "focus",
            "attribute": "mental_health",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Heal": [
        {
            "name": "medical_knowledge",
            "attribute": "skills.medical_knowledge",
            "satisfy_value": 40,
            "operator": "gt",
        },
        {
            "name": "compassion",
            "attribute": "personality_traits.agreeableness",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Gather": [
        {
            "name": "energy",
            "attribute": "energy",
            "satisfy_value": 20,
            "operator": "gt",
        },
        {
            "name": "curiosity",
            "attribute": "personality_traits.openness",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Build": [
        {
            "name": "construction_skill",
            "attribute": "skills.construction",
            "satisfy_value": 30,
            "operator": "gt",
        },
        {
            "name": "conscientiousness",
            "attribute": "personality_traits.conscientiousness",
            "satisfy_value": 40,
            "operator": "gt",
        },
    ],
    "Give Item": [
        {
            "name": "item_in_inventory",
            "attribute": "inventory.item_count",
            "satisfy_value": 1,
            "operator": "gt",
        },
        {
            "name": "generosity",
            "attribute": "personality_traits.agreeableness",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Receive Item": [
        {
            "name": "need",
            "attribute": "hunger_level",
            "satisfy_value": 5,
            "operator": "gt",
        },
        {
            "name": "social_wellbeing",
            "attribute": "social_wellbeing",
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
        home: tb.House = None,
        inventory: ItemInventory = None,
        motives: PersonalMotives = None,
        personality_traits: PersonalityTraits = None,
        career_goals: List[str] = [],
        possible_interactions: List[Action] = [],
    ):

        self.name = self.set_name(name)
        self.age = self.set_age(age)
        self.character_actions = [
            Action(
                "Talk",
                action_system.instantiate_condition(preconditions_dict["Talk"]),
                effects=effect_dict["Talk"],
                cost=1,
            ),
            Action(
                "Trade",
                action_system.instantiate_condition(preconditions_dict["Trade"]),
                effects=effect_dict["Trade"],
                cost=2,
            ),
            Action(
                "Help",
                action_system.instantiate_condition(preconditions_dict["Help"]),
                effects=effect_dict["Help"],
                cost=1,
            ),
            Action(
                "Attack",
                action_system.instantiate_condition(preconditions_dict["Attack"]),
                effects=effect_dict["Attack"],
                cost=3,
            ),
            Action(
                "Befriend",
                action_system.instantiate_condition(preconditions_dict["Befriend"]),
                effects=effect_dict["Befriend"],
                cost=1,
            ),
            Action(
                "Teach",
                action_system.instantiate_condition(preconditions_dict["Teach"]),
                effects=effect_dict["Teach"],
                cost=2,
            ),
            Action(
                "Learn",
                action_system.instantiate_condition(preconditions_dict["Learn"]),
                effects=effect_dict["Learn"],
                cost=1,
            ),
            Action(
                "Heal",
                action_system.instantiate_condition(preconditions_dict["Heal"]),
                effects=effect_dict["Heal"],
                cost=2,
            ),
            Action(
                "Gather",
                action_system.instantiate_condition(preconditions_dict["Gather"]),
                effects=effect_dict["Gather"],
                cost=1,
            ),
            Action(
                "Build",
                action_system.instantiate_condition(preconditions_dict["Build"]),
                effects=effect_dict["Build"],
                cost=2,
            ),
            Action(
                "Give Item",
                action_system.instantiate_condition(preconditions_dict["Give Item"]),
                effects=effect_dict["Give Item"],
                cost=1,
            ),
            Action(
                "Receive Item",
                action_system.instantiate_condition(preconditions_dict["Receive Item"]),
                effects=effect_dict["Receive Item"],
                cost=1,
            ),
        ]
        self.possible_interactions = possible_interactions + self.character_actions

        self.pronouns = self.set_pronouns(pronouns)
        self.friendship_grid = friendship_grid if friendship_grid else [{}]
        self.job = self.set_job(job)
        self.health_status = self.set_health_status(health_status)
        self.hunger_level = self.set_hunger_level(hunger_level)
        self.wealth_money = self.set_wealth_money(wealth_money)
        self.mental_health = self.set_mental_health(mental_health)
        self.social_wellbeing = self.set_social_wellbeing(social_wellbeing)

        self.luxury = 0  # fluctuates with environment

        self.job_performance = self.set_job_performance(job_performance)
        self.beauty = 0  # fluctuates with environment
        self.community = self.set_community(community)

        self.friendship_grid = self.set_friendship_grid(friendship_grid)
        self.recent_event = self.set_recent_event(recent_event)
        self.long_term_goal = self.set_long_term_goal(long_term_goal)
        self.inventory = self.set_inventory(inventory)
        self.material_goods = self.set_material_goods(self.calculate_material_goods())
        if home:
            self.home = self.set_home(home)
        else:
            self.home = self.set_home(tb.CreateBuilding().generate_random_house())
        self.shelter = self.set_shelter(self.home.calculate_shelter_value())
        self.success = self.set_success(self.calculate_success())
        self.control = self.set_control(self.calculate_control())
        self.hope = self.set_hope(self.calculate_hope())
        self.happiness = self.set_happiness(self.calculate_happiness())
        self.stability = self.set_stability(self.calculate_stability())
        self.personality_traits = self.set_personality_traits(personality_traits)
        self.motives = self.set_motives(motives)
        self.stamina = 0
        self.current_satisfaction = 0
        self.current_mood = 0
        self.skills = CharacterSkills([])
        self.career_goals = career_goals
        self.short_term_goals = []
        self.id = uuid.uuid4()
        self.memory_manager = MemoryManager(
            gametime_manager
        )  # Initialize MemoryManager with GameTimeManager instance

    def create_memory(self, description, timestamp, importance):
        memory = Memory(
            description, timestamp, importance
        )  # Assuming Memory class exists and is appropriately structured
        self.memory_manager.add_memory(memory)

    def recall_recent_memories(self):
        return self.memory_manager.flat_access.get_recent_memories()

    def make_decision_based_on_memories(self):
        # Example of how to use memories in decision-making
        important_memories = self.recall_recent_memories()
        # Decision logic here, possibly using the contents of important_memories
        pass

    def __repr__(self):
        return f"Character({self.name}, {self.age}, {self.pronouns}, {self.job}, {self.health_status}, {self.hunger_level}, {self.wealth_money}, {self.mental_health}, {self.social_wellbeing}, {self.job_performance}, {self.community}, {self.friendship_grid}, {self.recent_event}, {self.long_term_goal}, {self.home}, {self.inventory}, {self.motives}, {self.personality_traits})"

    def __str__(self):
        return f"Character named {self.name}, age {self.age}, pronouns {self.pronouns}, job {self.job}, health status {self.health_status}, hunger level {self.hunger_level}, wealth {self.wealth_money}, mental health {self.mental_health}, social wellbeing {self.social_wellbeing}, job performance {self.job_performance}, community {self.community}, friendship grid {self.friendship_grid}, recent event {self.recent_event}, long term goal {self.long_term_goal}, home {self.home}, inventory {self.inventory}, motives {self.motives}, personality traits {self.personality_traits}."

    def __eq__(self, other):
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
            and self.friendship_grid == other.friendship_grid
            and self.recent_event == other.recent_event
            and self.long_term_goal == other.long_term_goal
            and self.home == other.home
            and self.inventory == other.inventory
            and self.motives == other.motives
            and self.personality_traits == other.personality_traits
        )

    def get_possible_interactions(self):
        return self.possible_interactions

    def play_animation(self, animation):
        print(f"{self.name} is playing animation: {animation}")

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

    def define_descriptors(self):
        self.job
        return self.job

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

    def set_job(self, job):
        job_rules = JobRules()
        if isinstance(job, JobRoles):
            self.job = job
        elif isinstance(job, str):
            if job_rules.check_job_name_validity(job):
                for job_role in job_rules.ValidJobRoles:
                    if job_role.get_job_name() == job:
                        self.job = job_role
            else:
                self.job = random.choice(job_rules.ValidJobRoles)
        else:
            self.job = random.choice(job_rules.ValidJobRoles)
        return self.job

    def set_home(self, home):
        if isinstance(home, tb.House):
            self.home = home
        elif isinstance(home, str):
            self.home = tb.CreateBuilding().create_house_by_type(home)
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
        pass

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

    def calculate_material_goods(self):
        material_goods = round(
            tweener(self.inventory.count_total_items(), 1000, 0, 100, 2)
        )  # Tweening the wealth value
        return material_goods

    def calculate_stability(self):
        stability = 0
        stability += self.get_shelter()
        stability += round(
            tweener(self.get_luxury(), 100, 0, 10, 2)
        )  # Tweening the luxury value
        stability += self.get_hope()
        stability += round(
            tweener(self.get_success(), 100, 0, 10, 2)
        )  # Tweening the success value
        stability += round(
            tweener(self.get_control(), 100, 0, 10, 2)
        )  # Tweening the control value
        stability += round(
            tweener(self.get_beauty(), 100, 0, 10, 2)
        )  # Tweening the job performance value
        stability += self.get_community()
        stability += round(
            tweener(self.get_material_goods(), 100, 0, 10, 2)
        )  # Tweening the material goods value
        stability += self.get_social_wellbeing()
        return stability

    def calculate_happiness(self):
        happiness = 0
        happiness += self.get_hope()
        happiness += round(tweener(self.get_success(), 100, 0, 10, 2))
        happiness += round(tweener(self.get_control(), 100, 0, 10, 2))
        happiness += round(tweener(self.get_beauty(), 100, 0, 10, 2))
        happiness += self.get_community()
        happiness += round(tweener(self.get_material_goods(), 100, 0, 10, 2))
        happiness += self.get_social_wellbeing()
        return happiness

    def calculate_success(self):
        success = 0
        success += round(
            tweener(self.get_job_performance(), 100, 0, 50, 2)
        )  # Tweening the job performance value
        success += round(
            tweener(self.get_material_goods(), 100, 0, 20, 2)
        )  # Tweening the material goods value
        success += round(
            tweener(self.get_wealth_money(), 1000, 0, 20, 2)
        )  # Tweening the wealth value
        return success

    def calculate_control(self):
        control = 0
        control += self.get_shelter()
        control += round(tweener(self.get_success(), 100, 0, 10, 2))
        control += round(
            tweener(self.get_material_goods(), 100, 0, 20, 2)
        )  # Tweening the material goods value
        control += round(
            tweener(self.get_wealth_money(), 1000, 0, 20, 2)
        )  # Tweening the wealth value
        return control

    def calculate_hope(self):
        hope = 0
        hope += round(
            tweener(self.get_beauty(), 100, 0, 10, 2)
        )  # Tweening the luxury value
        hope += round(
            tweener(self.get_success(), 100, 0, 10, 2)
        )  # Tweening the success value
        hope += self.get_community()
        hope += round(
            tweener(self.get_material_goods(), 100, 0, 10, 2)
        )  # Tweening the material goods value
        hope += self.get_social_wellbeing()
        return hope

    def __repr__(self):
        return f"Character({self.name}, {self.job}, {self.health_status}, {self.hunger_level}, {self.wealth_money}, {self.long_term_goal}, {self.recent_event})"

    def __str__(self):
        return f"Character named {self.name} with job {self.job} and health status {self.health_status}."

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.job == other.job
            and self.health_status == other.health_status
            and self.hunger_level == other.hunger_level
            and self.wealth_money == other.wealth_money
            and self.long_term_goal == other.long_term_goal
            and self.recent_event == other.recent_event
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                self.job,
                self.health_status,
                self.hunger_level,
                self.wealth_money,
                self.long_term_goal,
                self.recent_event,
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

    def calculate_motives(self):
        social_wellbeing_motive = abs(
            self.personality_traits.get_openness()
            + (self.personality_traits.get_extraversion() * 2)
            + self.personality_traits.get_agreeableness()
            - self.personality_traits.get_neuroticism()
        )
        beauty_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            + social_wellbeing_motive
        )
        hunger_motive = abs(
            (10 - self.get_mental_health())
            + self.personality_traits.get_neuroticism()
            - self.personality_traits.get_conscientiousness()
            - beauty_motive
        )
        community_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            + social_wellbeing_motive
        )
        health_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            - hunger_motive
            + beauty_motive
            + self.personality_traits.get_conscientiousness()
        )
        mental_health_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            - hunger_motive
            + beauty_motive
            + self.personality_traits.get_conscientiousness()
            + health_motive
        )
        stability_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            + health_motive
            + community_motive
        )
        shelter_motive = abs(
            self.personality_traits.get_neuroticism()
            + self.personality_traits.get_conscientiousness()
            + health_motive
            + community_motive
            + beauty_motive
            + stability_motive
        )
        control_motive = abs(
            self.personality_traits.get_conscientiousness()
            + self.personality_traits.get_neuroticism()
            + shelter_motive
            + stability_motive
        )
        success_motive = abs(
            self.personality_traits.get_conscientiousness()
            + self.personality_traits.get_neuroticism()
            + shelter_motive
            + stability_motive
            + control_motive
        )
        material_goods_motive = abs(
            round(
                random.gauss(
                    abs(
                        (
                            shelter_motive
                            + stability_motive
                            + success_motive
                            + control_motive
                        )
                        / 1000
                    ),
                    (
                        self.personality_traits.get_conscientiousness()
                        + self.personality_traits.get_neuroticism()
                    )
                    * 10,
                )
            )
        )
        luxury_motive = abs(
            self.personality_traits.get_openness()
            + self.personality_traits.get_extraversion()
            + self.personality_traits.get_agreeableness()
            + self.personality_traits.get_neuroticism()
            + material_goods_motive
            + beauty_motive
        )
        wealth_motive = abs(
            round(
                random.gauss(
                    abs(
                        (
                            luxury_motive
                            + shelter_motive
                            + stability_motive
                            + success_motive
                            + control_motive
                            + material_goods_motive
                            + luxury_motive
                        )
                        / 1000
                    ),
                    (
                        self.personality_traits.get_conscientiousness()
                        + self.personality_traits.get_neuroticism()
                    )
                    * 10,
                )
            )
        )
        job_performance_motive = abs(
            round(
                random.gauss(
                    abs(
                        ((success_motive + material_goods_motive + wealth_motive) * 2)
                        / 1000
                    ),
                    (
                        self.personality_traits.get_conscientiousness()
                        + self.personality_traits.get_extraversion()
                        + self.personality_traits.get_agreeableness()
                        + self.personality_traits.get_neuroticism()
                    )
                    * 10,
                )
            )
        )
        happiness_motive = abs(
            round(
                random.gauss(
                    abs(
                        (
                            success_motive
                            + material_goods_motive
                            + wealth_motive
                            + job_performance_motive
                            + social_wellbeing_motive
                        )
                        / 1000
                    ),
                    (
                        self.personality_traits.get_openness()
                        + self.personality_traits.get_extraversion()
                        + self.personality_traits.get_agreeableness()
                        - self.personality_traits.get_neuroticism()
                    )
                    * 10,
                )
            )
        )
        hope_motive = abs(
            round(
                random.gauss(
                    abs(
                        (
                            mental_health_motive
                            + social_wellbeing_motive
                            + happiness_motive
                            + health_motive
                            + shelter_motive
                            + stability_motive
                            + luxury_motive
                            + success_motive
                            + control_motive
                            + job_performance_motive
                            + beauty_motive
                            + community_motive
                        )
                        / 1000
                    ),
                    (
                        self.personality_traits.get_openness()
                        + self.personality_traits.get_extraversion()
                        + self.personality_traits.get_agreeableness()
                        + self.personality_traits.get_neuroticism()
                    )
                    * 10,
                )
            )
        )
        return PersonalMotives(
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
        )

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
        }

    def get_state(self):
        return State(self.to_dict())


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

    def __str__(self):
        return f"CreateCharacter class."

    def create_new_character(
        self,
        mode: str = "auto",
        name: str = "John Doe",
        age: int = 18,
        pronouns: str = "they/them",
        job: str = "unemployed",
        health_status: int = 0,
        hunger_level: int = 0,
        wealth_money: int = 0,
        mental_health: int = 0,
        social_wellbeing: int = 0,
        job_performance: int = 0,
        community: int = 0,
        friendship_grid: dict = {},
        recent_event: str = "",
        long_term_goal: str = "",
        inventory: ItemInventory = None,
        personality_traits: PersonalityTraits = None,
        motives: PersonalMotives = None,
        home: str = "",
    ):
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
                        "How well does your character perform at their job? A number between 0 and 100"
                    )
                )
            if home == "":
                home = tb.CreateBuilding().create_house_by_type(
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
                home = tb.CreateBuilding().generate_random_house()

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
