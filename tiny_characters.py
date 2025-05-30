# This file contains the Character class, which is used to represent a character in the game.

from ast import arg
from calendar import c
import heapq

import importlib
from math import e
import random
import re
from typing import List
import uuid
import attr 
from numpy import rint
from tiny_types import PromptBuilder 
from pyparsing import Char
from sympy import im
from torch import Graph, eq, rand
import logging

logging.basicConfig(level=logging.DEBUG)
from tiny_types import House, CreateBuilding, Action as ActionType, State as StateType, ActionSystem 

from actions import (
    Action, 
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

from tiny_time_manager import GameTimeManager
from tiny_locations import Location, LocationManager


# --- Helper/Dependent Class Definitions (Moved to top) ---

class SimpleSocialGoal:
    def __init__(self, name, description, completion_conditions, priority, reward, target_character_id=None):
        self.name = name
        self.description = description
        self.completion_conditions = completion_conditions 
        self.priority = priority 
        self.reward = reward 
        self.target_character_id = target_character_id 
        self.achieved = False
        self.active = True 

    def __str__(self):
        return f"SimpleSocialGoal: {self.name} (Target: {self.target_character_id}, Priority: {self.priority}, Active: {self.active}, Achieved: {self.achieved})"

    def __repr__(self):
        return f"SimpleSocialGoal(name='{self.name}', target='{self.target_character_id}', priority={self.priority}, conditions={self.completion_conditions})"

    def set_achieved(self, status=True):
        self.achieved = status
        if status:
            self.active = False
        else: # If un-achieving the goal
            self.active = True # Reactivate it

    def set_active(self, status=True):
        self.active = status

class Motive:
    def __init__(self, name: str, description: str, score: float):
        self.name = name
        self.description = description
        self.score = score 
    def __repr__(self): return f"Motive({self.name}, {self.description}, {self.score})"
    def __eq__(self, other):
        if not isinstance(other, Motive):
            if isinstance(other, dict): return (self.name == other["name"] and self.description == other["description"] and self.score == other["score"])
            elif isinstance(other, tuple): return (self.name == other[0] and self.description == other[1] and self.score == other[2])
            elif isinstance(other, float) or isinstance(other, int): return self.score == float(other)
            return False
        return (self.name == other.name and self.description == other.description and self.score == other.score)
    def __hash__(self):
        def make_hashable(obj):
            if isinstance(obj, dict): return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list): return tuple(make_hashable(e) for e in obj)
            elif isinstance(obj, set): return frozenset(make_hashable(e) for e in obj)
            elif isinstance(obj, tuple): return tuple(make_hashable(e) for e in obj)
            return obj
        return hash(make_hashable(self.to_dict()))
    def get_name(self): return self.name
    def set_name(self, name): self.name = name; return self.name
    def get_description(self): return self.description
    def set_description(self, description): self.description = description; return self.description
    def get_score(self): return self.score
    def set_score(self, score): self.score = score; return self.score
    def to_dict(self): return {"name": self.name, "description": self.description, "score": self.score}

class PersonalMotives:
    def __init__(self, hunger_motive: Motive, wealth_motive: Motive, mental_health_motive: Motive, social_wellbeing_motive: Motive, happiness_motive: Motive, health_motive: Motive, shelter_motive: Motive, stability_motive: Motive, luxury_motive: Motive, hope_motive: Motive, success_motive: Motive, control_motive: Motive, job_performance_motive: Motive, beauty_motive: Motive, community_motive: Motive, material_goods_motive: Motive, family_motive: Motive):
        self.hunger_motive = self.set_hunger_motive(hunger_motive)
        self.wealth_motive = self.set_wealth_motive(wealth_motive)
        self.mental_health_motive = self.set_mental_health_motive(mental_health_motive); self.social_wellbeing_motive = self.set_social_wellbeing_motive(social_wellbeing_motive); self.happiness_motive = self.set_happiness_motive(happiness_motive); self.health_motive = self.set_health_motive(health_motive); self.shelter_motive = self.set_shelter_motive(shelter_motive); self.stability_motive = self.set_stability_motive(stability_motive); self.luxury_motive = self.set_luxury_motive(luxury_motive); self.hope_motive = self.set_hope_motive(hope_motive); self.success_motive = self.set_success_motive(success_motive); self.control_motive = self.set_control_motive(control_motive); self.job_performance_motive = self.set_job_performance_motive(job_performance_motive); self.beauty_motive = self.set_beauty_motive(beauty_motive); self.community_motive = self.set_community_motive(community_motive); self.material_goods_motive = self.set_material_goods_motive(material_goods_motive); self.family_motive = self.set_family_motive(family_motive)
        self._attributes = ["hunger_motive", "wealth_motive", "mental_health_motive", "social_wellbeing_motive", "happiness_motive", "health_motive", "shelter_motive", "stability_motive", "luxury_motive", "hope_motive", "success_motive", "control_motive", "job_performance_motive", "beauty_motive", "community_motive", "material_goods_motive", "family_motive"]
        self._index = 0
    def __repr__(self): return f"PersonalMotives(...)" 
    def __eq__(self, other): 
        if not isinstance(other, PersonalMotives): return False
        return self.to_dict() == other.to_dict()
    def __hash__(self): 
        return hash(tuple(sorted(self.to_dict().items())))
    def __iter__(self): self._index = 0; return self
    def __next__(self):
        if self._index < len(self._attributes):
            attr_name = self._attributes[self._index]; self._index += 1; return getattr(self, attr_name)
        else: raise StopIteration
    def set_family_motive(self, family_motive): self.family_motive = family_motive; return self.family_motive 
    def get_family_motive(self): return self.family_motive
    def get_hunger_motive(self): return self.hunger_motive
    def set_hunger_motive(self, hunger_motive): self.hunger_motive = hunger_motive; return self.hunger_motive
    def get_wealth_motive(self): return self.wealth_motive
    def set_wealth_motive(self, wealth_motive): self.wealth_motive = wealth_motive; return self.wealth_motive
    def get_mental_health_motive(self): return self.mental_health_motive
    def set_mental_health_motive(self, mental_health_motive): self.mental_health_motive = mental_health_motive; return self.mental_health_motive
    def get_social_wellbeing_motive(self): return self.social_wellbeing_motive
    def set_social_wellbeing_motive(self, social_wellbeing): self.social_wellbeing_motive = social_wellbeing; return self.social_wellbeing_motive
    def get_happiness_motive(self): return self.happiness_motive
    def set_happiness_motive(self, happiness_motive): self.happiness_motive = happiness_motive; return self.happiness_motive
    def get_health_motive(self): return self.health_motive
    def set_health_motive(self, health_motive): self.health_motive = health_motive; return self.health_motive
    def get_shelter_motive(self): return self.shelter_motive
    def set_shelter_motive(self, shelter_motive): self.shelter_motive = shelter_motive; return self.shelter_motive
    def get_stability_motive(self): return self.stability_motive
    def set_stability_motive(self, stability_motive): self.stability_motive = stability_motive; return self.stability_motive
    def get_luxury_motive(self): return self.luxury_motive
    def set_luxury_motive(self, luxury_motive): self.luxury_motive = luxury_motive; return self.luxury_motive
    def get_hope_motive(self): return self.hope_motive
    def set_hope_motive(self, hope_motive): self.hope_motive = hope_motive; return self.hope_motive
    def get_success_motive(self): return self.success_motive
    def set_success_motive(self, success_motive): self.success_motive = success_motive; return self.success_motive
    def get_control_motive(self): return self.control_motive
    def set_control_motive(self, control_motive): self.control_motive = control_motive; return self.control_motive
    def get_job_performance_motive(self): return self.job_performance_motive
    def set_job_performance_motive(self, job_performance_motive): self.job_performance_motive = job_performance_motive; return self.job_performance_motive
    def get_beauty_motive(self): return self.beauty_motive
    def set_beauty_motive(self, beauty_motive): self.beauty_motive = beauty_motive; return self.beauty_motive
    def get_community_motive(self): return self.community_motive
    def set_community_motive(self, community_motive): self.community_motive = community_motive; return self.community_motive
    def get_material_goods_motive(self): return self.material_goods_motive
    def set_material_goods_motive(self, material_goods_motive): self.material_goods_motive = material_goods_motive; return self.material_goods_motive
    def to_dict(self):
        return {attr.replace("_motive",""): getattr(self, attr) for attr in self._attributes}

class PersonalityTraits:
    def __init__( self, openness: float = 0.0, conscientiousness: float = 0.0, extraversion: float = 0.0, agreeableness: float = 0.0, neuroticism: float = 0.0 ):
        self.openness = ClampedIntScore().clamp_score(openness)
        self.conscientiousness = ClampedIntScore().clamp_score(conscientiousness)
        self.extraversion = ClampedIntScore().clamp_score(extraversion)
        self.agreeableness = ClampedIntScore().clamp_score(agreeableness)
        self.neuroticism = ClampedIntScore().clamp_score(neuroticism)
        self.motives = None # This will be set by set_motives
    def __repr__(self): return f"PersonalityTraits({self.openness}, {self.conscientiousness}, {self.extraversion}, {self.agreeableness}, {self.neuroticism})"
    def __eq__(self, other):
        if not isinstance(other, PersonalityTraits): return False
        return (self.openness == other.openness and self.conscientiousness == other.conscientiousness and self.extraversion == other.extraversion and self.agreeableness == other.agreeableness and self.neuroticism == other.neuroticism)
    def __hash__(self): return hash(tuple([self.openness, self.conscientiousness, self.extraversion, self.agreeableness, self.neuroticism]))
    def to_dict(self): return {"openness": self.openness, "conscientiousness": self.conscientiousness, "extraversion": self.extraversion, "agreeableness": self.agreeableness, "neuroticism": self.neuroticism}
    def get_openness(self): return self.openness
    def set_openness(self, openness): self.openness = openness; return self.openness
    def get_conscientiousness(self): return self.conscientiousness
    def set_conscientiousness(self, conscientiousness): self.conscientiousness = conscientiousness; return self.conscientiousness
    def get_extraversion(self): return self.extraversion
    def set_extraversion(self, extraversion): self.extraversion = extraversion; return self.extraversion
    def get_agreeableness(self): return self.agreeableness
    def set_agreeableness(self, agreeableness): self.agreeableness = agreeableness; return self.agreeableness
    def get_neuroticism(self): return self.neuroticism
    def set_neuroticism(self, neuroticism): self.neuroticism = neuroticism; return self.neuroticism
    def get_motives(self): return self.motives
    def get_personality_trait(self, trait): return getattr(self, trait, None)
    def set_motives(self, hunger_motive: float = 0.0, wealth_motive: float = 0.0, mental_health_motive: float = 0.0, social_wellbeing_motive: float = 0.0, happiness_motive: float = 0.0, health_motive: float = 0.0, shelter_motive: float = 0.0, stability_motive: float = 0.0, luxury_motive: float = 0.0, hope_motive: float = 0.0, success_motive: float = 0.0, control_motive: float = 0.0, job_performance_motive: float = 0.0, beauty_motive: float = 0.0, community_motive: float = 0.0, material_goods_motive: float = 0.0, family_motive: float = 0.0 ):
        self.motives = PersonalMotives(
            hunger_motive=Motive("hunger", "bias toward satisfying hunger", hunger_motive),
            wealth_motive=Motive("wealth", "bias toward accumulating wealth", wealth_motive),
            mental_health_motive=Motive("mental health", "bias toward maintaining mental health", mental_health_motive),
            social_wellbeing_motive=Motive("social wellbeing", "bias toward maintaining social wellbeing", social_wellbeing_motive),
            happiness_motive=Motive("happiness", "bias toward maintaining happiness", happiness_motive),
            health_motive=Motive("health", "bias toward maintaining health", health_motive),
            shelter_motive=Motive("shelter", "bias toward maintaining shelter", shelter_motive),
            stability_motive=Motive("stability", "bias toward maintaining stability", stability_motive),
            luxury_motive=Motive("luxury", "bias toward maintaining luxury", luxury_motive),
            hope_motive=Motive("hope", "bias toward maintaining hope", hope_motive),
            success_motive=Motive("success", "bias toward maintaining success", success_motive),
            control_motive=Motive("control", "bias toward maintaining control", control_motive),
            job_performance_motive=Motive("job performance", "bias toward maintaining job performance", job_performance_motive),
            beauty_motive=Motive("beauty", "bias toward maintaining beauty", beauty_motive),
            community_motive=Motive("community", "bias toward maintaining community", community_motive),
            material_goods_motive=Motive("material goods", "bias toward maintaining material goods", material_goods_motive),
            family_motive=Motive("family", "bias toward maintaining family", family_motive))

class CharacterSkills:
    def __init__(self, skills: List[Skill]): # Skill is from actions.py
        self.action_skills = []
        self.job_skills = []
        self.other_skills = []
        self.skills = []
        self.set_skills(skills)
    def __repr__(self): return f"CharacterSkills({self.skills}, {self.job_skills}, {self.action_skills}, {self.other_skills})"
    def __eq__(self, other):
        if not isinstance(other, CharacterSkills): return False
        return (self.skills == other.skills and self.job_skills == other.job_skills and self.action_skills == other.action_skills and self.other_skills == other.other_skills)
    def __hash__(self): # Simplified
        return hash((tuple(self.skills), tuple(self.job_skills), tuple(self.action_skills), tuple(self.other_skills)))
    def get_skills(self): return self.skills
    def get_skills_as_list_of_strings(self): return [skill.name for skill in self.skills]
    def set_skills(self, skills):
        for skill in skills:
            self.skills.append(skill)
            if isinstance(skill, JobSkill): self.job_skills.append(skill)
            elif isinstance(skill, ActionSkill): self.action_skills.append(skill)
            else: self.other_skills.append(skill)
    def add_skill(self, skill): self.skills.append(skill); return self.skills

class Goal: # This is the existing complex Goal class
    def __init__(
        self,
        description,
        character, 
        target,  
        score, 
        name,
        completion_conditions, 
        evaluate_utility_function, 
        difficulty,  
        completion_reward, 
        failure_penalty, 
        completion_message, 
        failure_message, 
        criteria, 
        graph_manager, 
        goal_type
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
        self.required_items = ( self.extract_required_items() ) 
        self.target = target
        self.graph_manager_ref = graph_manager 
        self.goal_type = goal_type
        self.achieved = False 
        self.active = True    

    def extract_required_items(self):
        required_items = []
        for criterion in self.criteria:
            if "node_attributes" in criterion:
                if ("item_type" in criterion["node_attributes"] or criterion["node_attributes"]["type"] == "item"):
                    required_items.append(criterion["node_attributes"])
        if self.completion_conditions: # Check if not None
            for condition_list in self.completion_conditions.values():
                for condition in condition_list:
                    if "inventory.check_has_item_by_type" in condition.attribute:
                        args = re.search(r"\[.*\]", condition.attribute).group().strip("[]")
                        args = [arg.strip("'") for arg in args.split(",")]
                        if args[0] not in [item["item_type"] for item in required_items if isinstance(item, dict) and "item_type" in item]: 
                            required_items.append(({"item_type": args[0]}, condition.satisfy_value))
        return required_items

    def __repr__(self):
        char_name = self.character.name if hasattr(self.character, 'name') else str(self.character)
        target_name = self.target.name if hasattr(self.target, 'name') else str(self.target)
        return f"Goal(Name: {self.name}, Character: {char_name}, Target: {target_name}, Score: {self.score}, Type: {self.goal_type})"

    def to_dict(self): 
        return {"name": self.name, "description": self.description, "score": self.score, "goal_type": self.goal_type}
        
    def __eq__(self, other):
        if not isinstance(other, Goal): return False
        return self.name == other.name and self.description == other.description 
        
    def __hash__(self): 
        return hash((self.name, self.description, self.score, self.goal_type))

    def check_completion(self): 
        if not self.completion_conditions: return False
        try:
            for condition_status, conditions in self.completion_conditions.items():
                if not condition_status: 
                    if not all(cond.check_condition() for cond in conditions):
                        return False
            return True 
        except Exception as e:
            return False 

    def set_achieved(self, status=True): 
        self.achieved = status
        if status: self.active = False
        else: self.active = True # Reactivate if un-achieved

    def set_active(self, status=True): 
        self.active = status
# --- End of Helper/Dependent Class Definitions ---

def heuristic(a, b): # Already defined above, ensure this isn't a duplicate
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(graph, start, goal): # Already defined above
    # ... (implementation)
    pass

class SteeringBehaviors: # Already defined above
    # ... (implementation)
    pass

class RandomNameGenerator: # Already defined above
    # ... (implementation)
    pass


# --- Character Class ---
class Character:
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
        possible_interactions: List[ActionType] = [], 
        move_speed: int = 1,
        graph_manager = None, 
        action_system: ActionSystem = None, 
        gametime_manager: GameTimeManager = None,
        location: Location = None,
        energy: int = 10,
        romanceable: bool = True,
        physical_appearance: str = "",
        physical_beauty: int = random.randint(0, 100),
        skills: CharacterSkills = None 
    ):
        if graph_manager is not None:
            if isinstance(graph_manager, type): 
                 pass 
            self.graph_manager = graph_manager
        else:
            try:
                GraphManager_module = importlib.import_module("tiny_graph_manager")
                self.graph_manager = GraphManager_module.GraphManager()
            except ModuleNotFoundError as e:
                print(f"CRITICAL WARNING: Failed to import or instantiate GraphManager due to: {e}. Character will have limited functionality.")
                self.graph_manager = None 
            except Exception as e:
                print(f"CRITICAL WARNING: An unexpected error occurred with GraphManager: {e}. Character will have limited functionality.")
                self.graph_manager = None

        self._updating = False 
        self._initialized = False 
        
        self.move_speed = move_speed
        if not action_system:
            action_system = ActionSystem() 
        self.action_system = action_system
        self.name = name 
        self.age = age   
        self.destination = None
        self.path = []
        self.velocity = (0,0) 
        self.speed = 1.0  
        self.energy = energy
        self.character_actions = [] 
        self.possible_interactions = possible_interactions + self.character_actions
        
        try:
            self.goap_planner = GOAPPlanner(self.graph_manager)
        except Exception as e:
            print(f"Warning: Failed to initialize GOAPPlanner for {self.name}: {e}")
            self.goap_planner = None

        from tiny_prompt_builder import PromptBuilder 
        self.prompt_builder = PromptBuilder(self)
        self.pronouns = pronouns 
        self.friendship_grid = friendship_grid if friendship_grid else [{}]
        self.job = None
        self.job = self.set_job(job) 
        self.health_status = health_status
        self.hunger_level = hunger_level
        self.wealth_money = wealth_money
        self.mental_health = mental_health
        self.social_wellbeing = social_wellbeing
        self.beauty = 0 
        self.community = community
        self.recent_event = recent_event
        self.long_term_goal = long_term_goal
        self.inventory = inventory if inventory else ItemInventory() 
        
        if home: self.home = self.set_home(home)
        else: self.home = self.set_home()

        self.personality_traits = personality_traits if personality_traits else PersonalityTraits()
        self.motives = motives 
        if self.motives is None and self.graph_manager : 
             try:
                self.motives = self.calculate_motives()
             except Exception as e:
                print(f"Warning: Failed to calculate motives for {self.name}: {e}. Setting to default.")
                default_motive_score = 0.5 
                all_motives_default = {m_name.replace("_motive", ""): Motive(m_name.replace("_motive", ""), f"Default motive for {m_name.replace('_motive', '')}", default_motive_score) for m_name in PersonalMotives(Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0))._attributes}
                self.motives = PersonalMotives(**all_motives_default)

        self.luxury = 0 
        self.job_performance = job_performance
        self.material_goods = self.calculate_material_goods() if self.inventory else 0
        self.shelter = self.home.calculate_shelter_value() if self.home else 0
        self.success = self.calculate_success()
        self.control = self.calculate_control()
        self.hope = self.calculate_hope()
        self.stability = self.calculate_stability()
        self.happiness = self.calculate_happiness()

        self.stamina = 0
        self.current_satisfaction = 0
        self.current_mood = 50
        self.current_activity = None
        self.skills = skills if skills else CharacterSkills([]) 
        self.career_goals = career_goals
        
        self.uuid = uuid.uuid4()
        
        if gametime_manager: self.gametime_manager = gametime_manager
        else: raise ValueError("GameTimeManager instance required.")
        
        try:
            MemoryManagerClass = importlib.import_module("tiny_memories").MemoryManager
            self.memory_manager = MemoryManagerClass(gametime_manager)
        except ModuleNotFoundError as e: 
            print(f"CRITICAL WARNING: Failed to import MemoryManager due to: {e}. Character {self.name} will have no memory system.")
            self.memory_manager = None 
        except Exception as e: 
            print(f"CRITICAL WARNING: An unexpected error occurred initializing MemoryManager for {self.name}: {e}. No memory system.")
            self.memory_manager = None

        if location is None: self.location = Location(name, 0, 0, 0, 0, self.action_system) 
        else: self.location = location
        self.coordinates_location = self.location.get_coordinates()

        self.needed_items = []
        self.goals = [] 
        self.social_goals = [] 

        self.state = self.get_state() 
        self.romantic_relationships = []
        self.romanceable = romanceable
        self.exclusive_relationship = None
        self.base_libido = self.calculate_base_libido()
        self.monogamy = 0
        self.investment_portfolio = InvestmentPortfolio([])
        self.monogamy = self.calculate_monogamy()
        
        if self.graph_manager: 
            try:
                self.goals = self.evaluate_goals()
            except Exception as e:
                print(f"Warning: Failed to evaluate initial goals for {self.name}: {e}")
                self.goals = []
        else:
            self.goals = []

        if self.graph_manager and not self.graph_manager.get_node(node_id=self):
            self.graph_manager.add_character_node(self)
        self.post_init()

    def add_simple_social_goal(self, goal: SimpleSocialGoal):
        if not hasattr(self, 'social_goals') or self.social_goals is None:
            self.social_goals = []
        self.social_goals.append(goal)

    def to_dict(self):
        return {
            "name": self.name, "age": self.age, "pronouns": self.pronouns,
            "job_name": getattr(self.job, 'job_title', str(self.job)), 
            "health_status": self.health_status, "hunger_level": self.hunger_level,
            "wealth_money": self.wealth_money, "mental_health": self.mental_health,
            "social_wellbeing": self.social_wellbeing, "energy": self.energy,
            "location_name": getattr(self.location, 'name', "Unknown"),
            "goals_count": len(self.goals) if self.goals else 0,
            "social_goals_count": len(self.social_goals) if hasattr(self, 'social_goals') else 0,
        }

    def get_state(self): 
        return State(self) 

    def get_base_libido(self): return self.base_libido
    def post_init(self): logging.info(f"Character {self.name} has been created\n"); self._initialized = True
    def add_to_inventory(self, item: ItemObject):
        if self.graph_manager and not self.graph_manager.G.has_node(item): self.graph_manager.add_item_node(item)
        self.inventory.add_item(item)
    def generate_goals(self):
        if not self.motives or not self.graph_manager or not self.goap_planner: 
            print(f"Cannot generate goals for {self.name} due to missing motives, graph_manager or goap_planner.")
            return []
        goal_generator = GoalGenerator(self.motives, self.graph_manager, self.goap_planner, self.prompt_builder)
        return goal_generator.generate_goals(self)
    def get_location(self): return self.location
    def set_location(self, *location):
        if len(location) == 1:
            if isinstance(location[0], Location): self.location = location[0]
            elif isinstance(location[0], tuple): self.location = Location(location[0][0], location[0][1], 0, 0, self.action_system) 
        elif len(location) == 2: self.location = Location(location[0], location[1], 0, 0, self.action_system) 
        return self.location
    def get_coordinates_location(self): return self.location.get_coordinates()
    def set_coordinates_location(self, *coordinates):
        if len(coordinates) == 1:
            if isinstance(coordinates[0], tuple): self.location.set_coordinates(coordinates[0][0], coordinates[0][1]) 
        elif len(coordinates) == 2: self.location.set_coordinates(coordinates[0], coordinates[1]) 
        return self.location.coordinates_location
    def set_destination(self, destination): self.destination = destination; self.path = a_star_search(self.graph_manager, self.location, destination) if self.graph_manager else []
    def move_towards_next_waypoint(self, potential_field):
        if not self.path: return
        next_waypoint = self.path[0]
        if not isinstance(self.location, tuple) and hasattr(self.location, 'get_coordinates'): self.location_coords = self.location.get_coordinates()
        else: self.location_coords = self.location # Assume it's already a tuple if not Location object
        if not isinstance(self.velocity, tuple): self.velocity = (0,0) 
        
        # Temporary character-like object for steering behavior if self.location is not directly (x,y)
        temp_steer_char = type('SteerChar', (), {'location': self.location_coords, 'velocity': self.velocity})()

        seek_force = SteeringBehaviors.seek(temp_steer_char, next_waypoint)
        self.velocity = temp_steer_char.velocity # Update velocity from steering behavior

        avoid_force = SteeringBehaviors.avoid(temp_steer_char, self.graph_manager.get_obstacles() if self.graph_manager else [], avoidance_radius=2.0)
        potential_field_force = SteeringBehaviors.potential_field_force(temp_steer_char, potential_field)
        combined_force = (seek_force[0] + avoid_force[0] + potential_field_force[0], seek_force[1] + avoid_force[1] + potential_field_force[1])
        
        new_x = self.location_coords[0] + combined_force[0]
        new_y = self.location_coords[1] + combined_force[1]
        self.set_coordinates_location(new_x, new_y)

        if self.graph_manager: self.graph_manager.update_location(self.name, self.get_coordinates_location())
        if self.get_coordinates_location() == next_waypoint:
            self.path.pop(0)
            if not self.path: self.destination = None

    def add_new_goal(self, goal): self.goals.append((0, goal)) 
    def create_memory(self, description, timestamp, importance):
        if not self.memory_manager: print(f"Warning: No memory manager for {self.name}, cannot create memory."); return
        # Memory class might not be defined if tiny_memories fails to import
        try:
            MemoryClass = importlib.import_module("tiny_memories").Memory
            memory = MemoryClass(description, timestamp, importance) 
            self.memory_manager.add_memory(memory)
        except Exception as e:
            print(f"Warning: Could not create memory for {self.name}: {e}")

    def recall_recent_memories(self): return self.memory_manager.flat_access.get_recent_memories() if self.memory_manager else []
    def make_decision_based_on_memories(self): 
        if not self.memory_manager or not hasattr(self.memory_manager, 'make_decision_based_on_memories'): return {} # Check attribute too
        return self.memory_manager.make_decision_based_on_memories() 
    def retrieve_specific_memories(self, query): return self.memory_manager.retrieve_memories(query) if self.memory_manager else []
    def update_memory_importance(self, description, new_importance):
        if not self.memory_manager or not hasattr(self.memory_manager.flat_access, 'get_all_memories'): return
        for memory in self.memory_manager.flat_access.get_all_memories(): 
            if memory.description == description: memory.importance_score = new_importance; break
    def __str__(self): return f"Character: {self.name}, Age: {self.age}" 
    def __repr__(self): return f"Character({self.name}, {self.age})" 
    def get_char_attribute(self, attribute): return getattr(self, attribute, None)
    def __eq__(self, other): 
        if not isinstance(other, Character): return False
        return self.uuid == other.uuid 
    def __hash__(self): return hash(self.uuid) 
    def get_possible_interactions(self): return self.possible_interactions
    def play_animation(self, animation): 
        try:
            from tiny_animation_system import get_animation_system
            animation_system = get_animation_system(); success = animation_system.play_animation(self.name, animation)
            if success: logging.info(f"{self.name} started playing animation: {animation}")
            else: logging.warning(f"Failed to play animation {animation} for {self.name}")
            return success
        except ImportError: logging.info(f"{self.name} is playing animation: {animation}"); return True
        except Exception as e: logging.error(f"Error playing animation {animation} for {self.name}: {e}"); return False
    def describe(self): print(f"{self.name} is a {self.age}-year-old {self.pronouns}...") 
    def decide_to_join_event(self, event): return self.personality_traits.extraversion > 50 if self.personality_traits else False
    def decide_to_explore(self):
        if not self.personality_traits: return False
        if self.personality_traits.openness > 75: return True
        elif self.personality_traits.openness > 40 and self.personality_traits.conscientiousness < 50: return True
        return False
    def decide_to_take_challenge(self): 
        if not self.personality_traits: return "unsure due to missing personality traits"
        if self.personality_traits.conscientiousness > 60 and self.personality_traits.neuroticism < 50: return "ready to tackle the challenge"
        elif self.personality_traits.agreeableness > 50 and self.personality_traits.neuroticism < 40: return "takes on the challenge to help others"
        return "too stressed to take on the challenge right now"
    def respond_to_conflict(self, conflict_level): 
        if not self.personality_traits: return "responds cautiously"
        if self.personality_traits.agreeableness > 65: return "seeks a peaceful resolution"
        elif self.personality_traits.neuroticism > 70: return "avoids the situation entirely"
        return "confronts the issue directly"
    def define_descriptors(self): 
        descriptors = {"name": self.name, "age": self.age} 
        return descriptors
    def get_name(self): return self.name
    def set_name(self, name): self.name = name; return self.name
    def get_age(self): return self.age
    def set_age(self, age): self.age = age; return self.age
    def get_pronouns(self): return self.pronouns
    def set_pronouns(self, pronouns): self.pronouns = pronouns; return self.pronouns
    def get_job(self): return self.job
    def check_graph_uuid(self): return self.graph_manager.unique_graph_id if self.graph_manager else None
    def set_job(self, job):
        job_rules = JobRules()
        if isinstance(job, JobRoles): self.job = job
        elif isinstance(job, str):
            valid_job_role = job_rules.get_job_role_by_name(job) 
            if valid_job_role: self.job = valid_job_role
            else: self.job = random.choice(job_rules.ValidJobRoles) if job_rules.ValidJobRoles else "unemployed"
        else: self.job = random.choice(job_rules.ValidJobRoles) if job_rules.ValidJobRoles else "unemployed"
        return self.job
    def set_home(self, home=None):
        from tiny_buildings import House, CreateBuilding 
        if isinstance(home, House): self.home = home
        elif isinstance(home, str) and home != "homeless": self.home = CreateBuilding().create_house_by_type(home)
        elif home is None: self.home = CreateBuilding().generate_random_house()
        else: self.home = None 
        return self.home
    def get_home(self): return self.home
    def get_job_role(self): return self.job 
    def set_job_role(self, job): self.job = self.set_job(job) 
    def get_health_status(self): return self.health_status
    def set_health_status(self, health_status): self.health_status = health_status; return self.health_status
    def get_hunger_level(self): return self.hunger_level
    def set_hunger_level(self, hunger_level): self.hunger_level = hunger_level; return self.hunger_level
    def get_wealth_money(self): return self.wealth_money
    def set_wealth_money(self, wealth_money): self.wealth_money = wealth_money; return self.wealth_money
    def get_mental_health(self): return self.mental_health
    def set_mental_health(self, mental_health): self.mental_health = mental_health; return self.mental_health
    def get_social_wellbeing(self): return self.social_wellbeing
    def set_social_wellbeing(self, social_wellbeing): self.social_wellbeing = social_wellbeing; return self.social_wellbeing
    def get_happiness(self): return self.happiness
    def set_happiness(self, happiness): self.happiness = happiness; return self.happiness
    def get_shelter(self): return self.shelter
    def set_shelter(self, shelter): self.shelter = shelter; return self.shelter
    def get_stability(self): return self.stability
    def set_stability(self, stability): self.stability = stability; return self.stability
    def get_luxury(self): return self.luxury
    def set_luxury(self, luxury): self.luxury = luxury; return self.luxury
    def get_hope(self): return self.hope
    def set_hope(self, hope): self.hope = hope; return self.hope
    def get_success(self): return self.success
    def set_success(self, success): self.success = success; return self.success
    def get_control(self): return self.control
    def set_control(self, control): self.control = control; return self.control
    def get_job_performance(self): return self.job_performance
    def set_job_performance(self, job_performance): self.job_performance = job_performance; return self.job_performance
    def get_beauty(self): return self.beauty
    def set_beauty(self, beauty): self.beauty = beauty; return self.beauty
    def get_community(self): return self.community
    def set_community(self, community): self.community = community; return self.community
    def get_material_goods(self): return self.material_goods
    def set_material_goods(self, material_goods): self.material_goods = material_goods; return self.material_goods
    def get_friendship_grid(self): return self.friendship_grid
    def generate_friendship_grid(self): 
        if not self.graph_manager: return [{}]
        friendship_grid = []; relationships = self.graph_manager.analyze_character_relationships(self)
        for neighbor, relationship_data in relationships.items():
            if hasattr(neighbor, "name"): 
                friendship_entry = {"character_name": neighbor.name, "character_id": neighbor, "emotional_impact": relationship_data.get("emotional",0), "trust_level": relationship_data.get("trust",0), "relationship_strength": relationship_data.get("strength",0), "historical_bond": relationship_data.get("historical",0), "interaction_frequency": relationship_data.get("interaction_frequency",0), "friendship_status": self.graph_manager.check_friendship_status(self, neighbor)}
                friendship_score = (friendship_entry["emotional_impact"]*0.3 + friendship_entry["trust_level"]*0.25 + friendship_entry["relationship_strength"]*0.25 + friendship_entry["historical_bond"]*0.1 + friendship_entry["interaction_frequency"]*0.1)
                friendship_entry["friendship_score"] = max(0, min(100, friendship_score)); friendship_grid.append(friendship_entry)
        return friendship_grid if friendship_grid else [{}]

    def set_friendship_grid(self, friendship_grid): self.friendship_grid = friendship_grid if (isinstance(friendship_grid, list) and len(friendship_grid)>0) else self.generate_friendship_grid()
    def get_recent_event(self): return self.recent_event
    def set_recent_event(self, recent_event): self.recent_event = recent_event; return self.recent_event
    def get_long_term_goal(self): return self.long_term_goal
    def set_long_term_goal(self, long_term_goal): self.long_term_goal = long_term_goal; return self.long_term_goal
    def get_inventory(self): return self.inventory
    def set_inventory(self, food_items: List[FoodItem]=[], clothing_items: List[ItemObject]=[], tools_items: List[ItemObject]=[], weapons_items: List[ItemObject]=[], medicine_items: List[ItemObject]=[], misc_items: List[ItemObject]=[]):
        self.inventory = ItemInventory(food_items, clothing_items, tools_items, weapons_items, medicine_items, misc_items)
        if food_items: self.inventory.set_food_items(food_items)
        if clothing_items: self.inventory.set_clothing_items(clothing_items)
        if tools_items: self.inventory.set_tools_items(tools_items)
        if weapons_items: self.inventory.set_weapons_items(weapons_items)
        if medicine_items: self.inventory.set_medicine_items(medicine_items)
        if misc_items: self.inventory.set_misc_items(misc_items)
        return self.inventory
    def set_goals(self, goals): self.goals = goals; return self.goals
    def evaluate_goals(self): 
        if not self.graph_manager or not self.goap_planner or not self.motives: return []
        goal_queue = []
        if not self.goals or len(self.goals) == 0:
            goal_generator = GoalGenerator(self.motives, self.graph_manager, self.goap_planner, self.prompt_builder)
            self.goals = goal_generator.generate_goals(self)
        for _, goal_obj in self.goals: 
            utility = 0
            if hasattr(goal_obj, 'evaluate_utility_function') and callable(goal_obj.evaluate_utility_function):
                try:
                    utility = goal_obj.evaluate_utility_function(self, self.graph_manager, goal_obj.difficulty, goal_obj.criteria) 
                except:
                    utility = goal_obj.score 
            else:
                utility = goal_obj.score 
            goal_queue.append((utility, goal_obj))
        goal_queue.sort(reverse=True, key=lambda x: x[0])
        return goal_queue

    def calculate_material_goods(self): return round(tweener(self.inventory.count_total_items(),1000,0,100.0,2)) if self.inventory else 0
    def calculate_stability(self): 
        stability = 0; stability += self.get_shelter() or 0; stability += round(tweener(self.get_luxury() or 0, 100.0,0,10,2)); stability += self.get_hope() or 0; stability += round(tweener(self.get_success() or 0, 100.0,0,10,2)); stability += round(tweener(self.get_control() or 0,100.0,0,10,2)); stability += round(tweener(self.get_beauty() or 0,100.0,0,10,2)); stability += self.get_community() or 0; stability += round(tweener(self.get_material_goods() or 0,100.0,0,10,2)); stability += self.get_social_wellbeing() or 0;
        from tiny_graph_manager import cached_sigmoid_raw_approx_optimized; return cached_sigmoid_raw_approx_optimized(stability, 100.0)
    def calculate_happiness(self): 
        happiness = 0; happiness += self.get_hope() or 0
        if self.motives: happiness += importlib.import_module("tiny_graph_manager").cached_sigmoid_raw_approx_optimized(min(self.get_success() or 0, self.motives.get_success_motive().get_score()*10),100.0) 
        return importlib.import_module("tiny_graph_manager").cached_sigmoid_raw_approx_optimized(happiness,100.0)
    def calculate_success(self): return round(tweener(self.get_job_performance() or 0,100.0,0,50,2)) + round(tweener(self.get_material_goods() or 0,100.0,0,20,2)) + round(tweener(self.get_wealth_money() or 0,1000,0,20,2))
    def calculate_control(self): return (self.get_shelter() or 0) + round(tweener(self.get_success() or 0,100.0,0,10,2)) + round(tweener(self.get_material_goods() or 0,100.0,0,20,2)) + round(tweener(self.get_wealth_money() or 0,1000,0,20,2))
    def calculate_monogamy(self): 
        if not self.personality_traits or not self.motives or not self._initialized: return 0 
        monogamy = 0; monogamy -= self.personality_traits.get_openness(); monogamy += self.personality_traits.get_conscientiousness(); monogamy -= self.personality_traits.get_extraversion(); monogamy += 1 if not self.has_job() else 0
        if self.graph_manager: monogamy += 1 if self.get_wealth_money() <= self.graph_manager.get_average_attribute_value("wealth_money") else 0
        monogamy += 1 if self.get_home() is None else 0; monogamy += max(0, (50 - (self.base_libido or 0)) // 10); monogamy += self.age / 100.0; monogamy += self.motives.get_control_motive().get_score(); monogamy += self.motives.get_hope_motive().get_score(); monogamy += self.motives.get_mental_health_motive().get_score(); monogamy += self.motives.get_stability_motive().get_score(); monogamy += self.motives.get_family_motive().get_score()*2; monogamy += (self.stability or 0) / 100.0
        return monogamy
    def calculate_hope(self): 
        hope = 0; hope += round(tweener(self.get_beauty() or 0, 100.0,0.0,10.0,2.0)); hope += round(tweener(self.get_success() or 0,100.0,0.0,10.0,2)); hope += self.get_community() or 0; hope += round(tweener(self.get_material_goods() or 0,100.0,0.0,10,2.0)); hope += self.get_social_wellbeing() or 0
        from tiny_graph_manager import cached_sigmoid_raw_approx_optimized; return cached_sigmoid_raw_approx_optimized(hope, 100.0)
    def calculate_base_libido(self): 
        if not self.personality_traits: return 0
        base_libido = 0; base_libido += self.personality_traits.get_openness(); base_libido += self.personality_traits.get_extraversion(); base_libido += self.personality_traits.get_agreeableness(); base_libido -= self.personality_traits.get_neuroticism(); base_libido = max(0, base_libido); base_libido -= self.get_age() / 10; base_libido = max(0, base_libido); base_libido += (self.get_health_status() or 0) / 10; base_libido += (self.get_happiness() or 0) / 10; base_libido += (self.get_social_wellbeing() or 0) / 10; base_libido += random.randint(-4,4); base_libido += (self.get_control() or 0) / 10; base_libido = max(0, base_libido); base_libido = round(tweener(base_libido,60,0,60,2)); return base_libido
    def has_job(self):
        if isinstance(self.job, str): return self.job != "unemployed"
        return self.job is not None and getattr(self.job, 'job_name', "unemployed") != "unemployed"
    def has_investment(self): return len(self.investment_portfolio.get_stocks()) > 0
    def update_required_items(self, item_node_attrs, item_count=1):
        if isinstance(item_node_attrs, list): 
            for item in item_node_attrs: self.needed_items.append((item, item_count))
        elif isinstance(item_node_attrs, dict): 
             self.needed_items.append((item_node_attrs, item_count))
    def calculate_motives(self): 
        if not self.graph_manager: 
            print(f"Warning: No graph_manager for {self.name}, cannot calculate motives. Returning default.")
            default_motive_score = 0.5; all_motives_default = {m_name.replace("_motive", ""): Motive(m_name.replace("_motive", ""), f"Default motive for {m_name.replace('_motive', '')}", default_motive_score) for m_name in PersonalMotives(Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0))._attributes}
            return PersonalMotives(**all_motives_default)
        return self.graph_manager.calculate_motives(self)


# CreateCharacter class to be placed AFTER Character and all its dependencies like PersonalityTraits etc.
class CreateCharacter:
    def __init__(self):
        self.description = "This class is used to create a character."
    def __repr__(self): return f"CreateCharacter()"
    def create_new_character( self, mode: str = "auto", name: str = "John Doe", age: int = 18, pronouns: str = "they/them", job: str = "unemployed", health_status: float = 0.0, hunger_level: float = 0.0, wealth_money: float = 0.0, mental_health: float = 0.0, social_wellbeing: float = 0.0, job_performance: float = 0.0, community: float = 0.0, friendship_grid: dict = {}, recent_event: str = "", long_term_goal: str = "", inventory: ItemInventory = None, personality_traits: PersonalityTraits = None, motives: PersonalMotives = None, home: str = "", graph_manager_instance = None, gametime_manager_instance = None, action_system_instance = None, skills_list: List[Skill] = None): # Added skills_list
        from tiny_buildings import House, CreateBuilding 
        if mode != "auto": pass 
        
        if pronouns == "they/them" and mode == "auto":
            r_val = random.random()
            if r_val < 0.45: pronouns = "he/him"
            elif r_val < 0.9: pronouns = "she/her"
            else: pronouns = "they/them"
        if name == "John Doe" and mode == "auto": name = RandomNameGenerator().generate_name(pronouns=pronouns)
        if age == 18 and mode == "auto": age = max(18, round(random.gauss(30, 10)))
        if job == "unemployed" and mode == "auto": job_choice = random.choice(JobRules().ValidJobRoles); job = job_choice.job_name if hasattr(job_choice, 'job_name') else str(job_choice)
        if wealth_money == 0 and mode == "auto": wealth_money = round(abs(random.triangular(10, 10000, 500)))
        if mental_health == 0 and mode == "auto": mental_health = random.randint(3, 10) 
        if social_wellbeing == 0 and mode == "auto": social_wellbeing = random.randint(3, 10) 
        if job_performance == 0 and mode == "auto": job_performance = round(random.gauss(50, 20))
        if home == "" and mode == "auto": home_obj = CreateBuilding().generate_random_house()
        elif isinstance(home, str) and home != "": home_obj = CreateBuilding().create_house_by_type(home)
        else: home_obj = home

        if personality_traits is None and mode == "auto":
            personality_traits = PersonalityTraits(
                openness=max(-4, min(4, random.gauss(0, 2))),
                conscientiousness=max(-4, min(4, random.gauss(0, 2))),
                extraversion=max(-4, min(4, random.gauss(0, 2))),
                agreeableness=max(-4, min(4, random.gauss(0, 2))),
                neuroticism=max(-4, min(4, random.gauss(0, 2))))
        if health_status == 0 and mode == "auto": health_status = random.randint(5, 10) 
        if hunger_level == 0 and mode == "auto": hunger_level = random.randint(0, 5) 
        if community == 0 and mode == "auto": community = random.randint(0,10)
        
        current_graph_manager = graph_manager_instance
        if current_graph_manager is None:
            try:
                GraphManagerModule = importlib.import_module("tiny_graph_manager")
                current_graph_manager = GraphManagerModule.GraphManager()
            except Exception as e:
                print(f"Failed to init GraphManager in CreateCharacter: {e}")
                current_graph_manager = None 

        current_gametime_manager = gametime_manager_instance if gametime_manager_instance else GameTimeManager()
        current_action_system = action_system_instance if action_system_instance else ActionSystem()
        char_skills = CharacterSkills(skills_list if skills_list else [])


        created_char = Character(
            name=name, age=age, pronouns=pronouns, job=job, health_status=health_status,
            hunger_level=hunger_level, wealth_money=wealth_money, mental_health=mental_health,
            social_wellbeing=social_wellbeing, job_performance=job_performance, community=community,
            home=home_obj, personality_traits=personality_traits, inventory=inventory if inventory else ItemInventory(),
            graph_manager=current_graph_manager, gametime_manager=current_gametime_manager, 
            action_system=current_action_system, skills=char_skills 
        )
        
        if recent_event == "" and mode == "auto": created_char.recent_event = recent_event_generator(created_char)
        else: created_char.recent_event = recent_event
        if long_term_goal == "" and mode == "auto": created_char.long_term_goal = default_long_term_goal_generator(created_char)
        else: created_char.long_term_goal = long_term_goal
        
        if motives is None and mode == "auto" and created_char.graph_manager:
             try:
                created_char.motives = created_char.calculate_motives()
             except Exception as e: 
                print(f"Warning: Auto-calculating motives failed for {name}: {e}. Using default.")
                default_motive_score = 0.5
                all_motives_default = {m_name.replace("_motive", ""): Motive(m_name.replace("_motive", ""), f"Default motive for {m_name.replace('_motive', '')}", default_motive_score) for m_name in PersonalMotives(Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0),Motive(""," ",0))._attributes}
                created_char.motives = PersonalMotives(**all_motives_default)
        elif motives:
            created_char.motives = motives
        
        created_char.happiness = created_char.calculate_happiness()
        created_char.stability = created_char.calculate_stability()
        return created_char


if __name__ == "__main__":
    graph_man = None
    try:
        GraphManagerModule = importlib.import_module("tiny_graph_manager")
        graph_man = GraphManagerModule.GraphManager()
    except ModuleNotFoundError as e:
        print(f"WARNING (main): GraphManager could not be initialized due to: {e}. Some Character functions might be limited.")
    except Exception as e:
        print(f"WARNING (main): GraphManager other error: {e}")

    gametime_manager = GameTimeManager()
    action_system = ActionSystem() 
    pass

[end of tiny_characters.py]
