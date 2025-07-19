"""Utilities for constructing LLM prompts for Tiny Village characters.

This module provides the :class:`PromptBuilder` and supporting classes used to
build rich text prompts that are sent to the language model. The prompts include
character state, goals and available actions so that the model can choose the
next behaviour for a character.  The classes here do not perform any network
calls; they only format information.
"""

import random
from typing import Dict, List, Optional

import tiny_characters as tc


class NeedsPriorities:
    """Calculate priority scores for all basic character needs."""
    def __init__(self):
        self.needs = [
            "health",
            "hunger",
            "wealth",
            "mental_health",
            "social_wellbeing",
            "happiness",
            "shelter",
            "stability",
            "luxury",
            "hope",
            "success",
            "control",
            "job_performance",
            "beauty",
            "community",
            "material_goods",
            "friendship_grid",
        ]
        # Value represents a character's current need level
        self.needs_priorities = {
            "health": 0,
            "hunger": 0,
            "wealth": 0,
            "mental_health": 0,
            "social_wellbeing": 0,
            "happiness": 0,
            "shelter": 0,
            "stability": 0,
            "luxury": 0,
            "hope": 0,
            "success": 0,
            "control": 0,
            "job_performance": 0,
            "beauty": 0,
            "community": 0,
            "material_goods": 0,
            "friendship_grid": 0,
        }

    def get_needs_priorities(self):
        return self.needs_priorities

    def get_needs_priorities_list(self):
        return self.needs_priorities.keys()

    def get_needs_priorities_values(self):
        return self.needs_priorities.values()

    def get_needs_priorities_sorted(self):
        return sorted(self.needs_priorities.items(), key=lambda x: x[1])

    def get_needs_priorities_sorted_list(self):
        return [x[0] for x in sorted(self.needs_priorities.items(), key=lambda x: x[1])]

    def get_needs_priorities_sorted_values(self):
        return [x[1] for x in sorted(self.needs_priorities.items(), key=lambda x: x[1])]

    def get_needs_priorities_sorted_reverse(self):
        return sorted(self.needs_priorities.items(), key=lambda x: x[1], reverse=True)

    def get_needs_priorities_sorted_list_reverse(self):
        return [
            x[0]
            for x in sorted(
                self.needs_priorities.items(), key=lambda x: x[1], reverse=True
            )
        ]

    def get_needs_priorities_sorted_values_reverse(self):
        return [
            x[1]
            for x in sorted(
                self.needs_priorities.items(), key=lambda x: x[1], reverse=True
            )
        ]

    def get_needs_priorities_sorted_by_value(self):
        return sorted(self.needs_priorities.items(), key=lambda x: x[1])

    def set_needs_priorities(self, needs_priorities):

        self.needs_priorities = needs_priorities

    def calculate_health_priority(self, character: tc.Character):
        # Health priority is based on health status
        # Health status is a value from 1-10
        # Health priority is a value from 1-100
        # Health priority is calculated by multiplying health status times 10 and subtracting from 100
        health_status = character.get_health_status()
        health_priority = (
            100 - (health_status * 10)
        ) + character.get_motives().get_health_motive()
        return health_priority

    def calculate_hunger_priority(self, character: tc.Character):
        # Hunger priority is based on hunger level
        # Hunger level is a value from 1-10
        # Hunger priority is a value from 1-100
        # Hunger priority is calculated by multiplying hunger level times 10
        hunger_level = character.get_hunger_level()
        hunger_priority = (
            hunger_level * 10 + character.get_motives().get_hunger_motive()
        )
        return hunger_priority

    def calculate_wealth_priority(self, character: tc.Character):
        # Wealth priority is based on wealth
        # Wealth is a value from 1-10
        # Wealth priority is a value from 1-100
        # Wealth priority is calculated by multiplying wealth times 10
        wealth = character.get_wealth()
        wealth_priority = character.get_motives().get_wealth_motive()
        return wealth_priority

    def calculate_mental_health_priority(self, character: tc.Character):
        # Mental health priority is based on mental health
        # Mental health is a value from 1-10
        # Mental health priority is a value from 1-100
        # Mental health priority is calculated by multiplying mental health times 10
        mental_health = character.get_mental_health()
        mental_health_priority = character.get_motives().get_mental_health_motive()
        return mental_health_priority

    def calculate_social_wellbeing_priority(self, character: tc.Character):
        # Social wellbeing priority is based on social wellbeing
        # Social wellbeing is a value from 1-10
        # Social wellbeing priority is a value from 1-100
        # Social wellbeing priority is calculated by multiplying social wellbeing times 10
        social_wellbeing = character.get_social_wellbeing()
        social_wellbeing_priority = (
            character.get_motives().get_social_wellbeing_motive()
        )
        return social_wellbeing_priority

    def calculate_happiness_priority(self, character: tc.Character):
        # Happiness priority is based on happiness
        # Happiness is a value from 1-10
        # Happiness priority is a value from 1-100
        # Happiness priority is calculated by multiplying happiness times 10
        happiness = character.get_happiness()
        happiness_priority = character.get_motives().get_happiness_motive()
        return happiness_priority

    def calculate_shelter_priority(self, character: tc.Character):
        # Shelter priority is based on shelter
        # Shelter is a value from 1-10
        # Shelter priority is a value from 1-100
        # Shelter priority is calculated by multiplying shelter times 10
        shelter = character.get_shelter()
        shelter_priority = character.get_motives().get_shelter_motive()
        return shelter_priority

    def calculate_stability_priority(self, character: tc.Character):
        # Stability priority is based on stability
        # Stability is a value from 1-10
        # Stability priority is a value from 1-100
        # Stability priority is calculated by multiplying stability times 10
        stability = character.get_stability()
        stability_priority = character.get_motives().get_stability_motive()
        return stability_priority

    def calculate_luxury_priority(self, character: tc.Character):
        # Luxury priority is based on luxury
        # Luxury is a value from 1-10
        # Luxury priority is a value from 1-100
        # Luxury priority is calculated by multiplying luxury times 10
        luxury = character.get_luxury()
        luxury_priority = character.get_motives().get_luxury_motive()
        return luxury_priority

    def calculate_hope_priority(self, character: tc.Character):
        # Hope priority is based on hope
        # Hope is a value from 1-10
        # Hope priority is a value from 1-100
        # Hope priority is calculated by multiplying hope times 10
        hope = character.get_hope()
        hope_priority = character.get_motives().get_hope_motive()
        return hope_priority

    def calculate_success_priority(self, character: tc.Character):
        # Success priority is based on success
        # Success is a value from 1-10
        # Success priority is a value from 1-100
        # Success priority is calculated by multiplying success times 10
        success = character.get_success()
        success_priority = character.get_motives().get_success_motive()
        return success_priority

    def calculate_control_priority(self, character: tc.Character):
        # Control priority is based on control
        # Control is a value from 1-10
        # Control priority is a value from 1-100
        # Control priority is calculated by multiplying control times 10
        control = character.get_control()
        control_priority = character.get_motives().get_control_motive()
        return control_priority

    def calculate_job_performance_priority(self, character: tc.Character):
        # Job performance priority is based on job performance
        # Job performance is a value from 1-10
        # Job performance priority is a value from 1-100
        # Job performance priority is calculated by multiplying job performance times 10
        job_performance = character.get_job_performance()
        job_performance_priority = character.get_motives().get_job_performance_motive()
        return job_performance_priority

    def calculate_beauty_priority(self, character: tc.Character):
        # Beauty priority is based on beauty
        # Beauty is a value from 1-10
        # Beauty priority is a value from 1-100
        # Beauty priority is calculated by multiplying beauty times 10
        beauty = character.get_beauty()
        beauty_priority = character.get_motives().get_beauty_motive() - beauty
        return beauty_priority

    def calculate_community_priority(self, character: tc.Character):
        # Community priority is based on community
        # Community is a value from 1-10
        # Community priority is a value from 1-100
        # Community priority is calculated by multiplying community times 10
        community = character.get_community()
        community_priority = character.get_motives().get_community_motive()
        return community_priority

    def calculate_material_goods_priority(self, character: tc.Character):
        # Material goods priority is based on material goods
        # Material goods is a value from 1-10
        # Material goods priority is a value from 1-100
        # Material goods priority is calculated by multiplying material goods times 10
        material_goods = character.get_material_goods()
        material_goods_priority = character.get_motives().get_material_goods_motive()
        return material_goods_priority

    def calculate_friendship_grid_priority(self, character: tc.Character):
        # Friendship grid priority is based on social connections and relationships
        # Use social_wellbeing_motive as the base since friendship relates to social wellbeing
        # Calculate aggregate friendship score from the character's friendship grid
        friendship_grid = character.get_friendship_grid()
        
        # Calculate average friendship score from the grid
        if friendship_grid and len(friendship_grid) > 0:
            # Filter out empty dictionaries and calculate average friendship score
            valid_friendships = [f for f in friendship_grid if f and 'friendship_score' in f]
            if valid_friendships:
                avg_friendship_score = sum(f['friendship_score'] for f in valid_friendships) / len(valid_friendships)
                # Convert to 0-10 scale for consistency with other priorities
                # Clamp to reasonable bounds (0-100 friendship score range)
                avg_friendship_score = max(0, min(100, avg_friendship_score))
                friendship_state = avg_friendship_score / 10.0
            else:
                friendship_state = 0  # No valid friendships
        else:
            friendship_state = 0  # No friendship data
        
        # Combine with social wellbeing motive (friendship is social)
        social_motive = character.get_motives().get_social_wellbeing_motive()
        
        # Calculate priority: higher motive with lower current state = higher priority
        # Ensure priority is always non-negative
        friendship_grid_priority = max(0, social_motive + (10 - friendship_state) * 2)
        return friendship_grid_priority

    def calculate_needs_priorities(self, character: tc.Character):
        # Calculate needs priorities based on character's current situation
        # Needs priorities are values from 1-100
        # Needs priorities are calculated by multiplying need level times 10
        # Needs priorities are calculated by adding motive value
        # Needs priorities are calculated by adding need level times motive value
        # Needs priorities are calculated by adding need level times motive value and subtracting from 100
        needs_priorities = {
            "health": self.calculate_health_priority(character),
            "hunger": self.calculate_hunger_priority(character),
            "wealth": self.calculate_wealth_priority(character),
            "mental_health": self.calculate_mental_health_priority(character),
            "social_wellbeing": self.calculate_social_wellbeing_priority(character),
            "happiness": self.calculate_happiness_priority(character),
            "shelter": self.calculate_shelter_priority(character),
            "stability": self.calculate_stability_priority(character),
            "luxury": self.calculate_luxury_priority(character),
            "hope": self.calculate_hope_priority(character),
            "success": self.calculate_success_priority(character),
            "control": self.calculate_control_priority(character),
            "job_performance": self.calculate_job_performance_priority(character),
            "beauty": self.calculate_beauty_priority(character),
            "community": self.calculate_community_priority(character),
            "material_goods": self.calculate_material_goods_priority(character),
            "friendship_grid": self.calculate_friendship_grid_priority(character),
        }

        return needs_priorities


class ActionOptions:
    """List and prioritize the actions a character can perform."""
    def __init__(self):
        self.actions = [
            "buy_food",
            "eat_food",
            "improve_job_performance",
            "increase_friendship",
            "improve_mental_health",
            "pursue_hobby",
            "volunteer_time",
            "set_goal",
            "leisure_activity",
            "organize_event",
            "research_new_technology",
            "buy_medicine",
            "take_medicine",
            "visit_doctor",
            "collaborate_colleagues",
            "gather_resource",
            "trade_goods",
            "repair_item",
            "get_educated",
            "social_visit",
            "attend_event",
            "go_to_work",
            "clean_up",
            "invest_wealth",
            "buy_property",
            "sell_property",
            "move_to_new_location",
            "commission_service",
            "start_business",
            "craft_item",
            "work_current_job",
        ]

    def prioritize_actions(self, character: tc.Character):
        # Prioritize actions based on character's current situation
        # Actions that are more likely to be chosen are placed earlier in the list
        # Actions that are less likely to be chosen are placed later in the list
        # Actions that are not possible are removed from the list
        # Actions that are possible are kept in the list
        # Actions that are possible but not likely are moved to the end

        # char_dict = character.to_dict()
        # inv_dict = character.inventory.to_dict()

        # Sample criteria for prioritizing actions
        needs_goals = {
            "buy_food": character.get_hunger_level() > 7
            and character.get_wealth_money() > 1
            and (
                character.get_inventory().count_food_items_total() < 5
                or character.get_inventory().count_food_calories_total()
                < character.get_hunger_level()
            ),
            "eat_food": character.get_hunger_level() > 5
            and character.get_inventory().count_food_items_total() > 0,
            "visit_doctor": character.get_health_status() < 3
            or character.get_mental_health() < 4,
            "take_medicine": character.get_health_status() < 5,
            "improve_shelter": character.get_shelter() < 4,
            "attend_event": character.get_social_wellbeing() < 5
            or character.get_community() < 5,
            "pursue_hobby": character.get_happiness() < 5 or character.get_beauty() < 5,
            "self_care": character.get_mental_health() < 5,
            "social_visit": character.get_friendship_grid() < 5,
            "volunteer_time": character.get_community() < 5,
            "improve_job_performance": character.get_job_performance() < 5,
            "get_educated": character.get_long_term_goal() == "career_advancement",
            "set_goal": character.get_hope() < 5,
            "start_business": character.get_long_term_goal() == "entrepreneurship",
            "trade_goods": character.get_wealth_money() > 5
            or character.get_material_goods() > 5,
            "invest_wealth": character.get_wealth_money() > 8,
            # ... additional mappings ...
        }

        prioritized_actions = []
        prioritized_actions += [action for action, need in needs_goals.items() if need]
        other_actions = [
            action for action in self.actions if action not in prioritized_actions
        ]
        if len(prioritized_actions) < 5:
            prioritized_actions += other_actions[: 5 - len(prioritized_actions)]
        return prioritized_actions


class DescriptorMatrices:
    """Repository of descriptors used to enrich generated text prompts."""
    def __init__(self):

        self.job_adjective = {
            "default": [
                "skilled",
                "hardworking",
                "friendly",
                "friendly, outgoing",
                "average",
            ]
        }

        self.job_pronoun = {
            "default": ["person"],
            "Engineer": [
                "person",
                "engineer",
                "programmer",
                "developer",
                "coder",
                "software engineer",
                "hardware engineer",
                "computer scientist",
                "computer engineer",
                "computer programmer",
                "computer scientist",
                "computer technician",
                "computer repair technician",
                "computer repairman",
                "computer repairwoman",
                "computer repair person",
                "computer repair specialist",
                "computer repair expert",
                "computer repair professional",
                "computer repair master",
                "computer repair guru",
                "computer repair wizard",
                "computer repair genius",
                "computer repair prodigy",
                "computer repair whiz",
                "computer repair wiz",
                "computer nerd",
                "computer geek",
            ],
            "Farmer": [
                "person",
                "farmer",
                "agriculturalist",
                "agricultural scientist",
                "agricultural engineer",
                "agricultural technician",
                "agricultural nerd",
                "agricultural geek",
            ],
        }

        self.job_place = {
            "default": ["at your job"],
            "Engineer": [""],
            "Farmer": ["at your farm"],
        }

        self.job_enjoys_verb = {
            "default": ["working with", "helping"],
            "Engineer": [
                "building",
                "designing",
                "creating",
                "developing",
                "programming",
                "testing",
                "debugging",
                "fixing",
                "improving",
                "optimizing",
                "learning",
                "teaching",
                "mentoring",
                "leading",
                "managing",
                "collaborating",
                "working",
                "writing",
                "reading",
                "researching",
                "analyzing",
                "planning",
                "documenting",
                "communicating",
                "presenting",
                "speaking",
                "talking",
                "discussing",
                "debating",
                "arguing",
                "solving",
                "simplifying",
                "automating",
                "optimizing",
            ],
            "Farmer": [
                "planting",
                "growing",
                "harvesting",
                "watering",
                "feeding",
                "tending",
                "caring",
                "cultivating",
                "nurturing",
                "pruning",
                "weeding",
                "fertilizing",
                "sowing",
                "reaping",
                "mowing",
                "raking",
                "plowing",
                "tilling",
                "hoeing",
                "digging",
                "shoveling",
                "raking",
            ],
        }

        self.job_verb_acts_on_noun = {
            "default": ["your hands", "others"],
            "Engineer": [
                "things",
                "machines",
                "doo-dads",
                "gizmos",
                "widgets",
                "programs",
                "software",
                "hardware",
                "systems",
                "components",
                "parts",
                "circuits",
                "circuits",
                "devices",
                "solutions",
            ],
            "Farmer": [
                "plants",
                "crops",
                "vegetables",
                "fruits",
                "grains",
                "flowers",
                "trees",
                "shrubs",
                "bushes",
                "grass",
                "weeds",
                "soil",
                "land",
                "fields",
                "gardens",
                "orchards",
                "vineyards",
                "pastures",
                "meadows",
                "ranches",
                "farms",
                "livestock",
                "animals",
                "cattle",
                "pigs",
                "chickens",
                "sheep",
                "goats",
                "horses",
                "llamas",
                "alpacas",
                "ostriches",
                "turkeys",
                "geese",
                "ducks",
                "fish",
                "aquatic life",
                "wildlife",
            ],
        }

        self.job_currently_working_on = {
            "Engineer": [
                "a new project",
                "a new software project",
                "a new hardware project",
                "a new product",
                "a new feature",
                "a new design",
                "a new system",
                "a new solution",
                "a new component",
                "a new part",
                "a new circuit",
                "a new device",
                "a new machine",
                "a new tool",
                "a new program",
                "a new algorithm",
                "a new technology",
                "a new language",
                "a new framework",
                "a new library",
                "a new interface",
                "a new API",
                "a new database",
                "a new website",
                "a new app",
                "a new game",
                "a new tool",
                "a new service",
                "a new business",
                "a new company",
                "a new startup",
                "a new project",
                "a new idea",
                "a new concept",
                "a new invention",
                "a new discovery",
                "a new theory",
                "a new hypothesis",
                "a new experiment",
                "a new method",
                "a new technique",
                "a new approach",
                "a new strategy",
                "a new plan",
                "a new goal",
                "a new objective",
                "a new target",
                "a new milestone",
                "a new task",
                "a new assignment",
                "a new mission",
                "a new quest",
                "a new adventure",
                "a new journey",
                "a new adventure",
                "a new experience",
                "a new opportunity",
                "a new challenge",
                "debugging a new bug",
                "fixing a new defect",
                "solving  a new error",
            ],
            "Farmer": [
                "a new crop",
                "a new harvest",
                "a new field",
                "a new garden",
                "a new orchard",
                "a new vineyard",
                "a new pasture",
                "a new meadow",
                "a new ranch",
                "a new farm",
                "a new livestock",
                "a new animal",
                "a new cattle",
                "a new pig",
                "a new chicken",
                "a new sheep",
                "a new goat",
                "a new horse",
                "a new llama",
                "a new alpaca",
                "a new ostrich",
                "a new turkey",
                "a new goose",
                "a new duck",
                "a new fish",
                "a new aquatic life",
                "a new wildlife",
                "a new plant",
                "a new vegetable",
                "a new fruit",
                "a new grain",
                "a new flower",
                "a new tree",
                "a new shrub",
                "a new bush",
                "a new grass",
                "a new weed",
                "a new soil",
                "a new land",
                "a new field",
                "a new garden",
                "a new orchard",
                "a new vineyard",
                "a new pasture",
                "a new meadow",
                "a new ranch",
                "a new farm",
                "a new livestock",
                "a new animal",
                "a new cattle",
                "a new pig",
                "a new chicken",
                "a new sheep",
                "a new goat",
                "a new horse",
                "a new llama",
                "a new alpaca",
                "a new ostrich",
                "a new turkey",
                "a new goose",
                "a new duck",
                "a new fish",
                "a new aquatic life",
                "a new wildlife",
            ],
        }

        self.job_planning_to_attend = {
            "Engineer": [
                "tech conference",
                "tech meetup",
                "developers conference",
                "maker faire",
                "hackathon",
                "startup conference",
                "tech talk",
                "tech event",
                "tech meetup",
                "tech gathering",
                "tech party",
                "tech event",
                "tech festival",
                "tech expo",
                "tech convention",
                "tech summit",
                "tech fair",
                "tech showcase",
                "tech competition",
            ],
            "Farmer": [
                "farmers market",
                "farmers conference",
                "farmers meetup",
                "farmers convention",
                "farmers fair",
                "farmers showcase",
                "farmers competition",
                "farmers festival",
                "farmers expo",
                "farmers gathering",
                "farmers party",
                "farmers event",
                "farmers summit",
                "farmers fair",
                "farmers showcase",
                "farmers competition",
            ],
        }

        self.job_hoping_to_there = {
            "Engineer": [
                "meet some of your colleagues",
                "encounter some new innovations",
            ],
            "Farmer": ["sell some of your produce", "buy a new tool"],
        }

        self.job_hoping_to_learn = {
            "Engineer": [
                "new programming languages",
                "new frameworks",
                "new libraries",
                "new technologies",
                "new tools",
                "new techniques",
                "new methods",
                "new approaches",
                "new strategies",
                "new plans",
                "new goals",
                "new objectives",
                "new targets",
                "new milestones",
                "new tasks",
                "new assignments",
                "new missions",
                "new quests",
                "new adventures",
                "new journeys",
                "new experiences",
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
            "Farmer": [
                "new farming techniques",
                "new farming methods",
                "new farming approaches",
                "new farming strategies",
                "new farming plans",
                "new farming goals",
                "new farming objectives",
                "new farming targets",
                "new farming milestones",
                "new farming tasks",
                "new farming assignments",
                "new farming missions",
                "new farming quests",
                "new farming adventures",
                "new farming journeys",
                "new farming experiences",
                "new farming opportunities",
                "new farming challenges",
                "new farming ideas",
                "new farming concepts",
                "new farming inventions",
                "new farming discoveries",
                "new farming theories",
                "new farming hypotheses",
                "new farming experiments",
                "new farming algorithms",
                "new farming designs",
                "new farming systems",
                "new farming solutions",
                "new farming components",
                "new farming parts",
                "new farming circuits",
                "new farming devices",
                "new farming machines",
                "new farming programs",
                "new farming software",
                "new farming hardware",
                "new farming products",
                "new farming features",
            ],
        }

        self.job_hoping_to_meet = {
            "Engineer": [
                "new people",
                "new friends",
                "new colleagues",
                "new mentors",
                "new leaders",
                "new managers",
                "new collaborators",
                "new partners",
                "new investors",
                "new customers",
                "new clients",
                "new users",
                "new developers",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
            ],
            "Farmer": [
                "new people",
                "new friends",
                "new colleagues",
                "new mentors",
                "new leaders",
                "new managers",
                "new collaborators",
                "new partners",
                "new investors",
                "new customers",
                "new clients",
                "new users",
                "new developers",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
            ],
        }

        self.job_hoping_to_find = {
            "Engineer": [
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
            "Farmer": [
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
        }

        self.feeling_health = {
            "healthy": [
                "in excellent health",
                "healthy",
                "doing well",
                "feeling good",
                "feeling great",
                "feeling amazing",
                "feeling fantastic",
                "feeling excellent",
                "feeling energetic",
                "strong",
                "fit",
                "feeling invincible",
            ],
            "sick": [
                "feeling sick",
                "feeling ill",
                "feeling unwell",
                "feeling bad",
                "feeling terrible",
                "feeling horrible",
                "feeling awful",
                "feeling absolutely dreadful",
                "miserable",
            ],
            "injured": ["injured", "hurt", "wounded", "damaged", "broken", "bruised"],
        }

        self.feeling_hunger = {
            "full": [
                "you are full",
                "you are satisfied",
                "you are not hungry",
                "you are barely peckish",
                "you are not hungry at all",
                "you are not hungry in the slightest",
                "you are not hungry whatsoever",
                "you are not hungry in the least",
            ],
            "moderate": [
                "your hunger is moderate",
                "you are only slightly hungry",
                "you are moderately hungry",
                "you are a bit hungry",
                "you could use a bite to eat",
                "you could do with a snack",
                "you could do with a meal",
                "you could do with a bite",
            ],
            "hungry": [
                "you are hungry",
                "you are starving",
                "you are famished",
                "you are ravenous",
                "you are starving",
            ],
            "starving": [
                "you are starving",
                "you are famished",
                "you are ravenous",
                "you are starving",
            ],
        }

        self.event_recent = {
            "default": ["Recently"],
            "craft fair": ["After your success at the craft fair"],
            "community center": ["After you helped at the community center"],
            "hospital": ["After you were recently in the hospital"],
            "nursing home": ["Since you helped out at the nursing home"],
            "outbreak": ["With the recent outbreak"],
            "rains": ["The recent rains"],
            "learning": ["Recently, you learned"],
        }

        self.financial_situation = {
            "default": ["financially, you are doing okay"],
            "rich": [
                "you are financially well-off",
                "you are rich",
                "you are wealthy",
                "you are well-off",
                "you are well-to-do",
                "you are well-heeled",
                "you are well-fixed",
                "you are well-situated",
                "you are well-provided",
                "you are well-provided for",
                "you are well-endowed",
                "you are well-furnished",
                "you are well-supplied",
                "you are well-stocked",
                "you are well-equipped",
                "you are well-prepared",
                "you are well-organized",
                "you are well-ordered",
                "you are well-regulated",
                "you are well-arranged",
                "you are well-balanced",
                "you are well-adjusted",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
            ],
            "stable": [
                "your financial situation is stable",
                "you are financially stable",
                "you are financially secure",
                "you are financially comfortable",
            ],
            "poor": [
                "you are financially poor",
                "you are financially struggling",
                "you are financially unstable",
                "you are financially insecure",
                "you are financially uncomfortable",
                "you are financially squeezed",
                "your finances are tight",
                "you are financially strapped",
                "you are financially stressed",
                "you are financially burdened",
                "you are struggling to make ends meet",
                "you are struggling to get by",
                "you are struggling to get through financially",
                "you are struggling to pay the bills",
                "you are struggling to pay the rent",
                "you are broke",
                "you are in debt",
                "you are in the red",
                "you are in the hole",
                "you are in the negative",
            ],
            "bankrupt": [
                "you are bankrupt",
                "you are insolvent",
                "you are in debt",
                "you are in the red",
                "you are in the hole",
                "you are in the negative",
                "you are destitute",
            ],
        }

        self.motivation = {
            "default": [
                "You're motivated to ",
                "Today, you aim to ",
                "Today offers the chance to",
                "You remind yourself of your goal to",
                "You're closer to your goal of",
            ]
        }

        self.weather_description = {
            "default": [
                "it's an average day outside",
                "it's a typical day outside",
                "it's a normal day outside",
                "it's a regular day outside",
                "it's a standard day outside",
                "it's a typical day outside",
                "it's a usual day outside",
                "it's a common day outside",
                "it's a standard day out there today",
            ],
            "sunny": [
                "it's a sunny day outside",
                "it's a bright day outsid",
                "it's a clear day out",
            ],
            "cloudy": [
                "it's a cloudy day outside",
                "it's a cloudy day out",
                "it's a bit overcast outside",
            ],
            "rainy": [
                "it's a rainy day outside",
                "it's a bit drizzly outside",
                "it's a bit rainy outside",
                "it's a bit wet outside",
                "it's a bit damp outside",
                "it's a bit moist outside",
            ],
            "snowy": [
                "it's a snowy day outside",
                "it's a bit snowy outside",
                "it's a bit icy outside",
                "it's a bit frosty outside",
                "it's a bit slushy outside",
                "it's a bit cold outside",
                "it's a bit chilly outside",
                "it's a bit freezing outside",
                "it's a bit frigid outside",
                "it's a bit wintry outside",
                "it's a bit wintery outside",
                "it's a bit frosty outside",
                "it's a bit icy outside",
                "it's a bit snowy outside",
                "it's a bit slushy outside",
                "it's a bit cold outside",
                "it's a bit chilly outside",
                "it's a bit freezing outside",
                "it's a bit frigid outside",
                "it's a bit wintry outside",
                "it's a bit wintery outside",
            ],
            "windy": [
                "it's a windy day outside",
                "it's a bit windy outside",
                "it's a bit breezy outside",
                "it's a bit gusty outside",
                "it's a bit blustery outside",
                "it's a bit windy out there today",
            ],
            "stormy": [
                "it's a stormy day outside",
                "it's a bit stormy outside",
                "it's a bit stormy out there today",
            ],
            "foggy": [
                "it's a foggy day outside",
                "it's a bit foggy outside",
                "it's a bit misty outside",
                "it's a bit hazy outside",
                "it's a bit smoky outside",
                "it's a bit smoggy outside",
                "it's a bit foggy out there today",
            ],
        }

        self.routine_question_framing = {
            "default": [
                "Considering the weather and your current situation, what do you choose to do next?\n",
                "What do you choose to do next?\n",
                "What do you do next?\n",
                "What will your focus be?\n",
                "What will you do?\n",
                "What will you focus on?\n",
                "What will you work on?\n",
                "What will you do next?\n",
                "What will you do?\n",
                "What is your next move?\n",
                "What is your next step?\n",
                "What is your next action?\n",
                "What is your next priority?\n",
            ]
        }

        self.action_descriptors = {
            "buy_food": [
                "Go to the market",
                "Go to the grocery store",
                "Go to the supermarket",
                "Go to the bodega",
                "Go to the corner store",
                "Go to the convenience store",
                "Go to the deli",
                "Go to the farmers market",
                "Go to the farm",
                "Go to the farm stand",
                "Go to the market",
            ]
        }

        # self.health_status = ["healthy", "sick", "injured", "disabled", "dying"]
        # self.hunger_level = ["full", "moderate", "hungry", "starving"]
        # self.wealth_money = ["rich", "moderate", "poor", "bankrupt"]
        # self.mental_health = ["stable", "unstable", "depressed", "anxious", "suicidal"]
        # self.social_wellbeing = ["connected", "lonely", "isolated", "disconnected"]
        # self.happiness = ["happy", "content", "sad", "depressed", "suicidal"]
        # self.shelter = ["stable", "unstable", "homeless"]
        # self.stability = ["stable", "unstable"]
        # self.luxury = ["luxurious", "comfortable", "uncomfortable", "unlivable"]
        # self.hope = ["hopeful", "hopeless"]
        # self.success = ["successful", "unsuccessful"]
        # self.control = ["in control", "out of control"]
        # self.job_performance = ["good", "bad"]
        # self.beauty = ["beautiful", "ugly"]
        # self.community = ["connected", "disconnected"]
        # self.material_goods = ["plentiful", "scarce"]
        # self.friendship_grid = ["connected", "disconnected"]

    def get_job_adjective(self, job):
        return random.choice(self.job_adjective.get(job, self.job_adjective["default"]))

    def get_job_pronoun(self, job):
        return random.choice(self.job_pronoun.get(job, self.job_pronoun["default"]))

    def get_job_enjoys_verb(self, job):
        return random.choice(
            self.job_enjoys_verb.get(job, self.job_enjoys_verb["default"])
        )

    def get_job_verb_acts_on_noun(self, job):
        return random.choice(
            self.job_verb_acts_on_noun.get(job, self.job_verb_acts_on_noun["default"])
        )

    def get_job_currently_working_on(self, job):
        return random.choice(
            self.job_currently_working_on.get(
                job, self.job_currently_working_on["default"]
            )
        )

    def get_job_place(self, job):
        return random.choice(self.job_place.get(job, self.job_place["default"]))

    def get_job_planning_to_attend(self, job):
        return random.choice(
            self.job_planning_to_attend.get(job, self.job_planning_to_attend["default"])
        )

    def get_job_hoping_to_there(self, job):
        return random.choice(
            self.job_hoping_to_there.get(job, self.job_hoping_to_there["default"])
        )

    def get_job_hoping_to_learn(self, job):
        return random.choice(
            self.job_hoping_to_learn.get(job, self.job_hoping_to_learn["default"])
        )

    def get_job_hoping_to_meet(self, job):
        return random.choice(
            self.job_hoping_to_meet.get(job, self.job_hoping_to_meet["default"])
        )

    def get_job_hoping_to_find(self, job):
        return random.choice(
            self.job_hoping_to_find.get(job, self.job_hoping_to_find["default"])
        )

    def get_feeling_health(self, health_status):
        return random.choice(
            self.feeling_health.get(health_status, self.feeling_health["default"])
        )

    def get_feeling_hunger(self, hunger_level):
        return random.choice(
            self.feeling_hunger.get(hunger_level, self.feeling_hunger["default"])
        )

    def get_event_recent(self, recent_event):
        return random.choice(
            self.event_recent.get(recent_event, self.event_recent["default"])
        )

    def get_financial_situation(self, wealth_money):
        return random.choice(
            self.financial_situation.get(
                wealth_money, self.financial_situation["default"]
            )
        )

    def get_motivation(self, motivation=None):
        """Return a motivational phrase.

        If ``motivation`` is ``None`` or not found in the matrix, a random
        choice from the ``"default"`` list is returned.
        """
        return random.choice(
            self.motivation.get(motivation, self.motivation["default"])
        )

    def get_motivation_zero(self, motivation, job):
        return (
            random.choice(self.motivation.get(motivation, self.motivation["default"]))
            + random.choice(
                self.job_enjoys_verb.get(job, self.job_enjoys_verb["default"])
            )
            + random.choice(
                self.job_verb_acts_on_noun.get(
                    job, self.job_verb_acts_on_noun["default"]
                )
            )
        )

    def get_weather_description(self, weather_description):
        return random.choice(
            self.weather_description.get(
                weather_description, self.weather_description["default"]
            )
        )

    def get_routine_question_framing(self, routine_question_framing=None):
        """Return a question framing string for routine prompts."""
        return random.choice(
            self.routine_question_framing.get(
                routine_question_framing, self.routine_question_framing["default"]
            )
        )

    def get_action_descriptors(self, action):
        return random.choice(
            self.action_descriptors.get(action, self.action_descriptors["default"])
        )


descriptors = DescriptorMatrices()


class PromptBuilder:
    """Build detailed prompts for Tiny Village characters."""

    def __init__(self, character: tc.Character) -> None:
        """Initialize the builder for ``character``."""

        self.character = character
        self.action_options = ActionOptions()
        self.needs_priorities_func = NeedsPriorities()

    def calculate_needs_priorities(self) -> None:
        """Compute and store the character's current need priorities."""

        self.needs_priorities = self.needs_priorities_func.calculate_needs_priorities(
            self.character
        )

    def prioritize_actions(self) -> List[str]:
        """Query the planning system for top actions and build choice strings."""

        try:
            from tiny_strategy_manager import StrategyManager
            from tiny_utility_functions import calculate_action_utility
        except ImportError:  # pragma: no cover - gracefully handle missing deps
            self.prioritized_actions = []
            self.action_choices = []
            return []

        manager = StrategyManager(use_llm=False)
        actions = manager.get_daily_actions(self.character)

        self.prioritized_actions = actions
        self.action_choices = []
        char_state = self._get_character_state_dict()
        current_goal = (
            self.character.get_current_goal()
            if hasattr(self.character, "get_current_goal")
            else None
        )

        for i, action in enumerate(actions[:5]):
            try:
                util = calculate_action_utility(char_state, action, current_goal)
            except (ValueError, TypeError):  # Replace with specific exceptions
                util = 0.0
                print(f"Error calculating utility for action {action}: {e}")  # Optional logging

            effects_str = ""
            if hasattr(action, "effects") and action.effects:
                parts = [
                    f"{eff.get('attribute', '')}: {eff.get('change_value', 0):+.1f}"
                    for eff in action.effects
                    if eff.get("attribute")
                ]
                if parts:
                    effects_str = f" - Effects: {', '.join(parts)}"

            desc = getattr(action, "description", getattr(action, "name", str(action)))
            choice = f"{i+1}. {desc} (Utility: {util:.1f}){effects_str}"
            self.action_choices.append(choice)

        return self.action_choices

    def generate_completion_message(self, character: tc.Character, action: str) -> str:
        """Return a short message describing successful completion of ``action``."""

        return f"{character.name} has {DescriptorMatrices.get_action_descriptors(action)} {action}."

    def generate_failure_message(self, character: tc.Character, action: str) -> str:
        """Return a short message describing failure to perform ``action``."""

        return f"{character.name} has failed to {DescriptorMatrices.get_action_descriptors(action)} {action}."

    def _get_character_state_dict(self) -> Dict[str, float]:
        """Return a simplified state dictionary for utility calculations."""

        state = {
            "hunger": getattr(self.character, "hunger_level", 5.0) / 10.0,
            "energy": getattr(self.character, "energy", 5.0) / 10.0,
            "health": getattr(self.character, "health_status", 5.0) / 10.0,
            "mental_health": getattr(self.character, "mental_health", 5.0) / 10.0,
            "social_wellbeing": getattr(self.character, "social_wellbeing", 5.0)
            / 10.0,
            "money": float(getattr(self.character, "wealth_money", 0.0)),
        }
        return state

    def calculate_action_utility(self, current_goal: Optional[object] = None) -> Dict[str, float]:
        """Calculate and return utility values for the prioritized actions."""
        from tiny_utility_functions import UtilityEvaluator, calculate_action_utility
        from tiny_output_interpreter import OutputInterpreter

        self.action_utilities = {}
        evaluator = UtilityEvaluator()
        char_state = self._get_character_state_dict()
        interpreter = OutputInterpreter()

        for action_name in self.prioritized_actions:
            try:
                action_cls = interpreter.action_class_map.get(action_name)
                action_obj = action_cls() if action_cls else None
            except Exception as e:
                print(f"Error creating action {action_name}: {e}")
                continue

            if not action_obj:
                print(f"Warning: Unknown action {action_name}")
                continue

            try:
                utility = evaluator.evaluate_action_utility(
                    self.character.name,
                    char_state,
                    action_obj,
                    current_goal,
                )
            except Exception:
                try:
                    utility = calculate_action_utility(char_state, action_obj, current_goal)
                except Exception as e:
                    print(f"Failed to evaluate utility for {action_name}: {e}")
                    continue

            self.action_utilities[action_name] = utility

        return self.action_utilities

    def generate_daily_routine_prompt(self, time: str, weather: str) -> str:
        """Generate a basic daily routine prompt."""
        prompt = "<|system|>"
        prompt += (
            f"You are {self.character.name}, a {self.character.job} in a small town. You are a {descriptors.get_job_adjective(self.character.job)} {descriptors.get_job_pronoun(self.character.job)} who enjoys {descriptors.get_job_enjoys_verb(self.character.job)} {descriptors.get_job_verb_acts_on_noun(self.character.job)}. You are currently working on {descriptors.get_job_currently_working_on(self.character.job)} {descriptors.get_job_place(self.character.job)}, and you are excited to see how it turns out. You are also planning to attend a {descriptors.get_job_planning_to_attend(self.character.job)} in the next few weeks, and you are hoping to {descriptors.get_job_hoping_to_there(self.character.job)} there.",
        )
        prompt += f"<|user|>"
        prompt += f"{self.character.name}, it's {time}, and {descriptors.get_weather_description(weather)}. You're feeling {descriptors.get_feeling_health(self.character.health_status)}, and {descriptors.get_feeling_hunger(self.character.hunger_level)}. "
        prompt += f"{descriptors.get_event_recent(self.character.recent_event)}, and {descriptors.get_financial_situation(self.character.wealth_money)}. {descriptors.get_motivation()} {self.long_term_goal}. {descriptors.get_routine_question_framing()}"
        prompt += "Options:\n"
        prompt += "1. Go to the market to Buy_Food.\n"
        prompt += f"2. Work at your job to Improve_{self.job_performance}.\n"
        prompt += "3. Visit a friend to Increase_Friendship.\n"
        prompt += "4. Engage in a Leisure_Activity to improve Mental_Health.\n"
        prompt += "5. Work on a personal project to Pursue_Hobby.\n"
        prompt += "</s>"
        prompt += "<|assistant|>"
        prompt += f"{self.character.name}, I choose "
        return prompt

    def generate_decision_prompt(
        self,
        time: str,
        weather: str,
        action_choices: List[str],
        character_state_dict: Optional[Dict[str, float]] = None,
        memories: Optional[List] = None,
    ) -> str:
        """Create a decision prompt incorporating goals, needs and context."""
        # Calculate needs priorities for character context
        needs_calculator = NeedsPriorities()
        needs_priorities = needs_calculator.calculate_needs_priorities(self.character)

        # Get character's current goals prioritized by importance
        try:
            goal_queue = self.character.evaluate_goals()
        except Exception as e:
            print(f"Warning: Could not evaluate goals for {self.character.name}: {e}")
            goal_queue = []

        # Build enhanced prompt with rich character context
        prompt = f"<|system|>"

        # Basic character identity and role
        prompt += (
            f"You are {self.character.name}, a {self.character.job} in a small town. "
        )
        prompt += f"You are a {descriptors.get_job_adjective(self.character.job)} {descriptors.get_job_pronoun(self.character.job)} "
        prompt += f"who enjoys {descriptors.get_job_enjoys_verb(self.character.job)} {descriptors.get_job_verb_acts_on_noun(self.character.job)}. "

        # Current goals and motivations
        if goal_queue and len(goal_queue) > 0:
            prompt += f"\n\nYour current goals (in order of importance):\n"
            for i, (utility_score, goal) in enumerate(goal_queue[:3]):  # Top 3 goals
                prompt += f"{i+1}. {goal.name}: {goal.description} (Priority: {utility_score:.1f})\n"

        # Character's pressing needs and motivations
        top_needs = sorted(needs_priorities.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        if top_needs:
            prompt += f"\nYour most pressing needs:\n"
            for need_name, priority_score in top_needs:
                need_desc = self._get_need_description(need_name, priority_score)
                prompt += f"- {need_desc}\n"

        # Character motives and personality context
        if hasattr(self.character, "motives") and self.character.motives:
            prompt += f"\nYour key motivations:\n"
            top_motives = self._get_top_motives(self.character.motives, 4)
            for motive_name, motive_score in top_motives:
                prompt += f"- {motive_name.replace('_', ' ').title()}: {self._get_motive_description(motive_name, motive_score)}\n"

        # Current comprehensive state
        prompt += f"\n<|user|>"
        prompt += f"{self.character.name}, it's {time}, and {descriptors.get_weather_description(weather)}. "

        # Enhanced status description
        prompt += f"Current state: "
        prompt += f"Health {self.character.health_status}/10, "
        prompt += f"Hunger {self.character.hunger_level}/10, "
        prompt += f"Energy {getattr(self.character, 'energy', 5):.1f}/10, "
        prompt += f"Mental Health {self.character.mental_health}/10, "
        prompt += f"Social Wellbeing {self.character.social_wellbeing}/10. "

        # Financial and life context
        prompt += f"{descriptors.get_event_recent(self.character.recent_event)}, and {descriptors.get_financial_situation(self.character.wealth_money)}. "

        # Long-term aspiration context
        if hasattr(self.character, "long_term_goal") and self.character.long_term_goal:
            prompt += f"Your long-term aspiration is: {self.character.long_term_goal}. "

 
        # Include short memory descriptions if provided
        if memories:
            prompt += "\nRecent memories influencing you:\n"
            for mem in memories[:2]:
                desc = getattr(mem, "description", str(mem))
                prompt += f"- {desc}\n"

        # Include any additional character state provided
        if isinstance(character_state_dict, dict):
            prompt += "\nAdditional state:\n"
            for key, value in character_state_dict.items():
                formatted_key = key.replace("_", " ").title()
                prompt += f"- {formatted_key}: {value}\n"
        elif character_state_dict is not None:
            raise TypeError("character_state_dict must be a dictionary.")
 

        prompt += f"\n{descriptors.get_routine_question_framing()}"

        # Enhanced action choices with better formatting
        prompt += f"\nAvailable actions:\n"
        for i, action_choice in enumerate(action_choices):
            prompt += f"{action_choice}\n"

        prompt += f"\nChoose the action that best aligns with your goals, needs, and current situation. "
        prompt += f"Consider both immediate benefits and long-term progress toward your aspirations."

        prompt += f"\n</s>"
        prompt += f"<|assistant|>"
        prompt += f"{self.character.name}, I choose "
        return prompt

    def _get_need_description(self, need_name: str, priority_score: float) -> str:
        """Generate human-readable description for character needs."""
        need_descriptions = {
            "health": f"Physical health needs attention (priority: {priority_score:.0f}/100)",
            "hunger": f"Nutritional needs are pressing (priority: {priority_score:.0f}/100)",
            "wealth": f"Financial security is important (priority: {priority_score:.0f}/100)",
            "mental_health": f"Mental wellness requires care (priority: {priority_score:.0f}/100)",
            "social_wellbeing": f"Social connections need nurturing (priority: {priority_score:.0f}/100)",
            "happiness": f"Personal happiness and fulfillment (priority: {priority_score:.0f}/100)",
            "shelter": f"Housing and shelter security (priority: {priority_score:.0f}/100)",
            "stability": f"Life stability and routine (priority: {priority_score:.0f}/100)",
            "luxury": f"Comfort and luxury desires (priority: {priority_score:.0f}/100)",
            "hope": f"Optimism and future outlook (priority: {priority_score:.0f}/100)",
            "success": f"Achievement and success drive (priority: {priority_score:.0f}/100)",
            "control": f"Sense of control and agency (priority: {priority_score:.0f}/100)",
            "job_performance": f"Professional excellence (priority: {priority_score:.0f}/100)",
            "beauty": f"Aesthetic and beauty appreciation (priority: {priority_score:.0f}/100)",
            "community": f"Community involvement and belonging (priority: {priority_score:.0f}/100)",
            "material_goods": f"Material possessions and acquisitions (priority: {priority_score:.0f}/100)",
        }
        return need_descriptions.get(
            need_name,
            f"{need_name.replace('_', ' ').title()} (priority: {priority_score:.0f}/100)",
        )

    def _get_top_motives(self, motives: object, count: int = 4) -> List[tuple]:
        """Get the top character motives by score."""
        try:
            motive_dict = motives.to_dict()
            motive_scores = [
                (name, motive.score) for name, motive in motive_dict.items()
            ]
            return sorted(motive_scores, key=lambda x: x[1], reverse=True)[:count]
        except Exception as e:
            print(f"Warning: Could not extract motives: {e}")
            return []

    def _get_motive_description(self, motive_name: str, score: float) -> str:
        """Generate human-readable description for character motives."""
        intensity = (
            "Very High"
            if score >= 8
            else "High" if score >= 6 else "Moderate" if score >= 4 else "Low"
        )
        return f"{intensity} ({score:.1f}/10)"


    def generate_crisis_response_prompt(self, crisis_description: str, urgency: str = "high"):
        """Generate a short crisis response prompt for the LLM.

        Parameters
        ----------
        crisis_description : str
            Description of the crisis situation.
        urgency : str, optional
            Qualitative urgency indicator (e.g. "low", "medium", "high").
        """

        prompt = "<|system|>"
        prompt += (
            f"You are {self.character.name}, a {descriptors.get_job_adjective(self.character.job)} "
            f"{descriptors.get_job_pronoun(self.character.job)} prepared for emergencies."
        )

        prompt += "<|user|>"
        prompt += (
            f"A crisis has occurred: {crisis_description}. Urgency: {urgency}. "
            f"{descriptors.get_event_recent(self.character.recent_event)} "
            f"{descriptors.get_financial_situation(self.character.wealth_money)}."
        )

        try:
            from tiny_utility_functions import UtilityEvaluator
            from tiny_output_interpreter import OutputInterpreter

            evaluator = UtilityEvaluator()
            interpreter = OutputInterpreter()
            actions = ActionOptions().prioritize_actions(self.character)
            action_objects = []
            for name in actions[:1]:
                cls = interpreter.action_class_map.get(name)
                if cls:
                    action_objects.append(cls())

            if action_objects:
                char_state = self._get_character_state_dict()
                _, analysis = evaluator.evaluate_plan_utility_advanced(
                    self.character.name, char_state, action_objects
                )
                breakdown = analysis.get("action_breakdown")
                if breakdown:
                    best_action = breakdown[0].get("action")
                    if best_action:
                        prompt += f" Recommended immediate action: {best_action}."
        except ImportError:
            # Utility evaluation is optional and may not work if dependencies are missing
            pass
        except Exception as e:
            # Log unexpected exceptions for debugging purposes
            import logging
            logging.error(f"An unexpected error occurred during utility evaluation: {e}")

        prompt += "<|assistant|>"
        return prompt
