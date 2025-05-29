import logging
import unittest
import unittest.util
import networkx as nx
from datetime import datetime

from actions import Action, State, ActionSystem

action_system = ActionSystem()
from tiny_graph_manager import GraphManager
from tiny_types import Character, Location, Event
from tiny_items import FoodItem, ItemInventory, ItemObject
from tiny_jobs import Job
from tiny_time_manager import GameTimeManager as tiny_time_manager

logging.basicConfig(level=logging.DEBUG)

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
    "Coding": [
        {
            "name": "energy",
            "attribute": "energy",
            "target": "initiator",
            "satisfy_value": 10,
            "operator": "gt",
        },
        {
            "name": "focus",
            "attribute": "mental_health",
            "target": "initiator",
            "satisfy_value": 30,
            "operator": "gt",
        },
    ],
    "Jogging": [
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
    "Reading": [
        {
            "name": "energy",
            "attribute": "energy",
            "target": "initiator",
            "satisfy_value": 10,
            "operator": "gt",
        },
        {
            "name": "openness",
            "attribute": "personality_traits.openness",
            "target": "initiator",
            "satisfy_value": 50,
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
    "Coding": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
        {"targets": ["initiator"], "attribute": "skills.coding", "change_value": 5},
    ],
    "Jogging": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
        {"targets": ["initiator"], "attribute": "health_status", "change_value": 5},
    ],
    "Reading": [
        {"targets": ["initiator"], "attribute": "energy", "change_value": -5},
        {"targets": ["initiator"], "attribute": "skills.knowledge", "change_value": 5},
    ],
}


class TestGraphManager(unittest.TestCase):

    def setUp(self):
        self.graph_manager = GraphManager()

    def test_initialize_graph(self):
        self.assertIsInstance(self.graph_manager.G, nx.MultiDiGraph)
        # logging.debug(self.graph_manager.G.nodes)
        # self.assertEqual(len(self.graph_manager.G.nodes), 0)
        # self.assertEqual(len(self.graph_manager.G.edges), 0)

    def test_add_character_node(self):
        from tiny_locations import Location
        from tiny_characters import Character

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        self.assertIn(char, self.graph_manager.G.nodes)

    def test_add_location_node(self):
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_location_node(loc)
        self.assertIn(loc, self.graph_manager.G.nodes)

    def test_add_event_node(self):
        from tiny_event_handler import Event
        from tiny_locations import Location

        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_event_node(event)
        self.assertIn(event, self.graph_manager.G.nodes)

    def test_add_object_node(self):
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_object_node(obj)
        logging.debug(self.graph_manager.G.nodes)
        self.assertIn(obj, self.graph_manager.G.nodes)

    def test_add_activity_node(self):
        from actions import Action, ActionSystem

        act = Action(
            name="Jogging",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        self.graph_manager.add_activity_node(act)
        self.assertIn(act.name, self.graph_manager.G.nodes)

    def test_add_job_node(self):
        from tiny_jobs import Job
        from tiny_locations import Location

        job = Job(
            job_name="Software Developer",
            job_description="Develop software",
            job_salary=5000,
            job_skills=["Coding"],
            job_education="Bachelor's degree",
            req_job_experience="2 years",
            job_motives=["Money", "Passion"],
            job_title="Software Developer",
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_job_node(job)
        self.assertIn(job, self.graph_manager.G.nodes)

    def test_add_character_character_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        self.assertTrue(self.graph_manager.G.has_edge(char1, char2))

    def test_add_character_location_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_character_location_edge(
            char, loc, 10, datetime.now(), ["Jogging"], "full"
        )
        self.assertTrue(self.graph_manager.G.has_edge(char, loc))

    def test_add_character_object_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_character_object_edge(
            char, obj, True, 10, 5, datetime.now()
        )
        self.assertTrue(self.graph_manager.G.has_edge(char, obj))

    def test_add_character_event_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location
        from tiny_event_handler import Event

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_event_node(event)
        self.graph_manager.add_character_event_edge(
            char,
            event,
            True,
            "Organizer",
            {"short_term": 5, "long_term": 10},
            3,
        )
        self.assertTrue(self.graph_manager.G.has_edge(char, event))

    def test_add_character_activity_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location
        from actions import Action, ActionSystem

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        act = Action(
            name="Jogging",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_activity_node(act)
        self.graph_manager.add_character_activity_edge(char, act, 5, 3, 10, 2)
        self.assertTrue(self.graph_manager.G.has_edge(char, act.name))

    def test_add_location_location_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        loc1 = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        loc2 = Location(
            name="Cafe",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_location_node(loc1)
        self.graph_manager.add_location_node(loc2)
        self.graph_manager.add_location_location_edge(
            loc1, loc2, 10, 5, 3, {"volume": 10, "value": 5}
        )
        self.assertTrue(self.graph_manager.G.has_edge(loc1, loc2))

    def test_add_location_item_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_location_item_edge(
            loc, obj, 5, {"primary": 10, "secondary": 5}
        )
        self.assertTrue(self.graph_manager.G.has_edge(loc, obj))

    def test_add_location_event_edge(self):
        from tiny_locations import Location
        from tiny_event_handler import Event

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_event_node(event)
        self.graph_manager.add_location_event_edge(
            loc,
            event,
            {"frequency": 5, "predictability": 3},
            "Main Venue",
            100,
            5,
        )
        self.assertTrue(self.graph_manager.G.has_edge(loc, event))

    def test_add_location_activity_edge(self):
        from tiny_locations import Location
        from actions import Action, ActionSystem

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        act = Action(
            name="Jogging",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_activity_node(act)
        self.graph_manager.add_location_activity_edge(loc, act, 5, 3, 2)
        self.assertTrue(self.graph_manager.G.has_edge(loc.name, act.name))

    def test_add_item_item_edge(self):

        obj1 = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        obj2 = ItemObject(
            name="Shield",
            item_type="Armor",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_object_node(obj1)
        self.graph_manager.add_object_node(obj2)
        self.graph_manager.add_item_item_edge(
            obj1, obj2, {"functional": 10, "aesthetic": 5}, 3, 2
        )
        self.assertTrue(self.graph_manager.G.has_edge(obj1, obj2))

    def test_add_item_activity_edge(self):
        from actions import Action, ActionSystem

        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        act = Action(
            name="Jogging",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_activity_node(act)
        self.graph_manager.add_item_activity_edge(obj, act.name, 5, 3, 2)
        self.assertTrue(self.graph_manager.G.has_edge(obj, act.name))

    def test_add_event_activity_edge(self):
        from actions import Action, ActionSystem
        from tiny_event_handler import Event
        from tiny_locations import Location

        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        act = Action(
            name="Jogging",
            preconditions=event.action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        self.graph_manager.add_event_node(event)
        self.graph_manager.add_activity_node(act)
        self.graph_manager.add_event_activity_edge(
            event.name,
            act.name,
            {"immediate": 5, "long_term": 3},
            {"direct": 2, "indirect": 1},
            4,
        )
        self.assertTrue(self.graph_manager.G.has_edge(event.name, act.name))

    def test_add_event_item_edge(self):
        from tiny_event_handler import Event
        from tiny_locations import Location

        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_event_node(event)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_event_item_edge(event.name, obj, 5, 3, 2)
        self.assertTrue(self.graph_manager.G.has_edge(event.name, obj))

    def test_add_activity_activity_edge(self):
        from actions import Action, ActionSystem

        act1 = Action(
            name="Jogging",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Jogging"]
            ),
            effects=effect_dict["Jogging"],
        )
        act2 = Action(
            name="Reading",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Reading"]
            ),
            effects=effect_dict["Reading"],
        )
        self.graph_manager.add_activity_node(act1)
        self.graph_manager.add_activity_node(act2)
        self.graph_manager.add_activity_activity_edge(act1.name, act2.name, 5, 3, 2)
        self.assertTrue(self.graph_manager.G.has_edge(act1.name, act2.name))

    def test_add_character_job_edge(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        job = Job(
            job_name="Software Developer",
            job_description="Develop software",
            job_salary=5000,
            job_skills=["Coding"],
            job_education="Bachelor's degree",
            req_job_experience="2 years",
            job_motives=["Money", "Passion"],
            job_title="Software Developer",
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_job_node(job)
        self.graph_manager.add_character_job_edge(
            char, job.name, "Developer", "Active", 5
        )
        self.assertTrue(self.graph_manager.G.has_edge(char, job.name))

    def test_add_job_location_edge(self):
        from tiny_locations import Location

        job = Job(
            job_name="Software Developer",
            job_description="Develop software",
            job_salary=5000,
            job_skills=["Coding"],
            job_education="Bachelor's degree",
            req_job_experience="2 years",
            job_motives=["Money", "Passion"],
            job_title="Software Developer",
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        loc = Location(
            name="Office", x=0, y=0, width=1, height=1, action_system=action_system
        )
        self.graph_manager.add_job_node(job)
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_job_location_edge(job, loc, True, 5)
        self.assertTrue(self.graph_manager.G.has_edge(job, loc))

    def test_add_job_activity_edge(self):
        from actions import Action, ActionSystem
        from tiny_jobs import Job
        from tiny_locations import Location

        job = Job(
            job_name="Software Developer",
            job_description="Develop software",
            job_salary=5000,
            job_skills=["Coding"],
            job_education="Bachelor's degree",
            req_job_experience="2 years",
            job_motives=["Money", "Passion"],
            job_title="Software Developer",
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        act = Action(
            name="Coding",
            preconditions=action_system.instantiate_conditions(
                preconditions_dict["Coding"]
            ),
            effects=effect_dict["Coding"],
        )
        self.graph_manager.add_job_node(job)
        self.graph_manager.add_activity_node(act)
        self.graph_manager.add_job_activity_edge(job.name, act.name, 5, 3)
        self.assertTrue(self.graph_manager.G.has_edge(job.name, act.name))

    def test_find_shortest_path(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        path = self.graph_manager.find_shortest_path(char1, char2)
        self.assertIsNotNone(path)
        self.assertEqual(path, [char1, char2])

    # def test_detect_communities(self):
    #     from tiny_characters import Character
    #     from tiny_locations import Location

    #     char1 = Character(
    #         name="Alice",
    #         age=25,
    #         pronouns="she/her",
    #         job="Waitress",
    #         health_status=10,
    #         hunger_level=2,
    #         wealth_money=10,
    #         mental_health=8,
    #         social_wellbeing=8,
    #         job_performance=20,
    #         community=5,
    #         friendship_grid=[],
    #         recent_event="new job",
    #         long_term_goal="Save for college",
    #         personality_traits={
    #             "extraversion": 50,
    #             "openness": 80,
    #             "conscientiousness": 70,
    #             "agreeableness": 60,
    #             "neuroticism": 30,
    #         },
    #         action_system=action_system,
    #         gametime_manager=tiny_time_manager,
    #         location=Location("Alice", 0, 0, 1, 1, action_system),
    #         graph_manager=self.graph_manager,
    #     )
    #     char2 = Character(
    #         name="Bob",
    #         age=25,
    #         pronouns="he/him",
    #         job="Mayor",
    #         health_status=10,
    #         hunger_level=2,
    #         wealth_money=10,
    #         mental_health=8,
    #         social_wellbeing=8,
    #         job_performance=20,
    #         community=5,
    #         friendship_grid=[],
    #         recent_event="new job",
    #         long_term_goal="Save for college",
    #         personality_traits={
    #             "extraversion": 50,
    #             "openness": 80,
    #             "conscientiousness": 70,
    #             "agreeableness": 60,
    #             "neuroticism": 30,
    #         },
    #         action_system=action_system,
    #         gametime_manager=tiny_time_manager,
    #         location=Location("Bob", 0, 0, 1, 1, action_system),
    #         graph_manager=self.graph_manager,
    #     )
    #     self.graph_manager.add_character_node(char1)
    #     self.graph_manager.add_character_node(char2)
    #     self.graph_manager.add_character_character_edge(
    #         char1, char2, "friend", 10, 5, 5, 4
    #     )
    #     communities = self.graph_manager.detect_communities()
    #     self.assertIsNotNone(communities)

    def test_calculate_centrality(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        centrality = self.graph_manager.calculate_centrality()
        self.assertIn(char1, centrality)

    def test_shortest_path_between_characters(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        path = self.graph_manager.shortest_path_between_characters(char1, char2)
        self.assertIsNotNone(path)
        self.assertEqual(path, [char1, char2])

    # def test_common_interests_cluster(self):
    #     from tiny_characters import Character

    #     char1 = Character(name="Alice", age=25, job="Engineer", interests=["Jogging"])
    #     char2 = Character(name="Bob", age=30, job="Doctor", interests=["Jogging"])
    #     self.graph_manager.add_character_node(char1)
    #     self.graph_manager.add_character_node(char2)
    #     clusters = self.graph_manager.common_interests_cluster()
    #     self.assertIsNotNone(clusters)

    # def test_most_influential_character(self):
    #     from tiny_characters import Character

    #     char1 = Character(
    #         name="Alice",
    #         age=25,
    #         pronouns="she/her",
    #         job="Waitress",
    #         health_status=10,
    #         hunger_level=2,
    #         wealth_money=10,
    #         mental_health=8,
    #         social_wellbeing=8,
    #         job_performance=20,
    #         community=5,
    #         friendship_grid=[],
    #         recent_event="new job",
    #         long_term_goal="Save for college",
    #         personality_traits={
    #             "extraversion": 50,
    #             "openness": 80,
    #             "conscientiousness": 70,
    #             "agreeableness": 60,
    #             "neuroticism": 30,
    #         },
    #         action_system=action_system,
    #         gametime_manager=tiny_time_manager,
    #         location=Location("Alice", 0, 0, 1, 1, action_system),
    #         graph_manager=self.graph_manager,
    #     )
    #     char2 = Character(
    #         name="Bob",
    #         age=25,
    #         pronouns="he/him",
    #         job="Mayor",
    #         health_status=10,
    #         hunger_level=2,
    #         wealth_money=10,
    #         mental_health=8,
    #         social_wellbeing=8,
    #         job_performance=20,
    #         community=5,
    #         friendship_grid=[],
    #         recent_event="new job",
    #         long_term_goal="Save for college",
    #         personality_traits={
    #             "extraversion": 50,
    #             "openness": 80,
    #             "conscientiousness": 70,
    #             "agreeableness": 60,
    #             "neuroticism": 30,
    #         },
    #         action_system=action_system,
    #         gametime_manager=tiny_time_manager,
    #         location=Location("Bob", 0, 0, 1, 1, action_system),
    #         graph_manager=self.graph_manager,
    #     )
    #     self.graph_manager.add_character_node(char1)
    #     self.graph_manager.add_character_node(char2)
    #     self.graph_manager.add_character_character_edge(
    #         char1, char2, "friend", 10,5, 5, 3, 4
    #     )
    #     influencer = self.graph_manager.most_influential_character()
    #     self.assertIn(influencer, [char1, char2])

    def test_update_node_attribute(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.update_node_attribute(char, "mood", "happy")
        self.assertEqual(self.graph_manager.G.nodes[char]["mood"], "happy")

    def test_update_edge_attribute(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        self.graph_manager.update_edge_attribute(char1, char2, "trust", 75)
        self.assertEqual(self.graph_manager.G[char1][char2][0]["trust"], 75)

    def test_analyze_location_popularity(self):
        from tiny_characters import Character
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_location_node(loc)
        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_character_location_edge(
            char, loc, 10, datetime.now(), ["Jogging"], "full"
        )
        popularity = self.graph_manager.analyze_location_popularity()
        self.assertIn(loc, popularity)

    def test_transfer_item_ownership(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 3, 3, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_character_object_edge(
            char, obj, True, 10, 5, datetime.now()
        )
        self.graph_manager.transfer_item_ownership(obj, char, char2)
        owner = self.graph_manager.determine_owner(obj)
        self.assertIsNotNone(owner)
        self.assertEqual(owner, char2)

    def test_transfer_item_ownership_b(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_character_object_edge(
            char1, obj, True, 10, 5, datetime.now()
        )
        self.graph_manager.transfer_item_ownership(obj, char1, char2)
        self.assertTrue(self.graph_manager.G.has_edge(char2, obj))

    def test_analyze_character_relationships(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(char1, char2, 1, 1, 1, 1)
        relationships = self.graph_manager.analyze_character_relationships(char1)
        logging.info(relationships)
        self.assertIn(char2, relationships)

    def test_location_popularity_analysis(self):
        from tiny_characters import Character
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_character_location_edge(
            char, loc, 10, datetime.now(), ["Jogging"], "full"
        )
        popularity = self.graph_manager.location_popularity_analysis()
        self.assertIn(loc, popularity)

    def test_track_item_ownership(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_character_object_edge(
            char, obj, True, 10, 5, datetime.now()
        )
        owner = self.graph_manager.determine_owner(obj)
        self.assertIsNotNone(owner)
        self.assertIn(char, owner)

    def test_predict_future_relationships(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        predictions = self.graph_manager.predict_future_relationships(char)
        self.assertIn("info", predictions)

    def test_update_node_attribute(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.update_node_attribute(char, "mood", "happy")
        self.assertEqual(self.graph_manager.G.nodes[char]["mood"], "happy")

    def test_update_edge_attribute(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 3, 4, 4
        )
        self.graph_manager.update_edge_attribute(char1, char2, "trust", 75)
        self.assertEqual(self.graph_manager.G[char1][char2][0]["trust"], 75)

    def test_evaluate_relationship_strength(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        strength = self.graph_manager.evaluate_relationship_strength(char1, char2)
        self.assertGreater(strength, 0)

    def test_check_friendship_status(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        status = self.graph_manager.check_friendship_status(char1, char2)
        self.assertEqual(status, "friends")

    def test_character_location_frequency(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_location_node(loc)
        self.graph_manager.add_character_location_edge(
            char, loc, 10, datetime.now(), ["Jogging"], "full"
        )
        frequency = self.graph_manager.character_location_frequency(char)
        self.assertIn(loc, frequency)

    def test_location_popularity(self):
        from tiny_characters import Character
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_location_node(loc)
        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_character_location_edge(
            char, loc, 10, datetime.now(), ["Jogging"], "full"
        )
        popularity = self.graph_manager.location_popularity(loc)
        self.assertGreater(popularity, 0)

    def test_item_ownership_history(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            value=10,
            weight=5,
            quantity=1,
            description="test",
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        self.graph_manager.add_character_object_edge(
            char, obj, True, 10, 5, datetime.now()
        )
        history = self.graph_manager.item_ownership_history(obj)
        self.assertGreater(len(history), 0)

    def test_can_interact_directly(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        conditions = {"relationship_type": "friend"}
        can_interact = self.graph_manager.can_interact_directly(
            char1, char2, conditions
        )
        self.assertTrue(can_interact)

    def test_get_nearest_resource(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 5, -5, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = FoodItem(
            name="Apple",
            value=10,
            weight=5,
            quantity=1,
            description="Red apple",
            coordinates_location=(2, -2),
            calories=5,
            perishable=True,
            action_system=action_system,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        resource_filter = {"item_type": "Food"}

        nearest_resource = self.graph_manager.get_nearest_resource(
            char, resource_filter
        )
        self.assertIsNotNone(nearest_resource)
        self.assertEqual(nearest_resource[0], obj.name)

    def test_track_event_participation(self):
        from tiny_characters import Character
        from tiny_locations import Location
        from tiny_event_handler import Event

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        event = Event(
            name="Festival",
            event_type="Cultural",
            date="2024-07-28",
            importance=5,
            impact=3,
            location=Location(
                name="Park",
                x=0,
                y=0,
                width=1,
                height=1,
                action_system=action_system,
                security=5,
                threat_level=0,
                popularity=5,
            ),
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_event_node(event)
        self.graph_manager.track_event_participation(char, event)
        self.assertTrue(self.graph_manager.G.has_edge(char, event))

    def test_check_safety_of_locations(self):
        from tiny_locations import Location

        loc = Location(
            name="Park",
            x=0,
            y=0,
            width=1,
            height=1,
            action_system=action_system,
            security=5,
            threat_level=0,
            popularity=5,
        )
        self.graph_manager.add_location_node(loc)
        safety_score = self.graph_manager.check_safety_of_locations(loc)
        self.assertGreater(safety_score, 0)

    def test_evaluate_trade_opportunities_by_char_surplus(self):
        from tiny_characters import Character, Goal, Condition
        from tiny_locations import Location
        from tiny_items import FoodItem
        from tiny_prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        example_criteria_d = [
            {
                "node_attributes": {"type": "item", "item_type": "food"},
                "max_distance": 20,
            }
        ]
        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char.add_to_inventory(
            FoodItem(
                "apple",
                "red apple",
                1,
                True,
                1,
                1,
                action_system=action_system,
                calories=5,
            )
        )

        char2.add_new_goal(
            Goal(
                name="Find Food",
                description="Search for food to satisfy hunger.",
                score=5,
                character=char2,
                target=char2,
                completion_conditions={
                    False: [
                        Condition(  # Remember the key is False because the condition is not met yet
                            name="has_food",
                            attribute="inventory.check_has_item_by_type(['food'])",
                            target=char2,
                            satisfy_value=True,
                            op="==",
                            weight=1,  # This is the weight representing the importance of this condition toward the goal. This will be used in a division operation to calculate the overall importance of the goal.
                        )
                    ]
                },
                evaluate_utility_function=char.goap_planner.evaluate_goal_importance,
                difficulty=self.graph_manager.calculate_goal_difficulty,
                completion_reward=self.graph_manager.calculate_reward,
                failure_penalty=self.graph_manager.calculate_penalty,
                completion_message=prompt_builder.generate_completion_message,
                failure_message=prompt_builder.generate_failure_message,
                criteria=example_criteria_d,
                graph_manager=self.graph_manager,
                goal_type="basic",
            )
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_character_node(char2)
        opportunities = self.graph_manager.evaluate_trade_opportunities_by_char_surplus(
            char
        )
        self.assertIn("apple", opportunities)

    def test_evaluate_trade_opportunities_for_item(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        obj = ItemObject(
            name="Sword",
            item_type="Weapon",
            quantity=1,
            description="test",
            value=10,
            weight=5,
        )
        self.graph_manager.add_character_node(char)
        self.graph_manager.add_object_node(obj)
        opportunities = self.graph_manager.evaluate_trade_opportunities_for_item(obj)
        self.assertIn(char, opportunities)

    def test_find_all_paths(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        paths = self.graph_manager.find_all_paths(char1, char2, max_length=2)
        self.assertGreater(len(paths), 0)

    def test_node_influence_spread(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(
            char1, char2, "friend", 10, 5, 5, 4
        )
        influence = self.graph_manager.node_influence_spread(char1)
        self.assertIn(char2, influence)

    def test_analyze_relationship_health(self):
        from tiny_characters import Character
        from tiny_locations import Location

        char1 = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        char2 = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            long_term_goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=action_system,
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, action_system),
            graph_manager=self.graph_manager,
        )
        self.graph_manager.add_character_node(char1)
        self.graph_manager.add_character_node(char2)
        self.graph_manager.add_character_character_edge(char1, char2, 2, 10, 5, 5, 4)
        health_score = self.graph_manager.analyze_relationship_health(char1, char2)
        self.assertGreater(health_score, 0)


if __name__ == "__main__":
    # unittest.main()
    try:
        import torch
        import sys

        cuda_version = torch.version.cuda
        sys.stderr.write(f"CUDA Version: {cuda_version}\n")
    except ImportError:
        print("PyTorch is not installed. Skipping CUDA version check.")
    # exit()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphManager)
    unittest.TextTestRunner(verbosity=2).run(suite)
