import logging
import unittest
import unittest.util
import networkx as nx
from datetime import datetime

from actions import Action, State, ActionSystem
from tiny_graph_manager import GraphManager
graph_mrg_instance_for_tests = GraphManager() # Module-level instance
action_system = ActionSystem(graph_manager=graph_mrg_instance_for_tests) # Pass graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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
            graph_manager=self.graph_manager
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

    def test_calculate_goal_difficulty(self):
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
        result = self.graph_manager.calculate_goal_difficulty(
            {"node_attributes": {"type": "item"}}, char
        )
        self.assertIsInstance(result, dict)
        self.assertIn("difficulty", result)

    def test_calculate_goal_difficulty_extended(self):
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

        class TestMgr(GraphManager):
            def __init__(self):
                super().__init__()
                self.G = nx.Graph()
                self.G.add_node("n1", type="item", item_type="food")
                self.G.add_node("n2", type="item", item_type="tool")
                self.G.add_edge("n1", "n2")

            def get_filtered_nodes(self, **kwargs):
                if kwargs.get("node_attributes", {}).get("item_type") == "food":
                    return {"n1": {}}
                if kwargs.get("node_attributes", {}).get("item_type") == "tool":
                    return {"n2": {}}
                return {}

            def calculate_action_viability_cost(self, node, goal, character):
                if node == "n1":
                    return {
                        "action_cost": {"eat": 1},
                        "viable": {"eat": True},
                        "goal_cost": {"eat": 2},
                        "conditions_fulfilled_by_action": {"eat": ["has_food"]},
                        "actions_that_fulfill_condition": {"has_food": [("eat", node)]},
                    }
                if node == "n2":
                    return {
                        "action_cost": {"use": 2},
                        "viable": {"use": True},
                        "goal_cost": {"use": 3},
                        "conditions_fulfilled_by_action": {"use": ["has_tool"]},
                        "actions_that_fulfill_condition": {"has_tool": [("use", node)]},
                    }
                return {}

            def calculate_edge_cost(self, u, v):
                return 1

        mgr = TestMgr()
        goal = DummyGoal(
            [
                {"node_attributes": {"item_type": "food"}},
                {"node_attributes": {"item_type": "tool"}},
            ]
        )
        character = None
        result = mgr.calculate_goal_difficulty(goal, character)
        self.assertIsInstance(result, dict)
        # Should be able to find a path and difficulty should be > 0
        self.assertGreater(result["difficulty"], 0)

    def test_no_viable_actions(self):
        class TestMgr(GraphManager):
            def __init__(self):
                super().__init__()
                self.G = nx.Graph()
                self.G.add_node("n1", type="item")

            def get_filtered_nodes(self, **kwargs):
                return {"n1": {}}

            def calculate_action_viability_cost(self, node, goal, character):
                return {
                    "action_cost": {"act1": 1},
                    "viable": {"act1": False},
                    "goal_cost": {"act1": 2},
                    "conditions_fulfilled_by_action": {"act1": ["cond1"]},
                    "actions_that_fulfill_condition": {"cond1": [("act1", node)]},
                }

            def calculate_edge_cost(self, u, v):
                return 0

        mgr = TestMgr()
        goal = DummyGoal([{"node_attributes": {"type": "item"}}])
        character = None
        result = mgr.calculate_goal_difficulty(goal, character)
        self.assertEqual(
            result,
            float("inf")
            or (isinstance(result, dict) and result.get("difficulty") == float("inf")),
        )

    def test_multiple_paths_choose_lowest_cost(self):
        class TestMgr(GraphManager):
            def __init__(self):
                super().__init__()
                self.G = nx.Graph()
                self.G.add_node("n1", type="item")
                self.G.add_node("n2", type="item")
                self.G.add_edge("n1", "n2")

            def get_filtered_nodes(self, **kwargs):
                return {"n1": {}, "n2": {}}

            def calculate_action_viability_cost(self, node, goal, character):
                if node == "n1":
                    return {
                        "action_cost": {"a": 1},
                        "viable": {"a": True},
                        "goal_cost": {"a": 2},
                        "conditions_fulfilled_by_action": {"a": ["cond"]},
                        "actions_that_fulfill_condition": {"cond": [("a", node)]},
                    }
                if node == "n2":
                    return {
                        "action_cost": {"b": 2},
                        "viable": {"b": True},
                        "goal_cost": {"b": 1},
                        "conditions_fulfilled_by_action": {"b": ["cond"]},
                        "actions_that_fulfill_condition": {"cond": [("b", node)]},
                    }
                return {}

            def calculate_edge_cost(self, u, v):
                return 0

        mgr = TestMgr()
        goal = DummyGoal([{"node_attributes": {"type": "item"}}])
        character = None
        result = mgr.calculate_goal_difficulty(goal, character)
        self.assertIsInstance(result, dict)
        # Should pick the path with lowest total cost (goal_cost + action_cost)
        self.assertEqual(result["difficulty"], 3)

    def test_character_specific_difficulty(self):
        class TestMgr(GraphManager):
            def __init__(self):
                super().__init__()
                self.G = nx.Graph()
                self.G.add_node("n1", type="item")

            def get_filtered_nodes(self, **kwargs):
                return {"n1": {}}

            def calculate_action_viability_cost(self, node, goal, character):
                # Action cost depends on character
                cost = 1 if character == "hero" else 10
                return {
                    "action_cost": {"act1": cost},
                    "viable": {"act1": True},
                    "goal_cost": {"act1": 2},
                    "conditions_fulfilled_by_action": {"act1": ["cond1"]},
                    "actions_that_fulfill_condition": {"cond1": [("act1", node)]},
                }

            def calculate_edge_cost(self, u, v):
                return 0

        mgr = TestMgr()
        goal = DummyGoal([{"node_attributes": {"type": "item"}}])
        result_hero = mgr.calculate_goal_difficulty(goal, "hero")
        result_villain = mgr.calculate_goal_difficulty(goal, "villain")
        self.assertLess(result_hero["difficulty"], result_villain["difficulty"])
