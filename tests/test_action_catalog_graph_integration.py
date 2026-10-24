import unittest
import sys
import types
from unittest.mock import MagicMock

from actions import (
    Action,
    BuyFoodAction,
    Condition,
    EatAction,
    GoToLocationAction,
    GreetAction,
    ImproveJobPerformanceAction,
    OfferComplimentAction,
    PursueHobbyAction,
    ShareNewsAction,
    SleepAction,
    SocialVisitAction,
    TalkAction,
    VisitDoctorAction,
    WorkAction,
)
from tests.social_action_test_utils import build_character
from tiny_globals import reset_global_graph_manager


_ORIGINAL_MODULES = {}
GraphManager = None


def setUpModule():
    global GraphManager

    fake_social_model = types.ModuleType("social_model")
    fake_goap_evaluator = types.ModuleType("goap_evaluator")
    fake_graph_analytics = types.ModuleType("graph_analytics")
    fake_networkx = types.ModuleType("networkx")

    class FakeNodeView:
        def __init__(self, graph):
            self.graph = graph

        def __contains__(self, node):
            return node in self.graph._nodes

        def __getitem__(self, node):
            return self.graph._nodes[node]

        def __call__(self, data=False):
            if data is True:
                return list(self.graph._nodes.items())
            if isinstance(data, str):
                return [(node, attrs.get(data)) for node, attrs in self.graph._nodes.items()]
            return list(self.graph._nodes.keys())

    class FakeMultiDiGraph:
        def __init__(self):
            self._nodes = {}
            self.nodes = FakeNodeView(self)

        def add_node(self, node, **attrs):
            self._nodes[node] = attrs

        def has_node(self, node):
            return node in self._nodes

        def add_edge(self, *args, **kwargs):
            return None

        def has_edge(self, *args, **kwargs):
            return False

    class FakeSocialModel:
        def __init__(self, *args, **kwargs):
            self.world_state = kwargs.get("world_state")

    class FakeGoapEvaluator:
        def calculate_action_effect_cost(self, *args, **kwargs):
            return 0

        def calculate_how_goal_impacts_character(self, *args, **kwargs):
            return 0

        def calculate_action_viability_cost(self, *args, **kwargs):
            return {}

        def will_action_fulfill_goal(self, *args, **kwargs):
            return {}

    class FakeGraphAnalytics:
        def __init__(self, *args, **kwargs):
            self.world_state = args[0] if args else None

    fake_social_model.SocialModel = FakeSocialModel
    fake_goap_evaluator.GoapEvaluator = FakeGoapEvaluator
    fake_goap_evaluator.WorldState = object
    fake_graph_analytics.GraphAnalytics = FakeGraphAnalytics
    fake_networkx.MultiDiGraph = FakeMultiDiGraph

    module_overrides = {
        "social_model": fake_social_model,
        "goap_evaluator": fake_goap_evaluator,
        "graph_analytics": fake_graph_analytics,
        "networkx": fake_networkx,
    }
    for module_name, replacement in module_overrides.items():
        _ORIGINAL_MODULES[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = replacement

    sys.modules.pop("tiny_graph_manager", None)
    from tiny_graph_manager import GraphManager as ImportedGraphManager

    GraphManager = ImportedGraphManager


def tearDownModule():
    sys.modules.pop("tiny_graph_manager", None)
    for module_name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


class TestConcreteActionDefaults(unittest.TestCase):
    def setUp(self):
        self.alice = build_character(
            "Alice",
            default_energy=6.0,
            hunger_level=6.0,
            health_status=5.0,
            mental_health=4.0,
            social_wellbeing=4.0,
            wealth_money=20.0,
            job_performance=15.0,
        )
        self.bob = build_character(
            "Bob",
            default_energy=6.0,
            social_wellbeing=4.0,
        )

    def test_concrete_actions_expose_meaningful_default_preconditions_and_effects(self):
        cases = {
            "Eat": EatAction("bread", initiator_id=self.alice),
            "GoToLocation": GoToLocationAction("Bakery", initiator_id=self.alice),
            "BuyFood": BuyFoodAction("bread", initiator_id=self.alice),
            "Work": WorkAction("baker", initiator_id=self.alice),
            "Sleep": SleepAction(initiator_id=self.alice),
            "SocialVisit": SocialVisitAction(self.bob, initiator_id=self.alice),
            "ImproveJobPerformance": ImproveJobPerformanceAction(initiator_id=self.alice),
            "PursueHobby": PursueHobbyAction("painting", initiator_id=self.alice),
            "VisitDoctor": VisitDoctorAction(initiator_id=self.alice),
            "Talk": TalkAction(initiator=self.alice, target=self.bob),
            "Greet": GreetAction(initiator=self.alice, target=self.bob),
            "ShareNews": ShareNewsAction(
                initiator=self.alice,
                target=self.bob,
                news_item="The market opens early.",
            ),
            "OfferCompliment": OfferComplimentAction(
                initiator=self.alice,
                target=self.bob,
                compliment_topic="their gardening",
            ),
        }

        for action_name, action in cases.items():
            with self.subTest(action=action_name):
                self.assertTrue(action.preconditions, f"{action_name} should have default preconditions")
                # Talk delegates its concrete state change to the target's talk-response path,
                # so the catalog only guarantees a default precondition for it here.
                self.assertTrue(action.effects or action_name == "Talk", f"{action_name} should have defined effects")

        self.assertEqual(
            [effect["attribute"] for effect in EatAction("bread", initiator_id=self.alice).effects],
            ["hunger_level", "energy"],
        )
        self.assertEqual(
            [(condition.attribute, condition.operator, condition.satisfy_value) for condition in EatAction("bread", initiator_id=self.alice).preconditions],
            [("hunger_level", "ge", 1), ("energy", "ge", 1)],
        )
        self.assertEqual(
            [effect["attribute"] for effect in WorkAction("baker", initiator_id=self.alice).effects],
            ["wealth_money", "energy", "job_performance"],
        )
        self.assertEqual(
            [(condition.attribute, condition.operator, condition.satisfy_value) for condition in WorkAction("baker", initiator_id=self.alice).preconditions],
            [("energy", "ge", 2), ("health_status", "ge", 3)],
        )
        self.assertEqual(
            [effect["attribute"] for effect in VisitDoctorAction(initiator_id=self.alice).effects],
            ["health_status", "wealth_money"],
        )
        self.assertEqual(
            [(condition.attribute, condition.operator, condition.satisfy_value) for condition in VisitDoctorAction(initiator_id=self.alice).preconditions],
            [("wealth_money", "ge", 10), ("health_status", "le", 9)],
        )


class TestActionExecutionAttributeMapping(unittest.TestCase):
    def setUp(self):
        self.graph_manager = MagicMock()
        self.alice = build_character(
            "Alice",
            default_energy=6.0,
            hunger_level=7.0,
            health_status=4.0,
            wealth_money=10.0,
        )

    def test_execute_maps_alias_effects_to_real_character_and_graph_fields(self):
        action = Action(
            name="AliasEffectAction",
            preconditions=[],
            effects=[
                {"targets": ["initiator"], "attribute": "hunger", "change_value": -2},
                {"targets": ["initiator"], "attribute": "money", "change_value": 5},
                {"targets": ["initiator"], "attribute": "health", "change_value": 2},
            ],
            initiator=self.alice,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.assertEqual(self.alice.hunger_level, 5.0)
        self.assertEqual(self.alice.wealth_money, 15.0)
        self.assertEqual(self.alice.health_status, 6.0)
        self.graph_manager.update_node_attribute.assert_any_call(
            self.alice.uuid,
            "hunger_level",
            5.0,
        )
        self.graph_manager.update_node_attribute.assert_any_call(
            self.alice.uuid,
            "wealth_money",
            15.0,
        )
        self.graph_manager.update_node_attribute.assert_any_call(
            self.alice.uuid,
            "health_status",
            6.0,
        )


class DummyActor:
    def __init__(self, name, energy=0.0, wealth_money=0.0):
        self.name = name
        self.uuid = f"{name}-uuid"
        self.energy = energy
        self.wealth_money = wealth_money

    def get_state(self):
        return {"energy": self.energy, "wealth_money": self.wealth_money}


class DummyGoal:
    def __init__(self, current_state, completion_conditions):
        self.current_state = current_state
        self.completion_conditions = completion_conditions


class TestGraphManagerActionInterpretation(unittest.TestCase):
    def setUp(self):
        reset_global_graph_manager()
        self.graph_manager = GraphManager()

    def tearDown(self):
        reset_global_graph_manager()

    def test_will_path_achieve_goal_handles_list_preconditions_and_mapped_effects(self):
        actor = DummyActor("Alice", energy=6.0, wealth_money=5.0)
        work_action = Action(
            name="Work",
            preconditions=[Condition("HasEnergy", "energy", actor, 5, ">=")],
            effects=[{"targets": ["initiator"], "attribute": "money", "change_value": 10}],
            initiator=actor,
        )
        goal = DummyGoal(
            current_state={"energy": actor.energy, "wealth_money": actor.wealth_money},
            completion_conditions=[Condition("EarnMoney", "wealth_money", actor, 15, ">=")],
        )

        self.graph_manager.G.add_node("job_node", possible_interactions=[work_action])

        self.assertTrue(self.graph_manager.will_path_achieve_goal(["job_node"], goal))


if __name__ == "__main__":
    unittest.main()
