"""Graph-backed tests for SocialModel core behavior."""

import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, ".")

from social_model import SocialModel, calculate_relationship_type


class GraphBackedWorldState:
    """Minimal real graph harness for SocialModel unit tests."""

    def __init__(self):
        self.G = nx.MultiGraph()
        self.G.add_node("Alice", type="character", wealth_money=1000)
        self.G.add_node("Bob", type="character", wealth_money=900)
        self.G.add_node("TownSquare", type="location")
        self.G.add_edge(
            "Alice",
            "Bob",
            relationship_type="friend",
            trust=0.8,
            emotional=0.6,
            strength=0.7,
            historical=45,
            interaction_frequency=0.5,
        )
        self.G.add_edge("Alice", "TownSquare", trust=0.95)

    def _attribute_values(self, attribute):
        values = [data.get(attribute, 0) for _, data in self.G.nodes(data=True)]
        return values or [0]

    def get_maximum_attribute_value(self, attribute):
        return max(self._attribute_values(attribute))

    def get_average_attribute_value(self, attribute):
        values = self._attribute_values(attribute)
        return sum(values) / len(values)

    def get_stddev_attribute_value(self, attribute):
        values = self._attribute_values(attribute)
        stddev = float(np.std(values))
        return stddev if stddev else 1.0


class PersonalityTraitsFixture:
    def __init__(self, **traits):
        self.traits = {
            "openness": 5,
            "extraversion": 5,
            "conscientiousness": 5,
            "agreeableness": 5,
            "neuroticism": 5,
        }
        self.traits.update(traits)

    def get_openness(self):
        return self.traits["openness"]

    def get_extraversion(self):
        return self.traits["extraversion"]

    def get_conscientiousness(self):
        return self.traits["conscientiousness"]

    def get_agreeableness(self):
        return self.traits["agreeableness"]

    def get_neuroticism(self):
        return self.traits["neuroticism"]


class MotiveFixture:
    def __init__(self, score):
        self.score = score


class MotivesFixture:
    def get_wealth_motive(self):
        return MotiveFixture(3)

    def get_family_motive(self):
        return MotiveFixture(7)

    def get_beauty_motive(self):
        return MotiveFixture(4)

    def get_luxury_motive(self):
        return MotiveFixture(2)

    def get_stability_motive(self):
        return MotiveFixture(6)

    def get_control_motive(self):
        return MotiveFixture(4)


class LocationFixture:
    def __init__(self, name="Shared Home"):
        self.name = name


class JobFixture:
    def __init__(self, location):
        self.location = location


class CharacterFixture:
    def __init__(self, name, home=None, traits=None, wealth_money=1000):
        shared_home = home or LocationFixture()
        self.name = name
        self.personality_traits = PersonalityTraitsFixture(**(traits or {}))
        self.age = 25
        self.beauty = 5
        self.energy = 50
        self.wealth_money = wealth_money
        self.stability = 5
        self.luxury = 3
        self.monogamy = 8
        self.shelter = 7
        self.success = 4
        self.job = JobFixture(shared_home)
        self.home = shared_home

    def get_motives(self):
        return MotivesFixture()

    def get_base_libido(self):
        return 50

    def get_control(self):
        return 5


class TestSocialModel(unittest.TestCase):
    def setUp(self):
        self.world_state = GraphBackedWorldState()
        self.social_model = SocialModel(self.world_state)

    def test_retrieve_characters_relationships_reads_real_neighbors(self):
        relationships = self.social_model.retrieve_characters_relationships("Alice")

        self.assertIn("Bob", relationships)
        self.assertEqual(relationships["Bob"]["relationship_type"], "friend")
        self.assertEqual(relationships["Bob"]["trust"], 0.8)

    def test_calculate_social_influence_filters_non_character_neighbors(self):
        influence = self.social_model.calculate_social_influence("Alice")

        self.assertEqual(influence, 0.8)

    def test_update_relationship_status_mutates_real_graph_edge(self):
        self.social_model.update_relationship_status("Alice", "Bob", {"trust": 0.1})

        self.assertEqual(self.world_state.G["Alice"]["Bob"]["trust"], 0.9)

    def test_analyze_relationship_health_uses_real_edge_metrics(self):
        health = self.social_model.analyze_relationship_health("Alice", "Bob")

        self.assertEqual(health["status"], "good")
        self.assertAlmostEqual(health["health_score"], 0.6425, places=4)

    def test_romance_calculations_stay_bounded_with_graph_statistics(self):
        shared_home = LocationFixture()
        alice = CharacterFixture(
            "Alice",
            home=shared_home,
            traits={"openness": 6, "extraversion": 7},
            wealth_money=1000,
        )
        bob = CharacterFixture(
            "Bob",
            home=shared_home,
            traits={"openness": 5, "extraversion": 6},
            wealth_money=900,
        )

        compatibility = self.social_model.calculate_romance_compatibility(alice, bob, 30)
        interest = self.social_model.calculate_romance_interest(
            alice,
            bob,
            compatibility,
            5,
            "friend",
            0.8,
            50,
            0.6,
            0.4,
            0.3,
        )

        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
        self.assertGreaterEqual(interest, 0.0)
        self.assertLessEqual(interest, 1.0)

    def test_calculate_relationship_type_prefers_shared_home_label(self):
        shared_home = LocationFixture()
        alice = CharacterFixture("Alice", home=shared_home)
        bob = CharacterFixture("Bob", home=shared_home)

        relationship_type = calculate_relationship_type(
            alice,
            bob,
            0.6,
            0.4,
            0.7,
            0.8,
            60,
        )

        self.assertEqual(relationship_type, "roommate")


if __name__ == "__main__":
    unittest.main()
