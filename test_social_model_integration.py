"""Integration tests for SocialModel against the real GraphManager."""

import logging
import sys
import types
import unittest

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from mock_character import MockCharacter, MockMotive, MockMotives

mock_memories = types.ModuleType("tiny_memories")
mock_memories.Memory = object
mock_memories.MemoryManager = object
sys.modules["tiny_memories"] = mock_memories

mock_characters = types.ModuleType("tiny_characters")
mock_characters.Character = MockCharacter
mock_characters.PersonalMotives = MockMotives
mock_characters.Motive = MockMotive
mock_characters.Goal = object
sys.modules["tiny_characters"] = mock_characters

from social_model import SocialModel
from tiny_graph_manager import GraphManager
from tiny_globals import reset_global_graph_manager


def seed_character_relationship(graph_manager):
    graph_manager.G.add_node("Alice", type="character", name="Alice", wealth_money=1000)
    graph_manager.G.add_node("Bob", type="character", name="Bob", wealth_money=900)
    graph_manager.G.add_edge(
        "Alice",
        "Bob",
        key="character_character",
        type="character_character",
        relationship_type="friend",
        trust=0.8,
        emotional=0.6,
        strength=0.7,
        historical=45,
        interaction_frequency=0.5,
    )
    graph_manager.G.add_edge(
        "Alice",
        "Bob",
        key="character_event",
        type="character_event",
        trust=0.05,
        event_id="festival",
    )


class TestSocialModelIntegration(unittest.TestCase):
    def setUp(self):
        reset_global_graph_manager()
        self.graph_manager = GraphManager()
        seed_character_relationship(self.graph_manager)
        self.graph_manager.social_model.set_world_state(self.graph_manager)

    def tearDown(self):
        reset_global_graph_manager()

    def test_graph_manager_has_social_model(self):
        self.assertIsInstance(self.graph_manager.social_model, SocialModel)
        self.assertIs(self.graph_manager.social_model.world_state, self.graph_manager)

    def test_retrieve_characters_relationships_reads_real_graph_manager_edges(self):
        relationships = self.graph_manager.retrieve_characters_relationships("Alice")

        self.assertIn("Bob", relationships)
        self.assertEqual(relationships["Bob"]["relationship_type"], "friend")
        self.assertEqual(relationships["Bob"]["trust"], 0.8)

    def test_calculate_social_influence_uses_real_character_neighbors(self):
        influence = self.graph_manager.calculate_social_influence("Alice")

        self.assertEqual(influence, 0.8)

    def test_update_relationship_status_updates_underlying_multigraph_edge(self):
        self.graph_manager.social_model.update_relationship_status(
            "Alice",
            "Bob",
            {"trust": 0.1},
        )

        relationships = self.graph_manager.retrieve_characters_relationships("Alice")
        self.assertEqual(relationships["Bob"]["trust"], 0.9)
        self.assertEqual(
            self.graph_manager.G["Alice"]["Bob"]["character_character"]["trust"],
            0.9,
        )
        self.assertEqual(
            self.graph_manager.G["Alice"]["Bob"]["character_event"]["trust"],
            0.05,
        )

    def test_analyze_relationship_health_uses_flattened_edge_attributes(self):
        health = self.graph_manager.analyze_relationship_health("Alice", "Bob")

        self.assertEqual(health["status"], "good")
        self.assertAlmostEqual(health["health_score"], 0.6425, places=4)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
