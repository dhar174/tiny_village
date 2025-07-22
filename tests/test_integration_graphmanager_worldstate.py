"""
Integration test for GraphManager and WorldState refactoring.

This test verifies that the GraphManager correctly delegates CRUD operations
to the WorldState instance while maintaining the existing interface.
"""

import unittest
import logging
from unittest.mock import Mock

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)

# Import the refactored classes
from world_state import WorldState
from tiny_graph_manager import GraphManager


class MockCharacter:
    """Mock character class for testing."""
    def __init__(self, name: str, age: int = 25):
        self.name = name
        self.age = age
        self.job = "test_job"
        self.happiness = 50
        self.energy = 80
        self.coordinates_location = (0, 0)
        self.inventory = []
        self.needed_items = []
        self.current_mood = "neutral"
        self.wealth_money = 1000
        self.health_status = "healthy"
        self.hunger_level = 5
        self.mental_health = 75
        self.social_wellbeing = 70
        self.shelter = 80
        self.has_investment = False
    
    def get_wealth_money(self):
        return self.wealth_money
    
    def has_investment(self):
        return False
    
    def to_dict(self):
        return {
            'name': self.name,
            'age': self.age,
            'job': self.job,
            'happiness': self.happiness,
            'energy': self.energy
        }
    
    def __str__(self):
        return f"Character({self.name})"
    
    def __repr__(self):
        return self.__str__()


class MockLocation:
    """Mock location class for testing."""
    def __init__(self, name: str, location_type: str = "building"):
        self.name = name
        self.location_type = location_type
        self.popularity = 50
        self.activities_available = []
        self.accessible = True
        self.security = 75
        self.coordinates_location = (0, 0)
        self.threat_level = 10
        self.visit_count = 0
    
    def to_dict(self):
        return {
            'name': self.name,
            'location_type': self.location_type,
            'popularity': self.popularity
        }
    
    def __str__(self):
        return f"Location({self.name})"
    
    def __repr__(self):
        return self.__str__()


class MockJob:
    """Mock job class for testing."""
    def __init__(self, job_name: str, salary: int = 50000):
        self.job_name = job_name
        self.job_salary = salary
        self.job_skills = []
        self.location = "office"
        self.job_title = job_name
        self.available = True
    
    def to_dict(self):
        return {
            'job_name': self.job_name,
            'job_salary': self.job_salary,
            'job_skills': self.job_skills
        }
    
    def __str__(self):
        return f"Job({self.job_name})"
    
    def __repr__(self):
        return self.__str__()


class TestGraphManagerWorldStateIntegration(unittest.TestCase):
    """Test that GraphManager correctly delegates to WorldState."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph_manager = GraphManager()
        # Clear the graph state for clean tests
        self.graph_manager.world_state.clear()
        
        self.character = MockCharacter("Alice", 30)
        self.location = MockLocation("Park", "outdoor")
        self.job = MockJob("Engineer", 75000)
    
    def test_graph_manager_uses_world_state(self):
        """Test that GraphManager has a WorldState instance."""
        self.assertIsInstance(self.graph_manager.world_state, WorldState)
        self.assertIs(self.graph_manager.G, self.graph_manager.world_state.graph)
    
    def test_graph_manager_delegates_dictionaries(self):
        """Test that GraphManager uses WorldState dictionaries."""
        self.assertIs(self.graph_manager.characters, self.graph_manager.world_state.characters)
        self.assertIs(self.graph_manager.locations, self.graph_manager.world_state.locations)
        self.assertIs(self.graph_manager.jobs, self.graph_manager.world_state.jobs)
    
    def test_add_character_node_delegation(self):
        """Test that GraphManager.add_character_node delegates to WorldState."""
        # Verify initial state
        self.assertEqual(len(self.graph_manager.characters), 0)
        self.assertEqual(self.graph_manager.world_state.get_node_count(), 0)
        
        # Add character through GraphManager
        self.graph_manager.add_character_node(self.character)
        
        # Verify character was added through WorldState
        self.assertEqual(len(self.graph_manager.characters), 1)
        self.assertEqual(self.graph_manager.world_state.get_node_count(), 1)
        self.assertIn("Alice", self.graph_manager.characters)
        self.assertTrue(self.graph_manager.world_state.has_node(self.character))
        
        # Verify node attributes
        attrs = self.graph_manager.world_state.get_node_attributes(self.character)
        self.assertEqual(attrs["type"], "character")
        self.assertEqual(attrs["name"], "Alice")
        self.assertEqual(attrs["age"], 30)
    
    def test_add_location_node_delegation(self):
        """Test that GraphManager.add_location_node delegates to WorldState."""
        # Add location through GraphManager
        self.graph_manager.add_location_node(self.location)
        
        # Verify location was added through WorldState
        self.assertEqual(len(self.graph_manager.locations), 1)
        self.assertEqual(self.graph_manager.world_state.get_node_count(), 1)
        self.assertIn("Park", self.graph_manager.locations)
        self.assertTrue(self.graph_manager.world_state.has_node(self.location))
        
        # Verify node attributes
        attrs = self.graph_manager.world_state.get_node_attributes(self.location)
        self.assertEqual(attrs["type"], "location")
        self.assertEqual(attrs["name"], "Park")
    
    def test_add_job_node_delegation(self):
        """Test that GraphManager.add_job_node delegates to WorldState."""
        # Add job through GraphManager
        self.graph_manager.add_job_node(self.job)
        
        # Verify job was added through WorldState
        self.assertEqual(len(self.graph_manager.jobs), 1)
        self.assertEqual(self.graph_manager.world_state.get_node_count(), 1)
        self.assertIn("Engineer", self.graph_manager.jobs)
        self.assertTrue(self.graph_manager.world_state.has_node(self.job))
        
        # Verify node attributes
        attrs = self.graph_manager.world_state.get_node_attributes(self.job)
        self.assertEqual(attrs["type"], "job")
        self.assertEqual(attrs["name"], "Engineer")
    
    def test_add_dict_of_nodes_delegation(self):
        """Test that GraphManager.add_dict_of_nodes delegates to WorldState."""
        nodes_dict = {
            "characters": [self.character],
            "locations": [self.location],
            "jobs": [self.job]
        }
        
        # Add nodes through GraphManager
        self.graph_manager.add_dict_of_nodes(nodes_dict)
        
        # Verify all nodes were added through WorldState
        self.assertEqual(self.graph_manager.world_state.get_node_count(), 3)
        self.assertEqual(len(self.graph_manager.characters), 1)
        self.assertEqual(len(self.graph_manager.locations), 1)
        self.assertEqual(len(self.graph_manager.jobs), 1)
    
    def test_update_node_attribute_delegation(self):
        """Test that GraphManager.update_node_attribute delegates to WorldState."""
        # Add a character and update its attribute
        self.graph_manager.add_character_node(self.character)
        self.graph_manager.update_node_attribute(self.character, "age", 31)
        
        # Verify attribute was updated through WorldState
        attrs = self.graph_manager.world_state.get_node_attributes(self.character)
        self.assertEqual(attrs["age"], 31)
    
    def test_update_edge_attribute_delegation(self):
        """Test that GraphManager.update_edge_attribute delegates to WorldState."""
        # Add nodes and edge
        self.graph_manager.add_character_node(self.character)
        char2 = MockCharacter("Bob", 25)
        self.graph_manager.add_character_node(char2)
        
        # Add edge through WorldState directly (since GraphManager edge methods are complex)
        self.graph_manager.world_state.add_edge(self.character, char2, "friendship", strength=75)
        
        # Update edge attribute through GraphManager
        self.graph_manager.update_edge_attribute(self.character, char2, "strength", 85)
        
        # Verify attribute was updated through WorldState
        attrs = self.graph_manager.world_state.get_edge_attributes(self.character, char2)
        self.assertEqual(attrs["strength"], 85)
    
    def test_singleton_behavior_preserved(self):
        """Test that GraphManager singleton behavior is preserved."""
        gm1 = GraphManager()
        gm2 = GraphManager()
        
        # Both instances should be the same
        self.assertIs(gm1, gm2)
        self.assertIs(gm1.world_state, gm2.world_state)
        
        # Adding a node through one should be visible in the other
        gm1.add_character_node(self.character)
        self.assertEqual(len(gm2.characters), 1)
        self.assertTrue(gm2.world_state.has_node(self.character))
    
    def test_graph_operations_still_work(self):
        """Test that direct graph operations still work as expected."""
        # Add nodes
        self.graph_manager.add_character_node(self.character)
        self.graph_manager.add_location_node(self.location)
        
        # Verify graph properties
        self.assertEqual(self.graph_manager.G.number_of_nodes(), 2)
        self.assertTrue(self.graph_manager.G.has_node(self.character))
        self.assertTrue(self.graph_manager.G.has_node(self.location))
        
        # Add edge directly through graph
        self.graph_manager.G.add_edge(self.character, self.location, type="visits")
        self.assertEqual(self.graph_manager.G.number_of_edges(), 1)
        self.assertTrue(self.graph_manager.G.has_edge(self.character, self.location))


class TestWorldStateStandaloneIntegration(unittest.TestCase):
    """Test WorldState as a standalone component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
        self.character = MockCharacter("Alice", 30)
        self.location = MockLocation("Park", "outdoor")
    
    def test_world_state_character_specific_methods(self):
        """Test WorldState character-specific methods."""
        # Add character using specific method
        self.world_state.add_character_node(self.character)
        
        # Verify attributes are correctly set
        attrs = self.world_state.get_node_attributes(self.character)
        self.assertEqual(attrs["type"], "character")
        self.assertEqual(attrs["age"], 30)
        self.assertEqual(attrs["job"], "test_job")
        self.assertEqual(attrs["wealth_money"], 1000)
        self.assertEqual(attrs["health_status"], "healthy")
    
    def test_world_state_location_specific_methods(self):
        """Test WorldState location-specific methods."""
        # Add location using specific method
        self.world_state.add_location_node(self.location)
        
        # Verify attributes are correctly set
        attrs = self.world_state.get_node_attributes(self.location)
        self.assertEqual(attrs["type"], "location")
        self.assertEqual(attrs["popularity"], 50)
        self.assertEqual(attrs["accessible"], True)
        self.assertEqual(attrs["security"], 75)
    
    def test_world_state_maintains_graph_integrity(self):
        """Test that WorldState maintains graph integrity."""
        # Add multiple nodes and edges
        char2 = MockCharacter("Bob", 25)
        self.world_state.add_character_node(self.character)
        self.world_state.add_character_node(char2)
        self.world_state.add_location_node(self.location)
        
        # Add edges
        self.world_state.add_edge(self.character, char2, "friendship", strength=75)
        self.world_state.add_edge(self.character, self.location, "visits", frequency=5)
        
        # Verify graph state
        self.assertEqual(self.world_state.get_node_count(), 3)
        self.assertEqual(self.world_state.get_edge_count(), 2)
        
        # Verify neighbors
        neighbors = self.world_state.get_neighbors(self.character)
        self.assertEqual(len(neighbors), 2)
        self.assertIn(char2, neighbors)
        self.assertIn(self.location, neighbors)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)