"""
Test module for WorldState class.

This module contains comprehensive tests for the WorldState class to ensure
all CRUD operations and graph management functionality work correctly.
"""

import unittest
import networkx as nx
import logging
from unittest.mock import Mock

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)

# Import the WorldState class
from world_state import WorldState


class MockCharacter:
    """Mock character class for testing."""
    def __init__(self, name: str, age: int = 25):
        self.name = name
        self.age = age
        self.job = "test_job"
        self.happiness = 50
        self.energy = 80
    
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
    
    def __str__(self):
        return f"Location({self.name})"
    
    def __repr__(self):
        return self.__str__()


class MockJob:
    """Mock job class for testing."""
    def __init__(self, job_name: str, salary: int = 50000):
        self.job_name = job_name
        self.salary = salary
        self.required_skills = []
    
    def __str__(self):
        return f"Job({self.job_name})"
    
    def __repr__(self):
        return self.__str__()


class TestWorldStateInitialization(unittest.TestCase):
    """Test WorldState initialization."""
    
    def test_initialization(self):
        """Test that WorldState initializes correctly."""
        world_state = WorldState()
        
        # Check that graph is initialized
        self.assertIsInstance(world_state.graph, nx.MultiDiGraph)
        self.assertEqual(world_state.get_node_count(), 0)
        self.assertEqual(world_state.get_edge_count(), 0)
        
        # Check that dictionaries are initialized
        self.assertEqual(len(world_state.characters), 0)
        self.assertEqual(len(world_state.locations), 0)
        self.assertEqual(len(world_state.objects), 0)
        self.assertEqual(len(world_state.events), 0)
        self.assertEqual(len(world_state.activities), 0)
        self.assertEqual(len(world_state.jobs), 0)
        
        # Check type mapping
        self.assertIn("character", world_state.type_to_dict_map)
        self.assertIn("location", world_state.type_to_dict_map)


class TestWorldStateNodeOperations(unittest.TestCase):
    """Test node CRUD operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
        self.character = MockCharacter("Alice", 30)
        self.location = MockLocation("Park", "outdoor")
        self.job = MockJob("Engineer", 75000)
    
    def test_add_node(self):
        """Test adding nodes to the graph."""
        # Add character node
        self.world_state.add_node(
            self.character, 
            "character", 
            age=self.character.age,
            job=self.character.job
        )
        
        # Verify node was added
        self.assertTrue(self.world_state.has_node(self.character))
        self.assertEqual(self.world_state.get_node_count(), 1)
        self.assertIn("Alice", self.world_state.characters)
        
        # Check node attributes
        attrs = self.world_state.get_node_attributes(self.character)
        self.assertEqual(attrs["type"], "character")
        self.assertEqual(attrs["name"], "Alice")
        self.assertEqual(attrs["age"], 30)
        self.assertEqual(attrs["job"], "test_job")
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes of different types."""
        # Add nodes
        self.world_state.add_node(self.character, "character", age=30)
        self.world_state.add_node(self.location, "location", popularity=75)
        self.world_state.add_node(self.job, "job", salary=75000)
        
        # Verify all nodes were added
        self.assertEqual(self.world_state.get_node_count(), 3)
        self.assertTrue(self.world_state.has_node(self.character))
        self.assertTrue(self.world_state.has_node(self.location))
        self.assertTrue(self.world_state.has_node(self.job))
        
        # Check type-specific storage
        self.assertIn("Alice", self.world_state.characters)
        self.assertIn("Park", self.world_state.locations)
        self.assertIn("Engineer", self.world_state.jobs)
    
    def test_remove_node(self):
        """Test removing nodes from the graph."""
        # Add and then remove a node
        self.world_state.add_node(self.character, "character", age=30)
        self.assertTrue(self.world_state.has_node(self.character))
        
        self.world_state.remove_node(self.character)
        self.assertFalse(self.world_state.has_node(self.character))
        self.assertEqual(self.world_state.get_node_count(), 0)
        self.assertNotIn("Alice", self.world_state.characters)
    
    def test_remove_nonexistent_node(self):
        """Test removing a node that doesn't exist."""
        # Should not raise an exception
        self.world_state.remove_node(self.character)
        self.assertEqual(self.world_state.get_node_count(), 0)
    
    def test_update_node_attribute(self):
        """Test updating node attributes."""
        # Add node and update attribute
        self.world_state.add_node(self.character, "character", age=30)
        self.world_state.update_node_attribute(self.character, "age", 31)
        
        attrs = self.world_state.get_node_attributes(self.character)
        self.assertEqual(attrs["age"], 31)
    
    def test_update_nonexistent_node_attribute(self):
        """Test updating attribute of non-existent node."""
        with self.assertRaises(ValueError):
            self.world_state.update_node_attribute(self.character, "age", 31)
    
    def test_get_nodes_by_type(self):
        """Test getting nodes filtered by type."""
        # Add different types of nodes
        char1 = MockCharacter("Alice")
        char2 = MockCharacter("Bob")
        loc1 = MockLocation("Park")
        
        self.world_state.add_node(char1, "character")
        self.world_state.add_node(char2, "character")
        self.world_state.add_node(loc1, "location")
        
        # Test filtering
        characters = self.world_state.get_nodes("character")
        self.assertEqual(len(characters), 2)
        self.assertIn(char1, characters)
        self.assertIn(char2, characters)
        
        locations = self.world_state.get_nodes("location")
        self.assertEqual(len(locations), 1)
        self.assertIn(loc1, locations)
        
        # Test getting all nodes
        all_nodes = self.world_state.get_nodes()
        self.assertEqual(len(all_nodes), 3)


class TestWorldStateEdgeOperations(unittest.TestCase):
    """Test edge CRUD operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
        self.character1 = MockCharacter("Alice")
        self.character2 = MockCharacter("Bob")
        self.location = MockLocation("Park")
        
        # Add nodes first
        self.world_state.add_node(self.character1, "character")
        self.world_state.add_node(self.character2, "character")
        self.world_state.add_node(self.location, "location")
    
    def test_add_edge(self):
        """Test adding edges between nodes."""
        # Add edge with attributes
        self.world_state.add_edge(
            self.character1, 
            self.character2, 
            "friendship",
            strength=75,
            trust=60
        )
        
        # Verify edge was added
        self.assertTrue(self.world_state.has_edge(self.character1, self.character2))
        self.assertEqual(self.world_state.get_edge_count(), 1)
        
        # Check edge attributes
        attrs = self.world_state.get_edge_attributes(self.character1, self.character2)
        self.assertEqual(attrs["type"], "friendship")
        self.assertEqual(attrs["strength"], 75)
        self.assertEqual(attrs["trust"], 60)
    
    def test_add_multiple_edges(self):
        """Test adding multiple edges."""
        # Add different types of edges
        self.world_state.add_edge(self.character1, self.character2, "friendship", strength=75)
        self.world_state.add_edge(self.character1, self.location, "visits", frequency=5)
        self.world_state.add_edge(self.character2, self.location, "visits", frequency=3)
        
        # Verify all edges were added
        self.assertEqual(self.world_state.get_edge_count(), 3)
        self.assertTrue(self.world_state.has_edge(self.character1, self.character2))
        self.assertTrue(self.world_state.has_edge(self.character1, self.location))
        self.assertTrue(self.world_state.has_edge(self.character2, self.location))
    
    def test_remove_edge(self):
        """Test removing edges."""
        # Add and then remove edge
        self.world_state.add_edge(self.character1, self.character2, "friendship")
        self.assertTrue(self.world_state.has_edge(self.character1, self.character2))
        
        self.world_state.remove_edge(self.character1, self.character2)
        self.assertFalse(self.world_state.has_edge(self.character1, self.character2))
        self.assertEqual(self.world_state.get_edge_count(), 0)
    
    def test_remove_nonexistent_edge(self):
        """Test removing an edge that doesn't exist."""
        # Should not raise an exception
        self.world_state.remove_edge(self.character1, self.character2)
        self.assertEqual(self.world_state.get_edge_count(), 0)
    
    def test_update_edge_attribute(self):
        """Test updating edge attributes."""
        # Add edge and update attribute
        self.world_state.add_edge(self.character1, self.character2, "friendship", strength=75)
        self.world_state.update_edge_attribute(self.character1, self.character2, "strength", 85)
        
        attrs = self.world_state.get_edge_attributes(self.character1, self.character2)
        self.assertEqual(attrs["strength"], 85)
    
    def test_update_nonexistent_edge_attribute(self):
        """Test updating attribute of non-existent edge."""
        with self.assertRaises(ValueError):
            self.world_state.update_edge_attribute(self.character1, self.character2, "strength", 85)
    
    def test_get_edges_by_type(self):
        """Test getting edges filtered by type."""
        # Add different types of edges
        self.world_state.add_edge(self.character1, self.character2, "friendship")
        self.world_state.add_edge(self.character1, self.location, "visits")
        self.world_state.add_edge(self.character2, self.location, "visits")
        
        # Test filtering
        friendships = self.world_state.get_edges("friendship")
        self.assertEqual(len(friendships), 1)
        self.assertIn((self.character1, self.character2), friendships)
        
        visits = self.world_state.get_edges("visits")
        self.assertEqual(len(visits), 2)
        
        # Test getting all edges
        all_edges = self.world_state.get_edges()
        self.assertEqual(len(all_edges), 3)
    
    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        # Add edges
        self.world_state.add_edge(self.character1, self.character2, "friendship")
        self.world_state.add_edge(self.character1, self.location, "visits")
        
        # Test neighbors
        neighbors = self.world_state.get_neighbors(self.character1)
        self.assertEqual(len(neighbors), 2)
        self.assertIn(self.character2, neighbors)
        self.assertIn(self.location, neighbors)
        
        # Character2 should have no outgoing edges (no neighbors)
        neighbors2 = self.world_state.get_neighbors(self.character2)
        self.assertEqual(len(neighbors2), 0)


class TestWorldStateObjectRetrieval(unittest.TestCase):
    """Test object retrieval methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
        self.character = MockCharacter("Alice")
        self.location = MockLocation("Park")
        self.job = MockJob("Engineer")
        
        # Add objects
        self.world_state.add_node(self.character, "character")
        self.world_state.add_node(self.location, "location")
        self.world_state.add_node(self.job, "job")
    
    def test_get_object_by_name(self):
        """Test retrieving objects by name."""
        # Test with type specified
        char = self.world_state.get_object_by_name("Alice", "character")
        self.assertEqual(char, self.character)
        
        # Test without type specified
        loc = self.world_state.get_object_by_name("Park")
        self.assertEqual(loc, self.location)
        
        # Test non-existent object
        result = self.world_state.get_object_by_name("NonExistent")
        self.assertIsNone(result)
    
    def test_get_all_objects_by_type(self):
        """Test retrieving all objects of a specific type."""
        characters = self.world_state.get_all_objects_by_type("character")
        self.assertEqual(len(characters), 1)
        self.assertIn("Alice", characters)
        self.assertEqual(characters["Alice"], self.character)
        
        jobs = self.world_state.get_all_objects_by_type("job")
        self.assertEqual(len(jobs), 1)
        self.assertIn("Engineer", jobs)
        
        # Test non-existent type
        result = self.world_state.get_all_objects_by_type("nonexistent")
        self.assertEqual(len(result), 0)


class TestWorldStateUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
        self.character = MockCharacter("Alice")
        self.location = MockLocation("Park")
    
    def test_clear(self):
        """Test clearing the world state."""
        # Add some data
        self.world_state.add_node(self.character, "character")
        self.world_state.add_node(self.location, "location")
        self.world_state.add_edge(self.character, self.location, "visits")
        
        # Verify data exists
        self.assertEqual(self.world_state.get_node_count(), 2)
        self.assertEqual(self.world_state.get_edge_count(), 1)
        
        # Clear and verify
        self.world_state.clear()
        self.assertEqual(self.world_state.get_node_count(), 0)
        self.assertEqual(self.world_state.get_edge_count(), 0)
        self.assertEqual(len(self.world_state.characters), 0)
        self.assertEqual(len(self.world_state.locations), 0)
    
    def test_get_graph_copy(self):
        """Test getting a copy of the graph."""
        # Add some data
        self.world_state.add_node(self.character, "character")
        self.world_state.add_edge(self.character, self.character, "self_loop")
        
        # Get copy
        graph_copy = self.world_state.get_graph_copy()
        
        # Verify it's a copy (different object)
        self.assertIsNot(graph_copy, self.world_state.graph)
        
        # Verify contents are the same
        self.assertEqual(graph_copy.number_of_nodes(), 1)
        self.assertEqual(graph_copy.number_of_edges(), 1)
        
        # Verify modifying copy doesn't affect original
        graph_copy.clear()
        self.assertEqual(self.world_state.get_node_count(), 1)
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        # Empty world state
        str_repr = str(self.world_state)
        self.assertIn("WorldState", str_repr)
        self.assertIn("nodes=0", str_repr)
        self.assertIn("edges=0", str_repr)
        
        repr_repr = repr(self.world_state)
        self.assertIn("WorldState", repr_repr)
        self.assertIn("characters=0", repr_repr)
        self.assertIn("locations=0", repr_repr)
        
        # Add some data and test again
        self.world_state.add_node(self.character, "character")
        self.world_state.add_node(self.location, "location")
        
        str_repr = str(self.world_state)
        self.assertIn("nodes=2", str_repr)
        
        repr_repr = repr(self.world_state)
        self.assertIn("characters=1", repr_repr)
        self.assertIn("locations=1", repr_repr)


class TestWorldStateEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.world_state = WorldState()
    
    def test_object_without_name_attribute(self):
        """Test handling objects without name attribute."""
        class NoNameObject:
            def __init__(self, value):
                self.value = value
            def __str__(self):
                return f"NoName({self.value})"
        
        obj = NoNameObject("test")
        self.world_state.add_node(obj, "test_type", value="test")
        
        # Should use string representation as name
        self.assertTrue(self.world_state.has_node(obj))
        attrs = self.world_state.get_node_attributes(obj)
        self.assertEqual(attrs["name"], str(obj))
    
    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        # These should not raise exceptions
        nodes = self.world_state.get_nodes()
        self.assertEqual(len(nodes), 0)
        
        edges = self.world_state.get_edges()
        self.assertEqual(len(edges), 0)
        
        neighbors = self.world_state.get_neighbors(MockCharacter("test"))
        self.assertEqual(len(neighbors), 0)
        
        attrs = self.world_state.get_node_attributes(MockCharacter("test"))
        self.assertEqual(len(attrs), 0)
        
        attrs = self.world_state.get_edge_attributes(MockCharacter("test1"), MockCharacter("test2"))
        self.assertEqual(len(attrs), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)