#!/usr/bin/env python3
"""
Unit tests for core global GraphManager functionality.

Tests the global GraphManager instance management in tiny_globals
without complex dependencies.
"""

import unittest
import sys

sys.path.insert(0, '.')

from tiny_globals import (
    get_global_graph_manager,
    has_global_graph_manager,
    initialize_global_graph_manager,
    set_global_graph_manager,
    reset_global_graph_manager
)


class TestGlobalGraphManagerCore(unittest.TestCase):
    """Test cases for core global GraphManager functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset the global graph manager before each test
        reset_global_graph_manager()
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset the global graph manager after each test
        reset_global_graph_manager()
    
    def test_singleton_behavior(self):
        """Test that get_global_graph_manager returns the same instance."""
        self.assertFalse(has_global_graph_manager())
        
        gm1 = get_global_graph_manager()
        self.assertIsNotNone(gm1)
        self.assertTrue(has_global_graph_manager())
        
        gm2 = get_global_graph_manager()
        self.assertIsNotNone(gm2)
        
        # Verify singleton behavior - same instance
        self.assertIs(gm1, gm2, "get_global_graph_manager should return the same instance")
    
    def test_initialize_creates_instance(self):
        """Test that initialize_global_graph_manager creates a GraphManager instance."""
        self.assertFalse(has_global_graph_manager())
        
        gm = initialize_global_graph_manager()
        self.assertIsNotNone(gm)
        self.assertTrue(has_global_graph_manager())
        
        # Verify it's a GraphManager instance
        from tiny_graph_manager import GraphManager
        self.assertIsInstance(gm, GraphManager)
    
    def test_multiple_initialize_calls_return_same_instance(self):
        """Test that multiple calls to initialize return the same instance."""
        gm1 = initialize_global_graph_manager()
        gm2 = initialize_global_graph_manager()
        
        self.assertIs(gm1, gm2, "Multiple initialize calls should return the same instance")
    
    def test_graph_manager_has_graph(self):
        """Test that the GraphManager has an initialized graph."""
        gm = get_global_graph_manager()
        
        self.assertTrue(hasattr(gm, 'G'), "GraphManager should have a 'G' attribute")
        self.assertIsNotNone(gm.G)
        
        # Verify it's a NetworkX graph
        import networkx as nx
        self.assertIsInstance(gm.G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    
    def test_graph_manager_has_world_state(self):
        """Test that the GraphManager has a WorldState instance."""
        gm = get_global_graph_manager()
        
        self.assertTrue(hasattr(gm, 'world_state'), "GraphManager should have 'world_state' attribute")
        self.assertIsNotNone(gm.world_state)
        
        from world_state import WorldState
        self.assertIsInstance(gm.world_state, WorldState)
    
    def test_world_state_graph_is_same_as_g(self):
        """Test that WorldState.graph is the same object as GraphManager.G."""
        gm = get_global_graph_manager()
        
        self.assertIs(gm.world_state.graph, gm.G,
                     "WorldState.graph should be the same object as GraphManager.G")
    
    def test_actions_use_global_graph_manager(self):
        """Test that Action class uses the global GraphManager."""
        from actions import Action
        
        gm = get_global_graph_manager()
        action = Action("TestAction", [], [], 1.0)
        
        self.assertIsNotNone(action.graph_manager)
        self.assertIs(action.graph_manager, gm,
                     "Action should use the global GraphManager instance")
    
    def test_action_generator_uses_global_graph_manager(self):
        """Test that ActionGenerator uses the global GraphManager."""
        from actions import ActionGenerator
        
        gm = get_global_graph_manager()
        ag = ActionGenerator()
        
        self.assertIsNotNone(ag.graph_manager)
        self.assertIs(ag.graph_manager, gm,
                     "ActionGenerator should use the global GraphManager instance")
    
    def test_set_global_graph_manager(self):
        """Test that set_global_graph_manager changes the global instance."""
        # Initialize with default
        gm1 = get_global_graph_manager()
        
        # Create a new GraphManager
        from tiny_graph_manager import GraphManager
        gm2 = GraphManager()
        
        # Set it as global
        set_global_graph_manager(gm2)
        
        # Verify the new instance is now global
        gm3 = get_global_graph_manager()
        self.assertIs(gm3, gm2, "set_global_graph_manager should change the global instance")
        self.assertIsNot(gm3, gm1, "New global instance should be different from old one")


if __name__ == "__main__":
    unittest.main()
