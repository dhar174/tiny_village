"""
Integration tests for GraphManager with GraphAnalytics delegation.

This test module verifies that GraphManager properly delegates graph analysis
methods to the GraphAnalytics class while maintaining backwards compatibility.
"""

import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, '.')

# Mock heavy dependencies to avoid import issues
import types
import logging

# Create mock modules
mock_memories = types.ModuleType('tiny_memories')
mock_memories.Memory = object
mock_memories.MemoryManager = object
sys.modules['tiny_memories'] = mock_memories

mock_characters = types.ModuleType('tiny_characters')
mock_characters.Character = object
mock_characters.PersonalMotives = object
mock_characters.Motive = object
sys.modules['tiny_characters'] = mock_characters

# Now import the graph manager and analytics
from tiny_graph_manager import GraphManager
from graph_analytics import GraphAnalytics


class TestGraphManagerGraphAnalyticsDelegation(unittest.TestCase):
    """Test that GraphManager properly delegates to GraphAnalytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph_manager = GraphManager()
        
        # Create some test data
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create test characters and add them to the graph."""
        # Create mock character objects with proper attributes
        self.char1 = Mock()
        self.char1.name = "Alice"
        self.char1.age = 25
        self.char1.coordinates_location = (0, 0)
        # Mock personality traits and motives to avoid calculation errors
        mock_traits = Mock()
        mock_traits.openness = 5
        mock_traits.extraversion = 6
        mock_traits.agreeableness = 7
        mock_traits.conscientiousness = 8
        mock_traits.neuroticism = 4
        mock_traits.to_dict.return_value = {
            'openness': 5, 'extraversion': 6, 'agreeableness': 7,
            'conscientiousness': 8, 'neuroticism': 4
        }
        self.char1.personality_traits = mock_traits
        
        # Mock motives
        mock_motives = Mock()
        mock_motives.to_dict.return_value = {}
        self.char1.motives = mock_motives
        
        self.char2 = Mock()
        self.char2.name = "Bob"
        self.char2.age = 30
        self.char2.coordinates_location = (1, 1)
        # Similar mock setup for char2
        mock_traits2 = Mock()
        mock_traits2.openness = 6
        mock_traits2.extraversion = 5
        mock_traits2.agreeableness = 6
        mock_traits2.conscientiousness = 7
        mock_traits2.neuroticism = 5
        mock_traits2.to_dict.return_value = {
            'openness': 6, 'extraversion': 5, 'agreeableness': 6,
            'conscientiousness': 7, 'neuroticism': 5
        }
        self.char2.personality_traits = mock_traits2
        
        mock_motives2 = Mock()
        mock_motives2.to_dict.return_value = {}
        self.char2.motives = mock_motives2
        
        # Add characters to the graph (just nodes, avoid complex edge logic for now)
        self.graph_manager.add_character_node(self.char1)
        self.graph_manager.add_character_node(self.char2)
        
        # Add a simple edge directly to the graph to avoid complex edge creation logic
        self.graph_manager.world_state.add_edge(
            self.char1, self.char2, "friendship", weight=0.8
        )
    
    def test_graph_manager_has_graph_analytics(self):
        """Test that GraphManager has GraphAnalytics instance."""
        self.assertTrue(hasattr(self.graph_manager, 'graph_analytics'))
        self.assertIsInstance(self.graph_manager.graph_analytics, GraphAnalytics)
    
    def test_graph_analytics_has_world_state_dependency(self):
        """Test that GraphAnalytics has proper WorldState dependency."""
        self.assertEqual(
            self.graph_manager.graph_analytics.world_state,
            self.graph_manager.world_state
        )
        self.assertEqual(
            self.graph_manager.graph_analytics.graph,
            self.graph_manager.world_state.graph
        )
    
    def test_find_shortest_path_delegation(self):
        """Test that find_shortest_path is properly delegated."""
        with patch.object(
            self.graph_manager.graph_analytics, 
            'find_shortest_path',
            return_value=['mocked_path']
        ) as mock_method:
            result = self.graph_manager.find_shortest_path(self.char1, self.char2)
            
            # Verify delegation occurred
            mock_method.assert_called_once_with(self.char1, self.char2)
            self.assertEqual(result, ['mocked_path'])
    
    def test_detect_communities_delegation(self):
        """Test that detect_communities is properly delegated."""
        with patch.object(
            self.graph_manager.graph_analytics,
            'detect_communities',
            return_value=[{'mocked_community'}]
        ) as mock_method:
            result = self.graph_manager.detect_communities()
            
            # Verify delegation occurred
            mock_method.assert_called_once()
            self.assertEqual(result, [{'mocked_community'}])
    
    def test_calculate_centrality_delegation(self):
        """Test that calculate_centrality is properly delegated."""
        with patch.object(
            self.graph_manager.graph_analytics,
            'calculate_centrality',
            return_value={'mocked': 'centrality'}
        ) as mock_method:
            result = self.graph_manager.calculate_centrality()
            
            # Verify delegation occurred
            mock_method.assert_called_once()
            self.assertEqual(result, {'mocked': 'centrality'})
    
    def test_shortest_path_between_characters_delegation(self):
        """Test that shortest_path_between_characters is properly delegated."""
        with patch.object(
            self.graph_manager.graph_analytics,
            'shortest_path_between_characters',
            return_value=['mocked_character_path']
        ) as mock_method:
            result = self.graph_manager.shortest_path_between_characters(self.char1, self.char2)
            
            # Verify delegation occurred
            mock_method.assert_called_once_with(self.char1, self.char2)
            self.assertEqual(result, ['mocked_character_path'])
    
    def test_common_interests_cluster_delegation(self):
        """Test that common_interests_cluster is properly delegated."""
        with patch.object(
            self.graph_manager.graph_analytics,
            'common_interests_cluster',
            return_value=[{'mocked_cluster'}]
        ) as mock_method:
            result = self.graph_manager.common_interests_cluster()
            
            # Verify delegation occurred
            mock_method.assert_called_once()
            self.assertEqual(result, [{'mocked_cluster'}])
    
    def test_get_filtered_nodes_delegation_basic_filters(self):
        """Test that get_filtered_nodes delegates basic filters to GraphAnalytics."""
        with patch.object(
            self.graph_manager.graph_analytics,
            'get_filtered_nodes',
            return_value={'mocked': 'filtered_nodes'}
        ) as mock_method:
            # Use basic filters that should be delegated
            result = self.graph_manager.get_filtered_nodes(
                node_type='character',
                max_distance=2
            )
            
            # Verify delegation occurred for basic filters
            mock_method.assert_called_once_with(
                node_type='character',
                max_distance=2
            )
            self.assertEqual(result, {'mocked': 'filtered_nodes'})
    
    def test_real_functionality_works(self):
        """Test that actual functionality works end-to-end."""
        # Test real pathfinding
        path = self.graph_manager.find_shortest_path(self.char1, self.char2)
        self.assertIsNotNone(path)
        self.assertIn(self.char1, path)
        self.assertIn(self.char2, path)
        
        # Test real centrality calculation
        centrality = self.graph_manager.calculate_centrality()
        self.assertIsInstance(centrality, dict)
        self.assertIn(self.char1, centrality)
        self.assertIn(self.char2, centrality)
        
        # Test real community detection
        communities = self.graph_manager.detect_communities()
        self.assertIsInstance(communities, list)
        
        # Test real filtered nodes
        characters = self.graph_manager.get_filtered_nodes(node_type='character')
        self.assertIsInstance(characters, dict)
        self.assertGreater(len(characters), 0)
    
    def test_graph_analytics_methods_available_on_graph_manager(self):
        """Test that all expected graph analytics methods are available on GraphManager."""
        expected_methods = [
            'find_shortest_path',
            'detect_communities',
            'calculate_centrality',
            'shortest_path_between_characters',
            'common_interests_cluster',
            'get_filtered_nodes'
        ]
        
        for method_name in expected_methods:
            self.assertTrue(
                hasattr(self.graph_manager, method_name),
                f"GraphManager should have method {method_name}"
            )
            self.assertTrue(
                callable(getattr(self.graph_manager, method_name)),
                f"GraphManager.{method_name} should be callable"
            )
    
    def test_backwards_compatibility(self):
        """Test that the interface remains backwards compatible."""
        # These methods should work exactly as they did before
        try:
            # Test method signatures haven't changed
            path = self.graph_manager.find_shortest_path(self.char1, self.char2)
            centrality = self.graph_manager.calculate_centrality()
            communities = self.graph_manager.detect_communities()
            char_path = self.graph_manager.shortest_path_between_characters(self.char1, self.char2)
            clusters = self.graph_manager.common_interests_cluster()
            filtered = self.graph_manager.get_filtered_nodes(node_type='character')
            
            # All should succeed without errors
            self.assertTrue(True, "All methods executed without errors")
            
        except Exception as e:
            self.fail(f"Backwards compatibility broken: {e}")


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main(verbosity=2)