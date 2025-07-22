"""
Unit tests for the GraphAnalytics class.

This test module verifies that the GraphAnalytics class properly provides
graph analysis and querying utilities for the tiny_village simulation.
"""

import sys
import unittest
from unittest.mock import Mock, patch
import logging

sys.path.insert(0, '.')

# Import the modules to test
from graph_analytics import GraphAnalytics
from world_state import WorldState


class TestGraphAnalytics(unittest.TestCase):
    """Test cases for GraphAnalytics class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.world_state = WorldState()
        self.graph_analytics = GraphAnalytics(self.world_state)
        
        # Create some test nodes and edges
        self.setup_test_graph()
    
    def setup_test_graph(self):
        """Create a test graph with sample nodes and edges."""
        # Create mock character objects
        self.char1 = Mock()
        self.char1.name = "Alice"
        self.char1.age = 25
        
        self.char2 = Mock()
        self.char2.name = "Bob"
        self.char2.age = 30
        
        self.char3 = Mock()
        self.char3.name = "Charlie"
        self.char3.age = 28
        
        # Create mock location objects
        self.loc1 = Mock()
        self.loc1.name = "Park"
        self.loc1.popularity = 8
        
        self.loc2 = Mock()
        self.loc2.name = "Library"
        self.loc2.popularity = 6
        
        # Create mock activity objects
        self.activity1 = Mock()
        self.activity1.name = "Reading"
        
        self.activity2 = Mock()
        self.activity2.name = "Jogging"
        
        # Add nodes to the graph
        self.world_state.add_character_node(self.char1)
        self.world_state.add_character_node(self.char2)
        self.world_state.add_character_node(self.char3)
        self.world_state.add_location_node(self.loc1)
        self.world_state.add_location_node(self.loc2)
        self.world_state.add_activity_node(self.activity1)
        self.world_state.add_activity_node(self.activity2)
        
        # Add some edges to create connections
        self.world_state.add_edge(self.char1, self.char2, "friendship", weight=0.8)
        self.world_state.add_edge(self.char2, self.char3, "friendship", weight=0.6)
        self.world_state.add_edge(self.char1, self.loc1, "visits")
        self.world_state.add_edge(self.char2, self.loc1, "visits")
        self.world_state.add_edge(self.char1, self.activity1, "participates")
        self.world_state.add_edge(self.char2, self.activity1, "participates")
        self.world_state.add_edge(self.char3, self.activity2, "participates")
    
    def test_initialization(self):
        """Test that GraphAnalytics initializes properly with WorldState dependency."""
        self.assertIsNotNone(self.graph_analytics.world_state)
        self.assertEqual(self.graph_analytics.world_state, self.world_state)
        self.assertEqual(self.graph_analytics.graph, self.world_state.graph)
    
    def test_find_shortest_path_existing_path(self):
        """Test finding shortest path between connected nodes."""
        path = self.graph_analytics.find_shortest_path(self.char1, self.char2)
        self.assertIsNotNone(path)
        self.assertIn(self.char1, path)
        self.assertIn(self.char2, path)
        self.assertEqual(len(path), 2)  # Direct connection
    
    def test_find_shortest_path_indirect_path(self):
        """Test finding shortest path between indirectly connected nodes."""
        path = self.graph_analytics.find_shortest_path(self.char1, self.char3)
        self.assertIsNotNone(path)
        self.assertIn(self.char1, path)
        self.assertIn(self.char3, path)
        # Path should go through char2
        self.assertEqual(len(path), 3)
    
    def test_find_shortest_path_no_path(self):
        """Test finding shortest path when no path exists."""
        # Create an isolated node
        isolated_char = Mock()
        isolated_char.name = "Isolated"
        self.world_state.add_character_node(isolated_char)
        
        path = self.graph_analytics.find_shortest_path(self.char1, isolated_char)
        self.assertIsNone(path)
    
    def test_find_shortest_path_same_node(self):
        """Test finding shortest path from a node to itself."""
        path = self.graph_analytics.find_shortest_path(self.char1, self.char1)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], self.char1)
    
    def test_detect_communities(self):
        """Test community detection functionality."""
        communities = self.graph_analytics.detect_communities()
        self.assertIsInstance(communities, list)
        # With our test graph, we should get at least one community
        if communities:
            for community in communities:
                self.assertIsInstance(community, set)
    
    def test_calculate_centrality(self):
        """Test centrality calculation."""
        centrality = self.graph_analytics.calculate_centrality()
        self.assertIsInstance(centrality, dict)
        
        # All nodes should have centrality scores
        for node in self.world_state.get_nodes():
            self.assertIn(node, centrality)
            self.assertIsInstance(centrality[node], (int, float))
            self.assertGreaterEqual(centrality[node], 0)
            self.assertLessEqual(centrality[node], 1)
    
    def test_shortest_path_between_characters(self):
        """Test shortest path between characters (alias method)."""
        path = self.graph_analytics.shortest_path_between_characters(self.char1, self.char2)
        self.assertIsNotNone(path)
        self.assertIn(self.char1, path)
        self.assertIn(self.char2, path)
    
    def test_common_interests_cluster(self):
        """Test clustering by common interests."""
        clusters = self.graph_analytics.common_interests_cluster()
        self.assertIsInstance(clusters, list)
        
        # Each cluster should be a set of characters
        for cluster in clusters:
            self.assertIsInstance(cluster, set)
            # Clusters should have more than one character
            self.assertGreater(len(cluster), 1)
    
    def test_get_filtered_nodes_by_type(self):
        """Test filtering nodes by type."""
        characters = self.graph_analytics.get_filtered_nodes(node_type='character')
        self.assertIsInstance(characters, dict)
        
        # Should only return character nodes
        for node, attrs in characters.items():
            self.assertEqual(attrs.get('type'), 'character')
        
        # Should include our test characters
        character_names = [attrs.get('name') for attrs in characters.values()]
        self.assertIn('Alice', character_names)
        self.assertIn('Bob', character_names)
        self.assertIn('Charlie', character_names)
    
    def test_get_filtered_nodes_by_attributes(self):
        """Test filtering nodes by attributes."""
        # Filter by a specific attribute value
        young_chars = self.graph_analytics.get_filtered_nodes(
            node_type='character',
            node_attributes={'age': 25}
        )
        
        # Should return Alice who is 25
        self.assertEqual(len(young_chars), 1)
        alice_node = list(young_chars.keys())[0]
        self.assertEqual(alice_node.name, 'Alice')
    
    def test_get_neighbors_by_type(self):
        """Test getting neighbors filtered by type."""
        # Get character neighbors of char1
        char_neighbors = self.graph_analytics.get_neighbors_by_type(self.char1, 'character')
        self.assertIsInstance(char_neighbors, list)
        self.assertIn(self.char2, char_neighbors)
        
        # Get location neighbors of char1
        loc_neighbors = self.graph_analytics.get_neighbors_by_type(self.char1, 'location')
        self.assertIsInstance(loc_neighbors, list)
        self.assertIn(self.loc1, loc_neighbors)
    
    def test_analyze_node_connectivity(self):
        """Test node connectivity analysis."""
        connectivity = self.graph_analytics.analyze_node_connectivity(self.char1)
        self.assertIsInstance(connectivity, dict)
        
        # Should contain expected connectivity metrics
        self.assertIn('degree', connectivity)
        self.assertIn('neighbor_count', connectivity)
        self.assertIn('neighbor_types', connectivity)
        self.assertIn('neighbors', connectivity)
        
        # Verify the data makes sense
        self.assertGreater(connectivity['degree'], 0)
        self.assertEqual(connectivity['neighbor_count'], len(connectivity['neighbors']))
    
    def test_find_nodes_within_distance(self):
        """Test finding nodes within a specific distance."""
        nearby_nodes = self.graph_analytics.find_nodes_within_distance(self.char1, 1)
        self.assertIsInstance(nearby_nodes, dict)
        
        # Should include the source node at distance 0
        self.assertIn(self.char1, nearby_nodes)
        self.assertEqual(nearby_nodes[self.char1], 0)
        
        # Should include direct neighbors at distance 1
        self.assertIn(self.char2, nearby_nodes)
        self.assertEqual(nearby_nodes[self.char2], 1)
    
    def test_get_graph_statistics(self):
        """Test getting graph statistics."""
        stats = self.graph_analytics.get_graph_statistics()
        self.assertIsInstance(stats, dict)
        
        # Should contain expected statistics
        self.assertIn('node_count', stats)
        self.assertIn('edge_count', stats)
        self.assertIn('node_types', stats)
        self.assertIn('edge_types', stats)
        
        # Verify counts make sense
        self.assertGreater(stats['node_count'], 0)
        self.assertGreater(stats['edge_count'], 0)
        self.assertIn('character', stats['node_types'])
        self.assertIn('location', stats['node_types'])
    
    def test_empty_graph_behavior(self):
        """Test behavior with an empty graph."""
        empty_world_state = WorldState()
        empty_analytics = GraphAnalytics(empty_world_state)
        
        # Should handle empty graph gracefully
        centrality = empty_analytics.calculate_centrality()
        self.assertEqual(centrality, {})
        
        communities = empty_analytics.detect_communities()
        self.assertEqual(communities, [])
        
        path = empty_analytics.find_shortest_path("nonexistent1", "nonexistent2")
        self.assertIsNone(path)
    
    def test_error_handling(self):
        """Test error handling in various methods."""
        # Test with non-existent nodes
        path = self.graph_analytics.find_shortest_path("nonexistent", self.char1)
        self.assertIsNone(path)
        
        connectivity = self.graph_analytics.analyze_node_connectivity("nonexistent")
        self.assertEqual(connectivity, {})
        
        neighbors = self.graph_analytics.get_neighbors_by_type("nonexistent")
        self.assertEqual(neighbors, [])


class TestGraphAnalyticsWithMockedNetworkX(unittest.TestCase):
    """Test GraphAnalytics behavior when NetworkX is not available."""
    
    def setUp(self):
        """Set up test with mocked NetworkX unavailability."""
        self.world_state = WorldState()
    
    @patch('graph_analytics.NETWORKX_AVAILABLE', False)
    def test_fallback_find_shortest_path(self):
        """Test fallback pathfinding when NetworkX is unavailable."""
        graph_analytics = GraphAnalytics(self.world_state)
        
        # Should use fallback implementation
        path = graph_analytics.find_shortest_path("source", "target")
        self.assertEqual(path, ["source", "target"])
        
        # Same node should return single-node path
        path = graph_analytics.find_shortest_path("node", "node")
        self.assertEqual(path, ["node"])
    
    @patch('graph_analytics.NETWORKX_COMMUNITY_AVAILABLE', False)
    def test_fallback_detect_communities(self):
        """Test fallback community detection when NetworkX community is unavailable."""
        graph_analytics = GraphAnalytics(self.world_state)
        
        # Add some test nodes
        char1 = Mock()
        char1.name = "Alice"
        char2 = Mock()
        char2.name = "Bob"
        loc1 = Mock()
        loc1.name = "Park"
        
        self.world_state.add_character_node(char1)
        self.world_state.add_character_node(char2)
        self.world_state.add_location_node(loc1)
        
        communities = graph_analytics.detect_communities()
        
        # Should group nodes by type as fallback
        self.assertIsInstance(communities, list)
        if communities:
            # Should have separate communities for characters and locations
            character_community = None
            location_community = None
            
            for community in communities:
                if any(self.world_state.graph.nodes[node].get('type') == 'character' for node in community):
                    character_community = community
                if any(self.world_state.graph.nodes[node].get('type') == 'location' for node in community):
                    location_community = community
            
            if character_community:
                self.assertIn(char1, character_community)
                self.assertIn(char2, character_community)
            if location_community:
                self.assertIn(loc1, location_community)
    
    @patch('graph_analytics.NETWORKX_AVAILABLE', False)
    def test_fallback_calculate_centrality(self):
        """Test fallback centrality calculation when NetworkX is unavailable."""
        graph_analytics = GraphAnalytics(self.world_state)
        
        # Add test nodes and edges
        char1 = Mock()
        char1.name = "Alice"
        char2 = Mock()
        char2.name = "Bob"
        
        self.world_state.add_character_node(char1)
        self.world_state.add_character_node(char2)
        self.world_state.add_edge(char1, char2, "friendship")
        
        centrality = graph_analytics.calculate_centrality()
        
        # Should use degree-based fallback calculation
        self.assertIsInstance(centrality, dict)
        self.assertIn(char1, centrality)
        self.assertIn(char2, centrality)
        
        # Both should have the same centrality (degree 1, normalized by n-1 = 1)
        self.assertEqual(centrality[char1], 1.0)
        self.assertEqual(centrality[char2], 1.0)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main(verbosity=2)