"""
Integration test for SocialModel with GraphManager
This test verifies that the GraphManager properly delegates social simulation tasks to SocialModel
"""
import sys
import unittest
sys.path.insert(0, '.')

from social_model import SocialModel

class MockGraphManager:
    """Mock GraphManager for testing SocialModel integration"""
    def __init__(self):
        self.G = MockGraph()
        self.social_model = SocialModel(world_state=self)
        
    def has_edge(self, node1, node2):
        return self.G.has_edge(node1, node2)
        
    def neighbors(self, node):
        return self.G.neighbors(node)
        
    def get_edge_data(self, node1, node2):
        return self.G.get_edge_data(node1, node2)

class MockGraph:
    """Mock graph for testing"""
    def __init__(self):
        self.edges = {
            ("Alice", "Bob"): {
                "relationship_type": "friend",
                "trust": 0.8,
                "emotional": 0.6,
                "strength": 0.7,
                "historical": 45,
                "interaction_frequency": 0.5
            }
        }
        self.node_neighbors = {
            "Alice": ["Bob"],
            "Bob": ["Alice"]
        }
        self.nodes = {
            "Alice": {"type": "character", "name": "Alice"},
            "Bob": {"type": "character", "name": "Bob"}
        }
        
    def has_edge(self, node1, node2):
        return (node1, node2) in self.edges or (node2, node1) in self.edges
        
    def neighbors(self, node):
        return self.node_neighbors.get(node, [])
        
    def get_edge_data(self, node1, node2):
        return self.edges.get((node1, node2), self.edges.get((node2, node1), {}))
        
    def __getitem__(self, node):
        """Allow graph[node1][node2] syntax"""
        return MockNodeEdges(node, self.edges, self.node_neighbors)

class MockNodeEdges:
    """Mock for graph[node1][node2] access"""
    def __init__(self, node, edges, neighbors):
        self.node = node
        self.edges = edges
        self.neighbors = neighbors.get(node, [])
        
    def __getitem__(self, other_node):
        key = (self.node, other_node)
        alt_key = (other_node, self.node)
        
        if key in self.edges:
            edge_data = self.edges[key]
            return MockEdgeData(edge_data, self.edges, key)
        elif alt_key in self.edges:
            edge_data = self.edges[alt_key]
            return MockEdgeData(edge_data, self.edges, alt_key)
        else:
            return MockEdgeData({}, self.edges, key)
        
class MockEdgeData:
    """Mock for edge data that allows both dict-like and attribute access"""
    def __init__(self, data, edges, key):
        self.data = data
        self.edges = edges
        self.key = key
        
    def __getitem__(self, item_key):
        return self.data[item_key]
        
    def __setitem__(self, item_key, value):
        self.data[item_key] = value
        # Update the main edges dict as well
        self.edges[self.key][item_key] = value
        
    def get(self, item_key, default=None):
        return self.data.get(item_key, default)

class MockCharacter:
    """Mock character for testing"""
    def __init__(self, name, traits=None):
        self.name = name
        self.personality_traits = MockPersonalityTraits(traits or {})
        self.age = 25
        self.beauty = 5
        self.energy = 50
        self.wealth_money = 1000
        self.stability = 5
        self.luxury = 3
        self.monogamy = 8
        self.shelter = 7
        self.success = 4
        self.job = MockJob()
        self.home = MockLocation()
        
    def get_motives(self):
        return MockMotives()
        
    def get_base_libido(self):
        return 50
        
    def get_control(self):
        return 5

class MockPersonalityTraits:
    def __init__(self, traits):
        self.traits = traits
        
    def get_openness(self):
        return self.traits.get('openness', 5)
        
    def get_extraversion(self):
        return self.traits.get('extraversion', 5)
        
    def get_conscientiousness(self):
        return self.traits.get('conscientiousness', 5)
        
    def get_agreeableness(self):
        return self.traits.get('agreeableness', 5)
        
    def get_neuroticism(self):
        return self.traits.get('neuroticism', 5)

class MockMotives:
    def get_wealth_motive(self):
        return MockMotive(3)
        
    def get_family_motive(self):
        return MockMotive(7)
        
    def get_beauty_motive(self):
        return MockMotive(4)
        
    def get_luxury_motive(self):
        return MockMotive(2)
        
    def get_stability_motive(self):
        return MockMotive(6)
        
    def get_control_motive(self):
        return MockMotive(4)
        
    def get_material_goods_motive(self):
        return MockMotive(3)
        
    def get_shelter_motive(self):
        return MockMotive(8)

class MockMotive:
    def __init__(self, score):
        self.score = score

class MockJob:
    def __init__(self):
        self.location = MockLocation()

class MockLocation:
    def __init__(self):
        self.name = "Test Location"

class TestSocialModelIntegration(unittest.TestCase):
    
    def setUp(self):
        self.graph_manager = MockGraphManager()
        
    def test_graph_manager_has_social_model(self):
        """Test that GraphManager has SocialModel instance"""
        self.assertIsInstance(self.graph_manager.social_model, SocialModel)
        self.assertEqual(self.graph_manager.social_model.world_state, self.graph_manager)
        
    def test_social_model_accesses_graph_data(self):
        """Test that SocialModel can access graph data through world_state"""
        # Test relationship retrieval
        relationships = self.graph_manager.social_model.retrieve_characters_relationships("Alice")
        
        # Should find the relationship we set up
        self.assertIn("Bob", relationships)
        self.assertEqual(relationships["Bob"]["relationship_type"], "friend")
        self.assertEqual(relationships["Bob"]["trust"], 0.8)
        
    def test_social_influence_calculation(self):
        """Test social influence calculation with mock data"""
        influence = self.graph_manager.social_model.calculate_social_influence("Alice")
        
        # Should calculate some influence based on the friend relationship
        self.assertIsInstance(influence, (int, float))
        
    def test_relationship_status_update(self):
        """Test updating relationship status"""
        # Update trust level
        self.graph_manager.social_model.update_relationship_status(
            "Alice", "Bob", {"trust": 0.1}
        )
        
        # Verify update was applied to the graph
        edge_data = self.graph_manager.G.get_edge_data("Alice", "Bob")
        self.assertEqual(edge_data["trust"], 0.9)  # 0.8 + 0.1
        
    def test_relationship_analysis(self):
        """Test character relationship analysis"""
        analysis = self.graph_manager.social_model.analyze_character_relationships("Alice")
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("total_relationships", analysis)
        self.assertEqual(analysis["total_relationships"], 1)
        self.assertIn("relationship_types", analysis)
        self.assertIn("friend", analysis["relationship_types"])
        
    def test_romance_compatibility(self):
        """Test romance compatibility calculation"""
        char1 = MockCharacter("Alice", {'openness': 6, 'extraversion': 7})
        char2 = MockCharacter("Bob", {'openness': 5, 'extraversion': 6})
        
        compatibility = self.graph_manager.social_model.calculate_romance_compatibility(
            char1, char2, 30
        )
        
        self.assertIsInstance(compatibility, float)
        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
        
    def test_romance_interest(self):
        """Test romance interest calculation"""
        char1 = MockCharacter("Alice")
        char2 = MockCharacter("Bob")
        
        interest = self.graph_manager.social_model.calculate_romance_interest(
            char1, char2, 0.7, 5, "friend", 0.8, 50, 0.6, 0.4, 0.3
        )
        
        self.assertIsInstance(interest, float)
        self.assertGreaterEqual(interest, 0.0)
        self.assertLessEqual(interest, 1.0)
        
    def test_dynamic_weights(self):
        """Test dynamic weights calculation"""
        weights = self.graph_manager.social_model.calculate_dynamic_weights(30)
        
        self.assertIsInstance(weights, dict)
        expected_keys = [
            'openness', 'extraversion', 'conscientiousness', 
            'agreeableness', 'neuroticism'
        ]
        for key in expected_keys:
            self.assertIn(key, weights)
            self.assertIsInstance(weights[key], (int, float))
            
    def test_relationship_health_analysis(self):
        """Test relationship health analysis"""
        health = self.graph_manager.social_model.analyze_relationship_health("Alice", "Bob")
        
        self.assertIsInstance(health, dict)
        self.assertIn("health_score", health)
        self.assertIn("status", health)
        self.assertIsInstance(health["health_score"], (int, float))
        
    def test_relationship_strength_evaluation(self):
        """Test relationship strength evaluation"""
        strength = self.graph_manager.social_model.evaluate_relationship_strength("Alice", "Bob")
        
        self.assertIsInstance(strength, (int, float))
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

if __name__ == '__main__':
    unittest.main()