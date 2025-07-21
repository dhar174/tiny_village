"""
Simple test for SocialModel to verify it works independently
"""
import sys
import unittest
sys.path.insert(0, '.')

from social_model import SocialModel, calculate_relationship_type

class MockWorldState:
    """Mock world state for testing"""
    def __init__(self):
        self.edges = {}
        self.nodes = {}
        
    def has_edge(self, node1, node2):
        return (node1, node2) in self.edges or (node2, node1) in self.edges
        
    def neighbors(self, node):
        return []
        
    def get_edge_data(self, node1, node2):
        return self.edges.get((node1, node2), self.edges.get((node2, node1), {}))

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

class MockJob:
    def __init__(self):
        self.location = MockLocation()

class MockLocation:
    def __init__(self):
        self.name = "Test Location"

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

class TestSocialModel(unittest.TestCase):
    
    def setUp(self):
        self.world_state = MockWorldState()
        self.social_model = SocialModel(self.world_state)
        
    def test_social_model_creation(self):
        """Test SocialModel can be created"""
        self.assertIsInstance(self.social_model, SocialModel)
        self.assertEqual(self.social_model.world_state, self.world_state)
        
    def test_calculate_dynamic_weights(self):
        """Test dynamic weights calculation"""
        # Test initial stage
        weights = self.social_model.calculate_dynamic_weights(10)
        self.assertIsInstance(weights, dict)
        self.assertIn('openness', weights)
        self.assertEqual(weights['openness'], 0.2)
        
        # Test middle stage
        weights = self.social_model.calculate_dynamic_weights(40)
        self.assertEqual(weights['openness'], 0.15)
        
        # Test mature stage
        weights = self.social_model.calculate_dynamic_weights(80)
        self.assertEqual(weights['openness'], 0.1)
        
    def test_calculate_romance_compatibility(self):
        """Test romance compatibility calculation"""
        char1 = MockCharacter("Alice", {'openness': 6, 'extraversion': 7})
        char2 = MockCharacter("Bob", {'openness': 5, 'extraversion': 6})
        
        compatibility = self.social_model.calculate_romance_compatibility(char1, char2, 30)
        
        self.assertIsInstance(compatibility, float)
        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
        
    def test_calculate_romance_interest(self):
        """Test romance interest calculation"""
        char1 = MockCharacter("Alice")
        char2 = MockCharacter("Bob")
        
        interest = self.social_model.calculate_romance_interest(
            char1, char2, 0.7, 5, "friend", 0.8, 50, 0.6, 0.4, 0.3
        )
        
        self.assertIsInstance(interest, float)
        self.assertGreaterEqual(interest, 0.0)
        self.assertLessEqual(interest, 1.0)
        
    def test_calculate_relationship_type(self):
        """Test relationship type calculation"""
        char1 = MockCharacter("Alice")
        char2 = MockCharacter("Bob")
        
        rel_type = calculate_relationship_type(
            char1, char2, 0.6, 0.4, 0.7, 0.8, 60
        )
        
        self.assertIsInstance(rel_type, str)
        
    def test_calculate_social_influence(self):
        """Test social influence calculation"""
        influence = self.social_model.calculate_social_influence("Alice")
        
        # Should return 0 since no relationships exist in mock
        self.assertEqual(influence, 0)
        
    def test_retrieve_characters_relationships(self):
        """Test relationship retrieval"""
        relationships = self.social_model.retrieve_characters_relationships("Alice")
        
        # Should return empty dict since no relationships exist in mock
        self.assertEqual(relationships, {})

if __name__ == '__main__':
    unittest.main()