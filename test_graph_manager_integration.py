"""
Simple GraphManager test to verify the SocialModel integration works in context
"""
import sys
import unittest
sys.path.insert(0, '.')

# Mock the heavy dependencies to avoid import issues
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

# Now import the graph manager
from tiny_graph_manager import GraphManager

class TestGraphManagerWithSocialModel(unittest.TestCase):
    
    def setUp(self):
        self.graph_manager = GraphManager()
        
    def test_graph_manager_has_social_model(self):
        """Test that GraphManager has SocialModel instance"""
        self.assertTrue(hasattr(self.graph_manager, 'social_model'))
        self.assertIsNotNone(self.graph_manager.social_model)
        
    def test_social_model_world_state_connection(self):
        """Test that SocialModel has proper world_state connection"""
        self.assertEqual(self.graph_manager.social_model.world_state, self.graph_manager)
        
    def test_dynamic_weights_delegation(self):
        """Test that dynamic weights calculation is properly delegated"""
        weights = self.graph_manager.calculate_dynamic_weights(30)
        
        self.assertIsInstance(weights, dict)
        self.assertIn('openness', weights)
        self.assertIn('extraversion', weights)
        
    def test_graph_manager_initialization(self):
        """Test basic graph manager initialization"""
        self.assertIsNotNone(self.graph_manager.G)
        self.assertTrue(hasattr(self.graph_manager, 'characters'))
        self.assertTrue(hasattr(self.graph_manager, 'locations'))
        
    def test_social_methods_available(self):
        """Test that social methods are available on GraphManager"""
        # Test that the methods exist and are callable
        self.assertTrue(hasattr(self.graph_manager, 'calculate_romance_compatibility'))
        self.assertTrue(callable(self.graph_manager.calculate_romance_compatibility))
        
        self.assertTrue(hasattr(self.graph_manager, 'calculate_social_influence'))
        self.assertTrue(callable(self.graph_manager.calculate_social_influence))
        
        self.assertTrue(hasattr(self.graph_manager, 'retrieve_characters_relationships'))
        self.assertTrue(callable(self.graph_manager.retrieve_characters_relationships))
        
        self.assertTrue(hasattr(self.graph_manager, 'analyze_relationship_health'))
        self.assertTrue(callable(self.graph_manager.analyze_relationship_health))

if __name__ == '__main__':
    # Suppress logging for cleaner test output
    logging.disable(logging.CRITICAL)
    unittest.main()