import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

# Import proper MockCharacter instead of using object
sys.path.insert(0, '.')
from mock_character import MockCharacter

# Create stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_attr = ModuleType('attr')

# Use the comprehensive MockCharacter instead of object
stub_tc.Character = MockCharacter


class CrisisPromptTests(unittest.TestCase):
    def test_prompt_contains_description_and_assistant_cue(self):
        """Test that crisis prompt includes the crisis description and ends with assistant cue."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            
            # Use MockCharacter instead of local mock
            char = MockCharacter()
            builder = PromptBuilder(char)
            prompt = builder.generate_crisis_response_prompt("barn fire", urgency="high")
            
        self.assertIn("barn fire", prompt)
        self.assertTrue(prompt.strip().endswith("<|assistant|>"))
    
    def test_mock_character_works_with_prompt_builder(self):
        """
        Test that MockCharacter provides a more realistic interface than object.
        """
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            
            # Test with comprehensive MockCharacter
            char = MockCharacter(name="TestCharacter", job="farmer", wealth_money=50)
            builder = PromptBuilder(char)
            prompt = builder.generate_crisis_response_prompt("drought", urgency="high")
            
            # Should work without errors
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
            self.assertIn("drought", prompt)
            
            # Test that character attributes are accessible
            self.assertEqual(char.get_name(), "TestCharacter")
            self.assertEqual(char.get_job(), "farmer")
            self.assertEqual(char.get_wealth_money(), 50)
    
    def test_mock_character_interface_completeness(self):
        """
        Test that MockCharacter has a much more complete interface than object.
        This ensures the fix addresses the original issue.
        """
        char = MockCharacter()
        
        # Test core character methods exist
        self.assertTrue(hasattr(char, 'get_name'))
        self.assertTrue(hasattr(char, 'get_job'))
        self.assertTrue(hasattr(char, 'get_health_status'))
        self.assertTrue(hasattr(char, 'get_wealth_money'))
        self.assertTrue(hasattr(char, 'get_mental_health'))
        self.assertTrue(hasattr(char, 'get_social_wellbeing'))
        
        # Test complex object attributes exist
        self.assertTrue(hasattr(char, 'get_motives'))
        self.assertTrue(hasattr(char, 'get_inventory'))
        self.assertTrue(hasattr(char, 'get_personality_traits'))
        
        # Test that methods actually work
        self.assertIsInstance(char.get_name(), str)
        self.assertIsInstance(char.get_health_status(), (int, float))
        motives = char.get_motives()
        self.assertIsNotNone(motives)
        
        # Verify this is much more than just object
        # object has only basic methods like __str__, __repr__, etc.
        object_methods = set(dir(object()))
        char_methods = set(dir(char))
        additional_methods = char_methods - object_methods
        
        # MockCharacter should have many more methods than basic object
        self.assertGreater(len(additional_methods), 20, 
                          "MockCharacter should have significantly more methods than basic object")

if __name__ == "__main__":
    unittest.main()
