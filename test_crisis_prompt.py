import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

# Create stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_attr = ModuleType('attr')

class MinimalCharacterMock:
    """
    Minimal mock that implements only the interface actually used by generate_crisis_response_prompt.
    
    This is a response to the question "Why can't we use the real Character class instead?"
    
    Reasons for using a mock instead of real Character class:
    1. Real Character has complex dependencies (GraphManager, ActionSystem, GameTimeManager, numpy, etc.)
    2. Real Character requires heavy imports that may not be available in test environments
    3. This test focuses on PromptBuilder logic, not Character implementation
    4. Mock allows fast, isolated unit testing
    
    However, this mock is much simpler than the previous version - it only implements
    what generate_crisis_response_prompt actually needs, reducing maintenance burden.
    """
    def __init__(self):
        # Only the attributes actually accessed by generate_crisis_response_prompt
        self.name = "Eve"
        self.job = "Farmer"
        self.recent_event = "outbreak"
        self.wealth_money = 10
        
        # Attributes used by _get_character_state_dict (called internally)
        self.hunger_level = 5
        self.energy = 5
        self.health_status = 7
        self.mental_health = 6
        self.social_wellbeing = 6
    
    # Some parts of PromptBuilder call getter methods instead of accessing attributes directly
    def get_hunger_level(self):
        return self.hunger_level

# Set the Character class to our minimal mock
stub_tc.Character = MinimalCharacterMock

class CrisisPromptTests(unittest.TestCase):
    def test_prompt_contains_description_and_assistant_cue(self):
        """Test that crisis prompt includes the crisis description and ends with assistant cue."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            
            # Use our minimal mock instead of complex one
            char = MinimalCharacterMock()
            builder = PromptBuilder(char)
            prompt = builder.generate_crisis_response_prompt("barn fire", urgency="high")
            
        self.assertIn("barn fire", prompt)
        self.assertTrue(prompt.strip().endswith("<|assistant|>"))

if __name__ == "__main__":
    unittest.main()
