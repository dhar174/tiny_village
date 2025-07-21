import sys
import unittest
from unittest.mock import patch
from types import ModuleType

# Create stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_tc.Character = object
stub_attr = ModuleType('attr')

class DescriptorFallbackTests(unittest.TestCase):
    def test_event_recent_fallback_to_default(self):
        """Test that unknown recent events fall back to default value."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            result = tiny_prompt_builder.descriptors.get_event_recent('unknown_event')
            # Should return the default value, not raise KeyError
            self.assertEqual(result, "Recently")

    def test_financial_situation_fallback_to_default(self):
        """Test that unknown financial situations fall back to default value."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            result = tiny_prompt_builder.descriptors.get_financial_situation('unknown_status')
            # Should return the default value, not raise KeyError
            self.assertEqual(result, "financially, you are doing okay")

    def test_known_event_recent_values(self):
        """Test that known recent events return expected values."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            # Test a known value
            result = tiny_prompt_builder.descriptors.get_event_recent('outbreak')
            self.assertEqual(result, "With the recent outbreak")

    def test_known_financial_situation_values(self):
        """Test that known financial situations return expected values."""
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            # Test known values - we need to check if they exist in the expected list
            result = tiny_prompt_builder.descriptors.get_financial_situation('poor')
            self.assertIn(result, [
                "you are financially poor",
                "you are financially struggling", 
                "you are financially unstable",
                "you are financially insecure",
                "you are financially uncomfortable",
                "you are financially squeezed",
                "your finances are tight",
                "you are financially strapped",
                "you are financially stressed",
                "you are financially burdened",
                "you are struggling to make ends meet",
                "you are struggling to get by",
                "you are struggling to get through financially",
                "you are struggling to pay the bills",
                "you are struggling to pay the rent",
                "you are broke",
                "you are in debt",
                "you are in the red",
                "you are in the hole",
                "you are in the negative",
            ])

if __name__ == "__main__":
    unittest.main()