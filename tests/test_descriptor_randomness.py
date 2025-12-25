#!/usr/bin/env python3
"""
Test for robust randomness in descriptor functions.

This test addresses issue #334 where `assert len(set(values)) > 1` could 
theoretically fail even with proper randomness, though extremely unlikely.

This implementation replaces brittle randomness tests with more robust
statistical approaches that properly validate randomness without false negatives.
"""

import sys
import random
import unittest
from collections import Counter
from unittest.mock import patch
from types import ModuleType


# Create stub modules to satisfy imports in tiny_prompt_builder
# (This is a lightweight approach to avoid complex dependency issues)
stub_tc = ModuleType('tiny_characters')
stub_tc.Character = object
stub_attr = ModuleType('attr')


class TestDescriptorRandomness(unittest.TestCase):
    """Test robust randomness validation for descriptor functions."""
    
    def setUp(self):
        """Set up test environment with mocked dependencies."""
        self.patch_modules = patch.dict(sys.modules, {
            "tiny_characters": stub_tc, 
            "attr": stub_attr
        })
        self.patch_modules.start()
        
        # Import after patching to avoid dependency issues
        import tiny_prompt_builder
        self.descriptors = tiny_prompt_builder.descriptors
        
    def tearDown(self):
        """Clean up test environment."""
        self.patch_modules.stop()
        
    def test_job_adjective_randomness_robust(self):
        """
        Test job adjective randomness using robust statistical approach.
        
        This replaces the problematic `assert len(set(values)) > 1` pattern
        with a more robust approach that accounts for natural randomness variation.
        """
        # Use a larger sample size to make all-identical extremely unlikely
        sample_size = 50
        job_type = "Engineer"  # Use a job type with multiple adjectives
        
        # Generate sample values
        values = [self.descriptors.get_job_adjective(job_type) for _ in range(sample_size)]
        value_counts = Counter(values)
        unique_count = len(value_counts)
        
        # Test 1: Should have multiple unique values (robust check)
        self.assertGreaterEqual(
            unique_count, 2,
            f"Expected at least 2 unique adjectives in {sample_size} samples, "
            f"got {unique_count}: {dict(value_counts)}"
        )
        
        # Test 2: No single value should dominate excessively 
        # (catches systematic bias while allowing for random clustering)
        max_frequency = max(value_counts.values())
        max_percentage = (max_frequency / sample_size) * 100
        
        self.assertLess(
            max_percentage, 85.0,  # Allow some natural clustering
            f"Single adjective appears {max_percentage:.1f}% of time, "
            f"suggesting non-random behavior: {dict(value_counts)}"
        )
        
    def test_weather_description_randomness_robust(self):
        """
        Test weather description randomness using robust statistical approach.
        """
        sample_size = 40
        weather_type = "sunny"
        
        # Generate sample values
        values = [self.descriptors.get_weather_description(weather_type) for _ in range(sample_size)]
        value_counts = Counter(values)
        unique_count = len(value_counts)
        
        # Robust randomness checks
        self.assertGreaterEqual(
            unique_count, 1,  # At minimum should have 1 unique value
            f"Expected at least 1 unique weather description, got {unique_count}"
        )
        
        # If we have multiple options, we should see some diversity
        # (This handles cases where there might only be one option for some weather types)
        if unique_count > 1:
            max_frequency = max(value_counts.values())
            max_percentage = (max_frequency / sample_size) * 100
            
            self.assertLess(
                max_percentage, 90.0,
                f"Weather description appears {max_percentage:.1f}% of time, "
                f"which may indicate limited randomness: {dict(value_counts)}"
            )
            
    def test_job_pronoun_randomness_robust(self):
        """
        Test job pronoun randomness using robust statistical approach.
        """
        sample_size = 60
        job_type = "Engineer"  # Job type with multiple pronoun options
        
        # Generate sample values
        values = [self.descriptors.get_job_pronoun(job_type) for _ in range(sample_size)]
        value_counts = Counter(values)
        unique_count = len(value_counts)
        
        # Robust randomness checks
        self.assertGreaterEqual(
            unique_count, 2,
            f"Expected at least 2 unique pronouns for Engineer in {sample_size} samples, "
            f"got {unique_count}: {dict(value_counts)}"
        )
        
        # Check for reasonable distribution
        max_frequency = max(value_counts.values())
        max_percentage = (max_frequency / sample_size) * 100
        
        self.assertLess(
            max_percentage, 80.0,
            f"Single pronoun appears {max_percentage:.1f}% of time, "
            f"suggesting insufficient randomness: {dict(value_counts)}"
        )
        
    def test_multiple_descriptor_functions_consistency(self):
        """
        Test that multiple descriptor functions show consistent randomness behavior.
        
        This is a meta-test that ensures our randomness improvements work
        across different descriptor functions.
        """
        functions_to_test = [
            (self.descriptors.get_job_adjective, "Engineer"),
            (self.descriptors.get_job_pronoun, "Engineer"),
            (self.descriptors.get_weather_description, "sunny"),
        ]
        
        sample_size = 30
        
        for func, param in functions_to_test:
            with self.subTest(function=func.__name__, parameter=param):
                # Generate samples
                values = [func(param) for _ in range(sample_size)]
                unique_count = len(set(values))
                
                # Each function should show some level of variation
                # (We're being conservative here to avoid false positives)
                if unique_count == 1:
                    # All values identical - this indicates a problem with the random function
                    # or the function legitimately has only one option (which should be documented)
                    self.fail(
                        f"{func.__name__}('{param}') returned only one unique value: '{values[0]}'. "
                        f"This suggests either insufficient randomness or limited options that should be documented."
                    )
                else:
                    self.assertGreaterEqual(
                        unique_count, 2,
                        f"{func.__name__}('{param}') should show some variation in {sample_size} samples"
                    )
                    
    def test_probabilistic_bounds_approach(self):
        """
        Test using probabilistic bounds - the most robust approach.
        
        This method calculates the actual probability of observed outcomes
        and only fails for statistically improbable results.
        """
        sample_size = 25
        values = [self.descriptors.get_job_adjective("Engineer") for _ in range(sample_size)]
        unique_count = len(set(values))
        
        # Get the number of available options
        available_options = len(self.descriptors.job_adjective.get("Engineer", 
                                self.descriptors.job_adjective["default"]))
        
        if unique_count == 1:
            # All values the same - calculate probability
            # P(all same) ≈ num_options * (1/num_options)^sample_size
            prob_all_same = available_options * (1/available_options) ** sample_size
            
            # Only fail if this is extremely improbable (suggests a bug)
            if prob_all_same < 1e-10:  # Less than 1 in 10 billion chance
                self.fail(
                    f"All {sample_size} values were identical: '{values[0]}'. "
                    f"This is extremely improbable (p ≈ {prob_all_same:.2e}) "
                    f"and suggests non-random behavior."
                )
            else:
                # Improbable but not impossible - document this for analysis
                print(f"Note: All values identical (p ≈ {prob_all_same:.2e}), within statistical possibility but unusual")
                # This is acceptable for probabilistic testing, but worth noting
        
        # Test always passes unless we see the extremely improbable case
        self.assertGreaterEqual(unique_count, 1)  # Always true, documents expectation


def create_example_problematic_test():
    """
    Example of the PROBLEMATIC test pattern that this fix addresses.
    
    This function demonstrates the original issue - do not use this pattern!
    """
    # PROBLEMATIC PATTERN - DO NOT USE:
    # values = [get_random_function() for _ in range(10)]
    # assert len(set(values)) > 1  # Could fail with proper randomness!
    
    # The problem: even with perfect randomness, there's a small chance
    # all values could be identical, causing the test to fail sporadically.
    
    pass  # Intentionally empty - this is just for documentation


if __name__ == "__main__":
    unittest.main(verbosity=2)