#!/usr/bin/env python3
"""
Example of migrating problematic randomness tests to robust versions.

This file demonstrates how to upgrade existing tests that use the problematic
`assert len(set(values)) > 1` pattern to more robust statistical approaches.
"""

import random
import unittest
from collections import Counter


class ExampleMigration:
    """Example showing before/after patterns for randomness testing."""
    
    def __init__(self):
        self.random_adjectives = ["skilled", "hardworking", "friendly", "creative", "dedicated"]
    
    def get_random_adjective(self):
        """Mock function that returns random adjectives."""
        return random.choice(self.random_adjectives)


class ProblematicTestExample(unittest.TestCase):
    """Example of PROBLEMATIC test patterns that should be avoided."""
    
    def setUp(self):
        self.example = ExampleMigration()
    
    def test_randomness_OLD_PROBLEMATIC_PATTERN(self):
        """
        ❌ PROBLEMATIC: This test could fail even with proper randomness!
        
        The pattern `assert len(set(values)) > 1` can cause flaky tests
        because there's always a small chance all random values are identical.
        """
        # ❌ DON'T USE THIS PATTERN:
        values = [self.example.get_random_adjective() for _ in range(10)]
        
        # This assertion could theoretically fail with proper randomness!
        # Probability with 5 choices: (1/5)^9 ≈ 0.0000512%
        # While rare, this causes intermittent test failures
        
        # assert len(set(values)) > 1  # COMMENTED OUT - DON'T USE!
        
        # For demonstration, we'll just show what the old test would do:
        unique_count = len(set(values))
        print(f"Old test would check: unique_count ({unique_count}) > 1")
        if unique_count <= 1:
            print("❌ Old test would FAIL here, even with proper randomness!")
        else:
            print("✅ Old test would pass")


class RobustTestExample(unittest.TestCase):
    """Example of ROBUST test patterns that should be used instead."""
    
    def setUp(self):
        self.example = ExampleMigration()
    
    def test_randomness_ROBUST_LARGE_SAMPLE(self):
        """
        ✅ ROBUST: Use larger sample size to make identical values effectively impossible.
        """
        # Use much larger sample size
        values = [self.example.get_random_adjective() for _ in range(100)]
        unique_count = len(set(values))
        
        # With 100 samples and 5 choices, probability of all same: (1/5)^99 ≈ 10^-70
        # This is so small it's effectively impossible
        self.assertGreater(
            unique_count, 1,
            f"Expected diverse values with large sample (n=100), got: {unique_count} unique values"
        )
        
    def test_randomness_ROBUST_STATISTICAL(self):
        """
        ✅ ROBUST: Use statistical bounds that allow for natural randomness variation.
        """
        sample_size = 50
        values = [self.example.get_random_adjective() for _ in range(sample_size)]
        value_counts = Counter(values)
        unique_count = len(value_counts)
        
        # Test 1: Should have multiple unique values (more forgiving)
        self.assertGreaterEqual(
            unique_count, 2,
            f"Expected at least 2 unique values, got {unique_count}: {dict(value_counts)}"
        )
        
        # Test 2: No single value should dominate excessively
        max_frequency = max(value_counts.values())
        max_percentage = (max_frequency / sample_size) * 100
        
        self.assertLess(
            max_percentage, 80.0,
            f"Single value appears {max_percentage:.1f}% of time, "
            f"suggesting non-random behavior: {dict(value_counts)}"
        )
        
    def test_randomness_ROBUST_PROBABILISTIC(self):
        """
        ✅ ROBUST: Use probabilistic bounds that only fail for truly improbable outcomes.
        """
        sample_size = 30
        values = [self.example.get_random_adjective() for _ in range(sample_size)]
        unique_count = len(set(values))
        
        if unique_count == 1:
            # All values the same - calculate probability
            num_choices = len(self.example.random_adjectives)
            prob_all_same = num_choices * (1/num_choices) ** sample_size
            
            # Only fail if extremely improbable (suggests a bug in randomness)
            if prob_all_same < 1e-10:  # Less than 1 in 10 billion
                self.fail(
                    f"All {sample_size} values were identical: '{values[0]}'. "
                    f"This is extremely improbable (p ≈ {prob_all_same:.2e}) "
                    f"and suggests non-random behavior."
                )
            else:
                # Improbable but not impossible - just warn
                print(f"Warning: All values identical (p ≈ {prob_all_same:.2e}), "
                      f"but within statistical possibility")
        
        # Test passes unless we see the extremely improbable case
        self.assertGreaterEqual(unique_count, 1)


def demonstrate_migration():
    """Show practical migration examples."""
    print("🔄 MIGRATION EXAMPLES")
    print("=" * 50)
    
    print("\n❌ OLD PROBLEMATIC PATTERN:")
    print("```python")
    print("values = [get_random_function() for _ in range(10)]")
    print("assert len(set(values)) > 1  # Could fail with proper randomness!")
    print("```")
    
    print("\n✅ NEW ROBUST PATTERNS:")
    print("\n1. LARGER SAMPLE SIZE (Simplest fix):")
    print("```python")
    print("values = [get_random_function() for _ in range(100)]  # Much larger sample")
    print("assert len(set(values)) > 1  # Now effectively impossible to fail")
    print("```")
    
    print("\n2. STATISTICAL BOUNDS (Recommended):")
    print("```python")
    print("values = [get_random_function() for _ in range(50)]")
    print("value_counts = Counter(values)")
    print("assert len(value_counts) >= 2  # Multiple unique values")
    print("assert max(value_counts.values()) / len(values) < 0.8  # No excessive dominance")
    print("```")
    
    print("\n3. PROBABILISTIC BOUNDS (Most robust):")
    print("```python")
    print("values = [get_random_function() for _ in range(30)]")
    print("if len(set(values)) == 1:")
    print("    prob = calculate_probability_all_same()")
    print("    if prob < 1e-10:  # Extremely improbable")
    print("        raise AssertionError('Non-random behavior detected')")
    print("```")


if __name__ == "__main__":
    # Show migration examples
    demonstrate_migration()
    
    print("\n" + "=" * 60)
    print("🧪 RUNNING MIGRATION TESTS")
    print("=" * 60)
    
    # Run the test examples
    unittest.main(verbosity=2)