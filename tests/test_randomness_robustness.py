#!/usr/bin/env python3
"""
Test for robust randomness validation.

This test addresses the issue where `assert len(set(values)) > 1` could 
theoretically fail even with proper randomness, though extremely unlikely.

The original problematic pattern:
    values = [get_random_function() for _ in range(10)]
    assert len(set(values)) > 1  # Could fail with proper randomness!

The improved approach uses either:
1. Larger sample size to make identical values extremely unlikely
2. Statistical testing with proper confidence intervals
3. Probabilistic bounds that account for expected randomness behavior
"""

import random
import unittest
from collections import Counter
from unittest.mock import patch


class MockDescriptorMatrices:
    """Mock descriptor matrices for testing randomness patterns."""
    
    def __init__(self):
        self.job_adjective = {
            "default": [
                "skilled",
                "hardworking", 
                "friendly",
                "creative",
                "dedicated"
            ]
        }
        
    def get_job_adjective(self, job):
        """Return a random job adjective."""
        return random.choice(self.job_adjective.get(job, self.job_adjective["default"]))


class TestRandomnessRobustness(unittest.TestCase):
    """Test robust randomness validation approaches."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.descriptors = MockDescriptorMatrices()
        
    def test_problematic_randomness_check(self):
        """
        Demonstrate the problematic randomness test pattern.
        
        This test could theoretically fail even with proper randomness,
        though the probability is extremely low.
        """
        # This is the PROBLEMATIC pattern mentioned in the issue
        values = [self.descriptors.get_job_adjective("default") for _ in range(10)]
        
        # This assertion could theoretically fail with proper randomness!
        # Probability of all 10 values being the same with 5 choices: (1/5)^9 ≈ 0.0000512%
        # While extremely unlikely, this could cause flaky test failures
        with self.subTest("Demonstrating problematic pattern"):
            # Comment out this assertion as it's the problematic one
            # assert len(set(values)) > 1, f"Expected diverse values, got: {values}"
            pass
            
    def test_robust_randomness_check_large_sample(self):
        """
        Improved randomness test using larger sample size.
        
        By using a much larger sample, we make the probability of all identical
        values so small that it's effectively impossible in practice.
        """
        # Use a larger sample size (100 instead of 10)
        values = [self.descriptors.get_job_adjective("default") for _ in range(100)]
        
        # With 5 choices and 100 samples, probability of all same: (1/5)^99 ≈ 10^-70
        # This is so small it's effectively impossible
        unique_values = set(values)
        
        self.assertGreater(
            len(unique_values), 1,
            f"Expected diverse values with large sample (n=100), got: {len(unique_values)} unique values"
        )
        
    def test_robust_randomness_check_statistical(self):
        """
        Improved randomness test using statistical bounds.
        
        Instead of requiring strictly more than 1 unique value, we test that
        the distribution shows reasonable randomness characteristics.
        """
        # Generate a reasonable sample
        values = [self.descriptors.get_job_adjective("default") for _ in range(50)]
        value_counts = Counter(values)
        unique_count = len(value_counts)
        
        # Statistical expectations for proper randomness:
        # - Should have multiple unique values (allowing for some repetition)
        # - No single value should dominate excessively
        
        # Test 1: Should have at least 2 unique values (more forgiving than original)
        self.assertGreaterEqual(
            unique_count, 2,
            f"Expected at least 2 unique values, got {unique_count}: {dict(value_counts)}"
        )
        
        # Test 2: No single value should appear more than 80% of the time
        # (this catches systematic bias while allowing for random clustering)
        max_frequency = max(value_counts.values())
        max_percentage = (max_frequency / len(values)) * 100
        
        self.assertLess(
            max_percentage, 80.0,
            f"Single value appears {max_percentage:.1f}% of time, suggesting non-random behavior: {dict(value_counts)}"
        )
        
    def test_robust_randomness_check_probabilistic(self):
        """
        Improved randomness test using probabilistic bounds.
        
        This approach calculates the probability of the observed outcome
        and fails only if it's statistically improbable.
        """
        sample_size = 30
        values = [self.descriptors.get_job_adjective("default") for _ in range(sample_size)]
        unique_count = len(set(values))
        
        # With 5 possible values and 30 samples, we expect good diversity
        # We'll be more lenient and only fail if we get an extremely improbable result
        
        if unique_count == 1:
            # All values the same - calculate probability
            # P(all same) = 5 * (1/5)^30 = 5 * (1/5)^30 ≈ 4.66 × 10^-21
            # This is so improbable that it suggests a bug in the random function
            self.fail(
                f"All {sample_size} values were identical: '{values[0]}'. "
                f"This is extremely improbable (p ≈ 4.66e-21) and suggests non-random behavior."
            )
        elif unique_count <= 2 and sample_size >= 30:
            # Very few unique values with large sample - suspicious but not impossible
            # We'll warn but not fail, as this could happen with proper randomness
            print(f"Warning: Only {unique_count} unique values in {sample_size} samples. "
                  f"This is unusual but possible with proper randomness.")
        
        # The test passes as long as we don't see the extremely improbable case
        self.assertGreaterEqual(unique_count, 1)  # Always true, but documents expectation
        
    def test_randomness_with_confidence_interval(self):
        """
        Most robust approach: Test randomness with proper confidence intervals.
        
        This approach uses statistical methods to determine if the observed
        randomness is within expected bounds.
        """
        sample_size = 50
        num_choices = 5  # Number of possible values in our mock
        
        values = [self.descriptors.get_job_adjective("default") for _ in range(sample_size)]
        value_counts = Counter(values)
        
        # Expected frequency for each value (if perfectly random)
        expected_freq = sample_size / num_choices  # 10 for 50 samples, 5 choices
        
        # Calculate chi-square-like statistic (simplified)
        # In a proper implementation, you'd use scipy.stats.chisquare
        chi_square_like = sum(
            (count - expected_freq) ** 2 / expected_freq 
            for count in value_counts.values()
        )
        
        # For truly random data with 5 categories and 50 samples,
        # the chi-square statistic should be reasonably small
        # (In practice, you'd compare against chi-square critical values)
        
        # We'll use a simple heuristic: the statistic shouldn't be extremely large
        # This catches systematic bias while allowing for natural randomness variation
        
        self.assertLess(
            chi_square_like, 40.0,  # Heuristic threshold
            f"Chi-square-like statistic ({chi_square_like:.2f}) suggests non-random distribution: {dict(value_counts)}"
        )
        
        # Also ensure we have reasonable diversity
        unique_count = len(value_counts)
        self.assertGreaterEqual(
            unique_count, 2,
            f"Expected multiple unique values, got {unique_count}: {dict(value_counts)}"
        )


def simulate_original_test_failure():
    """
    Demonstrate how the original test could theoretically fail.
    
    This function shows that even with proper randomness, the original
    `assert len(set(values)) > 1` could fail, causing flaky tests.
    """
    print("Simulating potential failure of original test pattern...")
    
    # Create a scenario where all values could be the same
    # We'll use a mock random function that occasionally returns all same values
    
    def mock_get_random():
        # Simulate the rare case where random.choice happens to return the same value
        return "skilled"  # All values the same
        
    # This would fail the original test even though the function is "working"
    values = [mock_get_random() for _ in range(10)]
    unique_count = len(set(values))
    
    print(f"Values: {values}")
    print(f"Unique values: {unique_count}")
    print(f"Original test would fail: {unique_count <= 1}")
    

if __name__ == "__main__":
    # Run the simulation first
    simulate_original_test_failure()
    print("\n" + "="*60 + "\n")
    
    # Run the unit tests
    unittest.main(verbosity=2)