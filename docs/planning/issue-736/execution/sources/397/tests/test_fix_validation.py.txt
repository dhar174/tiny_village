#!/usr/bin/env python3
"""
Validation test for the test counting logic fix.

This test validates that the fix correctly addresses the issue where 
test counting logic artificially manipulates results by treating 
multiple unittest methods as a single test unit.
"""

import unittest
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed function
from test_completed_implementations import run_unittest_with_proper_counting


class TestCountingFix(unittest.TestCase):
    """Test suite to validate the test counting fix."""

    def test_proper_counting_utility_function(self):
        """Test that the run_unittest_with_proper_counting function works correctly."""
        
        # Create a test case class with multiple methods
        class SampleTestCase(unittest.TestCase):
            def test_one(self):
                self.assertTrue(True)
            
            def test_two(self):
                self.assertEqual(1 + 1, 2)
            
            def test_three(self):
                # This includes the pattern from the issue
                feature_results = {"social_happiness": True, "romantic_happiness": False}
                for feature in ["social_happiness", "romantic_happiness", "family_happiness"]:
                    if not feature_results.get(feature, False):
                        print(f"⚠ Warning: {feature} feature not found - this may be acceptable if other features compensate")
                self.assertTrue(True)
        
        # Run the test using our fixed counting logic
        stats = run_unittest_with_proper_counting(SampleTestCase, "Sample Test")
        
        # Validate that individual methods are counted correctly
        self.assertEqual(stats['total'], 3, "Should count 3 individual test methods")
        self.assertEqual(stats['passed'], 3, "All 3 test methods should pass")
        self.assertEqual(stats['failures'], 0, "Should have no failures")
        self.assertEqual(stats['errors'], 0, "Should have no errors")

    def test_counting_with_failures(self):
        """Test counting logic when some test methods fail."""
        
        class SampleTestCaseWithFailures(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)
            
            def test_fail(self):
                self.fail("This test is designed to fail")
            
            def test_error(self):
                raise ValueError("This test raises an error")
        
        # Run the test using our fixed counting logic
        stats = run_unittest_with_proper_counting(SampleTestCaseWithFailures, "Failure Test")
        
        # Validate counting with failures and errors
        self.assertEqual(stats['total'], 3, "Should count 3 individual test methods")
        self.assertEqual(stats['passed'], 1, "Only 1 test method should pass")
        self.assertEqual(stats['failures'], 1, "Should have 1 failure")
        self.assertEqual(stats['errors'], 1, "Should have 1 error")

    def test_mixed_test_scenario(self):
        """Test a mixed scenario similar to the actual implementation."""
        
        class MixedTestCase(unittest.TestCase):
            def test_happiness_features_implementation(self):
                """Similar to the actual happiness test."""
                # Simulate checking features
                features = ["motive_satisfaction", "social_happiness", "romantic_happiness", "family_happiness"]
                implemented = ["motive_satisfaction", "social_happiness", "romantic_happiness", "family_happiness"]
                self.assertGreaterEqual(len(implemented), 3, "Should have at least 3 features")
            
            def test_individual_happiness_features(self):
                """Test with the specific pattern from the issue."""
                feature_results = {
                    "social_happiness": True,
                    "romantic_happiness": False,
                    "family_happiness": True
                }
                
                # This is the exact code pattern mentioned in the issue
                # For the other features, we'll be more lenient and just warn if missing
                for feature in ["social_happiness", "romantic_happiness", "family_happiness"]:
                    if not feature_results.get(feature, False):
                        print(f"⚠ Warning: {feature} feature not found - this may be acceptable if other features compensate")
                
                # Ensure at least core features work
                self.assertTrue(feature_results.get("social_happiness", False))
        
        # Run the mixed test case
        stats = run_unittest_with_proper_counting(MixedTestCase, "Mixed Test Scenario")
        
        # Validate proper counting of the mixed scenario
        self.assertEqual(stats['total'], 2, "Should count 2 individual test methods")
        self.assertEqual(stats['passed'], 2, "Both test methods should pass")


def main():
    """Run validation tests for the test counting fix."""
    print("Running validation tests for test counting logic fix...")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCountingFix)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("🎉 All validation tests passed!")
        print("✓ Test counting logic fix is working correctly")
        print("✓ Individual unittest methods are counted separately")
        print("✓ No artificial manipulation of test results")
        print("✓ Accurate feedback is provided")
    else:
        print("❌ Some validation tests failed")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)