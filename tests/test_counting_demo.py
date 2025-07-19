#!/usr/bin/env python3
"""
Demonstration of the test counting fix.

This file shows the difference between the old counting logic that treats 
unittest.TestCase classes as single units vs the new logic that properly 
counts individual test methods.
"""

import unittest
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DemoTestCase(unittest.TestCase):
    """Demo test case with multiple test methods to demonstrate counting issue."""

    def test_method_one(self):
        """First test method."""
        self.assertTrue(True, "This test should pass")

    def test_method_two(self):
        """Second test method."""
        self.assertEqual(2 + 2, 4, "Basic math should work")

    def test_method_three(self):
        """Third test method."""
        feature_results = {
            "social_happiness": True,
            "romantic_happiness": False,
            "family_happiness": True
        }
        
        # For the other features, we'll be more lenient and just warn if missing
        for feature in ["social_happiness", "romantic_happiness", "family_happiness"]:
            if not feature_results.get(feature, False):
                print(f"⚠ Warning: {feature} feature not found - this may be acceptable if other features compensate")
                
        # This demonstrates the code pattern mentioned in the issue
        self.assertTrue(feature_results.get("social_happiness", False))


def old_counting_logic():
    """
    Demonstrates the OLD counting logic that treats TestCase classes as single units.
    This artificially manipulates results and violates test accuracy principles.
    """
    print("=" * 60)
    print("OLD COUNTING LOGIC (PROBLEMATIC)")
    print("=" * 60)
    
    # Old logic would count the entire TestCase as 1 test
    test_classes = [DemoTestCase]
    
    passed = 0
    total = len(test_classes)  # This is the problem - only counting classes, not methods
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        # Old logic treats entire class as pass/fail regardless of individual methods
        if result.wasSuccessful():
            passed += 1
    
    print(f"Old logic result: {passed}/{total} tests passed")
    print("Problem: This counts the TestCase class as 1 test, ignoring individual methods!")
    return passed, total


def new_counting_logic():
    """
    Demonstrates the NEW counting logic that properly counts individual test methods.
    This provides accurate feedback about function behavior.
    """
    print("\n" + "=" * 60)
    print("NEW COUNTING LOGIC (FIXED)")
    print("=" * 60)
    
    # New logic counts individual test methods
    suite = unittest.TestLoader().loadTestsFromTestCase(DemoTestCase)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = tests_run - failures - errors
    
    print(f"New logic result: {passed}/{tests_run} individual test methods passed")
    print("Solution: This correctly counts each test method separately!")
    return passed, tests_run


def main():
    """Demonstrate the difference between old and new counting logic."""
    print("DEMONSTRATION: Test Counting Logic Fix")
    print("Issue: Multiple unittest methods treated as single test unit")
    print()
    
    old_passed, old_total = old_counting_logic()
    new_passed, new_total = new_counting_logic()
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Old logic: {old_passed}/{old_total} (inaccurate - treats class as 1 unit)")
    print(f"New logic: {new_passed}/{new_total} (accurate - counts individual methods)")
    print()
    print("The fix ensures that:")
    print("1. Each test method is counted separately")
    print("2. Feedback is granular and accurate")
    print("3. Test results reflect actual function behavior")
    print("4. No artificial manipulation of test counts")


if __name__ == "__main__":
    main()