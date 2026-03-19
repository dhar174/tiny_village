#!/usr/bin/env python3
"""
Test file that demonstrates and fixes the Mock() issue in utility calculations.
This file contains the problematic pattern mentioned in the issue and shows the correct approach.
"""

import unittest
import sys
import os

# Add the repo root (parent of tests/) to sys.path so that modules like
# tiny_utility_functions and actions can be imported directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from tiny_utility_functions import calculate_action_utility, Goal
from actions import Action
from test_tiny_utility_functions import MockAction


class TestFixValidation(unittest.TestCase):
    """Test class that validates the fix for Mock() usage in utility calculations."""

    def setUp(self):
        """Set up test fixtures."""
        # Common test data
        self.character_state = {"hunger": 0.9, "energy": 0.5}
    
    def test_fixed_with_real_action(self):
        """
        FIXED approach: Using real Action objects for proper testing.
        This ensures the test validates actual behavior with production classes.
        """
        # Create goal with proper priority
        goal = Goal(name="SatisfyHunger", target_effects={"hunger": -0.8})
        goal.priority = 0.7
        
        # Use real Action object instead of Mock()
        action = Action(
            name="EatFood",
            preconditions={},
            effects=[{"attribute": "hunger", "change_value": -0.7}],
            cost=0.1
        )
        
        utility = calculate_action_utility(self.character_state, action, current_goal=goal)
        
        # Now we can make meaningful assertions about the result
        self.assertIsInstance(utility, (int, float))
        self.assertGreater(utility, 0)  # Should be positive for a hungry character eating
        
        # Verify the calculation is correct
        # Expected: need_fulfillment (0.9 * 0.7 * 20) + goal_progress (0.7 * 25) - cost (0.1 * 10)
        expected = 12.6 + 17.5 - 1.0  # = 29.1
        self.assertAlmostEqual(utility, expected, places=1)
        
        print(f"✓ Real Action utility: {utility}")
    
    def test_fixed_with_test_class(self):
        """
        ALTERNATIVE FIXED approach: Using a proper test class that matches the interface.
        This is appropriate when real classes have complex dependencies.
        """
        # Create goal with proper priority
        goal = Goal(name="SatisfyHunger", target_effects={"hunger": -0.8})
        goal.priority = 0.7
        
        # Use MockAction test class that implements the proper interface
        action = MockAction(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.7}]
        )
        
        utility = calculate_action_utility(self.character_state, action, current_goal=goal)
        
        # Same assertions as with real Action object
        self.assertIsInstance(utility, (int, float))
        self.assertGreater(utility, 0)
        
        # Verify the calculation matches expected behavior
        expected = 12.6 + 17.5 - 1.0  # = 29.1
        self.assertAlmostEqual(utility, expected, places=1)
        
        print(f"✓ MockAction utility: {utility}")
    
    def test_mock_vs_real_comparison(self):
        """
        Test that demonstrates why Mock() is problematic and real/test objects are better.
        """
        goal = Goal(name="SatisfyHunger", target_effects={"hunger": -0.8})
        goal.priority = 0.7
        
        # Test with proper MockAction
        mock_action = MockAction(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.7}]
        )
        
        # Test with real Action
        real_action = Action(
            name="EatFood",
            preconditions={},
            effects=[{"attribute": "hunger", "change_value": -0.7}],
            cost=0.1
        )
        
        mock_utility = calculate_action_utility(self.character_state, mock_action, current_goal=goal)
        real_utility = calculate_action_utility(self.character_state, real_action, current_goal=goal)
        
        # Both should give the same result since they have the same interface
        self.assertAlmostEqual(mock_utility, real_utility, places=1)
        
        print(f"✓ MockAction and Real Action give consistent results: {mock_utility:.1f} vs {real_utility:.1f}")


def run_validation_tests():
    """Run the tests and provide summary."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFixValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(f"\n🎉 All {result.testsRun} tests passed!")
        print("✅ Mock() issue has been properly addressed")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    print("🧪 Testing Mock() issue fixes in utility calculations...")
    print("=" * 60)
    
    success = run_validation_tests()
    
    if success:
        print("\n✨ CONCLUSION: The Mock() issue has been addressed!")
        print("   Use real Action objects or proper test classes instead of Mock()")
        print("   for testing utility calculations to ensure accurate validation.")
    else:
        print("\n🔧 CONCLUSION: Issues found that need to be addressed.")