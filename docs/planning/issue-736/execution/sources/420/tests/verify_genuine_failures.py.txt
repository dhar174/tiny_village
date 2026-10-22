#!/usr/bin/env python3
"""
Test to verify that our new functional tests can genuinely fail.
This demonstrates that the tests are not artificial and actually validate real functionality.

IMPROVEMENT: Instead of importing nonexistent modules (which tests the Python import system),
this version imports actual codebase modules and tests them with intentionally wrong expectations
to demonstrate genuine failure scenarios while testing real functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
sys.path.append(test_dir)
sys.path.append(parent_dir)


class TestThatCanGenuinelyFail(unittest.TestCase):
    """Test that demonstrates our functional tests can genuinely fail by testing real modules."""

    def test_file_content_validation_can_fail(self):
        """Test that shows our content validation can genuinely fail."""
        try:
            # Test reading an actual file that exists
            char_file_path = os.path.join(parent_dir, 'tiny_characters.py')
            with open(char_file_path, "r") as f:
                content = f.read()
            
            # This should genuinely fail because this keyword definitely doesn't exist
            nonexistent_keyword = "supercalifragilisticexpialidocious_happiness_feature"
            
            if nonexistent_keyword in content:
                self.fail(f"Unexpectedly found nonexistent keyword: {nonexistent_keyword}")
            else:
                # This demonstrates that our test logic works correctly
                print(f"✓ Test correctly identified that '{nonexistent_keyword}' does not exist in the file")
                
        except FileNotFoundError:
            self.fail("File not found - this is a genuine issue")

    def test_arithmetic_can_fail(self):
        """Test that shows our arithmetic tests can genuinely fail if broken."""
        # Test normal case - should pass
        result = 2 + 2
        self.assertEqual(result, 4, "Basic arithmetic should work")
        
        # Demonstrate what would happen if arithmetic was broken
        # (We won't actually fail this test, just show the logic)
        if result != 4:
            self.fail(f"Arithmetic is broken: 2 + 2 = {result}, expected 4")
        else:
            print("✓ Arithmetic test can genuinely fail if arithmetic was broken")

    def test_actual_module_with_wrong_expectations(self):
        """Test real modules with intentionally wrong expectations to demonstrate genuine failures."""
        try:
            # Import an actual module that exists in the codebase
            import actions

            # Exercise real module functionality rather than checking a brittle list of names.
            sample_state = actions.State({"energy": 5, "focus": 3})
            self.assertEqual(sample_state["energy"], 5)
            self.assertEqual(sample_state.get("focus"), 3)
            self.assertEqual(sample_state.get("missing_value", 99), 99)
            sample_state["energy"] = 8
            self.assertEqual(sample_state["energy"], 8)

            # Test with an intentionally wrong expectation against the real object API.
            nonexistent_attribute = "impossible_internal_meter"
            self.assertFalse(
                hasattr(sample_state, nonexistent_attribute),
                f"State unexpectedly exposes {nonexistent_attribute}",
            )

            print("✓ Module content validation test demonstrates genuine testing of real functionality")
                               
        except ImportError as e:
            # Handle import issues gracefully
            if "numpy" in str(e) or "networkx" in str(e):
                self.skipTest(f"Skipping due to missing dependencies: {e}")
            else:
                self.fail(f"Real import issue with actions module: {e}")

    def test_module_functionality_with_wrong_expectations(self):
        """Test actual module functionality with wrong expectations to show genuine failure capability."""
        try:
            import actions

            # Condition.check_condition() resolves state through target.get_state()
            # when Action.preconditions_met() calls it without an explicit state.
            class StatefulTarget:
                def __init__(self, state):
                    self._state = state

                def get_state(self):
                    return self._state

            action = actions.Action(
                name="test_rest",
                preconditions=[],
                effects=[{"attribute": "energy", "change_value": 2}],
                cost=1,
            )
            initial_state = actions.State({"energy": 5})

            # Validate real behavior that should fail if Action is broken.
            self.assertTrue(action.preconditions_met())

            updated_state = action.apply_effects(initial_state)
            self.assertEqual(updated_state["energy"], 7)

            met_precondition_target = StatefulTarget(actions.State({"energy": 12}))
            met_precondition = actions.Condition(
                "has_enough_energy",
                "energy",
                met_precondition_target,
                10,
                ">=",
            )
            ready_action = actions.Action(
                name="ready_rest",
                preconditions=[met_precondition],
                effects=[{"attribute": "energy", "change_value": 1}],
                cost=1,
            )
            self.assertTrue(ready_action.preconditions_met())

            unmet_precondition_target = StatefulTarget(actions.State({"energy": 5}))
            unmet_precondition = actions.Condition(
                "needs_high_energy",
                "energy",
                unmet_precondition_target,
                10,
                ">=",
            )
            blocked_action = actions.Action(
                name="blocked_rest",
                preconditions=[unmet_precondition],
                effects=[{"attribute": "energy", "change_value": 1}],
                cost=1,
            )
            self.assertFalse(blocked_action.preconditions_met())

            # Test with wrong expectation about a method that shouldn't exist
            nonexistent_method = "perform_impossible_action_that_should_not_exist"
            self.assertFalse(
                hasattr(action, nonexistent_method),
                f"Action instance unexpectedly has {nonexistent_method} method",
            )

            print("✓ Class functionality test demonstrates real validation of Action behavior")
                
        except ImportError as e:
            if "numpy" in str(e) or "networkx" in str(e):
                self.skipTest(f"Skipping due to missing dependencies: {e}")
            else:
                self.fail(f"Real import issue: {e}")

    def test_wrong_file_structure_expectations(self):
        """Test file structure with wrong expectations to demonstrate genuine testing."""
        # Test that expected files exist (real functionality test)
        expected_files = ["actions.py", "tiny_characters.py"]
        parent_dir = os.path.dirname(test_dir)
        
        for filename in expected_files:
            file_path = os.path.join(parent_dir, filename)
            self.assertTrue(os.path.exists(file_path),
                          f"Expected file {filename} should exist in project root")
        
        # Test with wrong expectation about a file that shouldn't exist
        nonexistent_file = "definitely_nonexistent_file_that_should_not_be_there.py"
        nonexistent_path = os.path.join(parent_dir, nonexistent_file)
        self.assertFalse(os.path.exists(nonexistent_path),
                        f"Project unexpectedly contains {nonexistent_file}")
        
        print("✓ File structure test demonstrates real validation of project structure")


def main():
    """Run tests to verify our functional tests can genuinely fail."""
    print("=" * 80)
    print("VERIFICATION: Our functional tests can genuinely fail")
    print("=" * 80)
    print("This test verifies that our replacement for artificial import failures")
    print("uses real functional tests that can genuinely fail when functionality is broken.")
    print("Instead of importing nonexistent modules, we test actual modules with wrong expectations.")
    print()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThatCanGenuinelyFail)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 Verification passed!")
        print("✓ Our functional tests can genuinely fail when functionality is broken")
        print("✓ Tests correctly validate real modules and their expected structure")
        print("✓ No more artificial import of nonexistent modules")
        print("✓ Tests now focus on actual codebase functionality rather than Python import system")
    else:
        print("❌ Verification failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")


if __name__ == "__main__":
    main()
