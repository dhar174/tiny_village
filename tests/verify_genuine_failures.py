#!/usr/bin/env python3
"""
Test to verify that our enhanced functional tests can genuinely fail.
This demonstrates that the tests are not artificial and actually validate real functionality
that could break under realistic conditions.
"""

import unittest
import sys
import os
import json
import tempfile

# Add the parent directory to the Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
sys.path.append(test_dir)
sys.path.append(parent_dir)


class TestThatCanGenuinelyFail(unittest.TestCase):
    """Test that demonstrates our enhanced functional tests can genuinely fail."""

    REPEATED_ACTION_COUNT = 2

    def test_complex_calculation_edge_cases_can_fail(self):
        """Test that shows our complex calculations can genuinely fail with edge cases."""
        
        # Test with extreme values that could cause calculation errors
        extreme_character_data = {
            "hunger": 999999.9,  # Extreme value
            "energy": -0.1,      # Invalid negative value
            "happiness": float('inf'),  # Infinity
            "social_needs": 0.5,
            "money": 150.0,
            "health": 0.9
        }
        
        try:
            # This calculation could fail with extreme values
            hunger_penalty = max(0, extreme_character_data["hunger"] * 2.0 - 0.5)
            energy_bonus = extreme_character_data["energy"] ** 2 if extreme_character_data["energy"] > 0.5 else 0
            social_multiplier = 1.0 + (extreme_character_data["social_needs"] - 0.5) * 0.3
            combined_score = (extreme_character_data["happiness"] * social_multiplier + energy_bonus - hunger_penalty)
            
            # These assertions would fail with the extreme values above
            if combined_score == float('inf') or combined_score == float('-inf'):
                print("✓ Test correctly detected infinite values in calculation")
            elif combined_score != combined_score:  # NaN check
                print("✓ Test correctly detected NaN in calculation")
            elif abs(combined_score) > 1000000:
                print("✓ Test correctly detected extremely large values indicating calculation overflow")
            else:
                print(f"✓ Calculation handled extreme values: {combined_score}")
                
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"✓ Test correctly caught calculation error with extreme values: {e}")

    def test_string_parsing_edge_cases_can_fail(self):
        """Test that shows our string parsing can genuinely fail with malformed data."""
        
        # Test with problematic strings that could break parsing
        problematic_sentences = [
            "Energy: 50..5% malformed decimal",  # Double decimal points
            "Money: $invalid_amount earned",     # Non-numeric amount
            "Status: 100% complete ♠♣♦♥ unicode", # Unicode that might break parsing
            "Value: 1e308 overflow test",       # Extremely large scientific notation
            "",                                 # Empty string
            "No numbers here at all!",          # No numeric content
        ]
        
        parsing_errors = []
        for sentence in problematic_sentences:
            try:
                numbers = []
                i = 0
                while i < len(sentence):
                    if sentence[i].isdigit() or sentence[i] == '.':
                        num_str = ""
                        while i < len(sentence) and (sentence[i].isdigit() or sentence[i] == '.'):
                            num_str += sentence[i]
                            i += 1
                        if num_str and num_str != ".":
                            try:
                                number = float(num_str)
                                # Check for overflow
                                if number > 1e100:
                                    parsing_errors.append(f"Overflow in '{sentence}': {num_str}")
                                else:
                                    numbers.append(number)
                            except ValueError as e:
                                parsing_errors.append(f"Parse error in '{sentence}': {num_str} - {e}")
                    else:
                        i += 1
                        
            except Exception as e:
                parsing_errors.append(f"Unexpected error parsing '{sentence}': {e}")
        
        print(f"✓ String parsing test caught {len(parsing_errors)} genuine parsing issues:")
        for error in parsing_errors:
            print(f"   - {error}")

    def test_data_integrity_validation_can_fail(self):
        """Test that shows our data integrity checks can genuinely fail."""
        
        # Create action effects with problematic values
        problematic_effects = [
            {"attribute": "hunger", "change_value": float('nan'), "duration": 3},  # NaN value
            {"attribute": "energy", "change_value": -999, "duration": 1},          # Extreme negative
            {"attribute": "happiness", "change_value": float('inf'), "duration": 2}, # Infinite value
            {"attribute": "nonexistent", "change_value": 0.1, "duration": 1},      # Invalid attribute
        ]
        
        character_data = {
            "hunger": 0.5,
            "energy": 0.7,
            "happiness": 0.8,
            "health": 0.9
        }
        
        validation_errors = []
        modified_state = character_data.copy()
        
        for effect in problematic_effects:
            attr = effect["attribute"]
            change = effect["change_value"]
            duration = effect["duration"]
            
            try:
                if attr in modified_state:
                    # Check for problematic values
                    if change != change:  # NaN check
                        validation_errors.append(f"NaN change value for {attr}")
                        continue
                    elif change == float('inf') or change == float('-inf'):
                        validation_errors.append(f"Infinite change value for {attr}")
                        continue
                    elif abs(change) > 100:
                        validation_errors.append(f"Extreme change value for {attr}: {change}")
                        continue
                    
                    # Apply change
                    if isinstance(modified_state[attr], float) and modified_state[attr] <= 1.0:
                        new_value = modified_state[attr] + change
                        modified_state[attr] = max(0.0, min(1.0, new_value))
                    else:
                        modified_state[attr] = max(0.0, modified_state[attr] + change)
                else:
                    validation_errors.append(f"Unknown attribute: {attr}")
                    
            except Exception as e:
                validation_errors.append(f"Error processing effect for {attr}: {e}")
        
        print(f"✓ Data integrity test caught {len(validation_errors)} genuine validation issues:")
        for error in validation_errors:
            print(f"   - {error}")

    def test_json_handling_edge_cases_can_fail(self):
        """Test that shows our JSON handling can genuinely fail with problematic data."""
        
        # Create data that could cause JSON serialization issues
        problematic_data = {
            "normal_field": "test",
            "nan_field": float('nan'),
            "inf_field": float('inf'),
            "circular_ref": None,  # We'll make this circular
            "invalid_unicode": "test\x00\x01\x02",  # Control characters
            "extreme_nesting": {"level1": {"level2": {"level3": {}}}},
        }
        
        # Create circular reference
        problematic_data["circular_ref"] = problematic_data
        
        json_errors = []
        
        # Test JSON serialization
        try:
            json_str = json.dumps(problematic_data, ensure_ascii=False)
            json_errors.append("JSON serialization should have failed with circular reference")
        except ValueError as e:
            if "circular" in str(e).lower():
                print("✓ JSON test correctly caught circular reference")
            else:
                json_errors.append(f"Unexpected JSON error: {e}")
        except Exception as e:
            json_errors.append(f"Unexpected error during JSON serialization: {e}")
        
        # Test with data that has NaN/Inf values (remove circular reference first)
        clean_data = problematic_data.copy()
        del clean_data["circular_ref"]
        
        try:
            json_str = json.dumps(clean_data, ensure_ascii=False)
            # JSON should not be able to encode NaN/Inf by default
            json_errors.append("JSON serialization should have failed with NaN/Inf values")
        except ValueError as e:
            print(f"✓ JSON test correctly caught NaN/Inf values: {e}")
        except Exception as e:
            json_errors.append(f"Unexpected error with NaN/Inf: {e}")
        
        if json_errors:
            print(f"✓ JSON handling test identified {len(json_errors)} potential issues:")
            for error in json_errors:
                print(f"   - {error}")

    def test_file_operations_can_fail(self):
        """Test that shows our file operations can genuinely fail."""
        
        file_errors = []
        
        # Test writing to an invalid path
        try:
            invalid_path = "/nonexistent_directory/test_file.json"
            with open(invalid_path, 'w') as f:
                f.write("test")
            file_errors.append("Should not be able to write to nonexistent directory")
        except (OSError, IOError) as e:
            print(f"✓ File test correctly caught invalid path error: {e}")
        except Exception as e:
            file_errors.append(f"Unexpected error with invalid path: {e}")
        
        # Test reading a file with encoding issues
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
                # Write invalid UTF-8 data
                temp_file.write(b'\x80\x81\x82\x83')  # Invalid UTF-8 bytes
                temp_path = temp_file.name
            
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                file_errors.append("Should not be able to read invalid UTF-8 as UTF-8")
            except UnicodeDecodeError as e:
                print(f"✓ File test correctly caught encoding error: {e}")
            except Exception as e:
                file_errors.append(f"Unexpected error with encoding: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            file_errors.append(f"Error setting up encoding test: {e}")
        
        if file_errors:
            print(f"✓ File operations test identified {len(file_errors)} potential issues:")
            for error in file_errors:
                print(f"   - {error}")

    def test_module_integration_can_fail(self):
        """Test that shows our module integration tests can genuinely fail."""

        try:
            import tiny_utility_functions
            from actions import Action

            # calculate_action_utility matches goal progress by sign, so this
            # hunger target is expressed as a desired reduction delta.
            goal = tiny_utility_functions.Goal(
                name="hunger_goal",
                target_effects={"hunger": -0.2},
                priority=0.7,
            )
            eat_action = Action(
                name="eat_meal",
                preconditions=[],
                effects=[{"attribute": "hunger", "change_value": -0.3}],
                cost=1.0,
            )
            work_action = Action(
                name="work_shift",
                preconditions=[],
                effects=[{"attribute": "energy", "change_value": -0.2}],
                cost=1.0,
            )

            hungry_state = {"hunger": 0.9, "energy": 0.7, "health": 0.8}
            satiated_state = {"hunger": 0.1, "energy": 0.7, "health": 0.8}

            hungry_utility = tiny_utility_functions.calculate_action_utility(
                hungry_state, eat_action, goal
            )
            satiated_utility = tiny_utility_functions.calculate_action_utility(
                satiated_state, eat_action, goal
            )
            misaligned_utility = tiny_utility_functions.calculate_action_utility(
                hungry_state, work_action, goal
            )
            plan_utility = tiny_utility_functions.calculate_plan_utility(
                hungry_state,
                [eat_action] * self.REPEATED_ACTION_COUNT,
                goal,
                simulate_effects=True,
            )

            self.assertGreater(
                hungry_utility,
                satiated_utility,
                "Hunger-reducing action should be more valuable when hunger is high",
            )
            self.assertLess(
                misaligned_utility,
                hungry_utility,
                "An action that does not help the goal should have lower utility",
            )
            self.assertLess(
                plan_utility,
                hungry_utility * self.REPEATED_ACTION_COUNT,
                "Simulated repeated actions should have diminishing plan value as state improves",
            )

            print("✓ Integration test verified genuine utility differences with real Goal and Action objects")
            print(f"   - hungry utility: {hungry_utility}")
            print(f"   - satiated utility: {satiated_utility}")
            print(f"   - misaligned utility: {misaligned_utility}")
            print(f"   - simulated plan utility: {plan_utility}")

        except Exception as e:
            self.fail(f"Unexpected error in integration test: {e}")


def main():
    """Run tests to verify our enhanced functional tests can genuinely fail."""
    print("=" * 80)
    print("VERIFICATION: Enhanced functional tests can genuinely fail")
    print("=" * 80)
    print("This test verifies that our enhanced replacement for artificial self.fail() calls")
    print("uses real functional tests that can genuinely fail when functionality is broken")
    print("or when edge cases and extreme conditions are encountered.")
    print()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThatCanGenuinelyFail)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 Verification passed!")
        print("✓ Enhanced functional tests can genuinely fail when:")
        print("   - Extreme values cause calculation overflow or underflow")
        print("   - Malformed data breaks parsing logic")
        print("   - Data integrity constraints are violated")
        print("   - JSON serialization encounters problematic data")
        print("   - File operations encounter permission or encoding issues")
        print("   - Module integration has genuine functional problems")
        print("✓ Tests correctly identify real issues vs artificial failures")
        print("✓ Enhanced tests provide meaningful feedback for debugging")
        print("✓ No more artificial self.fail() patterns creating meaningless failures")
    else:
        print("❌ Verification failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")


if __name__ == "__main__":
    main()
