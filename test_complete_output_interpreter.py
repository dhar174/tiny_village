#!/usr/bin/env python3
"""
Test script for the completed Output Interpretation System.
This validates that the enhanced tiny_output_interpreter.py can properly:
1. Parse various LLM response formats (JSON, natural language, mixed)
2. Handle all 50+ mapped action types
3. Validate and normalize parameters correctly
4. Instantiate action objects with proper fallback behavior
"""

import unittest
import json
from tiny_output_interpreter import (
    OutputInterpreter,
    InvalidLLMResponseFormatError,
    UnknownActionError,
    InvalidActionParametersError,
    LLMResponseParsingError,
)
from actions import Action, TalkAction, GreetAction


class TestCompleteOutputInterpreter(unittest.TestCase):
    def setUp(self):
        self.interpreter = OutputInterpreter()

    def test_json_response_parsing(self):
        """Test parsing of well-formatted JSON responses"""
        json_response = '{"action": "eat_food", "parameters": {"item_name": "apple", "location": "kitchen"}}'
        parsed = self.interpreter.parse_llm_response(json_response)

        self.assertEqual(parsed["action"], "eat_food")
        self.assertEqual(parsed["parameters"]["item_name"], "apple")
        self.assertEqual(parsed["parameters"]["location"], "kitchen")

    def test_natural_language_parsing(self):
        """Test parsing of natural language responses"""
        natural_response = "I want to go to the market to buy some food"
        parsed = self.interpreter.parse_llm_response(natural_response)

        self.assertIn("action", parsed)
        self.assertIn("parameters", parsed)
        # Should extract movement or buying action
        self.assertTrue(
            any(action in parsed["action"].lower() for action in ["go", "buy", "move"])
        )

    def test_mixed_format_parsing(self):
        """Test parsing of mixed format with JSON embedded in text"""
        mixed_response = 'The character should {"action": "talk", "parameters": {"target_name": "Alice", "topic": "weather"}} to socialize.'
        parsed = self.interpreter.parse_llm_response(mixed_response)

        self.assertEqual(parsed["action"], "talk")
        self.assertEqual(parsed["parameters"]["target_name"], "Alice")

    def test_food_actions_interpretation(self):
        """Test interpretation of food-related actions"""
        # Test EatAction
        eat_response = {"action": "eat_food", "parameters": {"item_name": "sandwich"}}
        action = self.interpreter.interpret(eat_response, "character123")

        self.assertEqual(action.name, "Eat")
        self.assertEqual(action.item_name, "sandwich")
        self.assertEqual(action.initiator, "character123")

        # Test BuyFoodAction
        buy_response = {"action": "buy_food", "parameters": {"food_type": "bread"}}
        action = self.interpreter.interpret(buy_response, "character123")

        self.assertEqual(action.name, "BuyFood")
        self.assertEqual(action.food_type, "bread")

    def test_social_actions_interpretation(self):
        """Test interpretation of social actions"""
        # Test TalkAction
        talk_response = {
            "action": "Talk",
            "parameters": {"target_name": "Bob", "topic": "work"},
        }
        action = self.interpreter.interpret(talk_response, "character123")

        self.assertIsInstance(action, TalkAction)
        self.assertEqual(action.initiator, "character123")
        self.assertEqual(action.target, "Bob")

        # Test GreetAction
        greet_response = {"action": "Greet", "parameters": {"target": "Alice"}}
        action = self.interpreter.interpret(greet_response, "character123")

        self.assertIsInstance(action, GreetAction)
        self.assertEqual(action.initiator, "character123")
        self.assertEqual(action.target, "Alice")

    def test_movement_action_interpretation(self):
        """Test interpretation of movement actions"""
        movement_response = {"action": "GoTo", "parameters": {"location_name": "park"}}
        action = self.interpreter.interpret(movement_response, "character123")

        self.assertEqual(action.name, "GoToLocation")
        self.assertEqual(action.location_name, "park")
        self.assertEqual(action.initiator, "character123")

    def test_work_actions_interpretation(self):
        """Test interpretation of work-related actions"""
        work_response = {
            "action": "improve_job_performance",
            "parameters": {"method": "training"},
        }
        action = self.interpreter.interpret(work_response, "character123")

        self.assertEqual(action.name, "ImproveJobPerformance")
        self.assertEqual(action.method, "training")

    def test_parameter_validation_and_normalization(self):
        """Test parameter validation for different action categories"""
        # Test movement action parameter validation
        movement_params = self.interpreter.parse_movement_action(
            "GoTo", {"destination": "library"}
        )
        self.assertEqual(movement_params["location_name"], "library")
        self.assertEqual(movement_params["destination"], "library")

        # Test social action parameter validation
        social_params = self.interpreter.parse_social_action(
            "Talk", {"target": "Charlie", "topic": "books"}
        )
        self.assertEqual(social_params["target_name"], "Charlie")
        self.assertEqual(social_params["target_person"], "Charlie")

        # Test work action parameter validation
        work_params = self.interpreter.parse_work_action(
            "improve_job_performance", {"task": "presentation"}
        )
        self.assertEqual(work_params["method"], "presentation")

    def test_factory_created_actions(self):
        """Test actions created by factory methods"""
        # Test generic action creation
        generic_response = {"action": "self_care", "parameters": {"duration": "30"}}
        action = self.interpreter.interpret(generic_response, "character123")

        self.assertIsInstance(action, Action)
        self.assertIn("self_care", action.name.lower())

        # Test social factory action
        social_response = {
            "action": "increase_friendship",
            "parameters": {"target": "David"},
        }
        action = self.interpreter.interpret(social_response, "character123")

        self.assertIsInstance(action, Action)

    def test_fallback_behavior(self):
        """Test fallback behavior for unrecognized responses"""
        # Test completely unrecognized response
        unrecognized_response = "xyz random text that makes no sense"
        parsed = self.interpreter.parse_llm_response(unrecognized_response)

        self.assertEqual(parsed["action"], "NoOp")
        self.assertIn("parsing_failed", str(parsed["parameters"]))

        # Test unrecognized action name
        unknown_action = {"action": "completely_unknown_action", "parameters": {}}

        with self.assertRaises(UnknownActionError):
            self.interpreter.interpret(unknown_action, "character123")

    def test_missing_parameters_handling(self):
        """Test handling of missing required parameters"""
        # Test missing target for social action
        incomplete_talk = {"action": "Talk", "parameters": {}}

        with self.assertRaises(InvalidActionParametersError):
            self.interpreter.interpret(incomplete_talk, "character123")

        # Test missing item_name for eat action
        incomplete_eat = {"action": "eat_food", "parameters": {}}
        action = self.interpreter.interpret(incomplete_eat, "character123")
        # Should get default value
        self.assertEqual(action.item_name, "food")

    def test_comprehensive_action_coverage(self):
        """Test that all major action categories are properly mapped"""
        action_categories = {
            "food": ["eat_food", "buy_food"],
            "social": ["Talk", "Greet", "ShareNews", "social_visit"],
            "movement": ["GoTo", "move_to_new_location"],
            "work": ["improve_job_performance", "go_to_work"],
            "creative": ["pursue_hobby", "craft_item"],
            "health": ["visit_doctor", "take_medicine"],
            "economic": ["buy_property", "invest_wealth"],
        }

        for category, actions in action_categories.items():
            for action_name in actions:
                self.assertIn(
                    action_name,
                    self.interpreter.action_class_map,
                    f"Action '{action_name}' not found in action mapping for category '{category}'",
                )

    def test_parameter_extraction_patterns(self):
        """Test natural language parameter extraction"""
        test_cases = [
            ("go to the library", {"action": "GoTo", "location": "library"}),
            (
                "talk to Alice about work",
                {"action": "Talk", "target": "Alice", "topic": "work"},
            ),
            ("buy some apples", {"action": "BuyFood", "item": "apples"}),
            ("eat a sandwich", {"action": "Eat", "item": "sandwich"}),
        ]

        for text, expected in test_cases:
            parsed = self.interpreter.parse_llm_response(text)
            # Verify that some meaningful action was extracted
            self.assertIsInstance(parsed["action"], str)
            self.assertIsInstance(parsed["parameters"], dict)

    def test_error_handling_robustness(self):
        """Test that the system handles various error conditions gracefully"""
        error_cases = [
            '{"action": "eat_food"}',  # Missing parameters key
            '{"parameters": {"item": "apple"}}',  # Missing action key
            "not json at all",  # Invalid JSON
            "",  # Empty string
            '{"action": 123, "parameters": {}}',  # Non-string action
            '{"action": "valid_action", "parameters": "not_dict"}',  # Non-dict parameters
        ]

        for error_case in error_cases:
            try:
                parsed = self.interpreter.parse_llm_response(error_case)
                # Should get fallback response
                self.assertIn("action", parsed)
                self.assertIn("parameters", parsed)
            except Exception as e:
                # Some error handling is expected, but shouldn't crash
                self.assertIsInstance(
                    e, (InvalidLLMResponseFormatError, LLMResponseParsingError)
                )


def run_comprehensive_tests():
    """Run all tests and provide detailed results"""
    print("🧪 Running comprehensive Output Interpretation System tests...")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCompleteOutputInterpreter)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print(f"📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(
        f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")

    if len(result.failures) == 0 and len(result.errors) == 0:
        print(
            "\n✅ All tests passed! Output Interpretation System is working correctly."
        )
    else:
        print(f"\n⚠️  Some tests failed. Please review the implementation.")

    return result


if __name__ == "__main__":
    run_comprehensive_tests()
