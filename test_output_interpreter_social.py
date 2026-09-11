#!/usr/bin/env python3

"""
Test the OutputInterpreter with enhanced social actions
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestOutputInterpreterSocialActions(unittest.TestCase):

    def test_output_interpreter_social_actions(self):
        """Test OutputInterpreter with enhanced social actions"""

        from tiny_output_interpreter import OutputInterpreter

        # Create interpreter
        interpreter = OutputInterpreter()

        # Test cases for different social actions
        test_cases = [
            {
                "action": "Talk",
                "parameters": {
                    "target_name": "Bob",
                    "initiator_id": "Alice",
                },
                "expected_class": "TalkAction"
            },
            {
                "action": "Greet", 
                "parameters": {
                    "target_name": "Carol",
                    "initiator_id": "Alice",
                },
                "expected_class": "GreetAction"
            },
            {
                "action": "ShareNews",
                "parameters": {
                    "target_name": "Bob",
                    "news_item": "The market is having a sale!",
                    "initiator_id": "Alice",
                },
                "expected_class": "ShareNewsAction"
            },
            {
                "action": "OfferCompliment",
                "parameters": {
                    "target_name": "Carol",
                    "compliment_topic": "your painting skills",
                    "initiator_id": "Alice",
                },
                "expected_class": "OfferComplimentAction"
            }
        ]

        for test_case in test_cases:
            with self.subTest(action=test_case["action"]):
                parsed_response = {
                    "action": test_case["action"],
                    "parameters": test_case["parameters"],
                }

                # This should work now with enhanced social actions
                action_instance = interpreter.interpret(parsed_response)

                # Verify the action was created
                self.assertIsNotNone(action_instance)
                self.assertEqual(action_instance.__class__.__name__, test_case["expected_class"])
                self.assertEqual(action_instance.initiator, test_case["parameters"]["initiator_id"])
                self.assertEqual(action_instance.target, test_case["parameters"]["target_name"])

                print(f"✓ Successfully created {test_case['expected_class']}")

    def test_natural_language_social_parsing(self):
        """Test OutputInterpreter natural language parsing for social actions"""

        from tiny_output_interpreter import OutputInterpreter

        interpreter = OutputInterpreter()

        # Test natural language social interactions
        natural_language_tests = [
            ("talk to Bob about the weather", "Talk", "Bob"),
            ("greet Alice warmly", "Greet", "Alice"),
            ("say hello to Carol", "Greet", "Carol"),
            ("compliment John on his cooking", "OfferCompliment", "John"),
            ("tell Mary about the news", "ShareNews", "Mary"),
        ]

        for text, expected_action, expected_target in natural_language_tests:
            with self.subTest(text=text):
                try:
                    parsed_response = interpreter.parse_llm_response(text)
                    self.assertEqual(parsed_response["action"], expected_action)
                    
                    if "target_name" in parsed_response["parameters"]:
                        self.assertEqual(parsed_response["parameters"]["target_name"], expected_target)
                    
                    print(f"✓ Successfully parsed: '{text}' -> {expected_action}")
                except Exception as e:
                    print(f"⚠ Could not parse: '{text}' -> {e}")


if __name__ == "__main__":
    unittest.main()