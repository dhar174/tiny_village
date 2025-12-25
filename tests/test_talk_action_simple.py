#!/usr/bin/env python3

"""
Simple test for TalkAction with mocked GraphManager
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestTalkActionSimple(unittest.TestCase):

    @patch("actions.importlib.import_module")
    def test_talk_action_creation(self, mock_import):
        """Test TalkAction creation with mocked GraphManager"""

        # Mock the GraphManager
        mock_graph_manager_class = MagicMock()
        mock_graph_manager_instance = MagicMock()
        mock_graph_manager_class.return_value = mock_graph_manager_instance
        mock_import.return_value.GraphManager = mock_graph_manager_class

        # Now import TalkAction after setting up the mock
        from actions import TalkAction

        # Create TalkAction with extra parameters that would normally cause issues
        talk_action = TalkAction(
            initiator="char1", target="char2", topic="weather", target_name="char2"
        )

        # Verify the action was created successfully
        self.assertEqual(talk_action.name, "Talk")
        self.assertEqual(talk_action.initiator, "char1")
        self.assertEqual(talk_action.target, "char2")
        self.assertEqual(talk_action.cost, 0.1)

        print("SUCCESS: TalkAction created successfully with mocked GraphManager!")
        print(
            f"Action: {talk_action.name}, Initiator: {talk_action.initiator}, Target: {talk_action.target}"
        )

    @patch("actions.importlib.import_module")
    def test_output_interpreter_with_talk_action(self, mock_import):
        """Test OutputInterpreter with TalkAction using mocked GraphManager"""

        # Mock the GraphManager
        mock_graph_manager_class = MagicMock()
        mock_graph_manager_instance = MagicMock()
        mock_graph_manager_class.return_value = mock_graph_manager_instance
        mock_import.return_value.GraphManager = mock_graph_manager_class

        # Import after mocking
        from tiny_output_interpreter import OutputInterpreter

        # Create interpreter
        interpreter = OutputInterpreter()

        # Test TalkAction interpretation
        parsed_response = {
            "action": "Talk",
            "parameters": {
                "target_name": "John",
                "topic": "weather",
                "initiator_id": "char_speaker",
            },
        }

        # This should work now without GraphManager issues
        action_instance = interpreter.interpret(parsed_response)

        # Verify the action was created
        self.assertIsNotNone(action_instance)
        self.assertEqual(action_instance.name, "Talk")
        self.assertEqual(action_instance.initiator, "char_speaker")
        self.assertEqual(action_instance.target, "John")

        print("SUCCESS: OutputInterpreter.interpret() worked with TalkAction!")
        print(f"Created action: {action_instance.name}")


if __name__ == "__main__":
    unittest.main()
