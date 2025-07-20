import unittest
from unittest.mock import patch, MagicMock

# Assuming tiny_output_interpreter.py and actions.py are in the python path
from tiny_output_interpreter import (
    OutputInterpreter,
    InvalidLLMResponseFormatError,
    UnknownActionError,
    InvalidActionParametersError,
    EatAction as PlaceholderEatAction,  # Placeholder defined in tiny_output_interpreter
    GoToLocationAction as PlaceholderGoToAction,  # Placeholder defined in tiny_output_interpreter
    NoOpAction as PlaceholderNoOpAction,  # Placeholder defined in tiny_output_interpreter
)

# We also need to test against the actual TalkAction from actions.py
from actions import Action, TalkAction

# Since the placeholder actions in tiny_output_interpreter are also named EatAction, GoToLocationAction, etc.
# We use the 'as Placeholder...' for clarity when we intend to mock these specific placeholders.
# The actual TalkAction is imported directly.


class TestParseLLMResponse(unittest.TestCase):
    def setUp(self):
        self.interpreter = OutputInterpreter()

    def test_parse_valid_json(self):
        response_str = '{"action": "Eat", "parameters": {"item_name": "Apple"}}'
        expected = {"action": "Eat", "parameters": {"item_name": "Apple"}}
        self.assertEqual(self.interpreter.parse_llm_response(response_str), expected)

    def test_parse_invalid_json(self):
        response_str = '{"action": "Eat", "parameters": {"item_name": "Apple"'  # Missing closing brace
        # With our enhanced parser, this should fallback to natural language parsing and extract "Eat"
        result = self.interpreter.parse_llm_response(response_str)
        self.assertEqual(result["action"], "Eat")
        self.assertEqual(result["parameters"], {})

    def test_parse_empty_string(self):
        # Empty string should fallback to NoOp action in our enhanced parser
        result = self.interpreter.parse_llm_response("")
        self.assertEqual(result["action"], "NoOp")
        self.assertIn("reason", result["parameters"])

    def test_parse_missing_action_key(self):
        response_str = '{"parameters": {"item_name": "Apple"}}'
        # With our enhanced parser, this should fallback to NoOp action
        result = self.interpreter.parse_llm_response(response_str)
        self.assertEqual(result["action"], "NoOp")

    def test_parse_missing_parameters_key(self):
        response_str = '{"action": "Eat"}'
        # With our enhanced parser, this extracts "Eat" via natural language parsing
        result = self.interpreter.parse_llm_response(response_str)
        self.assertEqual(result["action"], "Eat")
        self.assertEqual(result["parameters"], {})

    def test_parse_action_not_string(self):
        response_str = '{"action": 123, "parameters": {}}'
        # With our enhanced parser, this should fallback to NoOp action since the action isn't a string
        result = self.interpreter.parse_llm_response(response_str)
        self.assertEqual(result["action"], "NoOp")

    def test_parse_parameters_not_dict(self):
        response_str = '{"action": "Eat", "parameters": "item_name"}'
        # With our enhanced parser, this extracts "Eat" via natural language parsing
        result = self.interpreter.parse_llm_response(response_str)
        self.assertEqual(result["action"], "Eat")
        self.assertEqual(result["parameters"], {})


class TestInterpret(unittest.TestCase):
    def setUp(self):
        self.interpreter = OutputInterpreter()
        # It's important that the paths to the classes are correct for patching.
        # These are the classes defined *within* tiny_output_interpreter.py for placeholders,
        # or imported into it (like TalkAction).
        self.patcher_eat = patch(
            "tiny_output_interpreter.EatAction", spec=PlaceholderEatAction
        )
        self.patcher_goto = patch(
            "tiny_output_interpreter.GoToLocationAction", spec=PlaceholderGoToAction
        )
        self.patcher_noop = patch(
            "tiny_output_interpreter.NoOpAction", spec=PlaceholderNoOpAction
        )
        # This is the TalkAction imported from actions.py and used in the map
        self.patcher_talk = patch("tiny_output_interpreter.TalkAction", spec=TalkAction)

        self.MockEatAction = self.patcher_eat.start()
        self.MockGoToAction = self.patcher_goto.start()
        self.MockNoOpAction = self.patcher_noop.start()
        self.MockTalkAction = self.patcher_talk.start()

        # Ensure mocks return a MagicMock instance when called, which itself can be asserted upon
        self.MockEatAction.return_value = MagicMock(spec=PlaceholderEatAction)
        self.MockGoToAction.return_value = MagicMock(spec=PlaceholderGoToAction)
        self.MockNoOpAction.return_value = MagicMock(spec=PlaceholderNoOpAction)
        self.MockTalkAction.return_value = MagicMock(spec=TalkAction)

    def tearDown(self):
        self.patcher_eat.stop()
        self.patcher_goto.stop()
        self.patcher_noop.stop()
        self.patcher_talk.stop()

    def test_interpret_eat_action_valid_with_param_initiator(self):
        parsed_response = {
            "action": "Eat",
            "parameters": {"item_name": "Apple", "initiator_id": "char1"},
        }
        action_instance = self.interpreter.interpret(
            parsed_response
        )  # No context initiator
        # Check that EatAction was called with item_name, initiator_id from params, and all params
        self.MockEatAction.assert_called_once_with(
            item_name="Apple", initiator_id="char1", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockEatAction.return_value)

    def test_interpret_eat_action_valid_with_context_initiator(self):
        parsed_response = {
            "action": "Eat",
            "parameters": {"item_name": "Pear"},
        }  # No initiator_id in params
        action_instance = self.interpreter.interpret(
            parsed_response, initiator_id_context="char_ctx"
        )
        self.MockEatAction.assert_called_once_with(
            item_name="Pear", initiator_id="char_ctx", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockEatAction.return_value)

    def test_interpret_eat_action_param_initiator_overrides_context(self):
        parsed_response = {
            "action": "Eat",
            "parameters": {"item_name": "Apple", "initiator_id": "char_param"},
        }
        action_instance = self.interpreter.interpret(
            parsed_response, initiator_id_context="char_ctx"
        )
        self.MockEatAction.assert_called_once_with(
            item_name="Apple",
            initiator_id="char_param",
            **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockEatAction.return_value)

    def test_interpret_goto_action_valid(self):
        parsed_response = {
            "action": "GoTo",
            "parameters": {"location_name": "Park", "speed": "fast"},
        }
        action_instance = self.interpreter.interpret(
            parsed_response, initiator_id_context="char2"
        )
        self.MockGoToAction.assert_called_once_with(
            location_name="Park", initiator_id="char2", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockGoToAction.return_value)

    def test_interpret_talk_action_valid(self):
        parsed_response = {
            "action": "Talk",
            "parameters": {
                "target_name": "John",
                "topic": "weather",
                "initiator_id": "char_speaker",
            },
        }
        action_instance = self.interpreter.interpret(parsed_response)
        # My interpreter's interpret method calls TalkAction:
        # TalkAction(initiator=initiator_id, target=parameters["target_name"], **parameters)
        self.MockTalkAction.assert_called_once_with(
            initiator="char_speaker", target="John", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockTalkAction.return_value)

    def test_interpret_talk_action_valid_with_context_initiator(self):
        parsed_response = {
            "action": "Talk",
            "parameters": {"target_name": "Jane", "topic": "news"},
        }  # No initiator_id in params
        action_instance = self.interpreter.interpret(
            parsed_response, initiator_id_context="char_ctx_speaker"
        )
        self.MockTalkAction.assert_called_once_with(
            initiator="char_ctx_speaker", target="Jane", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockTalkAction.return_value)

    def test_interpret_noop_action_valid(self):
        parsed_response = {"action": "NoOp", "parameters": {}}
        action_instance = self.interpreter.interpret(
            parsed_response, initiator_id_context="char_noop"
        )
        self.MockNoOpAction.assert_called_once_with(
            initiator_id="char_noop", **parsed_response["parameters"]
        )
        self.assertIs(action_instance, self.MockNoOpAction.return_value)

    def test_interpret_unknown_action(self):
        parsed_response = {"action": "Fly", "parameters": {"destination": "Moon"}}
        # Unknown actions should fallback to NoOp in our enhanced interpreter
        action_instance = self.interpreter.interpret(parsed_response)
        self.MockNoOpAction.assert_called_once()

    def test_interpret_eat_missing_item_name(self):
        parsed_response = {"action": "Eat", "parameters": {"initiator_id": "char1"}}
        # Our enhanced interpreter catches TypeError and wraps it
        with self.assertRaisesRegex(
            InvalidActionParametersError,
            "Parameter mismatch or missing required argument for action Eat",
        ):
            self.interpreter.interpret(parsed_response)

    def test_interpret_goto_missing_location_name(self):
        parsed_response = {"action": "GoTo", "parameters": {}}
        # Our enhanced interpreter catches TypeError and wraps it
        with self.assertRaisesRegex(
            InvalidActionParametersError,
            "Parameter mismatch or missing required argument for action GoTo",
        ):
            self.interpreter.interpret(parsed_response)

    def test_interpret_talk_missing_target_name(self):
        parsed_response = {
            "action": "Talk",
            "parameters": {"topic": "secrets", "initiator_id": "char_speaker"},
        }
        # Our enhanced interpreter catches TypeError and wraps it
        with self.assertRaisesRegex(
            InvalidActionParametersError,
            "Parameter mismatch or missing required argument for action Talk",
        ):
            self.interpreter.interpret(parsed_response)

    def test_interpret_talk_missing_initiator(self):
        # TalkAction requires initiator. If not in params and not in context, it should fail.
        parsed_response = {"action": "Talk", "parameters": {"target_name": "John"}}
        # Our enhanced interpreter catches TypeError and wraps it
        with self.assertRaisesRegex(
            InvalidActionParametersError,
            "Parameter mismatch or missing required argument for action Talk",
        ):
            self.interpreter.interpret(parsed_response)  # No context initiator

    def test_interpret_action_constructor_type_error(self):
        # This test simulates if an action (e.g., TalkAction) was called with parameters it cannot handle,
        # or if its __init__ raises a TypeError for other reasons.
        self.MockTalkAction.side_effect = TypeError("TalkAction internal TypeError")
        parsed_response = {
            "action": "Talk",
            "parameters": {"target_name": "John", "initiator_id": "char1"},
        }

        # The regex should match the message in InvalidActionParametersError raised by the interpreter
        with self.assertRaisesRegex(
            InvalidActionParametersError,
            "Parameter mismatch or missing required argument for action Talk: TalkAction internal TypeError",
        ):
            self.interpreter.interpret(parsed_response)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
