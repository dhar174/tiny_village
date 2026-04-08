import os
import sys
import unittest

# Ensure repository root is on path for module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tiny_output_interpreter import OutputInterpreter, InvalidLLMResponseFormatError


class TestOutputInterpreterResiliency(unittest.TestCase):
    def setUp(self):
        self.interpreter = OutputInterpreter()

    def test_structured_payload_passthrough(self):
        parsed = self.interpreter.parse_llm_response(
            {
                "parsed": {
                    "action": "Work",
                    "parameters": {"job_type": "coding"},
                    "reasoning": "need money",
                    "confidence": 0.9,
                    "raw": "raw-msg",
                }
            }
        )
        self.assertEqual(parsed["action"], "Work")
        self.assertEqual(parsed["parameters"]["job_type"], "coding")
        self.assertEqual(parsed.get("reasoning"), "need money")
        self.assertEqual(parsed.get("confidence"), 0.9)
        self.assertEqual(parsed.get("raw"), "raw-msg")

    def test_markdown_fenced_json(self):
        response = """Here is your action:
```json
{"action": "Eat", "parameters": {"item_name": "bread"}}
```
"""
        parsed = self.interpreter.parse_llm_response(response)
        self.assertEqual(parsed["action"], "Eat")
        self.assertEqual(parsed["parameters"]["item_name"], "bread")

    def test_prose_wrapped_json(self):
        response = "Great choice! {\"action\": \"Sleep\", \"parameters\": {\"duration\": 4}} Have a nice day."
        parsed = self.interpreter.parse_llm_response(response)
        self.assertEqual(parsed["action"], "Sleep")
        self.assertEqual(parsed["parameters"]["duration"], 4)

    def test_recovers_missing_closing_brace(self):
        response = '{"action": "Eat", "parameters": {"item_name": "Apple"'
        parsed = self.interpreter.parse_llm_response(response)
        self.assertEqual(parsed["action"], "Eat")
        self.assertEqual(parsed["parameters"]["item_name"], "Apple")

    def test_recovers_multiple_missing_braces(self):
        response = '{"action": "Work", "parameters": {"job_type": "dev"'
        parsed = self.interpreter.parse_llm_response(response)
        self.assertEqual(parsed["action"], "Work")
        self.assertEqual(parsed["parameters"]["job_type"], "dev")

    def test_recovers_outer_missing_brace(self):
        response = '"lead text" {"action": "Sleep", "parameters": {"duration": 2}'
        parsed = self.interpreter.parse_llm_response(response)
        self.assertEqual(parsed["action"], "Sleep")
        self.assertEqual(parsed["parameters"]["duration"], 2)

    def test_invalid_structured_payloads_raise(self):
        result = self.interpreter.parse_llm_response({"parsed": {"parameters": {}}})
        self.assertEqual(result["action"], "NoOp")
        result = self.interpreter.parse_llm_response({"action": "Eat", "parameters": "not-a-dict"})
        self.assertEqual(result["action"], "Eat")
        result = self.interpreter.parse_llm_response(
            {"parsed": {"action": "Eat", "parameters": {}, "confidence": 2.0}}
        )
        self.assertEqual(result["action"], "Eat")

    def test_unparseable_fallback(self):
        with self.assertLogs("tiny_output_interpreter", level="WARNING") as logs:
            parsed = self.interpreter.parse_llm_response("~~~")
        self.assertEqual(parsed["action"], "NoOp")
        self.assertTrue(any("LLM response parsing failed" in msg for msg in logs.output))

    def test_parser_recovery_sequence(self):
        # ensure later parsers are reached when earlier extraction fails
        text = "Prose then malformed {\"action\": \"Talk\", \"parameters\": {\"target_name\": \"Sam\"}"
        parsed = self.interpreter.parse_llm_response(text)
        self.assertEqual(parsed["action"], "Talk")
        self.assertEqual(parsed["parameters"]["target_name"], "Sam")


if __name__ == "__main__":
    unittest.main()
