import os
import sys
import unittest

# Ensure repository root is on path for module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tiny_output_interpreter import OutputInterpreter


class TestOutputInterpreterResiliency(unittest.TestCase):
    def setUp(self):
        self.interpreter = OutputInterpreter()

    def test_structured_payload_passthrough(self):
        parsed = self.interpreter.parse_llm_response(
            {"parsed": {"action": "Work", "parameters": {"job_type": "coding"}}}
        )
        self.assertEqual(parsed["action"], "Work")
        self.assertEqual(parsed["parameters"]["job_type"], "coding")

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

    def test_unparseable_fallback(self):
        parsed = self.interpreter.parse_llm_response("~~~")
        self.assertEqual(parsed["action"], "NoOp")


if __name__ == "__main__":
    unittest.main()
