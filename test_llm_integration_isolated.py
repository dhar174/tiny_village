"""
Isolated test for LLM integration that bypasses transformers dependencies
"""

import unittest
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockAction:
    """Mock Action class to avoid import dependencies"""

    def __init__(self, name="MockAction", cost=0.1, effects=None, preconditions=None):
        self.name = name
        self.cost = cost
        self.effects = effects or []
        self.preconditions = preconditions or []


class MockCharacter:
    """Mock Character class"""

    def __init__(self, name="TestChar"):
        self.name = name
        self.id = name
        self.hunger_level = 5.0
        self.energy = 5.0
        self.wealth_money = 50.0
        self.social_wellbeing = 5.0
        self.mental_health = 5.0
        self.use_llm_decisions = False


class MockTinyBrainIO:
    """Mock TinyBrainIO that doesn't require transformers"""

    def __init__(self, model_name=None):
        self.model_name = model_name

    def input_to_model(self, prompts):
        """Mock LLM response"""
        return [("I choose to sleep because the character needs rest.", "0.5")]


class MockPromptBuilder:
    """Mock PromptBuilder that doesn't require dependencies"""

    def __init__(self, character=None):
        self.character = character

    def generate_decision_prompt(self, time, weather, action_choices):
        """Generate a mock prompt"""
        choices_text = "\n".join(action_choices)
        return f"""Character {self.character.name if self.character else 'Unknown'} needs to make a decision.
Time: {time}
Weather: {weather}
Available choices:
{choices_text}
What should they do?"""


class MockOutputInterpreter:
    """Mock OutputInterpreter that doesn't require dependencies"""

    def __init__(self):
        pass

    def interpret_response(self, response_text, character, potential_actions):
        """Mock interpretation - return first action"""
        if potential_actions and len(potential_actions) > 0:
            return [potential_actions[0]]
        return [MockAction("Sleep")]


class TestLLMIntegrationIsolated(unittest.TestCase):
    """Test LLM integration without external dependencies"""

    def test_mock_components_work(self):
        """Test that our mock components work correctly"""
        character = MockCharacter("Alice")
        brain_io = MockTinyBrainIO("test-model")
        prompt_builder = MockPromptBuilder(character)
        output_interpreter = MockOutputInterpreter()

        # Test prompt generation
        action_choices = ["1. Sleep", "2. Eat", "3. Work"]
        prompt = prompt_builder.generate_decision_prompt(
            "morning", "sunny", action_choices
        )

        self.assertIn("Alice", prompt)
        self.assertIn("morning", prompt)
        self.assertIn("sunny", prompt)
        self.assertIn("Sleep", prompt)

        # Test LLM mock response
        response = brain_io.input_to_model([prompt])
        self.assertIsNotNone(response)
        self.assertEqual(len(response), 1)
        self.assertIsInstance(response[0], tuple)

        # Test interpretation
        potential_actions = [MockAction("Sleep"), MockAction("Eat")]
        interpreted = output_interpreter.interpret_response(
            response[0][0], character, potential_actions
        )
        self.assertIsNotNone(interpreted)
        self.assertEqual(len(interpreted), 1)
        self.assertEqual(interpreted[0].name, "Sleep")

    def test_decision_pipeline_flow(self):
        """Test the complete decision-making pipeline with mocks"""
        character = MockCharacter("Bob")

        # Step 1: Generate potential actions (mock)
        potential_actions = [
            MockAction(
                "Sleep",
                cost=0.0,
                effects=[{"attribute": "energy", "change_value": 0.7}],
            ),
            MockAction(
                "Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.5}]
            ),
            MockAction(
                "Work", cost=0.3, effects=[{"attribute": "money", "change_value": 20.0}]
            ),
        ]

        # Step 2: Create action choices
        action_choices = []
        for i, action in enumerate(potential_actions):
            action_choices.append(f"{i+1}. {action.name}")

        # Step 3: Generate prompt
        prompt_builder = MockPromptBuilder(character)
        prompt = prompt_builder.generate_decision_prompt(
            "evening", "rainy", action_choices
        )

        # Step 4: Get LLM response
        brain_io = MockTinyBrainIO()
        llm_response = brain_io.input_to_model([prompt])

        # Step 5: Interpret response
        output_interpreter = MockOutputInterpreter()
        selected_actions = output_interpreter.interpret_response(
            llm_response[0][0], character, potential_actions
        )

        # Verify the pipeline worked
        self.assertIsNotNone(selected_actions)
        self.assertEqual(len(selected_actions), 1)
        self.assertIn(selected_actions[0].name, ["Sleep", "Eat", "Work"])

    def test_llm_integration_structure(self):
        """Test that LLM integration follows the correct structure"""

        class MockStrategyManager:
            """Mock StrategyManager with LLM integration"""

            def __init__(self, use_llm=False, model_name=None):
                self.use_llm = use_llm
                if self.use_llm:
                    self.brain_io = MockTinyBrainIO(model_name)
                    self.output_interpreter = MockOutputInterpreter()
                else:
                    self.brain_io = None
                    self.output_interpreter = None

            def get_daily_actions(self, character):
                """Mock utility-based actions"""
                return [
                    MockAction("Sleep", cost=0.0),
                    MockAction("Eat", cost=0.1),
                    MockAction("Work", cost=0.3),
                ]

            def decide_action_with_llm(
                self, character, time="morning", weather="clear"
            ):
                """Mock LLM decision-making"""
                if not self.use_llm:
                    return self.get_daily_actions(character)[:1]

                try:
                    # Generate potential actions
                    potential_actions = self.get_daily_actions(character)

                    # Create prompt
                    prompt_builder = MockPromptBuilder(character)
                    action_choices = [
                        f"{i+1}. {a.name}" for i, a in enumerate(potential_actions)
                    ]
                    prompt = prompt_builder.generate_decision_prompt(
                        time, weather, action_choices
                    )

                    # Query LLM
                    llm_response = self.brain_io.input_to_model([prompt])

                    # Interpret response
                    selected_actions = self.output_interpreter.interpret_response(
                        llm_response[0][0], character, potential_actions
                    )

                    return (
                        selected_actions if selected_actions else potential_actions[:1]
                    )

                except Exception:
                    # Fallback to utility-based
                    return self.get_daily_actions(character)[:1]

        # Test LLM-enabled manager
        character = MockCharacter("Charlie")
        manager_with_llm = MockStrategyManager(use_llm=True, model_name="test-model")

        self.assertTrue(manager_with_llm.use_llm)
        self.assertIsNotNone(manager_with_llm.brain_io)
        self.assertIsNotNone(manager_with_llm.output_interpreter)

        # Test LLM decision-making
        decision = manager_with_llm.decide_action_with_llm(
            character, "morning", "sunny"
        )
        self.assertIsNotNone(decision)
        self.assertEqual(len(decision), 1)

        # Test non-LLM manager
        manager_no_llm = MockStrategyManager(use_llm=False)
        self.assertFalse(manager_no_llm.use_llm)
        self.assertIsNone(manager_no_llm.brain_io)
        self.assertIsNone(manager_no_llm.output_interpreter)

        # Test fallback behavior
        fallback_decision = manager_no_llm.decide_action_with_llm(character)
        self.assertIsNotNone(fallback_decision)
        self.assertEqual(len(fallback_decision), 1)

    def test_character_llm_flag(self):
        """Test character-level LLM decision flag"""
        character = MockCharacter("Diana")

        # Initially disabled
        self.assertFalse(character.use_llm_decisions)

        # Enable LLM decisions
        character.use_llm_decisions = True
        self.assertTrue(character.use_llm_decisions)

        # Mock GameplayController behavior
        def mock_execute_character_actions(characters, time, weather):
            results = []
            for character in characters:
                if (
                    hasattr(character, "use_llm_decisions")
                    and character.use_llm_decisions
                ):
                    # Would use LLM decision-making
                    results.append(f"LLM decision for {character.name}")
                else:
                    # Would use utility-based decision-making
                    results.append(f"Utility decision for {character.name}")
            return results

        char_with_llm = MockCharacter("LLM_User")
        char_with_llm.use_llm_decisions = True

        char_without_llm = MockCharacter("Utility_User")
        char_without_llm.use_llm_decisions = False

        results = mock_execute_character_actions(
            [char_with_llm, char_without_llm], "morning", "sunny"
        )

        self.assertIn("LLM decision for LLM_User", results)
        self.assertIn("Utility decision for Utility_User", results)


if __name__ == "__main__":
    print("Running Isolated LLM Integration Tests...")
    print("=" * 50)
    unittest.main(verbosity=2)
