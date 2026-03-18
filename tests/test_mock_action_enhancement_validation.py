import unittest

from tiny_utility_functions import (
    UtilityEvaluator,
    safe_calculate_action_utility,
    validate_action,
)

try:
    from test_tiny_utility_functions import MockAction, MockGoal
except ImportError:
    from tests.test_tiny_utility_functions import MockAction, MockGoal


class OldStyleMockAction:
    """Simplified legacy mock used to show why the enhanced mock is needed."""

    def __init__(self, name, cost, effects=None):
        self.name = name
        self.cost = cost
        self.effects = effects or []


class TestMockActionEnhancementValidation(unittest.TestCase):
    def test_preconditions_can_block_execution(self):
        blocked_action = MockAction(
            "BlockedAction",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.5}],
            preconditions=[True, False],
        )

        self.assertFalse(blocked_action.preconditions_met())

    def test_invalid_effects_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must have 'attribute' key"):
            MockAction(
                "InvalidAction",
                cost=0.1,
                effects=[{"missing_change_value": "invalid"}],
            )

        with self.assertRaisesRegex(ValueError, "change_value must be numeric"):
            MockAction(
                "InvalidAction2",
                cost=0.1,
                effects=[{"attribute": "hunger", "change_value": "not_a_number"}],
            )

    def test_target_and_initiator_relationships_are_preserved(self):
        self_action = MockAction(
            "SelfTargetingAction",
            cost=0.1,
            initiator="character_bob",
            default_target_is_initiator=True,
        )
        social_action = MockAction(
            "SocialAction",
            cost=0.1,
            initiator="character_alice",
            target="character_bob",
        )

        self.assertEqual(self_action.target, "character_bob")
        self.assertEqual(self_action.initiator, "character_bob")
        self.assertEqual(social_action.initiator, "character_alice")
        self.assertEqual(social_action.target, "character_bob")

    def test_urgency_influences_utility_evaluator(self):
        evaluator = UtilityEvaluator()
        char_state = {"hunger": 0.9, "energy": 0.5, "health": 0.8}

        urgent_goal = MockGoal(
            "UrgentSurvival",
            target_effects={"hunger": -0.8},
            priority=1.0,
            urgency=0.95,
        )
        urgent_action = MockAction(
            "UrgentEat",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.8}],
            priority=0.9,
            related_goal=urgent_goal,
        )
        normal_goal = MockGoal(
            "NormalEat",
            target_effects={"hunger": -0.8},
            priority=1.0,
            urgency=0.5,
        )
        normal_action = MockAction(
            "NormalEat",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.8}],
            priority=0.5,
        )

        urgent_utility = evaluator.evaluate_action_utility(
            "survivor", char_state, urgent_action, urgent_goal
        )
        normal_utility = evaluator.evaluate_action_utility(
            "survivor", char_state, normal_action, normal_goal
        )

        self.assertGreater(urgent_utility, normal_utility)

    def test_enhanced_mock_works_with_validation_helpers(self):
        valid_action = MockAction(
            "ValidAction",
            cost=0.2,
            effects=[
                {"attribute": "hunger", "change_value": -0.5},
                {"attribute": "energy", "change_value": 0.3},
            ],
            preconditions=[True],
            priority=0.7,
        )

        is_valid, error = validate_action(valid_action)
        utility, utility_error = safe_calculate_action_utility(
            {"hunger": 0.8, "energy": 0.3, "health": 0.9},
            valid_action,
            validate_inputs=True,
        )

        self.assertTrue(is_valid, error)
        self.assertEqual(utility_error, "")
        self.assertGreater(utility, 0)

    def test_enhanced_mock_supports_validation_better_than_legacy_mock(self):
        old_action = OldStyleMockAction(
            "OldAction",
            0.1,
            [{"attribute": "hunger", "change_value": -0.5}],
        )
        enhanced_action = MockAction(
            "EnhancedEquivalent",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.5}],
        )

        old_is_valid, _ = validate_action(old_action)
        enhanced_is_valid, enhanced_error = validate_action(enhanced_action)

        self.assertTrue(old_is_valid)
        self.assertTrue(enhanced_is_valid, enhanced_error)
        self.assertTrue(hasattr(enhanced_action, "preconditions_met"))
        self.assertTrue(hasattr(enhanced_action, "add_effect"))
        self.assertTrue(hasattr(enhanced_action, "add_precondition"))

    def test_enhanced_mock_mutation_helpers_work(self):
        action = MockAction("TestAction", cost=0.1)

        action.add_precondition(True)
        action.add_effect({"attribute": "test", "change_value": 1.0})

        self.assertEqual(len(action.preconditions), 1)
        self.assertEqual(len(action.effects), 1)
        self.assertTrue(action.preconditions_met())


if __name__ == "__main__":
    unittest.main()
