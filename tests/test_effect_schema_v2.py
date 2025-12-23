"""
Comprehensive tests for Effect Schema v2 (Typed + Validated).

Tests cover:
- Valid effect parsing/validation
- Missing required fields
- Invalid types/operators
- Invalid target specs
- Effect application with dispatcher
- Backward compatibility
"""

import sys
import os
import unittest
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from effect_schema import (
    EffectV2,
    EffectType,
    EffectCondition,
    OperatorType,
    validate_effect_dict,
    create_canonical_effects
)
from effect_dispatcher import EffectDispatcher


class TestEntity:
    """Simple test entity for testing effects without mocks."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestEffectCondition(unittest.TestCase):
    """Test EffectCondition validation and evaluation."""
    
    def test_condition_evaluation(self):
        """Test various condition operators."""
        # Greater than or equal
        cond = EffectCondition("energy", ">=", 50)
        self.assertTrue(cond.evaluate(60))
        self.assertTrue(cond.evaluate(50))
        self.assertFalse(cond.evaluate(40))
        
        # Greater than
        cond = EffectCondition("health", ">", 30)
        self.assertTrue(cond.evaluate(40))
        self.assertFalse(cond.evaluate(30))
        
        # Less than or equal
        cond = EffectCondition("stress", "<=", 70)
        self.assertTrue(cond.evaluate(60))
        self.assertTrue(cond.evaluate(70))
        self.assertFalse(cond.evaluate(80))
        
        # Equal
        cond = EffectCondition("level", "==", 5)
        self.assertTrue(cond.evaluate(5))
        self.assertFalse(cond.evaluate(4))
        
        # Not equal
        cond = EffectCondition("status", "!=", "dead")
        self.assertTrue(cond.evaluate("alive"))
        self.assertFalse(cond.evaluate("dead"))
    
    def test_invalid_operator(self):
        """Test that invalid operators are rejected."""
        with self.assertRaises(ValueError) as context:
            EffectCondition("energy", "~=", 50)
        self.assertIn("Invalid condition operator", str(context.exception))


class TestEffectV2Validation(unittest.TestCase):
    """Test EffectV2 validation logic."""
    
    def test_valid_effect_creation(self):
        """Test creating valid effects."""
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10
        )
        self.assertEqual(effect.type, EffectType.ATTRIBUTE_CHANGE)
        self.assertEqual(effect.targets, ["participants"])
        self.assertEqual(effect.attribute, "happiness")
        self.assertEqual(effect.change_value, 10)
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        # Missing targets
        with self.assertRaises(TypeError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                attribute="happiness",
                change_value=10
            )
        
        # Missing attribute
        with self.assertRaises(TypeError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets=["participants"],
                change_value=10
            )
    
    def test_invalid_effect_type(self):
        """Test that invalid effect types are rejected."""
        with self.assertRaises(ValueError):
            EffectV2(
                type="invalid_type",
                targets=["participants"],
                attribute="happiness",
                change_value=10
            )
    
    def test_invalid_targets(self):
        """Test that invalid target specs are rejected."""
        # Empty targets
        with self.assertRaises(ValueError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets=[],
                attribute="happiness",
                change_value=10
            )
        
        # Non-list targets
        with self.assertRaises(ValueError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets="participants",
                attribute="happiness",
                change_value=10
            )
    
    def test_invalid_operator(self):
        """Test that invalid operators are rejected."""
        with self.assertRaises(ValueError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets=["participants"],
                attribute="happiness",
                change_value=10,
                operator="invalid_op"
            )
    
    def test_invalid_change_value_for_numeric_operator(self):
        """Test that non-numeric change values are rejected for numeric operators."""
        with self.assertRaises(ValueError):
            EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets=["participants"],
                attribute="happiness",
                change_value="not a number",
                operator=OperatorType.ADD
            )
    
    def test_effect_with_conditions(self):
        """Test effect with conditions."""
        condition = EffectCondition("energy", ">=", 50)
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="productivity",
            change_value=5,
            conditions=[condition]
        )
        self.assertEqual(len(effect.conditions), 1)
        self.assertEqual(effect.conditions[0].attribute, "energy")
    
    def test_effect_with_chain(self):
        """Test effect with chained attributes."""
        effect = EffectV2(
            type=EffectType.RELATIONSHIP_CHANGE,
            targets=["participants"],
            attribute="trust",
            change_value=5,
            chain=["friendship_level", "loyalty"]
        )
        self.assertEqual(len(effect.chain), 2)
        self.assertIn("friendship_level", effect.chain)


class TestEffectV2Serialization(unittest.TestCase):
    """Test EffectV2 serialization and deserialization."""
    
    def test_to_dict(self):
        """Test converting effect to dictionary."""
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10,
            description="Test effect"
        )
        effect_dict = effect.to_dict()
        
        self.assertEqual(effect_dict["type"], "attribute_change")
        self.assertEqual(effect_dict["targets"], ["participants"])
        self.assertEqual(effect_dict["attribute"], "happiness")
        self.assertEqual(effect_dict["change_value"], 10)
        self.assertEqual(effect_dict["description"], "Test effect")
    
    def test_from_dict(self):
        """Test creating effect from dictionary."""
        effect_dict = {
            "type": "attribute_change",
            "targets": ["participants"],
            "attribute": "happiness",
            "change_value": 10,
            "operator": "add"
        }
        effect = EffectV2.from_dict(effect_dict)
        
        self.assertEqual(effect.type, EffectType.ATTRIBUTE_CHANGE)
        self.assertEqual(effect.targets, ["participants"])
        self.assertEqual(effect.attribute, "happiness")
        self.assertEqual(effect.change_value, 10)
    
    def test_from_dict_with_conditions(self):
        """Test creating effect with conditions from dictionary."""
        effect_dict = {
            "type": "attribute_change",
            "targets": ["participants"],
            "attribute": "productivity",
            "change_value": 5,
            "conditions": [
                {"attribute": "energy", "operator": ">=", "threshold": 50}
            ]
        }
        effect = EffectV2.from_dict(effect_dict)
        
        self.assertEqual(len(effect.conditions), 1)
        self.assertEqual(effect.conditions[0].attribute, "energy")
        self.assertEqual(effect.conditions[0].threshold, 50)
    
    def test_validate_effect_dict(self):
        """Test effect dict validation function."""
        valid_dict = {
            "type": "attribute_change",
            "targets": ["participants"],
            "attribute": "happiness",
            "change_value": 10
        }
        self.assertTrue(validate_effect_dict(valid_dict))
        
        invalid_dict = {
            "type": "invalid_type",
            "targets": ["participants"],
            "attribute": "happiness"
        }
        self.assertFalse(validate_effect_dict(invalid_dict))


class TestEffectDispatcher(unittest.TestCase):
    """Test the EffectDispatcher functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.get_node = Mock(return_value=None)
        self.mock_graph_manager.update_character_character_edge = Mock()
        
        self.dispatcher = EffectDispatcher(self.mock_graph_manager)
        
        # Create mock event
        self.mock_event = Mock()
        self.mock_event.name = "Test Event"
        self.mock_event.participants = []
        self.mock_event.location = None
    
    def test_apply_attribute_change_to_participants(self):
        """Test applying attribute change to participants."""
        # Create test participant
        participant = TestEntity(happiness=50)
        self.mock_event.participants = [participant]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(participant.happiness, 60)
    
    def test_apply_attribute_change_with_condition(self):
        """Test applying conditional attribute change."""
        # Create participants with different energy levels
        participant1 = TestEntity(energy=60, productivity=50)
        participant2 = TestEntity(energy=30, productivity=50)
        
        self.mock_event.participants = [participant1, participant2]
        
        # Effect only applies if energy >= 50
        condition = EffectCondition("energy", ">=", 50)
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="productivity",
            change_value=5,
            conditions=[condition]
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        
        # Only participant1 should have increased productivity
        self.assertEqual(participant1.productivity, 55)
        self.assertEqual(participant2.productivity, 50)
    
    def test_apply_different_operators(self):
        """Test applying effects with different operators."""
        participant = TestEntity(score=100)
        self.mock_event.participants = [participant]
        
        # Test SET operator
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="score",
            change_value=50,
            operator=OperatorType.SET
        )
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(participant.score, 50)
        
        # Test MULTIPLY operator
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="score",
            change_value=2,
            operator=OperatorType.MULTIPLY
        )
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(participant.score, 100)
        
        # Test MAX operator
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="score",
            change_value=150,
            operator=OperatorType.MAX
        )
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(participant.score, 150)
    
    def test_apply_location_change(self):
        """Test applying effect to location."""
        location = TestEntity(development_level=10)
        self.mock_event.location = location
        
        effect = EffectV2(
            type=EffectType.LOCATION_CHANGE,
            targets=["location"],
            attribute="development_level",
            change_value=5
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(location.development_level, 15)
    
    def test_apply_relationship_change(self):
        """Test applying relationship change effect."""
        participant1 = Mock()
        participant2 = Mock()
        self.mock_event.participants = [participant1, participant2]
        
        # Set up graph manager to say edge exists
        self.mock_graph_manager.G.has_edge = Mock(return_value=True)
        
        effect = EffectV2(
            type=EffectType.RELATIONSHIP_CHANGE,
            targets=["participants"],
            attribute="trust",
            change_value=5
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        
        # Verify relationship update was called
        self.mock_graph_manager.update_character_character_edge.assert_called_once()
    
    def test_effect_with_chain(self):
        """Test applying effect with chained attributes."""
        participant = TestEntity(trust=50, friendship_level=30, loyalty=20)
        self.mock_event.participants = [participant]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="trust",
            change_value=5,
            chain=["friendship_level", "loyalty"]
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        
        # Check main attribute and chained attributes
        self.assertEqual(participant.trust, 55)
        self.assertEqual(participant.friendship_level, 35)
        self.assertEqual(participant.loyalty, 25)
    
    def test_dispatcher_summary(self):
        """Test getting applied effects summary."""
        participant = TestEntity(happiness=50, energy=100)
        self.mock_event.participants = [participant]
        
        effect1 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10
        )
        
        effect2 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=-5
        )
        
        self.dispatcher.apply_effect(effect1, self.mock_event)
        self.dispatcher.apply_effect(effect2, self.mock_event)
        
        summary = self.dispatcher.get_applied_effects_summary()
        self.assertEqual(summary["total_effects"], 2)
        self.assertEqual(summary["by_type"]["attribute_change"], 2)


class TestCanonicalEffects(unittest.TestCase):
    """Test canonical effect examples."""
    
    def test_canonical_effects_creation(self):
        """Test that all canonical effects are valid."""
        effects = create_canonical_effects()
        
        self.assertIn("happiness_boost", effects)
        self.assertIn("conditional_energy_drain", effects)
        self.assertIn("relationship_trust_boost", effects)
        self.assertIn("location_development", effects)
        self.assertIn("world_economy_boost", effects)
        
        # Verify each effect is valid
        for name, effect in effects.items():
            self.assertIsInstance(effect, EffectV2)
            # Should not raise validation errors
            effect.validate()
    
    def test_canonical_happiness_boost(self):
        """Test the canonical happiness boost effect."""
        effects = create_canonical_effects()
        effect = effects["happiness_boost"]
        
        self.assertEqual(effect.type, EffectType.ATTRIBUTE_CHANGE)
        self.assertEqual(effect.attribute, "happiness")
        self.assertEqual(effect.change_value, 10)
    
    def test_canonical_conditional_effect(self):
        """Test the canonical conditional effect."""
        effects = create_canonical_effects()
        effect = effects["conditional_energy_drain"]
        
        self.assertEqual(len(effect.conditions), 1)
        self.assertEqual(effect.conditions[0].attribute, "energy")
        self.assertEqual(effect.conditions[0].operator, ">=")
    
    def test_canonical_chained_effect(self):
        """Test the canonical chained effect."""
        effects = create_canonical_effects()
        effect = effects["relationship_trust_boost"]
        
        self.assertGreater(len(effect.chain), 0)
        self.assertIn("friendship_level", effect.chain)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old dict-based effects."""
    
    def test_convert_old_effect_format(self):
        """Test converting old dict format to EffectV2."""
        old_effect = {
            "type": "attribute_change",
            "targets": ["participants"],
            "attribute": "happiness",
            "change_value": 10
        }
        
        effect = EffectV2.from_dict(old_effect)
        self.assertIsInstance(effect, EffectV2)
        self.assertEqual(effect.type, EffectType.ATTRIBUTE_CHANGE)
    
    def test_old_relationship_change_format(self):
        """Test converting old relationship change format."""
        old_effect = {
            "type": "relationship_change",
            "targets": ["participants"],
            "attribute": "relationship_strength",
            "change_value": 5
        }
        
        effect = EffectV2.from_dict(old_effect)
        self.assertIsInstance(effect, EffectV2)
        self.assertEqual(effect.type, EffectType.RELATIONSHIP_CHANGE)


def run_tests():
    """Run all tests and provide summary."""
    print("Running Effect Schema v2 Tests...")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEffectCondition))
    suite.addTests(loader.loadTestsFromTestCase(TestEffectV2Validation))
    suite.addTests(loader.loadTestsFromTestCase(TestEffectV2Serialization))
    suite.addTests(loader.loadTestsFromTestCase(TestEffectDispatcher))
    suite.addTests(loader.loadTestsFromTestCase(TestCanonicalEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(traceback)
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(traceback)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
