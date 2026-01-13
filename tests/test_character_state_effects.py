"""
Tests for Character State Effects Implementation

Tests cover:
- Attribute mapping from template names to actual Character fields
- Effect handlers for character stats (health, energy, wealth, hunger, etc.)
- Bounds/clamping enforcement
- Graceful failure on missing attributes
- Integration with demo characters
"""

import sys
import os
import unittest
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from character_attribute_mapper import AttributeMapper
from effect_schema import EffectV2, EffectType
from effect_dispatcher import EffectDispatcher
from demo_character_factory import create_demo_character


class TestAttributeMapper(unittest.TestCase):
    """Test the AttributeMapper functionality."""
    
    def test_map_health_attribute(self):
        """Test mapping 'health' to 'health_status'."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("health")
        self.assertEqual(actual, "health_status")
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 10)
        self.assertEqual(default, 5)
    
    def test_map_happiness_to_social_wellbeing(self):
        """Test mapping 'happiness' to 'social_wellbeing'."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("happiness")
        self.assertEqual(actual, "social_wellbeing")
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 10)
    
    def test_map_wealth_attribute(self):
        """Test mapping 'wealth' to 'wealth_money'."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("wealth")
        self.assertEqual(actual, "wealth_money")
        self.assertEqual(min_val, 0)
        self.assertIsNone(max_val)  # No upper bound on wealth
    
    def test_map_energy_attribute(self):
        """Test mapping 'energy' to 'energy'."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("energy")
        self.assertEqual(actual, "energy")
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 10)
    
    def test_map_hunger_attribute(self):
        """Test mapping 'hunger' to 'hunger_level'."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("hunger")
        self.assertEqual(actual, "hunger_level")
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 10)
    
    def test_map_job_performance_attribute(self):
        """Test mapping 'job_performance' and aliases."""
        actual1, _, _, _ = AttributeMapper.map_attribute("job_performance")
        actual2, _, _, _ = AttributeMapper.map_attribute("productivity")
        self.assertEqual(actual1, "job_performance")
        self.assertEqual(actual2, "job_performance")
    
    def test_map_unmapped_attribute(self):
        """Test that unmapped attributes are returned as-is."""
        actual, min_val, max_val, default = AttributeMapper.map_attribute("custom_attr")
        self.assertEqual(actual, "custom_attr")
        self.assertIsNone(min_val)
        self.assertIsNone(max_val)
    
    def test_get_supported_attributes(self):
        """Test getting all supported attribute mappings."""
        supported = AttributeMapper.get_supported_attributes()
        self.assertIn("health", supported)
        self.assertIn("happiness", supported)
        self.assertIn("wealth", supported)
        self.assertIn("energy", supported)
        self.assertEqual(supported["health"], "health_status")
    
    def test_is_bounded_attribute(self):
        """Test checking if attributes have bounds."""
        self.assertTrue(AttributeMapper.is_bounded_attribute("health"))
        self.assertTrue(AttributeMapper.is_bounded_attribute("energy"))
        self.assertTrue(AttributeMapper.is_bounded_attribute("wealth"))  # Has min bound
        self.assertFalse(AttributeMapper.is_bounded_attribute("unmapped_attr"))


class TestEntity:
    """Simple test entity for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestAttributeMapperWithEntities(unittest.TestCase):
    """Test AttributeMapper with actual entities."""
    
    def test_get_attribute_value(self):
        """Test getting attribute value through mapper."""
        entity = TestEntity(health_status=8, social_wellbeing=7)
        
        health = AttributeMapper.get_attribute_value(entity, "health")
        self.assertEqual(health, 8)
        
        happiness = AttributeMapper.get_attribute_value(entity, "happiness")
        self.assertEqual(happiness, 7)
    
    def test_get_missing_attribute_returns_default(self):
        """Test that missing attributes return default value."""
        entity = TestEntity()
        
        health = AttributeMapper.get_attribute_value(entity, "health")
        self.assertEqual(health, 5)  # Default for health
    
    def test_set_attribute_value(self):
        """Test setting attribute value through mapper."""
        entity = TestEntity(health_status=5)
        
        success = AttributeMapper.set_attribute_value(entity, "health", 8)
        self.assertTrue(success)
        self.assertEqual(entity.health_status, 8)
    
    def test_set_attribute_with_bounds_clamping(self):
        """Test that bounds are applied when setting values."""
        entity = TestEntity(health_status=5)
        
        # Try to set above max
        AttributeMapper.set_attribute_value(entity, "health", 15, apply_bounds=True)
        self.assertEqual(entity.health_status, 10)  # Clamped to max
        
        # Try to set below min
        AttributeMapper.set_attribute_value(entity, "health", -5, apply_bounds=True)
        self.assertEqual(entity.health_status, 0)  # Clamped to min
    
    def test_set_attribute_without_bounds(self):
        """Test setting attribute without bounds enforcement."""
        entity = TestEntity(health_status=5)
        
        AttributeMapper.set_attribute_value(entity, "health", 15, apply_bounds=False)
        self.assertEqual(entity.health_status, 15)  # Not clamped


class TestCharacterHealthEffects(unittest.TestCase):
    """Test health-related character effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Health Event"
        self.mock_event.participants = []
        self.mock_event.location = None
    
    def test_health_increase_effect(self):
        """Test increasing character health."""
        character = TestEntity(health_status=5)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=3
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.health_status, 8)
    
    def test_health_decrease_effect(self):
        """Test decreasing character health."""
        character = TestEntity(health_status=5)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=-3
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.health_status, 2)
    
    def test_health_clamped_at_minimum(self):
        """Test that health is clamped at 0."""
        character = TestEntity(health_status=2)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=-10
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.health_status, 0)  # Clamped at min
    
    def test_health_clamped_at_maximum(self):
        """Test that health is clamped at 10."""
        character = TestEntity(health_status=8)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=10
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.health_status, 10)  # Clamped at max


class TestCharacterEnergyEffects(unittest.TestCase):
    """Test energy-related character effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Energy Event"
        self.mock_event.participants = []
    
    def test_energy_increase(self):
        """Test increasing character energy."""
        character = TestEntity(energy=5)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=2
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.energy, 7)
    
    def test_energy_decrease(self):
        """Test decreasing character energy."""
        character = TestEntity(energy=8)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=-5
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.energy, 3)
    
    def test_energy_bounds(self):
        """Test energy bounds enforcement."""
        character = TestEntity(energy=9)
        self.mock_event.participants = [character]
        
        # Test upper bound
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=5
        )
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(character.energy, 10)  # Clamped at max
        
        # Test lower bound
        effect2 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=-15
        )
        self.dispatcher.apply_effect(effect2, self.mock_event)
        self.assertEqual(character.energy, 0)  # Clamped at min


class TestCharacterWealthEffects(unittest.TestCase):
    """Test wealth/money-related character effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Wealth Event"
        self.mock_event.participants = []
    
    def test_wealth_increase(self):
        """Test increasing character wealth."""
        character = TestEntity(wealth_money=50)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="wealth",
            change_value=25
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.wealth_money, 75)
    
    def test_wealth_decrease(self):
        """Test decreasing character wealth."""
        character = TestEntity(wealth_money=50)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="wealth",
            change_value=-20
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.wealth_money, 30)
    
    def test_wealth_minimum_bound(self):
        """Test that wealth cannot go below 0."""
        character = TestEntity(wealth_money=10)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="wealth",
            change_value=-50
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.wealth_money, 0)  # Clamped at min
    
    def test_wealth_no_maximum_bound(self):
        """Test that wealth can grow indefinitely."""
        character = TestEntity(wealth_money=100)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="wealth",
            change_value=500
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.wealth_money, 600)  # No upper limit


class TestCharacterHungerEffects(unittest.TestCase):
    """Test hunger-related character effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Hunger Event"
        self.mock_event.participants = []
    
    def test_hunger_increase(self):
        """Test increasing hunger level (getting hungrier)."""
        character = TestEntity(hunger_level=3)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="hunger",
            change_value=2
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.hunger_level, 5)
    
    def test_hunger_decrease(self):
        """Test decreasing hunger level (getting less hungry)."""
        character = TestEntity(hunger_level=7)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="hunger",
            change_value=-3
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.hunger_level, 4)
    
    def test_hunger_bounds(self):
        """Test hunger level bounds."""
        character = TestEntity(hunger_level=8)
        self.mock_event.participants = [character]
        
        # Test upper bound
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="hunger",
            change_value=5
        )
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(character.hunger_level, 10)
        
        # Test lower bound
        effect2 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="hunger",
            change_value=-15
        )
        self.dispatcher.apply_effect(effect2, self.mock_event)
        self.assertEqual(character.hunger_level, 0)


class TestCharacterMentalHealthEffects(unittest.TestCase):
    """Test mental health and morale effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Mental Health Event"
        self.mock_event.participants = []
    
    def test_happiness_effect(self):
        """Test happiness effect (maps to social_wellbeing)."""
        character = TestEntity(social_wellbeing=5)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=3
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.social_wellbeing, 8)
    
    def test_morale_effect(self):
        """Test morale effect (maps to mental_health)."""
        character = TestEntity(mental_health=6)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="morale",
            change_value=2
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.mental_health, 8)
    
    def test_mental_health_bounds(self):
        """Test mental health bounds."""
        character = TestEntity(mental_health=9)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="mental_health",
            change_value=5
        )
        
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(character.mental_health, 10)  # Clamped at max


class TestCharacterJobPerformanceEffects(unittest.TestCase):
    """Test job performance and skill effects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Job Performance Event"
        self.mock_event.participants = []
    
    def test_job_performance_increase(self):
        """Test increasing job performance."""
        character = TestEntity(job_performance=50)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="job_performance",
            change_value=10
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.job_performance, 60)
    
    def test_productivity_alias(self):
        """Test that 'productivity' maps to job_performance."""
        character = TestEntity(job_performance=40)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="productivity",
            change_value=15
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.job_performance, 55)
    
    def test_job_performance_bounds(self):
        """Test job performance bounds (0-100)."""
        character = TestEntity(job_performance=95)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="job_performance",
            change_value=20
        )
        
        self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertEqual(character.job_performance, 100)  # Clamped at max


class TestMissingAttributeHandling(unittest.TestCase):
    """Test graceful handling of missing attributes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Missing Attribute Event"
        self.mock_event.participants = []
    
    def test_missing_attribute_creates_default(self):
        """Test that missing attributes are created with default values."""
        character = TestEntity()  # No attributes set
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=3
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        # Should create attribute with default (5) + change (3) = 8
        self.assertTrue(hasattr(character, "health_status"))
        self.assertEqual(character.health_status, 8)
    
    def test_effect_fails_gracefully_on_error(self):
        """Test that effect application doesn't crash on errors."""
        # Create an entity that will cause AttributeMapper to fail
        class FailingEntity:
            def __setattr__(self, name, value):
                raise AttributeError("Test error: cannot set attribute")
        
        character = FailingEntity()
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=3
        )
        
        # Should not raise exception, just return False or log error
        try:
            result = self.dispatcher.apply_effect(effect, self.mock_event)
            # The key is it shouldn't crash, and it should signal failure via the result
            self.assertFalse(result)
        except Exception as e:
            self.fail(f"Effect application should not raise exception: {e}")


class TestDemoCharacterIntegration(unittest.TestCase):
    """Test effects with actual DemoRealCharacter instances."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = EffectDispatcher(None)
        self.mock_event = Mock()
        self.mock_event.name = "Demo Character Event"
        self.mock_event.participants = []
    
    def test_demo_character_health_effect(self):
        """Test health effect on demo character."""
        character = create_demo_character("Alice", health_status=7)
        self.mock_event.participants = [character]
        
        effect = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=-2
        )
        
        result = self.dispatcher.apply_effect(effect, self.mock_event)
        self.assertTrue(result)
        self.assertEqual(character.health_status, 5)
    
    def test_demo_character_multiple_effects(self):
        """Test multiple effects on demo character."""
        character = create_demo_character(
            "Bob",
            health_status=8,
            energy=6,
            wealth_money=100
        )
        self.mock_event.participants = [character]
        
        # Apply health effect
        effect1 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="health",
            change_value=-3
        )
        self.dispatcher.apply_effect(effect1, self.mock_event)
        
        # Apply energy effect
        effect2 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=2
        )
        self.dispatcher.apply_effect(effect2, self.mock_event)
        
        # Apply wealth effect
        effect3 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="wealth",
            change_value=25
        )
        self.dispatcher.apply_effect(effect3, self.mock_event)
        
        # Verify all effects applied
        self.assertEqual(character.health_status, 5)
        self.assertEqual(character.energy, 8)
        self.assertEqual(character.wealth_money, 125)
    
    def test_demo_character_with_aliases(self):
        """Test that attribute aliases work with demo characters."""
        character = create_demo_character(
            "Charlie",
            social_wellbeing=5,
            mental_health=6,
            job_performance=50
        )
        self.mock_event.participants = [character]
        
        # Use alias 'happiness' for social_wellbeing
        effect1 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=2
        )
        self.dispatcher.apply_effect(effect1, self.mock_event)
        self.assertEqual(character.social_wellbeing, 7)
        
        # Use alias 'morale' for mental_health
        effect2 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="morale",
            change_value=1
        )
        self.dispatcher.apply_effect(effect2, self.mock_event)
        self.assertEqual(character.mental_health, 7)
        
        # Use alias 'productivity' for job_performance
        effect3 = EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="productivity",
            change_value=10
        )
        self.dispatcher.apply_effect(effect3, self.mock_event)
        self.assertEqual(character.job_performance, 60)


def run_tests():
    """Run all tests and provide summary."""
    print("Running Character State Effects Tests...")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAttributeMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestAttributeMapperWithEntities))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterHealthEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterEnergyEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterWealthEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterHungerEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterMentalHealthEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterJobPerformanceEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingAttributeHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestDemoCharacterIntegration))
    
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
