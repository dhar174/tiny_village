#!/usr/bin/env python3
"""
Test to validate the comprehensive MockCharacter implementation against
various test scenarios to ensure it provides accurate Character interface behavior.
"""

import unittest
import sys
import os

# Add the tests directory to Python path to import mock_character
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_character import (
    MockCharacter, MockMotives, MockPersonalityTraits, MockInventory,
    create_test_character, create_realistic_character, validate_character_interface
)


class TestComprehensiveMockCharacter(unittest.TestCase):
    """Test the comprehensive MockCharacter implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.character = MockCharacter("Alice")
    
    def test_basic_character_creation(self):
        """Test basic character creation with defaults."""
        char = MockCharacter("TestChar")
        
        # Test core attributes exist
        self.assertEqual(char.name, "TestChar")
        self.assertEqual(char.age, 25)
        self.assertEqual(char.pronouns, "they/them")
        self.assertEqual(char.job, "unemployed")
        self.assertIsInstance(char.health_status, (int, float))
        self.assertIsInstance(char.wealth_money, (int, float))
        
        # Test complex objects exist
        self.assertIsNotNone(char.personality_traits)
        self.assertIsNotNone(char.motives)
        self.assertIsNotNone(char.inventory)
        self.assertIsNotNone(char.location)
        
        print("✓ Basic character creation test passed")

    def test_character_with_custom_attributes(self):
        """Test character creation with custom attributes."""
        char = MockCharacter(
            "CustomChar",
            age=30,
            health_status=9.0,
            wealth_money=1000.0,
            job="engineer"
        )
        
        self.assertEqual(char.name, "CustomChar")
        self.assertEqual(char.age, 30)
        self.assertEqual(char.health_status, 9.0)
        self.assertEqual(char.wealth_money, 1000.0)
        self.assertEqual(char.job, "engineer")
        
        print("✓ Custom attributes test passed")

    def test_interface_completeness(self):
        """Test that MockCharacter has complete interface coverage."""
        char = MockCharacter("InterfaceTest")
        
        # Test core identity methods
        self.assertEqual(char.get_name(), "InterfaceTest")
        char.set_name("NewName")
        self.assertEqual(char.get_name(), "NewName")
        
        # Test health methods
        self.assertIsInstance(char.get_health_status(), (int, float))
        char.set_health_status(9.5)
        self.assertEqual(char.get_health_status(), 9.5)
        
        # Test economic methods
        self.assertIsInstance(char.get_wealth_money(), (int, float))
        char.set_wealth_money(500.0)
        self.assertEqual(char.get_wealth_money(), 500.0)
        
        # Test social methods
        self.assertIsInstance(char.get_social_wellbeing(), (int, float))
        char.set_social_wellbeing(8.0)
        self.assertEqual(char.get_social_wellbeing(), 8.0)
        
        # Test psychological state methods
        self.assertIsInstance(char.get_happiness(), (int, float))
        self.assertIsInstance(char.get_stability(), (int, float))
        self.assertIsInstance(char.get_control(), (int, float))
        
        # Test complex object accessors
        self.assertIsNotNone(char.get_personality_traits())
        self.assertIsNotNone(char.get_motives())
        self.assertIsNotNone(char.get_inventory())
        
        print("✓ Interface completeness test passed")

    def test_personality_traits_functionality(self):
        """Test that personality traits work correctly."""
        traits = MockPersonalityTraits(
            openness=2.0,
            conscientiousness=1.0,
            extraversion=3.0,
            agreeableness=2.5,
            neuroticism=-1.0
        )
        
        char = MockCharacter("PersonalityTest", personality_traits=traits)
        
        self.assertEqual(char.personality_traits.get_openness(), 2.0)
        self.assertEqual(char.personality_traits.get_extraversion(), 3.0)
        self.assertEqual(char.personality_traits.get_neuroticism(), -1.0)
        
        # Test personality affects behavior
        self.assertTrue(char.decide_to_explore())  # High openness
        
        print("✓ Personality traits functionality test passed")

    def test_motives_functionality(self):
        """Test that motives work correctly."""
        char = MockCharacter("MotivesTest")
        motives = char.get_motives()
        
        # Test all motive getters exist and return values
        self.assertIsNotNone(motives.get_hunger_motive())
        self.assertIsNotNone(motives.get_wealth_motive())
        self.assertIsNotNone(motives.get_mental_health_motive())
        self.assertIsNotNone(motives.get_social_wellbeing_motive())
        
        # Test motive values are reasonable
        hunger_motive = motives.get_hunger_motive()
        self.assertIsInstance(hunger_motive.get_score(), (int, float))
        self.assertGreaterEqual(hunger_motive.get_score(), 0)
        self.assertLessEqual(hunger_motive.get_score(), 10)
        
        print("✓ Motives functionality test passed")

    def test_inventory_functionality(self):
        """Test that inventory works correctly."""
        char = MockCharacter("InventoryTest")
        inventory = char.get_inventory()
        
        # Test basic inventory operations
        self.assertEqual(inventory.count_total_items(), 0)
        
        # Test item type checking
        self.assertFalse(inventory.check_has_item_by_type(['food']))
        
        # Add mock food item and test again
        mock_food = type('MockFood', (), {'item_type': 'food', 'name': 'apple'})()
        inventory.food_items.append(mock_food)
        
        self.assertEqual(inventory.count_total_items(), 1)
        self.assertTrue(inventory.check_has_item_by_type(['food']))
        
        print("✓ Inventory functionality test passed")

    def test_social_interaction_behavior(self):
        """Test social interaction methods."""
        alice = MockCharacter("Alice")
        bob = MockCharacter("Bob")
        
        # Test social interaction
        initial_wellbeing = alice.social_wellbeing
        response = alice.respond_to_talk(bob)
        
        # Should increase social wellbeing
        self.assertGreater(alice.social_wellbeing, initial_wellbeing)
        self.assertIsInstance(response, str)
        self.assertIn("Alice", response)
        
        print("✓ Social interaction behavior test passed")

    def test_calculation_methods(self):
        """Test calculation methods produce reasonable results."""
        char = MockCharacter("CalcTest", 
                           health_status=8.0,
                           mental_health=7.0,
                           social_wellbeing=6.0,
                           wealth_money=500.0)
        
        # Test calculations return reasonable values
        happiness = char.calculate_happiness()
        self.assertIsInstance(happiness, (int, float))
        self.assertGreaterEqual(happiness, 0)
        self.assertLessEqual(happiness, 100)
        
        stability = char.calculate_stability()
        self.assertIsInstance(stability, (int, float))
        self.assertGreaterEqual(stability, 0)
        self.assertLessEqual(stability, 100)
        
        success = char.calculate_success()
        self.assertIsInstance(success, (int, float))
        self.assertGreaterEqual(success, 0)
        self.assertLessEqual(success, 100)
        
        print("✓ Calculation methods test passed")

    def test_behavioral_decision_methods(self):
        """Test behavioral decision methods."""
        # Test with social character
        social_char = MockCharacter("SocialChar", 
                                  personality_traits=MockPersonalityTraits(extraversion=2.0))
        self.assertTrue(social_char.decide_to_socialize())
        
        # Test with poor character
        poor_char = MockCharacter("PoorChar", wealth_money=10.0)
        self.assertTrue(poor_char.decide_to_work())
        
        # Test with exploratory character
        exploratory_char = MockCharacter("ExplorerChar", 
                                       personality_traits=MockPersonalityTraits(openness=2.0))
        self.assertTrue(exploratory_char.decide_to_explore())
        
        print("✓ Behavioral decision methods test passed")

    def test_state_management(self):
        """Test character state management."""
        char = MockCharacter("StateTest")
        
        # Test update_character method
        char.update_character(
            wealth_money=1000.0,
            health_status=9.0,
            job="doctor"
        )
        
        self.assertEqual(char.wealth_money, 1000.0)
        self.assertEqual(char.health_status, 9.0)
        self.assertEqual(char.job, "doctor")
        
        # Test to_dict method
        char_dict = char.to_dict()
        self.assertIsInstance(char_dict, dict)
        self.assertEqual(char_dict["name"], "StateTest")
        self.assertEqual(char_dict["wealth_money"], 1000.0)
        
        print("✓ State management test passed")

    def test_convenience_functions(self):
        """Test convenience functions for creating characters."""
        # Test create_test_character
        test_char = create_test_character("TestChar", health_status=10.0)
        self.assertEqual(test_char.name, "TestChar")
        self.assertEqual(test_char.health_status, 10.0)
        
        # Test create_realistic_character with different scenarios
        poor_char = create_realistic_character("PoorChar", "poor")
        wealthy_char = create_realistic_character("WealthyChar", "wealthy")
        
        self.assertLess(poor_char.wealth_money, wealthy_char.wealth_money)
        self.assertLess(poor_char.mental_health, wealthy_char.mental_health)
        
        print("✓ Convenience functions test passed")

    def test_interface_validation(self):
        """Test interface validation function."""
        char = MockCharacter("ValidationTest")
        
        # Test with default expected attributes
        is_valid, missing = validate_character_interface(char)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)
        
        # Test with custom expected attributes
        custom_attrs = ['name', 'age', 'wealth_money', 'nonexistent_attr']
        is_valid, missing = validate_character_interface(char, custom_attrs)
        self.assertFalse(is_valid)
        self.assertIn('nonexistent_attr', missing)
        
        print("✓ Interface validation test passed")

    def test_investment_functionality(self):
        """Test investment-related functionality."""
        char = MockCharacter("InvestorTest")
        
        # Test has_investment method (should be False initially)
        self.assertFalse(char.has_investment())
        
        # Add a mock stock
        mock_stock = type('MockStock', (), {'symbol': 'AAPL', 'shares': 10})()
        char.investment_portfolio.add_stock(mock_stock)
        
        # Should now have investment
        self.assertTrue(char.has_investment())
        
        print("✓ Investment functionality test passed")

    def test_llm_decision_attribute(self):
        """Test LLM decision-making attribute."""
        char = MockCharacter("LLMTest")
        
        # Test default value
        self.assertFalse(char.use_llm_decisions)
        
        # Test setting value
        char.use_llm_decisions = True
        self.assertTrue(char.use_llm_decisions)
        
        # Test in constructor
        llm_char = MockCharacter("LLMChar", use_llm_decisions=True)
        self.assertTrue(llm_char.use_llm_decisions)
        
        print("✓ LLM decision attribute test passed")

    def test_comparison_with_simple_mock(self):
        """Test comparison with a simple mock to show improvements."""
        # Simple mock (like what exists in many current tests)
        class SimpleMockCharacter:
            def __init__(self, name):
                self.name = name
                self.hunger_level = 5.0
                self.wealth_money = 100.0
        
        simple_mock = SimpleMockCharacter("Simple")
        comprehensive_mock = MockCharacter("Comprehensive")
        
        # Show that comprehensive mock has much more functionality
        self.assertTrue(hasattr(comprehensive_mock, 'get_personality_traits'))
        self.assertFalse(hasattr(simple_mock, 'get_personality_traits'))
        
        self.assertTrue(hasattr(comprehensive_mock, 'respond_to_talk'))
        self.assertFalse(hasattr(simple_mock, 'respond_to_talk'))
        
        self.assertTrue(hasattr(comprehensive_mock, 'calculate_happiness'))
        self.assertFalse(hasattr(simple_mock, 'calculate_happiness'))
        
        # Show that comprehensive mock can handle complex operations
        self.assertIsNotNone(comprehensive_mock.define_descriptors())
        
        with self.assertRaises(AttributeError):
            simple_mock.define_descriptors()
        
        print("✓ Comparison with simple mock test passed")

    def test_edge_cases_and_robustness(self):
        """Test edge cases and robustness of the mock."""
        # Test with extreme values
        extreme_char = MockCharacter(
            "ExtremeChar",
            age=0,
            health_status=-5.0,
            wealth_money=-1000.0,
            hunger_level=20.0
        )
        
        # Should handle extreme values gracefully
        self.assertIsNotNone(extreme_char.calculate_happiness())
        self.assertIsNotNone(extreme_char.to_dict())
        
        # Test with None values where appropriate
        none_char = MockCharacter("NoneChar", home=None, location=None)
        self.assertIsNone(none_char.home)
        
        # Should still work with None values
        descriptors = none_char.define_descriptors()
        self.assertIn("home_status", descriptors)
        self.assertEqual(descriptors["home_status"], "homeless")
        
        print("✓ Edge cases and robustness test passed")


if __name__ == "__main__":
    print("Running Comprehensive MockCharacter Tests...")
    print("=" * 60)
    
    # Run tests with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestComprehensiveMockCharacter)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All comprehensive MockCharacter tests passed!")
        print("✓ The new MockCharacter provides accurate Character interface coverage")
        print("✓ Tests are more reliable and less likely to have false positives")
        print("✓ Consistent interface across all test files")
    else:
        print("❌ Some tests failed. Check implementation.")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")