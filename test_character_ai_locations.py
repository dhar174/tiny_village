#!/usr/bin/env python3
"""
Test character decision-making with location properties using the REAL Character class.

This test demonstrates the correct approach of testing actual Character class methods
rather than creating mock implementations that could hide bugs in the real code.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock Building and Location classes for testing
class MockLocation:
    def __init__(self, security=5, popularity=5, activities=None):
        self.security = security
        self.popularity = popularity
        self.activities_available = activities or []

class MockBuilding:
    def __init__(self, name, building_type="building", security=5, popularity=5, activities=None):
        self.name = name
        self.building_type = building_type
        self._location = MockLocation(security, popularity, activities)
        
    def get_location(self):
        return self._location

class MockPersonalityTraits:
    def __init__(self, neuroticism=50, extraversion=50):
        self.neuroticism = neuroticism
        self.extraversion = extraversion


class TestCharacterLocationDecisions(unittest.TestCase):
    """Test Character location decision-making using the REAL Character class"""
    
    def setUp(self):
        """Set up test with minimal Character dependencies"""
        # We'll patch the complex dependencies to focus on testing the location evaluation logic
        # This allows us to test the REAL Character methods without requiring all dependencies
        pass
    
    def test_character_location_evaluation_with_real_character_class(self):
        """Test that the REAL Character class can evaluate locations based on personality"""
        
        # Import the real Character class
        try:
            # Mock heavy dependencies to focus on testing the location evaluation logic
            with patch.dict('sys.modules', {
                'networkx': MagicMock(),
                'numpy': MagicMock(),
                'torch': MagicMock(),
                'faiss': MagicMock(),
                'pandas': MagicMock(),
                'scipy': MagicMock(),
                'pygame': MagicMock()
            }):
                from tiny_characters import Character
                
                # Create a minimal Character with mocked dependencies
                with patch('tiny_characters.GraphManager'), \
                     patch('tiny_characters.ActionSystem'), \
                     patch('tiny_characters.GameTimeManager'):
                    
                    # Create a real Character instance
                    character = Character(
                        name="Test Character",
                        age=25,
                        energy=3,  # Low energy
                        social_wellbeing=2,  # Low social wellbeing
                        graph_manager=MagicMock(),
                        action_system=MagicMock(),
                        gametime_manager=MagicMock()
                    )
                    
                    # Add personality traits
                    character.personality_traits = MockPersonalityTraits(
                        neuroticism=80,  # High anxiety
                        extraversion=30  # Introverted
                    )
                    
                    # Test buildings with different properties
                    secure_house = MockBuilding(
                        "Secure House", 
                        building_type="house",
                        security=9, 
                        popularity=3,
                        activities=["rest", "sleep", "secure_shelter"]
                    )
                    
                    crowded_mall = MockBuilding(
                        "Crowded Mall",
                        building_type="commercial", 
                        security=3,
                        popularity=9,
                        activities=["shop", "conduct_business"]
                    )
                    
                    # Test if Character has the location evaluation method
                    if hasattr(character, 'evaluate_location_for_visit'):
                        # Test the REAL implementation
                        secure_score = character.evaluate_location_for_visit(secure_house)
                        mall_score = character.evaluate_location_for_visit(crowded_mall)
                        
                        # Anxious, introverted character should prefer secure, less popular locations
                        self.assertGreater(secure_score, mall_score,
                            "Anxious character should prefer secure house over crowded mall")
                        
                        print(f"✓ Real Character evaluation - Secure house: {secure_score}, Mall: {mall_score}")
                        
                    else:
                        # If the method doesn't exist, this test documents what should be implemented
                        self.fail("Character class is missing evaluate_location_for_visit method. "
                                 "This method should be implemented to enable location-based AI decisions.")
                    
                    # Test other location decision methods if they exist
                    if hasattr(character, 'find_suitable_locations'):
                        buildings = [secure_house, crowded_mall]
                        suitable = character.find_suitable_locations(buildings, min_score=50)
                        self.assertIsInstance(suitable, list, "find_suitable_locations should return a list")
                        print(f"✓ Real Character found {len(suitable)} suitable locations")
                    
                    if hasattr(character, 'make_location_decision'):
                        buildings = [secure_house, crowded_mall]
                        chosen, factors = character.make_location_decision(buildings)
                        self.assertIsNotNone(chosen, "Character should be able to make a location decision")
                        print(f"✓ Real Character chose: {chosen.name if chosen else 'None'}")
                        
        except ImportError as e:
            self.skipTest(f"Cannot import Character class: {e}")
        except Exception as e:
            # Character was imported but methods don't exist or failed for other reasons
            print(f"⚠️  Character class imported but location methods not available: {e}")
            self.fail(f"Character class exists but location evaluation methods are not properly implemented: {e}")
    
    def test_character_energy_affects_location_preference(self):
        """Test that low energy characters prefer rest locations"""
        
        try:
            # Mock heavy dependencies
            with patch.dict('sys.modules', {
                'networkx': MagicMock(),
                'numpy': MagicMock(), 
                'torch': MagicMock(),
                'faiss': MagicMock(),
                'pandas': MagicMock(),
                'scipy': MagicMock(),
                'pygame': MagicMock()
            }):
                from tiny_characters import Character
                
                with patch('tiny_characters.GraphManager'), \
                     patch('tiny_characters.ActionSystem'), \
                     patch('tiny_characters.GameTimeManager'):
                    
                    # Create tired character
                    tired_character = Character(
                        name="Tired Character",
                        age=25,
                        energy=1,  # Very low energy
                        graph_manager=MagicMock(),
                        action_system=MagicMock(),
                        gametime_manager=MagicMock()
                    )
                    
                    # Create energized character
                    energetic_character = Character(
                        name="Energetic Character", 
                        age=25,
                        energy=10,  # High energy
                        graph_manager=MagicMock(),
                        action_system=MagicMock(),
                        gametime_manager=MagicMock()
                    )
                    
                    # Test location with rest activities
                    rest_location = MockBuilding(
                        "Rest Area",
                        activities=["rest", "sleep", "relax"]
                    )
                    
                    # Work location without rest activities
                    work_location = MockBuilding(
                        "Office",
                        activities=["work", "meet", "conduct_business"]
                    )
                    
                    if hasattr(tired_character, 'evaluate_location_for_visit'):
                        # Tired character should prefer rest location
                        tired_rest_score = tired_character.evaluate_location_for_visit(rest_location)
                        tired_work_score = tired_character.evaluate_location_for_visit(work_location)
                        
                        # Energetic character may not have strong preference for rest
                        energetic_rest_score = energetic_character.evaluate_location_for_visit(rest_location)
                        energetic_work_score = energetic_character.evaluate_location_for_visit(work_location)
                        
                        # Tired character should show stronger preference for rest location
                        tired_preference = tired_rest_score - tired_work_score
                        energetic_preference = energetic_rest_score - energetic_work_score
                        
                        self.assertGreater(tired_preference, energetic_preference,
                            "Tired character should show stronger preference for rest locations")
                        
                        print(f"✓ Tired character rest preference: {tired_preference}")
                        print(f"✓ Energetic character rest preference: {energetic_preference}")
                    else:
                        self.fail("Character class missing evaluate_location_for_visit method")
                        
        except ImportError as e:
            self.skipTest(f"Cannot import Character class: {e}")
        except Exception as e:
            print(f"⚠️  Character class imported but location methods not available: {e}")
            self.fail(f"Character class exists but location evaluation methods are not properly implemented: {e}")


    def test_demonstrates_proper_testing_approach(self):
        """This test demonstrates the correct testing approach and documents the antipattern"""
        
        print("\n" + "="*70)
        print("CORRECT TESTING APPROACH - What TO do:")
        print("="*70)
        print("1. Import and use the REAL Character class")
        print("2. Mock only the external dependencies (GraphManager, ActionSystem, etc.)")
        print("3. Test the actual implementation methods")
        print("4. Verify real behavior, not fake behavior")
        print("5. Fail when the real implementation doesn't work")
        print("\nBenefits:")
        print("- Tests catch bugs in the real implementation")
        print("- Tests verify actual integration works")
        print("- Tests document expected behavior")
        print("- Changes to Character class will break tests appropriately")
        
        # This always passes to show the concept
        self.assertTrue(True, "This demonstrates the correct testing philosophy")


def demonstrate_testing_antipattern():
    """
    This function shows the WRONG way to test - creating a mock that reimplements 
    the logic instead of testing the real implementation.
    
    This approach is problematic because:
    1. The test could pass even if the real Character implementation is broken
    2. The mock might not accurately represent actual behavior  
    3. It doesn't validate that the real integration works correctly
    """
    
    print("\n" + "="*60)
    print("ANTIPATTERN EXAMPLE - What NOT to do:")
    print("="*60)
    
    # BAD: Creating a mock that reimplements the logic
    class TestCharacterAntipattern:
        def __init__(self, name, energy=10):
            self.name = name
            self.energy = energy
            
        def evaluate_location_for_visit(self, building):
            # BAD: This reimplements the logic instead of testing the real implementation
            # This could pass even if the real Character class is broken!
            if self.energy < 5:
                return 80  # Always return high score for tired characters
            return 50
    
    # This test might pass but doesn't test the real Character class
    fake_character = TestCharacterAntipattern("Fake", energy=2)
    fake_score = fake_character.evaluate_location_for_visit(MockBuilding("Test"))
    
    print(f"Fake character score: {fake_score}")
    print("Problem: This test passes but doesn't verify the real Character class works!")
    print("The real Character implementation could be completely broken and this test would still pass.")


def test_character_location_decisions():
    """Run the tests using unittest framework"""
    
    print("Testing Character Location Decisions with REAL Character class")
    print("="*70)
    
    # Run the proper tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Show the antipattern example
    demonstrate_testing_antipattern()


if __name__ == '__main__':
    test_character_location_decisions()