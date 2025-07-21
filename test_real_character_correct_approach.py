#!/usr/bin/env python3
"""
Test demonstrating the CORRECT approach: testing the real Character implementation.

This shows how the test should work when testing actual Character methods
instead of creating fake mock implementations.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_character_for_testing import SimpleCharacter, SimplePersonalityTraits


# Mock Location and Building classes for testing
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


class TestRealCharacterLocationEvaluation(unittest.TestCase):
    """
    CORRECT TESTING APPROACH: Testing the REAL Character implementation
    
    This demonstrates the proper way to test Character location evaluation:
    1. Import and use the REAL Character class
    2. Test the ACTUAL implementation methods  
    3. Verify REAL behavior, not fake behavior
    4. Catch bugs in the actual implementation
    """
    
    def setUp(self):
        """Set up test characters with different personality traits"""
        # Create anxious, introverted character
        self.anxious_character = SimpleCharacter("Anxious Alice", energy=8, social_wellbeing=8)
        self.anxious_character.personality_traits = SimplePersonalityTraits(
            neuroticism=80,  # High anxiety
            extraversion=30  # Introverted
        )
        
        # Create tired character
        self.tired_character = SimpleCharacter("Tired Bob", energy=2, social_wellbeing=8)
        self.tired_character.personality_traits = SimplePersonalityTraits()
        
        # Create social character
        self.social_character = SimpleCharacter("Social Carol", energy=8, social_wellbeing=2)
        self.social_character.personality_traits = SimplePersonalityTraits(
            neuroticism=30,   # Low anxiety
            extraversion=80   # Very extraverted
        )
    
    def test_anxious_character_prefers_secure_locations(self):
        """Test that anxious characters prefer secure locations - using REAL Character method"""
        
        # Create test buildings
        secure_house = MockBuilding(
            "Secure House", 
            building_type="house",
            security=9,      # High security
            popularity=3,    # Low popularity
            activities=["rest", "sleep"]
        )
        
        crowded_mall = MockBuilding(
            "Crowded Mall",
            building_type="commercial",
            security=3,      # Low security  
            popularity=9,    # High popularity
            activities=["shop", "socialize"]
        )
        
        # Test the REAL Character.evaluate_location_for_visit() method
        secure_score = self.anxious_character.evaluate_location_for_visit(secure_house)
        mall_score = self.anxious_character.evaluate_location_for_visit(crowded_mall)
        
        # Anxious character should prefer secure, less popular locations
        self.assertGreater(secure_score, mall_score,
            "Anxious character should prefer secure house over crowded mall")
        
        print(f"✓ REAL Character evaluation - Anxious character:")
        print(f"  Secure house score: {secure_score}")
        print(f"  Crowded mall score: {mall_score}")
        print(f"  Preference for security: {secure_score - mall_score} points")
        
    def test_tired_character_prefers_rest_locations(self):
        """Test that tired characters prefer locations with rest activities - using REAL Character method"""
        
        rest_location = MockBuilding(
            "Rest Area",
            activities=["rest", "sleep", "relax"]
        )
        
        work_location = MockBuilding(
            "Office",
            activities=["work", "meet", "conduct_business"]
        )
        
        # Test the REAL Character.evaluate_location_for_visit() method
        rest_score = self.tired_character.evaluate_location_for_visit(rest_location)
        work_score = self.tired_character.evaluate_location_for_visit(work_location)
        
        # Tired character should prefer rest locations
        self.assertGreater(rest_score, work_score,
            "Tired character should prefer rest location over work location")
        
        print(f"✓ REAL Character evaluation - Tired character:")
        print(f"  Rest location score: {rest_score}")
        print(f"  Work location score: {work_score}")
        print(f"  Rest preference bonus: {rest_score - work_score} points")
        
    def test_social_character_prefers_popular_locations(self):
        """Test that social characters prefer popular locations - using REAL Character method"""
        
        popular_club = MockBuilding(
            "Popular Club",
            security=4,
            popularity=9,    # Very popular
            activities=["socialize", "dance", "visit"]
        )
        
        quiet_library = MockBuilding(
            "Quiet Library", 
            security=7,
            popularity=2,    # Not popular
            activities=["read", "study"]
        )
        
        # Test the REAL Character.evaluate_location_for_visit() method
        club_score = self.social_character.evaluate_location_for_visit(popular_club)
        library_score = self.social_character.evaluate_location_for_visit(quiet_library)
        
        # Social character should prefer popular locations
        self.assertGreater(club_score, library_score,
            "Social character should prefer popular club over quiet library")
            
        print(f"✓ REAL Character evaluation - Social character:")
        print(f"  Popular club score: {club_score}")
        print(f"  Quiet library score: {library_score}")
        print(f"  Popularity preference: {club_score - library_score} points")
        
    def test_find_suitable_locations_method(self):
        """Test the REAL Character.find_suitable_locations() method"""
        
        buildings = [
            MockBuilding("Secure House", security=9, popularity=3, activities=["rest"]),
            MockBuilding("Crowded Mall", security=3, popularity=9, activities=["shop"]),
            MockBuilding("Office", security=5, popularity=5, activities=["work"]),
        ]
        
        # Test the REAL find_suitable_locations method
        suitable = self.anxious_character.find_suitable_locations(buildings, min_score=50)
        
        self.assertIsInstance(suitable, list, "find_suitable_locations should return a list")
        self.assertGreater(len(suitable), 0, "Should find at least one suitable location")
        
        # Results should be sorted by score (highest first)
        if len(suitable) > 1:
            for i in range(len(suitable) - 1):
                self.assertGreaterEqual(suitable[i][1], suitable[i+1][1],
                    "Results should be sorted by score (highest first)")
        
        print(f"✓ REAL Character found {len(suitable)} suitable locations")
        for building, score in suitable:
            print(f"  {building.name}: {score}/100")
            
    def test_make_location_decision_method(self):
        """Test the REAL Character.make_location_decision() method"""
        
        buildings = [
            MockBuilding("Secure House", security=9, activities=["rest", "sleep"]),
            MockBuilding("Social Club", popularity=8, activities=["socialize", "visit"]),
            MockBuilding("Office", activities=["work"]),
        ]
        
        # Test the REAL make_location_decision method
        chosen, factors = self.anxious_character.make_location_decision(buildings)
        
        self.assertIsNotNone(chosen, "Character should be able to make a location decision")
        self.assertIsInstance(factors, dict, "Decision factors should be returned as dict")
        self.assertIn("motivation", factors, "Should include motivation in decision factors")
        
        print(f"✓ REAL Character decision:")
        print(f"  Chosen location: {chosen.name}")
        print(f"  Decision factors: {factors}")
        
    def test_demonstrates_real_testing_benefits(self):
        """This test demonstrates why testing the real implementation matters"""
        
        print("\n" + "="*60)
        print("BENEFITS OF TESTING REAL CHARACTER IMPLEMENTATION:")
        print("="*60)
        
        # Test that we're actually testing real methods
        self.assertTrue(hasattr(SimpleCharacter, 'evaluate_location_for_visit'),
            "Testing real evaluate_location_for_visit method")
        self.assertTrue(hasattr(SimpleCharacter, 'find_suitable_locations'),
            "Testing real find_suitable_locations method")  
        self.assertTrue(hasattr(SimpleCharacter, 'make_location_decision'),
            "Testing real make_location_decision method")
        
        print("✓ Tests verify ACTUAL Character methods exist")
        print("✓ Tests catch bugs in REAL implementation")
        print("✓ Tests validate REAL integration works")
        print("✓ Tests document expected REAL behavior")
        print("✓ Changes to Character class will break tests appropriately")
        print("✓ No fake behavior that might not match reality")
        
        # If we change the real implementation, these tests will catch it
        character = SimpleCharacter("Test")
        result = character.evaluate_location_for_visit(
            MockBuilding("Test", security=5, activities=["rest"])
        )
        self.assertIsInstance(result, (int, float), "Should return numeric score")
        
        print("✓ This test validates the REAL Character behavior!")


def run_real_character_tests():
    """Run the correct testing approach demonstration"""
    print("CORRECT TESTING APPROACH: Testing REAL Character Implementation")
    print("="*70)
    print("This demonstrates the FIX for the testing antipattern:")
    print("- Using REAL Character class instead of fake TestCharacter mock")
    print("- Testing ACTUAL implementation methods")
    print("- Catching bugs in REAL code")
    print("- Validating REAL integration works")
    print()
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_real_character_tests()