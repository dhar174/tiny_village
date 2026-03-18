#!/usr/bin/env python3
"""
Test suite for enhanced building interactions system.

This tests the implementation of building-type specific interactions
that go beyond the basic 'Enter Building' action.
"""

import unittest
from unittest.mock import Mock, MagicMock
from tiny_buildings import Building, BUILDING_TYPE_INTERACTIONS


class MockCharacter:
    """Mock character for testing interaction filtering."""
    
    def __init__(self, energy=50, extraversion=60, wealth=100, **attributes):
        self.energy = energy
        self.wealth = wealth
        self.hunger = attributes.get('hunger', 20)
        self.thirst = attributes.get('thirst', 30)
        
        # Mock personality traits
        self.personality_traits = Mock()
        self.personality_traits.extraversion = extraversion
        self.personality_traits.openness = attributes.get('openness', 50)
        self.personality_traits.conscientiousness = attributes.get('conscientiousness', 50)
        self.personality_traits.agreeableness = attributes.get('agreeableness', 50)
        
        # Mock skills
        self.skills = Mock()
        self.skills.crafting = attributes.get('crafting_skill', 20)
        self.skills.farming = attributes.get('farming_skill', 15)
        self.skills.animal_care = attributes.get('animal_care_skill', 10)


class TestBuildingInteractions(unittest.TestCase):
    """Test building interaction system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_character = MockCharacter()
    
    def test_residential_building_interactions(self):
        """Test that residential buildings have correct interactions."""
        house = Building('Test House', 0, 0, 10, 10, 10, building_type='residential')
        
        interaction_names = [action.name for action in house.possible_interactions]
        expected_interactions = ['Enter Building', 'Rest Inside', 'Visit Residents', 'Use Facilities']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_commercial_building_interactions(self):
        """Test that commercial buildings have correct interactions."""
        shop = Building('Test Shop', 0, 0, 10, 10, 10, building_type='commercial')
        
        interaction_names = [action.name for action in shop.possible_interactions]
        expected_interactions = ['Enter Building', 'Browse Goods', 'Buy Items', 'Trade with Merchants']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_social_building_interactions(self):
        """Test that social buildings have correct interactions."""
        tavern = Building('Test Tavern', 0, 0, 10, 10, 10, building_type='social')
        
        interaction_names = [action.name for action in tavern.possible_interactions]
        expected_interactions = ['Enter Building', 'Socialize with Patrons', 'Get a Drink', 'Join Activity']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_crafting_building_interactions(self):
        """Test that crafting buildings have correct interactions."""
        workshop = Building('Test Workshop', 0, 0, 10, 10, 10, building_type='crafting')
        
        interaction_names = [action.name for action in workshop.possible_interactions]
        expected_interactions = ['Enter Building', 'Commission Item', 'Learn Crafting', 'Use Equipment']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_agricultural_building_interactions(self):
        """Test that agricultural buildings have correct interactions."""
        farm = Building('Test Farm', 0, 0, 10, 10, 10, building_type='agricultural')
        
        interaction_names = [action.name for action in farm.possible_interactions]
        expected_interactions = ['Enter Building', 'Help with Crops', 'Gather Food', 'Tend Animals']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_educational_building_interactions(self):
        """Test that educational buildings have correct interactions."""
        school = Building('Test School', 0, 0, 10, 10, 10, building_type='educational')
        
        interaction_names = [action.name for action in school.possible_interactions]
        expected_interactions = ['Enter Building', 'Attend Class', 'Study Books', 'Access Resources']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_civic_building_interactions(self):
        """Test that civic buildings have correct interactions."""
        city_hall = Building('City Hall', 0, 0, 10, 10, 10, building_type='civic')
        
        interaction_names = [action.name for action in city_hall.possible_interactions]
        expected_interactions = ['Enter Building', 'Attend Meeting', 'Get Information', 'File Complaint']
        
        self.assertEqual(len(interaction_names), 4)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
    
    def test_building_type_aliases(self):
        """Test that building type aliases work correctly."""
        # Test house alias for residential
        house = Building('Test House', 0, 0, 10, 10, 10, building_type='house')
        house_interactions = [action.name for action in house.possible_interactions]
        
        residential = Building('Test Residential', 0, 0, 10, 10, 10, building_type='residential')
        residential_interactions = [action.name for action in residential.possible_interactions]
        
        self.assertEqual(house_interactions, residential_interactions)
        
        # Test shop alias for commercial
        shop = Building('Test Shop', 0, 0, 10, 10, 10, building_type='shop')
        shop_interactions = [action.name for action in shop.possible_interactions]
        
        commercial = Building('Test Commercial', 0, 0, 10, 10, 10, building_type='commercial')
        commercial_interactions = [action.name for action in commercial.possible_interactions]
        
        self.assertEqual(shop_interactions, commercial_interactions)
    
    def test_library_special_case(self):
        """Test that libraries have specialized interactions."""
        library = Building('Test Library', 0, 0, 10, 10, 10, building_type='library')
        
        interaction_names = [action.name for action in library.possible_interactions]
        expected_interactions = ['Enter Building', 'Study Books', 'Access Resources']
        
        self.assertEqual(len(interaction_names), 3)
        for expected in expected_interactions:
            self.assertIn(expected, interaction_names)
        
        # Libraries should not have 'Attend Class' unlike regular educational buildings
        self.assertNotIn('Attend Class', interaction_names)
    
    def test_unknown_building_type_fallback(self):
        """Test that unknown building types fall back to basic interactions."""
        unknown_building = Building('Unknown Building', 0, 0, 10, 10, 10, building_type='unknown_type')
        
        interaction_names = [action.name for action in unknown_building.possible_interactions]
        
        # Should fall back to just "Enter Building"
        self.assertEqual(len(interaction_names), 1)
        self.assertIn('Enter Building', interaction_names)
    
    def test_building_type_mapping_completeness(self):
        """Test that all defined building types have interactions."""
        for building_type, interactions in BUILDING_TYPE_INTERACTIONS.items():
            self.assertIsInstance(interactions, list)
            self.assertGreater(len(interactions), 0)
            self.assertIn('Enter Building', interactions)  # All should have basic entry
    
    def test_get_possible_interactions_returns_actions(self):
        """Test that get_possible_interactions returns Action objects."""
        building = Building('Test Building', 0, 0, 10, 10, 10, building_type='commercial')
        
        # Mock a character with sufficient energy
        mock_char = MockCharacter(energy=100, extraversion=80, wealth=200)
        
        interactions = building.get_possible_interactions(mock_char)
        
        # Should return Action objects, not strings
        self.assertGreater(len(interactions), 0)
        for action in interactions:
            self.assertTrue(hasattr(action, 'name'))
            self.assertTrue(hasattr(action, 'preconditions'))
    
    def test_character_energy_filtering(self):
        """Test basic character energy filtering."""
        building = Building('Test Building', 0, 0, 10, 10, 10, building_type='commercial')
        
        # Character with sufficient energy should get interactions
        high_energy_char = MockCharacter(energy=50)
        interactions_high = building.get_possible_interactions(high_energy_char)
        self.assertGreater(len(interactions_high), 0)
        
        # Character with very low energy should get fewer/no interactions
        low_energy_char = MockCharacter(energy=2)
        interactions_low = building.get_possible_interactions(low_energy_char)
        
        # Low energy character should have fewer available interactions
        self.assertLessEqual(len(interactions_low), len(interactions_high))
    
    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        # Any building should still support the basic "Enter Building" action
        building_types = ['residential', 'commercial', 'social', 'crafting', 
                         'agricultural', 'educational', 'civic', 'house', 'shop']
        
        for building_type in building_types:
            building = Building(f'Test {building_type}', 0, 0, 10, 10, 10, building_type=building_type)
            interaction_names = [action.name for action in building.possible_interactions]
            
            self.assertIn('Enter Building', interaction_names, 
                         f"Building type '{building_type}' should have 'Enter Building' interaction")
    
    def test_each_building_type_has_multiple_interactions(self):
        """Test that each building type has more than just 'Enter Building'."""
        core_building_types = ['residential', 'commercial', 'social', 'crafting', 
                              'agricultural', 'educational', 'civic']
        
        for building_type in core_building_types:
            building = Building(f'Test {building_type}', 0, 0, 10, 10, 10, building_type=building_type)
            interaction_names = [action.name for action in building.possible_interactions]
            
            self.assertGreater(len(interaction_names), 1, 
                             f"Building type '{building_type}' should have more than just 'Enter Building'")
            self.assertGreaterEqual(len(interaction_names), 3, 
                                   f"Building type '{building_type}' should have at least 3 interactions")


if __name__ == '__main__':
    unittest.main()