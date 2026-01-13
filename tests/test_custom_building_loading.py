#!/usr/bin/env python3
"""
Integration tests for custom building loading system.

Tests the robustness of loading buildings from custom_buildings.json
and validates all defined building properties and interactions.
"""

import unittest
import json
import os
import tempfile
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_gameplay_controller import GameplayController
from tiny_buildings import Building, BUILDING_TYPE_INTERACTIONS
from actions import ActionSystem


class TestCustomBuildingLoading(unittest.TestCase):
    """Test loading buildings from custom_buildings.json with validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_building_file(self):
        """Test loading a valid building configuration file."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Test Town Hall",
                    "type": "civic",
                    "x": 100,
                    "y": 150,
                    "width": 60,
                    "height": 55,
                    "length": 55,
                    "stories": 2,
                    "num_rooms": 8,
                    "address": "1 Main Square",
                    "description": "Test building"
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "test_buildings.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        # Initialize pygame properly for testing
        import pygame
        pygame.init()
        with patch("pygame.display.set_mode") as mock_display:
            mock_display.return_value = Mock()
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        pygame.quit()
        
        self.assertEqual(len(buildings_data), 1)
        building = buildings_data[0]
        
        # Validate all properties were loaded correctly
        self.assertEqual(building["name"], "Test Town Hall")
        self.assertEqual(building["type"], "civic")
        self.assertEqual(building["length"], 55)
        self.assertEqual(building["stories"], 2)
        self.assertEqual(building["num_rooms"], 8)
        self.assertEqual(building["address"], "1 Main Square")
        self.assertEqual(building["description"], "Test building")
    
    def test_load_minimal_building(self):
        """Test loading a building with only required fields."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Minimal Building",
                    "x": 50,
                    "y": 50
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "minimal_buildings.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(len(buildings_data), 1)
        building = buildings_data[0]
        
        # Check defaults were applied
        self.assertEqual(building["name"], "Minimal Building")
        self.assertEqual(building["type"], "building")
        self.assertIsNotNone(building["rect"])
    
    def test_load_all_building_types(self):
        """Test loading buildings of all supported types."""
        building_types = [
            "residential", "house", "commercial", "shop", "social", "tavern",
            "crafting", "workshop", "agricultural", "farm", "educational",
            "school", "library", "civic", "office"
        ]
        
        test_buildings = {
            "buildings": [
                {
                    "name": f"Test {btype.capitalize()}",
                    "type": btype,
                    "x": idx * 50,
                    "y": 100,
                    "width": 40,
                    "height": 40
                }
                for idx, btype in enumerate(building_types)
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "all_types.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(len(buildings_data), len(building_types))
        
        # Verify each building type was loaded correctly
        for idx, building in enumerate(buildings_data):
            expected_type = building_types[idx]
            self.assertEqual(building["type"], expected_type)
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        buildings_file = os.path.join(self.temp_dir, "invalid.json")
        with open(buildings_file, "w") as f:
            f.write("{ invalid json }")
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Should return empty list, not crash
        self.assertEqual(buildings_data, [])
    
    def test_missing_required_fields(self):
        """Test handling of buildings with missing required fields."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Valid Building",
                    "x": 100,
                    "y": 100
                },
                {
                    "x": 200,
                    "y": 200
                    # Missing name
                },
                {
                    "name": "Missing Coords"
                    # Missing x, y
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "partial_buildings.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Only the valid building should be loaded
        self.assertEqual(len(buildings_data), 1)
        self.assertEqual(buildings_data[0]["name"], "Valid Building")
    
    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values with fallback to defaults."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Invalid Numbers Building",
                    "x": "not a number",
                    "y": 100,
                    "width": "invalid",
                    "height": 50,
                    "stories": "two"
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "invalid_numbers.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Should load with default values
        self.assertEqual(len(buildings_data), 1)
        building = buildings_data[0]
        self.assertEqual(building["name"], "Invalid Numbers Building")
        # Coordinates should default to 0, 40, 40
        self.assertIsNotNone(building["rect"])
    
    def test_custom_properties_preserved(self):
        """Test that custom properties not in the standard set are preserved."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Custom Props Building",
                    "x": 100,
                    "y": 100,
                    "custom_field": "custom_value",
                    "special_marker": 42,
                    "tags": ["important", "landmark"]
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "custom_props.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(len(buildings_data), 1)
        building = buildings_data[0]
        
        # Custom properties should be preserved
        self.assertEqual(building["custom_field"], "custom_value")
        self.assertEqual(building["special_marker"], 42)
        self.assertEqual(building["tags"], ["important", "landmark"])
    
    def test_unrecognized_building_type(self):
        """Test handling of unrecognized building types."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Unknown Type Building",
                    "type": "mysterious",
                    "x": 100,
                    "y": 100
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "unknown_type.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Should load, but with a warning logged
        self.assertEqual(len(buildings_data), 1)
        self.assertEqual(buildings_data[0]["type"], "mysterious")
    
    def test_file_not_found(self):
        """Test handling of missing file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.json")
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(nonexistent_file)
        
        # Should return empty list gracefully
        self.assertEqual(buildings_data, [])
    
    def test_empty_buildings_array(self):
        """Test handling of empty buildings array."""
        test_buildings = {"buildings": []}
        
        buildings_file = os.path.join(self.temp_dir, "empty.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(buildings_data, [])
    
    def test_missing_buildings_key(self):
        """Test handling of JSON without 'buildings' key."""
        test_data = {"some_other_key": []}
        
        buildings_file = os.path.join(self.temp_dir, "no_buildings_key.json")
        with open(buildings_file, "w") as f:
            json.dump(test_data, f)
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(buildings_data, [])


class TestBuildingInteractionIntegration(unittest.TestCase):
    """Test that loaded buildings have correct interactions."""
    
    def test_building_interactions_from_type(self):
        """Test that buildings loaded from JSON get correct interactions based on type."""
        from actions import ActionSystem
        
        # Test each building type gets correct interactions
        test_cases = [
            ("civic", ["Enter Building", "Attend Meeting", "Get Information", "File Complaint"]),
            ("commercial", ["Enter Building", "Browse Goods", "Buy Items", "Trade with Merchants"]),
            ("tavern", ["Enter Building", "Socialize with Patrons", "Get a Drink", "Join Activity"]),
            ("workshop", ["Enter Building", "Commission Item", "Learn Crafting", "Use Equipment"]),
        ]
        
        for building_type, expected_interactions in test_cases:
            with self.subTest(building_type=building_type):
                building = Building(
                    name=f"Test {building_type}",
                    x=0,
                    y=0,
                    height=40,
                    width=40,
                    length=40,
                    building_type=building_type,
                    action_system=ActionSystem()
                )
                
                interaction_names = [action.name for action in building.possible_interactions]
                for expected in expected_interactions:
                    self.assertIn(expected, interaction_names)


class TestRealCustomBuildingsFile(unittest.TestCase):
    """Test loading the actual custom_buildings.json file."""
    
    def test_load_actual_custom_buildings(self):
        """Test loading the real custom_buildings.json from the repository."""
        buildings_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "custom_buildings.json"
        )
        
        # Verify file exists
        self.assertTrue(os.path.exists(buildings_file), "custom_buildings.json not found")
        
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})
            buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Should load at least some buildings
        self.assertGreater(len(buildings_data), 0, "No buildings loaded from custom_buildings.json")
        
        # Verify each building has required properties
        for building in buildings_data:
            self.assertIn("name", building)
            self.assertIn("type", building)
            self.assertIn("rect", building)
            self.assertIsNotNone(building["rect"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
