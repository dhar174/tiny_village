#!/usr/bin/env python3
"""
Direct unit tests for building loading functionality.

Tests _load_buildings_from_file without requiring full GameplayController initialization.
"""

import unittest
import json
import os
import tempfile
import sys
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildingLoadingDirect(unittest.TestCase):
    """Direct tests for building loading without full controller initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock pygame for imports
        sys.modules['pygame'] = MagicMock()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_controller(self):
        """Create a minimal mock controller with just the loading method."""
        # Import after pygame is mocked
        from tiny_gameplay_controller import GameplayController
        
        controller = Mock(spec=GameplayController)
        # Bind the actual method to the mock
        controller._load_buildings_from_file = GameplayController._load_buildings_from_file.__get__(controller)
        return controller
    
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
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(buildings_file)
        
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
        self.assertIsNotNone(building["rect"])
    
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
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
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(buildings_file)
        
        # Should load, but with a warning logged
        self.assertEqual(len(buildings_data), 1)
        self.assertEqual(buildings_data[0]["type"], "mysterious")
    
    def test_file_not_found(self):
        """Test handling of missing file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.json")
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(nonexistent_file)
        
        # Should return empty list gracefully
        self.assertEqual(buildings_data, [])
    
    def test_empty_buildings_array(self):
        """Test handling of empty buildings array."""
        test_buildings = {"buildings": []}
        
        buildings_file = os.path.join(self.temp_dir, "empty.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(buildings_data, [])
    
    def test_missing_buildings_key(self):
        """Test handling of JSON without 'buildings' key."""
        test_data = {"some_other_key": []}
        
        buildings_file = os.path.join(self.temp_dir, "no_buildings_key.json")
        with open(buildings_file, "w") as f:
            json.dump(test_data, f)
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(buildings_data, [])
    
    def test_building_with_all_optional_fields(self):
        """Test loading building with all optional fields specified."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Complete Building",
                    "type": "commercial",
                    "x": 100,
                    "y": 100,
                    "width": 50,
                    "height": 45,
                    "length": 45,
                    "stories": 3,
                    "num_rooms": 10,
                    "address": "123 Complete St",
                    "owner": "John Doe",
                    "description": "A fully specified building",
                    "door": {"x": 110, "y": 100}
                }
            ]
        }
        
        buildings_file = os.path.join(self.temp_dir, "complete.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)
        
        controller = self._create_mock_controller()
        buildings_data = controller._load_buildings_from_file(buildings_file)
        
        self.assertEqual(len(buildings_data), 1)
        building = buildings_data[0]
        
        # Verify all fields are present
        self.assertEqual(building["name"], "Complete Building")
        self.assertEqual(building["type"], "commercial")
        self.assertEqual(building["length"], 45)
        self.assertEqual(building["stories"], 3)
        self.assertEqual(building["num_rooms"], 10)
        self.assertEqual(building["address"], "123 Complete St")
        self.assertEqual(building["owner"], "John Doe")
        self.assertEqual(building["description"], "A fully specified building")
        self.assertEqual(building["door"], {"x": 110, "y": 100})


if __name__ == "__main__":
    unittest.main(verbosity=2)
