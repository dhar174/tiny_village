#!/usr/bin/env python3
"""
Test error handling and edge cases in GOAPPlanner.get_current_world_state.
"""

import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import os

# Ensure we can import from the root directory
sys.path.insert(0, os.getcwd())

from tiny_goap_system import GOAPPlanner
from actions import State

class TestGOAPErrorHandling(unittest.TestCase):
    def setUp(self):
        self.planner = GOAPPlanner(graph_manager=None)

    def test_get_current_world_state_outer_exception(self):
        """Test Scenario 1: Outer Exception Handling in get_current_world_state"""
        # Mock character whose get_state method raises an exception
        mock_character = MagicMock()
        mock_character.get_state.side_effect = Exception("Character state error")
        mock_character.name = "TestChar"

        # Call the method under test
        with self.assertLogs(level='WARNING') as cm:
            result_state = self.planner.get_current_world_state(mock_character)

        # Verify result is a fallback State
        self.assertIsInstance(result_state, State)
        self.assertEqual(result_state.dict_or_obj, {'character': mock_character})
        # Verify logging
        self.assertTrue(any("Error getting current world state: Character state error" in output for output in cm.output))

    def test_get_current_world_state_inner_exception(self):
        """Test Scenario 2: Inner Exception Handling in get_current_world_state"""
        # Mock character
        mock_character = MagicMock()
        char_state_dict = {"energy": 50}
        mock_character.get_state.return_value = State(char_state_dict)
        mock_character.name = "TestChar"

        # Mock graph manager that raises an exception
        mock_graph_manager = MagicMock()
        mock_graph_manager.get_character_state.side_effect = Exception("Graph manager error")

        self.planner.graph_manager = mock_graph_manager

        # Call the method under test
        with self.assertLogs(level='WARNING') as cm:
            result_state = self.planner.get_current_world_state(mock_character)

        # Verify result is the character's original state
        self.assertIsInstance(result_state, State)
        self.assertEqual(result_state.dict_or_obj, char_state_dict)
        # Verify logging
        self.assertTrue(any("Could not get world context from graph manager: Graph manager error" in output for output in cm.output))

    def test_get_current_world_state_success_merge(self):
        """Test Scenario 3: Normal State Enrichment in get_current_world_state"""
        # Mock character
        mock_character = MagicMock()
        char_state_dict = {"energy": 50}
        mock_character.get_state.return_value = State(char_state_dict)
        mock_character.name = "TestChar"

        # Mock graph manager
        mock_graph_manager = MagicMock()
        world_context = {"weather": "sunny", "location": "village"}
        mock_graph_manager.get_character_state.return_value = world_context

        self.planner.graph_manager = mock_graph_manager

        # Call the method under test
        result_state = self.planner.get_current_world_state(mock_character)

        # Verify result is a merged state
        self.assertIsInstance(result_state, State)
        expected_combined = char_state_dict.copy()
        expected_combined.update(world_context)
        self.assertEqual(result_state.dict_or_obj, expected_combined)

    def test_get_current_world_state_no_get_state_attr(self):
        """Test when character doesn't have get_state method"""
        mock_character = MagicMock(spec=['name']) # No get_state
        mock_character.name = "TestChar"

        result_state = self.planner.get_current_world_state(mock_character)

        self.assertIsInstance(result_state, State)
        # Should return State({}) if it doesn't have get_state and no graph_manager
        self.assertEqual(result_state.dict_or_obj, {})

    def test_get_current_world_state_non_dict_state(self):
        """Test when character state is not a dict"""
        mock_character = MagicMock()
        char_obj = "some_object"
        mock_character.get_state.return_value = State(char_obj)
        mock_character.name = "TestChar"

        # Mock graph manager
        mock_graph_manager = MagicMock()
        world_context = {"weather": "sunny"}
        mock_graph_manager.get_character_state.return_value = world_context

        self.planner.graph_manager = mock_graph_manager

        # Call the method under test
        result_state = self.planner.get_current_world_state(mock_character)

        # Verify result is a merged state
        self.assertIsInstance(result_state, State)
        # It should put the object into 'character' key
        self.assertEqual(result_state.dict_or_obj['character'], char_obj)
        self.assertEqual(result_state.dict_or_obj['weather'], "sunny")

if __name__ == "__main__":
    unittest.main()
