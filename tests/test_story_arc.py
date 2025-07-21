#!/usr/bin/env python3
"""
Unit tests for the StoryArc implementation.
Tests the story arc system for narrative progression tracking.
"""

import unittest
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_story_arc import StoryArc, StoryArcManager


class TestStoryArc(unittest.TestCase):
    """Test cases for the StoryArc class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.arc = StoryArc("Test Arc", importance=5)
    
    def test_constants_defined(self):
        """Test that required constants are properly defined."""
        # Test constants exist
        self.assertTrue(hasattr(StoryArc, 'STARTING_THRESHOLD'))
        self.assertTrue(hasattr(StoryArc, 'DEVELOPING_THRESHOLD'))
        self.assertTrue(hasattr(StoryArc, 'CLIMAX_THRESHOLD'))
        
        # Test constants have correct values
        self.assertEqual(StoryArc.STARTING_THRESHOLD, 0.2)
        self.assertEqual(StoryArc.DEVELOPING_THRESHOLD, 0.6)
        self.assertEqual(StoryArc.CLIMAX_THRESHOLD, 0.9)
        
        # Test constants are in logical order
        self.assertLess(StoryArc.STARTING_THRESHOLD, StoryArc.DEVELOPING_THRESHOLD)
        self.assertLess(StoryArc.DEVELOPING_THRESHOLD, StoryArc.CLIMAX_THRESHOLD)
        self.assertLess(StoryArc.CLIMAX_THRESHOLD, 1.0)
    
    def test_initial_state(self):
        """Test initial state of a new story arc."""
        self.assertEqual(self.arc.name, "Test Arc")
        self.assertEqual(self.arc.importance, 5)
        self.assertEqual(self.arc.progression, 0.0)
        self.assertEqual(self.arc.phase, "setup")
        self.assertFalse(self.arc.completed)
        self.assertIsNone(self.arc.resolution_type)
        self.assertEqual(len(self.arc.events), 0)
        self.assertEqual(len(self.arc.characters_involved), 0)
    
    def test_phase_progression(self):
        """Test progression through different story phases."""
        # Test setup phase
        self.arc.progression = 0.1
        self.assertEqual(self.arc.get_current_phase(), "setup")
        
        # Test rising action phase
        self.arc.progression = 0.4
        self.assertEqual(self.arc.get_current_phase(), "rising_action")
        
        # Test climax phase
        self.arc.progression = 0.8
        self.assertEqual(self.arc.get_current_phase(), "climax")
        
        # Test resolution phase
        self.arc.progression = 0.95
        self.assertEqual(self.arc.get_current_phase(), "resolution")
    
    def test_advance_progression(self):
        """Test advancing story progression."""
        initial_progression = self.arc.progression
        self.arc.advance_progression(0.1)
        self.assertEqual(self.arc.progression, initial_progression + 0.1)
        
        # Test progression capping at 1.0
        self.arc.progression = 0.95
        self.arc.advance_progression(0.2)
        self.assertEqual(self.arc.progression, 1.0)
        self.assertTrue(self.arc.completed)
    
    def test_phase_detection_with_constants(self):
        """Test that phase detection correctly uses the defined constants."""
        # Test boundary conditions
        self.arc.progression = StoryArc.STARTING_THRESHOLD - 0.01
        self.assertEqual(self.arc.get_current_phase(), "setup")
        
        self.arc.progression = StoryArc.STARTING_THRESHOLD
        self.assertEqual(self.arc.get_current_phase(), "rising_action")
        
        self.arc.progression = StoryArc.DEVELOPING_THRESHOLD
        self.assertEqual(self.arc.get_current_phase(), "climax")
        
        self.arc.progression = StoryArc.CLIMAX_THRESHOLD
        self.assertEqual(self.arc.get_current_phase(), "resolution")
    
    def test_add_character(self):
        """Test adding characters to story arc."""
        # Test adding character with name attribute
        class MockCharacter:
            def __init__(self, name):
                self.name = name
        
        char = MockCharacter("Alice")
        self.arc.add_character(char)
        self.assertIn("Alice", self.arc.characters_involved)
        
        # Test adding character as string
        self.arc.add_character("Bob")
        self.assertIn("Bob", self.arc.characters_involved)
        
        self.assertEqual(len(self.arc.characters_involved), 2)
    
    def test_add_event(self):
        """Test adding events to story arc."""
        class MockEvent:
            def __init__(self, importance):
                self.importance = importance
        
        event = MockEvent(8)
        initial_progression = self.arc.progression
        
        self.arc.add_event(event)
        
        self.assertEqual(len(self.arc.events), 1)
        self.assertGreater(self.arc.progression, initial_progression)
    
    def test_get_progression_percentage(self):
        """Test progression percentage calculation."""
        self.arc.progression = 0.75
        self.assertEqual(self.arc.get_progression_percentage(), 75)
        
        self.arc.progression = 0.0
        self.assertEqual(self.arc.get_progression_percentage(), 0)
        
        self.arc.progression = 1.0
        self.assertEqual(self.arc.get_progression_percentage(), 100)
    
    def test_is_in_phase(self):
        """Test phase checking method."""
        self.arc.progression = 0.1
        self.assertTrue(self.arc.is_in_phase("setup"))
        self.assertFalse(self.arc.is_in_phase("climax"))
        
        self.arc.progression = 0.8
        self.assertTrue(self.arc.is_in_phase("climax"))
        self.assertFalse(self.arc.is_in_phase("setup"))
    
    def test_set_resolution(self):
        """Test setting resolution type."""
        self.arc.set_resolution("success")
        self.assertEqual(self.arc.resolution_type, "success")
        self.assertTrue(self.arc.completed)
        self.assertEqual(self.arc.progression, 1.0)
        self.assertEqual(self.arc.phase, "resolution")
    
    def test_get_status(self):
        """Test status reporting."""
        status = self.arc.get_status()
        
        required_keys = [
            "name", "importance", "progression", "progression_percentage",
            "phase", "phase_progress", "completed", "events_count",
            "characters_involved", "start_time", "duration_minutes", "resolution_type"
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status["name"], "Test Arc")
        self.assertEqual(status["importance"], 5)
        self.assertEqual(status["progression"], 0.0)
        self.assertEqual(status["progression_percentage"], 0)
        self.assertEqual(status["phase"], "setup")
        self.assertFalse(status["completed"])


class TestStoryArcManager(unittest.TestCase):
    """Test cases for the StoryArcManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = StoryArcManager()
    
    def test_initial_state(self):
        """Test initial manager state."""
        self.assertEqual(len(self.manager.active_arcs), 0)
        self.assertEqual(len(self.manager.completed_arcs), 0)
        self.assertEqual(self.manager.arc_counter, 0)
    
    def test_create_arc(self):
        """Test creating story arcs."""
        arc = self.manager.create_arc("Test Arc", importance=7)
        
        self.assertEqual(arc.name, "Test Arc")
        self.assertEqual(arc.importance, 7)
        self.assertEqual(len(self.manager.active_arcs), 1)
        self.assertEqual(self.manager.arc_counter, 1)
    
    def test_get_arc(self):
        """Test retrieving story arcs by name."""
        arc = self.manager.create_arc("Findable Arc")
        
        found_arc = self.manager.get_arc("Findable Arc")
        self.assertEqual(found_arc, arc)
        
        not_found = self.manager.get_arc("Nonexistent Arc")
        self.assertIsNone(not_found)
    
    def test_update_arcs(self):
        """Test updating and moving completed arcs."""
        arc1 = self.manager.create_arc("Active Arc")
        arc2 = self.manager.create_arc("Completed Arc")
        
        # Complete one arc
        arc2.set_resolution("success")
        
        self.manager.update_arcs()
        
        self.assertEqual(len(self.manager.active_arcs), 1)
        self.assertEqual(len(self.manager.completed_arcs), 1)
        self.assertIn(arc1, self.manager.active_arcs)
        self.assertIn(arc2, self.manager.completed_arcs)
    
    def test_get_arcs_in_phase(self):
        """Test filtering arcs by phase."""
        arc1 = self.manager.create_arc("Setup Arc")
        arc2 = self.manager.create_arc("Rising Arc")
        
        arc2.progression = 0.4  # Put in rising_action phase
        
        setup_arcs = self.manager.get_arcs_in_phase("setup")
        rising_arcs = self.manager.get_arcs_in_phase("rising_action")
        
        self.assertEqual(len(setup_arcs), 1)
        self.assertEqual(len(rising_arcs), 1)
        self.assertIn(arc1, setup_arcs)
        self.assertIn(arc2, rising_arcs)
    
    def test_get_statistics(self):
        """Test statistics generation."""
        arc1 = self.manager.create_arc("Arc 1")
        arc2 = self.manager.create_arc("Arc 2")
        arc3 = self.manager.create_arc("Arc 3")
        
        # Progress one arc
        arc2.progression = 0.4
        
        # Complete one arc
        arc3.set_resolution("success")
        self.manager.update_arcs()
        
        stats = self.manager.get_statistics()
        
        self.assertEqual(stats["total_arcs"], 3)
        self.assertEqual(stats["active_arcs"], 2)
        self.assertEqual(stats["completed_arcs"], 1)
        self.assertIn("phase_distribution", stats)
        self.assertIn("average_progression", stats)


class TestAttributeErrorResolution(unittest.TestCase):
    """Test that the original AttributeError issue is resolved."""
    
    def test_constants_accessible_from_class(self):
        """Test accessing constants from the class itself."""
        # These should not raise AttributeError
        starting = StoryArc.STARTING_THRESHOLD
        developing = StoryArc.DEVELOPING_THRESHOLD
        climax = StoryArc.CLIMAX_THRESHOLD
        
        self.assertIsInstance(starting, (int, float))
        self.assertIsInstance(developing, (int, float))
        self.assertIsInstance(climax, (int, float))
    
    def test_constants_accessible_from_instance(self):
        """Test accessing constants from class instances."""
        arc = StoryArc("Test")
        
        # These should not raise AttributeError
        starting = arc.STARTING_THRESHOLD
        developing = arc.DEVELOPING_THRESHOLD
        climax = arc.CLIMAX_THRESHOLD
        
        self.assertEqual(starting, StoryArc.STARTING_THRESHOLD)
        self.assertEqual(developing, StoryArc.DEVELOPING_THRESHOLD)
        self.assertEqual(climax, StoryArc.CLIMAX_THRESHOLD)
    
    def test_usage_in_conditional_logic(self):
        """Test using constants in conditional logic (typical usage scenario)."""
        arc = StoryArc("Conditional Test")
        
        # Test typical usage patterns that would previously cause AttributeError
        if arc.progression < StoryArc.STARTING_THRESHOLD:
            phase = "setup"
        elif arc.progression < StoryArc.DEVELOPING_THRESHOLD:
            phase = "rising_action"
        elif arc.progression < StoryArc.CLIMAX_THRESHOLD:
            phase = "climax"
        else:
            phase = "resolution"
        
        self.assertEqual(phase, "setup")  # Initial progression is 0.0


if __name__ == '__main__':
    unittest.main()