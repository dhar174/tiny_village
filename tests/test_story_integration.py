#!/usr/bin/env python3
"""
Integration test for tiny_storytelling_engine and tiny_storytelling_system.

This test validates the complete workflow from story event creation to narrative arcs,
ensuring that the storytelling engine and system work together properly.
"""

import unittest
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Suppress logging during tests
logging.disable(logging.CRITICAL)

from tiny_storytelling_engine import StorytellingEventHandler, NarrativeImpact
from tiny_storytelling_system import StorytellingSystem, StoryArc, StoryArcStatus
from tiny_event_handler import Event


class MockCharacter:
    """Mock character for testing."""
    
    def __init__(self, name: str, uuid: str = None):
        self.name = name
        self.uuid = uuid or f"char_{name.lower()}"
        self.energy = 80
        self.health_status = 90


class TestStoryIntegration(unittest.TestCase):
    """Test integration between storytelling engine and system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = StorytellingEventHandler()
        self.system = StorytellingSystem(self.handler)
        
        # Wire them together
        self.handler.storytelling_system = self.system
        
        # Create mock characters
        self.char_a = MockCharacter("Alice")
        self.char_b = MockCharacter("Bob") 
        self.characters = [self.char_a, self.char_b]
    
    def test_story_arc_thresholds_defined(self):
        """Test that story arc progression thresholds are defined."""
        self.assertEqual(StoryArc.STARTING_THRESHOLD, 3)
        self.assertEqual(StoryArc.DEVELOPING_THRESHOLD, 6)
        self.assertEqual(StoryArc.CLIMAX_THRESHOLD, 9)
    
    def test_event_forwarding_to_system(self):
        """Test that events are forwarded from handler to system."""
        # Create a high-importance event that should create an arc
        test_event = Event(
            name="Community Festival",
            date=datetime.now(),
            event_type="social",
            importance=7,  # Above threshold for arc creation
            impact=4,
            participants=self.characters
        )
        
        # Add event to handler
        self.handler.add_event(test_event)
        
        # Check that system received and processed the event
        self.assertGreater(len(self.system.arc_manager.active_arcs), 0)
        
        # Verify an arc was created
        arc = list(self.system.arc_manager.active_arcs.values())[0]
        self.assertEqual(arc.status, StoryArcStatus.STARTING)
        self.assertGreater(len(arc.elements), 0)
    
    def test_importance_scaling_with_narrative_impact(self):
        """Test that narrative impact scales importance correctly."""
        # Create mock narrative context
        self.handler._current_narrative_context = Mock()
        self.handler._current_narrative_context.narrative_impact = NarrativeImpact.MAJOR  # value = 4
        
        # Create story event from template
        story_event = self.handler.create_story_event_from_template(
            "meet_cute", 
            "Meet Cute #1", 
            self.characters
        )
        
        # Should scale: MAJOR (4) * 2 = 8, which is >= 6 threshold
        self.assertIsNotNone(story_event)
        self.assertGreaterEqual(story_event.importance, 6)
        self.assertEqual(story_event.importance, 8)  # 4 * 2
    
    def test_meet_cute_creates_love_arc(self):
        """Test that meet_cute template creates an arc with love theme."""
        # Create meet cute event
        meet_cute_event = self.handler.create_story_event_from_template(
            "meet_cute",
            "Meet Cute #1", 
            self.characters
        )
        
        # Add to handler to trigger system processing
        self.handler.add_event(meet_cute_event)
        
        # Check that an arc was created
        active_arcs = self.system.arc_manager.active_arcs
        self.assertGreater(len(active_arcs), 0)
        
        # Find arc with love theme
        love_arcs = []
        for arc in active_arcs.values():
            if "love" in arc.themes:
                love_arcs.append(arc)
        
        self.assertGreater(len(love_arcs), 0, "No arc with 'love' theme found")
        
        # Verify the love arc details
        love_arc = love_arcs[0]
        self.assertIn("love", love_arc.themes)
        self.assertEqual(love_arc.status, StoryArcStatus.STARTING)
        self.assertIn(self.char_a.name, love_arc.participants)
        self.assertIn(self.char_b.name, love_arc.participants)
    
    def test_story_arc_progression(self):
        """Test that story arcs advance through status progression."""
        # Create an arc manually for testing
        arc = self.system.arc_manager.create_arc_from_event(
            Event(
                name="Test Romance",
                date=datetime.now(),
                event_type="social",
                importance=8,
                impact=4,
                participants=self.characters
            )
        )
        
        self.assertEqual(arc.status, StoryArcStatus.STARTING)
        
        # Add elements to trigger progression
        for i in range(StoryArc.STARTING_THRESHOLD + 1):
            element = Mock()
            element.timestamp = datetime.now()
            element.narrative_text = f"Story element {i}"
            element.significance = 5
            element.character_impact = {}
            element.emotional_tone = "positive"
            arc.add_element(element)
        
        # Update arcs to trigger status advancement
        self.system.arc_manager.update_arcs()
        
        # Arc should have advanced from STARTING to DEVELOPING
        self.assertEqual(arc.status, StoryArcStatus.DEVELOPING)
    
    def test_multiple_events_same_participants(self):
        """Test that multiple events with same participants update the same arc."""
        # Create first event
        event1 = Event(
            name="First Meeting",
            date=datetime.now(),
            event_type="social",
            importance=7,
            impact=3,
            participants=self.characters
        )
        
        self.handler.add_event(event1)
        initial_arc_count = len(self.system.arc_manager.active_arcs)
        first_arc = list(self.system.arc_manager.active_arcs.values())[0]
        initial_elements = len(first_arc.elements)
        
        # Create second event with same participants
        event2 = Event(
            name="Second Encounter", 
            date=datetime.now() + timedelta(hours=1),
            event_type="social",
            importance=6,
            impact=2,
            participants=self.characters
        )
        
        self.handler.add_event(event2)
        
        # Should update existing arc, not create new one
        final_arc_count = len(self.system.arc_manager.active_arcs)
        self.assertEqual(final_arc_count, initial_arc_count)
        
        # Should add element to existing arc
        updated_arc = list(self.system.arc_manager.active_arcs.values())[0]
        final_elements = len(updated_arc.elements)
        self.assertGreater(final_elements, initial_elements)
    
    def test_low_importance_events_ignored(self):
        """Test that low importance events don't create arcs."""
        # Create low importance event
        low_event = Event(
            name="Minor Chat",
            date=datetime.now(),
            event_type="social", 
            importance=3,  # Below threshold for arc creation
            impact=1,
            participants=self.characters
        )
        
        self.handler.add_event(low_event)
        
        # Should not create any arcs
        self.assertEqual(len(self.system.arc_manager.active_arcs), 0)
    
    def test_system_recovery_integration(self):
        """Test that recovery manager properly wires storytelling components."""
        # Skip this test due to pygame dependency
        self.skipTest("Skipping pygame-dependent test")
        
        # Note: In a real environment with pygame, this test would verify:
        # - SystemRecoveryManager can recover StorytellingEventHandler
        # - Recovery properly wires handler to storytelling system
    
    def test_story_event_templates_exist(self):
        """Test that required story templates exist."""
        templates = self.handler.get_event_templates() if hasattr(self.handler, 'get_event_templates') else {}
        
        # Check for meet_cute template in story templates
        self.assertTrue(hasattr(self.handler, 'story_templates'))
        self.assertIn("meet_cute", self.handler.story_templates)
        
        template = self.handler.story_templates["meet_cute"]
        self.assertEqual(template["type"], "romance")
        self.assertGreaterEqual(template["importance"], 6)
    
    def test_narrative_impact_enum_values(self):
        """Test that NarrativeImpact enum has expected values."""
        self.assertEqual(NarrativeImpact.MINOR.value, 1)
        self.assertEqual(NarrativeImpact.MODERATE.value, 2)
        self.assertEqual(NarrativeImpact.SIGNIFICANT.value, 3)
        self.assertEqual(NarrativeImpact.MAJOR.value, 4)  # Key for scaling test
        self.assertEqual(NarrativeImpact.LEGENDARY.value, 5)


if __name__ == '__main__':
    # Enable logging for debugging if needed
    # logging.basicConfig(level=logging.DEBUG)
    
    unittest.main(verbosity=2)