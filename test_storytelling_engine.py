#!/usr/bin/env python3
"""
Test suite for the Event-Driven Storytelling Engine.

Tests the narrative enhancements to the event system including:
- Character action monitoring and triggers
- Story event creation and chaining
- World state change tracking
- Narrative context and story arc management
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add the project directory to path
sys.path.insert(0, "/home/runner/work/tiny_village/tiny_village")

from tiny_storytelling_engine import (
    StorytellingEventHandler,
    StoryEventType,
    NarrativeImpact,
    StoryContext,
    CharacterActionTrigger,
    WorldStateChange
)
from tiny_event_handler import Event


class TestStorytellingEngine(unittest.TestCase):
    """Test the core storytelling engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock graph manager
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.add_event_node = Mock()
        
        # Create storytelling handler
        self.story_handler = StorytellingEventHandler(self.mock_graph_manager)
        
        # Create mock character
        self.mock_character = Mock()
        self.mock_character.name = "Alice"
        self.mock_character.happiness = 75
        self.mock_character.energy = 60
        self.mock_character.location = Mock()
        self.mock_character.location.name = "Village Square"
    
    def test_initialization(self):
        """Test that the storytelling handler initializes correctly."""
        self.assertIsInstance(self.story_handler, StorytellingEventHandler)
        self.assertEqual(len(self.story_handler.character_action_monitors), 0)
        self.assertEqual(len(self.story_handler.world_state_history), 0)
        self.assertEqual(len(self.story_handler.active_story_threads), 0)
        self.assertTrue(hasattr(self.story_handler, 'story_templates'))
    
    def test_character_action_trigger_creation(self):
        """Test creation and management of character action triggers."""
        # Create a simple trigger
        trigger = CharacterActionTrigger(
            action_names={"talk", "socialize"},
            character_conditions={"happiness": {"operator": ">=", "value": 50}},
            probability=0.8
        )
        
        # Add the trigger
        self.story_handler.add_character_action_monitor("social_trigger", trigger)
        
        self.assertIn("social_trigger", self.story_handler.character_action_monitors)
        self.assertEqual(self.story_handler.trigger_counts["social_trigger"], 0)
    
    def test_character_action_monitoring(self):
        """Test monitoring character actions for story triggers."""
        # Setup trigger
        trigger = CharacterActionTrigger(
            action_names={"talk"},
            character_conditions={"happiness": {"operator": ">=", "value": 50}},
            probability=1.0  # Ensure it triggers
        )
        self.story_handler.add_character_action_monitor("talk_trigger", trigger)
        
        # Monitor a matching action
        action_result = {"success": True, "target": "Bob"}
        triggered_events = self.story_handler.monitor_character_action(
            self.mock_character, "talk", action_result
        )
        
        # Should trigger a story event
        self.assertEqual(len(triggered_events), 1)
        self.assertEqual(triggered_events[0].type, "story")
        self.assertIn(self.mock_character, triggered_events[0].participants)
    
    def test_character_condition_checking(self):
        """Test character condition evaluation."""
        # Test simple conditions
        conditions = {"happiness": {"operator": ">=", "value": 50}}
        result = self.story_handler._check_character_conditions(self.mock_character, conditions)
        self.assertTrue(result)  # Character has happiness = 75
        
        # Test failing condition
        conditions = {"happiness": {"operator": ">=", "value": 100}}
        result = self.story_handler._check_character_conditions(self.mock_character, conditions)
        self.assertFalse(result)  # Character has happiness = 75
        
        # Test direct value comparison
        conditions = {"name": "Alice"}
        result = self.story_handler._check_character_conditions(self.mock_character, conditions)
        self.assertTrue(result)
    
    def test_location_condition_checking(self):
        """Test location-based condition evaluation."""
        conditions = {"location_name": "Village Square"}
        result = self.story_handler._check_location_conditions(self.mock_character, conditions)
        self.assertTrue(result)
        
        conditions = {"location_name": "Forest"}
        result = self.story_handler._check_location_conditions(self.mock_character, conditions)
        self.assertFalse(result)
    
    def test_trigger_cooldown(self):
        """Test trigger cooldown functionality."""
        trigger = CharacterActionTrigger(
            action_names={"talk"},
            cooldown_hours=1,
            probability=1.0
        )
        self.story_handler.add_character_action_monitor("cooldown_trigger", trigger)
        
        # First trigger should work
        events1 = self.story_handler.monitor_character_action(
            self.mock_character, "talk", {}
        )
        self.assertEqual(len(events1), 1)
        
        # Immediate second trigger should be blocked by cooldown
        events2 = self.story_handler.monitor_character_action(
            self.mock_character, "talk", {}
        )
        self.assertEqual(len(events2), 0)
    
    def test_max_triggers_limit(self):
        """Test maximum triggers limit."""
        trigger = CharacterActionTrigger(
            action_names={"talk"},
            max_triggers=1,
            cooldown_hours=0,  # No cooldown for this test
            probability=1.0
        )
        self.story_handler.add_character_action_monitor("limited_trigger", trigger)
        
        # First trigger should work
        events1 = self.story_handler.monitor_character_action(
            self.mock_character, "talk", {}
        )
        self.assertEqual(len(events1), 1)
        
        # Second trigger should be blocked by max_triggers limit
        events2 = self.story_handler.monitor_character_action(
            self.mock_character, "talk", {}
        )
        self.assertEqual(len(events2), 0)
    
    def test_world_state_change_tracking(self):
        """Test tracking and responding to world state changes."""
        # Create a world state change
        change = WorldStateChange(
            change_type="economic",
            affected_entities=[self.mock_character],
            magnitude=80,  # High magnitude
            timestamp=datetime.now(),
            description="Major economic shift in the village"
        )
        
        # Track the change
        initial_event_count = len(self.story_handler.events)
        self.story_handler.track_world_state_change(change)
        
        # Should add the change to history
        self.assertEqual(len(self.story_handler.world_state_history), 1)
        
        # Should potentially trigger story events for high magnitude changes
        final_event_count = len(self.story_handler.events)
        self.assertGreaterEqual(final_event_count, initial_event_count)
    
    def test_story_chain_creation(self):
        """Test creation of connected story event chains."""
        chain_events = [
            {
                "delay_hours": 0,
                "importance": 6,
                "impact": 4,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "curiosity",
                        "change_value": 10
                    }
                ]
            },
            {
                "delay_hours": 24,
                "importance": 7,
                "impact": 5,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "determination",
                        "change_value": 15
                    }
                ]
            }
        ]
        
        # Create story chain
        story_events = self.story_handler.create_story_chain(
            "mystery_investigation",
            chain_events,
            [self.mock_character]
        )
        
        # Should create the expected number of events
        self.assertEqual(len(story_events), 2)
        
        # Should track the story thread
        self.assertIn("mystery_investigation", self.story_handler.active_story_threads)
        
        # Events should have the correct participants
        for event in story_events:
            self.assertIn(self.mock_character, event.participants)
            self.assertEqual(event.type, "story_chain")
    
    def test_story_template_system(self):
        """Test story template creation and usage."""
        # Test predefined template usage
        romance_event = self.story_handler.create_story_event_from_template(
            "meet_cute",
            "Alice Meets Bob",
            [self.mock_character]
        )
        
        self.assertIsNotNone(romance_event)
        self.assertEqual(romance_event.name, "Alice Meets Bob")
        self.assertIn(self.mock_character, romance_event.participants)
        
        # Test adding custom template
        self.story_handler.add_story_template("custom_event", {
            "type": "adventure",
            "importance": 8,
            "impact": 6,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "bravery",
                    "change_value": 20
                }
            ]
        })
        
        custom_event = self.story_handler.create_story_event_from_template(
            "custom_event",
            "Brave Adventure",
            [self.mock_character]
        )
        
        self.assertIsNotNone(custom_event)
        self.assertEqual(custom_event.importance, 8)
    
    def test_character_story_summary(self):
        """Test generation of character story summaries."""
        # Add some story events involving the character
        story_event1 = Event(
            name="Alice's Adventure",
            date=datetime.now(),
            event_type="story",
            importance=6,
            impact=4,
            participants=[self.mock_character]
        )
        
        story_event2 = Event(
            name="Alice's Romance",
            date=datetime.now(),
            event_type="story",
            importance=5,
            impact=3,
            participants=[self.mock_character]
        )
        
        self.story_handler.add_event(story_event1)
        self.story_handler.add_event(story_event2)
        
        # Generate summary
        summary = self.story_handler.get_character_story_summary(self.mock_character)
        
        self.assertEqual(summary["total_story_events"], 2)
        self.assertEqual(len(summary["events"]), 2)
        self.assertIn("character", summary)
    
    def test_narrative_summary_generation(self):
        """Test generation of narrative summaries."""
        # Add some processed story events
        story_event = Event(
            name="Village Festival",
            date=datetime.now(),
            event_type="story",
            importance=7,
            impact=5,
            participants=[self.mock_character]
        )
        story_event.last_triggered = datetime.now()
        
        self.story_handler.processed_events.append(story_event)
        
        # Generate narrative summary
        summary = self.story_handler.generate_narrative_summary(timeframe_days=7)
        
        self.assertIn("timeframe_days", summary)
        self.assertIn("total_story_events", summary)
        self.assertIn("story_themes", summary)
        self.assertIn("character_involvement", summary)
        self.assertIn("narrative_momentum", summary)
    
    def test_story_thread_management(self):
        """Test story thread tracking and completion."""
        # Create a story chain to establish a thread
        chain_events = [{"delay_hours": 0, "importance": 5, "impact": 3}]
        self.story_handler.create_story_chain(
            "test_thread",
            chain_events,
            [self.mock_character]
        )
        
        # Check active threads
        active_threads = self.story_handler.get_active_story_threads()
        self.assertIn("test_thread", active_threads)
        self.assertEqual(active_threads["test_thread"]["status"], "active")
        
        # Complete the thread
        self.story_handler.complete_story_thread("test_thread")
        
        # Check thread is marked as completed
        updated_threads = self.story_handler.get_active_story_threads()
        self.assertEqual(updated_threads["test_thread"]["status"], "completed")
        self.assertIn("completed_at", updated_threads["test_thread"])


class TestStoryContextAndEvents(unittest.TestCase):
    """Test story context and narrative event functionality."""
    
    def test_story_context_creation(self):
        """Test creation of story context objects."""
        context = StoryContext(
            theme=StoryEventType.ROMANCE,
            narrative_impact=NarrativeImpact.SIGNIFICANT,
            character_arcs={"alice": "love_interest", "bob": "romantic_lead"},
            emotional_tone="romantic"
        )
        
        self.assertEqual(context.theme, StoryEventType.ROMANCE)
        self.assertEqual(context.narrative_impact, NarrativeImpact.SIGNIFICANT)
        self.assertEqual(context.emotional_tone, "romantic")
        self.assertIn("alice", context.character_arcs)
    
    def test_narrative_impact_levels(self):
        """Test narrative impact enum values."""
        self.assertEqual(NarrativeImpact.MINOR.value, 1)
        self.assertEqual(NarrativeImpact.MODERATE.value, 2)
        self.assertEqual(NarrativeImpact.SIGNIFICANT.value, 3)
        self.assertEqual(NarrativeImpact.MAJOR.value, 4)
        self.assertEqual(NarrativeImpact.LEGENDARY.value, 5)
    
    def test_story_event_types(self):
        """Test story event type enumeration."""
        # Test a few key story types
        self.assertEqual(StoryEventType.ROMANCE.value, "romance")
        self.assertEqual(StoryEventType.ADVENTURE.value, "adventure")
        self.assertEqual(StoryEventType.MYSTERY.value, "mystery")
        self.assertEqual(StoryEventType.HEROIC_JOURNEY.value, "heroic_journey")
    
    def test_world_state_change_creation(self):
        """Test creation of world state change objects."""
        change = WorldStateChange(
            change_type="political",
            affected_entities=["village_council", "mayor"],
            magnitude=65,
            timestamp=datetime.now(),
            description="New mayor elected"
        )
        
        self.assertEqual(change.change_type, "political")
        self.assertEqual(change.magnitude, 65)
        self.assertIn("village_council", change.affected_entities)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complex integration scenarios for storytelling."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.add_event_node = Mock()
        
        self.story_handler = StorytellingEventHandler(self.mock_graph_manager)
        
        # Create multiple characters for complex scenarios
        self.alice = Mock()
        self.alice.name = "Alice"
        self.alice.happiness = 70
        self.alice.location = Mock()
        self.alice.location.name = "Village Square"
        
        self.bob = Mock()
        self.bob.name = "Bob"
        self.bob.happiness = 65
        self.bob.location = Mock()
        self.bob.location.name = "Village Square"
    
    def test_romance_storyline_scenario(self):
        """Test a complete romance storyline from meeting to relationship."""
        # Set up romance triggers
        meet_trigger = CharacterActionTrigger(
            action_names={"talk", "greet"},
            location_conditions={"location_name": "Village Square"},
            probability=1.0
        )
        self.story_handler.add_character_action_monitor("romance_meet", meet_trigger)
        
        # Simulate first meeting
        triggered_events = self.story_handler.monitor_character_action(
            self.alice, "talk", {"target": self.bob}
        )
        
        self.assertEqual(len(triggered_events), 1)
        
        # Create a romance story chain
        romance_chain = [
            {
                "delay_hours": 6,
                "importance": 5,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "romantic_interest",
                        "change_value": 10
                    }
                ]
            },
            {
                "delay_hours": 24,
                "importance": 6,
                "effects": [
                    {
                        "type": "relationship_change",
                        "targets": ["participants"],
                        "attribute": "relationship_status",
                        "change_value": 15
                    }
                ]
            }
        ]
        
        story_events = self.story_handler.create_story_chain(
            "alice_bob_romance",
            romance_chain,
            [self.alice, self.bob]
        )
        
        self.assertEqual(len(story_events), 2)
        self.assertIn("alice_bob_romance", self.story_handler.active_story_threads)
    
    def test_village_crisis_scenario(self):
        """Test a village-wide crisis scenario with multiple story threads."""
        # Create a major world state change
        crisis_change = WorldStateChange(
            change_type="disaster",
            affected_entities=[self.alice, self.bob, "village"],
            magnitude=90,
            timestamp=datetime.now(),
            description="Natural disaster strikes the village"
        )
        
        # Track the crisis
        self.story_handler.track_world_state_change(crisis_change)
        
        # Create crisis response story chain
        crisis_response = [
            {
                "delay_hours": 1,
                "importance": 9,
                "impact": -5,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "stress",
                        "change_value": 30
                    }
                ]
            },
            {
                "delay_hours": 12,
                "importance": 8,
                "impact": 3,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "community_spirit",
                        "change_value": 20
                    }
                ]
            },
            {
                "delay_hours": 72,
                "importance": 7,
                "impact": 5,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "resilience",
                        "change_value": 25
                    }
                ]
            }
        ]
        
        story_events = self.story_handler.create_story_chain(
            "village_crisis_response",
            crisis_response,
            [self.alice, self.bob]
        )
        
        self.assertEqual(len(story_events), 3)
        
        # Generate narrative summary to see the impact
        summary = self.story_handler.generate_narrative_summary(timeframe_days=1)
        self.assertGreaterEqual(summary["total_story_events"], 0)
    
    def test_character_development_arc(self):
        """Test tracking a character's development through multiple events."""
        # Create events that form a character development arc
        development_events = [
            ("Alice's Challenge", "adventure", 6, 4),
            ("Alice's Growth", "coming_of_age", 7, 5),
            ("Alice's Triumph", "heroic_journey", 8, 6)
        ]
        
        for event_name, event_type, importance, impact in development_events:
            story_event = Event(
                name=event_name,
                date=datetime.now(),
                event_type=event_type,
                importance=importance,
                impact=impact,
                participants=[self.alice]
            )
            story_event.last_triggered = datetime.now()
            
            self.story_handler.add_event(story_event)
            self.story_handler.processed_events.append(story_event)
        
        # Get character story summary
        summary = self.story_handler.get_character_story_summary(self.alice)
        
        self.assertEqual(summary["total_story_events"], 3)
        
        # Check that events show progression in importance
        events = summary["events"]
        self.assertEqual(events[0]["importance"], 6)
        self.assertEqual(events[1]["importance"], 7)
        self.assertEqual(events[2]["importance"], 8)


def run_storytelling_tests():
    """Run all storytelling engine tests."""
    print("Running Event-Driven Storytelling Engine Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStorytellingEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestStoryContextAndEvents))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_storytelling_tests()
    sys.exit(0 if success else 1)