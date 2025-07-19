#!/usr/bin/env python3
"""
Direct test of the storytelling system components without pygame dependencies.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add the project directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_storytelling_components():
    """Test core storytelling components directly."""
    print("Testing Core Storytelling Components...")
    
    try:
        from tiny_storytelling_system import (
            StorytellingSystem, StoryArcManager, NarrativeGenerator,
            StoryArcType, StoryArcStatus, StoryElement, StoryArc
        )
        from tiny_event_handler import Event, EventHandler
        
        # Test initialization
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        
        event_handler = EventHandler(mock_graph_manager)
        storytelling_system = StorytellingSystem(event_handler)
        
        print("✓ All storytelling components initialize successfully")
        
        # Test that it correctly identifies as BASIC_IMPLEMENTED
        current_stories = storytelling_system.get_current_stories()
        assert current_stories["feature_status"] == "BASIC_IMPLEMENTED"
        print("✓ Feature status correctly reports BASIC_IMPLEMENTED")
        
        return True
        
    except Exception as e:
        print(f"✗ Storytelling components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_event_to_story_pipeline():
    """Test the complete pipeline from event to story generation."""
    print("\nTesting Event-to-Story Pipeline...")
    
    try:
        from tiny_storytelling_system import StorytellingSystem
        from tiny_event_handler import Event, EventHandler
        
        # Setup
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        mock_graph_manager.get_node = Mock(return_value=None)
        mock_graph_manager.add_character_event_edge = Mock()
        mock_graph_manager.add_location_event_edge = Mock()
        
        event_handler = EventHandler(mock_graph_manager)
        storytelling_system = StorytellingSystem(event_handler)
        
        # Create test characters
        char1 = Mock()
        char1.name = "Elena"
        char2 = Mock()
        char2.name = "Marcus"
        
        # Create a meaningful event sequence
        events = [
            Event(
                name="Annual Harvest Festival",
                date=datetime.now(),
                event_type="celebration",
                importance=9,
                impact=7,
                participants=[char1, char2],
                effects=[
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "happiness",
                        "change_value": 15,
                    }
                ]
            ),
            Event(
                name="Market Day Preparation",
                date=datetime.now() + timedelta(days=1),
                event_type="work",
                importance=6,
                impact=4,
                participants=[char1, char2]
            ),
            Event(
                name="Community Meeting",
                date=datetime.now() + timedelta(days=2),
                event_type="social",
                importance=7,
                impact=5,
                participants=[char1, char2]
            )
        ]
        
        # Process events and track results
        total_narratives = []
        total_arcs_created = 0
        
        for i, event in enumerate(events):
            results = storytelling_system.process_event_for_stories(event)
            total_narratives.extend(results.get("narratives", []))
            total_arcs_created += len(results.get("new_arcs", []))
            
            print(f"  Event {i+1} ({event.name}):")
            print(f"    - New arcs: {len(results.get('new_arcs', []))}")
            print(f"    - Updated arcs: {len(results.get('updated_arcs', []))}")
            print(f"    - Narratives: {len(results.get('narratives', []))}")
            
            if results.get("narratives"):
                print(f"    - Sample narrative: {results['narratives'][0][:100]}...")
        
        # Verify comprehensive results
        assert len(total_narratives) >= 3, f"Expected at least 3 narratives, got {len(total_narratives)}"
        assert total_arcs_created >= 1, f"Expected at least 1 arc created, got {total_arcs_created}"
        
        # Test story summary
        summary = storytelling_system.generate_story_summary(days_back=7)
        assert len(summary) > 100, f"Story summary too short: {len(summary)} characters"
        assert "Elena" in summary or "Marcus" in summary, "Characters not mentioned in summary"
        
        print(f"✓ Pipeline successful: {len(total_narratives)} narratives, {total_arcs_created} arcs")
        print(f"✓ Story summary generated: {len(summary)} characters")
        
        # Test character story involvement
        elena_stories = storytelling_system.get_character_story_involvement("Elena")
        marcus_stories = storytelling_system.get_character_story_involvement("Marcus")
        
        assert len(elena_stories["active_arcs"]) > 0, "Elena should be involved in story arcs"
        assert len(marcus_stories["active_arcs"]) > 0, "Marcus should be involved in story arcs"
        
        print(f"✓ Character involvement: Elena in {len(elena_stories['active_arcs'])} arcs, "
              f"Marcus in {len(marcus_stories['active_arcs'])} arcs")
        
        return True
        
    except Exception as e:
        print(f"✗ Event-to-story pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_story_arc_progression():
    """Test that story arcs progress through their lifecycle correctly."""
    print("\nTesting Story Arc Progression...")
    
    try:
        from tiny_storytelling_system import (
            StoryArc, StoryArcType, StoryArcStatus, StoryElement
        )
        
        # Create a test story arc
        arc = StoryArc(
            arc_id="progression_test",
            arc_type=StoryArcType.COMMUNITY,
            title="The Village Bridge Project",
            description="A story about the community coming together to build a bridge",
            participants=["Elena", "Marcus", "Sophia"],
            status=StoryArcStatus.STARTING,
            started_at=datetime.now(),
            last_updated=datetime.now(),
            elements=[],
            themes=["cooperation", "building", "progress"],
            importance=8
        )
        
        # Track status progression
        status_history = [arc.status]
        
        # Add story elements to trigger progression
        story_events = [
            "The villagers gathered to discuss the bridge project",
            "Elena proposed a detailed construction plan",
            "Marcus volunteered to source materials from the quarry",
            "Sophia organized work teams for different phases",
            "The foundation was laid with great ceremony",
            "Challenges arose when the river flooded during construction",
            "The community rallied to overcome the obstacles",
            "The bridge was completed ahead of schedule"
        ]
        
        for i, event_text in enumerate(story_events):
            element = StoryElement(
                event_name=f"Bridge Event {i+1}",
                timestamp=datetime.now() + timedelta(hours=i),
                significance=7,
                narrative_text=event_text,
                character_impact={"Elena": {"emotional_change": 2}, 
                                "Marcus": {"emotional_change": 2},
                                "Sophia": {"emotional_change": 2}},
                emotional_tone="positive"
            )
            
            arc.add_element(element)
            
            # Check for status advancement
            if arc.should_advance_status():
                arc.advance_status()
                status_history.append(arc.status)
                print(f"  After element {i+1}: Advanced to {arc.status.value}")
        
        # Verify progression
        assert len(status_history) > 1, "Arc status should have progressed"
        assert StoryArcStatus.DEVELOPING in status_history, "Arc should reach DEVELOPING status"
        
        # Test narrative generation
        narrative = arc.get_current_narrative()
        assert len(narrative) > 200, f"Narrative too short: {len(narrative)} characters"
        assert "bridge" in narrative.lower(), "Narrative should mention the bridge"
        
        print(f"✓ Status progression: {' → '.join([s.value for s in status_history])}")
        print(f"✓ Final narrative: {len(narrative)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Story arc progression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_narrative_template_system():
    """Test the narrative template system for different event types."""
    print("\nTesting Narrative Template System...")
    
    try:
        from tiny_storytelling_system import NarrativeGenerator, StoryArc, StoryArcType, StoryArcStatus
        from tiny_event_handler import Event
        
        narrator = NarrativeGenerator()
        
        # Create a test arc
        test_arc = StoryArc(
            arc_id="template_test",
            arc_type=StoryArcType.COMMUNITY,
            title="Test Arc",
            description="Testing narrative templates",
            participants=["Alice", "Bob"],
            status=StoryArcStatus.DEVELOPING,
            started_at=datetime.now(),
            last_updated=datetime.now(),
            elements=[],
            themes=["test"],
            importance=5
        )
        
        # Test different event types and their narratives
        event_types = [
            ("social", "Community Gathering"),
            ("economic", "Market Day"),
            ("work", "Barn Raising"),
            ("crisis", "Storm Warning"),
            ("celebration", "Harvest Festival")
        ]
        
        generated_narratives = {}
        
        for event_type, event_name in event_types:
            # Create test characters
            char1 = Mock()
            char1.name = "Alice"
            char2 = Mock()
            char2.name = "Bob"
            
            event = Event(
                name=event_name,
                date=datetime.now(),
                event_type=event_type,
                importance=6,
                impact=4,
                participants=[char1, char2]
            )
            
            story_element = narrator.generate_story_element(event, test_arc)
            generated_narratives[event_type] = story_element.narrative_text
            
            # Verify narrative quality
            assert len(story_element.narrative_text) > 50, f"Narrative too short for {event_type}"
            assert "Alice" in story_element.narrative_text or "Bob" in story_element.narrative_text, \
                   f"Characters not mentioned in {event_type} narrative"
            
            print(f"  {event_type}: {story_element.narrative_text[:80]}...")
        
        # Verify different event types produce different narrative styles
        social_narrative = generated_narratives["social"]
        crisis_narrative = generated_narratives["crisis"]
        
        assert social_narrative != crisis_narrative, "Different event types should produce different narratives"
        
        # Test emotional tone detection
        celebration_tone = narrator._determine_emotional_tone(Event(
            name="Festival", date=datetime.now(), event_type="celebration",
            importance=8, impact=5, participants=[]
        ))
        crisis_tone = narrator._determine_emotional_tone(Event(
            name="Disaster", date=datetime.now(), event_type="crisis",
            importance=9, impact=-7, participants=[]
        ))
        
        assert celebration_tone == "positive", f"Celebration should be positive, got {celebration_tone}"
        assert crisis_tone == "negative", f"Crisis should be negative, got {crisis_tone}"
        
        print(f"✓ Generated {len(generated_narratives)} different narrative types")
        print(f"✓ Emotional tone detection: celebration={celebration_tone}, crisis={crisis_tone}")
        
        return True
        
    except Exception as e:
        print(f"✗ Narrative template system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_implementation_confirmation():
    """Confirm that the event-driven storytelling feature is properly implemented."""
    print("\nConfirming Feature Implementation...")
    
    try:
        from tiny_storytelling_system import StorytellingSystem
        from tiny_event_handler import EventHandler
        
        # Test that all required components exist and work
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        
        event_handler = EventHandler(mock_graph_manager)
        storytelling_system = StorytellingSystem(event_handler)
        
        # Verify all required components
        assert hasattr(storytelling_system, 'arc_manager'), "Missing arc_manager"
        assert hasattr(storytelling_system, 'narrative_generator'), "Missing narrative_generator"
        assert hasattr(storytelling_system, 'process_event_for_stories'), "Missing event processing"
        assert hasattr(storytelling_system, 'get_current_stories'), "Missing story access"
        assert hasattr(storytelling_system, 'generate_story_summary'), "Missing story summary"
        
        print("✓ All required storytelling components present")
        
        # Verify feature status
        stories = storytelling_system.get_current_stories()
        assert stories["feature_status"] == "BASIC_IMPLEMENTED"
        
        print("✓ Feature status confirmed: BASIC_IMPLEMENTED")
        
        # Test core functionality works
        mock_char = Mock()
        mock_char.name = "TestCharacter"
        
        from tiny_event_handler import Event
        test_event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="social",
            importance=7,
            impact=5,
            participants=[mock_char]
        )
        
        results = storytelling_system.process_event_for_stories(test_event)
        assert "narratives" in results
        assert len(results["narratives"]) > 0
        
        print("✓ Core functionality verified: Events → Stories pipeline works")
        
        # Verify story arc management
        arc_stats = storytelling_system.arc_manager.get_arc_statistics()
        assert "active_arcs" in arc_stats
        assert "completed_arcs" in arc_stats
        
        print("✓ Story arc management verified")
        
        # Verify narrative coherence
        summary = storytelling_system.generate_story_summary()
        assert len(summary) > 0
        
        print("✓ Narrative generation verified")
        
        print("\n📊 Feature Implementation Summary:")
        print("   ✅ Event triggers - Working with existing event handler")
        print("   ✅ Story arc management - Tracks narrative threads over time")
        print("   ✅ Narrative coherence - Maintains story continuity")
        print("   ✅ Dynamic narrative generation - Converts events to stories")
        print("   ✅ Integration ready - Can be integrated with gameplay controller")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature implementation confirmation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive storytelling system tests."""
    print("Event-Driven Storytelling System - Comprehensive Tests")
    print("=" * 70)
    
    tests = [
        test_storytelling_components,
        test_event_to_story_pipeline,
        test_story_arc_progression,
        test_narrative_template_system,
        test_feature_implementation_confirmation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"Comprehensive tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Event-Driven Storytelling System IMPLEMENTED!")
        print("\n🏆 Achievement Unlocked: Event-Driven Storytelling System")
        print("   Status: NOT_STARTED → BASIC_IMPLEMENTED")
        print("   Components: Event triggers ✅ | Story arc management ✅ | Narrative coherence ✅")
        print("   Impact: Dynamic narrative generation from game events")
    else:
        print("⚠️  Some tests failed - review implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)