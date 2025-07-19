#!/usr/bin/env python3
"""
Test the event-driven storytelling system functionality.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add the project directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_storytelling_system():
    """Test basic storytelling system functionality."""
    print("Testing Storytelling System...")
    
    try:
        from tiny_storytelling_system import (
            StorytellingSystem, StoryArcManager, NarrativeGenerator,
            StoryArcType, StoryArcStatus
        )
        from tiny_event_handler import Event, EventHandler
        
        # Create mock participants
        mock_char1 = Mock()
        mock_char1.name = "Alice"
        mock_char2 = Mock()
        mock_char2.name = "Bob"
        
        # Create test events
        social_event = Event(
            name="Village Gathering",
            date=datetime.now(),
            event_type="social",
            importance=7,
            impact=5,
            participants=[mock_char1, mock_char2]
        )
        
        economic_event = Event(
            name="Market Day",
            date=datetime.now(),
            event_type="economic", 
            importance=6,
            impact=3,
            participants=[mock_char1]
        )
        
        # Test StoryArcManager
        print("✓ Testing StoryArcManager...")
        arc_manager = StoryArcManager()
        
        # Create arc from event
        arc = arc_manager.create_arc_from_event(social_event)
        assert arc is not None
        assert arc.arc_type in [StoryArcType.PERSONAL_GROWTH, StoryArcType.SEASONAL]
        assert "Alice" in arc.participants
        print(f"  - Created arc: {arc.title}")
        
        # Test NarrativeGenerator
        print("✓ Testing NarrativeGenerator...")
        narrator = NarrativeGenerator()
        
        story_element = narrator.generate_story_element(social_event, arc)
        assert story_element.event_name == "Village Gathering"
        assert "Alice" in story_element.narrative_text or "Bob" in story_element.narrative_text
        print(f"  - Generated narrative: {story_element.narrative_text}")
        
        # Test StorytellingSystem
        print("✓ Testing StorytellingSystem...")
        storytelling_system = StorytellingSystem()
        
        # Process events
        results = storytelling_system.process_event_for_stories(social_event)
        assert "new_arcs" in results
        assert len(results["new_arcs"]) > 0 or len(results["updated_arcs"]) > 0
        print(f"  - Processed social event: {len(results['narratives'])} narratives generated")
        
        results2 = storytelling_system.process_event_for_stories(economic_event)
        print(f"  - Processed economic event: {len(results2['narratives'])} narratives generated")
        
        # Get current stories
        stories = storytelling_system.get_current_stories()
        assert "active_narratives" in stories
        assert stories["feature_status"] == "BASIC_IMPLEMENTED"
        print(f"  - Current active arcs: {stories['total_active_arcs']}")
        
        # Generate story summary
        summary = storytelling_system.generate_story_summary(days_back=1)
        assert len(summary) > 0
        print(f"  - Generated story summary: {len(summary)} characters")
        
        # Test character involvement
        involvement = storytelling_system.get_character_story_involvement("Alice")
        assert "active_arcs" in involvement
        print(f"  - Alice's story involvement: {len(involvement['active_arcs'])} arcs")
        
        return True
        
    except Exception as e:
        print(f"✗ Storytelling system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_story_arc_progression():
    """Test story arc status progression."""
    print("\nTesting Story Arc Progression...")
    
    try:
        from tiny_storytelling_system import StoryArc, StoryArcType, StoryArcStatus, StoryElement
        from datetime import datetime
        
        # Create a test arc
        arc = StoryArc(
            arc_id="test_001",
            arc_type=StoryArcType.PERSONAL_GROWTH,
            title="Test Arc",
            description="A test story arc",
            participants=["Alice"],
            status=StoryArcStatus.STARTING,
            started_at=datetime.now(),
            last_updated=datetime.now(),
            elements=[],
            themes=["growth", "challenge"],
            importance=5
        )
        
        # Test initial state
        assert arc.status == StoryArcStatus.STARTING
        assert not arc.should_advance_status()
        print("✓ Initial arc state correct")
        
        # Add elements to trigger progression
        for i in range(3):
            element = StoryElement(
                event_name=f"Event {i+1}",
                timestamp=datetime.now(),
                significance=5,
                narrative_text=f"Something happened in event {i+1}",
                character_impact={"Alice": {"emotional_change": 2}},
                emotional_tone="positive"
            )
            arc.add_element(element)
        
        # Should advance to developing
        assert arc.should_advance_status()
        arc.advance_status()
        assert arc.status == StoryArcStatus.DEVELOPING
        print("✓ Arc advanced to DEVELOPING")
        
        # Add more elements
        for i in range(3):
            element = StoryElement(
                event_name=f"Event {i+4}",
                timestamp=datetime.now(),
                significance=6,
                narrative_text=f"More development in event {i+4}",
                character_impact={"Alice": {"emotional_change": 3}},
                emotional_tone="mixed"
            )
            arc.add_element(element)
        
        # Should advance to climax
        assert arc.should_advance_status()
        arc.advance_status()
        assert arc.status == StoryArcStatus.CLIMAX
        print("✓ Arc advanced to CLIMAX")
        
        # Test narrative generation
        narrative = arc.get_current_narrative()
        assert len(narrative) > 0
        assert "test story arc" in narrative.lower()
        print(f"✓ Generated narrative: {len(narrative)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Story arc progression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_narrative_templates():
    """Test narrative template system."""
    print("\nTesting Narrative Templates...")
    
    try:
        from tiny_storytelling_system import NarrativeGenerator
        from tiny_event_handler import Event
        from unittest.mock import Mock
        
        narrator = NarrativeGenerator()
        
        # Test different event types
        event_types = ["social", "economic", "work", "crisis", "celebration"]
        
        for event_type in event_types:
            mock_char = Mock()
            mock_char.name = "TestChar"
            
            event = Event(
                name=f"Test {event_type.title()} Event",
                date=datetime.now(),
                event_type=event_type,
                importance=5,
                impact=3,
                participants=[mock_char]
            )
            
            # Test emotional tone detection
            tone = narrator._determine_emotional_tone(event)
            assert tone in ["positive", "negative", "neutral", "mixed"]
            
            # Test participant formatting
            formatted = narrator._format_participants(event.participants)
            assert "TestChar" in formatted
            
            print(f"  ✓ {event_type}: tone={tone}, participants='{formatted}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Narrative templates test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all storytelling system tests."""
    print("Event-Driven Storytelling System Tests")
    print("=" * 50)
    
    tests = [
        test_storytelling_system,
        test_story_arc_progression,
        test_narrative_templates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All storytelling system tests passed!")
        print("\nFeature Status: NOT_STARTED → BASIC_IMPLEMENTED")
    else:
        print("⚠️  Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)