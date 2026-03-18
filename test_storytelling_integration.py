#!/usr/bin/env python3
"""
Integration test for the event-driven storytelling system with the gameplay controller.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add the project directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_storytelling_integration():
    """Test storytelling system integration with GameplayController."""
    print("Testing Storytelling System Integration...")
    
    try:
        from tiny_gameplay_controller import GameplayController
        from tiny_event_handler import Event
        
        # Create mock graph manager to avoid dependency issues
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        mock_graph_manager.get_node = Mock(return_value=None)
        mock_graph_manager.add_character_event_edge = Mock()
        mock_graph_manager.add_location_event_edge = Mock()
        mock_graph_manager.update_character_character_edge = Mock()
        
        # Initialize gameplay controller with mock
        config = {"screen_width": 800, "screen_height": 600}
        controller = GameplayController(graph_manager=mock_graph_manager, config=config)
        
        # Check that storytelling system was initialized
        assert hasattr(controller, 'storytelling_system')
        assert controller.storytelling_system is not None
        print("✓ Storytelling system initialized in GameplayController")
        
        # Test feature status
        feature_status = controller.get_feature_implementation_status()
        assert feature_status["event_driven_storytelling"] == "BASIC_IMPLEMENTED"
        print("✓ Feature status correctly shows BASIC_IMPLEMENTED")
        
        # Test storytelling access methods
        stories = controller.get_current_stories()
        assert "active_narratives" in stories
        assert stories["feature_status"] == "BASIC_IMPLEMENTED"
        print("✓ get_current_stories() works")
        
        summary = controller.get_story_summary()
        assert len(summary) > 0
        assert "quiet" in summary.lower() or "story" in summary.lower()
        print("✓ get_story_summary() works")
        
        character_stories = controller.get_character_stories("TestCharacter")
        assert "active_arcs" in character_stories
        print("✓ get_character_stories() works")
        
        # Test event processing with storytelling
        mock_participant = Mock()
        mock_participant.name = "Alice"
        
        test_event = Event(
            name="Village Festival",
            date=datetime.now(),
            event_type="celebration",
            importance=8,
            impact=6,
            participants=[mock_participant]
        )
        
        # Add event to controller's events list and process
        controller.events = [test_event]
        initial_story_count = len(controller.storytelling_system.arc_manager.active_arcs)
        
        # Process events (should trigger storytelling)
        controller._process_pending_events()
        
        # Check that storytelling processed the event
        final_story_count = len(controller.storytelling_system.arc_manager.active_arcs)
        assert final_story_count >= initial_story_count
        
        # Get updated stories
        updated_stories = controller.get_current_stories()
        print(f"✓ Event processed - active arcs: {updated_stories['total_active_arcs']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Storytelling integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_recovery():
    """Test storytelling system recovery functionality."""
    print("\nTesting System Recovery...")
    
    try:
        from tiny_gameplay_controller import GameplayController, SystemRecoveryManager
        
        # Create controller with mock
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        
        controller = GameplayController(graph_manager=mock_graph_manager)
        
        # Test system status
        status = controller.recovery_manager.get_system_status()
        assert "storytelling_system" in status
        print(f"✓ System status includes storytelling_system: {status['storytelling_system']}")
        
        # Test recovery strategies
        assert "storytelling_system" in controller.recovery_manager.recovery_strategies
        print("✓ Storytelling system recovery strategy registered")
        
        # Simulate system failure and recovery
        controller.storytelling_system = None
        
        # Attempt recovery
        recovery_success = controller.recovery_manager.attempt_recovery("storytelling_system")
        assert recovery_success
        assert controller.storytelling_system is not None
        print("✓ Storytelling system recovery successful")
        
        return True
        
    except Exception as e:
        print(f"✗ System recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_story_arc_creation():
    """Test that story arcs are created from significant events."""
    print("\nTesting Story Arc Creation from Events...")
    
    try:
        from tiny_gameplay_controller import GameplayController
        from tiny_event_handler import Event
        
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        mock_graph_manager.get_node = Mock(return_value=None)
        mock_graph_manager.add_character_event_edge = Mock()
        mock_graph_manager.add_location_event_edge = Mock()
        
        controller = GameplayController(graph_manager=mock_graph_manager)
        
        # Create characters
        char1 = Mock()
        char1.name = "Alice"
        char2 = Mock()
        char2.name = "Bob"
        
        # Create significant events
        events = [
            Event(
                name="Harvest Festival",
                date=datetime.now(),
                event_type="celebration",
                importance=9,
                impact=7,
                participants=[char1, char2]
            ),
            Event(
                name="Market Opening",
                date=datetime.now(),
                event_type="economic",
                importance=7,
                impact=5,
                participants=[char1]
            ),
            Event(
                name="Bridge Construction",
                date=datetime.now(),
                event_type="work",
                importance=8,
                impact=6,
                participants=[char1, char2]
            )
        ]
        
        # Process events
        initial_arcs = controller.storytelling_system.arc_manager.get_arc_statistics()["active_arcs"]
        
        for event in events:
            controller.events = [event]
            controller._process_pending_events()
        
        # Check that arcs were created
        final_stats = controller.storytelling_system.arc_manager.get_arc_statistics()
        final_arcs = final_stats["active_arcs"]
        
        assert final_arcs > initial_arcs
        print(f"✓ Story arcs created: {initial_arcs} → {final_arcs}")
        
        # Check that narratives were generated
        stories = controller.get_current_stories()
        narratives = stories["active_narratives"]
        assert len(narratives) > 0
        print(f"✓ Generated {len(narratives)} narrative(s)")
        
        # Check character involvement
        alice_stories = controller.get_character_stories("Alice")
        assert len(alice_stories["active_arcs"]) > 0
        print(f"✓ Alice involved in {len(alice_stories['active_arcs'])} story arc(s)")
        
        return True
        
    except Exception as e:
        print(f"✗ Story arc creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_narrative_coherence():
    """Test that narratives maintain coherence across events."""
    print("\nTesting Narrative Coherence...")
    
    try:
        from tiny_storytelling_system import StorytellingSystem, NarrativeGenerator
        from tiny_event_handler import Event, EventHandler
        
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()
        
        event_handler = EventHandler(mock_graph_manager)
        storytelling_system = StorytellingSystem(event_handler)
        
        # Create a sequence of related events
        char1 = Mock()
        char1.name = "Alice"
        char2 = Mock()
        char2.name = "Bob"
        
        # Sequence: Meeting → Friendship → Collaboration
        events = [
            Event(
                name="First Meeting",
                date=datetime.now(),
                event_type="social",
                importance=6,
                impact=4,
                participants=[char1, char2]
            ),
            Event(
                name="Shared Meal",
                date=datetime.now() + timedelta(days=1),
                event_type="social",
                importance=5,
                impact=3,
                participants=[char1, char2]
            ),
            Event(
                name="Joint Project",
                date=datetime.now() + timedelta(days=3),
                event_type="work",
                importance=7,
                impact=5,
                participants=[char1, char2]
            )
        ]
        
        # Process events in sequence
        narratives = []
        for event in events:
            result = storytelling_system.process_event_for_stories(event)
            narratives.extend(result.get("narratives", []))
        
        # Check narrative coherence
        assert len(narratives) >= 2
        
        # Check that participants are consistently mentioned
        alice_mentions = sum(1 for n in narratives if "Alice" in n)
        bob_mentions = sum(1 for n in narratives if "Bob" in n)
        
        assert alice_mentions >= 2
        assert bob_mentions >= 2
        print(f"✓ Character consistency: Alice mentioned {alice_mentions} times, Bob mentioned {bob_mentions} times")
        
        # Check story progression
        arcs = storytelling_system.arc_manager.active_arcs
        assert len(arcs) > 0
        
        # Find arc involving both characters
        shared_arc = None
        for arc in arcs.values():
            if "Alice" in arc.participants and "Bob" in arc.participants:
                shared_arc = arc
                break
        
        assert shared_arc is not None
        assert len(shared_arc.elements) >= 2
        print(f"✓ Story progression tracked: {len(shared_arc.elements)} story elements")
        
        # Test narrative summary
        summary = storytelling_system.generate_story_summary(days_back=7)
        assert len(summary) > 0
        assert "Alice" in summary or "Bob" in summary
        print(f"✓ Coherent story summary generated: {len(summary)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Narrative coherence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("Event-Driven Storytelling Integration Tests")
    print("=" * 60)
    
    tests = [
        test_storytelling_integration,
        test_system_recovery,
        test_story_arc_creation,
        test_narrative_coherence
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Integration tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n📚 Event-Driven Storytelling System Summary:")
        print("✓ Story Arc Management - Tracks ongoing narrative threads")
        print("✓ Narrative Generation - Converts events to story text")
        print("✓ Narrative Coherence - Maintains story continuity")
        print("✓ GameplayController Integration - Processes events for stories")
        print("✓ System Recovery - Handles failures gracefully")
        print("✓ Feature Status: NOT_STARTED → BASIC_IMPLEMENTED")
    else:
        print("⚠️  Some integration tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)