#!/usr/bin/env python3
"""
Demonstration of the event-driven storytelling system in action.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add the project directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def demonstrate_storytelling_system():
    """Demonstrate the storytelling system with a realistic scenario."""
    print("🎭 Event-Driven Storytelling System Demonstration")
    print("=" * 60)
    
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
    
    # Create village characters
    elena = Mock()
    elena.name = "Elena"
    
    marcus = Mock()
    marcus.name = "Marcus"
    
    sophia = Mock()
    sophia.name = "Sophia"
    
    print("👥 Village Characters: Elena, Marcus, and Sophia")
    print()
    
    # Simulate a week of village events
    events = [
        Event(
            name="Spring Festival Planning",
            date=datetime.now(),
            event_type="social",
            importance=8,
            impact=6,
            participants=[elena, marcus, sophia]
        ),
        Event(
            name="Market Stall Construction",
            date=datetime.now() + timedelta(days=1),
            event_type="work",
            importance=7,
            impact=5,
            participants=[marcus, sophia]
        ),
        Event(
            name="Flower Garden Planting",
            date=datetime.now() + timedelta(days=2),
            event_type="work",
            importance=6,
            impact=4,
            participants=[elena, sophia]
        ),
        Event(
            name="Merchant Caravan Arrival",
            date=datetime.now() + timedelta(days=3),
            event_type="economic",
            importance=7,
            impact=5,
            participants=[elena, marcus]
        ),
        Event(
            name="Spring Festival Celebration",
            date=datetime.now() + timedelta(days=4),
            event_type="celebration",
            importance=9,
            impact=8,
            participants=[elena, marcus, sophia]
        ),
        Event(
            name="Festival Cleanup",
            date=datetime.now() + timedelta(days=5),
            event_type="work",
            importance=5,
            impact=3,
            participants=[elena, marcus, sophia]
        )
    ]
    
    print("📅 Simulating a Week of Village Events...")
    print()
    
    # Process events and watch stories unfold
    for i, event in enumerate(events, 1):
        print(f"Day {i}: {event.name}")
        print("-" * 40)
        
        results = storytelling_system.process_event_for_stories(event)
        
        if results.get("new_arcs"):
            print(f"📖 New story arc created!")
            
        if results.get("narratives"):
            for narrative in results["narratives"]:
                print(f"📝 {narrative}")
        
        print()
    
    print("📚 Final Story Summary:")
    print("=" * 60)
    summary = storytelling_system.generate_story_summary(days_back=7)
    print(summary)
    print()
    
    print("👤 Character Story Involvement:")
    print("-" * 30)
    for character_name in ["Elena", "Marcus", "Sophia"]:
        involvement = storytelling_system.get_character_story_involvement(character_name)
        print(f"{character_name}: {len(involvement['active_arcs'])} active story arcs")
        if involvement['recent_developments']:
            print(f"  Recent: {involvement['recent_developments'][0][:60]}...")
    print()
    
    print("📊 Story Statistics:")
    print("-" * 20)
    stats = storytelling_system.arc_manager.get_arc_statistics()
    print(f"Active story arcs: {stats['active_arcs']}")
    print(f"Total story elements: {stats['total_story_elements']}")
    print(f"Story types: {', '.join([k for k, v in stats['arc_types'].items() if v > 0])}")
    print()
    
    print("🎯 System Status:")
    print("-" * 15)
    current_stories = storytelling_system.get_current_stories()
    print(f"Feature Status: {current_stories['feature_status']}")
    print(f"Last Updated: {current_stories['last_update']}")
    print()
    
    print("✨ The event-driven storytelling system has successfully transformed")
    print("   discrete game events into a cohesive, evolving narrative!")
    
    return True


if __name__ == "__main__":
    demonstrate_storytelling_system()