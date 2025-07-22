#!/usr/bin/env python3
"""
Example demonstration of Event-Driven Storytelling integration with TinyVillage.

This script shows how the storytelling engine can be integrated into the existing
game systems to create dynamic narratives based on character actions and world events.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add the project directory to path
# sys.path.insert(0, "/home/runner/work/tiny_village/tiny_village")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from storytelling_integration import StorytellingGameIntegration
from tiny_storytelling_engine import StoryEventType, NarrativeImpact


def create_mock_character(name: str, **attributes):
    """Create a mock character with specified attributes."""
    character = Mock()
    character.name = name
    
    # Default attributes
    default_attrs = {
        'happiness': 60,
        'energy': 70,
        'curiosity': 50,
        'bravery': 40,
        'empathy': 55,
        'intelligence': 65,
        'stress': 25,
        'wealth': 100,
        'social_status': 3
    }
    
    # Apply defaults and overrides
    for attr, value in default_attrs.items():
        setattr(character, attr, attributes.get(attr, value))
    
    # Create location
    character.location = Mock()
    character.location.name = attributes.get('location', 'Village Square')
    
    return character


def demonstrate_romance_storyline(integration: StorytellingGameIntegration):
    """Demonstrate a romance storyline between two characters."""
    print("\n🌹 ROMANCE STORYLINE DEMONSTRATION")
    print("-" * 40)
    
    # Create two characters
    alice = create_mock_character("Alice", happiness=75, empathy=80, location="Village Square")
    bob = create_mock_character("Bob", happiness=70, empathy=75, location="Village Square")
    
    print(f"Characters: {alice.name} and {bob.name}")
    
    # Simulate meeting and talking
    events1 = integration.monitor_character_action(
        alice, "talk", 
        {"target": bob}, 
        {"success": True, "emotional_response": "positive"}
    )
    print(f"Alice talks to Bob -> {len(events1)} story events triggered")
    
    # Create a romance arc for both characters
    alice_arc = integration.create_character_arc_storyline(alice, "romance", 7)
    bob_arc = integration.create_character_arc_storyline(bob, "romance", 7)
    
    print(f"Romance arcs created: Alice ({len(alice_arc)} events), Bob ({len(bob_arc)} events)")
    
    # Simulate more romantic interactions
    romantic_actions = ["compliment", "gift", "flirt"]
    for action in romantic_actions:
        events = integration.monitor_character_action(
            alice, action,
            {"target": bob},
            {"success": True}
        )
        if events:
            print(f"  {action.title()} -> {len(events)} story events")
    
    # Get story summary
    alice_status = integration.get_character_story_status(alice)
    print(f"Alice's story: {alice_status['story_summary']['total_story_events']} events, "
          f"{len(alice_status['active_threads'])} active threads")


def demonstrate_hero_journey(integration: StorytellingGameIntegration):
    """Demonstrate a hero's journey storyline."""
    print("\n⚔️  HERO'S JOURNEY DEMONSTRATION")
    print("-" * 40)
    
    # Create a brave character
    hero = create_mock_character("Marcus", bravery=85, curiosity=90, intelligence=75)
    
    print(f"Hero: {hero.name}")
    
    # Start with exploration that triggers adventure
    events1 = integration.monitor_character_action(
        hero, "explore",
        {"target": "mysterious_cave"},
        {"success": True, "discovery": "ancient_artifact"}
    )
    print(f"Hero explores mysterious cave -> {len(events1)} story events triggered")
    
    # Create hero's journey arc
    hero_arc = integration.create_character_arc_storyline(hero, "hero_journey", 10)
    print(f"Hero's journey arc created with {len(hero_arc)} events over 10 days")
    
    # Simulate heroic actions
    heroic_actions = ["investigate", "search", "venture"]
    for action in heroic_actions:
        events = integration.monitor_character_action(
            hero, action,
            {"target": "unknown_territory"},
            {"success": True, "danger_level": "high"}
        )
        if events:
            print(f"  {action.title()} -> {len(events)} story events")
    
    # Get story summary
    hero_status = integration.get_character_story_status(hero)
    print(f"Hero's story: {hero_status['story_summary']['total_story_events']} events")


def demonstrate_village_crisis(integration: StorytellingGameIntegration):
    """Demonstrate a village-wide crisis scenario."""
    print("\n🔥 VILLAGE CRISIS DEMONSTRATION")
    print("-" * 40)
    
    # Create multiple villagers
    villagers = [
        create_mock_character("Sarah", empathy=90, bravery=60),
        create_mock_character("Tom", bravery=85, stress=40),
        create_mock_character("Emma", intelligence=80, empathy=75),
        create_mock_character("Jack", bravery=70, stress=30)
    ]
    
    print(f"Villagers: {', '.join(v.name for v in villagers)}")
    
    # Trigger a major crisis
    crisis_events = integration.track_world_state_change(
        "natural_disaster",
        villagers + ["village_infrastructure"],
        magnitude=95,
        description="Massive storm threatens the village",
        related_actions=["weather_monitoring", "emergency_preparation"]
    )
    print(f"Crisis triggered -> {len(crisis_events)} immediate story events")
    
    # Simulate villagers responding to crisis
    for villager in villagers:
        # Each villager helps in their own way
        if villager.empathy > 80:
            action = "help"
        elif villager.bravery > 80:
            action = "lead_rescue"
        else:
            action = "support"
        
        events = integration.monitor_character_action(
            villager, action,
            {"crisis_response": True},
            {"success": True, "community_impact": "positive"}
        )
        if events:
            print(f"  {villager.name} performs {action} -> {len(events)} story events")
    
    # Create a community recovery storyline
    recovery_events = [
        {
            "delay_hours": 2,
            "importance": 8,
            "impact": -3,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "crisis_stress",
                    "change_value": 25
                }
            ]
        },
        {
            "delay_hours": 24,
            "importance": 7,
            "impact": 4,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "community_unity",
                    "change_value": 30
                }
            ]
        },
        {
            "delay_hours": 72,
            "importance": 6,
            "impact": 5,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "resilience",
                    "change_value": 20
                }
            ]
        }
    ]
    
    recovery_arc = integration.story_handler.create_story_chain(
        "village_recovery",
        recovery_events,
        villagers
    )
    print(f"Village recovery storyline created with {len(recovery_arc)} events")


def demonstrate_friendship_development(integration: StorytellingGameIntegration):
    """Demonstrate friendship development between characters."""
    print("\n👥 FRIENDSHIP DEVELOPMENT DEMONSTRATION")
    print("-" * 40)
    
    # Create characters with different personalities
    outgoing = create_mock_character("Luna", empathy=85, happiness=80, stress=15)
    shy = create_mock_character("Oliver", empathy=70, happiness=50, stress=60)
    
    print(f"Characters: {outgoing.name} (outgoing) and {shy.name} (shy)")
    
    # Luna helps Oliver (friendship trigger)
    events1 = integration.monitor_character_action(
        outgoing, "help",
        {"target": shy, "help_type": "emotional_support"},
        {"success": True, "recipient_response": "grateful"}
    )
    print(f"Luna helps Oliver -> {len(events1)} story events triggered")
    
    # Create friendship arcs for both
    luna_friendship = integration.create_character_arc_storyline(outgoing, "friendship", 5)
    oliver_friendship = integration.create_character_arc_storyline(shy, "friendship", 5)
    
    print(f"Friendship arcs: Luna ({len(luna_friendship)} events), Oliver ({len(oliver_friendship)} events)")
    
    # Simulate friendship-building activities
    friendship_actions = [
        ("collaborate", {"project": "garden", "duration": "afternoon"}),
        ("support", {"situation": "family_troubles", "type": "listening"}),
        ("assist", {"task": "house_repairs", "effort": "significant"})
    ]
    
    for action, params in friendship_actions:
        events = integration.monitor_character_action(
            outgoing, action, params, {"success": True, "bond_strengthened": True}
        )
        if events:
            print(f"  {action.title()} -> {len(events)} story events")


def demonstrate_character_development_tracking(integration: StorytellingGameIntegration):
    """Demonstrate tracking character development over time."""
    print("\n📈 CHARACTER DEVELOPMENT TRACKING")
    print("-" * 40)
    
    # Create a character
    character = create_mock_character("Maya", curiosity=60, bravery=40, intelligence=70)
    
    print(f"Tracking development for: {character.name}")
    
    # Simulate various character actions over time
    development_sequence = [
        ("study", {"subject": "ancient_history"}, {"knowledge_gained": 15}),
        ("experiment", {"type": "alchemy"}, {"discovery": "minor", "confidence": 5}),
        ("investigate", {"mystery": "village_legend"}, {"clues_found": 3}),
        ("explore", {"location": "old_library"}, {"rare_books_found": True}),
        ("research", {"topic": "magical_artifacts"}, {"breakthrough": True})
    ]
    
    for i, (action, params, result) in enumerate(development_sequence):
        events = integration.monitor_character_action(character, action, params, result)
        print(f"Day {i+1}: {action} -> {len(events)} story events")
        
        # Simulate character growth
        character.intelligence += 2
        character.curiosity += 3
        if i >= 2:  # Gain bravery after a few successful actions
            character.bravery += 4
    
    # Get comprehensive character story
    status = integration.get_character_story_status(character)
    print(f"Final character status:")
    print(f"  - Total story events: {status['story_summary']['total_story_events']}")
    print(f"  - Active storylines: {len(status['active_threads'])}")
    print(f"  - Recent actions: {len(status['recent_actions'])}")
    
    # Show attribute progression
    print(f"  - Final stats: Intelligence {character.intelligence}, "
          f"Curiosity {character.curiosity}, Bravery {character.bravery}")


def demonstrate_village_narrative_overview(integration: StorytellingGameIntegration):
    """Show overall village narrative status."""
    print("\n🏘️  VILLAGE NARRATIVE OVERVIEW")
    print("-" * 40)
    
    # Get village-wide narrative status
    village_status = integration.get_village_narrative_status()
    
    print("Village Storytelling Statistics:")
    print(f"  - Total events: {village_status['event_statistics']['total_events']}")
    print(f"  - Active storylines: {village_status['active_storylines']}")
    print(f"  - Recent world changes: {village_status['recent_world_changes']}")
    
    narrative_summary = village_status['narrative_summary']
    print(f"  - Story events (last 7 days): {narrative_summary['total_story_events']}")
    print(f"  - Narrative momentum: {narrative_summary['narrative_momentum']:.2f}")
    
    if narrative_summary['story_themes']:
        print("  - Active story themes:")
        for theme, count in narrative_summary['story_themes'].items():
            print(f"    * {theme}: {count} events")
    
    if narrative_summary['character_involvement']:
        print("  - Character involvement:")
        for character, count in list(narrative_summary['character_involvement'].items())[:5]:
            print(f"    * {character}: {count} events")



# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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


def main():
    """Run the complete storytelling demonstration."""
    print("🎭 EVENT-DRIVEN STORYTELLING DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how the storytelling engine creates dynamic")
    print("narratives based on character actions and world events.")
    print("=" * 60)
    
    # Create the integration
    integration = StorytellingGameIntegration()
    
    # Run different storyline demonstrations
    demonstrate_romance_storyline(integration)
    demonstrate_hero_journey(integration)
    demonstrate_village_crisis(integration)
    demonstrate_friendship_development(integration)
    demonstrate_character_development_tracking(integration)
    demonstrate_village_narrative_overview(integration)
    
    print("\n✨ STORYTELLING DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The event-driven storytelling system successfully created")
    print("dynamic narratives based on character actions and world events,")
    print("demonstrating how AI characters can experience meaningful")
    print("story arcs that emerge from their autonomous behaviors.")


if __name__ == "__main__":
    main()
    demonstrate_storytelling_system()

