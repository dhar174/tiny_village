#!/usr/bin/env python3
"""
Integration module for Event-Driven Storytelling with TinyVillage game systems.

This module demonstrates how to integrate the storytelling engine with the existing
character actions, world state changes, and game events to create dynamic narratives.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from tiny_storytelling_engine import (
    StorytellingEventHandler,
    CharacterActionTrigger,
    WorldStateChange,
    StoryEventType,
    NarrativeImpact
)
from tiny_event_handler import Event


class StorytellingGameIntegration:
    """
    Integration layer that connects the storytelling engine with game systems.
    
    This class monitors character actions and world changes, triggering appropriate
    story events to create dynamic narratives based on gameplay.
    """
    
    def __init__(self, graph_manager=None, time_manager=None):
        self.story_handler = StorytellingEventHandler(graph_manager, time_manager)
        self.action_history: List[Dict[str, Any]] = []
        self.world_state_baseline: Dict[str, Any] = {}
        
        # Setup default story triggers
        self._setup_default_story_triggers()
        
        logging.info("StorytellingGameIntegration initialized")
    
    def _setup_default_story_triggers(self):
        """Set up common story triggers for typical character actions."""
        
        # Romance story triggers
        romance_trigger = CharacterActionTrigger(
            action_names={"talk", "flirt", "compliment", "gift"},
            character_conditions={
                "happiness": {"operator": ">=", "value": 50},
                "energy": {"operator": ">=", "value": 30}
            },
            location_conditions={"location_name": "Village Square"},
            cooldown_hours=12,
            probability=0.7
        )
        self.story_handler.add_character_action_monitor("romance_trigger", romance_trigger)
        
        # Adventure story triggers
        adventure_trigger = CharacterActionTrigger(
            action_names={"explore", "investigate", "search", "venture"},
            character_conditions={
                "curiosity": {"operator": ">=", "value": 60},
                "bravery": {"operator": ">=", "value": 40}
            },
            cooldown_hours=24,
            probability=0.5
        )
        self.story_handler.add_character_action_monitor("adventure_trigger", adventure_trigger)
        
        # Conflict story triggers
        conflict_trigger = CharacterActionTrigger(
            action_names={"argue", "confront", "challenge", "disagree"},
            character_conditions={
                "stress": {"operator": ">=", "value": 70}
            },
            cooldown_hours=6,
            probability=0.8
        )
        self.story_handler.add_character_action_monitor("conflict_trigger", conflict_trigger)
        
        # Friendship story triggers
        friendship_trigger = CharacterActionTrigger(
            action_names={"help", "support", "assist", "collaborate"},
            character_conditions={
                "empathy": {"operator": ">=", "value": 50}
            },
            cooldown_hours=18,
            probability=0.6
        )
        self.story_handler.add_character_action_monitor("friendship_trigger", friendship_trigger)
        
        # Discovery story triggers
        discovery_trigger = CharacterActionTrigger(
            action_names={"study", "research", "experiment", "learn"},
            character_conditions={
                "intelligence": {"operator": ">=", "value": 65}
            },
            cooldown_hours=48,
            probability=0.4
        )
        self.story_handler.add_character_action_monitor("discovery_trigger", discovery_trigger)
    
    def monitor_character_action(self, character, action_name: str, 
                               action_params: Dict[str, Any] = None,
                               action_result: Dict[str, Any] = None) -> List[Event]:
        """
        Monitor a character action and trigger appropriate story events.
        
        Args:
            character: The character performing the action
            action_name: Name of the action being performed
            action_params: Parameters passed to the action
            action_result: Result/outcome of the action
        
        Returns:
            List of triggered story events
        """
        # Record the action
        action_record = {
            "character": character,
            "action_name": action_name,
            "timestamp": datetime.now(),
            "params": action_params or {},
            "result": action_result or {}
        }
        self.action_history.append(action_record)
        
        # Check for story triggers
        triggered_events = self.story_handler.monitor_character_action(
            character, action_name, action_result or {}
        )
        
        # Log any triggered events
        if triggered_events:
            character_name = getattr(character, 'name', 'Unknown')
            logging.info(f"Character {character_name} action '{action_name}' triggered {len(triggered_events)} story events")
        
        return triggered_events
    
    def track_world_state_change(self, change_type: str, affected_entities: List[Any],
                                magnitude: float, description: str,
                                related_actions: List[str] = None) -> List[Event]:
        """
        Track a significant world state change and trigger story events.
        
        Args:
            change_type: Type of change (economic, political, environmental, etc.)
            affected_entities: Entities affected by the change
            magnitude: Significance of the change (0-100)
            description: Human-readable description of the change
            related_actions: Actions that led to this change
        
        Returns:
            List of triggered story events
        """
        change = WorldStateChange(
            change_type=change_type,
            affected_entities=affected_entities,
            magnitude=magnitude,
            timestamp=datetime.now(),
            description=description,
            related_actions=related_actions or []
        )
        
        # Track the change and get any triggered events
        initial_event_count = len(self.story_handler.events)
        self.story_handler.track_world_state_change(change)
        final_event_count = len(self.story_handler.events)
        
        triggered_count = final_event_count - initial_event_count
        if triggered_count > 0:
            logging.info(f"World change '{description}' triggered {triggered_count} story events")
        
        # Return the newly added events
        return self.story_handler.events[-triggered_count:] if triggered_count > 0 else []
    
    def create_character_arc_storyline(self, character, arc_type: str,
                                     duration_days: int = 7) -> List[Event]:
        """
        Create a multi-part storyline for a character's development arc.
        
        Args:
            character: The character for the storyline
            arc_type: Type of character arc (romance, hero_journey, redemption, etc.)
            duration_days: How long the storyline should span
        
        Returns:
            List of story events in the arc
        """
        character_name = getattr(character, 'name', 'Character')
        
        if arc_type == "romance":
            return self._create_romance_arc(character, character_name, duration_days)
        elif arc_type == "hero_journey":
            return self._create_hero_journey_arc(character, character_name, duration_days)
        elif arc_type == "redemption":
            return self._create_redemption_arc(character, character_name, duration_days)
        elif arc_type == "friendship":
            return self._create_friendship_arc(character, character_name, duration_days)
        else:
            logging.warning(f"Unknown arc type: {arc_type}")
            return []
    
    def _create_romance_arc(self, character, character_name: str, duration_days: int) -> List[Event]:
        """Create a romance storyline arc."""
        romance_events = [
            {
                "delay_hours": 0,
                "importance": 5,
                "impact": 3,
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
                "delay_hours": duration_days * 24 // 4,  # 1/4 through
                "importance": 6,
                "impact": 4,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "emotional_connection",
                        "change_value": 15
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 2,  # Halfway through
                "importance": 7,
                "impact": 5,
                "effects": [
                    {
                        "type": "relationship_change",
                        "targets": ["participants"],
                        "attribute": "relationship_depth",
                        "change_value": 20
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24,  # End
                "importance": 8,
                "impact": 6,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "happiness",
                        "change_value": 25
                    }
                ]
            }
        ]
        
        return self.story_handler.create_story_chain(
            f"{character_name}_romance_arc",
            romance_events,
            [character]
        )
    
    def _create_hero_journey_arc(self, character, character_name: str, duration_days: int) -> List[Event]:
        """Create a hero's journey storyline arc."""
        hero_events = [
            {
                "delay_hours": 0,
                "importance": 6,
                "impact": 2,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "call_to_adventure",
                        "change_value": 20
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 6,  # Early challenge
                "importance": 7,
                "impact": -2,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "obstacle_faced",
                        "change_value": 15
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 3,  # Mentorship/growth
                "importance": 6,
                "impact": 4,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "wisdom_gained",
                        "change_value": 18
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 * 2 // 3,  # Major trial
                "importance": 9,
                "impact": -3,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "trial_endured",
                        "change_value": 25
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24,  # Triumph and return
                "importance": 10,
                "impact": 8,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "heroic_achievement",
                        "change_value": 30
                    }
                ]
            }
        ]
        
        return self.story_handler.create_story_chain(
            f"{character_name}_hero_journey",
            hero_events,
            [character]
        )
    
    def _create_redemption_arc(self, character, character_name: str, duration_days: int) -> List[Event]:
        """Create a redemption storyline arc."""
        redemption_events = [
            {
                "delay_hours": 0,
                "importance": 5,
                "impact": -3,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "guilt_realization",
                        "change_value": 20
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 4,
                "importance": 6,
                "impact": 1,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "repentance_begins",
                        "change_value": 15
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 2,
                "importance": 7,
                "impact": 3,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "making_amends",
                        "change_value": 18
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24,
                "importance": 8,
                "impact": 6,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "redemption_achieved",
                        "change_value": 25
                    }
                ]
            }
        ]
        
        return self.story_handler.create_story_chain(
            f"{character_name}_redemption_arc",
            redemption_events,
            [character]
        )
    
    def _create_friendship_arc(self, character, character_name: str, duration_days: int) -> List[Event]:
        """Create a friendship development storyline arc."""
        friendship_events = [
            {
                "delay_hours": 0,
                "importance": 4,
                "impact": 2,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "social_openness",
                        "change_value": 12
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 // 3,
                "importance": 5,
                "impact": 3,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "trust_building",
                        "change_value": 15
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24 * 2 // 3,
                "importance": 6,
                "impact": 4,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "shared_experiences",
                        "change_value": 18
                    }
                ]
            },
            {
                "delay_hours": duration_days * 24,
                "importance": 7,
                "impact": 5,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "deep_friendship",
                        "change_value": 22
                    }
                ]
            }
        ]
        
        return self.story_handler.create_story_chain(
            f"{character_name}_friendship_arc",
            friendship_events,
            [character]
        )
    
    def generate_daily_story_events(self) -> Dict[str, Any]:
        """Generate daily story events and return summary."""
        daily_results = self.story_handler.check_daily_events()
        
        # Process any pending cascading events
        cascading_processed = self.story_handler.process_cascading_queue()
        
        return {
            "daily_events": daily_results,
            "cascading_processed": cascading_processed,
            "story_summary": self.story_handler.generate_narrative_summary(1)
        }
    
    def get_character_story_status(self, character) -> Dict[str, Any]:
        """Get comprehensive story status for a character."""
        return {
            "story_summary": self.story_handler.get_character_story_summary(character),
            "active_threads": [
                name for name, thread in self.story_handler.get_active_story_threads().items()
                if str(character) in thread.get("characters", [])
            ],
            "recent_actions": [
                action for action in self.action_history[-10:]
                if action["character"] == character
            ]
        }
    
    def get_village_narrative_status(self) -> Dict[str, Any]:
        """Get overall village narrative status."""
        stats = self.story_handler.get_event_statistics()
        narrative_summary = self.story_handler.generate_narrative_summary(7)
        
        return {
            "event_statistics": stats,
            "narrative_summary": narrative_summary,
            "active_storylines": len([
                t for t in self.story_handler.get_active_story_threads().values()
                if t["status"] == "active"
            ]),
            "recent_world_changes": len(self.story_handler.world_state_history[-10:])
        }
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old action history and story data."""
        # Clean up action history
        cutoff_date = datetime.now() - timedelta(days=days_old)
        initial_action_count = len(self.action_history)
        self.action_history = [
            action for action in self.action_history
            if action["timestamp"] > cutoff_date
        ]
        
        # Clean up old events in story handler
        cleaned_events = self.story_handler.cleanup_old_events(days_old)
        
        logging.info(f"Cleaned up {initial_action_count - len(self.action_history)} old actions and {cleaned_events} old events")


# Utility functions for easy integration

def create_storytelling_integration(graph_manager=None, time_manager=None) -> StorytellingGameIntegration:
    """Create a new storytelling integration instance."""
    return StorytellingGameIntegration(graph_manager, time_manager)


def setup_character_for_storytelling(integration: StorytellingGameIntegration, 
                                   character, preferred_arc: str = None):
    """Set up a character for storytelling by creating an initial arc."""
    if preferred_arc:
        integration.create_character_arc_storyline(character, preferred_arc)
    
    logging.info(f"Character {getattr(character, 'name', 'Unknown')} set up for storytelling")


def monitor_action_with_storytelling(integration: StorytellingGameIntegration,
                                   character, action_name: str, 
                                   action_params: Dict[str, Any] = None,
                                   action_result: Dict[str, Any] = None) -> List[Event]:
    """Convenience function to monitor an action with storytelling."""
    return integration.monitor_character_action(character, action_name, action_params, action_result)


if __name__ == "__main__":
    # Example usage demonstration
    print("Event-Driven Storytelling Integration Example")
    print("=" * 50)
    
    # Create integration
    integration = create_storytelling_integration()
    
    # Create mock character with proper attributes
    from unittest.mock import Mock
    character = Mock()
    character.name = "TestCharacter"
    character.happiness = 75
    character.curiosity = 80
    character.bravery = 65
    character.energy = 85
    character.empathy = 70
    character.intelligence = 75
    character.stress = 30
    character.location = Mock()
    character.location.name = "Village Square"
    
    # Monitor some actions
    events1 = monitor_action_with_storytelling(
        integration, character, "explore", 
        {"target": "forest"}, {"success": True}
    )
    print(f"Exploration triggered {len(events1)} story events")
    
    # Create a character arc
    arc_events = integration.create_character_arc_storyline(character, "hero_journey", 5)
    print(f"Hero journey arc created with {len(arc_events)} events")
    
    # Track a world change
    world_events = integration.track_world_state_change(
        "environmental", [character], 85, "Mysterious forest changes detected"
    )
    print(f"World change triggered {len(world_events)} story events")
    
    # Get status
    status = integration.get_character_story_status(character)
    print(f"Character has {status['story_summary']['total_story_events']} total story events")
    
    village_status = integration.get_village_narrative_status()
    print(f"Village has {village_status['active_storylines']} active storylines")
    
    print("\nStorytelling integration example completed successfully!")