#!/usr/bin/env python3
"""
Event-Driven Storytelling Engine for TinyVillage

This module enhances the existing event system with narrative-focused components
that create meaningful story experiences through character actions and world changes.

Key Features:
- Character action monitoring for story triggers
- World state change detection for narrative events
- Story-focused event templates and mini-storyline chains
- Narrative context and impact tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random

from tiny_event_handler import Event, EventHandler


class StoryEventType(Enum):
    """Categories of story events for narrative organization."""
    ROMANCE = "romance"
    CONFLICT = "conflict"
    ADVENTURE = "adventure"
    MYSTERY = "mystery"
    TRAGEDY = "tragedy"
    COMEDY = "comedy"
    DISCOVERY = "discovery"
    FRIENDSHIP = "friendship"
    BETRAYAL = "betrayal"
    REDEMPTION = "redemption"
    COMING_OF_AGE = "coming_of_age"
    HEROIC_JOURNEY = "heroic_journey"


class NarrativeImpact(Enum):
    """Levels of narrative significance for story events."""
    MINOR = 1          # Small character moments
    MODERATE = 2       # Notable events affecting one character
    SIGNIFICANT = 3    # Events affecting multiple characters
    MAJOR = 4         # Village-wide or life-changing events
    LEGENDARY = 5     # Epic events that become village lore


@dataclass
class StoryContext:
    """Narrative context for events to enable coherent storytelling."""
    theme: StoryEventType
    narrative_impact: NarrativeImpact
    character_arcs: Dict[str, str] = field(default_factory=dict)  # character_id -> arc_type
    plot_threads: List[str] = field(default_factory=list)  # ongoing story threads
    emotional_tone: str = "neutral"  # emotional atmosphere
    setting_details: Dict[str, Any] = field(default_factory=dict)
    foreshadowing: List[str] = field(default_factory=list)  # hints for future events
    callbacks: List[str] = field(default_factory=list)  # references to past events


@dataclass
class CharacterActionTrigger:
    """Defines conditions for triggering story events based on character actions."""
    action_names: Set[str]
    character_conditions: Dict[str, Any] = field(default_factory=dict)
    relationship_conditions: Dict[str, Any] = field(default_factory=dict)
    location_conditions: Dict[str, Any] = field(default_factory=dict)
    world_state_conditions: Dict[str, Any] = field(default_factory=dict)
    cooldown_hours: int = 24  # minimum time between triggers
    max_triggers: int = -1    # max times this trigger can fire (-1 = unlimited)
    probability: float = 1.0  # chance of triggering when conditions are met


@dataclass
class WorldStateChange:
    """Represents a significant change in the world state."""
    change_type: str
    affected_entities: List[Any]
    magnitude: float
    timestamp: datetime
    description: str
    related_actions: List[str] = field(default_factory=list)


class StorytellingEventHandler(EventHandler):
    """Enhanced EventHandler with narrative storytelling capabilities."""
    
    def __init__(self, graph_manager=None, time_manager=None):
        super().__init__(graph_manager, time_manager)
        
        # Story-specific tracking
        self.character_action_monitors: Dict[str, CharacterActionTrigger] = {}
        self.world_state_history: List[WorldStateChange] = []
        self.active_story_threads: Dict[str, Dict[str, Any]] = {}
        self.character_story_arcs: Dict[str, List[str]] = {}
        self.narrative_memory: Dict[str, Any] = {}
        
        # Cooldown tracking
        self.trigger_cooldowns: Dict[str, datetime] = {}
        self.trigger_counts: Dict[str, int] = {}
        
        # Storytelling system bridge (set by GameplayController)
        self.storytelling_system: Optional[Any] = None
        
        # Initialize story templates
        self._setup_story_templates()
        
        logging.info("StorytellingEventHandler initialized with narrative capabilities")
    
    def add_event(self, event):
        """Add an event and forward it to the storytelling system if available."""
        # Call parent method to handle standard event processing
        super().add_event(event)
        
        # Forward to storytelling system for narrative processing
        if hasattr(self, 'storytelling_system') and self.storytelling_system:
            try:
                self.storytelling_system.process_event_for_stories(event)
                logging.debug(f"Forwarded event '{event.name}' to storytelling system for narrative processing")
            except Exception as e:
                logging.error(f"Error forwarding event to storytelling system: {e}")
    
    def add_character_action_monitor(self, trigger_id: str, trigger: CharacterActionTrigger):
        """Add a monitor for character actions that can trigger story events."""
        self.character_action_monitors[trigger_id] = trigger
        self.trigger_counts[trigger_id] = 0
        logging.info(f"Added character action monitor: {trigger_id}")
    
    def monitor_character_action(self, character, action_name: str, action_result: Dict[str, Any]):
        """Monitor a character action and check for story event triggers."""
        triggered_events = []
        
        for trigger_id, trigger in self.character_action_monitors.items():
            if self._should_trigger_story_event(trigger_id, trigger, character, action_name, action_result):
                story_event = self._create_triggered_story_event(trigger_id, trigger, character, action_name, action_result)
                if story_event:
                    self.add_event(story_event)
                    triggered_events.append(story_event)
                    self._update_trigger_tracking(trigger_id)
        
        return triggered_events
    
    def _should_trigger_story_event(self, trigger_id: str, trigger: CharacterActionTrigger, 
                                  character, action_name: str, action_result: Dict[str, Any]) -> bool:
        """Check if a story event should be triggered based on the conditions."""
        
        # Check if action matches
        if action_name not in trigger.action_names:
            return False
        
        # Check cooldown
        if trigger_id in self.trigger_cooldowns:
            if datetime.now() - self.trigger_cooldowns[trigger_id] < timedelta(hours=trigger.cooldown_hours):
                return False
        
        # Check max triggers
        if trigger.max_triggers > 0 and self.trigger_counts.get(trigger_id, 0) >= trigger.max_triggers:
            return False
        
        # Check probability
        if random.random() > trigger.probability:
            return False
        
        # Check character conditions
        if not self._check_character_conditions(character, trigger.character_conditions):
            return False
        
        # Check relationship conditions
        if not self._check_relationship_conditions(character, trigger.relationship_conditions):
            return False
        
        # Check location conditions
        if not self._check_location_conditions(character, trigger.location_conditions):
            return False
        
        # Check world state conditions
        if not self._check_world_state_conditions(trigger.world_state_conditions):
            return False
        
        return True
    
    def _check_character_conditions(self, character, conditions: Dict[str, Any]) -> bool:
        """Check if character meets the specified conditions."""
        if not conditions:
            return True
        
        for attribute, expected_value in conditions.items():
            if hasattr(character, attribute):
                actual_value = getattr(character, attribute)
                if isinstance(expected_value, dict) and "operator" in expected_value:
                    # Handle complex conditions like {"operator": ">=", "value": 50}
                    operator = expected_value["operator"]
                    threshold = expected_value["value"]
                    if not self._evaluate_condition(actual_value, operator, threshold):
                        return False
                else:
                    # Handle direct value comparison
                    if actual_value != expected_value:
                        return False
            else:
                # If character doesn't have the attribute, condition fails
                return False
        
        return True
    
    def _check_relationship_conditions(self, character, conditions: Dict[str, Any]) -> bool:
        """Check relationship-based conditions using the graph manager."""
        if not conditions or not self.graph_manager:
            return True
        
        # This would need access to relationship data from the graph
        # For now, return True as a placeholder
        return True
    
    def _check_location_conditions(self, character, conditions: Dict[str, Any]) -> bool:
        """Check location-based conditions."""
        if not conditions:
            return True
        
        if hasattr(character, 'location'):
            location = character.location
            for condition_type, condition_value in conditions.items():
                if condition_type == "location_name" and hasattr(location, 'name'):
                    if location.name != condition_value:
                        return False
                elif condition_type == "location_type" and hasattr(location, 'type'):
                    if location.type != condition_value:
                        return False
        
        return True
    
    def _check_world_state_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check global world state conditions."""
        if not conditions:
            return True
        
        # This would check global state like time of day, weather, village statistics, etc.
        # For now, return True as a placeholder
        return True
    
    def _evaluate_condition(self, value, operator: str, threshold) -> bool:
        """Evaluate a condition with the given operator."""
        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return True
    
    def _create_triggered_story_event(self, trigger_id: str, trigger: CharacterActionTrigger,
                                    character, action_name: str, action_result: Dict[str, Any]) -> Optional[Event]:
        """Create a story event based on the triggered action."""
        
        # This would use story templates and context to create meaningful narrative events
        # For now, create a basic story event
        event_name = f"Story Event: {character.name if hasattr(character, 'name') else 'Character'} {action_name}"
        
        story_context = StoryContext(
            theme=random.choice(list(StoryEventType)),
            narrative_impact=NarrativeImpact.MODERATE,
            character_arcs={str(character): "development"},
            emotional_tone="engaging"
        )
        
        effects = [
            {
                "type": "attribute_change",
                "targets": ["participants"],
                "attribute": "story_experience",
                "change_value": 5
            }
        ]
        
        story_event = Event(
            name=event_name,
            date=datetime.now(),
            event_type="story",
            importance=story_context.narrative_impact.value,
            impact=3,
            participants=[character],
            effects=effects,
            # Store story context in event metadata
            preconditions=[{
                "type": "story_context",
                "story_context": story_context.__dict__
            }]
        )
        
        return story_event
    
    def _update_trigger_tracking(self, trigger_id: str):
        """Update tracking data for the triggered event."""
        self.trigger_cooldowns[trigger_id] = datetime.now()
        self.trigger_counts[trigger_id] = self.trigger_counts.get(trigger_id, 0) + 1
    
    def track_world_state_change(self, change: WorldStateChange):
        """Track significant world state changes that might trigger narrative events."""
        self.world_state_history.append(change)
        
        # Check if this change should trigger story events
        triggered_events = self._check_world_change_triggers(change)
        for event in triggered_events:
            self.add_event(event)
        
        # Keep history manageable
        if len(self.world_state_history) > 1000:
            self.world_state_history = self.world_state_history[-500:]
    
    def _check_world_change_triggers(self, change: WorldStateChange) -> List[Event]:
        """Check if a world state change should trigger narrative events."""
        triggered_events = []
        
        # Example: if a major economic change occurs, trigger related story events
        if change.change_type == "economic" and change.magnitude > 50:
            economic_story = self.create_economic_story_event(change)
            if economic_story:
                triggered_events.append(economic_story)
        
        return triggered_events
    
    def create_story_chain(self, chain_name: str, events: List[Dict[str, Any]], 
                          characters: List[Any] = None) -> List[Event]:
        """Create a chain of connected story events that form a mini-storyline."""
        story_events = []
        
        for i, event_def in enumerate(events):
            event_name = f"{chain_name}: Part {i+1}"
            
            # Calculate delay for chained events
            delay_hours = event_def.get("delay_hours", i * 24)  # Default: one event per day
            event_date = datetime.now() + timedelta(hours=delay_hours)
            
            story_event = Event(
                name=event_name,
                date=event_date,
                event_type="story_chain",
                importance=event_def.get("importance", 5),
                impact=event_def.get("impact", 3),
                participants=characters or [],
                effects=event_def.get("effects", []),
                preconditions=event_def.get("preconditions", []),
                cascading_events=event_def.get("cascading_events", [])
            )
            
            story_events.append(story_event)
            self.add_event(story_event)
        
        # Track this story chain
        self.active_story_threads[chain_name] = {
            "events": [e.name for e in story_events],
            "characters": [str(c) for c in (characters or [])],
            "started_at": datetime.now(),
            "status": "active"
        }
        
        logging.info(f"Created story chain '{chain_name}' with {len(story_events)} events")
        return story_events
    
    def _setup_story_templates(self):
        """Set up predefined story event templates for different narrative scenarios."""
        
        # Romance storyline templates
        self.add_story_template("meet_cute", {
            "type": "romance",
            "importance": 6,
            "impact": 4,
            "effects": [
                {
                    "type": "relationship_change",
                    "targets": ["participants"],
                    "attribute": "romantic_interest",
                    "change_value": 10
                }
            ],
            "cascading_events": [
                {
                    "name": "first_conversation",
                    "delay": 2,
                    "effects": [
                        {
                            "type": "attribute_change",
                            "targets": ["participants"],
                            "attribute": "social_confidence",
                            "change_value": 5
                        }
                    ]
                }
            ]
        })
        
        # Adventure storyline templates
        self.add_story_template("mysterious_discovery", {
            "type": "adventure",
            "importance": 7,
            "impact": 5,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "curiosity",
                    "change_value": 15
                }
            ],
            "cascading_events": [
                {
                    "name": "investigation_begins",
                    "delay": 6,
                    "effects": [
                        {
                            "type": "attribute_change",
                            "targets": ["participants"],
                            "attribute": "determination",
                            "change_value": 10
                        }
                    ]
                }
            ]
        })
        
        # Conflict storyline templates
        self.add_story_template("brewing_tension", {
            "type": "conflict",
            "importance": 5,
            "impact": -2,
            "effects": [
                {
                    "type": "relationship_change",
                    "targets": ["participants"],
                    "attribute": "tension",
                    "change_value": 15
                }
            ],
            "cascading_events": [
                {
                    "name": "confrontation",
                    "delay": 12,
                    "effects": [
                        {
                            "type": "attribute_change",
                            "targets": ["participants"],
                            "attribute": "stress",
                            "change_value": 20
                        }
                    ]
                }
            ]
        })
    
    def add_story_template(self, template_name: str, template_data: Dict[str, Any]):
        """Add a new story template for event creation."""
        if not hasattr(self, 'story_templates'):
            self.story_templates = {}
        self.story_templates[template_name] = template_data
    
    def create_story_event_from_template(self, template_name: str, event_name: str, 
                                       characters: List[Any], **overrides) -> Optional[Event]:
        """Create a story event from a predefined template."""
        if not hasattr(self, 'story_templates') or template_name not in self.story_templates:
            logging.error(f"Story template '{template_name}' not found")
            return None
        
        template = self.story_templates[template_name].copy()
        template.update(overrides)
        
        # Calculate importance with scaling for narrative impact
        importance = template.get("importance", 5)
        
        # Check if we have narrative impact context to scale importance
        # NarrativeImpact.MAJOR (4) should map to importance >= 6 for arc creation
        if hasattr(self, '_current_narrative_context'):
            narrative_impact = getattr(self._current_narrative_context, 'narrative_impact', None)
            if narrative_impact:
                # Scale: MAJOR (4) * 2 = 8, ensuring >= 6 threshold is met
                scaled_importance = narrative_impact.value * 2
                importance = max(importance, scaled_importance)
        
        return Event(
            name=event_name,
            date=datetime.now(),
            event_type="story",
            importance=importance,
            impact=template.get("impact", 3),
            participants=characters,
            effects=template.get("effects", []),
            preconditions=template.get("preconditions", []),
            cascading_events=template.get("cascading_events", [])
        )
    
    def create_economic_story_event(self, change: WorldStateChange) -> Optional[Event]:
        """Create a story event based on economic changes."""
        if change.magnitude > 75:
            return Event(
                name="Economic Upheaval",
                date=datetime.now(),
                event_type="economic_crisis",
                importance=8,
                impact=-5,
                effects=[
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "economic_anxiety",
                        "change_value": 25
                    }
                ]
            )
        elif change.magnitude > 50:
            return Event(
                name="Market Changes",
                date=datetime.now(),
                event_type="economic_change",
                importance=5,
                impact=2,
                effects=[
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "economic_awareness",
                        "change_value": 10
                    }
                ]
            )
        return None
    
    def get_active_story_threads(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active story threads."""
        return self.active_story_threads.copy()
    
    def complete_story_thread(self, thread_name: str):
        """Mark a story thread as completed."""
        if thread_name in self.active_story_threads:
            self.active_story_threads[thread_name]["status"] = "completed"
            self.active_story_threads[thread_name]["completed_at"] = datetime.now()
    
    def get_character_story_summary(self, character) -> Dict[str, Any]:
        """Get a summary of story events involving a specific character."""
        character_id = str(character)
        
        story_events = []
        for event in self.events:
            if character in event.participants:
                story_events.append({
                    "name": event.name,
                    "date": event.date,
                    "type": event.type,
                    "importance": event.importance,
                    "impact": event.impact
                })
        
        return {
            "character": character_id,
            "total_story_events": len(story_events),
            "events": story_events,
            "active_threads": [
                name for name, thread in self.active_story_threads.items()
                if character_id in thread.get("characters", []) and thread["status"] == "active"
            ]
        }
    
    def generate_narrative_summary(self, timeframe_days: int = 7) -> Dict[str, Any]:
        """Generate a narrative summary of recent story events."""
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        
        recent_events = [
            event for event in self.processed_events
            if event.last_triggered and event.last_triggered > cutoff_date
        ]
        
        story_themes = {}
        character_involvement = {}
        
        for event in recent_events:
            if hasattr(event, 'type') and event.type == "story":
                # Count story themes
                theme = getattr(event, 'theme', 'general')
                story_themes[theme] = story_themes.get(theme, 0) + 1
                
                # Count character involvement
                for participant in event.participants:
                    char_id = str(participant)
                    character_involvement[char_id] = character_involvement.get(char_id, 0) + 1
        
        return {
            "timeframe_days": timeframe_days,
            "total_story_events": len(recent_events),
            "story_themes": story_themes,
            "character_involvement": character_involvement,
            "active_threads": len([t for t in self.active_story_threads.values() if t["status"] == "active"]),
            "narrative_momentum": sum(story_themes.values()) / max(timeframe_days, 1)
        }