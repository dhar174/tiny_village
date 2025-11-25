"""
Event-Driven Storytelling System for Tiny Village

This module implements dynamic narrative generation from game events,
managing story arcs and ensuring narrative coherence.

Components:
- StoryArc: Represents ongoing narrative threads
- StoryArcManager: Manages and tracks multiple story arcs
- NarrativeGenerator: Converts events to story text
- StorytellingSystem: Main coordinator class
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Local imports
from tiny_event_handler import Event, EventHandler

logger = logging.getLogger(__name__)


class StoryArcType(Enum):
    """Types of story arcs supported by the system."""
    PERSONAL_GROWTH = "personal_growth"
    RELATIONSHIP = "relationship"
    COMMUNITY = "community"
    CONFLICT = "conflict"
    MYSTERY = "mystery"
    ECONOMIC = "economic"
    SEASONAL = "seasonal"


class StoryArcStatus(Enum):
    """Status of a story arc."""
    STARTING = "starting"
    DEVELOPING = "developing"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    COMPLETED = "completed"
    DORMANT = "dormant"


@dataclass
class StoryElement:
    """Individual element within a story arc."""
    event_name: str
    timestamp: datetime
    significance: int  # 1-10, how important to the arc
    narrative_text: str
    character_impact: Dict[str, Any]
    emotional_tone: str  # positive, negative, neutral, mixed


@dataclass
class StoryArc:
    """Represents an ongoing narrative thread."""
    arc_id: str
    arc_type: StoryArcType
    title: str
    description: str
    participants: List[str]  # Character names
    status: StoryArcStatus
    started_at: datetime
    last_updated: datetime
    elements: List[StoryElement]
    themes: List[str]
    importance: int  # 1-10
    expected_duration: Optional[timedelta] = None
    completion_conditions: Optional[Dict[str, Any]] = None
    
    def add_element(self, element: StoryElement):
        """Add a new story element to this arc."""
        self.elements.append(element)
        self.last_updated = datetime.now()
        
    def get_current_narrative(self) -> str:
        """Get the current narrative summary of this arc."""
        if not self.elements:
            return f"{self.title}: {self.description}"
        
        recent_elements = sorted(self.elements, key=lambda x: x.timestamp)[-3:]
        narrative_parts = [self.description]
        
        for element in recent_elements:
            narrative_parts.append(element.narrative_text)
            
        return " ".join(narrative_parts)
    
    def should_advance_status(self) -> bool:
        """Check if the arc status should advance based on elements."""
        if self.status == StoryArcStatus.COMPLETED:
            return False
            
        element_count = len(self.elements)
        
        # Simple progression logic using thresholds
        if self.status == StoryArcStatus.STARTING and element_count >= self.STARTING_THRESHOLD:
            return True
        elif self.status == StoryArcStatus.DEVELOPING and element_count >= self.DEVELOPING_THRESHOLD:
            return True
        elif self.status == StoryArcStatus.CLIMAX and element_count >= self.CLIMAX_THRESHOLD:
            return True
            
        return False
    
    def advance_status(self):
        """Advance the story arc to the next status."""
        if self.status == StoryArcStatus.STARTING:
            self.status = StoryArcStatus.DEVELOPING
        elif self.status == StoryArcStatus.DEVELOPING:
            self.status = StoryArcStatus.CLIMAX
        elif self.status == StoryArcStatus.CLIMAX:
            self.status = StoryArcStatus.RESOLUTION
        elif self.status == StoryArcStatus.RESOLUTION:
            self.status = StoryArcStatus.COMPLETED
            
        logger.info(f"Story arc '{self.title}' advanced to {self.status.value}")


# Define story arc progression thresholds
StoryArc.STARTING_THRESHOLD = 3
StoryArc.DEVELOPING_THRESHOLD = 6  
StoryArc.CLIMAX_THRESHOLD = 9


class StoryArcManager:
    """Manages multiple story arcs and their relationships."""
    
    def __init__(self):
        self.active_arcs: Dict[str, StoryArc] = {}
        self.completed_arcs: Dict[str, StoryArc] = {}
        self.arc_templates = self._initialize_arc_templates()
        self.next_arc_id = 1
        
    def _initialize_arc_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined story arc templates."""
        return {
            "character_development": {
                "type": StoryArcType.PERSONAL_GROWTH,
                "themes": ["growth", "learning", "challenge"],
                "expected_duration": timedelta(days=30),
                "completion_conditions": {"character_attribute_change": 20}
            },
            "village_romance": {
                "type": StoryArcType.RELATIONSHIP,
                "themes": ["love", "relationships", "community"],
                "expected_duration": timedelta(days=60),
                "completion_conditions": {"relationship_strength": 80}
            },
            "community_project": {
                "type": StoryArcType.COMMUNITY,
                "themes": ["cooperation", "building", "progress"],
                "expected_duration": timedelta(days=45),
                "completion_conditions": {"project_completion": True}
            },
            "merchant_rivalry": {
                "type": StoryArcType.CONFLICT,
                "themes": ["competition", "trade", "resolution"],
                "expected_duration": timedelta(days=21),
                "completion_conditions": {"conflict_resolved": True}
            },
            "seasonal_celebration": {
                "type": StoryArcType.SEASONAL,
                "themes": ["tradition", "celebration", "community"],
                "expected_duration": timedelta(days=14),
                "completion_conditions": {"event_completion": True}
            }
        }
    
    def create_arc_from_event(self, event: Event) -> Optional[StoryArc]:
        """Create a new story arc based on an event."""
        template_name = self._determine_arc_template(event)
        if not template_name:
            return None
            
        template = self.arc_templates[template_name]
        arc_id = f"arc_{self.next_arc_id:04d}"
        self.next_arc_id += 1
        
        arc = StoryArc(
            arc_id=arc_id,
            arc_type=template["type"],
            title=self._generate_arc_title(event, template),
            description=self._generate_arc_description(event, template),
            participants=self._extract_participants(event),
            status=StoryArcStatus.STARTING,
            started_at=datetime.now(),
            last_updated=datetime.now(),
            elements=[],
            themes=template["themes"].copy(),
            importance=max(event.importance, 5),
            expected_duration=template.get("expected_duration"),
            completion_conditions=template.get("completion_conditions", {}).copy()
        )
        
        self.active_arcs[arc_id] = arc
        logger.info(f"Created new story arc: {arc.title} ({arc_id})")
        return arc
    
    def _determine_arc_template(self, event: Event) -> Optional[str]:
        """Determine which story arc template to use for an event."""
        event_type = event.type.lower()
        
        # Check for story events with romance context
        if event_type == "story":
            # Look for romance indicators in the event name or context
            event_name = event.name.lower()
            if "meet" in event_name and "cute" in event_name:
                return "village_romance"
            # Check if event has romance-related effects
            if hasattr(event, 'effects'):
                for effect in event.effects:
                    if effect.get('attribute') == 'romantic_interest':
                        return "village_romance"
        
        if event_type in ["social", "celebration", "holiday"]:
            if event.importance >= 8:
                return "seasonal_celebration"
            else:
                return "character_development"
        elif event_type in ["economic", "trade"]:
            return "merchant_rivalry"
        elif event_type in ["work", "building"]:
            return "community_project"
        elif event_type in ["meeting", "gathering", "romance"]:
            return "village_romance"
        else:
            return "character_development"  # Default
    
    def _generate_arc_title(self, event: Event, template: Dict[str, Any]) -> str:
        """Generate a title for the story arc."""
        arc_type = template["type"]
        
        if arc_type == StoryArcType.PERSONAL_GROWTH:
            return f"The Journey of {event.participants[0].name if event.participants and hasattr(event.participants[0], 'name') else 'Unknown'}"
        elif arc_type == StoryArcType.RELATIONSHIP:
            return f"A Growing Bond"
        elif arc_type == StoryArcType.COMMUNITY:
            return f"The {event.name} Project"
        elif arc_type == StoryArcType.CONFLICT:
            return f"The {event.name} Rivalry"
        elif arc_type == StoryArcType.SEASONAL:
            return f"The {event.name} Chronicles"
        else:
            return f"The Story of {event.name}"
    
    def _generate_arc_description(self, event: Event, template: Dict[str, Any]) -> str:
        """Generate a description for the story arc."""
        participants_str = ", ".join([
            p.name if hasattr(p, 'name') else str(p) 
            for p in event.participants[:3]
        ])
        
        if len(event.participants) > 3:
            participants_str += " and others"
        
        themes_str = ", ".join(template["themes"][:2])
        
        return (f"A story of {themes_str} involving {participants_str}, "
                f"beginning with the {event.name} and unfolding over time.")
    
    def _extract_participants(self, event: Event) -> List[str]:
        """Extract participant names from an event."""
        participants = []
        for p in event.participants:
            if hasattr(p, 'name'):
                participants.append(p.name)
            else:
                participants.append(str(p))
        return participants
    
    def find_relevant_arcs(self, event: Event) -> List[StoryArc]:
        """Find existing story arcs that are relevant to an event."""
        relevant_arcs = []
        event_participants = set(self._extract_participants(event))
        
        for arc in self.active_arcs.values():
            arc_participants = set(arc.participants)
            
            # Check for participant overlap
            if event_participants & arc_participants:
                relevant_arcs.append(arc)
                continue
                
            # Check for thematic relevance
            if self._is_thematically_relevant(event, arc):
                relevant_arcs.append(arc)
        
        return sorted(relevant_arcs, key=lambda a: a.importance, reverse=True)
    
    def _is_thematically_relevant(self, event: Event, arc: StoryArc) -> bool:
        """Check if an event is thematically relevant to a story arc."""
        event_themes = self._extract_event_themes(event)
        return bool(set(event_themes) & set(arc.themes))
    
    def _extract_event_themes(self, event: Event) -> List[str]:
        """Extract themes from an event."""
        event_type = event.type.lower()
        themes = []
        
        if event_type in ["social", "celebration"]:
            themes.extend(["community", "relationships"])
        elif event_type in ["economic", "trade"]:
            themes.extend(["trade", "progress"])
        elif event_type in ["work", "building"]:
            themes.extend(["building", "cooperation"])
        elif event_type == "crisis":
            themes.extend(["challenge", "survival"])
            
        return themes
    
    def update_arcs(self):
        """Update all active arcs, advancing their status as needed."""
        arcs_to_complete = []
        
        for arc_id, arc in self.active_arcs.items():
            if arc.should_advance_status():
                arc.advance_status()
                
            if arc.status == StoryArcStatus.COMPLETED:
                arcs_to_complete.append(arc_id)
        
        # Move completed arcs
        for arc_id in arcs_to_complete:
            arc = self.active_arcs.pop(arc_id)
            self.completed_arcs[arc_id] = arc
            logger.info(f"Completed story arc: {arc.title}")
    
    def get_active_narratives(self) -> List[str]:
        """Get current narrative summaries for all active arcs."""
        narratives = []
        for arc in sorted(self.active_arcs.values(), key=lambda a: a.importance, reverse=True):
            narratives.append(arc.get_current_narrative())
        return narratives
    
    def get_arc_statistics(self) -> Dict[str, Any]:
        """Get statistics about story arcs."""
        return {
            "active_arcs": len(self.active_arcs),
            "completed_arcs": len(self.completed_arcs),
            "arc_types": {arc_type.value: len([a for a in self.active_arcs.values() if a.arc_type == arc_type]) 
                         for arc_type in StoryArcType},
            "total_story_elements": sum(len(arc.elements) for arc in self.active_arcs.values())
        }


class NarrativeGenerator:
    """Converts events to narrative text and story elements."""
    
    def __init__(self):
        self.narrative_templates = self._initialize_narrative_templates()
        self.emotional_tone_patterns = self._initialize_emotional_patterns()
    
    def _initialize_narrative_templates(self) -> Dict[str, List[str]]:
        """Initialize narrative templates for different event types."""
        return {
            "social": [
                "The village gathered as {participants} participated in {event_name}, bringing the community closer together.",
                "A sense of joy filled the air when {participants} joined {event_name}, creating lasting memories.",
                "The bonds between villagers grew stronger during {event_name}, with {participants} at the center of it all."
            ],
            "economic": [
                "Trade flourished when {participants} engaged in {event_name}, bringing prosperity to the village.",
                "The marketplace buzzed with activity as {participants} participated in {event_name}.",
                "Economic opportunities arose from {event_name}, with {participants} leading the way."
            ],
            "work": [
                "Progress was made on {event_name} as {participants} worked together toward a common goal.",
                "The sound of productive work filled the air during {event_name}, with {participants} contributing their skills.",
                "Through determination and cooperation, {participants} advanced {event_name} for the benefit of all."
            ],
            "crisis": [
                "Challenges arose during {event_name}, testing the resolve of {participants}.",
                "The village faced difficulties with {event_name}, but {participants} stood ready to help.",
                "In times of hardship like {event_name}, the true character of {participants} shone through."
            ],
            "celebration": [
                "Joy and merriment filled the village as {participants} celebrated {event_name}.",
                "The festive spirit of {event_name} brought {participants} together in celebration.",
                "Laughter and happiness marked {event_name}, with {participants} at the heart of the festivities."
            ]
        }
    
    def _initialize_emotional_patterns(self) -> Dict[str, str]:
        """Initialize emotional tone patterns for events."""
        return {
            "social": "positive",
            "celebration": "positive", 
            "holiday": "positive",
            "economic": "neutral",
            "work": "neutral",
            "crisis": "negative",
            "weather": "mixed"
        }
    
    def generate_story_element(self, event: Event, arc: StoryArc) -> StoryElement:
        """Generate a story element from an event within a story arc context."""
        narrative_text = self._generate_narrative_text(event, arc)
        emotional_tone = self._determine_emotional_tone(event)
        significance = self._calculate_significance(event, arc)
        character_impact = self._calculate_character_impact(event)
        
        return StoryElement(
            event_name=event.name,
            timestamp=datetime.now(),
            significance=significance,
            narrative_text=narrative_text,
            character_impact=character_impact,
            emotional_tone=emotional_tone
        )
    
    def _generate_narrative_text(self, event: Event, arc: StoryArc) -> str:
        """Generate narrative text for an event."""
        event_type = event.type.lower()
        templates = self.narrative_templates.get(event_type, self.narrative_templates["social"])
        
        # Choose template based on arc status
        template_index = min(len(templates) - 1, len(arc.elements) % len(templates))
        template = templates[template_index]
        
        # Format participants
        participants_text = self._format_participants(event.participants)
        
        return template.format(
            participants=participants_text,
            event_name=event.name
        )
    
    def _format_participants(self, participants) -> str:
        """Format participant list for narrative text."""
        if not participants:
            return "the villagers"
        
        names = []
        for p in participants[:3]:
            if hasattr(p, 'name'):
                names.append(p.name)
            else:
                names.append(str(p))
        
        if len(participants) > 3:
            names.append("others")
        
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return f"{', '.join(names[:-1])}, and {names[-1]}"
    
    def _determine_emotional_tone(self, event: Event) -> str:
        """Determine the emotional tone of an event."""
        event_type = event.type.lower()
        base_tone = self.emotional_tone_patterns.get(event_type, "neutral")
        
        # Modify based on event impact
        event_impact = getattr(event, 'impact', 0)
        if event_impact > 5:
            if base_tone == "neutral":
                base_tone = "positive"
        elif event_impact < -3:
            base_tone = "negative"
        
        return base_tone
    
    def _calculate_significance(self, event: Event, arc: StoryArc) -> int:
        """Calculate how significant an event is to a story arc."""
        base_significance = min(event.importance, 10)
        
        # Increase significance for pivotal arc moments
        if arc.status == StoryArcStatus.CLIMAX:
            base_significance = min(base_significance + 2, 10)
        
        return base_significance
    
    def _calculate_character_impact(self, event: Event) -> Dict[str, Any]:
        """Calculate the impact of an event on characters."""
        impact = {}
        
        for participant in event.participants:
            char_name = participant.name if hasattr(participant, 'name') else str(participant)
            impact[char_name] = {
                "emotional_change": event.impact,
                "relationship_changes": {},
                "skill_development": 1 if event.type == "work" else 0
            }
        
        return impact


class StorytellingSystem:
    """Main coordinator for the event-driven storytelling system."""
    
    def __init__(self, event_handler: EventHandler = None):
        self.event_handler = event_handler
        self.arc_manager = StoryArcManager()
        self.narrative_generator = NarrativeGenerator()
        self.story_cache = {}
        self.last_update = datetime.now()
        
        logger.info("Storytelling system initialized")
    
    def process_event_for_stories(self, event: Event) -> Dict[str, Any]:
        """Process an event for storytelling purposes."""
        results = {
            "new_arcs": [],
            "updated_arcs": [],
            "story_elements": [],
            "narratives": []
        }
        
        # Find relevant existing arcs
        relevant_arcs = self.arc_manager.find_relevant_arcs(event)
        
        if relevant_arcs:
            # Add to existing arcs
            for arc in relevant_arcs[:2]:  # Limit to top 2 most relevant
                story_element = self.narrative_generator.generate_story_element(event, arc)
                arc.add_element(story_element)
                results["updated_arcs"].append(arc.arc_id)
                results["story_elements"].append(story_element)
                results["narratives"].append(story_element.narrative_text)
        else:
            # Create new arc if event is significant enough
            if event.importance >= 6:
                new_arc = self.arc_manager.create_arc_from_event(event)
                if new_arc:
                    story_element = self.narrative_generator.generate_story_element(event, new_arc)
                    new_arc.add_element(story_element)
                    results["new_arcs"].append(new_arc.arc_id)
                    results["story_elements"].append(story_element)
                    results["narratives"].append(story_element.narrative_text)
        
        # Update all arcs
        self.arc_manager.update_arcs()
        self.last_update = datetime.now()
        
        logger.info(f"Processed event '{event.name}' for storytelling: "
                   f"{len(results['new_arcs'])} new arcs, "
                   f"{len(results['updated_arcs'])} updated arcs")
        
        return results
    
    def get_current_stories(self) -> Dict[str, Any]:
        """Get current story state and narratives."""
        return {
            "active_narratives": self.arc_manager.get_active_narratives(),
            "arc_statistics": self.arc_manager.get_arc_statistics(),
            "last_update": self.last_update.isoformat(),
            "total_active_arcs": len(self.arc_manager.active_arcs),
            "feature_status": "BASIC_IMPLEMENTED"
        }
    
    def generate_story_summary(self, days_back: int = 7) -> str:
        """Generate a summary of recent story developments."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_elements = []
        for arc in self.arc_manager.active_arcs.values():
            for element in arc.elements:
                if element.timestamp >= cutoff_date:
                    recent_elements.append((arc, element))
        
        if not recent_elements:
            return "The village has been quiet lately, with no significant story developments."
        
        # Sort by timestamp
        recent_elements.sort(key=lambda x: x[1].timestamp)
        
        summary_parts = [f"Story developments over the past {days_back} days:"]
        
        for arc, element in recent_elements:
            summary_parts.append(f"• {element.narrative_text}")
        
        return "\n".join(summary_parts)
    
    def get_character_story_involvement(self, character_name: str) -> Dict[str, Any]:
        """Get a character's involvement in current stories."""
        involvement = {
            "active_arcs": [],
            "total_story_elements": 0,
            "recent_developments": []
        }
        
        for arc in self.arc_manager.active_arcs.values():
            if character_name in arc.participants:
                involvement["active_arcs"].append({
                    "arc_id": arc.arc_id,
                    "title": arc.title,
                    "status": arc.status.value,
                    "role": arc.character_roles.get(character_name, CharacterRole.PARTICIPANT).value
                })
                
                # Count elements involving this character
                char_elements = [e for e in arc.elements if character_name in e.character_impact]
                involvement["total_story_elements"] += len(char_elements)
                
                # Get recent developments
                recent_elements = [e for e in char_elements if 
                                 e.timestamp >= datetime.now() - timedelta(days=3)]
                for element in recent_elements:
                    involvement["recent_developments"].append(element.narrative_text)
        
        return involvement