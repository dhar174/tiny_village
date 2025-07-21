"""
Story Arc Management System for Tiny Village

This module provides narrative progression tracking and story arc management
for the event-driven storytelling system.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging


class StoryArc:
    """
    Manages narrative progression and story phases for events and character interactions.
    
    Story arcs track the progression of narrative elements through different phases:
    - Setup (0.0 - 0.2): Introduction and establishing elements
    - Rising Action (0.2 - 0.6): Building tension and developing plot
    - Climax (0.6 - 0.9): Peak tension and critical events
    - Resolution (0.9 - 1.0): Conclusion and aftermath
    """
    
    # Threshold constants for story progression phases
    STARTING_THRESHOLD = 0.2
    DEVELOPING_THRESHOLD = 0.6
    CLIMAX_THRESHOLD = 0.9
    
    def __init__(self, name: str, importance: int = 5):
        """
        Initialize a new story arc.
        
        Args:
            name: Name/identifier for this story arc
            importance: Importance level (1-10) affecting progression speed
        """
        self.name = name
        self.importance = importance
        self._progression = 0.0
        self.phase = "setup"
        self.events = []
        self.characters_involved = set()
        self.start_time = datetime.now()
        self.completed = False
        self.resolution_type = None
    
    @property
    def progression(self) -> float:
        """Get the current progression value."""
        return self._progression
    
    @progression.setter
    def progression(self, value: float):
        """Set the progression value and update phase."""
        self._progression = max(0.0, min(1.0, value))
        self._update_phase()
        if self._progression >= 1.0:
            self.completed = True
        
    def get_current_phase(self) -> str:
        """
        Get the current narrative phase based on progression.
        
        Returns:
            str: Current phase ('setup', 'rising_action', 'climax', 'resolution')
        """
        if self.progression < self.STARTING_THRESHOLD:
            return "setup"
        elif self.progression < self.DEVELOPING_THRESHOLD:
            return "rising_action"
        elif self.progression < self.CLIMAX_THRESHOLD:
            return "climax"
        else:
            return "resolution"
    
    def _update_phase(self):
        """Update the current phase based on progression."""
        self.phase = self.get_current_phase()
    
    def advance_progression(self, amount: float = 0.1):
        """
        Advance the story progression by the specified amount.
        
        Args:
            amount: Amount to advance (0.0-1.0)
        """
        if not self.completed:
            old_phase = self.phase
            self.progression = self.progression + amount
            
            if old_phase != self.phase:
                logging.info(f"Story arc '{self.name}' advanced from {old_phase} to {self.phase}")
                
            if self.progression >= 1.0:
                logging.info(f"Story arc '{self.name}' completed")
    
    class EventProtocol(Protocol):
        """
        Protocol for event objects expected by the StoryArc class.
        """
        importance: int
    
    def add_event(self, event: EventProtocol):
        """
        Add an event to this story arc and potentially advance progression.
        
        Args:
            event: Event object to associate with this arc
        """
        if not isinstance(event, EventProtocol):
            raise TypeError(f"Event must conform to EventProtocol with an 'importance' attribute.")
        
        self.events.append(event)
        
        # Advance progression based on event importance
        progression_amount = (event.importance / 10.0) * 0.1
        self.advance_progression(progression_amount)
    
    def add_character(self, character: Union['Character', str]):
        """
        Add a character as involved in this story arc.
        
        Args:
            character: Character object or character name
        """
        if isinstance(character, Character):
            self.characters_involved.add(character.name)
        elif isinstance(character, str):
            self.characters_involved.add(character)
        else:
            raise TypeError("Invalid character type. Expected a Character object or string.")
    
    def get_progression_percentage(self) -> int:
        """
        Get progression as a percentage.
        
        Returns:
            int: Progression percentage (0-100)
        """
        return int(self.progression * 100)
    
    def is_in_phase(self, phase: str) -> bool:
        """
        Check if the story arc is currently in the specified phase.
        
        Args:
            phase: Phase to check ('setup', 'rising_action', 'climax', 'resolution')
            
        Returns:
            bool: True if in the specified phase
        """
        return self.phase == phase
    
    def get_phase_progress(self) -> float:
        """
        Get progress within the current phase (0.0-1.0).
        
        Returns:
            float: Progress within current phase
        """
        if self.phase == "setup":
            return self.progression / self.STARTING_THRESHOLD if self.STARTING_THRESHOLD > 0 else 1.0
        elif self.phase == "rising_action":
            phase_start = self.STARTING_THRESHOLD
            phase_length = self.DEVELOPING_THRESHOLD - self.STARTING_THRESHOLD
            return (self.progression - phase_start) / phase_length if phase_length > 0 else 1.0
        elif self.phase == "climax":
            phase_start = self.DEVELOPING_THRESHOLD
            phase_length = self.CLIMAX_THRESHOLD - self.DEVELOPING_THRESHOLD
            return (self.progression - phase_start) / phase_length if phase_length > 0 else 1.0
        else:  # resolution
            phase_start = self.CLIMAX_THRESHOLD
            phase_length = 1.0 - self.CLIMAX_THRESHOLD
            return (self.progression - phase_start) / phase_length if phase_length > 0 else 1.0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the story arc.
        
        Returns:
            dict: Status information including phase, progression, events, etc.
        """
        return {
            "name": self.name,
            "importance": self.importance,
            "progression": self.progression,
            "progression_percentage": self.get_progression_percentage(),
            "phase": self.phase,
            "phase_progress": self.get_phase_progress(),
            "completed": self.completed,
            "events_count": len(self.events),
            "characters_involved": list(self.characters_involved),
            "start_time": self.start_time.isoformat(),
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "resolution_type": self.resolution_type
        }
    
    def set_resolution(self, resolution_type: str):
        """
        Set the resolution type when the story arc completes.
        
        Args:
            resolution_type: Type of resolution ('success', 'failure', 'mixed', 'open')
        """
        self.resolution_type = resolution_type
        if not self.completed:
            self.progression = 1.0
    
    def __str__(self) -> str:
        return f"StoryArc('{self.name}', {self.phase}, {self.get_progression_percentage()}%)"
    
    def __repr__(self) -> str:
        return self.__str__()


class StoryArcManager:
    """
    Manages multiple story arcs and their interactions.
    """
    
    def __init__(self):
        """Initialize the story arc manager."""
        self.active_arcs = []
        self.completed_arcs = []
        self.arc_counter = 0
    
    def create_arc(self, name: str, importance: int = 5) -> StoryArc:
        """
        Create a new story arc.
        
        Args:
            name: Name for the story arc
            importance: Importance level (1-10)
            
        Returns:
            StoryArc: The created story arc
        """
        arc = StoryArc(name, importance)
        self.active_arcs.append(arc)
        self.arc_counter += 1
        logging.info(f"Created story arc: {name}")
        return arc
    
    def get_arc(self, name: str) -> Optional[StoryArc]:
        """
        Get a story arc by name.
        
        Args:
            name: Name of the story arc
            
        Returns:
            StoryArc: The story arc if found, None otherwise
        """
        for arc in self.active_arcs + self.completed_arcs:
            if arc.name == name:
                return arc
        return None
    
    def update_arcs(self):
        """
        Update all active story arcs and move completed ones.
        """
        completed_this_update = []
        
        for arc in self.active_arcs:
            if arc.completed:
                completed_this_update.append(arc)
        
        for arc in completed_this_update:
            self.active_arcs.remove(arc)
            self.completed_arcs.append(arc)
            logging.info(f"Story arc '{arc.name}' moved to completed")
    
    def get_arcs_in_phase(self, phase: str) -> List[StoryArc]:
        """
        Get all active arcs in a specific phase.
        
        Args:
            phase: Phase to filter by
            
        Returns:
            List[StoryArc]: Arcs in the specified phase
        """
        return [arc for arc in self.active_arcs if arc.is_in_phase(phase)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all story arcs.
        
        Returns:
            dict: Statistics including counts, phases, etc.
        """
        total_arcs = len(self.active_arcs) + len(self.completed_arcs)
        phase_counts = {}
        
        for arc in self.active_arcs:
            phase_counts[arc.phase] = phase_counts.get(arc.phase, 0) + 1
        
        return {
            "total_arcs": total_arcs,
            "active_arcs": len(self.active_arcs),
            "completed_arcs": len(self.completed_arcs),
            "phase_distribution": phase_counts,
            "average_progression": sum(arc.progression for arc in self.active_arcs) / len(self.active_arcs) if self.active_arcs else 0
        }