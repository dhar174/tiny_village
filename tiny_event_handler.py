# tiny_event_handler.py
"""4. Event Handler (event_handler.py)
This module will be responsible for detecting and handling game events, triggering strategic updates.
"""
from datetime import datetime, timedelta
import importlib
import logging
from typing import List, Dict, Any, Optional, Callable

# from tiny_graph_manager import GraphManager as graph_manager
from actions import Action, ActionSystem
from tiny_locations import Location
import tiny_time_manager as time_manager
from tiny_types import GraphManager


class Event:
    def __init__(
        self,
        name,
        date,
        event_type,
        importance,
        impact,
        required_items=None,
        coordinates_location=None,
        location: Location = None,
        action_system=None,
        recurrence_pattern=None,
        effects=None,
        preconditions=None,
        cascading_events=None,
        participants=None,
        max_participants=None,
    ):
        ActionSystem = importlib.import_module("actions").ActionSystem
        self.name = name
        self.date = date
        self.type = event_type
        self.importance = importance
        self.impact = impact
        self.required_items = required_items if required_items else []
        self.coordinates_location = (
            coordinates_location if coordinates_location else (0, 0)
        )
        self.location = location
        self.action_system = action_system if action_system else ActionSystem()

        # Enhanced event properties
        self.recurrence_pattern = recurrence_pattern  # Dict: {"type": "daily/weekly/monthly/yearly", "interval": 1, "end_date": None}
        self.effects = (
            effects if effects else []
        )  # List of effect dicts that modify game state
        self.preconditions = (
            preconditions if preconditions else []
        )  # Conditions that must be met for event to trigger
        self.cascading_events = (
            cascading_events if cascading_events else []
        )  # Events triggered by this event
        self.participants = (
            participants if participants else []
        )  # Characters participating in event
        self.max_participants = max_participants  # Maximum number of participants

        # Event state tracking
        self.is_active = False
        self.last_triggered = None
        self.trigger_count = 0
        self.created_at = datetime.now()

    def is_recurring(self) -> bool:
        """Check if this event has a recurrence pattern."""
        return self.recurrence_pattern is not None

    def get_next_occurrence(self, from_date: datetime = None) -> Optional[datetime]:
        """Calculate the next occurrence of this event."""
        if not self.is_recurring():
            return None

        if from_date is None:
            from_date = datetime.now()

        pattern = self.recurrence_pattern
        interval = pattern.get("interval", 1)
        recurrence_type = pattern.get("type", "daily")

        base_date = (
            datetime.strptime(self.date, "%Y-%m-%d")
            if isinstance(self.date, str)
            else self.date
        )

        if recurrence_type == "daily":
            next_date = base_date
            while next_date <= from_date:
                next_date += timedelta(days=interval)

        elif recurrence_type == "weekly":
            next_date = base_date
            while next_date <= from_date:
                next_date += timedelta(weeks=interval)

        elif recurrence_type == "monthly":
            next_date = base_date
            while next_date <= from_date:
                if next_date.month == 12:
                    next_date = next_date.replace(year=next_date.year + 1, month=1)
                else:
                    next_date = next_date.replace(month=next_date.month + interval)

        elif recurrence_type == "yearly":
            next_date = base_date
            while next_date <= from_date:
                next_date = next_date.replace(year=next_date.year + interval)
        else:
            return None

        # Check if we've passed the end date
        end_date = pattern.get("end_date")
        if end_date and next_date > end_date:
            return None

        return next_date

    def should_trigger(self, current_time: datetime) -> bool:
        """Check if event should trigger at the given time."""
        # Check preconditions first
        if not self.check_preconditions():
            return False

        if self.is_recurring():
            next_occurrence = self.get_next_occurrence(current_time - timedelta(days=1))
            if (
                next_occurrence
                and abs((next_occurrence - current_time).total_seconds()) < 3600
            ):  # Within 1 hour
                return True
        else:
            event_date = (
                datetime.strptime(self.date, "%Y-%m-%d")
                if isinstance(self.date, str)
                else self.date
            )
            if abs((event_date - current_time).total_seconds()) < 3600:  # Within 1 hour
                return True

        return False

    def check_preconditions(self) -> bool:
        """Check if all preconditions for the event are met."""
        for condition in self.preconditions:
            condition_type = condition.get("type")

            if condition_type == "attribute_check":
                target = condition.get("target")
                attribute = condition.get("attribute")
                operator = condition.get("operator", ">=")
                threshold = condition.get("threshold", 0)

                # Check participant attributes
                if target == "participants":
                    for participant in self.participants:
                        if hasattr(participant, attribute):
                            value = getattr(participant, attribute)
                            if not self._evaluate_condition(value, operator, threshold):
                                return False

            elif condition_type == "time_check":
                current_time = datetime.now()
                start_time = condition.get("start_time")
                end_time = condition.get("end_time")

                if start_time and current_time < start_time:
                    return False
                if end_time and current_time > end_time:
                    return False

            elif condition_type == "weather_check":
                # Placeholder for weather condition checking
                required_weather = condition.get("weather")
                # This would integrate with a weather system
                pass

            elif condition_type == "location_check":
                required_location = condition.get("location")
                if self.location and self.location.name != required_location:
                    return False

        return True

    def _evaluate_condition(self, value, operator: str, threshold) -> bool:
        """Evaluate a condition based on operator."""
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

    def check_required_items(self, available_items: List[Any]) -> bool:
        """Check if all required items are available."""
        if not self.required_items:
            return True

        for required_item in self.required_items:
            if required_item not in available_items:
                return False
        return True

    def add_participant(self, character) -> bool:
        """Add a character as participant if space available."""
        if self.max_participants and len(self.participants) >= self.max_participants:
            return False
        if character not in self.participants:
            self.participants.append(character)
        return True

    def remove_participant(self, character) -> bool:
        """Remove a character from participants."""
        if character in self.participants:
            self.participants.remove(character)
            return True
        return False

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "type": self.type,
            "importance": self.importance,
            "impact": self.impact,
            "required_items": self.required_items,
            "coordinates_location": self.coordinates_location,
            "recurrence_pattern": self.recurrence_pattern,
            "effects": self.effects,
            "preconditions": self.preconditions,
            "cascading_events": self.cascading_events,
            "participants": [
                p.name if hasattr(p, "name") else str(p) for p in self.participants
            ],
            "max_participants": self.max_participants,
            "is_active": self.is_active,
            "last_triggered": (
                self.last_triggered.isoformat() if self.last_triggered else None
            ),
            "trigger_count": self.trigger_count,
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self):
        recurrence_info = ""
        if self.is_recurring():
            pattern = self.recurrence_pattern
            recurrence_info = (
                f" (Recurring: {pattern['type']} every {pattern.get('interval', 1)})"
            )
        return f"{self.name} - {self.date} - {self.type} - {self.importance} - {self.impact}{recurrence_info} - Participants: {len(self.participants)}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.date == other.date
            and self.type == other.type
            and self.importance == other.importance
            and self.impact == other.impact
            and self.required_items == other.required_items
            and self.coordinates_location == other.coordinates_location
            and self.location == other.location
            and self.recurrence_pattern == other.recurrence_pattern
        )

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.name,
                    self.date,
                    self.type,
                    self.importance,
                    self.impact,
                    self.coordinates_location,
                    self.location,
                    str(self.recurrence_pattern),
                ]
            )
        )


class EventHandler:
    def __init__(self, graph_manager=None, time_manager=None):
        # Only import GraphManager if we don't have a graph_manager passed in
        if graph_manager is None:
            try:
                GraphManager = importlib.import_module(
                    "tiny_graph_manager"
                ).GraphManager
                self.graph_manager = GraphManager()
            except ImportError:
                print("Warning: Could not import GraphManager, using None")
                self.graph_manager = None
        else:
            self.graph_manager = graph_manager

        self.events = []
        self.active_events = []
        self.processed_events = []
        self.time_manager = time_manager
        self.event_triggers = {}  # Dict to store custom event triggers
        self.cascading_event_queue = []  # Queue for cascading events

    def add_event(self, event):
        """Add an event to the handler and register it in the graph."""
        self.events.append(event)
        self.graph_manager.add_event_node(event)
        logging.info(f"Added event: {event.name}")

    def remove_event(self, event_name: str) -> bool:
        """Remove an event by name."""
        for event in self.events:
            if event.name == event_name:
                self.events.remove(event)
                if event in self.active_events:
                    self.active_events.remove(event)
                logging.info(f"Removed event: {event_name}")
                return True
        return False

    def update_event(self, event_name: str, **kwargs) -> bool:
        """Update an existing event with new properties."""
        for event in self.events:
            if event.name == event_name:
                for key, value in kwargs.items():
                    if hasattr(event, key):
                        setattr(event, key, value)
                logging.info(f"Updated event: {event_name}")
                return True
        return False

    def get_event(self, event_name: str) -> Optional["Event"]:
        """Get an event by name."""
        for event in self.events:
            if event.name == event_name:
                return event
        return None

    def check_events(self, current_time: datetime = None) -> List["Event"]:
        """Check for and return any relevant game events that should trigger."""
        if current_time is None:
            current_time = datetime.now()

        triggered_events = []

        for event in self.events:
            if event.should_trigger(current_time):
                triggered_events.append(event)

        return triggered_events

    def process_events(self, current_time: datetime = None) -> Dict[str, Any]:
        """Process all triggered events and their effects."""
        if current_time is None:
            current_time = datetime.now()

        triggered_events = self.check_events(current_time)
        results = {
            "processed_events": [],
            "failed_events": [],
            "cascading_events": [],
            "state_changes": {},
        }

        for event in triggered_events:
            try:
                if self._process_single_event(event, current_time):
                    results["processed_events"].append(event.name)
                    self.processed_events.append(event)

                    # Handle cascading events
                    cascading = self._trigger_cascading_events(event)
                    results["cascading_events"].extend(cascading)
                else:
                    results["failed_events"].append(event.name)

            except Exception as e:
                logging.error(f"Error processing event {event.name}: {e}")
                results["failed_events"].append(event.name)

        return results

    def _process_single_event(self, event: "Event", current_time: datetime) -> bool:
        """Process a single event and apply its effects."""
        logging.info(f"Processing event: {event.name}")

        # Check if required items are available
        if not self._check_event_requirements(event):
            logging.warning(f"Event {event.name} requirements not met")
            return False

        # Apply event effects
        self._apply_event_effects(event)

        # Update event state
        event.is_active = True
        event.last_triggered = current_time
        event.trigger_count += 1

        if event not in self.active_events:
            self.active_events.append(event)

        # Update graph relationships for participants
        self._update_event_relationships(event)

        return True

    def _check_event_requirements(self, event: "Event") -> bool:
        """Check if all event requirements are met."""
        if not event.required_items:
            return True

        # Get available items from the graph
        available_items = []
        if event.location:
            # Check items at the event location
            for node in self.graph_manager.G.nodes():
                if hasattr(
                    node, "coordinates_location"
                ) and event.location.contains_point(*node.coordinates_location):
                    if hasattr(node, "item_type"):  # It's an item
                        available_items.append(node)

        return event.check_required_items(available_items)

    def _apply_event_effects(self, event: "Event"):
        """Apply the effects of an event to the game state."""
        for effect in event.effects:
            try:
                self._apply_single_effect(event, effect)
            except Exception as e:
                logging.error(
                    f"Error applying effect {effect} for event {event.name}: {e}"
                )

    def _apply_single_effect(self, event: "Event", effect: Dict[str, Any]):
        """Apply a single effect from an event."""
        targets = effect.get("targets", [])
        attribute = effect.get("attribute")
        change_value = effect.get("change_value", 0)
        effect_type = effect.get("type", "attribute_change")

        if effect_type == "attribute_change":
            for target in targets:
                if target == "participants":
                    for participant in event.participants:
                        self._modify_entity_attribute(
                            participant, attribute, change_value
                        )
                elif target == "location" and event.location:
                    self._modify_entity_attribute(
                        event.location, attribute, change_value
                    )
                else:
                    # Try to find the target in the graph
                    target_node = self.graph_manager.get_node(target)
                    if target_node:
                        self._modify_entity_attribute(
                            target_node, attribute, change_value
                        )

        elif effect_type == "relationship_change":
            # Modify relationships between participants
            for i, participant1 in enumerate(event.participants):
                for participant2 in event.participants[i + 1 :]:
                    if self.graph_manager.G.has_edge(participant1, participant2):
                        self.graph_manager.update_character_character_edge(
                            participant1,
                            participant2,
                            impact_factor=1,
                            impact_value=change_value,
                        )

    def _modify_entity_attribute(self, entity, attribute: str, change_value):
        """Modify an attribute of an entity."""
        if hasattr(entity, "get_state"):
            state = entity.get_state()
            current_value = state.get(attribute, 0)
            if isinstance(current_value, (int, float)):
                new_value = current_value + change_value
                state[attribute] = new_value
                logging.debug(
                    f"Modified {entity} {attribute}: {current_value} -> {new_value}"
                )
        elif hasattr(entity, attribute):
            current_value = getattr(entity, attribute)
            if isinstance(current_value, (int, float)):
                new_value = current_value + change_value
                setattr(entity, attribute, new_value)
                logging.debug(
                    f"Modified {entity} {attribute}: {current_value} -> {new_value}"
                )

    def _update_event_relationships(self, event: "Event"):
        """Update graph relationships based on event participation."""
        for participant in event.participants:
            # Add character-event edge if not exists
            if not self.graph_manager.G.has_edge(participant, event):
                self.graph_manager.add_character_event_edge(
                    participant,
                    event,
                    participation_status=True,
                    role="participant",
                    impact_on_character=event.impact,
                    emotional_outcome=max(1, event.importance),
                )

            # Add location-event edge if not exists
            if event.location and not self.graph_manager.G.has_edge(
                event.location, event
            ):
                self.graph_manager.add_location_event_edge(
                    event.location,
                    event,
                    event_occurrence={"frequency": 1, "predictability": 3},
                    location_role="venue",
                    capacity=event.max_participants or 100,
                    preparation_level=5,
                )

    def _trigger_cascading_events(self, parent_event: "Event") -> List[str]:
        """Trigger cascading events from a parent event."""
        triggered_cascading = []

        for cascading_event_def in parent_event.cascading_events:
            try:
                # Create new event from definition
                cascading_event = self._create_event_from_definition(
                    cascading_event_def, parent_event
                )

                if cascading_event:
                    # Add delay if specified
                    delay = cascading_event_def.get("delay", 0)
                    if delay > 0:
                        # Schedule for later processing
                        trigger_time = datetime.now() + timedelta(hours=delay)
                        cascading_event.date = trigger_time
                        self.cascading_event_queue.append(
                            (trigger_time, cascading_event)
                        )
                    else:
                        # Process immediately
                        self.add_event(cascading_event)
                        if self._process_single_event(cascading_event, datetime.now()):
                            triggered_cascading.append(cascading_event.name)

            except Exception as e:
                logging.error(f"Error triggering cascading event: {e}")

        return triggered_cascading

    def _create_event_from_definition(
        self, event_def: Dict[str, Any], parent_event: "Event"
    ) -> Optional["Event"]:
        """Create an event from a cascading event definition."""
        try:
            # Inherit some properties from parent if not specified
            name = event_def.get(
                "name", f"{parent_event.name}_cascade_{len(self.events)}"
            )
            date = event_def.get("date", datetime.now())
            event_type = event_def.get("type", parent_event.type)
            importance = event_def.get(
                "importance", max(1, parent_event.importance - 1)
            )
            impact = event_def.get("impact", parent_event.impact)
            location = event_def.get("location", parent_event.location)

            # Inherit participants or specify new ones
            participants = event_def.get(
                "participants", parent_event.participants.copy()
            )

            return Event(
                name=name,
                date=date,
                event_type=event_type,
                importance=importance,
                impact=impact,
                location=location,
                participants=participants,
                effects=event_def.get("effects", []),
                preconditions=event_def.get("preconditions", []),
                max_participants=event_def.get("max_participants"),
            )

        except Exception as e:
            logging.error(f"Error creating cascading event: {e}")
            return None

    def process_cascading_queue(self, current_time: datetime = None) -> List[str]:
        """Process any scheduled cascading events."""
        if current_time is None:
            current_time = datetime.now()

        processed = []
        remaining_queue = []

        for trigger_time, event in self.cascading_event_queue:
            if trigger_time <= current_time:
                self.add_event(event)
                if self._process_single_event(event, current_time):
                    processed.append(event.name)
            else:
                remaining_queue.append((trigger_time, event))

        self.cascading_event_queue = remaining_queue
        return processed

    def check_daily_events(self, current_time: datetime = None) -> Dict[str, Any]:
        """Enhanced daily event detection with recurring event scheduling."""
        if current_time is None:
            current_time = datetime.now()

        current_date = current_time.strftime("%Y-%m-%d")
        daily_events = []

        # Check for recurring events that should trigger today
        for event in self.events:
            if event.is_recurring():
                next_occurrence = event.get_next_occurrence(current_time)
                if (
                    next_occurrence
                    and next_occurrence.date() == current_time.date()
                    and event.should_trigger(current_time)
                ):
                    daily_events.append(event)

        # Process any scheduled cascading events
        cascading_processed = self.process_cascading_queue(current_time)

        # Check for special date-based events
        special_events = self._check_special_date_events(current_time)
        daily_events.extend(special_events)

        return {
            "event": "daily_check",
            "date": current_date,
            "recurring_events": [e.name for e in daily_events if e.is_recurring()],
            "cascading_events": cascading_processed,
            "special_events": [e.name for e in special_events],
            "total_events": len(daily_events)
            + len(cascading_processed)
            + len(special_events),
        }

    def _check_special_date_events(self, current_time: datetime) -> List["Event"]:
        """Check for special date-based events (holidays, seasons, etc.)."""
        special_events = []

        # Check for seasonal events
        month = current_time.month
        day = current_time.day

        # Example seasonal events
        if month == 12 and day == 25:  # Christmas
            special_events.append(self.create_holiday_event("Christmas", current_time))
        elif month == 1 and day == 1:  # New Year
            special_events.append(self.create_holiday_event("New Year", current_time))
        elif month == 10 and day == 31:  # Halloween
            special_events.append(self.create_holiday_event("Halloween", current_time))

        # Check for market days (every 7 days)
        if current_time.weekday() == 6:  # Sunday
            special_events.append(self.create_market_event(current_time))

        return [e for e in special_events if e is not None]

    # Event Creation Helper Methods
    def create_holiday_event(
        self, holiday_name: str, date: datetime, location=None
    ) -> "Event":
        """Create a holiday event with predefined effects."""
        effects = [
            {
                "type": "attribute_change",
                "targets": ["participants"],
                "attribute": "happiness",
                "change_value": 10,
            },
            {
                "type": "relationship_change",
                "targets": ["participants"],
                "attribute": "relationship_strength",
                "change_value": 5,
            },
        ]

        return Event(
            name=f"{holiday_name} Celebration",
            date=date,
            event_type="holiday",
            importance=8,
            impact=5,
            location=location,
            effects=effects,
            max_participants=50,
        )

    def create_market_event(self, date: datetime, location=None) -> "Event":
        """Create a market day event."""
        effects = [
            {
                "type": "attribute_change",
                "targets": ["location"],
                "attribute": "economic_activity",
                "change_value": 20,
            }
        ]

        return Event(
            name="Market Day",
            date=date,
            event_type="economic",
            importance=5,
            impact=3,
            location=location,
            effects=effects,
            recurrence_pattern={"type": "weekly", "interval": 1},
            max_participants=100,
        )

    def create_weather_event(
        self, weather_type: str, date: datetime, severity: int = 5
    ) -> "Event":
        """Create a weather-based event."""
        effects = []

        if weather_type == "storm":
            effects = [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "safety",
                    "change_value": -severity,
                }
            ]
        elif weather_type == "drought":
            effects = [
                {
                    "type": "attribute_change",
                    "targets": ["location"],
                    "attribute": "water_level",
                    "change_value": -severity * 2,
                }
            ]

        return Event(
            name=f"{weather_type.title()} Event",
            date=date,
            event_type="weather",
            importance=severity,
            impact=severity,
            effects=effects,
        )

    def create_social_event(
        self, event_name: str, date: datetime, participants=None, location=None
    ) -> "Event":
        """Create a social gathering event."""
        effects = [
            {
                "type": "attribute_change",
                "targets": ["participants"],
                "attribute": "social_satisfaction",
                "change_value": 8,
            },
            {
                "type": "relationship_change",
                "targets": ["participants"],
                "attribute": "relationship_strength",
                "change_value": 3,
            },
        ]

        return Event(
            name=event_name,
            date=date,
            event_type="social",
            importance=6,
            impact=4,
            location=location,
            participants=participants or [],
            effects=effects,
            max_participants=20,
        )

    def create_work_event(
        self, task_name: str, date: datetime, required_items=None, location=None
    ) -> "Event":
        """Create a work or task-based event."""
        effects = [
            {
                "type": "attribute_change",
                "targets": ["participants"],
                "attribute": "productivity",
                "change_value": 5,
            },
            {
                "type": "attribute_change",
                "targets": ["location"],
                "attribute": "development_level",
                "change_value": 2,
            },
        ]

        return Event(
            name=task_name,
            date=date,
            event_type="work",
            importance=4,
            impact=3,
            location=location,
            required_items=required_items or [],
            effects=effects,
            max_participants=10,
        )

    # Event Template System
    def get_event_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined event templates for easy event creation."""
        return {
            "village_festival": {
                "type": "social",
                "importance": 9,
                "impact": 6,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "happiness",
                        "change_value": 15,
                    },
                    {
                        "type": "relationship_change",
                        "targets": ["participants"],
                        "attribute": "community_bond",
                        "change_value": 10,
                    },
                ],
                "max_participants": 100,
                "required_items": ["decorations", "food", "music"],
            },
            "harvest_season": {
                "type": "economic",
                "importance": 8,
                "impact": 7,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["location"],
                        "attribute": "food_supply",
                        "change_value": 50,
                    },
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "wealth",
                        "change_value": 20,
                    },
                ],
                "recurrence_pattern": {"type": "yearly", "interval": 1},
                "max_participants": 30,
            },
            "merchant_arrival": {
                "type": "economic",
                "importance": 6,
                "impact": 4,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["location"],
                        "attribute": "trade_activity",
                        "change_value": 25,
                    }
                ],
                "cascading_events": [
                    {
                        "name": "market_expansion",
                        "type": "economic",
                        "delay": 24,  # 24 hours later
                        "effects": [
                            {
                                "type": "attribute_change",
                                "targets": ["location"],
                                "attribute": "economic_development",
                                "change_value": 10,
                            }
                        ],
                    }
                ],
                "max_participants": 15,
            },
            "natural_disaster": {
                "type": "crisis",
                "importance": 10,
                "impact": -8,
                "effects": [
                    {
                        "type": "attribute_change",
                        "targets": ["participants"],
                        "attribute": "safety",
                        "change_value": -20,
                    },
                    {
                        "type": "attribute_change",
                        "targets": ["location"],
                        "attribute": "infrastructure",
                        "change_value": -30,
                    },
                ],
                "cascading_events": [
                    {
                        "name": "rebuilding_effort",
                        "type": "work",
                        "delay": 72,  # 3 days later
                        "effects": [
                            {
                                "type": "attribute_change",
                                "targets": ["location"],
                                "attribute": "infrastructure",
                                "change_value": 15,
                            }
                        ],
                    }
                ],
            },
        }

    def create_event_from_template(
        self, template_name: str, event_name: str, date: datetime, **overrides
    ) -> Optional["Event"]:
        """Create an event from a predefined template with optional overrides."""
        templates = self.get_event_templates()

        if template_name not in templates:
            logging.error(f"Template '{template_name}' not found")
            return None

        template = templates[template_name].copy()

        # Apply any overrides
        template.update(overrides)

        return Event(
            name=event_name,
            date=date,
            event_type=template.get("type", "general"),
            importance=template.get("importance", 5),
            impact=template.get("impact", 3),
            required_items=template.get("required_items", []),
            location=template.get("location"),
            recurrence_pattern=template.get("recurrence_pattern"),
            effects=template.get("effects", []),
            preconditions=template.get("preconditions", []),
            cascading_events=template.get("cascading_events", []),
            max_participants=template.get("max_participants"),
        )

    # Integration and Utility Methods
    def schedule_recurring_events(self, time_span_days: int = 365):
        """Schedule recurring events for the specified time span."""
        current_time = datetime.now()
        end_time = current_time + timedelta(days=time_span_days)

        scheduled_count = 0

        for event in self.events:
            if event.is_recurring():
                next_occurrence = event.get_next_occurrence(current_time)

                while next_occurrence and next_occurrence <= end_time:
                    # Create new instance for this occurrence
                    occurrence_event = Event(
                        name=f"{event.name}_{next_occurrence.strftime('%Y%m%d')}",
                        date=next_occurrence,
                        event_type=event.type,
                        importance=event.importance,
                        impact=event.impact,
                        required_items=event.required_items.copy(),
                        location=event.location,
                        effects=event.effects.copy(),
                        preconditions=event.preconditions.copy(),
                        cascading_events=event.cascading_events.copy(),
                        max_participants=event.max_participants,
                    )

                    self.add_event(occurrence_event)
                    scheduled_count += 1

                    # Get next occurrence
                    next_occurrence = event.get_next_occurrence(next_occurrence)

        logging.info(f"Scheduled {scheduled_count} recurring event occurrences")
        return scheduled_count

    def get_events_by_type(self, event_type: str) -> List["Event"]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.type == event_type]

    def get_events_by_location(self, location) -> List["Event"]:
        """Get all events at a specific location."""
        return [event for event in self.events if event.location == location]

    def get_events_in_timeframe(
        self, start_time: datetime, end_time: datetime
    ) -> List["Event"]:
        """Get all events within a specific timeframe."""
        events_in_timeframe = []

        for event in self.events:
            event_date = (
                datetime.strptime(event.date, "%Y-%m-%d")
                if isinstance(event.date, str)
                else event.date
            )

            if start_time <= event_date <= end_time:
                events_in_timeframe.append(event)

            # Check recurring events
            elif event.is_recurring():
                next_occurrence = event.get_next_occurrence(start_time)
                while next_occurrence and next_occurrence <= end_time:
                    events_in_timeframe.append(event)
                    break  # Only add once per recurring event

        return events_in_timeframe

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about events in the handler."""
        stats = {
            "total_events": len(self.events),
            "active_events": len(self.active_events),
            "processed_events": len(self.processed_events),
            "recurring_events": len([e for e in self.events if e.is_recurring()]),
            "events_by_type": {},
            "events_by_importance": {},
            "average_importance": 0,
            "scheduled_cascading": len(self.cascading_event_queue),
        }

        # Count by type
        for event in self.events:
            stats["events_by_type"][event.type] = (
                stats["events_by_type"].get(event.type, 0) + 1
            )
            stats["events_by_importance"][event.importance] = (
                stats["events_by_importance"].get(event.importance, 0) + 1
            )

        # Calculate average importance
        if self.events:
            stats["average_importance"] = sum(e.importance for e in self.events) / len(
                self.events
            )

        return stats

    def cleanup_old_events(self, days_old: int = 30):
        """Remove old processed events to keep memory usage manageable."""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        initial_count = len(self.processed_events)
        self.processed_events = [
            event
            for event in self.processed_events
            if event.last_triggered and event.last_triggered > cutoff_date
        ]

        removed_count = initial_count - len(self.processed_events)
        logging.info(f"Cleaned up {removed_count} old processed events")
        return removed_count
