# tiny_event_handler.py
""" 4. Event Handler (event_handler.py)
This module will be responsible for detecting and handling game events, triggering strategic updates. """
from datetime import datetime
from tiny_graph_manager import GraphManager as graph_manager


class Event:
    def __init__(
        self,
        name,
        date,
        event_type,
        importance,
        impact,
        required_items=None,
        coordinate_location=None,
    ):
        self.name = name
        self.date = date
        self.type = event_type
        self.importance = importance
        self.impact = impact
        self.required_items = required_items if required_items else []
        self.coordinate_location = (
            coordinate_location if coordinate_location else (0, 0)
        )

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "importance": self.importance,
            "impact": self.impact,
            "required_items": self.required_items,
            "coordinate_location": self.coordinate_location,
        }


class EventHandler:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)
        # Assume GraphManager instance is available and named graph_manager
        graph_manager.add_event_node(event)

    def check_events(self):
        # Check for and return any relevant game events
        current_date = datetime.now().strftime("%Y-%m-%d")
        return [event for event in self.events if event.date == current_date]

    def check_daily_events(self):
        # Logic to detect the start of a new day
        current_date = datetime.now().strftime("%Y-%m-%d")
        return {"event": "new_day", "date": current_date}
