# tiny_event_handler.py
""" 4. Event Handler (event_handler.py)
This module will be responsible for detecting and handling game events, triggering strategic updates. """
from datetime import datetime
import importlib

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

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "importance": self.importance,
            "impact": self.impact,
            "required_items": self.required_items,
            "coordinates_location": self.coordinates_location,
        }

    def __str__(self):
        return f"{self.name} - {self.date} - {self.type} - {self.importance} - {self.impact} - {self.required_items} - {self.coordinates_location} - {self.location}"

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
                ]
            )
        )


class EventHandler:
    def __init__(self, graph_manager: GraphManager):
        GraphManager = importlib.import_module("tiny_graph_manager").GraphManager
        self.events = []
        self.graph_manager = graph_manager

    def add_event(self, event):
        self.events.append(event)
        # Assume GraphManager instance is available and named graph_manager
        self.graph_manager.add_event_node(event)

    def check_events(self):
        # Check for and return any relevant game events
        current_date = datetime.now().strftime("%Y-%m-%d")
        return [event for event in self.events if event.date == current_date]

    def check_daily_events(self):
        # Logic to detect the start of a new day
        current_date = datetime.now().strftime("%Y-%m-%d")
        return {"event": "new_day", "date": current_date}
