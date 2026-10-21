#!/usr/bin/env python3
"""Integration tests for storytelling with a concrete graph harness."""

from datetime import datetime, timedelta

import logging
import networkx as nx

from tiny_event_handler import Event, EventHandler
from tiny_gameplay_controller import GameplayController
from tiny_storytelling_system import StorytellingSystem


logging.disable(logging.CRITICAL)


class GraphManagerHarness:
    """Small concrete graph manager used for storytelling integration tests."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.characters = {}
        self.locations = {}
        self.event_nodes = {}
        self.character_event_edges = []
        self.location_event_edges = []
        self.relationship_updates = []

    def add_event_node(self, event):
        self.event_nodes[event.name] = event
        self.G.add_node(event, type="event", event=event)

    def get_node(self, node_name):
        return self.event_nodes.get(node_name)

    def add_character_event_edge(self, character, event, **attributes):
        self.G.add_node(character, type="character")
        self.G.add_edge(character, event, **attributes)
        self.character_event_edges.append((character, event, attributes))

    def add_location_event_edge(self, location, event, **attributes):
        self.G.add_node(location, type="location")
        self.G.add_edge(location, event, **attributes)
        self.location_event_edges.append((location, event, attributes))

    def update_character_character_edge(self, source, target, **attributes):
        self.G.add_node(source, type="character")
        self.G.add_node(target, type="character")
        self.G.add_edge(source, target, **attributes)
        self.relationship_updates.append((source, target, attributes))


class StoryCharacter:
    def __init__(self, name):
        self.name = name


def create_controller():
    graph_manager = GraphManagerHarness()
    controller = GameplayController(
        graph_manager=graph_manager,
        config={"screen_width": 800, "screen_height": 600},
    )
    return controller, graph_manager


def test_storytelling_integration_processes_events_into_story_arcs():
    controller, graph_manager = create_controller()
    alice = StoryCharacter("Alice")

    stories = controller.get_current_stories()
    assert controller.storytelling_system is not None
    assert stories["feature_status"] == "BASIC_IMPLEMENTED"
    assert "active_narratives" in stories

    festival = Event(
        name="Village Festival",
        date=datetime.now(),
        event_type="celebration",
        importance=8,
        impact=6,
        participants=[alice],
    )

    initial_story_count = len(controller.storytelling_system.arc_manager.active_arcs)
    controller.events = [festival]
    controller._process_pending_events()

    assert len(controller.storytelling_system.arc_manager.active_arcs) > initial_story_count
    assert "Village Festival" in graph_manager.event_nodes
    assert graph_manager.G.has_edge(alice, festival)
    assert any(edge[0] is alice and edge[1] is festival for edge in graph_manager.character_event_edges)


def test_system_recovery_recreates_storytelling_system():
    controller, _ = create_controller()

    status = controller.recovery_manager.get_system_status()
    assert status["storytelling_system"] == "healthy"

    controller.storytelling_system = None
    recovery_success = controller.recovery_manager.attempt_recovery("storytelling_system")

    assert recovery_success
    assert controller.storytelling_system is not None


def test_story_arc_creation_tracks_participants_across_events():
    controller, _ = create_controller()
    alice = StoryCharacter("Alice")
    bob = StoryCharacter("Bob")

    events = [
        Event(
            name="Harvest Festival",
            date=datetime.now(),
            event_type="celebration",
            importance=9,
            impact=7,
            participants=[alice, bob],
        ),
        Event(
            name="Market Opening",
            date=datetime.now(),
            event_type="economic",
            importance=7,
            impact=5,
            participants=[alice],
        ),
        Event(
            name="Bridge Construction",
            date=datetime.now(),
            event_type="work",
            importance=8,
            impact=6,
            participants=[alice, bob],
        ),
    ]

    initial_arcs = controller.storytelling_system.arc_manager.get_arc_statistics()["active_arcs"]
    for event in events:
        controller.events = [event]
        controller._process_pending_events()

    final_stats = controller.storytelling_system.arc_manager.get_arc_statistics()
    alice_stories = controller.get_character_stories("Alice")

    assert final_stats["active_arcs"] > initial_arcs
    assert len(controller.get_current_stories()["active_narratives"]) > 0
    assert len(alice_stories["active_arcs"]) > 0


def test_narrative_coherence_uses_real_story_elements():
    graph_manager = GraphManagerHarness()
    event_handler = EventHandler(graph_manager)
    storytelling_system = StorytellingSystem(event_handler)
    alice = StoryCharacter("Alice")
    bob = StoryCharacter("Bob")

    events = [
        Event(
            name="First Meeting",
            date=datetime.now(),
            event_type="social",
            importance=6,
            impact=4,
            participants=[alice, bob],
        ),
        Event(
            name="Shared Meal",
            date=datetime.now() + timedelta(days=1),
            event_type="social",
            importance=5,
            impact=3,
            participants=[alice, bob],
        ),
        Event(
            name="Joint Project",
            date=datetime.now() + timedelta(days=3),
            event_type="work",
            importance=7,
            impact=5,
            participants=[alice, bob],
        ),
    ]

    narratives = []
    for event in events:
        result = storytelling_system.process_event_for_stories(event)
        narratives.extend(result.get("narratives", []))

    shared_arc = None
    for arc in storytelling_system.arc_manager.active_arcs.values():
        if "Alice" in arc.participants and "Bob" in arc.participants:
            shared_arc = arc
            break

    summary = storytelling_system.generate_story_summary(days_back=7)

    assert len(narratives) >= 2
    assert shared_arc is not None
    assert len(shared_arc.elements) >= 2
    assert "Alice" in summary or "Bob" in summary
