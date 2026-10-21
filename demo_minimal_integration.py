#!/usr/bin/env python3
"""Bounded startup/event/basic-memory checks using the actual TinyVillage runtime.

This is a lightweight testing profile. AI is required in the final product;
these checks neither execute gameplay turns nor validate AI model behavior.
"""

from copy import deepcopy
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _require(condition, description):
    if not condition:
        raise RuntimeError(f"Minimal startup check failed: {description}")


def demonstrate_minimal_integration(config=None):
    """Return inspectable real state after checks and owned-resource cleanup.

    Rendering, semantic memory and LLM decisions are explicitly disabled only
    for this testing entry point. Caller configuration is never mutated.
    """
    from tiny_gameplay_controller import GameplayController
    from tiny_characters import Character
    from tiny_event_handler import Event

    settings = deepcopy(config or {})
    characters = settings.setdefault("characters", {})
    count = characters.setdefault("count", 2)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Minimal startup/memory checks require at least one Character")
    characters["use_llm"] = False
    settings["headless"] = True
    settings["memory"] = {"mode": "basic"}
    settings["checkpoint"] = {"auto_enabled": False}
    logger.info("Lightweight testing profile: renderer, LLM decisions and semantic memory disabled")
    logger.info("AI remains a core requirement of the final TinyVillage product")
    controller = GameplayController(config=settings)
    try:
        _require(not controller.initialization_errors, str(controller.initialization_errors))
        _require(len(controller.characters) == count, "requested Character count")
        graph = controller.graph_manager
        _require(graph.G is graph.world_state.graph, "authoritative graph identity")
        for service in (controller.strategy_manager, controller.action_resolver, controller.event_handler):
            _require(service.graph_manager is graph, "composed service graph ownership")
        _require(controller.action_system.action_generator.graph_manager is graph,
                 "action-system graph ownership")
        for character in controller.characters.values():
            _require(isinstance(character, Character) and character._initialized,
                     "complete real Character construction")
            _require(character.graph_manager is graph and graph.G.has_node(character),
                     "Character graph registration")
            _require(character.goap_planner.graph_manager is graph, "Character planner ownership")
            _require(all(action.graph_manager is graph for action in character.character_actions),
                     "Character action ownership")
        logger.info("Startup checks passed: %d real Characters; shared graph/actions/events/time", count)

        timestamp = datetime.now()
        event = Event("Minimal Startup Check", timestamp, "general", 1, 0,
                      action_system=controller.action_system,
                      participants=[c.name for c in controller.characters.values()])
        controller.event_handler.add_event(event)
        _require(controller.event_handler.get_event(event.name) is event, "event handler registration")
        _require(graph.G.has_node(event), "event registration in the authoritative graph")
        controller.add_event_notification(f"Registered {event.name}")
        logger.info("Event registration checks passed: %s", event.name)

        for character in controller.characters.values():
            description = f"{character.name} observed {event.name}"
            character.create_memory(description, timestamp, 1)
            recent = character.recall_recent_memories()
            _require(bool(recent), "recent memory exists")
            record = recent[-1]
            _require((record.description, record.creation_time, record.importance_score)
                     == (description, timestamp, 1), "memory fields retained")
            logger.info("Memory capabilities for %s: %s", character.name,
                        character.memory_manager.get_capabilities())
        logger.info("Basic memory checks passed: %d records stored and recalled", count)
        logger.info("Gameplay turns were not exercised; actions executed: 0")
        logger.info("Minimal startup/event/memory checks passed")
        return controller
    finally:
        controller.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    try:
        demonstrate_minimal_integration()
    except Exception:
        logger.exception("Minimal startup/event/memory checks failed")
        raise SystemExit(1)
