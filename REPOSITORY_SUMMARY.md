# Repository Summary: dhar174/tiny_village

## Overview
TinyVillage is a 2D simulation game where AI characters autonomously go about their lives in a dynamic village. Each AI character has a personal history, likes and dislikes, relationships, careers, and financial state.

## Technical Architecture
The core engine runs on Python (3.8+) and is predominantly Python-based. The game heavily leverages AI for character decision-making:
- **Language Model**: Uses TinyLlama (based on Llama 2), which runs locally.
- **Decision Engine**: Relies on Goal-Oriented Action Planning (GOAP) and utility-based evaluations for strategic AI planning and simulation.

### Key Components
- **Game Controllers**: `tiny_gameplay_controller.py`, `tiny_map_controller.py` coordinate the game loop and visuals.
- **AI Brain & GOAP**: `tiny_brain_io.py`, `tiny_goap_system.py`, `tiny_utility_functions.py` process character actions and decision trees.
- **Entity Management**: `tiny_characters.py`, `tiny_buildings.py`, `tiny_items.py`, `tiny_jobs.py` simulate the dynamic entities in the world.
- **Memory & Story Systems**: `tiny_memories.py`, `tiny_storytelling_engine.py`, `tiny_story_arc.py` handle short-term and long-term memories, shaping personal histories.
- **Events & State**: `tiny_event_handler.py`, `tiny_strategy_manager.py` react to changing conditions within the simulated village.

## Conclusion
TinyVillage represents a sophisticated approach to marrying local LLMs with traditional game AI (GOAP), creating highly interactive and autonomous agents in a sandbox setting.
