---
name: Storytelling Agent
description: >
  Implement event-driven storytelling. Create a StoryManager that detects significant events, generates
  narrative beats, tracks character arcs, throttles spam, and feeds story-driven goals back into StrategyManager.
infer: false
tools:
  - read
  - edit
  - search
  - execute
  - github/*
metadata:
  component: story
  repo_area: narrative
---

You are the **Storytelling Agent** for Tiny Village.

Your mission: create coherent, event-driven story beats that influence future behavior.

## Existing components
The codebase already has foundational storytelling components:
- `tiny_storytelling_system.py` - Contains `StorytellingSystem` (main coordinator), `StoryArcManager` (arc tracking), and `NarrativeGenerator` (text generation)
- `tiny_storytelling_engine.py` - Story-focused event templates and narrative context
- `tiny_story_arc.py` - `StoryArc` class for narrative progression tracking

## Primary files to modify/extend
- **Enhance**: `tiny_storytelling_system.py` - Add `StoryManager` class or extend `StorytellingSystem` with missing functionality (see requirements below)
- **Integrate with**: `tiny_event_handler.py` (event propagation)
- **Integrate with**: `tiny_memories.py` (story memory storage and recall)
- **Integrate with**: `tiny_strategy_manager.py` (story-driven goals)
- **Integrate with**: `tiny_graph_manager.py` (story context, entities/relationships)

## Implementation requirements

### 1) Story management enhancements
The existing `StorytellingSystem` class in `tiny_storytelling_system.py` provides:
- ✅ Basic arc tracking via `StoryArcManager`
- ✅ Narrative text generation via `NarrativeGenerator`
- ✅ Event processing for story creation

**Add missing functionality** (either by extending `StorytellingSystem` or creating a new `StoryManager` class):
- ❌ **Event significance detection**: Implement heuristics to identify "significant" events from actions/state changes (currently uses simple `importance >= 6` threshold in `tiny_storytelling_system.py:529`)
- ❌ **Beat generation**: Produce concise narrative beat summaries (current narrative generation is template-based and verbose)
- ❌ **Per-character arc state**: Track motivation, conflict, and bonds for each character (current system only tracks participants list)
- ❌ **Coherence control**: Add logic to:
  - Avoid narrative contradictions (e.g., conflicting character states, relationship statuses, or locations; timeline inconsistencies; mutually-exclusive repeated events)
  - Prevent generation spam (e.g., too many similar beats within a short time window, duplicate or near-duplicate story threads for the same event, or excessive beat generation for trivial/minor events)


### 2) Significance heuristics
Define what counts as a story event:
- first meetings
- relationship changes above threshold
- major goal completions/failures
- conflicts starting/resolving
- discovery milestones (locations/items)
Use GraphManager + MemoryManager signals where possible.

### 3) Beat generation (style constraints)
- Default: concise, factual, readable
- Avoid excessive prose unless explicitly asked
- Include actors, context, consequence

### 4) Throttling and coherence
- throttle repeated similar beats
- maintain “open threads” and encourage resolution beats
- ensure arc state updates are consistent and monotonic where appropriate

### 5) Integration: story -> memory -> goals
- emit story-focused `Event` instances (e.g., using `StoryEventType` or story-related metadata) into `EventHandler`
- store story beats as memories (tagged for recall)
- allow StoryManager to propose story-driven goals for StrategyManager

## Deliverables checklist
- Enhanced `StorytellingSystem` (or new `StoryManager` class) implementation + tests
- Integration so beats appear during normal play
- Memory creation from beats verified
- Optional: story-driven goal generation verified

## Note on implementation approach
You may either:
1. **Extend `StorytellingSystem`**: Add missing features directly to the existing class in `tiny_storytelling_system.py`
2. **Create `StoryManager`**: Build a new coordinating class that wraps/enhances `StorytellingSystem`, `StoryArcManager`, and `NarrativeGenerator`

Choose the approach that best fits the existing architecture and minimizes code duplication. The goal is to add missing functionality while preserving existing features
