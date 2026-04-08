# Tiny Village - System Integration Status

## Summary

This file is a high-level integration reference. Earlier revisions described the
system as fully demo-ready, but current verification shows that conclusion was
too strong.

## What Is Verified

### Entry Point and Assets ✅

- `main.py` exists at the repository root.
- `python main.py --help` succeeds.
- `assets/default_map.png` exists.
- The CLI exposes `visual`, `minimal`, and `test` modes, plus `--headless`.

### Integration-Oriented Subsystems Exist ✅

- `GameplayController`, `ActionResolver`, `CheckpointManager`, and the related
  graph/strategy modules are present in the codebase.
- The documentation no longer needs to describe `main.py` or the default map as
  missing.

## What Is Not Currently Verified

### Demo Scripts in This Clone ❌

Direct checks in a fresh clone on 2026-04-08 produced:

- `python demo_minimal_integration.py` → import-time failure
- `python test_integration_minimal.py` → import-time failure

Both currently fail before demo execution because the import path through
`tiny_utility_functions`, `tiny_goap_system`, and `tiny_characters` reaches
`NameError: Goal is not defined`.

### Bare-Environment Runtime ❌

`python main.py --mode minimal --headless` currently stops at dependency checks
in a bare environment and reports missing required packages such as `pygame`,
`networkx`, `numpy`, `pydantic`, and `faiss-cpu`.

### Visual Mode Still Needs Target-Machine Validation ⚠️

Visual mode may work on a machine with the required dependencies and a valid
pygame display environment, but that was not re-validated here.

## How to Run

### Quick Start

```bash
# Install dependencies
python3.12 -m pip install -r requirements.txt

# Inspect the available runtime modes
python main.py --help

# Re-check minimal/headless mode after dependencies are installed
python main.py --mode minimal --headless

# Re-check the direct demo and integration scripts
python demo_minimal_integration.py
python test_integration_minimal.py

# Try visual demo on a display-capable machine
python main.py --mode visual
```

### Advanced Options

```bash
python main.py --characters 3
python main.py --no-llm
python main.py --fps 30
python main.py --headless
python main.py --verbose
```

## Integration Architecture

### Core Integration Loop

```text
Character Turn
    ↓
Strategy Manager (get_daily_actions)
    ↓
Action Resolver (resolve_action)
    ↓
Action Execution (execute)
    ↓
Effect Application (update character state)
    ↓
Graph Manager (update relationships)
    ↓
Memory System (store experience)
    ↓
Event Handler (process consequences)
```

### Error Handling Chain

```text
Action Execution Fails
    ↓
Try Strategy Manager Actions
    ↓
Try GOAP Fallback
    ↓
Try Simple Rest Action
    ↓
Minimal Energy Update
```

### System Integration

```text
Event Detected
    ↓
EventHandler.check_events()
    ↓
StrategyManager.update_strategy(events)
    ↓
GameplayController.apply_decision()
    ↓
Action Execution
    ↓
State Updates
```

## Working Assumptions to Re-Validate

### Core Systems

- ✅ GameplayController - Main game loop
- ✅ EventHandler - Event detection and processing
- ✅ StrategyManager - Decision coordination
- ✅ ActionResolver - Action execution
- ✅ GOAPPlanner - Intelligent planning
- ✅ GraphManager - Relationship tracking
- ✅ StorytellingSystem - Narrative generation
- ✅ CheckpointManager - Auto-save
- ✅ Analytics - Performance tracking

### Integration Points

- ✅ Event → Strategy → Action pipeline
- ✅ Character turn processing
- ✅ Action resolution and execution
- ✅ Error handling and recovery
- ✅ Fallback mechanisms
- ✅ Performance monitoring
- ✅ State persistence

### Error Handling

- ✅ LLM timeout (falls back to GOAP)
- ✅ Invalid JSON (falls back to safe action)
- ✅ Invalid action (provides fallback)
- ✅ Plan failure (replans or falls back)
- ✅ Memory errors (continues with partial data)
- ✅ Subsystem failures (don't crash sim)

These items describe architectural intent and implemented components, but they
should not be read as proof that the demo paths currently pass in every
environment.

## Recommendations

1. Treat `docs/reference/MINIMUM_DEMO_STATUS.md` as the current source of truth
   for demo-readiness notes.
2. Re-run the demo/test commands after installing `requirements.txt`
   dependencies.
3. Fix the current `Goal` import failure before restoring any "works now" claims
   about `demo_minimal_integration.py` or `test_integration_minimal.py`.
4. Validate visual mode on the actual target machine instead of assuming
   headless verification covers it.

## Conclusion

The repository contains the documented integration pieces and the current entry
point, but the stronger "system integration complete" wording from older
revisions is no longer accurate enough. Keep this file as an architectural
overview and use current verification output before claiming demo readiness.
