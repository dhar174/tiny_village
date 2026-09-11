## Why This Project Exists

- Building a believable village simulation with autonomous AI characters
  requires coordinated systems for planning, memory, world state, action
  execution, time, and character interactions.
- TinyVillage exists to explore and implement that kind of emergent simulation
  in a Python codebase where characters can make context-sensitive decisions
  and interact meaningfully with each other and the environment.

## Problems It Solves

- Provides a shared simulation framework for characters, locations, jobs,
  items, and relationships instead of isolated scripts or toy demos.
- Connects high-level character goals to concrete actions through GOAP,
  utility-based reasoning, and optional LLM-assisted decision support.
- Centralizes world knowledge in a graph-based model (`GraphManager`) so
  multiple systems operate on a common, authoritative source of truth.
- Preserves room for both lightweight demos and richer feature paths, including
  persistent memory, storytelling, analytics, and save/checkpoint systems.

## Intended User Experience

- A developer or player should be able to run `python main.py` and observe
  characters behaving autonomously inside a living village simulation.
- The system should produce understandable event → decision → action flows
  rather than opaque black-box character updates.
- Optional advanced features such as LLM-backed decision logic should enrich
  behavior without being mandatory for basic operation.

## UX Goals Reflected in the Repository

- The README presents `python main.py` as the main entry point, with `visual`,
  `minimal`, and `test` modes for different usage contexts.
- The docs tree is organized to support onboarding, technical reference,
  testing guidance, and design deep-dives.
- Demo scripts (`demo_*.py`) and integration-oriented test files support
  showing system behavior without requiring every advanced subsystem to be
  fully complete.
