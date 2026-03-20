## Current Work Focus

- The current documentation focus is correcting stale Memory Bank files that
  still describe an unrelated `langgraph_system_generator` project.
- The active goal is to keep Memory Bank content aligned with TinyVillage's
  actual architecture: root-level runtime modules, graph-based world state,
  GOAP planning, memory systems, optional LLM routing, and simulation demos.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Immediate Next Steps

- Replace stale Memory Bank summaries that still reference
  `langgraph_system_generator` or unrelated notebook-generation workflows.
- Create and maintain the first real task record under `memory-bank/tasks/`.
- Keep the Memory Bank synchronized with the current repo entry points,
  architecture docs, testing guidance, and known demo/runtime gaps.

## Active Considerations

- Some current docs may disagree with older status notes; root `README.md` and
  root `AGENTS.md` should generally be prioritized over older archived docs
  when they conflict.
- Optional dependencies for LLM/NLP features should be documented without
  making them sound mandatory for every workflow.
- Testing guidance strongly favors realistic objects and conservative use of
  mocks, especially for memory-related tests.
