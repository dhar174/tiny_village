# Memory Bank Instructions

This repository's Memory Bank should stay grounded in the actual TinyVillage codebase and documentation, not in template assumptions from unrelated projects.

## Project Context
- TinyVillage is a 2D simulation game where autonomous AI characters live in a dynamic village.
- Core simulation concerns include characters, memories, relationships, buildings, items, jobs, events, time progression, GOAP-style planning, and utility-based decision making.
- The primary runtime entry point described in the repository README is `python main.py`, with support for `visual`, `minimal`, and `test` modes.

## Memory Bank Expectations
- At the start of each task, read all core Memory Bank files under `memory-bank/`.
- Treat `memory-bank/projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, and `progress.md` as the canonical summary of current repository context.
- Keep these files synchronized with the real repository state described in `README.md`, `docs/README.md`, `AGENTS.md`, design docs, and current code.
- When a file contains stale details from another architecture or project, replace them with TinyVillage-specific content instead of layering contradictory notes on top.

## Documentation Sources to Prefer
- `README.md` for top-level project goals and run commands.
- `docs/README.md` for the documentation map and which docs are current vs historical.
- `AGENTS.md` and `docs/reference/AGENTS.md` for contributor guidance and architecture background.
- `design_docs/` for system-level architecture and design deep-dives.
- `docs/reference/` for current technical reference.
- `docs/testing/` for testing expectations and conventions.

## Task Tracking Guidance
- Use `memory-bank/tasks/_index.md` as the task registry.
- Create task files as `memory-bank/tasks/TASKID-taskname.md`.
- Each task file should preserve the original request, thought process, implementation plan, progress tracking, and dated progress log entries.
- When task progress changes, update both the task file and `memory-bank/tasks/_index.md` on the same date.

## Current Repository-Specific Notes
- The repository contains substantial documentation beyond the root README; do not assume the README alone is complete.
- The docs tree was reorganized on 2025-12-26 to distinguish current documentation from archived historical materials.
- Historical documents under `docs/archived/` should not override current implementation guidance.
- If Memory Bank files currently describe `langgraph_system_generator` or notebook-generation workflows, treat that as stale context to be corrected.

## Update Standard
When updating the Memory Bank:
1. Verify changes against the repository's current files.
2. Prefer concrete file and subsystem names from TinyVillage.
3. Record active work and known limitations in a way that helps future sessions resume accurately.
4. Keep instructions concise, specific, and tied to this repository.
