## Current Work Focus

- Current implementation work is #718: restore the utility Goal fallback and
  validate existing PR #706 against current main (TASK002, 2026-10-13).
- PR #706 is rebased and its cold imports, actual fallback, and five GOAP tests
  pass. The reduced minimal demo exits 0; broader startup is still incomplete.
- Validation evidence lives in `docs/testing/issue-718-goal-fallback-validation.md`.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Immediate Next Steps

- Maintainer reviews PR #706 and the recorded failures before merge/issue closure.
- Follow #719 for recursive global graph initialization; do not replace the lock
  without resolving import-time construction and graph ownership.
- Triage #773 (integration fixtures) and #774 (utility Goal contracts and fixture
  IDs) alongside existing ordered tasks. Human startup review #721 is pending.

## Active Considerations

- Some current docs may disagree with older status notes; root `README.md` and
  root `AGENTS.md` should generally be prioritized over older archived docs
  when they conflict.
- Optional dependencies for LLM/NLP features should be documented without
  making them sound mandatory for every workflow.
- Testing guidance strongly favors realistic objects and conservative use of
  mocks, especially for memory-related tests.
