## Current Work Focus

- #718 is complete: PR #706 merged and the issue closed on 2026-10-14.
- TASK003 / #719 repairs graph initialization and ownership on branch
  `codex/issue-719-graph-initialization`; human code review/merge remain pending.
- 53 required tests and the combined 23-test rerun pass. Cold graph access now
  completes; reduced minimal mode still executes zero actions.
- Evidence and limits: `docs/testing/issue-719-graph-initialization-validation.md`.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Immediate Next Steps

- Maintainer reviews #719's implementation/evidence before merge and closure.
- #720 cache repair follows #719. Human startup review #721 and gate #722 remain
  pending; passing scoped tests does not approve them.
- Retain #773/#774 integration/utility failures and #775/#776 Door-definition/
  optional-spaCy limits, scheduled under existing parent tasks and human gates.

## Active Considerations

- Some current docs may disagree with older status notes; root `README.md` and
  root `AGENTS.md` should generally be prioritized over older archived docs
  when they conflict.
- Optional dependencies for LLM/NLP features should be documented without
  making them sound mandatory for every workflow.
- Testing guidance strongly favors realistic objects and conservative use of
  mocks, especially for memory-related tests.
