## Current Work Focus

- #718 is complete: PR #706 merged and the issue closed on 2026-10-14.
- TASK003 / #719 is complete: PR #777 merged as `0ba9061`; #719 closed.
- Post-merge verification passes 54/54 tests plus cold-getter/fallback/minimal
  probes. Reduced minimal mode still executes zero actions.
- TASK004 / #720 is implementing owned cache configuration on branch
  `codex/issue-720-owned-cache`; 81 unique checks pass in final bounded batches (27 cache).
  The single-module initialization deadline limitation is retained for review.
- Evidence and limits: `docs/testing/issue-719-graph-initialization-validation.md`.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Immediate Next Steps

- Codex/Copilot review fixes for PR #778 are implemented; 81 unique checks
  pass in final batches, including one cold-import rerun after an initial native
  timeout. Original failures and platform limits are retained. Maintainer merge
  and #721/#722 human decisions remain pending.
- Human startup review #721 and gate #722 follow; passing scoped tests does
  not approve either decision.
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
