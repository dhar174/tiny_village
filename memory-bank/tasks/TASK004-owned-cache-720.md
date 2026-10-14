# [TASK004] - Use an owned model cache (#720)

**Status:** In Progress

**Added:** 2026-10-14

**Updated:** 2026-10-14

## Original Request

Perform post-merge verification of #719/#777, then begin #720 immediately.

## Thought Process

Verify merged main before cache edits. PR #701's chmod change alone leaves a
shared tempfile path and destructive reproduction script. A stdlib helper must
configure owned caches before optional imports and serve memory and brain
loaders without overriding user paths or changing download verification.

## Implementation Plan

See [the scoped implementation plan](../../docs/superpowers/plans/2026-10-14-issue-720-owned-cache.md).

## Progress Tracking

**Overall Status:** In Progress — Runtime repair and cache regressions complete; startup validation/review/publication pending.

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 4.1 | Verify merged #719 | Complete | 2026-10-14 | 54/54 tests; cold-getter/fallback/minimal pass |
| 4.2 | Owned-cache helper | Complete | 2026-10-14 | 15 isolated filesystem/configuration checks pass |
| 4.3 | Memory/brain integration | Complete | 2026-10-14 | Full cache suite 17/17; both import orders and real loader methods |
| 4.4 | Startup validation and fresh review | In Progress | 2026-10-14 | Bounded matrix and one fresh whole-branch review |
| 4.5 | Publish evidence and PR | Pending | 2026-10-14 | #720 remains open for human review/merge |

## Progress Log

### 2026-10-14

- Confirmed #777 merged at `0ba9061a90db776d49a5959b6eb35f711754530c`;
  verified its added direct-owner regression before changing runtime code.
- User authorized #720 immediately; branch `codex/issue-720-owned-cache`
  starts from that merged main. No additional approval pause.
- Reviewed PR #701 at `8c19e2bcf6f8d517999e4456e72df269ee87a620`.
  Do not run or include its shared-cache deletion script. Candidate remains
  open for maintainer disposition; this repair supersedes its narrow patch.
- Added `tiny_cache.configure_model_cache()` with owned POSIX directories,
  descriptor-based permission changes and preserved caller overrides. Initial
  14 feature tests fail before the helper and pass after it; a stdlib-only
  import guard additionally passes.
- Import/loader regressions fail four assertions before integration, then the
  full 17-test cache suite passes. Optional model downloads are controlled
  boundaries; no full inference or Windows ACL validation is claimed.
- #773–#776 and human review/phase gates #721/#722 remain separate.
