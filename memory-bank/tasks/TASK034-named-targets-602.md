# TASK034 — #602 named targets with #816 synchronized writers first

Updated: 2026-11-25 UTC. Status: implementation and final local verification complete.
Authorization: "Implement the proposed plan." Plan:
docs/superpowers/plans/2026-11-25-issue-602-named-targets.md.
Base: origin/main7b2c913dc9bf76f1e7cfd5b7a6f419f67173500c, isolated worktree
/workspace/tiny_village-602, branch codex/issue-602-named-targets.

## Execution ledger

- Planning baseline: eight existing named-target/weather assertions pass;
  job-parser ERROR output remains. Confirmed market lost-reward and copied-actor
  nearby-target defects. Historical UUID/hash defects are historical.
- Pre-flight shared interface: Task1 supplies synchronized canonical scalar writes;
  Task2 retains the same execution path and field rollback contract. No conflict.
- Coordinator only writes/runs commands. Repository-defined read-only specialists
  use supplied evidence through Codex; native Zcode loading is not claimed.
- Ruling: use this canonical task file as the execution ledger and existing test
  artifact directories; cloud skill helper scripts are not installed locally.
  This avoids a competing status record and preserves the repo Memory Bank policy.
- First RED: market/setter/social/nested rollback, mapper/event and building
  state assertions fail on the old behavior. Two fixture mistakes (complete
  PersonalityTraits and BuildingService signature) were corrected before relying
  on their results. Sequential Action and constructor controls already pass.
- Shared updater now stages nested same-field writes; commit-phase reentry is
  rejected; other fields retain independent commits. Existing setter APIs route
  through it. Economic/building/mapped-event writers use graph-backed arithmetic.
- Ruling: managerless compatibility belongs to legacy adapters, not initialized
  native Characters stripped of their owner. Their setters/mapped writes reject
  owner loss; observed ownerless/mapping RED regressions protect this boundary.
- Corrected an introduced missing AttributeMapper import and a test's incorrect
  KeyError assumption (actual UnknownNodeError subclasses ValueError).
- Focused action/identity/legacy run:36 tests OK/exit0; adapter follow-up3 OK.
  Independent world-state review found no blocker in supplied helper/source
  excerpts; exact presence and fault-injection coverage remains in verification.
- Additional building/event check:102 tests,3 failures. Exact-base comparison is
  completed: the same3 named failures occur in all102 checks on unmodified
  main7b2c913. No fix is attributed to this change; canonical core is pending.
- First canonical core attempt: components hit the unchanged110s worker deadline
  after125 completed tests, no assertion failures/errors/skips at that point.
  Active-plan coverage exposed5 new failures: the public shared updater was made
  numeric-only, rejecting existing job='Baker' updates. These are real regressions
  to repair, not fixture changes or baseline failures.
- Ruling: preserve the shared helper's nonnumeric state-update compatibility;
  finite numeric validation applies to numeric fields/results. Existing5 RED
  active-plan cases protect job updates. Private numeric Character writers remain
  scalar operations. Plan cost: numeric validation must be field-aware.
- Ruling: repartition only existing PROFILES module selectors into contiguous
  components/world-execution/strategy batches, preserving the exact flattened
  sequence and all110s/120s/20s/25s limits. The test-CI specialist agrees this is
  the minimum existing-runner change; the first timeout receipt stays retained.

- Task1 verification: canonical repartitioned core reports236 tests and3 CLI
  probes PASS; every batch exits0, zero failures/errors/skips. Source hashes and
  complete counts are retained in the milestone receipt. Five nonnumeric
  compatibility cases observed RED->GREEN5. No worker deadline was increased.

- Task2 observed RED5 cases/3 failures (copied actor, stale handle location,
  wrong typed UUID proxy); current resolver fix GREEN16 with six legacy and five
  identity checks. Registered handles resolve before matching/proximity; optional
  irrelevant unknown visitors retain compatibility; required targets fail closed.
- Fresh independent whole-branch reviewer found2 Important compatibility issues:
  mapper/dispatcher text SET rejected, native service thirst bounds lost. Both
  observed RED2 then GREEN, plus RED/GREEN for nonfinite thirst concealment.
  Shared field-aware validation now applies before caller-specific bounds.
- Final audit also found age/beauty/community/material_goods and job setters that
  could leave existing graph fields stale. Observed remaining-writer RED, routed
  them through the same helper; job selection logic computes a local result
  before committing. Zero updates and constructor controls remain covered.
- Earlier final dev20 and visual70 PASS; core was deliberately CANCELLED109
  completed tests/zero assertions before this final writer correction. These
  receipts are historical and will not be presented as final-source verification.
- Review boundary ruling: arbitrary callback side effects and concurrent writers
  retain existing D035 limits; this task provides synchronous per-field updates,
  not a whole-action transaction or concurrent-world mutation guarantee.

## Final verification and handoff

- Final unchanged source hashes:9 files; canonical core244 tests+3CLI probes,
  dev20,visual70 PASS; all batches exit0, zero failures/errors/skips. Complete
  source hashes, commands, numeric traces and receipts:
  docs/testing/evidence/issue-602-local-results.json.
- Independent review's2 Important findings are fixed with RED/GREEN regressions;
  the later writer inventory correction also has RED/GREEN plus constructor
  control. Review scope/limits: issue-602-independent-review.json.
- Additional102 building/event tests still have exactly3 failures, independently
  reproduced on unmodified base7b2c913 and final source; parser ERROR remains.
  See docs/testing/issue-602-named-targets.md for exact cases and boundaries.
- Delivery branch codex/issue-602-named-targets targets main. Publication/PR identity
  and integration status are tracked by GitHub and the attached chat artifact.
  No merge, #742/#743 approval or product/model readiness is authorized here.
