# TASK023 — Zcode specialist team and coordination integration

**Status:** In Progress — deliverables complete, independently reviewed and fixed; runtime invocation re-verification pending a fresh session  
**Added:** 2026-10-22  
**Updated:** 2026-10-22

## Request and decision

dhar174 requested a small team of custom Zcode subagents for this repository
plus durable coordination instructions for the primary Zcode session. The
deliverables are agent configuration, a canonical coordination policy,
platform entry-point integration, installation guidance, and focused
verification. Runtime gameplay changes, dependency upgrades, unrelated
documentation cleanup, external issue mutations, PR publication, and merges
are outside the task. The detailed implementation plan was approved through
the Zcode plan-mode approval gate; no `docs/decisions.md` entry was created
(D003 requires a verbatim human confirmation quote, which a plan-gate approval
does not supply).

## Plan

1. Reconnaissance: repository configuration landscape, module ownership
   verified via imports, Zcode/Antigravity platform documentation and local
   installation facts.
2. Canonical coordination policy at `docs/agents/team-coordination.md`; nine
   read-only specialist source templates under `.zcode/agents/`.
3. Entry-point integration: root `AGENTS.md` (placement reconciliation plus
   coordinator contract), root `GEMINI.md`, `.agents/rules/team-coordination.mdc`,
   and `.zcode/README.md`.
4. Install definitions to `~/.zcode/agents/`; record this task; run static
   verification; obtain independent review; report with the defined/installed/
   discovered/invoked layers kept separate.

## Progress

- Reconnaissance complete: 57 Copilot agents under `.github/agents/` (Copilot
  schema, unaffected); no prior coordination/team doc existed; Memory Bank
  task workflow confirmed (this file + `_index.md`).
- Created `docs/agents/team-coordination.md` (canonical policy: entry-point
  map, coordinator contract, one-writer rule, routing table for nine
  specialists, lifecycle, assignment/return contracts, invariants,
  escalation, fallback, verification layers).
- Created nine source templates `.zcode/agents/tiny-village-*.md`
  (architecture-scout, planning-actions, world-state, llm-contracts,
  runtime-memory, social-narrative, ui-simulation, test-ci, reviewer); all
  read-only (`Read, Grep, Glob`), `model: inherit`, `injectAgentsMd: true`.
- Created `.zcode/README.md` (templates-vs-installed, install/update/removal,
  usage, frontmatter guidance) and `GEMINI.md`; created the always-on
  Antigravity rule `.agents/rules/team-coordination.mdc`; updated root
  `AGENTS.md` placement language and added the coordinator contract.
- Installed the nine definitions to `~/.zcode/agents/` (first install; no
  prior files, no conflicts, backups not needed).
- Observed platform behavior: the running Zcode session discovered the
  workspace `.zcode/agents/` definitions as invocable agent types without
  user-level installation; recorded as version-dependent in
  `.zcode/README.md` (official docs document user-level discovery only).

## Verification

- Static checks pass: nine templates and nine installed copies parse with
  required frontmatter (unique `tiny-village-*` names, `tools` exactly
  Read/Grep/Glob, `model: inherit`, positive `maxTurns`, `injectAgentsMd:
  true`); no content drift between template and installed copies; 46
  referenced repository paths exist; routing table agrees with definitions;
  no machine-specific absolute paths; `.mdc` frontmatter valid.
- Runtime: two micro-assignments dispatched to `tiny-village-planning-actions`
  and `tiny-village-test-ci` were blocked by a provider-side filter ("unusual
  activity") and never ran; discovery succeeded but invocation is unverified.
  Re-verify in a fresh session: confirm the nine agents are listed, dispatch
  one small read-only assignment, then one two-specialist exercise.
- Independent review (read-only reviewer agent, not a repair pass) examined
  all deliverables and git scope. Findings fixed: broken D013/D014 anchor
  links in the policy; stale ML-profile names in AGENTS.md reconciled to the
  actual runner/workflow (`nlp`, `llm-cpu`; requirements listing corrected);
  world-state "enforce this" claim softened to what the tests actually check;
  ui-simulation `--headless` wording corrected (rejected with `--mode
  visual`); README backup timestamp computed once, dual-location wording and
  discovery-only claim made precise. Remaining (no action this task):
  `.mimosa/` session-runtime hook writes are platform side effects, not task
  edits.
- All fixes reapplied to the installed user-level copies; static checks
  re-run clean after the fixes.
