# SDD ledger — plan: docs/superpowers/plans/2026-10-20-issue-726-supported-profile-ci.md

**Status:** In progress
**Added/Updated:** 2026-10-20

## Request / approved scope

Implement the reviewed #726 plan. Required core/dev/visual CI; NLP and CPU-LLM
checks manual only. No automatic merge, closure or phase approval. AI remains
required in the final product (D015).

## Tasks / shared interfaces

1. Post-merge baseline and plan: complete, base eb5f4c9 (#788 merged).
2. Bounded runner: pending. Produces --profile / --artifacts and exit/report contract.
3. Required workflow: pending. Consumes runner profiles and success-only gate.
4. Manual ML workflow: pending. Consumes nlp/llm-cpu profiles; never scheduled.
5. Validation/review/PR: pending. Hosted main/manual/enforcement rollout follows merge.

Pre-flight: tasks 3/4 consume task 2's exact profile names and artifact directory.
Both require existing hash locks; no application runtime API changes.

## Rulings and evidence

- Ruling: use a feature branch in the existing clone and this resumable ledger
  plus /workspace/scratch/issue-726 — managed cloud constraint and cloud-only
  skill helper scripts; cost is manual ledger bookkeeping rather than worktree
  isolation/helper automation. Preserve evidence rather than deleting it.
- Ruling: execute the approved explicit matrix rather than bare whole-repo
  pytest — unsupported/live-model/known legacy suites are outside this plan;
  cost is no whole-repository-green claim.

## Log

- 2026-10-20: user approved implementation. Branch from merged main eb5f4c9.
  Post-merge baseline running; no runtime or dependency-lock edits.

- Task 1: complete — baseline.json retains 47 passing methods and CLI exits
  0/0/2. Task 2 RED: 12 tests fail because the bounded runner does not exist.

- Task 2: initial 12-method RED→GREEN pass. New runner uses stdlib parent and
  separate session per batch, real unittest/pytest counts and explicit failing
  evidence for empty/skipped/incomplete/nonzero/timeout/cancellation cases.
  The existing subprocess_capture interface and diagnostic limits are unchanged.
