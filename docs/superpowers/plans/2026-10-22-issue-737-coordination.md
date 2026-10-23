# #737 — PR ownership, delivery order and Project configuration

Approved by dhar174 in chat on 2026-10-22. Start from merged main after #805.

1. Snapshot all 26 remaining PRs and the 328 unique roadmap/triage/runtime issues; retain heads, source, hierarchy, dependencies, assignees, reviews, checks and mergeability.
2. Review direct handoffs #701/#704/#709/#711/#715. Present retain/rebase, split, supersede or defer recommendations with evidence before closing or rewriting PRs. Useful code changes receive separate ordered issues.
3. Map every PR to ownership, prerequisites, gate, validation and refresh policy. Preserve existing assignees and runtime requirements #799–#803/#741/#774.
4. Preview all 12 existing Project fields and six views, preserving unrelated items and human edits. Provide a scoped resumable updater with stale checks, duplicate detection and readback; never replay the original creator.
5. Validate missing/duplicate rows, cycles/order, human gates, interrupted writes, stale state and idempotence. Publish one focused review PR. Keep #737 open until merge and verified publication; leave #738 Pending.

Codex supplies technical review; dhar174 retains human acceptance. All 328 issues are in synchronization scope. No runtime/locks/UI/package changes, PR disposition approval or phase approval is supplied. If Project access remains forbidden, supply a portable local package and retain the blocker.
