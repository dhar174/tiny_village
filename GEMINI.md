# GEMINI.md

Entry point for Gemini CLI / Antigravity sessions in the Tiny Village
repository. Tiny Village is a single-repo Python simulation built around an
autonomous character decision loop (GOAP, graph state, memory, optional LLM
routing).

## Read first

1. Root `AGENTS.md` — the canonical contributor guide (folder layout, core
   behaviors, conventions, working agreements).
2. [`docs/agents/team-coordination.md`](docs/agents/team-coordination.md) — the
   canonical specialist-team coordination policy. Read it before any
   substantive delegated work.

Antigravity additionally loads the always-on rule
`.agents/rules/team-coordination.mdc`; keep it consistent with this file and
with `AGENTS.md`.

## Coordination contract (summary)

- The primary session is the repository coordinator and the only writer of
  repository files. It interprets objectives, recovers context, decomposes
  work, routes specialists, reconciles conflicting findings against evidence,
  makes edits, runs checks, obtains independent review, and reports.
- Specialists investigate, review, and return findings under the standard
  assignment and return contracts in the coordination policy (§6, §7). A
  specialist's recommendation is not human acceptance; accepted decisions live
  only in `docs/decisions.md`.
- Use the five-stage lifecycle proportionately; trivial changes need no
  specialists. If specialist invocation is unavailable, the primary agent
  performs the checks itself and says plainly that no independent specialist
  review occurred.
- Preserve the repository invariants in the coordination policy §9: single
  GraphManager source of truth, prompt/parse/action alignment, explicit
  integration boundaries, both headless and Pygame presentations (D013), AI
  required in the final product (D015), no permissive mocks, and no reopening
  settled decisions.
