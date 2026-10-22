# `.zcode/` — Zcode workspace configuration for Tiny Village

This directory holds repository-owned Zcode configuration. Two things live
here:

- `agents/` — **source templates** for the nine `tiny-village-*` specialist
  subagents. Repository-owned, reviewed, and versioned with the code.
- `plans/` — per-session plan files written by the Zcode plan-mode UI (not part
  of the deliverable; safe to ignore or delete).

The coordination policy these agents operate under is
[`docs/agents/team-coordination.md`](../docs/agents/team-coordination.md).
Start there; it defines the routing table, assignment and return contracts,
and the one-writer rule.

## Templates vs installed agents

Official Zcode documentation places custom-agent discovery at the user-level
directory `~/.zcode/agents/<name>.md` (on Windows:
`%USERPROFILE%\.zcode\agents\<name>.md`) and states that project-level
creation is not yet available. **Observed behavior (2026-10-22, installed
Zcode version):** the running session discovered the workspace templates in
this directory and exposed all nine `tiny-village-*` agents as invocable
types without user-level installation; actual dispatch to the new types was
not completed in that session (provider-side block), so end-to-end
invocation still needs re-verification in a fresh session. Treat workspace
discovery as version-dependent: it was observed here, but the documented,
portable guarantee is still the user-level install below. Installing is a plain file copy, so one agent name maps to exactly one file
per directory; if both this repository's templates and user-level copies are
present, keep their content in sync (current Zcode versions that discover
both resolve the same names).
Editing a template has **no effect** on running sessions until it is
re-installed (if you rely on user-level copies) and a **fresh session** is
started — running sessions never hot-reload agent definitions.

## Install (first time)

Git Bash:

```bash
mkdir -p ~/.zcode/agents
cp .zcode/agents/tiny-village-*.md ~/.zcode/agents/
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.zcode\agents" | Out-Null
Copy-Item .zcode\agents\tiny-village-*.md "$HOME\.zcode\agents\"
```

Then start a **new Zcode session** in this repository and confirm the nine
`tiny-village-*` agents appear as invocable agent types.

## Update

Before overwriting an installed definition that has drifted from the template,
back it up **outside** the discovery directory (discovery is per-file, and a
leftover copy under `~/.zcode/agents/` would stay active):

```bash
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p ~/.zcode/backup/agents/$stamp
cp ~/.zcode/agents/tiny-village-*.md ~/.zcode/backup/agents/$stamp/ 2>/dev/null || true
cp .zcode/agents/tiny-village-*.md ~/.zcode/agents/
```

If an installed file you did not create (an unrelated user agent) ever shares
a name with a template, do **not** overwrite it silently — back it up, note
the conflict, and let the owner decide. Re-running install is otherwise
idempotent: same names, same paths, content replaced.

## Removal

```bash
rm ~/.zcode/agents/tiny-village-*.md
```

The templates in this repository are unaffected, and unrelated agents in
`~/.zcode/agents/` are untouched. Restart any open sessions afterwards.

## Usage

The primary Zcode session coordinates; these agents never spawn each other.
Dispatch happens from the primary session's Agent tool (or `@name` in the chat
box), using the assignment template in
[`docs/agents/team-coordination.md` §6](../docs/agents/team-coordination.md).
All nine are read-only (`Read`, `Grep`, `Glob`): they investigate, review, and
propose; the coordinator runs commands and edits files.

Example (from a primary session): ask the architecture scout to map which
subsystems a planned change touches, dispatch the matching domain specialists
in parallel, implement the change yourself from their reports, then have the
reviewer examine the result before closeout.

## Definition frontmatter (for template authors)

Fields used by these templates: `name`, `description` (required), `model:
inherit`, `tools` (exhaustive allow-list; valid identifiers are `Read`,
`Grep`, `Glob`, `Bash`, `Edit`, `Write`, `WebFetch`, `WebSearch`,
`TodoWrite`), `injectAgentsMd: true`, and `maxTurns`. Do not add unsupported
fields, other tools' schemas (e.g. Copilot `.agent.md` frontmatter or
Antigravity `view_file`-style tool names), or machine-specific absolute paths
— definitions are installed user-level and must verify the workspace is
TinyVillage from repository-relative paths.
