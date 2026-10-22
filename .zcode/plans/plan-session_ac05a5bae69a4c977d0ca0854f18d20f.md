Refresh the root AGENTS.md in place (/init) — the existing guide's structure and behavior claims are all verified accurate, so this is a targeted update to cover what's changed since it was last written. No code changes, no new files, no test edits.

## Edits to `AGENTS.md`

1. **Folder Structure section** — add the three recent additions it currently omits:
   - `requirements/`: dependency-profile system (`core`, `dev`, `visual`, `nlp`, `llm`, `retrieval`, `analytics` `.in` files) with hash-locked `locks/*.txt` files; root `requirements.txt` is a pointer file installed with `pip install --require-hashes --only-binary=:all: -r requirements.txt`.
   - `scripts/`: `run_ci_checks.py` (bounded unittest/pytest/CLI check runner used by CI, per-profile) and `compile_requirements.py`.
   - `docs/`: extend the existing bullet to also mention `docs/planning/` and `docs/superpowers/` alongside guides/reference/testing/archived.

2. **Working Agreements section** — replace/extend the "no discoverable type-check command" item with accurate command guidance:
   - Canonical test path: `python -B scripts/run_ci_checks.py --profile <core|dev|visual>` (what CI runs via `.github/workflows/python-ci.yml`; offline ML profiles run via `python-ml-profiles.yml`).
   - Keep the statement that there is no pyproject/pytest/lint/type-check config — but update it to note there is also no lint/format step in CI (still true).

3. **Overview/misc** — one-line touch-ups if needed: Python 3.12 is the pinned runtime (`.python-version`), and note that `docs/reference/AGENTS.md` + `.github/copilot-instructions.md` are supplemental (the latter is already implied by the memory-bank bullet).

## Explicitly NOT doing
- Not creating `CLAUDE.md` or any duplicate guide (repo convention: root `AGENTS.md` is canonical, avoid duplication).
- Not touching `docs/reference/AGENTS.md` (labeled supplemental; still consistent with reality).
- Not adding tests, lint, or formatting work (per the repo's ad-hoc-request agreement).

Verification: re-read the edited AGENTS.md and cross-check every path/command it names against the filesystem (all were already verified to exist during exploration).