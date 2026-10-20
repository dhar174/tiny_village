# [TASK011] — Reproducible dependency profiles (#725)

**Status:** In Progress — implementation and local validation complete; PR publication pending

**Added:** 2026-10-20

**Updated:** 2026-10-20 (America/New_York)

## Original request

“Let's do it,” after confirming #725 is ready to start following merged #784
and closed #724.

## Scope and approach

Audit actual imports and #704. Provide hash-locked Python 3.12 Linux core and
development profiles, a separate Pygame profile, explicit optional feature
profiles and reproducible install/regeneration instructions. Preserve the old
full manifest and optional capabilities under D014. Make entry-point checks
match the selected mode and remove the unnecessary FAISS requirement. Do not
claim #654's lightweight runtime or live inference is complete.

## Progress

| ID | Work | Status | Updated | Evidence |
| --- | --- | --- | --- | --- |
| 11.1 | Audit baseline and #704 | Complete | 2026-10-20 | main 8d6b254; #704 head 5636167 only removes mandatory FAISS/hint |
| 11.2 | Inputs, locks and mode checks | Complete | 2026-10-20 | Eight hash-locked profiles, pinned generator, preserved legacy manifest, main checks by mode |
| 11.3 | Clean/repeat installation and focused checks | Complete | 2026-10-20 | Four fresh env combinations; pip checks/idempotent repeats; 85 existing methods; real library probes; corrupt hash rejected |
| 11.4 | Documentation, issue update and PR | In Progress | 2026-10-20 | Setup guide and exact validation report ready; publish against main |

## Log

### 2026-10-20

- Verified #784 merged to main as 8d6b254 and #724's native blocker is closed.
- Created codex/issue-725-dependency-profiles from current main, with no worktree
  or delegation. Read core Memory Bank, accepted MVP contract and D014.
- Reviewed #704 at 56361679188f19df228291c306c3827becb916b3. Its scoped FAISS
  removal is correct; installation hints and mode-specific dependency checks
  also need the new profiles. No candidate PR is merged or approved automatically.
- Audited import boundaries: attrs, NetworkX, NumPy, Pydantic and pyparsing
  support core planning/state objects; rendering and NLP/ML remain explicit
  optional profiles. Existing full Character/no-spaCy and mocked minimal-demo
  boundaries remain under #776/#654.
- Local evidence and isolated install outputs: /workspace/scratch/issue-725/.
- Implemented core, visual, development, analytics, NLP, retrieval, CPU-LLM and
  CUDA 12.6 locks, all sharing identical pinned core packages. Ordinary default
  requirements install core with hashes/wheels only; the old broad manifest
  is preserved byte-for-byte under requirements/legacy-full.txt.
- The pinned uv helper reproduces all eight locks byte-for-byte. Torch uses
  scoped CPU/CUDA resolution and emits the needed pip wheel index without
  changing any resolved hash. A global-index attempt was rejected because it
  shadowed current requests; retained its failed log and corrected the source
  selection, not the verification contract.
- Fresh core, dev+visual, NLP and CPU-LLM installs pass; repeated commands leave
  frozen packages identical and pip checks clean. GPU is resolved only. The
  corrupt-hash negative check rejects installation into an isolated target.
- 85 existing methods pass with zero failures/errors/skips. Real core graph
  identity, mode checks with optional libraries absent, SDL-dummy Pygame,
  blank-spaCy tokenization, CPU tensor and FAISS search checks pass. No model
  weights, corpora, GPU execution or gameplay-readiness claim is supplied.
- Corrected one standalone probe's mistaken ActionSystem attribute after
  reading the class; preserved its original failure. No production behavior
  was changed for that harness mistake, and no regression suite was retried.
- Legacy minimal mode exits 0 while retaining mock patches, map initialization
  errors and contradictory tracked/executed counts. #654/#776/#740 retain the
  runtime work; do not treat this as the integrated MVP behavior proof.
- Updated README, Quickstart and dependency guide to use locked profiles and
  state current limits. Validation report: docs/testing/issue-725-dependency-profile-validation.md.
