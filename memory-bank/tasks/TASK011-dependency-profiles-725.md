# [TASK011] — Reproducible dependency profiles (#725)

**Status:** Completed — PR #785 merged as 8a899650; #725 closed

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
| 11.4 | Documentation, issue update and PR | Complete | 2026-10-20 | Ready PR #785 targets main and closes #725 on merge; issue handoff read back |
| 11.5 | PR #785 Pygame recovery review fix | Complete | 2026-10-20 | Quickstart selects the visual lock; fresh core-to-visual install and SDL-dummy probe pass |

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
- Published and attached [PR #785](https://github.com/dhar174/tiny_village/pull/785)
  against main, with a closing reference for #725. Validated code/locks are in
  bec88aac2a2b4b367ebb75a6340bb5f6240f4a11; subsequent edits are handoff docs.
- Posted and read back the [#725 implementation handoff](https://github.com/dhar174/tiny_village/issues/725#issuecomment-6297885936).
  Refreshed the dedicated status block and checked the three completed work
  criteria while preserving their text and native dependencies. #725 remains
  open until PR merge; no phase approval or Project field update supplied.
- Additional real pandas/SciPy/scikit-learn calculations and an Agg plot pass
  inside the NLP superset, supporting the optional analytics package boundary.
- Reviewed PR #785's sole unresolved finding, discussion_r4328579247. The
  Pygame troubleshooting command still selected the core-only requirements.
  Changed it to the same hash-verified visual lock used by the visual demo,
  explicitly using the activated virtual environment's Python.
- In a new isolated Python 3.12 environment, installed core and confirmed
  Pygame absent, then ran the exact corrected recovery command successfully.
  pip check and real Pygame 2.6.1 import/display/draw/shutdown with SDL dummy
  drivers pass. Install logs: /workspace/scratch/issue-725/review/.
  Runtime code and requirement locks are unchanged by this review fix.

- PR #785 merged as 8a899650 and closed #725. Subsequent runtime work is TASK012.
