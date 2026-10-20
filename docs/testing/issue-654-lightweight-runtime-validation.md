# Issue #654/#776 — Lightweight runtime validation

## Scope and revision

Implementation follows the [approved plan](../superpowers/plans/2026-10-20-issue-654-lightweight-runtime.md)
and D015/D016 in the [single decision record](../decisions.md). AI remains a
required final-product capability. These checks establish real startup, event
registration and basic memory, not gameplay turns or AI model readiness.

- Base: merged main `8a899650e47484add62b3269e7188de59d221870` (#785).
- Final runtime checked: `ccc2469880a04bdf03531da049a8421d0de65368`, including the
  independent review corrections. Earlier matrix at `56a1698` and its cache-test
  correction remain retained separately. Dependency locks are unchanged.
- Execution: CPython 3.12/Linux x86-64, 2026-10-20.
- Evidence retained locally: `/workspace/scratch/issue-654/`, with commands,
  stdout/stderr, baseline failures, source hashes, install receipts and durable
  child-process artifacts. No failed observation is removed or labeled PASS.
- Published receipts: [original baseline](evidence/issue-654-baseline.json) and
  [final commands, results, install receipts and source hashes](evidence/issue-654-final-results.json).

## Baseline and corrections

Core-only baseline imports fail at unconditional Pygame loading. Real memory
construction fails at `spacy.tokens` with spaCy absent. `baseline.json` retains
both original exceptions and the base revision.

Focused regressions were observed failing before implementation. Intermediate
observations exposed an additional eager NMF import, decorator-definition order,
discarded recent records during restoration, invalid sample jobs terminating the
process, visual configuration mutating caller data, and obsolete demo assertions.
Each relevant correction has a subsequent passing check; original logs remain.

The first broad cache batch had two failing import-order subcases in one method:
it expected `tiny_sr_mapping` to have loaded eagerly. The corrected test first
asserts it remains unloaded, then explicitly requests the actual mapping module
and retains the original cache-path, environment and module-identity assertions.
The corrected 27-method cache batch passes. No runtime change or deadline
increase was used to satisfy that test.

## Independent review and correction

One fresh whole-branch review of `8a899650..84e56b3` found no Critical findings,
five Important findings, and no independently actionable Minor findings. All
five were accepted and fixed in one pass in `ccc2469`:

1. Preserve requested lazy AI in the public LLM integration helper.
2. Route configured Character turns through the reusable strategy provider;
   retain configured model/status and avoid repeating failed loads.
3. Validate restored flat stores and rebind them to the receiving manager so
   restoring full-mode data cannot bypass basic-mode guards.
4. Guard theme extraction with the analysis capability boundary.
5. Initialize/clean up display and font separately, preserving externally
   initialized subsystems even when top-level Pygame is not initialized.

Regressions failed before those changes (`review-core-red.log`: six failures
among 16 methods, including both memory-mode cases and two provider cases;
`review-visual-red.log`: one failure). Afterwards the 19-method core selection
and two-method real visual selection pass. Original failures remain retained.
The complete approved matrix was then rerun once on the corrected runtime;
every selection passes, as recorded below. No second independent review was
requested. Provider tests inject only the external model-load boundary and do
not establish inference or gameplay correctness. No review finding is deferred.

## Installation and checks

Three fresh independent environments were installed using ordinary pip:

```bash
python3.12 -m venv /path/to/environment
python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
# Development environment instead selects the dev lock.
# Visual environment selects both dev and visual locks.
python -m pip check
```

Fresh environments are `/workspace/.venvs/issue654-core`, `issue654-dev` and
`issue654-visual`. Install receipts retain the full commands. All installs and
pip checks pass. Core and dev omit Pygame and AI packages; no locks changed.
The existing locked NLP and CPU-LLM environments from #725 supply the installed
dependency cases; they are reused, not described as new installations.

| Environment / selection | Executed checks | Result |
| --- | ---: | --- |
| Core: actions, named targets, GOAP errors, event payloads | 21 unittest methods | Pass |
| Core: memory runtime, headless controller, minimal runner | 19 unittest methods | Pass |
| Dev without Pygame: public minimal CLI smoke cases | 2 pytest cases | Pass |
| Visual: actual Pygame lifecycle and owned/borrowed cleanup | 2 unittest methods | Pass |
| Visual: graph import orders, lifecycle and ownership | 24 unittest methods, including 8 import-order subcases | Pass |
| Visual: owned model-cache tests, corrected lazy-mapping contract | 27 unittest methods | Pass |
| Visual: durable subprocess capture | 13 unittest methods | Pass |
| NLP installed: actual spaCy/basic runtime and controlled missing model | 2 unittest methods | Pass |
| CPU-LLM environment: actual FAISS index without embedding-model initialization | 1 unittest method | Pass |
| CPU-LLM environment: repeat minimal-runner contract with installed Torch/Transformers/FAISS blocked from import | 1 additional execution of an existing case | Pass |

Total: **111 distinct cases; 112 executions across profiles**. No skips or
timeouts in the final post-review matrix. Three standalone CLI probes additionally
return their expected exit codes (0, 0 and 2). The initial cache subcase failures
remain reported above; the earlier broad matrix is not labeled an all-green run.

Representative test commands (select the environment shown above):

```bash
python -B -m unittest -v tests.test_memory_runtime \
  tests.test_lightweight_runtime.TestHeadlessRuntime \
  tests.test_lightweight_runtime.TestMinimalRunner
python -B -m pytest -q tests/test_minimal_demo_smoke.py
python -B -m unittest -v tests.test_graph_initialization tests.test_graph_ownership
python -B -m unittest -v tests.test_model_cache tests.test_subprocess_capture
python -B -m unittest -v tests.test_memory_optional_profiles.TestInstalledNLP
python -B -m unittest -v tests.test_memory_optional_profiles.TestInstalledRetrieval
```

New cold processes retain a 20-second fatal diagnostic / 25-second parent
deadline via the existing capture helper. Existing limits are unchanged; the
updated CLI smoke keeps its 120-second parent limit. Each outer matrix batch
has a separate 120-second supervisor limit and incremental result recording.
Offline Hugging Face settings, isolated cache/artifact directories and bounded
BLAS threads are used; TLS and artifact checks remain enabled.

## Acceptance mapping

- **Core-only minimal path:** both canonical commands below exit 0 in the fresh
  core environment. They construct 2 and 5 real Characters respectively.
- **No production mocks:** the supported runner has no Mock, `sys.modules`
  replacement or simplified Character. Import guards fail the checks if heavy
  dependencies are requested, even when those packages are installed.
- **Actual state:** checked shared graph/action/planner ownership, complete
  Character initialization, event presence in handler and authoritative graph,
  and recorded/recalled description, timestamp and importance for each Character.
- **Basic memory:** independent stores, latest-50 eviction, equal-importance
  records, per-instance timestamps and restoration without eager AI setup pass.
  Restored flat stores retain the receiving manager's feature restrictions.
- **Observable failures:** disabled semantic calls raise
  `MemoryFeatureUnavailableError`; missing NLP packages/model assets retain
  their causes and report unavailable. Required Character failures propagate.
- **Renderer separation:** headless screen/clock/map/fonts/panels remain absent;
  renderer recovery stays disabled. Real SDL-dummy setup and idempotent cleanup
  pass, including preserving independently initialized display/font subsystems.
- **Retained AI integration:** helpers preserve requested enablement; controlled
  model-load success/failure proves provider reuse and observable availability,
  without performing pretrained inference.
- **CLI:** incompatible visual/headless and empty/negative minimal populations
  fail before dependency checking. A controlled required-demo failure returns 1.

```bash
python main.py --mode minimal --headless --no-llm --characters 2
python main.py --mode minimal
```

The actual `main.py --headless` command also exits 2 with explicit minimal-mode
guidance, instead of attempting display initialization.

## Limits and follow-ups

- The smoke performs **zero gameplay turns**. Economic-loop and real action
  acceptance remain #751/#658. Legacy integration/utility failures under
  #773/#774 are not rerun or declared resolved by this selection.
- Real spaCy and FAISS libraries are exercised. The missing-spaCy-model test
  injects that failure at the loader boundary. No pretrained weights, live
  inference, semantic-analysis correctness, GPU execution, native Windows/macOS
  support or final-product AI readiness is established. Those remain separate
  integration work, including #585.
- The existing job catalog logs its 117-versus-116 parsing discrepancy during
  real construction. Its broader data/role consistency is not repaired here.
  Unsupported sample roles were replaced with existing catalog roles; invalid
  job input now raises rather than terminating the interpreter. Follow-up: #787.
- Dictionary-valued `Event.impact` is currently unhashable when registering the
  Event as a graph node. This startup fixture uses the existing supported neutral
  numeric impact `0`. Event identity/equality changes are outside this PR and
  are tracked separately in #786; the failed observation remains retained.
- No blanket full-suite or full visual-gameplay claim is made. Later human
  engineering/product reviews and phase gate #732 remain separate decisions.
