# Issue #725: dependency profiles and verified installation

Validation date: **2026-10-20**, Python **3.12.14**, Linux x86-64 with glibc
**2.41**, from main `8d6b254357fadff390e3dae228acd7f1e55f5b4a`. The generated
lock target is Python 3.12/manylinux glibc 2.28; this current-instance check
does not establish native Ubuntu 24.04, Windows, macOS or ARM readiness.

## Result and scope

The default core, development/Pygame, optional NLP and optional CPU-LLM profiles
install into fresh environments with SHA-256 verification and wheels only.
Their repeat installations leave frozen package lists unchanged and `pip check`
reports no broken requirements. **85 existing test methods pass, with zero
failures, errors or skips.** New dependency/library probes also pass.

Core has nine resolved packages and excludes Pygame, FAISS, pandas, SciPy,
scikit-learn, spaCy, Torch and Transformers. Development plus visual has 16
packages, NLP 67 and CPU-LLM 52. Analytics/retrieval are resolved separately and
their functionality is checked inside the NLP/CPU-LLM supersets; standalone
clean installs for those two profiles were not separately executed. The CUDA
12.6 profile is **resolved only**, not installed or executed on GPU hardware.

All eight locks preserve identical core pins. The original full requirements
file remains byte-for-byte unchanged in `requirements/legacy-full.txt`; no
optional subsystem or implementation is removed. Documentation distinguishes
the supported default from historical/experimental feature combinations.

## Reviewed candidate PR and implementation

PR [#704](https://github.com/dhar174/tiny_village/pull/704), head
`56361679188f19df228291c306c3827becb916b3`, is a draft against old main
`de13463`. Its patch only removes the mandatory FAISS import check and the
FAISS installation hint in `main.py`. That intent is correct for the existing
optional memory/index paths. This change incorporates it on current main,
adds mode-specific dependency checks and points missing-package errors to
the verified lock rather than an unpinned package list. #704 is not merged
or automatically approved by this work; the profile PR supersedes its intent
if merged.

Minimal/test modes check attrs, NetworkX, NumPy, Pydantic and pyparsing; visual
also checks Pygame. The legacy zero-argument `check_dependencies()` retains
visual checking as its default. FAISS remains available through retrieval/LLM
profiles and is never required just to enter the application.

## Installation and integrity evidence

Fresh environments were created under `/workspace/.venvs/issue725-*`, without
system-site packages. These are separate from the prepared Phase 1 environment.
Each installed profile was reinstalled with the same command, followed by
`pip check`; before/after `pip freeze` files compare identically.

| Environment | Installed requirement(s) | Fresh / repeat / check |
| --- | --- | --- |
| `issue725-core` | Root `requirements.txt` → core lock | Pass / unchanged / pass |
| `issue725-dev` | Development and visual locks together | Pass / unchanged / pass |
| `issue725-nlp` | NLP lock | Pass / unchanged / pass |
| `issue725-llm-cpu` | CPU-LLM lock, including official CPU Torch source | Pass / unchanged / pass |

The command shape, using each environment's own interpreter, was:

```bash
python3.12 -m venv /workspace/.venvs/issue725-core
/workspace/.venvs/issue725-core/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
/workspace/.venvs/issue725-core/bin/python -m pip check
```

For the other rows, replace the environment and use
`requirements/locks/<profile>-py312-linux-x86_64.txt`; dev used both dev and
visual locks. No TLS bypass, trusted-host override, source build or changed
expected artifact hash was used. A separate deliberately corrupt temporary
attrs hash was rejected with a nonzero install result, and no package was
installed into its isolated target. Production lock hashes were not modified
for that negative check.

The pinned development uv regenerated **all eight locks byte-for-byte** from
their inputs. Core is resolved first and constrains the other profiles. Torch
is bound to its official backend source during resolution. Because uv omits
that scoped index from requirements output, the helper emits the public wheel
index afterward without changing resolved versions/hashes. Declaring that
index globally during resolution was rejected: its older requests entry would
shadow PyPI. The final CPU profile installs successfully with ordinary pip,
contains CPU Torch and has no NVIDIA/CUDA packages.

## Behavioral checks

Offline model settings, owned HF/plotting caches, two BLAS/OpenMP threads and
dummy SDL were used for the bounded startup/component checks. No model weights,
spaCy models or NLTK corpora were downloaded. Each command had a 120-second
parent bound; existing graph child 20/25-second and cache child 25/30-second
limits and assertions were unchanged. No timed-out or failed suite was retried.

| Existing target | Environment | Methods | Result | Parent elapsed |
| --- | --- | ---: | --- | ---: |
| `tests.test_actions`, `tests.test_action_named_target_resolution`, `tests.test_goap_error_handling`, `tests.test_event_handler_payloads` | Core | 21 | OK | 0.264 s |
| `tests.test_subprocess_capture` | Dev + visual | 13 | OK | 3.027 s |
| `tests.test_graph_initialization.TestGraphImports` | Dev + visual | 1 (eight orders) | OK | 1.926 s |
| `tests.test_graph_initialization.TestGraphLifecycle` | Dev + visual | 12 | OK | 3.380 s |
| `tests.test_graph_ownership` | Dev + visual | 11 | OK | 3.644 s |
| `tests.test_model_cache` | Dev + visual | 27 | OK | 1.016 s |

All use `python -m unittest -v <target(s)>`; totals are **85 executed, zero
failures/errors/skips**. Exit status and final unittest summaries were retained.
The core's absence of optional libraries is genuine, not simulated by replacing
their module entries.

Additional probes:

- **Core dependency/ownership:** minimal/test dependency checks succeed;
  visual correctly reports missing Pygame. Optional libraries are absent.
  Real cold global getter preserves graph identity and `G is world_state.graph`;
  real `ActionSystem.action_generator` retains that graph and `State` reads
  actual values. Optional embedding initialization still reports unavailable.
- **Pygame:** real SDL-dummy initialization, surface drawing/display flip and
  shutdown pass in dev/visual without FAISS. This is library verification,
  not a full UI/gameplay or human display review.
- **NLP:** spaCy blank-English tokenization produces `Tiny`, `Village`; NLTK,
  RAKE and TextBlob import. No trained model/corpus feature claim is supplied.
- **CPU-LLM/retrieval:** CPU Torch tensor arithmetic passes, Transformers and
  sentence-transformers import, and a real FAISS index returns its inserted
  vector with distance zero. No pretrained-model download or inference-quality
  validation is supplied.
- **Legacy minimal command:** `python main.py --mode minimal --no-llm --headless`
  exits 0 in 0.418 s in core. It still injects mocks, reports a MapController
  initialization error, tracks two simplified actions while game statistics
  report zero executed actions, and does not prove the production economic
  loop. Preserve #654/#776/#740 and the later scenario work rather than count
  this as real gameplay acceptance.

An initial standalone ownership probe incorrectly asserted a direct
`ActionSystem.graph_manager` attribute. The API stores it on
`action_generator.graph_manager`; the probe was corrected after inspecting
the class. The original failed probe log is retained; no runtime change was
made to accommodate it. This harness mistake is separate from the 85 existing
regression methods, all of which passed on their first run here.

## Retained evidence and remaining work

Local evidence: `/workspace/scratch/issue-725/`, including install/repeat logs,
frozen package lists, all lock SHA-256 values, corrected/original probe logs,
per-target stdout/stderr, `validation-results.json`, `validation-audit.json`
and the corrupt-hash rejection record. Child capture artifacts also remain in
the repository's ignored `.test-artifacts/subprocess/` directory. Raw paths are
local evidence, not public artifact URLs; the substantive results are above.

This completes #725's scoped profile/installation work for review. #654 still
owns real renderer-free initialization and removal of production mock patches;
#726 owns required CI. #727 owns subsequent UX planning. Preserve D014's broader
feature vision and the separate #730/#731/#732 and release decisions. GPU
execution, native Windows/macOS, live model artifacts/inference, GGUF source
builds and arbitrary optional-profile combinations remain unverified. No phase
gate or Project Decision field is automatically approved or updated.
