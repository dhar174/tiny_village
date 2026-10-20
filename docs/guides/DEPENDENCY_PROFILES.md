# Python 3.12 dependency profiles

TinyVillage's default installation supplies the graph, planning and state
dependencies without display, NLP, model or GPU packages. Add the visual or
optional feature profiles when needed. This preserves the full TinyVillage
vision and its existing feature paths; a small default install is not a feature
removal. The accepted [MVP contract](../superpowers/specs/2026-10-20-issue-724-mvp-contract.md)
defines future release behavior, not what these package installations prove.

AI capabilities are required in the final product. Here, optional AI packages
allow lightweight testing without installing or loading those components;
passing that profile does not establish final-product readiness. See
[D015](../decisions.md#d015--require-ai-capabilities-in-the-final-product).

## Install a locked profile

The lock target is **CPython 3.12, Linux x86-64, glibc 2.28 or newer**. The MVP
support target remains Ubuntu 24.04. Current validation uses the cloud instance;
Windows, macOS, ARM and other Python versions are not established by these locks.
Keep TLS and artifact verification enabled. Run commands from the repository
root using the same interpreter that will run the application.

```bash
python3.12 -m venv .venv
# POSIX shell; select the equivalent interpreter path in other shells.
. .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
python -m pip check
```

The root `requirements.txt` selects the core lock and enables hash checking and
wheels only. The explicit flags are repeated here so installation remains
verified when selecting an individual profile directly.

```bash
# Add real Pygame for the visual surface; still no ML/model requirements.
python -m pip install --require-hashes --only-binary=:all: -r requirements/locks/visual-py312-linux-x86_64.txt

# Development tools plus core. Add the visual lock too for display-related tests.
python -m pip install --require-hashes --only-binary=:all: -r requirements/locks/dev-py312-linux-x86_64.txt
```

Each lock is a complete dependency closure including core, so a clean
environment can install a profile directly. Core versions are constrained
identically across profiles. Core + development + visual are checked together.
Arbitrary combinations of optional profiles are not claimed compatible: use a
separate environment per feature profile unless that combination is validated.
`pip install` adds packages; it does not remove extras left by an older profile.
Use a fresh environment when checking that optional packages are absent.

## Profile inventory

Inputs are reviewed in `requirements/*.in`; install the corresponding generated
`requirements/locks/<profile>-py312-linux-x86_64.txt`, not the inputs.

| Profile | Purpose and contents | Boundary |
| --- | --- | --- |
| `core` | attrs, NetworkX, NumPy, Pydantic and pyparsing, with pinned transitive dependencies | Real graph/state/action foundations. Does not require Pygame, FAISS, pandas, SciPy, scikit-learn, spaCy, Torch or Transformers. |
| `visual` | Core plus Pygame | Installs the renderer library; does not certify full visual gameplay, assets or human UX. |
| `dev` | Core plus pytest and the pinned uv resolver | Existing focused tests and lock regeneration; does not install all optional dependencies used by unrelated suites. |
| `analytics` | Core plus pandas, SciPy, scikit-learn, Matplotlib and Seaborn | Optional classic vector/statistical/plotting paths in the memory and analysis modules. |
| `nlp` | Analytics plus spaCy, NLTK, RAKE and TextBlob | Library packages only. spaCy models and NLTK corpora are separate artifacts; no models/data are downloaded by these installation commands. |
| `retrieval` | Core plus FAISS CPU | Optional vector-index capability. FAISS is not an entry-point requirement. |
| `llm-cpu` | Analytics/retrieval plus Transformers, sentence-transformers, Accelerate, tokenizers, requests and CPU Torch | No CUDA runtime or NVIDIA packages in this lock. Model weights and live inference are separate validation work. |
| `llm-gpu` | LLM packages plus CUDA 12.6 Torch, bitsandbytes and torch-geometric | Resolution target only until installed/tested on suitable GPU hardware with a compatible driver. Not part of the core install. |

For example, install NLP into its own environment using:

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements/locks/nlp-py312-linux-x86_64.txt
```

The LLM locks declare PyPI and the official PyTorch CPU or CUDA 12.6 wheel
index. uv's resolver binds the Torch packages to that backend; the generated
requirements expose the same wheel source to pip. All versions and artifact
hashes remain fixed. Do not substitute an arbitrary extra index or disable
verification to resolve an unavailable artifact.

The original broad manifest is preserved byte-for-byte in
`requirements/legacy-full.txt`. It retains historical packages and direct model
URLs for experimental paths, including GGUF/llama.cpp, image generation and
other tools outside these profiles. It is **not** the supported default or a
validated combined environment. Existing APIs and optional implementation paths
remain; platform-specific source builds, model files and unsupported legacy
combinations need their own verification. Do not install it just to hide a
failure in a core optional-dependency boundary.

## What works now and what remains separate

- `python main.py --help` does not load optional models or require a display.
- Dependency checking is mode-aware: minimal/test check core, visual also
  checks Pygame, and no mode requires FAISS as a prerequisite.
- Minimal mode uses real Characters, shared runtime services, event registration
  and basic memory. It performs no gameplay turns. It does not patch dependency
  modules or substitute mock Characters, even when AI packages are installed.
- Headless construction skips Pygame, map rendering, fonts and display setup.
  Choose `--mode minimal` explicitly; visual/headless CLI combinations are errors.
- Memory and strategy services defer heavy dependencies/model initialization
  until requested. Memory imports perform no model loading or NLTK downloads.
- Installing an NLP/LLM library or resolving a CUDA wheel does not establish
  native Windows behavior, model downloads/inference, GPU execution or game
  readiness. Those remain separate roadmap evidence and human reviews.

For bounded checks without model downloads:

```bash
export HF_HOME="$PWD/.test-artifacts/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python main.py --mode minimal --no-llm --headless
```

See the [#725 validation report](../testing/issue-725-dependency-profile-validation.md)
for exact checked profiles, commands, counts and remaining limitations.

## Memory and renderer boundaries

`MemoryManager(clock, mode="basic")` and `Character(..., memory_mode="basic")`
support real recording and recent recall without NLP, embeddings or FAISS. Each
store retains the latest 50 records, preserving description, timestamp and
importance. Disabled semantic operations raise `MemoryFeatureUnavailableError`;
they never manufacture embeddings, sentiment or retrieval results. A semantic
index filename cannot be supplied to basic mode.

The default memory mode is `full`. Its services initialize on demand and report
missing packages/model assets explicitly. `get_capabilities()` returns actual
states (`disabled`, `not_initialized`, `ready`, `unavailable`) without loading
anything. Library installation alone does not make a service ready. NLTK data
must be provisioned separately; imports do not download corpora.

`GameplayController` accepts `headless=True`, `memory.mode`, and
`characters.use_llm` independently. The minimal runner explicitly selects basic
memory/no LLM and returns inspectable state after releasing its owned resources.
The controller's `close()` is idempotent and preserves externally initialized
Pygame resources. This boundary does not implement an AI-enabled headless game
loop or establish full product readiness.

See the [#654/#776 validation report](../testing/issue-654-lightweight-runtime-validation.md)
for runtime checks, remaining job-catalog/event limitations and review evidence.

## Regenerate locks

Use Python 3.12 with uv **0.12.23**, supplied by the development lock. The helper
always resolves core first, then constrains each selected profile to that
exact core closure. It targets manylinux/glibc 2.28 and refuses source builds.

```bash
# Default: core, visual, dev.
python scripts/compile_requirements.py

# Optional profiles, resolved separately from the default install.
python scripts/compile_requirements.py analytics nlp retrieval llm-cpu llm-gpu
```

Review input/version changes, the generated transitive packages and SHA-256
hashes together. Existing outputs constrain routine regeneration; change the
relevant reviewed input constraint when intentionally upgrading. Do not edit
generated versions or hashes to work around an install failure. Optional model
artifacts retain their own verification requirements. Re-run clean install,
repeat install, `pip check` and relevant existing behavioral checks for the
profiles changed; keep failed/incomplete outcomes visible. #726 will automate
the supported checks in CI.
