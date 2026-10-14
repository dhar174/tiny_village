# Issue #720 Owned Model Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for sequential implementation. Track steps with checkboxes. The user authorized implementation immediately after post-merge verification; no further plan approval is required.

**Goal:** Replace shared temporary and machine-specific model caches with validated directories owned by the runtime user, retaining repeat startup and download verification.

**Architecture:** A standard-library-only `tiny_cache` helper selects and prepares the configured Hugging Face cache before optional model imports. Memory and brain loaders pass its returned model-cache directory explicitly. Configuration preserves caller overrides and never deletes or recursively changes existing cache contents.

**Tech Stack:** Python 3.12, pathlib/os/stat, existing unittest/subprocess support; no additional dependencies.

**Spec:** [Issue #720](https://github.com/dhar174/tiny_village/issues/720), candidate [PR #701](https://github.com/dhar174/tiny_village/pull/701), and the user's instruction to verify merged #777 and then begin #720 immediately.

## Global Constraints

- “Review PR #701 and verify cache ownership, permissions, and repeat startup.”
- “Exclude the repro_check.py script that deletes an existing shared cache; use an isolated temporary fixture.”
- “Preserve TLS/checksum verification and avoid importing models solely to initialize a cache.”
- Existing checkout, fresh feature branch from merged main; preserve all #719 initialization/ownership contracts.
- No cache deletion, migration, recursive chmod, global umask changes, model downloads or dependency installs for validation.
- No repair of unrelated #773/#774 fixtures, #775 Door definitions, #776 spaCy initialization, or gameplay/UI behavior.
- No automatic merge, approval, or #721/#722 human phase decisions.

## Review Focus

1. Existing or caller-configured directories retain their data; permission changes apply only to validated owned directory inodes. Task 1 covers repeat calls and sentinels.
2. Symlinks, non-directories and foreign ownership must fail before permissions or environment are changed. Task 1 covers real unsafe fixtures and controlled ownership metadata.
3. Cache overrides, including distinct legacy Transformers and Hub locations, survive import order. Tasks 1–2 cover precedence and real module imports.
4. Filesystem permission failures give an actionable cache-path error; no insecure fallback. Task 1 covers a real non-directory and controlled denied operation.
5. Cache setup works without optional models and does not disable TLS/checksum checks. Tasks 1–3 cover library-import guards, boundary calls and unchanged verification settings.

## Research and baseline

- Base main is `0ba9061a90db776d49a5959b6eb35f711754530c`, merged #777. Post-merge verification passed 54 required tests and cold-getter/fallback/minimal probes before runtime edits.
- PR #701 head `8c19e2bcf6f8d517999e4456e72df269ee87a620` is an open draft with successful CodeQL checks. Its two-line permission fix retains a predictable shared tempfile path. Its `repro_check.py` deletes an existing shared directory and executes a line-number slice; exclude it completely.
- `tiny_memories.py` chooses `tempfile.gettempdir()/hf_test_cache`, chmods it `0777`, and overwrites HF_HOME/TRANSFORMERS_CACHE after optional model imports.
- `tiny_brain_io.py` independently overwrites TRANSFORMERS_CACHE and two loader cache arguments with `/mnt/d/transformers_cache`. Restrict edits there to this same cache contract.
- [Official Hugging Face environment documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables): variables are read at library import; HF_HOME defaults to `~/.cache/huggingface` or `$XDG_CACHE_HOME/huggingface`; HF_HUB_CACHE defaults to `$HF_HOME/hub`.
- `platformdirs` is listed in full requirements but absent from the prepared environment. Standard-library selection matching Hugging Face defaults avoids requiring it merely to configure a cache.

## File map and interfaces

| File | Responsibility |
| --- | --- |
| `tiny_cache.py` (new) | `configure_model_cache() -> str`; path selection, owned-directory validation and private POSIX permissions |
| `tiny_memories.py` | Call helper before optional imports, remove shared-temp block, retain explicit loader cache argument |
| `tiny_brain_io.py` | Call helper before optional imports, remove drive-specific overrides, use returned cache for model/tokenizer |
| `tests/test_model_cache.py` (new) | Real isolated filesystem behavior and bounded module/loader integration checks |
| `docs/guides/QUICKSTART.md`, CHANGELOG, validation report | Document configuration, permissions, repeat startup, supported boundaries and evidence |
| `memory-bank/tasks/TASK004-owned-cache-720.md` and current summaries | Resume-ready progress; record #719 merged and #720 execution |

`configure_model_cache()` selects HF_HOME from its explicit setting, otherwise XDG_CACHE_HOME/huggingface, otherwise Path.home()/.cache/huggingface. HF_HUB_CACHE independently overrides HF_HOME/hub. TRANSFORMERS_CACHE, when supplied, takes precedence for these existing Transformers loaders; it is not newly introduced or overwritten. Expand `~` and use absolute filesystem paths internally. Preserve explicit variable values and unrelated security settings; publish missing HF_HOME/HF_HUB_CACHE defaults only after directory validation succeeds.

Prepare distinct selected directories idempotently. On POSIX, validate directory type and effective-user ownership before setting mode `0700`; use non-following directory descriptors for permission changes and search-only descriptors for ancestors. Reject symlink directory paths, including symlinked ancestors, and `..` components with an actionable error. Windows retains its inherited user-profile ACL behavior; do not claim POSIX modes prove Windows ACL isolation. Never alter parent permissions or existing file contents. Filesystem errors identify the offending path and advise selecting an owned HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE location.

## Ordered tasks

### Task 1: Establish owned-cache selection and filesystem behavior

**Files:** Create `tiny_cache.py` and `tests/test_model_cache.py`.

**Interfaces:** Produce `configure_model_cache() -> str` without importing runtime/model libraries. Consume environment path settings described above.

- [x] Write selection tests for default home, XDG, explicit HF_HOME, HF_HUB_CACHE, legacy TRANSFORMERS_CACHE and distinct overrides; all paths live in an owned TemporaryDirectory.
- [x] Write filesystem tests for `0700` despite permissive umask, repeat calls preserving a sentinel, existing permissive owned directories, file/symlink/ancestor-symlink rejection, foreign ownership, denied operations and unchanged environment on rejected configuration.
- [x] Run `python -m unittest tests.test_model_cache -v`; observe feature failures before creating the helper.
- [x] Implement the stdlib-only helper with the exact contract above; validate existing directories and use descriptor-based POSIX permission changes. Fail closed on invalid configuration.
- [x] Rerun the tests; require all executed assertions to pass. Commit this independently testable deliverable.

### Task 2: Wire memory and brain loaders without machine-specific side effects

**Files:** Modify `tiny_memories.py`, `tiny_brain_io.py`; extend `tests/test_model_cache.py`.

**Interfaces:** Consume Task 1 helper before optional library imports. Preserve existing public constructors/model IDs, fallback behavior and download verification options.

- [x] Add bounded fresh-process regressions for both module import orders and repeated startup. Use isolated HOME-cache settings and temporary roots; preserve an unrelated shared-cache sentinel.
- [x] Guard optional library import boundaries: before Transformers is imported, cache settings must already be configured. Isolate unavailable optional NLP/model packages to avoid real downloads; cache tests do not claim full model inference readiness.
- [x] Exercise the real EmbeddingModel and TinyBrainIO loader methods with controlled from_pretrained boundaries, asserting each model/tokenizer receives the selected cache directory, and TLS/checksum-disabling flags/settings are not added. Observe failure on current hardcoded/late setup behavior.
- [x] Call the helper at the top of both modules, remove the shared-temp chmod/environment block and Windows-drive overrides, and forward the selected directory. Keep unrelated import-time NLP/model work and inference policy outside this task.
- [x] Rerun cache and targeted startup checks. Commit integration and accompanying cache configuration documentation.

### Task 3: Validate, review and publish the scoped repair

**Files:** Validation report, CHANGELOG and Memory Bank records; runtime edits only if review finds a requirement gap.

**Interfaces:** Consume repaired startup/cache contracts. Produce one reviewable PR for #720 and preserve its dependency on completed #719.

- [x] Run cache tests and the same 54-test post-merge startup matrix, plus cold-getter/fallback/minimal probes, under isolated directories with bounded subprocess deadlines. Record executed/pass/failure/error/skipped counts and exact revision.
- [x] Verify repeated process startup preserves populated cache files, owned leaf directories have private POSIX modes, the old shared-cache sentinel remains intact, and no destructive repro script is added.
- [x] Record PR #701's reviewed revision, limitations, successful CI versus runtime evidence, and the reason the new branch supersedes its patch without merging it.
- [x] Obtain one fresh whole-branch code review, address Important/Critical findings through failing regressions followed by a green agreed matrix, and document remaining limitations accurately.
- [x] Publish a draft PR linked to #720, attach it to this task, and update issue evidence without merging or approving human gates. Draft PR #778 is published; #720 remains open pending review/merge.

## Completion and handoff

All three #720 acceptance criteria must map to executed evidence. No shared cache is deleted or trusted on sight; repeat startup and explicit configuration must work. Existing broader failures remain tracked. #721 is the subsequent human startup engineering review; #722 remains the separate phase decision. Developer UX review checks actionable path errors and clear configuration documentation; aesthetic/UI redesign remains in later planned phases.

Implementation is already authorized by the user's latest instruction. Follow the existing sequential execution method and do not pause for another plan approval.
