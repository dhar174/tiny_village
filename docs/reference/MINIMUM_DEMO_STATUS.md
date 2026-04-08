# Minimum Demo Requirements - Implementation Status

## Executive Summary

This document tracks **verified demo-readiness facts** rather than a percentage
estimate.

On 2026-04-08, the following checks were run in a fresh clone of the repository:

- `python main.py --help` ✅
- `python main.py --mode minimal --headless` ❌
- `python demo_minimal_integration.py` ❌
- `python test_integration_minimal.py` ❌

The current state is:

- `main.py` **does** exist and is the correct top-level entry point.
- `assets/default_map.png` **does** exist.
- The CLI surface for `visual`, `minimal`, and `test` modes is present.
- A bare environment without installed dependencies cannot start the runtime.
- In this clone, the direct minimal demo and minimal integration test scripts
  currently fail during import because `tiny_utility_functions`,
  `tiny_goap_system`, and `tiny_characters` end up with
  `NameError: Goal is not defined`.

## Verified Repository Facts ✅

### Entry Points and Assets

1. **Entry point present**
   - ✅ `main.py` exists at the repository root.
   - ✅ `python main.py --help` succeeds and documents `visual`, `minimal`,
     `test`, and `headless` options.

2. **Packaged map asset present**
   - ✅ `assets/default_map.png` exists in the repository.

3. **Current docs should not describe `main.py` or the default map asset as missing**
   - ✅ Those older claims are historical only.

## Current Blockers Observed in Verification 🔧

### 1. Required Dependencies Must Be Installed

`python main.py --mode minimal --headless` currently exits early in a bare
environment with missing dependency checks for:

- `pygame`
- `networkx`
- `numpy`
- `pydantic`
- `faiss-cpu`

This means the existence of a headless/minimal CLI mode does **not** imply that
the project runs without installing the documented Python dependencies first.

### 2. Demo/Test Scripts Are Not Currently Verified as Passing

In this clone:

- `python demo_minimal_integration.py` fails during import with
  `NameError: Goal is not defined`
- `python test_integration_minimal.py` fails during import with the same error

Those failures occur before the scripts reach the earlier, more optimistic
"works now" assertions that appeared in older documentation revisions.

### 3. Visual Mode Still Requires Environment-Specific Validation

`python main.py --mode visual` was not validated in this headless environment.
Even once dependencies are installed, visual mode still depends on a functioning
local pygame/display setup.

## Recommended Validation Sequence

### 1. Install documented dependencies

```bash
python3.12 -m pip install -r requirements.txt
```

### 2. Confirm the entry-point CLI

```bash
python main.py --help
```

### 3. Re-check the runnable paths after dependencies are installed

```bash
python main.py --mode minimal --headless
python demo_minimal_integration.py
python test_integration_minimal.py
```

### 4. Validate visual mode only on a display-capable target machine

```bash
python main.py --mode visual
```

## Notes on Older Claims

Earlier versions of this file were correct to remove outdated "missing
`main.py`" and "missing map asset" claims, but they went too far in the other
direction by describing the demo paths as already working. The current
documentation should distinguish between:

- **Repository facts that are verified** (`main.py` exists, assets exist, CLI is
  present)
- **Runtime claims that still require validation** (demo scripts passing,
  minimal mode running end-to-end, visual mode working on a target machine)

## Conclusion

The minimum-demo story is **partially aligned** rather than "done":

- ✅ The repo now contains the documented entry point and packaged map asset.
- ✅ The CLI surface is present and discoverable.
- ❌ The minimal demo and minimal integration scripts are not currently verified
  as passing in this clone.
- ❌ A bare environment still fails before runtime until the required
  dependencies are installed.

Use this file as a verification checklist, not as a claim that the demo is
already running end-to-end everywhere.
