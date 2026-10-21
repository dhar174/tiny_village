# #735 — Utility PR consolidation

## Selection and scope

Dhar174 selected **retain #703 and consolidate useful tests from #714** on
2026-10-21; the single decision record is [D028](https://github.com/dhar174/tiny_village/blob/36deba5/docs/decisions.md#d028--retain-pr-703-as-the-735-utility-consolidation-carrier).
Both original PRs change only `tiny_util_funcs.py` and add the same test path.
Both contain the identical `except (ValueError, TypeError)` fix. Keep one test
module, rebase #703 onto merged main, retain useful coverage from both, document
existing behavior and run that module in the existing core CI selection. Close
#714 as superseded only after the consolidated #703 is published and verified.
#735 closes after #703 merges, with no Phase 3 gate approval inferred.

## Baseline and evidence

- Merged main: `2c3bc5753b15aad4a1e345c020c5872d6ba8b25c`.
- Original #703: `c1528b5bf668774a0fd63cab4f81cca3b8ee2ace`.
- Original #714: `f99d49f9d3073719d28069de7751e7a3311242a9`.
- Both were draft, mergeable but behind main, with no submitted reviews or
  inline comments. Old check results do not verify a rebased head: #703 had
  CodeQL results; #714 additionally had the Python matrix and aggregate check.
- Rebased source plus consolidation:
  `80df5fcc8a7878ec3eb87b07f382c42979b81db1`.
  The original #703 author and both PRs' Jules attribution are retained.

The candidate comparison runs the exact extracted original utility modules and
existing candidate tests in fresh processes, without replacing domain objects
or fabricating outcomes. [Comparison](issue-735/comparison.json) retains all four
captures, including failures. #703 passes 10 tests on its own module and has two
TypeError errors against main (`[]`, `None`). #714 passes 12 tests and has one
TypeError error against main (`None`, which prevents its grouped test reaching
later inputs). These are reproducible current-main failures, not two different
production fixes. Candidate sources are retained as text, outside test discovery.

## Behavior retained

| Utility | Contract and limits |
| --- | --- |
| `is_numeric` | Tests whether `float(value)` succeeds without ValueError/TypeError. `None`, lists and dictionaries return False. Numeric strings, scientific notation, booleans and non-finite floats retain Python conversion behavior. This is not a finiteness check or proof that the original value supports arithmetic. Other exceptions are not swallowed. |
| `tweener` | Linear interpolation with positive `max_input`; returns `end` at/above the endpoint. Descending ranges work; negative inputs extrapolate. `steps` remains accepted but unused; no quantization or lower-bound clamp is introduced. Nonpositive input maxima remain outside the documented supported call contract. |
| `ClampedIntScore` | Retains historical magnitude scaling outside configured thresholds, not strict min/max saturation. With default bounds, 500 produces 5 and -600 produces -6. Custom bounds change thresholds, not the scaling formula. The stored score retains a fractional value; the return value truncates toward zero. No personality-trait behavior is redesigned here. |

No existing prose specification for these utilities was found outside their
source/callers and the PR descriptions. New method docstrings describe the
observed retained behavior rather than presenting the class name as a strict
clamping guarantee. Any change to that policy needs its own scoped caller and
compatibility review.

The consolidated module retains #703's scientific notation, empty/malformed
strings, default/custom ranges, setters and representations; imports #714's
reversed-range interpolation and stored-score checks; and checks fractional
stored/returned scores, negative interpolation and the compatibility-only steps
argument. There are **11 test methods**, not two competing test files.

## Validation

Python 3.12.14 / Linux x86_64, existing hash-locked environments:

```sh
/workspace/.venvs/issue726-core/bin/python -B scripts/run_ci_checks.py \
  --profile core --artifacts docs/testing/issue-735/core
```

[Core receipt](issue-735/core/summary.json): **51 tests and 3 CLI probes pass**
(32 component tests, including 11 utility methods; 19 runtime tests). Zero
failures, errors or skips. Existing worker deadlines, CLI contracts, packages
and locks are unchanged. The canonical/default CLI probes exit 0; invalid
visual/headless exits the expected 2. These probes still prove bounded startup
and basic memory, not gameplay turns or final AI readiness.

The existing `tests.test_ci_checks` suite ran in the dev environment through
`tests.subprocess_capture.capture_python` with 20s diagnostic/25s parent
limits: **18 tests pass**, no failures/errors/skips or unexpected timeout.
[Receipt and exact output](issue-735/runner-result.json) include the controlled
failure/timeout cases tested by that suite. This adds no CI jobs or dependency
packages; the core component list gains the utility module.

Runs occurred before committing the consolidation. The runner's revision field
therefore names parent `eeef271`; [source mapping](issue-735/validated-source.json)
checks the exact three changed files against committed source `80df5fc`. Retained
captures are original evidence, not rewritten to claim an earlier clean tree.

No new visual/NLP/live-model or broad legacy-suite validation is claimed. The
existing action/named-target checks use their current fixtures; #602's real
Character/Location identity failures remain separate. Hosted checks on the
updated PR must be assessed independently from these local results.
