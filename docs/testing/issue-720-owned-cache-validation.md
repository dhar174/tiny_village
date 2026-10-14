# Issue #720: Owned model cache validation

Published in [draft PR #778](https://github.com/dhar174/tiny_village/pull/778);
human review and merge remain pending.

## Scope and revisions

The user authorized post-merge verification of #777, followed immediately by
#720 implementation. The [scoped plan](../superpowers/plans/2026-10-14-issue-720-owned-cache.md)
records that authorization and the acceptance criteria.

- Base main: `0ba9061a90db776d49a5959b6eb35f711754530c` (merged #777; #719 closed).
- Runtime/test revision: `2ec8259f7bee9426c16a986006af95dd63ebcbdc`.
- Subsequent publication commits update documentation only.
- Post-merge verification **before cache edits**: 54/54 tests, zero skips or
  timeouts, plus cold-getter, actual Goal fallback and minimal mode probes.
  See the [#719 report](issue-719-graph-initialization-validation.md#post-merge-verification--2026-10-14).

## Candidate PR and resulting repair

[PR #701](https://github.com/dhar174/tiny_village/pull/701) was reviewed at
`8c19e2bcf6f8d517999e4456e72df269ee87a620`: open draft, mergeable, successful
CodeQL checks at inspection. It changes shared-temp permissions from `0777` to
`0700`, but retains a predictable shared cache and adds `repro_check.py`, which
deletes an existing shared directory and executes a source-line slice. That
script was neither run nor included. #701 remains open for maintainer disposition;
this repair implements its intent on merged main with ownership validation.

`tiny_memories.py` previously created/chmodded `tempfile.gettempdir()/hf_test_cache`
and overwrote HF settings after optional library imports. `tiny_brain_io.py`
independently overwrote TRANSFORMERS_CACHE and loader arguments with a drive path.
Both now call the same stdlib-only helper before optional imports and explicitly
pass its selected model directory to the existing loaders.

[Official Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables),
consulted on 2026-10-14, states that environment variables are read at library
import, HF_HOME defaults to `~/.cache/huggingface` or
`$XDG_CACHE_HOME/huggingface`, and HF_HUB_CACHE defaults to `$HF_HOME/hub`.

| Configuration | Behavior |
| --- | --- |
| HF_HOME | Preserve explicit value; otherwise publish the supported default |
| HF_HUB_CACHE | Preserve explicit repository cache; otherwise HF_HOME/hub |
| TRANSFORMERS_CACHE | Preserve existing legacy override and use it for these Transformers loaders; do not introduce it when absent |
| POSIX directories | Validate effective-user ownership, reject unsafe paths, restrict selected directory inodes to `0700` |
| Existing contents/parents | No deletion, migration, recursive chmod, parent permission changes or production umask change |

Descriptor traversal does not follow symlinks; ancestors use search descriptors,
while selected leaves use descriptors suitable for permission changes. Explicit
paths containing `..` are rejected before normalization so preserved environment
values cannot point at an unvalidated different location. Errors identify the
cache path/condition and the available owned-directory configuration settings.

## Verification commands and results

Prepared Python 3.12.14 CPU environment, Linux POSIX, repository root. HF_HOME,
HF_HUB_CACHE and TMPDIR were directed to owned isolated fixtures. Dummy SDL,
two BLAS/OpenMP threads and model offline flags were set for startup checks;
the prepared plotting cache was reused for the final batches. No models or
dependencies were downloaded/installed for validation.

Executed final targets:

```bash
python -m unittest -v tests.test_graph_initialization.TestGraphImports
python -m unittest -v tests.test_graph_initialization.TestGraphLifecycle
python -m unittest discover -s tests -p test_graph_ownership.py -v
python -m unittest -v test_core_global_graph_manager
python -m unittest discover -s tests -p test_action_named_target_resolution.py -v
python -m unittest discover -s tests -p test_actions.py -v
python -m unittest discover -s tests -p test_event_handler_payloads.py -v
python -m unittest discover -s tests -p test_goap_error_handling.py -v
python -m unittest -v tests.test_model_cache
python main.py --mode minimal --no-llm --headless
```

Each batch used a 110-second fatal traceback timer and 120-second parent deadline.
Existing graph test children retain their 20-second traceback/25-second parent
limits. Cache integration children use 25/30 seconds. Required scenarios were
not skipped, assertions were not removed and timeout limits were not increased.

| Target | Executed | Passed | Failures/errors/skips |
| --- | ---: | ---: | --- |
| Cold import orders (eight subcases) | 1 | 1 | 0/0/0 |
| Graph lifecycle | 12 | 12 | 0/0/0 |
| Ownership | 11 | 11 | 0/0/0 |
| Core global manager | 9 | 9 | 0/0/0 |
| Named-target | 6 | 6 | 0/0/0 |
| Actions | 5 | 5 | 0/0/0 |
| Event payloads | 5 | 5 | 0/0/0 |
| GOAP errors | 5 | 5 | 0/0/0 |
| Cache configuration/integration | 19 | 19 | 0/0/0 |

**73/73 unique tests passed in the final bounded batches, with zero skips.**
The import batch took 51.73 seconds and lifecycle batch 91.53 seconds. The
single-module aggregate deadline failures remain documented below; they are not
reclassified as passing runs.

Cold getter exited 0 in 5.32 seconds, returning the same ready object on repeated
access with `G is world_state.graph`. The actual Goal fallback probe, with the
optional character import denied, passed its name/effects/priority/score/urgency/
attributes and explicit-zero assertions. Minimal mode exited 0 in 2.72 seconds
and still reported **zero executed actions**; it does not establish a gameplay turn.

## Filesystem and loader evidence

Cache tests exercise real isolated directories, `0700` modes independent of a
permissive umask, effective ownership, existing permissive owned directories,
file-content/mode preservation, symlink rejection, search-only parents and
ambiguous parent-traversal rejection. Foreign ownership and denied mkdir are
controlled metadata/OS-boundary cases; validation does not chown real directories.

Both fresh module import orders preserve distinct HF_HOME/HF_HUB_CACHE/
TRANSFORMERS_CACHE settings. A guard verifies that directories exist before the
Transformers import boundary. Repeated processes preserve a populated model-cache
sentinel. An unrelated isolated `hf_test_cache` sentinel and its mode `0700`
remain intact, including throughout the startup matrix.

Actual EmbeddingModel and TinyBrainIO loader methods are exercised with controlled
from_pretrained download boundaries. All four model/tokenizer calls receive the
selected cache; the brain's revision argument is retained. These fixtures write
only a small isolated marker. The stdlib helper separately runs with imports of
model/runtime libraries forbidden. Existing security environment settings and
loader verification arguments remain intact. No live transport, checksum or
inference execution is claimed.

## Test-first and independent review

- Fourteen initial feature checks fail before the helper exists, then pass;
  an additional stdlib-import guard makes the first helper suite 15/15.
- Import-order/loader regressions fail four assertions across two methods on
  the original runtime wiring, then the full 17-test cache suite passes.
- One fresh-context reviewer independently passes all 15 helper tests and finds
  two Important defects, zero Critical and zero Minor findings: read-only
  descriptors reject valid search-only ancestors, and early normalization hides
  symlink/non-directory components followed by `..`.
- Both findings are reproduced before repair: one EACCES error and two failed
  rejection subcases. Search descriptors and early parent-traversal rejection
  make the expanded 19-test cache suite pass. Fixes are verified by regressions
  and the final scoped matrix; no second reviewer dispatch is substituted for tests.

## Timing failures and execution decisions

The first parallel matrix at `8372bef` failed a Character ownership child at its
20-second timer (no Python frame) and interrupted initialization at the aggregate
110-second timer. The exact Character case then passed alone in 11.0 seconds
with unchanged limits. A subsequent sequential whole-module initialization run
at `2ec8259` completed 12 methods before its 110-second timer interrupted the last.
Both failed runs are retained; they are not green results or omitted scenarios.

The same initialization methods are therefore verified as separate import and
lifecycle batches, with the original deadlines. This does **not** establish that
the combined module completes within 110 seconds. Native-import timing varied;
the exact native delay was not established. A helper-only probe completed twenty
configurations in 0.008 seconds, providing no evidence of multi-second cache
preparation overhead. #721 should inspect this timing limitation alongside the
successful bounded batches.

Execution decisions and their costs:

1. Use the prepared lightweight environment and scoped tests; full-repository,
   heavy-model and visual behavior remain unverified.
2. Implement #701's intent on merged main instead of merging its unsafe fixture;
   the old candidate remains open for maintainer disposition.
3. Unavailable workflow helper/template resources are replaced with equivalent
   manual ledger/review records; bookkeeping is manual, required checks retained.
4. Reject `..` paths before normalization; callers must provide the direct path.
5. Model/inference/TLS/checksum execution was declined by review; downloader
   settings are preserved, but end-to-end model and transport behavior is unverified.
6. Windows ACL/platform execution was declined; Linux POSIX evidence does not
   establish Windows ACL isolation or other platform-specific behavior.
7. Startup verification remains the executor's responsibility; initial failures
   are retained and completion depends on the final result table above.
8. Broader simulation/initialization defects remain separate: #773/#774 contain
   previously observed integration/utility failures, while #775/#776 retain Door
   and spaCy limits. Those suites/features were not rerun or repaired for #720.
9. Run startup suites sequentially with the prepared plotting cache; suite-parallel
   timing remains unverified, while the eight-thread singleton scenario is retained.
10. Split initialization into bounded class batches; a single combined module
    did not meet its aggregate deadline, and that limitation remains for review.

Raw first/second-run traces, final batch logs, command/result JSON, review
findings and execution ledger are retained locally under
`/workspace/scratch/issue-720-validation/` after publication. No deferred Minor
findings were reported.

## Human review and ordering

#720 remains open pending maintainer review/merge. The new PR supersedes #701's
narrow patch; neither PR is automatically merged. #721 is the next human startup
engineering review, followed by #722's separate phase decision. Review diagnostics,
configuration precedence, filesystem boundaries and the timing limitation above;
passing tests do not supply either human decision.
