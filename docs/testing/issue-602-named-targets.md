# #602 named-target execution and #816 direct writer synchronization

Implementation authorization: D039. Base: main
`7b2c913dc9bf76f1e7cfd5b7a6f419f67173500c`.
Plan: [implementation contract](../superpowers/plans/2026-11-25-issue-602-named-targets.md).

## Baseline and required behavior

Eight existing named-target/weather checks report OK on the base. This does not
close #602: a copied initiator handle in `current_visitors` can still receive the
nearby effect. With actor/partner energy20, the result is actor25/partner20,
instead of actor20/partner25.

#816 is also reproducible: market wealth20 plus3 changes the object to23 while
the graph remains20; the subsequent +10 Action writes30 to both, losing the
reward. The required trace is20->23->33 in both stores. The sequential energy
acceptance is20->15->25, with one graph write for each declared Action.

Historical #734 UUID-key/mutable-Location-hash failures remain historical evidence;
the current base includes #799/#220/#741 repairs. New failures are not attributed
to that old implementation.

## Synchronized writer changes

- Native scalar setters/social responses, weather, market/trade rewards,
  building fees/scalar effects, mapped event effects and the legacy event scalar
  entry point use the existing graph-backed update boundary.
- Same-field nested legacy callbacks stage their object values and commit once;
  other fields retain independent commits. Commit-phase reentry is rejected.
- Failed fields restore prior object/graph/alias values and presence. Earlier
  fields remain committed; service resource/inventory transactions are not added.
- Construction remains raw until registration; initialized native Characters
  cannot fall back to object-only writes after removal or owner loss.
- Event mapping meanings remain separate from native Action aliases. Native
  needs keep0..100 units and native happiness0..1; credits are unscaled.
- Shared helper nonnumeric updates such as job='Baker' remain compatible.

## Named-target corrections

Registered visitor handles resolve to canonical Characters before self exclusion,
requested identity, co-location and deduplication. Copied actor handles cannot
receive nearby effects. Stale visitor names/locations do not override the current
registered entity. Object requests retain concrete-type identity; UUID and name
requests remain supported.

The visitor fast path, empty-list registry fallback, optional missing-target no-op
and required-target failure remain intact. Irrelevant unknown visitors are skipped;
eligible unresolved requested targets fail preparation before writes. Named target
preparation stays ahead of mutation, with the existing per-field rollback policy.

Real target tests cover three Characters, duplicate/copy handles, overlapping
selectors, all requested-identity forms, stale locations in both directions,
ambiguity, blocked/required targets, partial failure and named drain/recovery.

## Final verification

The final corrected source passes these existing canonical profiles:

| Profile | Executed result |
| --- | --- |
| Core | 244 tests and 3 CLI probes, all PASS |
| Development | 20 tests, all PASS |
| Visual | 70 tests, all PASS |

Every batch exits 0; no assertion failures, errors or skips. Nine frozen source
hashes are retained and must match the committed files. Commands, per-batch counts,
elapsed times, lock hashes and actual child state traces are in the
[final receipt](evidence/issue-602-local-results.json).

Observed real traces include ordinary and named energy **20 → 15 → 25**,
market reward plus Action wealth **20 → 23 → 33**, and copied-actor discovery
leaving actor energy **20** while the canonical partner receives **25**.

The fresh independent reviewer found two Important compatibility defects: mapped
text assignments were rejected and native service thirst bounds were bypassed.
Both were fixed with RED/GREEN regressions. Final inventory checks also cover the
remaining numeric and job setters, without changing job selection rules.
[Review scope and corrections](evidence/issue-602-independent-review.json).

## Retained verification history

Focused action/identity/legacy verification reports36 tests OK; an adapter
follow-up reports3 OK. Initial introduced import/fixture errors and the five
job-update compatibility regressions were corrected, with the latter5 observed
RED then GREEN. Raw results remain under `/workspace/scratch/issue-602/`.

The first canonical core run is a failure: components exceeded the existing110s
worker deadline after125 completed tests; five active-plan assertions also failed
before the job-update correction. It is not a passing or complete profile.

Only existing core module selectors were repartitioned into smaller contiguous
components/world-execution/strategy batches. The flattened selector sequence,
duplicates and order are identical; all other profiles and20s/25s/110s/120s limits
remain unchanged. The completed #816 milestone reports 236 core tests and 3 CLI
probes PASS, with every batch exit 0 and no failures/errors/skips. Its earlier
source and receipts remain in [the milestone record](evidence/issue-602-milestone1.json).

An intermediate core run was deliberately cancelled after 109 completed tests
before the remaining-writer correction; its development/visual runs are historical.
Only the final corrected-source receipts above establish final verification.

## Known limitations

The additional102-check building/event suites have three failures on both the
modified tree and an independently materialized exact base7b2c913:

- `TestEffectDispatcher.test_apply_attribute_change_to_participants`
- `TestEffectDispatcher.test_apply_attribute_change_with_condition`
- `TestIntegrationScenarios.test_event_lifecycle`

Their outcomes are retained, not counted as passing or silently repaired under
this task. Known job-role parser ERROR output (117 counted/116 parsed) also
persists. Expected fault-injection/invalid-input logs are distinct from that
baseline parser issue. Successful assertion summaries do not imply clean stderr.

Independent specialists assess coordinator-supplied current source and execution
evidence; they run no commands and make no edits. Native Zcode execution is not
claimed. #742/#743 remain separate human reviews; no model/product readiness,
inventory/catalog repair or scheduler redesign is inferred.
