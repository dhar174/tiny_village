# Runtime need and resource units

This is the #740 contract for production state and supported action descriptors.
Values are native at every magnitude: energy `5` means five points, never 50
points or a fraction inferred from its size. No saved-world migration is implied.

## Fields and bounds

| Canonical field | Unit | Direction | Character default | Scalar result bounds |
| --- | --- | --- | --- | --- |
| `energy` | points | Higher is more rested | 100 | 0–100 |
| `hunger_level` | points | Higher is hungrier; zero is full | 20 | 0–100 |
| `health_status` | points | Higher is healthier | 100 | 0–100 |
| `mental_health` | points | Higher is healthier | 80 | 0–100 |
| `social_wellbeing` | points | Higher is better | 80 | 0–100 |
| `job_performance` | points | Higher is better | 20 | 0–100 |
| `current_satisfaction` | points | Higher is more satisfied | 0 | 0–100 |
| `happiness` | fraction | Higher is happier | Computed | 0–1 |
| `wealth_money` | credits | Higher is more money | 10 | No need-scale cap |
| Food quantity | discrete items/meals | A count, not a need | Inventory-specific | Inventory contract |

These are post-effect bounds, not automatic conversion rules for constructor
arguments. Numeric/add/set effects and dynamic scalar results use shared bounds;
invalid numeric descriptors are rejected rather than silently coerced. Existing
per-attribute rollback remains responsible for both stores. Credits are not
divided, multiplied, or clamped as though they were need points.

## Names and authoritative state

Actions and planning recognize `hunger` → `hunger_level`, `health` →
`health_status`, `money`/`wealth` → `wealth_money`, and `satisfaction` →
`current_satisfaction`. Character's `satisfaction` property is a compatibility
spelling of the same field, not a second mutable value. The graph's seeded
satisfaction alias is updated within the synchronized write/rollback scope.

Registered graph values win over stale object attributes. Graph-free callers
fall back to object values. `native_character_state` copies and canonicalizes
each source before overlaying graph state. It emits consistent aliases without
mutating its inputs. Strategy dictionaries are **native inputs**, just like
Character objects; passing a previously normalized consumer snapshot back as
world state would divide need values a second time.

Legacy event-template names have their own explicit mapping in
`AttributeMapper`: template `happiness` and `satisfaction` mean social-wellbeing
points; `morale` means mental-health points. They do not identify Character's
fractional happiness or its current-satisfaction field. Use canonical event
field names for unambiguous effects. This template compatibility is distinct
from the Action aliases above. Registered native need event writes use the
synchronized updater and the same finite-result validation and bounds.

## Boundaries and scoring

World snapshots, action deltas, conditions, projections, and absolute goal
targets remain native. Positive numeric targets are minimums; negative targets
encode maximums by magnitude (hunger target `-60` means hunger ≤60).
Canonical and alias goal lookups refer to the same native value.

Only the normalized utility consumer divides need points and effective deltas
by 100. Happiness fractions and credits pass through. Effects are first
projected sequentially through `_AttributeEffect.scalar_result` against a
private native copy; the scorer receives **realizable change**, not an unclamped
declaration or the payload of `set:`. For example:

| Input | World result | Utility input |
| --- | --- | --- |
| Energy 20, Rest +15 | 35 | state .20, delta .15 |
| Energy 95, +10 then +10 | 100 | deltas .05 then 0 |
| Money 100, `set:25` | 25 credits | delta −75 credits |
| Happiness .5, hobby +.2 | .7 | delta .2 |

`calculate_native_action_utility` and `calculate_native_plan_utility` bridge
native callers to the existing normalized scorers. The older
`calculate_action_utility`/`calculate_plan_utility` public functions retain their
normalized state/delta contract. Do not feed raw runtime actions to them.
Goal bonuses use the normalized goal direction; satisfied positive minimums
have zero remainder. The existing negative-target bonus policy is retained.

Costs are dimensionless effort/time ratings: `.1` is still `.1`, with the
existing scorer cost multiplier of ten. Different catalog and strategy action
definitions may have different native deltas; a small integer does not establish
a second unit system. The legacy no-effects `Action.satisfaction` benefit
fallback remains a separate utility-rating interface. General balancing is #187.

GOAP's action-cost, action-ranking and plan-scoring consumers now use the native
bridge. `Plan._calculate_action_priority` has no state snapshot and currently
uses its existing cost-only branch; no normalized state is fabricated for it.
The distance heuristic's energy factors explicitly divide native graph energy
by 100. Its older weighted relationship branch has a separate MultiDiGraph
metadata incompatibility; see the closeout report's limitations.

## Prompts, displays, and fixtures

Strategy and prompt utility snapshots are normalized consumers. Prompt effect
projection takes a complete native snapshot, including resources and fractions.
Prompt state labels and the selected-character panel display native needs `/100`
and money in credits. The panel reads actual hunger; it does not infer hunger
from energy. Few-shot fixture states are native and converted before similarity
comparison with normalized consumer state.

Sleep is offered below 30 native energy at Home. Generated Work requires at
least 30 energy and consumes 30; blocked Work is excluded from daily selection.
Rest restores 15 energy and 8 satisfaction points. Catalog Work/Eat/Sleep have
their own smaller point deltas; their amounts are not rescaled automatically.
Shelter keeps its separate house score (1–12); monetary cutoffs stay in credits.
Motive scores and legacy goal-priority ratings labeled `/10` are not need fields.

The versioned `village-economy-v1` planning fixture uses 0–10 need units.
`fixture_units_to_native` multiplies those fixture values/deltas/thresholds by
ten on entry; `native_units_to_fixture` divides by ten for fixture reporting.
Runtime values never pass through this boundary merely because they are small.
This conversion does not implement the later economic episode or settlement.

The shared demo factory and its runtime-facing example inputs/labels use native
need points. They remain lightweight tutorial substitutes, not real-Character
acceptance evidence. `demo_llm_integration_features.py` imports isolated test
copies and retains historical `/10` example prose; it is not a production state
source. Its examples must not be passed into runtime without an explicit adapter.

See [the #740 investigation](../testing/issue-740-closeout.md) for reproductions,
executed checks, evidence limits, and the conditional issue-closure decision.
