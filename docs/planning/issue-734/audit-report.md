# #734 — Mandatory audit results and limitations

Source: merged main `13b8e374d07e7e7ea6f8f562ea6a290fcab967f7`.
Linux/Python 3.12; existing locked #726 core, development and visual environments.
Package versions are retained in [environment-manifest.json](evidence/environment-manifest.json).
No dependency installation, source repair, model download or live inference was
performed. Existing D025 evidence remains scoped; no later gate is approved.

## Results

| Issue | Executed evidence | Conclusion for review |
|---|---|---|
| #482 | Real Pygame/font/Surface, valid map asset, rainy WeatherPanel rendering; 29px rendered height and changed surface pixels | Missing-constant symptom no longer reproduces. Propose narrow completion; not full weather acceptance. |
| #355 | Real Plan/ActionWrapper/GraphManager; equal-priority insertion, four-field heap rows, replan and retained dependencies; source inspection of both real execution heap-pop consumers | Tuple-format contract is satisfied. Propose narrow completion. Replan ties use object IDs; cross-process deterministic ordering is not proven. |
| #602 | Six existing fixture-based regressions pass; corrected real Character/Action/Graph probe fails | **Keep open.** UUID effect updates miss object-keyed graph nodes; real visitor mutations expose invalid/mutable Location/Character hashing. Existing Phase 4 prerequisites remain open. |
| #596 | Actual bounded visual loop frame records tick/events/update/render/checkpoint; separate sample probe checks bounded history and disabled collection | Propose narrow instrumentation completion. The real frame logs unrelated integration errors; this is not a healthy-gameplay or frame-budget pass. |
| #597 | Empty, normal and above-budget inputs; render/update/events/checkpoint dominant-stage suggestions; oldest samples evicted at 10, exact average 10.5ms/peak 15ms; disabled collector preserves state | Propose narrow analysis-hook completion. Dominant-stage hints can appear below budget under current policy; no benchmark acceptance is claimed. |
| #609 | Two real Characters and actual MultiDiGraph edge; reader returns empty with an unhashable-dict diagnostic; actual relationship fields do not decay | **Keep open.** Existing graph-backed reader/update implementation is not correct. Delegation/status strings cannot establish a single working authority. |

The extra organization check for #594 verifies all #113–#122 and #598/#599
are native children of #594, itself under #56. Its organizational acceptance
is met; none of its graphics children is thereby implemented or approved.

## Retained runtime gaps

- **Named-target/identity:** the real #602 probe logs a missing UUID graph node.
  It then fails when reading the graph after adding actual visitors because
  `Location.__hash__` reads nonexistent `dict_or_obj`. Character hashing also
  includes mutable fields. Proposed R734-01 supplies one identity-contract
  remediation; no hashing/lookup design is chosen silently here.
- **Relationship reads/decay:** `get_character_relationships` passes `{}` as the
  third `MultiDiGraph.get_edge_data` argument (a key, not a default mapping),
  and the implementation treats keyed multiedge adjacency as flat attributes.
  The captured reader error and unchanged fields belong to existing #609.
- **Map initialization:** the live frame logs `NoneType ... locations`.
  Source inspection identifies the specific cause: MapController constructs a
  LocationManager and later resets it to None. An isolated real visual
  controller with two Characters confirms it. This is not a new global graph
  initialization deadlock. Proposed R734-02 retains this bounded repair.
- **Animation API:** the controller calls AnimationSystem.update, but the actual
  provider exposes update_all. The same isolated visual probe confirms the
  mismatch. Proposed R734-03 keeps this separate from delivering sprite art.

The first timing frame intentionally had zero configured Characters. Its
empty-character recovery diagnostics are retained but are **not** promoted into
a new general Character-startup defect. The separate two-Character probe
isolates the map/animation interfaces without that fixture limitation.

## Exact executed selection

Existing focused tests, using the #726 development interpreter:

```bash
/workspace/.venvs/issue726-dev/bin/python -m pytest -q \
  tests/test_action_named_target_resolution.py \
  tests/test_goap_planner_integration_real.py --tb=short
```

Actual captured invocation uses `pytest.main` with those arguments inside the
standard durable capture helper. Result: **12 passed, 0 failed, 0 errors,
0 skips, 2 warnings, 0 timeouts**. Warnings are deprecated `uu` and the
non-collectable helper class `TestCharacter`. The six named-target cases use
DemoCharacter/RecordingGraphManager fixtures; the planner tests also use bounded
fixture actors. Their passes are not real Character/world acceptance.

Scratch probe source is [audit_probes.py](evidence/audit_probes.py). Each named
entrypoint ran in its own fresh interpreter through the existing
`tests.subprocess_capture.capture_python` helper:

```python
capture_python(source, cwd=repo, env=environment, test_id=name,
               fatal_timeout=20, timeout=25)
```

The child source uses `runpy.run_path` with exactly one named entrypoint.
The `.source.txt` files retain the exact command source. Named environments:

- `weather`, `live_timing`, `visual_interfaces`: #726 visual interpreter.
- `queue`, `named_target`, `frame_analysis`, `social`: #726 core interpreter.
- `existing_regressions`: #726 development interpreter after correcting the
  original attempt's environment selection.

Selected execution variables: `SDL_VIDEODRIVER=dummy`,
`SDL_AUDIODRIVER=dummy`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`MPLBACKEND=Agg`, and a task-owned cache/artifact root. No credential values or
process environment dump are retained. Real Character probes copy the repository's
actual job_roles.json into an isolated temporary working directory; no fake
Character, graph, Action, provider or renderer replaces a domain object.

For local replay, select the matching interpreter/profile and entrypoint:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1 \
  /workspace/.venvs/issue726-visual/bin/python \
  docs/planning/issue-734/evidence/audit_probes.py weather
```

The **captured** executions use the 20/25-second helper deadlines above; direct
replay alone does not retain equivalent timeout/evidence guarantees. Paths in
raw run metadata describe the original machine; retained artifacts resolve by
the capture directory basename under `evidence/captures/`.

## Initial failures and corrections

[audit-results.json](evidence/audit-results.json) is the complete **12-attempt**
record, not a retry-only success summary. The initial JSON is its first seven
attempts and must not be counted again. The master record identifies the exact
retained probe-source version/hash used by each attempt.

1. Initial real Character probes lacked job_roles.json in their temporary
   directory and failed construction. Both original failures remain retained.
   Supplying the actual fixture allows the targeted application defects to be
   observed; both corrected application probes still fail.
2. Initial frame-analysis probe assumed below-budget frames must produce no
   stage hint. Current code permits a stage hint independently of the over-budget
   warning. The corrected assertion checks that policy and is explicitly a
   **probe correction, not an application fix**. Original assertion failure kept.
3. Initial pytest attempt used the core-only interpreter, which lacks pytest.
   It failed before collection. The first proper development-profile collection
   executes all 12 cases successfully. Both attempts are retained.
4. The later visual-interface probe isolates the newly observed map and
   animation concerns with two real Characters. It fails as expected against
   the actual composed interfaces; no source fix or passing retry is supplied.

Overall: **5 zero-exit captures and 7 nonzero captures**, including setup/probe
errors and real application failures. All 12 parent captures have terminal
receipts; **0 timeouts**. Of the six distinct mandatory probe scenarios,
**4 pass their scoped checks and 2 expose failures**. The separate interface
probe fails; the existing 12-test selection passes. Catalog-integrity validation
passing does not turn these runtime failures into passing evidence.
