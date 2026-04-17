# TinyVillage Repo Visualization Bundle

Generated: 2026-04-17 03:49:05 UTC

## At a Glance

- Root-level Python modules analyzed: 133
- Internal repo-local import edges: 262
- Strongest inbound dependency hub: `actions.py` (27 inbound imports)
- Runtime/core nodes rendered in the filtered graph: 38
- Root shape: 29 `tiny_*`, 26 `demo_*`, 48 `test_*`, 9 `validate_/verify_*`, 21 other

## Key Insights

- `actions.py` is the strongest inbound dependency hub across the root-level Python modules.
- `tiny_event_handler.py`, `tiny_strategy_manager.py`, and `tiny_gameplay_controller.py` form the clearest event-to-plan orchestration spine in the runtime slice.
- The repository root is unusually flat: tests and demos outnumber the core `tiny_*` modules, which makes topology views more useful than a plain file tree.

## Markdown Mindmaps

- [README.md](../markmap/README.html) - Project framing, run modes, and the high-level runtime surface.
- [AGENTS.md](../markmap/AGENTS.html) - Contributor-oriented map of the root layout, core systems, and working agreements.
- [design_docs/high_level_architecture.md](../markmap/design_docs/high_level_architecture.html) - Conceptual component map showing how orchestration, planning, graph state, memory, and map control fit together.
- [design_docs/data_flow_decision_cycle.md](../markmap/design_docs/data_flow_decision_cycle.html) - Detailed event-to-decision-to-action sequence for a character's daily behavior loop.
- [docs/README.md](../markmap/docs/README.html) - Documentation taxonomy showing how guides, reference material, analysis, and archived docs are organized.

## Mermaid Diagrams

- [runtime-topology SVG](../mermaid/runtime-topology.svg) - The runtime is organized around an orchestration spine: entrypoint to gameplay controller to event handling to strategy to actions and graph state, with the optional LLM path hanging off strategy selection. ([source](../mermaid/runtime-topology.mmd))
- [decision-cycle-sequence SVG](../mermaid/decision-cycle-sequence.svg) - The character decision loop is event-triggered and only reaches memory updates after state changes, which keeps planning, execution, and recollection as distinct phases. ([source](../mermaid/decision-cycle-sequence.mmd))
- [repo-shape SVG](../mermaid/repo-shape.svg) - The root directory is dominated by tests and demos around a smaller `tiny_*` runtime core, which explains why topology diagrams are more revealing than a simple tree listing. ([source](../mermaid/repo-shape.mmd))
- [artifact-map SVG](../mermaid/artifact-map.svg) - The repo's docs and validation artifacts mirror the runtime subsystems closely, so the most useful visual bundle needs both documentation-derived and code-derived views. ([source](../mermaid/artifact-map.mmd))

## Data-Derived Graph Views

- [Runtime import graph PNG](../graphs/runtime-import-graph.png) - Filtered to the entrypoint, action catalog, `tiny_*` modules, and direct runtime support modules so the rendered topology stays readable.
- [Module hotspots CSV](../graphs/module-hotspots.csv) - Inbound, outbound, and total internal dependency counts for every root-level Python module.
- [Import analysis summary JSON](../graphs/import-analysis-summary.json) - Machine-readable summary with counts, top hubs, and bundle insights.
- [Full import graph JSON](../graphs/runtime-import-graph.json) - Node-link export of the complete root-level import graph.
- [Full import graph GraphML](../graphs/runtime-import-graph.graphml) - Graph database or external visualization export for follow-on analysis.
