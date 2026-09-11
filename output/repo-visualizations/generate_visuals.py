#!/usr/bin/env python3
"""Generate a repo visualization bundle for TinyVillage."""

from __future__ import annotations

import ast
import csv
import html
import json
import math
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("matplotlib is required to render the import graph.") from exc

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("networkx is required to export the import graph.") from exc


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parent.parent
MARKMAP_DIR = BUNDLE_ROOT / "markmap"
MERMAID_DIR = BUNDLE_ROOT / "mermaid"
GRAPHS_DIR = BUNDLE_ROOT / "graphs"
INDEX_DIR = BUNDLE_ROOT / "index"
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

MARKDOWN_TARGETS = [
    (
        "README.md",
        "Project framing, run modes, and the high-level runtime surface.",
    ),
    (
        "AGENTS.md",
        "Contributor-oriented map of the root layout, core systems, and working agreements.",
    ),
    (
        "design_docs/high_level_architecture.md",
        "Conceptual component map showing how orchestration, planning, graph state, memory, and map control fit together.",
    ),
    (
        "design_docs/data_flow_decision_cycle.md",
        "Detailed event-to-decision-to-action sequence for a character's daily behavior loop.",
    ),
    (
        "docs/README.md",
        "Documentation taxonomy showing how guides, reference material, analysis, and archived docs are organized.",
    ),
]

RUNTIME_SUPPORT_MODULES = {
    "actions",
    "main",
    "world_state",
    "graph_analytics",
    "goap_evaluator",
    "llm_character_utils",
    "llm_integration_utils",
    "social_model",
    "effect_dispatcher",
    "effect_schema",
    "storytelling_integration",
    "recent_event_list",
}


def log(message: str) -> None:
    print(f"[visuals] {message}")


def resolve_command(*names: str) -> str:
    for name in names:
        candidate = shutil.which(name)
        if candidate:
            return candidate
    raise SystemExit(f"Unable to find any of: {', '.join(names)}")


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    log("Running: " + " ".join(command))
    subprocess.run(
        command,
        cwd=cwd or REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def clean_output_tree() -> None:
    for directory in (MARKMAP_DIR, MERMAID_DIR, GRAPHS_DIR, INDEX_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def markmap_output_path(relative_path: str) -> Path:
    source = Path(relative_path)
    return (MARKMAP_DIR / source).with_suffix(".html")


def render_markmaps() -> list[dict[str, str]]:
    markmap_cmd = resolve_command("markmap", "markmap.cmd")
    outputs: list[dict[str, str]] = []

    for relative_path, insight in MARKDOWN_TARGETS:
        source_path = REPO_ROOT / relative_path
        output_path = markmap_output_path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                markmap_cmd,
                str(source_path),
                "-o",
                str(output_path),
                "--no-open",
                "--offline",
            ]
        )
        outputs.append(
            {
                "source": relative_path,
                "output": output_path.relative_to(BUNDLE_ROOT).as_posix(),
                "insight": insight,
            }
        )

    return outputs


def extract_local_imports(file_path: Path, local_modules: set[str]) -> set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                base = imported.name.split(".")[0]
                if base in local_modules and base != file_path.stem:
                    imports.add(base)
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            base = node.module.split(".")[0]
            if base in local_modules and base != file_path.stem:
                imports.add(base)

    return imports


def graph_family(module_name: str) -> str:
    if module_name == "main":
        return "entrypoint"
    if module_name == "actions":
        return "actions"
    if module_name.startswith("tiny_"):
        return "tiny"
    if module_name.startswith("demo_"):
        return "demo"
    if module_name.startswith("test_"):
        return "test"
    if module_name.startswith("validate_") or module_name.startswith("verify_"):
        return "validation"
    return "support"


def shape_family(module_name: str) -> str:
    if module_name.startswith("tiny_"):
        return "tiny_*"
    if module_name.startswith("demo_"):
        return "demo_*"
    if module_name.startswith("test_"):
        return "test_*"
    if module_name.startswith("validate_") or module_name.startswith("verify_"):
        return "validate_/verify_*"
    return "other"


def is_runtime_core(module_name: str) -> bool:
    return module_name.startswith("tiny_") or module_name in RUNTIME_SUPPORT_MODULES


def analyze_import_graph() -> tuple[nx.DiGraph, nx.DiGraph, dict[str, object]]:
    python_files = sorted(REPO_ROOT.glob("*.py"))
    local_modules = {file_path.stem for file_path in python_files}
    full_graph = nx.DiGraph()

    for file_path in python_files:
        module_name = file_path.stem
        full_graph.add_node(
            module_name,
            path=file_path.name,
            family=graph_family(module_name),
            shape_family=shape_family(module_name),
        )

    for file_path in python_files:
        module_name = file_path.stem
        for imported_module in sorted(extract_local_imports(file_path, local_modules)):
            full_graph.add_edge(module_name, imported_module, relation="imports")

    hotspot_rows: list[dict[str, object]] = []
    for module_name in sorted(full_graph.nodes()):
        in_degree = int(full_graph.in_degree(module_name))
        out_degree = int(full_graph.out_degree(module_name))
        total_degree = in_degree + out_degree
        core_flag = is_runtime_core(module_name)
        full_graph.nodes[module_name]["in_degree"] = in_degree
        full_graph.nodes[module_name]["out_degree"] = out_degree
        full_graph.nodes[module_name]["total_degree"] = total_degree
        full_graph.nodes[module_name]["is_runtime_core"] = int(core_flag)
        hotspot_rows.append(
            {
                "module": module_name,
                "path": f"{module_name}.py",
                "family": graph_family(module_name),
                "shape_family": shape_family(module_name),
                "is_runtime_core": core_flag,
                "inbound": in_degree,
                "outbound": out_degree,
                "total": total_degree,
            }
        )

    hotspot_rows.sort(
        key=lambda row: (-int(row["total"]), -int(row["inbound"]), str(row["module"]))
    )

    core_nodes = [
        node
        for node in full_graph.nodes()
        if is_runtime_core(node) and (full_graph.degree(node) > 0 or node in {"main", "actions"})
    ]
    core_graph = full_graph.subgraph(core_nodes).copy()

    shape_counts = Counter(shape_family(file_path.stem) for file_path in python_files)
    top_inbound = [
        {"module": name, "count": count}
        for name, count in Counter(dict(full_graph.in_degree())).most_common(15)
    ]
    top_outbound = [
        {"module": name, "count": count}
        for name, count in Counter(dict(full_graph.out_degree())).most_common(15)
    ]

    orchestration_modules = [
        name
        for name in (
            "tiny_event_handler",
            "tiny_strategy_manager",
            "tiny_gameplay_controller",
            "tiny_graph_manager",
        )
        if name in full_graph
    ]
    orchestration_hubs = sorted(
        orchestration_modules,
        key=lambda name: (
            -int(full_graph.in_degree(name)),
            -int(full_graph.out_degree(name)),
            name,
        ),
    )

    summary = {
        "generated_at": GENERATED_AT,
        "module_count": len(python_files),
        "edge_count": full_graph.number_of_edges(),
        "core_runtime_node_count": core_graph.number_of_nodes(),
        "shape_counts": dict(shape_counts),
        "top_inbound": top_inbound,
        "top_outbound": top_outbound,
        "orchestration_hubs": orchestration_hubs,
        "key_insights": [
            f"`{top_inbound[0]['module']}.py` is the strongest inbound dependency hub across the root-level Python modules."
            if top_inbound
            else "No import hotspots were detected.",
            "`tiny_event_handler.py`, `tiny_strategy_manager.py`, and `tiny_gameplay_controller.py` form the clearest event-to-plan orchestration spine in the runtime slice.",
            "The repository root is unusually flat: tests and demos outnumber the core `tiny_*` modules, which makes topology views more useful than a plain file tree.",
        ],
        "hotspots": hotspot_rows,
    }

    return full_graph, core_graph, summary


def export_graph_artifacts(
    full_graph: nx.DiGraph,
    core_graph: nx.DiGraph,
    summary: dict[str, object],
) -> dict[str, str]:
    graphml_path = GRAPHS_DIR / "runtime-import-graph.graphml"
    json_path = GRAPHS_DIR / "runtime-import-graph.json"
    png_path = GRAPHS_DIR / "runtime-import-graph.png"
    hotspots_csv_path = GRAPHS_DIR / "module-hotspots.csv"
    summary_path = GRAPHS_DIR / "import-analysis-summary.json"

    nx.write_graphml(full_graph, graphml_path)

    json_path.write_text(
        json.dumps(nx.node_link_data(full_graph), indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with hotspots_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "module",
                "path",
                "family",
                "shape_family",
                "is_runtime_core",
                "inbound",
                "outbound",
                "total",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["hotspots"])

    render_runtime_graph(core_graph, png_path)

    return {
        "graphml": graphml_path.relative_to(BUNDLE_ROOT).as_posix(),
        "json": json_path.relative_to(BUNDLE_ROOT).as_posix(),
        "png": png_path.relative_to(BUNDLE_ROOT).as_posix(),
        "csv": hotspots_csv_path.relative_to(BUNDLE_ROOT).as_posix(),
        "summary": summary_path.relative_to(BUNDLE_ROOT).as_posix(),
    }


def render_runtime_graph(graph: nx.DiGraph, output_path: Path) -> None:
    color_map = {
        "entrypoint": "#c084fc",
        "actions": "#fb923c",
        "tiny": "#38bdf8",
        "support": "#34d399",
    }

    figure_size = (18, 13)
    plt.figure(figsize=figure_size)
    node_count = max(graph.number_of_nodes(), 1)
    layout = nx.spring_layout(
        graph,
        seed=7,
        k=2.0 / math.sqrt(node_count),
    )

    node_colors = []
    node_sizes = []
    labels = {}
    for node_name in graph.nodes():
        family = graph.nodes[node_name].get("family", "support")
        node_colors.append(color_map.get(family, "#94a3b8"))
        degree = int(graph.nodes[node_name].get("total_degree", 1))
        node_sizes.append(1100 + (degree * 170))
        labels[node_name] = node_name

    nx.draw_networkx_edges(
        graph,
        pos=layout,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#64748b",
        width=1.2,
        alpha=0.35,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#0f172a",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(
        graph,
        pos=layout,
        labels=labels,
        font_size=8,
        font_family="DejaVu Sans",
        font_weight="bold",
    )

    legend_handles = []
    for family, color in color_map.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=family,
                markerfacecolor=color,
                markeredgecolor="#0f172a",
                markersize=10,
            )
        )

    plt.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        title="Module family",
    )
    plt.title(
        "TinyVillage runtime/core import graph\n"
        "Filtered to the entrypoint, action catalog, tiny_* modules, and direct runtime support modules.",
        fontsize=15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_mermaid_sources(summary: dict[str, object]) -> list[dict[str, str]]:
    shape_counts = summary["shape_counts"]
    module_count = int(summary["module_count"])

    runtime_topology = """flowchart LR
    Main["main.py\\nEntry point"] --> Controller["tiny_gameplay_controller.py\\nMain loop and orchestration"]
    Controller --> Event["tiny_event_handler.py\\nEvent detection and routing"]
    Event --> Strategy["tiny_strategy_manager.py\\nGoal selection and routine updates"]
    Strategy --> GOAP["tiny_goap_system.py\\nGOAP planning"]
    Strategy --> Actions["actions.py\\nAction catalog and execution rules"]
    GOAP --> Actions
    GOAP --> Graph["tiny_graph_manager.py\\nShared world model"]
    Controller --> Graph
    Graph --> Strategy
    Graph --> Memories["tiny_memories.py\\nMemory storage and retrieval"]
    Graph --> Map["tiny_map_controller.py\\nMap and presentation"]
    Strategy --> Prompt["tiny_prompt_builder.py\\nPrompt construction"]
    Prompt --> Brain["tiny_brain_io.py\\nModel I/O"]
    Brain --> Interpret["tiny_output_interpreter.py\\nDecision parsing"]
    Interpret --> Actions
    Actions --> Graph
    Actions --> Memories
    Controller --> Story["tiny_storytelling_system.py\\nNarrative hooks"]
    Controller --> Animation["tiny_animation_system.py\\nVisual feedback"]
    Globals["tiny_globals.py\\nGlobal accessors"] -.-> Graph
    """

    decision_cycle = """sequenceDiagram
    participant EH as tiny_event_handler.py
    participant Main as main.py
    participant GC as tiny_gameplay_controller.py
    participant SM as tiny_strategy_manager.py
    participant GM as tiny_graph_manager.py
    participant GOAP as tiny_goap_system.py
    participant PB as tiny_prompt_builder.py
    participant IO as tiny_brain_io.py
    participant OI as tiny_output_interpreter.py
    participant ACT as actions.py
    participant MEM as tiny_memories.py

    EH->>Main: New day or gameplay event
    Main->>GC: Advance simulation tick
    GC->>SM: Refresh character strategy
    SM->>GM: Read character state and world context
    GM-->>SM: Current graph-backed state
    SM->>GOAP: Evaluate goals and candidate plans
    GOAP->>GM: Query difficulty and dependencies
    GM-->>GOAP: Graph-derived constraints
    GOAP-->>SM: Prioritized plan or action options
    SM->>PB: Build prompt/context package
    PB->>IO: Send optional model request
    IO-->>OI: Raw model response
    OI-->>GC: Executable action choices
    GC->>ACT: Execute selected action
    ACT->>GM: Apply state changes
    ACT->>MEM: Record outcome memory
    MEM-->>GC: Updated memory context
    """

    repo_shape = f"""pie title Root-level Python file families ({module_count} modules)
    "tiny_* core runtime ({shape_counts.get('tiny_*', 0)})" : {shape_counts.get('tiny_*', 0)}
    "demo_* experiments ({shape_counts.get('demo_*', 0)})" : {shape_counts.get('demo_*', 0)}
    "test_* regressions ({shape_counts.get('test_*', 0)})" : {shape_counts.get('test_*', 0)}
    "validate_/verify_ smoke checks ({shape_counts.get('validate_/verify_*', 0)})" : {shape_counts.get('validate_/verify_*', 0)}
    "other support modules ({shape_counts.get('other', 0)})" : {shape_counts.get('other', 0)}
    """

    artifact_map = """flowchart TD
    subgraph Runtime["Runtime code"]
        Main["main.py"]
        Tiny["tiny_*.py core systems"]
        Actions["actions.py"]
        Support["world_state.py, graph_analytics.py, llm_* helpers"]
    end

    subgraph Docs["Documentation"]
        Readme["README.md"]
        Agents["AGENTS.md"]
        Design["design_docs/"]
        DocsReadme["docs/README.md"]
    end

    subgraph Quality["Validation surfaces"]
        Demos["demo_*.py"]
        Tests["test_*.py and tests/"]
        Checks["validate_*.py and verify_*.py"]
    end

    subgraph Bundle["Generated visualization bundle"]
        Markmap["markmap/*.html"]
        Mermaid["mermaid/*.mmd + *.svg"]
        Graphs["graphs/*.graphml *.json *.png *.csv"]
        Index["index/README.md + index.html"]
    end

    Readme --> Bundle
    Agents --> Bundle
    Design --> Bundle
    DocsReadme --> Bundle
    Runtime --> Bundle
    Quality --> Bundle
    Main --> Tiny
    Tiny --> Actions
    Tiny --> Support
    Demos --> Runtime
    Tests --> Runtime
    Checks --> Runtime
    Bundle --> Index
    """

    return [
        {
            "name": "runtime-topology",
            "source": runtime_topology,
            "insight": "The runtime is organized around an orchestration spine: entrypoint to gameplay controller to event handling to strategy to actions and graph state, with the optional LLM path hanging off strategy selection.",
        },
        {
            "name": "decision-cycle-sequence",
            "source": decision_cycle,
            "insight": "The character decision loop is event-triggered and only reaches memory updates after state changes, which keeps planning, execution, and recollection as distinct phases.",
        },
        {
            "name": "repo-shape",
            "source": repo_shape,
            "insight": "The root directory is dominated by tests and demos around a smaller `tiny_*` runtime core, which explains why topology diagrams are more revealing than a simple tree listing.",
        },
        {
            "name": "artifact-map",
            "source": artifact_map,
            "insight": "The repo's docs and validation artifacts mirror the runtime subsystems closely, so the most useful visual bundle needs both documentation-derived and code-derived views.",
        },
    ]


def render_mermaid_diagrams(summary: dict[str, object]) -> list[dict[str, str]]:
    npx_cmd = resolve_command("npx", "npx.cmd")
    outputs: list[dict[str, str]] = []

    for diagram in build_mermaid_sources(summary):
        source_path = MERMAID_DIR / f"{diagram['name']}.mmd"
        svg_path = MERMAID_DIR / f"{diagram['name']}.svg"
        source_path.write_text(diagram["source"] + "\n", encoding="utf-8")
        run_command(
            [
                npx_cmd,
                "--yes",
                "@mermaid-js/mermaid-cli",
                "-i",
                str(source_path),
                "-o",
                str(svg_path),
                "-t",
                "neutral",
                "-b",
                "transparent",
            ]
        )
        outputs.append(
            {
                "name": diagram["name"],
                "source": source_path.relative_to(BUNDLE_ROOT).as_posix(),
                "svg": svg_path.relative_to(BUNDLE_ROOT).as_posix(),
                "insight": diagram["insight"],
            }
        )

    return outputs


def write_index_files(
    markmaps: list[dict[str, str]],
    mermaid_outputs: list[dict[str, str]],
    graph_outputs: dict[str, str],
    summary: dict[str, object],
) -> None:
    readme_path = INDEX_DIR / "README.md"
    html_path = INDEX_DIR / "index.html"

    top_hub = summary["top_inbound"][0]["module"]
    top_hub_count = summary["top_inbound"][0]["count"]
    module_count = summary["module_count"]
    edge_count = summary["edge_count"]
    shape_counts = summary["shape_counts"]

    markmap_lines = []
    for item in markmaps:
        markmap_lines.append(
            f"- [{item['source']}](../{item['output']}) - {item['insight']}"
        )

    mermaid_lines = []
    for item in mermaid_outputs:
        mermaid_lines.append(
            f"- [{item['name']} SVG](../{item['svg']}) - {item['insight']} "
            f"([source](../{item['source']}))"
        )

    readme = f"""# TinyVillage Repo Visualization Bundle

Generated: {GENERATED_AT}

## At a Glance

- Root-level Python modules analyzed: {module_count}
- Internal repo-local import edges: {edge_count}
- Strongest inbound dependency hub: `{top_hub}.py` ({top_hub_count} inbound imports)
- Runtime/core nodes rendered in the filtered graph: {summary['core_runtime_node_count']}
- Root shape: {shape_counts.get('tiny_*', 0)} `tiny_*`, {shape_counts.get('demo_*', 0)} `demo_*`, {shape_counts.get('test_*', 0)} `test_*`, {shape_counts.get('validate_/verify_*', 0)} `validate_/verify_*`, {shape_counts.get('other', 0)} other

## Key Insights

- {summary['key_insights'][0]}
- {summary['key_insights'][1]}
- {summary['key_insights'][2]}

## Markdown Mindmaps

{chr(10).join(markmap_lines)}

## Mermaid Diagrams

{chr(10).join(mermaid_lines)}

## Data-Derived Graph Views

- [Runtime import graph PNG](../{graph_outputs['png']}) - Filtered to the entrypoint, action catalog, `tiny_*` modules, and direct runtime support modules so the rendered topology stays readable.
- [Module hotspots CSV](../{graph_outputs['csv']}) - Inbound, outbound, and total internal dependency counts for every root-level Python module.
- [Import analysis summary JSON](../{graph_outputs['summary']}) - Machine-readable summary with counts, top hubs, and bundle insights.
- [Full import graph JSON](../{graph_outputs['json']}) - Node-link export of the complete root-level import graph.
- [Full import graph GraphML](../{graph_outputs['graphml']}) - Graph database or external visualization export for follow-on analysis.
"""
    readme_path.write_text(readme, encoding="utf-8")

    markmap_cards = []
    for item in markmaps:
        markmap_cards.append(
            "<li>"
            f"<a href=\"../{html.escape(item['output'])}\">{html.escape(item['source'])}</a>"
            f"<p>{html.escape(item['insight'])}</p>"
            "</li>"
        )

    mermaid_cards = []
    for item in mermaid_outputs:
        mermaid_cards.append(
            "<section class=\"card\">"
            f"<h3>{html.escape(item['name'])}</h3>"
            f"<p>{html.escape(item['insight'])}</p>"
            f"<p><a href=\"../{html.escape(item['svg'])}\">Open SVG</a> | "
            f"<a href=\"../{html.escape(item['source'])}\">Open source</a></p>"
            f"<img src=\"../{html.escape(item['svg'])}\" alt=\"{html.escape(item['name'])} diagram\">"
            "</section>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TinyVillage Repo Visualization Bundle</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --surface: #ffffff;
      --line: #dbe4ee;
      --text: #0f172a;
      --muted: #475569;
      --accent: #0284c7;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #eef6ff 0%, var(--bg) 240px);
      color: var(--text);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
    }}
    h1, h2, h3 {{
      line-height: 1.2;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.6;
    }}
    a {{
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin: 1rem 0 2rem;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1rem 1.1rem;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }}
    .card img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      margin-top: 0.75rem;
    }}
    ul {{
      padding-left: 1.25rem;
    }}
    .hero {{
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 1.5rem;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .metrics {{
      display: grid;
      gap: 0.75rem;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-top: 1rem;
    }}
    .metric {{
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0.9rem;
    }}
    .metric strong {{
      display: block;
      color: var(--text);
      font-size: 1.35rem;
      margin-bottom: 0.2rem;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>TinyVillage Repo Visualization Bundle</h1>
      <p>Generated on {html.escape(GENERATED_AT)} from docs plus static import analysis. This bundle is architecture-first: it highlights the decision loop, the runtime topology, and the flat root-level module landscape.</p>
      <div class="metrics">
        <div class="metric"><strong>{module_count}</strong>root Python modules analyzed</div>
        <div class="metric"><strong>{edge_count}</strong>repo-local import edges</div>
        <div class="metric"><strong>{html.escape(top_hub)}</strong>top inbound hub</div>
        <div class="metric"><strong>{shape_counts.get('test_*', 0)}</strong>root test modules</div>
      </div>
    </section>

    <section>
      <h2>Key insights</h2>
      <ul>
        <li>{html.escape(summary['key_insights'][0])}</li>
        <li>{html.escape(summary['key_insights'][1])}</li>
        <li>{html.escape(summary['key_insights'][2])}</li>
      </ul>
    </section>

    <section>
      <h2>Markdown mindmaps</h2>
      <ul>
        {''.join(markmap_cards)}
      </ul>
    </section>

    <section>
      <h2>Mermaid diagrams</h2>
      <div class="grid">
        {''.join(mermaid_cards)}
      </div>
    </section>

    <section>
      <h2>Data-derived graph views</h2>
      <div class="grid">
        <section class="card">
          <h3>Runtime import graph</h3>
          <p>Filtered to the entrypoint, action catalog, <code>tiny_*</code> modules, and direct runtime support modules so the rendered graph stays readable.</p>
          <p><a href="../{html.escape(graph_outputs['png'])}">Open PNG</a> | <a href="../{html.escape(graph_outputs['summary'])}">Summary JSON</a></p>
          <img src="../{html.escape(graph_outputs['png'])}" alt="Runtime import graph">
        </section>
        <section class="card">
          <h3>Machine-readable exports</h3>
          <p>Use the CSV, JSON, and GraphML artifacts for deeper analysis or to import the graph into other tools.</p>
          <ul>
            <li><a href="../{html.escape(graph_outputs['csv'])}">module-hotspots.csv</a></li>
            <li><a href="../{html.escape(graph_outputs['json'])}">runtime-import-graph.json</a></li>
            <li><a href="../{html.escape(graph_outputs['graphml'])}">runtime-import-graph.graphml</a></li>
          </ul>
        </section>
      </div>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


def main() -> int:
    log("Cleaning output directories")
    clean_output_tree()

    log("Rendering Markdown mindmaps")
    markmaps = render_markmaps()

    log("Analyzing root-level Python imports")
    full_graph, core_graph, summary = analyze_import_graph()
    graph_outputs = export_graph_artifacts(full_graph, core_graph, summary)

    log("Rendering Mermaid diagrams")
    mermaid_outputs = render_mermaid_diagrams(summary)

    log("Writing landing pages")
    write_index_files(markmaps, mermaid_outputs, graph_outputs, summary)

    log(f"Visualization bundle ready at {BUNDLE_ROOT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[visuals] External tool failed with exit code {exc.returncode}", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
