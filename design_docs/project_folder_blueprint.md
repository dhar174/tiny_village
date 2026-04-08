# Project Folder Structure Blueprint

## Configuration Variables
* **Project Type:** Python Application (Simulation/AI Environment)
* **Monorepo:** `false`
* **Microservices:** `false`
* **Frontend Included:** `false` (Terminal/CLI and script-based visual outputs)
* **Visualization Format:** `markdown_list`
* **Max Depth:** `2` (adjusted dynamically for key areas)
* **Include Templates:** `false`

## Generated Prompt

### Initial Auto-detection Phase
The system detected a single-module Python repository focused on a simulation environment (`tiny_village`). It heavily utilizes AI integrations (LLM agents, vector embeddings), graph-based spatial relationships, goal-oriented action planning (GOAP), and diverse character implementations. The project contains an extensive test suite, substantial documentation, and custom GitHub skill configurations for Copilot.

### 1. Structural Overview
The `tiny_village` repository uses a **flat monolithic structure** at the root for core Python modules, supported by highly organized secondary directories for documentation, testing, and continuous integration.

- **Primary Paradigm:** Monolithic Python application.
- **Architectural Style:** Modular monolith where distinct gameplay/simulation systems (e.g., memory, graph management, GOAP, storytelling) are defined in separate `tiny_*.py` files, while `main.py` and `world_state.py` orchestrate them.
- **Testing Approach:** Comprehensive external test suite (`/tests`) containing unit, integration, mock-validation, and specialized feature tests.
- **Documentation:** Multi-tiered documentation separating design (`/design_docs`), active development contexts (`/memory-bank`), and technical reference (`/docs`).

### 2. Directory Visualization

- **`/`** (Root: Core Application Logic)
  - **`.github/`** (CI/CD Workflows, Prompts, and Copilot Skills)
  - **`assets/`** (Static assets like maps, e.g., `default_map.png`)
  - **`critical_analysis/`** (Analytical reports, logs, and evaluation metrics)
  - **`design_docs/`** (Design specifications and architecture documents)
  - **`docs/`** (Project reference, guides, and testing documentation)
    - `archived/`
    - `guides/`
    - `reference/`
    - `testing/`
  - **`memory-bank/`** (Active development context, rules, and task states)
  - **`tests/`** (Test suites for modules, integration, and mocking)
  - *Core Python Modules* (`main.py`, `tiny_*.py`, `actions.py`, `world_state.py`, `demo_*.py`)
  - *Configuration & Data Files* (`*.json`, `*.txt`, `*.csv`)

### 3. Key Directory Analysis

#### `/` (Root Directory)
**Purpose:** Hosts the core domain logic, simulation engine, LLM integration utilities, and entry points.
- `tiny_*.py`: Modular domain implementations (e.g., `tiny_characters.py`, `tiny_buildings.py`, `tiny_goap_system.py`, `tiny_memories.py`).
- `main.py` / `tiny_gameplay_controller.py`: Entry points and primary game loop orchestration.
- `demo_*.py`: Specialized implementation proofs and interactive demonstrations.
- `actions.py` / `social_model.py` / `world_state.py`: Core simulation state and logic.

#### `/.github/`
**Purpose:** Automation, workflows, and AI assistance configuration.
- Holds GitHub Actions CI/CD workflows.
- Contains `.github/prompts/` and `.github/skills/` tailored for custom Copilot interactions, workflow enforcement, and project planning (e.g., `breakdown-plan`, `folder-structure-blueprint-generator`, `context-map`).

#### `/tests/`
**Purpose:** Quality assurance and system verification.
- Extensive test coverage including `test_goap_*.py`, `test_llm_*.py`, `test_tiny_*.py`.
- Incorporates specific integration tests, mock setups (e.g., `mock_character.py`), and issue-specific regression validation scripts.

#### `/docs/` & `/design_docs/`
**Purpose:** Developer onboarding, architecture reference, and structural decisions.
- `/docs/guides/` & `/docs/reference/`: Technical usage and implementation guides.
- `/design_docs/`: Pre-implementation architectural choices and conceptual blueprints.

#### `/memory-bank/`
**Purpose:** Context retention for AI coding assistants and strict adherence to the Kiro-Lite workflow.
- Houses standard operating procedures (`copilot-rules.md`), active context (`activeContext.md`), and progress states.

### 4. File Placement Patterns
- **Entry Points:** `main.py` and top-level scripts (e.g., `demo_*.py`) reside directly in the root to facilitate immediate execution from the terminal.
- **Domain Logic:** Core modules prefixed with `tiny_` to prevent namespace collisions and quickly identify internal library elements.
- **Data Stores:** Flat file data (`.json`, `.txt`, `.csv`, `.pkl`, `.bin`) required at runtime are stored in root (e.g., `random_characters.json`, `l2.bin`) alongside their respective parsers.
- **Test Files:** Strictly segregated into `/tests/` with a `test_` prefix mirroring the source file they validate (e.g., `test_tiny_graph_manager.py`).

### 5. Naming and Organization Conventions
- **Files:** `snake_case` is universally applied to Python files. Prefixing (e.g., `tiny_`, `demo_`, `test_`) groups related files naturally when sorted alphabetically.
- **Directories:** `snake_case` or `kebab-case` (e.g., `memory-bank`, `design_docs`).
- **Classes/Interfaces:** Standard Python `PascalCase` definitions within the modules.
- **Test Suites:** Named descriptively `test_[module]_[focus].py` to clarify test intent without opening the file (e.g., `test_strategy_manager_event_planning.py`).

### 6. Navigation and Development Workflow
- **Getting Started:** Begin with `/README.md` and `/memory-bank/memory-bank-instructions.md` to initialize project context.
- **Finding Implementation:** Domain specific logic resides in the respective `tiny_{domain}.py` file. System orchestration lives in `tiny_gameplay_controller.py` and `world_state.py`.
- **Testing:** Navigate to `/tests/`. Tests can be executed using `pytest` test discovery from the root. Observe the rule: *Do not over-mock or fake classes if avoidable. Good tests fail when there is an error.*
- **Adding Features:** Follow the Kiro-Lite workflow: `PRD` → `Design (/design_docs)` → `Tasks` → `Code (Root)`. Register context updates in `/memory-bank/`.

### 7. Build and Output Organization
- **Dependencies:** Managed via standard Python `requirements.txt` in the root.
- **Artifacts:** Output models, `.pkl` memory dumps (e.g., `flat_access_memories.pkl`), and `.bin` embeddings are generated at runtime and generally ignored from source control via `.gitignore` (unless utilized as specific seed data).
- **Build Process:** No distinct compilation step as it is a standard interpreted Python environment.

### 8. Technology-Specific Organization
- **LLM/AI Assets:** Contains raw `.bin` files for FAISS vector embeddings (e.g., `ip_norm.bin`, `l2.bin`), and integration logic scripts like `llm_integration_utils.py` and `tiny_prompt_builder.py`.
- **Data Serialization:** Pickled cache files and JSON files are utilized extensively for state serialization and hydration efficiency across agent sessions.

### 9. Extension and Evolution
- **Refactoring Opportunity:** Moving root `.py` source modules into a `src/` or `tiny_village/` python package directory would clean up the root structure, clearly separating the application package from configurations, tests, and active scripts.
- **Asset/Data Segregation:** The root currently contains a significant number of raw static data files (`.txt`, `.json`, `.csv`). Consolidating these into a `data/` or `config/` directory would substantially improve root directory readability.

### 10. Structure Enforcement
- Review `.gitignore` periodically to ensure `.pkl`, `.bin`, and log `.txt` files do not inadvertently commit unnecessary binary bloat to the repository.
- Utilize the `.github/skills/context-map` and other GitHub skills strictly before introducing new architectural directories to ensure parity with the monolithic, context-aware design paradigm. Update `/memory-bank/` accordingly.

