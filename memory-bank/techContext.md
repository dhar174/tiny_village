## Technology Stack

- Python-only repository targeting Python 3.12 with code primarily in root-level
  modules and a `tests/` subdirectory for organized test coverage.
- Core simulation and planning technologies include:
  - `networkx.MultiDiGraph` for graph-based world state in `tiny_graph_manager.py`
  - GOAP-style planning in `tiny_goap_system.py`
  - `pygame` for the visual runtime and map rendering
  - Optional NLP/embedding/LLM integrations for memory and decision systems
    (e.g., transformers, sentence-transformers, spaCy, TinyLlama)

## Development Setup

- The README describes standard Python setup via
  `python3.12 -m pip install -r requirements.txt`.
- The main documented runtime entry point is:
  ```
  python main.py
  ```
- Additional documented run modes:
  - `python main.py --mode visual` — full pygame display
  - `python main.py --mode minimal` — console output, no display
  - `python main.py --mode test` — integration test verification
  - `python main.py --no-llm` — disable LLM, use fallback logic
  - `python main.py --verbose` — enable debug logging

## Configuration Model

- Some optional runtime behavior depends on environment variables such as
  `TRANSFORMERS_CACHE`, `PYGAME_DISPLAY_WIDTH`, and `PYGAME_DISPLAY_HEIGHT`.
- LLM- and NLP-related functionality may require additional dependencies beyond
  the minimal runtime requirements listed in `requirements.txt`.
- The project uses both core runtime modules and auxiliary scripts:
  - `demo_*.py` — subsystem demos and integration experiments
  - `test_*.py` (root-level) — integration and regression tests
  - `validate_*.py` / `verify_*.py` — ad hoc smoke-check scripts

## Technical Constraints

- Optional advanced features may depend on heavier libraries such as
  transformers, sentence-transformers, spaCy, or related NLP tooling; those
  should remain optional and degrade gracefully.
- Visual runtime flows can be sensitive to pygame display initialization and
  map/rendering setup.
- No canonical repo-wide type-check command is currently documented; do not
  invent one in Memory Bank notes or contributor guidance.
- Some historical docs mention partial demo gaps or compatibility issues; Memory
  Bank updates should note unresolved limitations rather than assuming every
  documented path is fully complete.
