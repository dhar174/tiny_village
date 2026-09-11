# Tiny Village Documentation

This directory contains organized documentation for the Tiny Village project.

## Directory Structure

### 📚 `/guides/` - User Guides
User-facing documentation for getting started and using the system.

- **QUICKSTART.md** - Quick start guide for running demos
- **QUICKSTART_CHECKPOINTS.md** - Guide to the checkpoint system
- **LLM_INTEGRATION_USAGE_GUIDE.md** - How to use LLM integration features

### 📖 `/reference/` - Technical Reference
Current technical documentation and API references for the system's components.

- **AGENTS.md** - Supplemental developer-agent architecture background (see root `/AGENTS.md` for the canonical guide)
- **SYSTEM_INTEGRATION_COMPLETE.md** - Complete system integration documentation
- **MINIMUM_DEMO_STATUS.md** - Current implementation status and demo capabilities
- **CHECKPOINT_SYSTEM_DOCUMENTATION.md** - Checkpoint and save system documentation
- **STORYTELLING_DOCUMENTATION.md** - Storytelling system documentation
- **STORY_ARC_DOCUMENTATION.md** - Story arc system documentation
- **EFFECT_SCHEMA_V2_DOCUMENTATION.md** - Effect schema version 2 documentation
- **ENHANCED_MOCK_ACTION_DOCUMENTATION.md** - Mock action system documentation
- **SIGMOID_FIX_DOCUMENTATION.md** - Sigmoid function fixes
- **LLM_INTEGRATION_SUMMARY.md** - LLM integration technical summary
- **MAP_INTERACTIVITY.md** - Map interaction system documentation
- **MINIMAP_FEATURES.md** - Minimap features and usage
- **GUI_ANALYSIS.md** - GUI system analysis
- **graph_manager_high_level_summary.md** - Graph manager overview
- **graphmanager_descriptive_overview.md** - Detailed graph manager description

### 🧪 `/testing/` - Testing Documentation
Guidelines and best practices for testing.

- **MEMORY_TESTING_BEST_PRACTICES.md** - Best practices for memory testing
- **MEMORY_TESTING_GUIDELINES.md** - Guidelines for memory system tests
- **RANDOMNESS_TESTING_GUIDE.md** - How to handle randomness in tests
- **TEST_FILES_README.md** - Overview of test file organization

### 🗄️ `/archived/` - Historical Documentation
Historical fix summaries, implementation reports, and outdated documentation. These are kept for reference but describe past issues or implementations that have been superseded.

#### Fix Summaries
- ALIGNMENT_FIX_SUMMARY.md
- FIX_SUMMARY_ISSUE_445.md
- ISSUE_329_FIX_SUMMARY.md
- ISSUE_332_FIX_SUMMARY.md
- ISSUE_334_FIX_SUMMARY.md
- ISSUE_471_FIX_SUMMARY.md
- TESTING_ANTIPATTERN_FIX_SUMMARY.md

#### Implementation Summaries (Historical)
- DYNAMIC_ACTION_CHOICES_IMPLEMENTATION.md
- EFFECT_SCHEMA_V2_IMPLEMENTATION_SUMMARY.md
- ERROR_HANDLING_IMPLEMENTATION_SUMMARY.md
- GOAP_IMPLEMENTATION_SUMMARY.md
- IMPLEMENTATION_SUMMARY.md
- INTEGRATION_SUMMARY.md
- GRAPHMANAGER_REFACTORING_SUMMARY.md
- SYSTEM_INTEGRATION_SUMMARY.md

#### Outdated Reports
- MISSING_FUNCTIONALITY_OVERVIEW.md (outdated - main.py now exists)
- missing_demo_elements.md (outdated - issues resolved)
- missing_demo_requirements.md (outdated - issues resolved)

#### Testing Antipattern Fixes (Historical)
- MEMORY_TESTING_ANTIPATTERN_FIXES.md
- MOCKSTOCK_ANTIPATTERN_FIX.md
- MockCharacter_Solution_Summary.md

## Other Documentation Locations

### `/design_docs/` - Architecture and Design
Deep-dive documentation on system architecture and design decisions.

- action_system_deep_dive.md
- data_flow_decision_cycle.md
- deconstructive_analysis_summary.md
- documentation_summary.md
- graph_manager_deep_dive.md
- gui_display_analysis.md
- high_level_architecture.md
- memory_manager_deep_dive.md
- module_connectivity_map.md
- strategy_management_architecture.md

### `/critical_analysis/` - Critical Analysis
Analysis reports for various system components.

- CALCULATE_GOAL_DIFFICULTY_IMPROVEMENTS.md
- ENHANCED_EVENT_HANDLER_COMPLETION.md
- IMPLEMENTATION_COMPLETION_SUMMARY.md
- IMPLEMENTATION_PLAN.md
- TODO_report.md
- UTILITY_FUNCTIONS_COMPLETION_SUMMARY.md
- ai_systems_analysis.md
- controller_analysis.md
- github_issue_templates.md
- graph_manager_analysis.md
- graph_manager_code_analysis.md
- gui_display_analysis.md
- stub_systems_analysis.md
- world_interaction_analysis.md

### `/.github/` - GitHub Configuration
GitHub-specific documentation and agent definitions.

- `.github/copilot-instructions.md` - Copilot agent instructions
- `.github/IMPLEMENTATION_ISSUES_INDEX.md` - Index of implementation issues
- `.github/agents/*.agent.md` - Copilot agent definitions
- `.github/instructions/*.instructions.md` - Repository-specific Copilot instructions and workflow rules
- `.github/issue_templates/*.md` - Issue templates
- `.github/skills/**/SKILL.md` - Reusable agent skills and bundled references

## Quick Links

### Getting Started
- **First time users**: Start with [/guides/QUICKSTART.md](guides/QUICKSTART.md)
- **Main README**: See [/README.md](../README.md) in the root directory

### For Developers
- **System architecture**: See [/design_docs/high_level_architecture.md](../design_docs/high_level_architecture.md)
- **Module connectivity**: See [/design_docs/module_connectivity_map.md](../design_docs/module_connectivity_map.md)
- **Agent guidelines**: Start with [`../AGENTS.md`](../AGENTS.md), then use [/reference/AGENTS.md](reference/AGENTS.md) for supplemental background
- **Current status**: See [/reference/MINIMUM_DEMO_STATUS.md](reference/MINIMUM_DEMO_STATUS.md)

### For Testers
- **Testing guidelines**: See [/testing/](testing/)
- **Memory testing**: See [/testing/MEMORY_TESTING_BEST_PRACTICES.md](testing/MEMORY_TESTING_BEST_PRACTICES.md)

## Documentation Status

### Current (Up-to-date)
All documents in `/guides/`, `/reference/`, and `/testing/` reflect the current state of the codebase.

### Historical (Archived)
Documents in `/archived/` are kept for historical reference but describe past implementations or issues that have been resolved. They are not updated to reflect current code.

### Analysis (Point-in-time)
Documents in `/design_docs/` and `/critical_analysis/` represent analysis done at specific points in time and may not reflect current implementation details, but provide valuable architectural insights.

## Maintenance Notes

This documentation was reorganized on 2025-12-26 to:
- Separate current documentation from historical records
- Group related documents by purpose (guides, reference, testing)
- Identify outdated content (e.g., documents mentioning missing main.py)
- Improve discoverability and navigation

When adding new documentation:
- **User guides** → `/guides/`
- **Technical reference** → `/reference/`
- **Testing documentation** → `/testing/`
- **Historical/completed work** → `/archived/`
- **Deep architecture** → `/design_docs/`
- **Analysis reports** → `/critical_analysis/`
