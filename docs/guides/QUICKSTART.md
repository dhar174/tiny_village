# Tiny Village Quickstart Guide

This guide describes installation and the current Tiny Village demo entry points.

## Prerequisites

- Python 3.12
- pip package manager

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dhar174/tiny_village.git
cd tiny_village
```

### 2. Install Minimum Dependencies
```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
python -m pip check
```

The default install is the locked core profile for Python 3.12/Linux x86-64.
Add [visual, development or optional feature profiles](DEPENDENCY_PROFILES.md)
as needed. NLP/LLM/GPU capabilities remain available through separate profiles;
their model artifacts and hardware need independent validation.

**Current runtime boundary:** the legacy minimal demo uses mocked components
and may exit 0 with zero actions. It is not the accepted MVP economic loop.
The visual application and broader integration tests have separate unresolved
runtime/fixture work. See [the dependency validation report](../testing/issue-725-dependency-profile-validation.md)
before interpreting a startup or test result as full gameplay readiness.

## Running Your First Demo

### Option 1: Minimal Console Demo (Fastest)
No display required, perfect for testing:
```bash
python main.py --mode minimal
```

**Expected output:**
- System initialization
- Character creation
- Event handling demonstration
- Action execution examples
- Performance analytics


### Option 2: Integration Tests
Run the existing integration tests (known fixture/runtime failures are separate):
```bash
python main.py --mode test
```

**Expected output:**
- 11 integration tests
- 9+ tests passing
- System validation report


### Option 3: Full Visual Demo
Pygame window with characters moving on a map:
```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements/locks/visual-py312-linux-x86_64.txt
python main.py --mode visual
```

**Expected output:**
- Game window opens
- Characters appear on map
- UI panels show stats
- Characters make decisions

**Controls:**
- `SPACE` - Pause/unpause
- `ESC` - Quit
- `S` - Save game
- `L` - Load game
- Click characters to select


## Common Issues & Solutions

### "No module named 'pygame'"
**Solution**: Install the visual profile in your activated virtual environment
```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements/locks/visual-py312-linux-x86_64.txt
```

### "Failed to initialize MapController"
**Solution**: Use minimal mode which doesn't require display
```bash
python main.py --mode minimal
```

### "Character class not available"
**Solution**: This is a warning, not an error. The demo uses simplified characters.

### "LLM components not available"
**Solution**: This is expected. LLM is optional. Use `--no-llm` flag to suppress warnings:
```bash
python main.py --no-llm
```

## Customizing Your Demo

### Create Fewer Characters (Faster)
```bash
python main.py --characters 3
```

### Run at Lower FPS (Slower machines)
```bash
python main.py --fps 30
```

### Enable Debug Logging
```bash
python main.py --verbose
```

### Combine Options
```bash
python main.py --mode visual --characters 3 --no-llm --fps 30
```

## What's Working

✅ **Core Systems**
- Game loop with error recovery
- Event detection and handling
- Character decision making
- Action execution with fallbacks
- Performance monitoring
- Auto-save functionality

✅ **Integration**
- Event → Strategy → Action pipeline
- Character turn processing
- Error handling and recovery
- Analytics and metrics

✅ **Demo Modes**
- Minimal console demo
- Integration tests
- Full visual demo (with minor UI issues)

## What's Optional

⚠️ **Advanced Features** (Not required for demo)
- LLM decision making (transformers)
- Advanced memory system (spacy)
- Complex social networks
- Full NLP features

## Next Steps

### For Developers
1. Read `MINIMUM_DEMO_STATUS.md` for implementation details
2. Run integration tests to understand system behavior
3. Check test results in console output
4. Review code in `tiny_gameplay_controller.py`

### For Users
1. Try minimal demo first
2. If it works, try visual demo
3. Experiment with different character counts
4. Watch the game statistics

### For Contributors
1. Run tests: `python main.py --mode test`
2. Check existing issues on GitHub
3. Read contributing guide (if available)
4. Start with small improvements

## Performance Expectations

**Initialization**: ~500ms  
**Turn Processing**: ~10ms per character  
**Event Processing**: ~5ms per event  
**Memory Usage**: ~150MB baseline  
**FPS**: 60 (configurable)

## Troubleshooting

### Demo doesn't start
1. Check Python version: `python --version` (need 3.12)
2. Check dependencies: `pip list | grep -E "pygame|networkx|numpy"`
3. Run minimal demo: `python main.py --mode minimal`
4. Check error messages carefully

### Characters don't move
1. This is normal in minimal mode (no display)
2. Try visual mode: `python main.py --mode visual`
3. Check if pygame window opened
4. Try lower FPS: `python main.py --fps 30`

### High memory usage
1. Reduce character count: `python main.py --characters 3`
2. Lower FPS: `python main.py --fps 30`
3. Check for memory leaks (report to developers)

## Getting Help

1. **Check logs**: Enable verbose mode with `--verbose`
2. **Run tests**: `python main.py --mode test`
3. **GitHub Issues**: Report problems with:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

## Success Indicators

✅ You should see:
- "Controller initialized" message
- Character count in logs
- "All game systems initialized successfully"
- No critical errors (warnings are OK)

## Files Created

The demo creates these directories/files:
- `saves/` - Game save files
- `saves/checkpoints/` - Auto-save checkpoints
- A user-owned Hugging Face model cache (see below)

Save/checkpoint directories contain game state. Model cache files persist across startups.

### Model cache configuration

The default Hugging Face home is `$XDG_CACHE_HOME/huggingface`, or
`~/.cache/huggingface` when XDG_CACHE_HOME is unset. Model repositories use its
`hub/` subdirectory. To select a different owned location before startup:

```bash
export HF_HOME=/absolute/path/to/your/huggingface
python main.py --mode minimal --no-llm --headless
```

`HF_HUB_CACHE` can select a separate repository cache. An existing
`TRANSFORMERS_CACHE` setting takes precedence for TinyVillage's Transformers
loaders; the application preserves it instead of replacing it or introducing a
new legacy setting. Cache setup runs before optional model libraries are imported
and does not import or download models itself.

On POSIX systems, selected directories must belong to the effective runtime user
and are restricted to mode `0700`. Ancestors must belong to that user or root;
an unrelated owner can replace children even when the ancestor's mode is `0755`.
Existing files and unselected parent permissions remain intact. Symlink/reparse
directory paths, paths containing `..`, and parents that allow other users to
replace entries are rejected with a path-specific error; use the direct path
beneath trusted ancestors. Windows prefixes are checked before directory creation
and use inherited directory ACLs; POSIX mode checks do not establish Windows ACL
isolation.

An absolute HF_HOME works without resolving a user-profile directory. The default
home directory is needed only when neither HF_HOME nor XDG_CACHE_HOME supplies
the location. Semantic-role module imports also preserve these settings.

The previous shared `hf_test_cache` and machine-specific drive cache are not
deleted or automatically migrated. TLS and downloader verification settings are
preserved. This cache configuration does not change optional model availability.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Full visual demo |
| `python main.py --mode minimal` | Console demo |
| `python main.py --mode test` | Run tests |
| `python main.py --help` | Show all options |

## Still Have Issues?

Create a GitHub issue with:
1. Command you ran
2. Error message (full output)
3. Python version (`python --version`)
4. OS (Windows/Mac/Linux)

---

**Happy simulating!** 🏘️
