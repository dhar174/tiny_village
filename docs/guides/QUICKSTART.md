# Tiny Village Quickstart Guide

This guide will get you running a Tiny Village demo in under 5 minutes.

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
python3.12 -m pip install -r requirements.txt
```

**Note**: Tiny Village now targets Python 3.12. The full `requirements.txt` is the
supported dependency set; optional dependencies (transformers, spacy) add
advanced features but aren't required for the minimal demo paths.

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

**Time**: ~5 seconds

### Option 2: Integration Tests
Validate all systems are working:
```bash
python main.py --mode test
```

**Expected output:**
- 11 integration tests
- 9+ tests passing
- System validation report

**Time**: ~5 seconds

### Option 3: Full Visual Demo
Pygame window with characters moving on a map:
```bash
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

**Time**: Runs continuously until quit

## Common Issues & Solutions

### "No module named 'pygame'"
**Solution**: Install dependencies
```bash
python3.12 -m pip install -r requirements.txt
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
- `/tmp/hf_test_cache/` - Model cache (if using LLM)

These are safe to delete if you want to start fresh.

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
