# Game State Persistence and Checkpointing System

## Overview

The game state persistence and checkpointing system provides automatic and manual saving of game state to enable recovery from crashes, gameplay experimentation, and long-term save management.

## Features

### 1. Automatic Checkpointing
- **Configurable intervals**: Default 5 minutes, minimum 10 seconds
- **Automatic cleanup**: Maintains a history of the last 10 checkpoints by default
- **Non-intrusive**: Runs in background during game loop
- **User notification**: Optional UI notification when checkpoint is created

### 2. Manual Save/Load
- **Quick save**: Press 'S' to save current game state
- **Quick load**: Press 'L' to load saved game state
- **Manual checkpoint**: Press 'C' to create a named checkpoint
- **Restore checkpoint**: Press 'V' to restore the most recent checkpoint

### 3. Checkpoint Management
- **History tracking**: Maintains list of all checkpoints with metadata
- **Selective restoration**: Can restore any checkpoint from history
- **Corruption recovery**: Automatically tries previous checkpoints if latest is corrupted
- **File management**: Automatically removes old checkpoints beyond limit

## Configuration

Add to your game configuration:

```python
config = {
    "checkpoint": {
        "directory": "saves/checkpoints",  # Where to store checkpoints
        "interval_ms": 300000,              # 5 minutes (300,000 milliseconds)
        "auto_enabled": True,               # Enable automatic checkpointing
        "max_checkpoints": 10               # Keep last 10 checkpoints
    }
}
```

## Key Bindings

| Key | Action |
|-----|--------|
| S | Save game (quick save) |
| L | Load game (quick load) |
| C | Create manual checkpoint |
| V | Restore last checkpoint |

## Architecture

### CheckpointManager Class

The `CheckpointManager` class handles all checkpoint operations:

- **Initialization**: Sets up checkpoint directory and configuration
- **Automatic checkpointing**: Monitors game time and creates checkpoints at intervals
- **Manual operations**: Provides API for manual checkpoint creation/restoration
- **History management**: Tracks checkpoint metadata and manages file cleanup
- **Error recovery**: Handles corrupted files and provides fallback options
- **Concurrency safety**: Uses an internal lock so create/restore operations do
  not overlap
- **Failure escalation**: Tracks consecutive failures and can emit a high-priority
  notification after repeated save problems

### Internal Robustness Helpers

The public API below is the main surface area for gameplay code, but the
implementation also relies on internal helpers to keep the checkpoint history
consistent and to recover cleanly from bad states:

- `_check_failure_threshold()` - raises user-visible warnings after repeated
  checkpoint failures
- `_validate_checkpoint_history()` - removes history entries whose files no
  longer exist
- `_cleanup_old_checkpoints()` - deletes older checkpoint files beyond the
  configured retention limit

### Integration with GameplayController

The checkpoint manager is integrated into the main game loop:

```python
def game_loop(self):
    while self.running:
        # ... game update logic ...
        
        # Automatic checkpointing
        current_time = pygame.time.get_ticks()
        if self.checkpoint_manager.should_checkpoint(current_time):
            self.checkpoint_manager.create_checkpoint()
```

## Saved Game State

Checkpoints save the following game state:

- **Timestamp**: When the checkpoint was created
- **Characters**: All character data (positions, stats, inventory)
- **Statistics**: Game statistics (actions executed, errors, etc.)
- **Achievements**: Unlocked achievements
- **Weather**: Current weather state
- **Quest System**: Active and completed quests
- **Social Networks**: Character relationships (via GraphManager)

## API Reference

### CheckpointManager Methods

#### `create_checkpoint(checkpoint_name: str = None) -> bool`
Creates a new checkpoint with optional custom name.

**Parameters:**
- `checkpoint_name`: Optional name for checkpoint (auto-generated if None)

**Returns:**
- `True` if checkpoint created successfully, `False` otherwise

**Example:**
```python
checkpoint_manager.create_checkpoint("before_boss_fight")
```

#### `restore_checkpoint(checkpoint_index: int = -1) -> bool`
Restores game state from a checkpoint.

**Parameters:**
- `checkpoint_index`: Index in history (-1 for most recent, -2 for second most recent, etc.)

**Returns:**
- `True` if restoration successful, `False` otherwise

**Example:**
```python
# Restore most recent checkpoint
checkpoint_manager.restore_checkpoint(-1)

# Restore second most recent
checkpoint_manager.restore_checkpoint(-2)
```

#### `should_checkpoint(current_time: int) -> bool`
Checks if it's time for an automatic checkpoint.

**Parameters:**
- `current_time`: Current game time in milliseconds

**Returns:**
- `True` if checkpoint should be created

#### `get_checkpoint_list() -> list`
Returns list of available checkpoints with metadata.

**Returns:**
- List of dictionaries containing checkpoint information

**Example:**
```python
checkpoints = checkpoint_manager.get_checkpoint_list()
for cp in checkpoints:
    print(f"Checkpoint {cp['index']}: {cp['filename']} - {cp['character_count']} characters")
```

#### `set_checkpoint_interval(interval_ms: int)`
Sets the automatic checkpoint interval.

**Parameters:**
- `interval_ms`: Interval in milliseconds (minimum 10000)

**Example:**
```python
# Set to 10 minutes
checkpoint_manager.set_checkpoint_interval(600000)
```

#### `enable_auto_checkpoint(enabled: bool)`
Enables or disables automatic checkpointing.

**Parameters:**
- `enabled`: True to enable, False to disable

**Example:**
```python
# Disable auto-checkpointing during cutscenes
checkpoint_manager.enable_auto_checkpoint(False)
```

#### `recover_from_corruption() -> bool`
Attempts to recover from corrupted save file.

**Returns:**
- `True` if recovery successful

**Example:**
```python
if not checkpoint_manager.restore_checkpoint(-1):
    # Try corruption recovery
    if checkpoint_manager.recover_from_corruption():
        print("Recovered from corrupted save")
```

## Testing

### Running Tests

```bash
# Run standalone checkpoint tests (no game dependencies)
python tests/test_checkpoint_standalone.py

# Run focused tests (basic functionality)
python tests/test_checkpoint_focused.py

# Run full integration tests (requires full game setup)
python tests/test_checkpoint_system.py
```

These commands verify the checkpoint system itself. They do not guarantee that a
full Tiny Village demo session is currently runnable in the local environment.

### Test Coverage

- ✓ Checkpoint creation
- ✓ Checkpoint restoration
- ✓ Multiple checkpoint handling
- ✓ Old checkpoint cleanup
- ✓ Automatic checkpoint timing (with configuration)
- ✓ Auto-checkpoint enable/disable
- ✓ Checkpoint list retrieval
- ✓ Basic save/load functionality
- ✓ Corruption recovery

## Implementation Notes

### Performance Considerations

1. **Non-blocking**: Checkpoint creation happens between game frames
2. **Incremental**: Only modified data is saved
3. **Efficient cleanup**: Old checkpoints are removed automatically
4. **Minimal overhead**: Timing checks are lightweight

### Error Handling

The system includes comprehensive error handling:

- **Save failures**: Logged but don't crash the game
- **Load failures**: Fall back to previous checkpoint
- **Corrupted files**: Automatic recovery attempts
- **Missing directories**: Created automatically
- **Disk space**: No explicit handling (relies on OS)

### Future Enhancements

Potential improvements for future versions:

1. **Compression**: Compress checkpoint files to save disk space
2. **Cloud sync**: Sync checkpoints to cloud storage
3. **Metadata**: Store more gameplay context (difficulty, playtime, etc.)
4. **Thumbnail**: Save screenshot with each checkpoint
5. **Incremental saves**: Only save changed data since last checkpoint
6. **Backup rotation**: Keep daily/weekly checkpoints longer
7. **UI browser**: In-game checkpoint browser with preview

## Troubleshooting

### Checkpoints not created automatically

- Check that `auto_enabled` is `True` in configuration
- Verify checkpoint interval is reasonable (>= 10 seconds)
- Check log output for error messages
- Ensure checkpoint directory is writable

### Cannot restore checkpoint

- Verify checkpoint file exists in checkpoint directory
- Check file permissions
- Try `recover_from_corruption()` if file is corrupted
- Check logs for specific error messages

### Disk space issues

- Reduce `max_checkpoints` in configuration
- Manually delete old checkpoints from `saves/checkpoints/`
- Consider implementing compression (future enhancement)

## Examples

### Example 1: Custom Checkpoint Before Risky Action

```python
# Before trying something risky
if controller.checkpoint_manager.create_checkpoint("before_experiment"):
    print("Checkpoint created - safe to experiment")
    # ... do risky thing ...
else:
    print("Couldn't create checkpoint - skipping experiment")
```

### Example 2: Browse and Restore Specific Checkpoint

```python
# Get list of checkpoints
checkpoints = controller.checkpoint_manager.get_checkpoint_list()

# Show to user
for cp in checkpoints:
    print(f"{cp['index']}: {cp['filename']} ({cp['character_count']} characters)")

# User selects checkpoint 3
if controller.checkpoint_manager.restore_checkpoint(3):
    print("Restored selected checkpoint")
```

### Example 3: Temporary Disable Auto-Checkpoint

```python
# Disable during cinematic
controller.checkpoint_manager.enable_auto_checkpoint(False)

# Play cinematic
play_cinematic()

# Re-enable after
controller.checkpoint_manager.enable_auto_checkpoint(True)
```

## License

This implementation is part of the Tiny Village project and follows the same license terms.
