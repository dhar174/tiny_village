#!/usr/bin/env python3
"""
Standalone test for CheckpointManager without game dependencies.
Extracts CheckpointManager logic for isolated testing.
"""

import sys
import os
import json
import tempfile
import shutil
from unittest.mock import Mock


# Minimal CheckpointManager implementation for testing
class CheckpointManager:
    """Manages automatic game state checkpointing and restoration."""
    
    def __init__(self, gameplay_controller, checkpoint_dir: str = "saves/checkpoints"):
        self.gameplay_controller = gameplay_controller
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 300000  # 5 minutes in milliseconds by default
        self.last_checkpoint_time = 0
        self.max_checkpoints = 10  # Keep last 10 checkpoints
        self.checkpoint_history = []
        self.auto_checkpoint_enabled = True
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def should_checkpoint(self, current_time: int) -> bool:
        """Check if it's time for an automatic checkpoint."""
        if not self.auto_checkpoint_enabled:
            return False
        
        time_since_last = current_time - self.last_checkpoint_time
        return time_since_last >= self.checkpoint_interval
    
    def create_checkpoint(self, checkpoint_name: str = None) -> bool:
        """Create a checkpoint of the current game state."""
        try:
            current_time = 1000  # Mock time
            
            # Generate checkpoint filename
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{current_time}.json"
            elif not checkpoint_name.endswith('.json'):
                checkpoint_name = f"{checkpoint_name}.json"
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            
            # Create the checkpoint using the gameplay controller's save method
            if self.gameplay_controller.save_game_state(checkpoint_path):
                # Add to checkpoint history
                checkpoint_info = {
                    "filename": checkpoint_name,
                    "path": checkpoint_path,
                    "timestamp": current_time,
                    "game_ticks": current_time,
                    "character_count": len(self.gameplay_controller.characters)
                }
                self.checkpoint_history.append(checkpoint_info)
                
                # Update last checkpoint time
                self.last_checkpoint_time = current_time
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error creating checkpoint: {e}")
            return False
    
    def restore_checkpoint(self, checkpoint_index: int = -1) -> bool:
        """Restore game state from a checkpoint."""
        try:
            if not self.checkpoint_history:
                return False
            
            # Get checkpoint info
            if checkpoint_index < 0:
                checkpoint_index = len(self.checkpoint_history) + checkpoint_index
            
            if checkpoint_index < 0 or checkpoint_index >= len(self.checkpoint_history):
                return False
            
            checkpoint_info = self.checkpoint_history[checkpoint_index]
            checkpoint_path = checkpoint_info["path"]
            
            # Verify checkpoint file exists
            if not os.path.exists(checkpoint_path):
                return False
            
            # Restore the checkpoint
            return self.gameplay_controller.load_game_state(checkpoint_path)
                
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the maximum limit."""
        while len(self.checkpoint_history) > self.max_checkpoints:
            # Remove oldest checkpoint
            old_checkpoint = self.checkpoint_history.pop(0)
            
            # Delete the file if it exists
            if os.path.exists(old_checkpoint["path"]):
                try:
                    os.remove(old_checkpoint["path"])
                except Exception:
                    pass
    
    def get_checkpoint_list(self) -> list:
        """Get list of available checkpoints."""
        return [
            {
                "index": i,
                "filename": cp["filename"],
                "timestamp": cp["timestamp"],
                "character_count": cp["character_count"]
            }
            for i, cp in enumerate(self.checkpoint_history)
        ]
    
    def set_checkpoint_interval(self, interval_ms: int):
        """Set the automatic checkpoint interval in milliseconds."""
        if interval_ms < 10000:  # Minimum 10 seconds
            interval_ms = 10000
        self.checkpoint_interval = interval_ms
    
    def enable_auto_checkpoint(self, enabled: bool):
        """Enable or disable automatic checkpointing."""
        self.auto_checkpoint_enabled = enabled
    
    def recover_from_corruption(self) -> bool:
        """Attempt to recover from corrupted save by restoring the most recent valid checkpoint."""
        try:
            # Try checkpoints from most recent to oldest
            for i in range(len(self.checkpoint_history) - 1, -1, -1):
                checkpoint_info = self.checkpoint_history[i]
                
                # Verify checkpoint file is readable
                try:
                    with open(checkpoint_info["path"], 'r') as f:
                        json.load(f)
                    
                    # If we can read it, try to restore it
                    if self.restore_checkpoint(i):
                        return True
                        
                except Exception:
                    continue
            
            return False
            
        except Exception:
            return False


def run_standalone_tests():
    """Run standalone checkpoint manager tests."""
    print("\n" + "="*70)
    print(" "*15 + "CHECKPOINT MANAGER STANDALONE TESTS")
    print("="*70)
    
    test_dir = tempfile.mkdtemp(prefix="checkpoint_standalone_")
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    
    try:
        # Create mock controller
        mock_controller = Mock()
        mock_controller.characters = {}
        mock_controller.game_statistics = {
            "actions_executed": 10,
            "actions_failed": 2
        }
        
        # Mock save/load methods
        def mock_save(filepath):
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                data = {
                    "timestamp": 1000,
                    "characters": {},
                    "statistics": mock_controller.game_statistics.copy()
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            except Exception:
                return False
        
        def mock_load(filepath):
            try:
                if not os.path.exists(filepath):
                    return False
                with open(filepath, 'r') as f:
                    data = json.load(f)
                mock_controller.game_statistics.update(data.get("statistics", {}))
                return True
            except Exception:
                return False
        
        mock_controller.save_game_state = mock_save
        mock_controller.load_game_state = mock_load
        
        # Create CheckpointManager
        checkpoint_mgr = CheckpointManager(mock_controller, checkpoint_dir)
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Create checkpoint
        print("\n[Test 1] Checkpoint Creation")
        tests_total += 1
        result = checkpoint_mgr.create_checkpoint("test_checkpoint")
        checkpoint_file = os.path.join(checkpoint_dir, "test_checkpoint.json")
        if result and os.path.exists(checkpoint_file):
            print("  ✓ PASS: Checkpoint created successfully")
            tests_passed += 1
        else:
            print("  ✗ FAIL: Checkpoint creation failed")
        
        # Test 2: Restore checkpoint
        print("\n[Test 2] Checkpoint Restoration")
        tests_total += 1
        original_value = mock_controller.game_statistics["actions_executed"]
        mock_controller.game_statistics["actions_executed"] = 999
        restore_result = checkpoint_mgr.restore_checkpoint(-1)
        restored_value = mock_controller.game_statistics["actions_executed"]
        
        if restore_result and restored_value == original_value:
            print(f"  ✓ PASS: Checkpoint restored (value: {original_value})")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Restoration failed (expected {original_value}, got {restored_value})")
        
        # Test 3: Multiple checkpoints
        print("\n[Test 3] Multiple Checkpoint Creation")
        tests_total += 1
        for i in range(3):
            checkpoint_mgr.create_checkpoint(f"checkpoint_{i}")
        
        checkpoint_count = len(checkpoint_mgr.checkpoint_history)
        if checkpoint_count == 4:  # Including first test checkpoint
            print(f"  ✓ PASS: {checkpoint_count} checkpoints created")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Expected 4 checkpoints, got {checkpoint_count}")
        
        # Test 4: Checkpoint cleanup
        print("\n[Test 4] Old Checkpoint Cleanup")
        tests_total += 1
        checkpoint_mgr.max_checkpoints = 2
        checkpoint_mgr._cleanup_old_checkpoints()
        
        remaining_count = len(checkpoint_mgr.checkpoint_history)
        if remaining_count == 2:
            print(f"  ✓ PASS: Cleaned up to {remaining_count} checkpoints")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Expected 2 checkpoints, got {remaining_count}")
        
        # Test 5: Checkpoint timing
        print("\n[Test 5] Automatic Checkpoint Timing")
        tests_total += 1
        checkpoint_mgr.set_checkpoint_interval(5000)
        checkpoint_mgr.last_checkpoint_time = 0
        should_cp = checkpoint_mgr.should_checkpoint(6000)
        
        if should_cp:
            print("  ✓ PASS: Timing logic correct")
            tests_passed += 1
        else:
            print("  ✗ FAIL: Timing logic incorrect")
        
        # Test 6: Auto-checkpoint disable
        print("\n[Test 6] Auto-Checkpoint Disable")
        tests_total += 1
        checkpoint_mgr.enable_auto_checkpoint(False)
        should_not_cp = checkpoint_mgr.should_checkpoint(10000)
        
        if not should_not_cp:
            print("  ✓ PASS: Auto-checkpoint disabled correctly")
            tests_passed += 1
        else:
            print("  ✗ FAIL: Auto-checkpoint disable failed")
        
        # Test 7: Checkpoint list
        print("\n[Test 7] Checkpoint List Retrieval")
        tests_total += 1
        cp_list = checkpoint_mgr.get_checkpoint_list()
        
        if isinstance(cp_list, list) and len(cp_list) == 2:
            print(f"  ✓ PASS: Retrieved {len(cp_list)} checkpoints")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Expected list of 2, got {len(cp_list) if isinstance(cp_list, list) else 'not a list'}")
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {tests_passed}/{tests_total}")
        
        if tests_passed == tests_total:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n❌ {tests_total - tests_passed} test(s) failed")
            return False
        
    except Exception as e:
        print(f"\n✗ Tests failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = run_standalone_tests()
    sys.exit(0 if success else 1)
