#!/usr/bin/env python3
"""
Standalone test for CheckpointManager without full game dependencies.
Tests the actual CheckpointManager implementation with mocked dependencies.
"""

import sys
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all the problematic modules before any imports
sys.modules['pygame'] = MagicMock()
sys.modules['pygame.time'] = MagicMock()
sys.modules['pygame.font'] = MagicMock()
sys.modules['pygame.display'] = MagicMock()
sys.modules['pygame.math'] = MagicMock()

# Mock time to be controllable
_mock_time = [0]
def mock_get_ticks():
    """Return and increment mock time."""
    _mock_time[0] += 100
    return _mock_time[0]

sys.modules['pygame'].time.get_ticks = mock_get_ticks

# Mock the problematic time manager
mock_calendar = MagicMock()
mock_calendar.get_game_time_string = lambda: "Day 1, 00:00"
sys.modules['tiny_time_manager'] = MagicMock()
sys.modules['tiny_time_manager'].GameCalendar = lambda *args, **kwargs: mock_calendar

# Mock tiny_globals to avoid initialization issues
sys.modules['tiny_globals'] = MagicMock()
sys.modules['tiny_globals'].global_calendar = mock_calendar

# Now we can safely import
from tiny_gameplay_controller import CheckpointManager


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
        
        # Mock save/load methods with success/failure control
        save_should_fail = [False]  # Mutable to allow test control
        
        def mock_save(filepath):
            if save_should_fail[0]:
                return False
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                data = {
                    "timestamp": _mock_time[0],
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
        mock_controller.add_event_notification = Mock()  # Mock notification method
        
        # Reset mock time
        _mock_time[0] = 0
        
        # Create CheckpointManager with actual implementation
        checkpoint_mgr = CheckpointManager(mock_controller, checkpoint_dir)
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Create checkpoint
        print("\n[Test 1] Checkpoint Creation")
        tests_total += 1
        _mock_time[0] = 1000
        result = checkpoint_mgr.create_checkpoint("test_checkpoint")
        checkpoint_file = os.path.join(checkpoint_dir, "test_checkpoint_1100.json")  # With timestamp
        if result and os.path.exists(checkpoint_file):
            print("  ✓ PASS: Checkpoint created successfully")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Checkpoint creation failed (file exists: {os.path.exists(checkpoint_file)})")
        
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
        
        # Test 3: Multiple checkpoints with unique names
        print("\n[Test 3] Multiple Checkpoint Creation with Unique Names")
        tests_total += 1
        for i in range(3):
            _mock_time[0] += 1000
            checkpoint_mgr.create_checkpoint(f"checkpoint_{i}")
        
        checkpoint_count = len(checkpoint_mgr.checkpoint_history)
        if checkpoint_count == 4:  # Including first test checkpoint
            print(f"  ✓ PASS: {checkpoint_count} checkpoints created with unique names")
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
        checkpoint_mgr.last_checkpoint_time = 1000
        _mock_time[0] = 7000
        should_cp = checkpoint_mgr.should_checkpoint(7000)
        
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
        
        # Test 8: Consecutive failure tracking
        print("\n[Test 8] Consecutive Failure Tracking")
        tests_total += 1
        checkpoint_mgr.enable_auto_checkpoint(True)
        checkpoint_mgr.consecutive_failures = 0
        save_should_fail[0] = True  # Make saves fail
        
        # Try to create checkpoints that will fail
        for i in range(4):
            checkpoint_mgr.create_checkpoint(f"failing_checkpoint_{i}")
        
        if checkpoint_mgr.consecutive_failures >= 3:
            print(f"  ✓ PASS: Consecutive failures tracked ({checkpoint_mgr.consecutive_failures})")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Expected 3+ failures, got {checkpoint_mgr.consecutive_failures}")
        
        # Test 9: History validation
        print("\n[Test 9] Checkpoint History Validation")
        tests_total += 1
        save_should_fail[0] = False  # Re-enable saves
        
        # Create a checkpoint then delete its file manually
        _mock_time[0] += 1000
        checkpoint_mgr.create_checkpoint("to_be_deleted")
        deleted_checkpoint = checkpoint_mgr.checkpoint_history[-1]
        os.remove(deleted_checkpoint["path"])
        
        # Validate history
        checkpoint_mgr._validate_checkpoint_history()
        
        # Check if the deleted checkpoint was removed from history
        if deleted_checkpoint not in checkpoint_mgr.checkpoint_history:
            print(f"  ✓ PASS: Invalid checkpoint removed from history")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Invalid checkpoint still in history")
        
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
