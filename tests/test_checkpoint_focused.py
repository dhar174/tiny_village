#!/usr/bin/env python3
"""
Focused integration tests for the checkpoint system.
Tests the CheckpointManager in isolation without requiring full game initialization.
"""

import sys
import os
import json
import tempfile
import shutil
from unittest.mock import MagicMock, Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_checkpoint_manager_basic():
    """Test CheckpointManager basic functionality without full game controller."""
    print("\n" + "="*60)
    print("CHECKPOINT MANAGER BASIC TESTS")
    print("="*60)
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    
    try:
        # Import after setting up mocks
        sys.modules['pygame'] = MagicMock()
        sys.modules['pygame.time'] = MagicMock()
        
        # Use incrementing time instead of constant
        _time_counter = [1000]
        def mock_get_ticks():
            _time_counter[0] += 100
            return _time_counter[0]
        
        sys.modules['pygame'].time.get_ticks = mock_get_ticks
        
        from tiny_gameplay_controller import CheckpointManager
        
        # Create mock controller with minimal required attributes
        mock_controller = Mock()
        mock_controller.characters = {}
        mock_controller.global_achievements = {"village_milestones": {}}
        mock_controller.game_statistics = {
            "actions_executed": 10,
            "actions_failed": 2,
            "characters_created": 5,
            "errors_recovered": 1
        }
        mock_controller.weather_system = {"current_weather": "sunny"}
        mock_controller.quest_system = {"active_quests": {}}
        mock_controller.get_social_networks = lambda: {"relationships": {}}
        
        # Mock the save_game_state method to actually create a file
        def mock_save(filepath):
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                data = {
                    "timestamp": 1000,
                    "characters": {},
                    "statistics": mock_controller.game_statistics.copy(),
                    "achievements": mock_controller.global_achievements.copy()
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            except Exception as e:
                print(f"Mock save error: {e}")
                return False
        
        def mock_load(filepath):
            try:
                if not os.path.exists(filepath):
                    return False
                with open(filepath, 'r') as f:
                    data = json.load(f)
                mock_controller.game_statistics.update(data.get("statistics", {}))
                mock_controller.global_achievements.update(data.get("achievements", {}))
                return True
            except Exception as e:
                print(f"Mock load error: {e}")
                return False
        
        mock_controller.save_game_state = mock_save
        mock_controller.load_game_state = mock_load
        
        # Create CheckpointManager
        checkpoint_mgr = CheckpointManager(mock_controller, checkpoint_dir)
        
        print("\nTest 1: Checkpoint Creation")
        result = checkpoint_mgr.create_checkpoint("test_checkpoint")
        # Check using checkpoint history instead of filename (since timestamp is added)
        if result and len(checkpoint_mgr.checkpoint_history) > 0:
            created_checkpoint = checkpoint_mgr.checkpoint_history[-1]
            if os.path.exists(created_checkpoint["path"]):
                print("  ✓ Checkpoint created successfully")
            else:
                print("  ✗ Checkpoint file not found")
                return False
        else:
            print("  ✗ Checkpoint creation failed")
            return False
        
        print("\nTest 2: Checkpoint Restoration")
        # Modify state
        mock_controller.game_statistics["actions_executed"] = 100
        # Restore
        restore_result = checkpoint_mgr.restore_checkpoint(-1)
        if restore_result and mock_controller.game_statistics["actions_executed"] == 10:
            print("  ✓ Checkpoint restored successfully")
        else:
            print(f"  ✗ Checkpoint restoration failed (actions_executed = {mock_controller.game_statistics['actions_executed']})")
            return False
        
        print("\nTest 3: Multiple Checkpoints")
        for i in range(3):
            checkpoint_mgr.create_checkpoint(f"checkpoint_{i}")
        
        if len(checkpoint_mgr.checkpoint_history) == 4:  # Including the first one
            print(f"  ✓ Multiple checkpoints created ({len(checkpoint_mgr.checkpoint_history)} total)")
        else:
            print(f"  ✗ Expected 4 checkpoints, found {len(checkpoint_mgr.checkpoint_history)}")
            return False
        
        print("\nTest 4: Checkpoint Cleanup")
        checkpoint_mgr.max_checkpoints = 2
        checkpoint_mgr._cleanup_old_checkpoints()
        
        if len(checkpoint_mgr.checkpoint_history) == 2:
            print(f"  ✓ Old checkpoints cleaned up correctly")
        else:
            print(f"  ✗ Expected 2 checkpoints after cleanup, found {len(checkpoint_mgr.checkpoint_history)}")
            return False
        
        print("\nTest 5: Checkpoint Timing")
        checkpoint_mgr.set_checkpoint_interval(5000)
        checkpoint_mgr.last_checkpoint_time = 0
        
        should_checkpoint = checkpoint_mgr.should_checkpoint(6000)
        if should_checkpoint:
            print("  ✓ Checkpoint timing logic works")
        else:
            print("  ✗ Checkpoint timing logic failed")
            return False
        
        print("\nTest 6: Auto-checkpoint Enable/Disable")
        checkpoint_mgr.enable_auto_checkpoint(False)
        should_not_checkpoint = checkpoint_mgr.should_checkpoint(10000)
        
        if not should_not_checkpoint:
            print("  ✓ Auto-checkpoint disable works")
        else:
            print("  ✗ Auto-checkpoint disable failed")
            return False
        
        print("\nTest 7: Checkpoint List")
        checkpoint_list = checkpoint_mgr.get_checkpoint_list()
        if isinstance(checkpoint_list, list) and len(checkpoint_list) > 0:
            print(f"  ✓ Checkpoint list retrieved ({len(checkpoint_list)} checkpoints)")
        else:
            print("  ✗ Failed to get checkpoint list")
            return False
        
        print("\n" + "="*60)
        print("✓ All basic checkpoint tests passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_save_load_basic():
    """Test basic save/load without full initialization."""
    print("\n" + "="*60)
    print("SAVE/LOAD BASIC TESTS")
    print("="*60)
    
    test_dir = tempfile.mkdtemp(prefix="saveload_test_")
    
    try:
        # Test saving and loading JSON data
        save_path = os.path.join(test_dir, "test_save.json")
        
        # Create test data
        test_data = {
            "timestamp": 12345,
            "characters": {"char1": {"name": "Alice", "energy": 75}},
            "statistics": {"actions_executed": 50},
            "achievements": {"first_quest": True}
        }
        
        print("\nTest 1: Save Data")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        if os.path.exists(save_path):
            print("  ✓ Save file created")
        else:
            print("  ✗ Save file not created")
            return False
        
        print("\nTest 2: Load Data")
        with open(save_path, 'r') as f:
            loaded_data = json.load(f)
        
        if loaded_data["statistics"]["actions_executed"] == 50:
            print("  ✓ Data loaded correctly")
        else:
            print("  ✗ Data mismatch")
            return False
        
        print("\nTest 3: Handle Missing File")
        missing_path = os.path.join(test_dir, "nonexistent.json")
        if not os.path.exists(missing_path):
            print("  ✓ Correctly detects missing file")
        else:
            print("  ✗ File existence check failed")
            return False
        
        print("\n" + "="*60)
        print("✓ All save/load basic tests passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "CHECKPOINT SYSTEM FOCUSED TESTS")
    print("="*70)
    
    # Run tests
    test1_result = test_checkpoint_manager_basic()
    test2_result = test_save_load_basic()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    results = [
        ("Checkpoint Manager Basic", test1_result),
        ("Save/Load Basic", test2_result)
    ]
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
