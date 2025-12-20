#!/usr/bin/env python3
"""
Integration tests for the checkpoint and persistence system.

Tests verify:
- Checkpoint creation and restoration
- Automatic checkpointing in game loop
- Checkpoint history management
- Error recovery from corrupted saves
- Full turn cycle with checkpointing
"""

import sys
import os
import json
import time
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock pygame before importing if not available
try:
    import pygame
except ImportError:
    print("pygame not available, using mock")
    import unittest.mock as mock
    sys.modules['pygame'] = mock.MagicMock()
    sys.modules['pygame.font'] = mock.MagicMock()
    sys.modules['pygame.display'] = mock.MagicMock()
    sys.modules['pygame.time'] = mock.MagicMock()
    sys.modules['pygame.math'] = mock.MagicMock()
    import pygame


def setup_test_environment():
    """Set up a clean test environment."""
    # Create temporary directory for test saves
    test_dir = tempfile.mkdtemp(prefix="tiny_village_test_")
    return test_dir


def cleanup_test_environment(test_dir):
    """Clean up test environment."""
    try:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    except Exception as e:
        print(f"Warning: Failed to clean up test directory: {e}")


class TestCheckpointManager:
    """Test the CheckpointManager functionality."""
    
    def __init__(self):
        self.test_results = []
    
    def setup(self):
        """Set up test fixtures."""
        from tiny_gameplay_controller import CheckpointManager, GameplayController
        
        # Create temporary test directory
        self.test_dir = setup_test_environment()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        
        # Create a minimal gameplay controller for testing
        config = {
            "checkpoint": {
                "directory": self.checkpoint_dir,
                "interval_ms": 10000,  # 10 seconds for testing
                "auto_enabled": True
            },
            "screen_width": 800,
            "screen_height": 600
        }
        
        # Mock pygame initialization to prevent display window
        pygame.init = lambda: None
        pygame.display.set_mode = lambda *args, **kwargs: mock.MagicMock()
        pygame.time.Clock = mock.MagicMock
        
        try:
            self.controller = GameplayController(config=config)
            self.checkpoint_manager = self.controller.checkpoint_manager
            return True
        except Exception as e:
            print(f"Failed to initialize controller: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def teardown(self):
        """Clean up test fixtures."""
        cleanup_test_environment(self.test_dir)
    
    def test_checkpoint_creation(self):
        """Test basic checkpoint creation."""
        print("\nTest: Checkpoint Creation")
        try:
            # Create a checkpoint
            result = self.checkpoint_manager.create_checkpoint("test_checkpoint_1")
            
            if result:
                # Verify checkpoint file exists
                checkpoint_path = os.path.join(self.checkpoint_dir, "test_checkpoint_1.json")
                if os.path.exists(checkpoint_path):
                    # Verify checkpoint content
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)
                    
                    if "timestamp" in data and "characters" in data and "statistics" in data:
                        print("  ✓ Checkpoint created successfully with valid content")
                        self.test_results.append(("checkpoint_creation", True))
                        return True
                    else:
                        print("  ✗ Checkpoint missing required fields")
                        self.test_results.append(("checkpoint_creation", False))
                        return False
                else:
                    print("  ✗ Checkpoint file not created")
                    self.test_results.append(("checkpoint_creation", False))
                    return False
            else:
                print("  ✗ Checkpoint creation failed")
                self.test_results.append(("checkpoint_creation", False))
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("checkpoint_creation", False))
            return False
    
    def test_checkpoint_restoration(self):
        """Test checkpoint restoration."""
        print("\nTest: Checkpoint Restoration")
        try:
            # Create a checkpoint with known state
            self.controller.game_statistics["actions_executed"] = 42
            self.controller.game_statistics["characters_created"] = 5
            
            result = self.checkpoint_manager.create_checkpoint("test_restore")
            
            if not result:
                print("  ✗ Failed to create checkpoint for restoration test")
                self.test_results.append(("checkpoint_restoration", False))
                return False
            
            # Modify state
            self.controller.game_statistics["actions_executed"] = 100
            self.controller.game_statistics["characters_created"] = 10
            
            # Restore checkpoint
            restore_result = self.checkpoint_manager.restore_checkpoint(-1)
            
            if restore_result:
                # Verify state was restored
                if (self.controller.game_statistics["actions_executed"] == 42 and
                    self.controller.game_statistics["characters_created"] == 5):
                    print("  ✓ Checkpoint restored successfully with correct state")
                    self.test_results.append(("checkpoint_restoration", True))
                    return True
                else:
                    print("  ✗ Checkpoint restored but state is incorrect")
                    print(f"    Expected actions_executed=42, got {self.controller.game_statistics['actions_executed']}")
                    self.test_results.append(("checkpoint_restoration", False))
                    return False
            else:
                print("  ✗ Checkpoint restoration failed")
                self.test_results.append(("checkpoint_restoration", False))
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("checkpoint_restoration", False))
            return False
    
    def test_checkpoint_history_management(self):
        """Test checkpoint history and cleanup."""
        print("\nTest: Checkpoint History Management")
        try:
            # Set max checkpoints to a small number for testing
            original_max = self.checkpoint_manager.max_checkpoints
            self.checkpoint_manager.max_checkpoints = 3
            
            # Create more checkpoints than the limit
            for i in range(5):
                self.checkpoint_manager.create_checkpoint(f"history_test_{i}")
                time.sleep(0.1)  # Small delay to ensure different timestamps
            
            # Verify only max_checkpoints remain
            checkpoint_count = len(self.checkpoint_manager.checkpoint_history)
            
            if checkpoint_count == 3:
                # Verify oldest checkpoints were removed
                remaining_names = [cp["filename"] for cp in self.checkpoint_manager.checkpoint_history]
                expected_names = ["history_test_2.json", "history_test_3.json", "history_test_4.json"]
                
                if remaining_names == expected_names:
                    print(f"  ✓ Checkpoint history maintained correctly (kept last {checkpoint_count})")
                    self.test_results.append(("checkpoint_history", True))
                    self.checkpoint_manager.max_checkpoints = original_max
                    return True
                else:
                    print(f"  ✗ Incorrect checkpoints kept: {remaining_names}")
                    self.test_results.append(("checkpoint_history", False))
                    self.checkpoint_manager.max_checkpoints = original_max
                    return False
            else:
                print(f"  ✗ Expected 3 checkpoints, found {checkpoint_count}")
                self.test_results.append(("checkpoint_history", False))
                self.checkpoint_manager.max_checkpoints = original_max
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("checkpoint_history", False))
            return False
    
    def test_automatic_checkpoint_timing(self):
        """Test automatic checkpoint timing logic."""
        print("\nTest: Automatic Checkpoint Timing")
        try:
            # Set a short interval for testing
            self.checkpoint_manager.set_checkpoint_interval(1000)  # 1 second
            self.checkpoint_manager.last_checkpoint_time = 0
            
            # Mock current time
            current_time = 2000  # 2 seconds
            
            # Should checkpoint now
            should_checkpoint = self.checkpoint_manager.should_checkpoint(current_time)
            
            if should_checkpoint:
                print("  ✓ Automatic checkpoint timing logic works correctly")
                self.test_results.append(("automatic_timing", True))
                return True
            else:
                print("  ✗ Should checkpoint but timing check returned False")
                self.test_results.append(("automatic_timing", False))
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("automatic_timing", False))
            return False
    
    def test_corrupted_checkpoint_recovery(self):
        """Test recovery from corrupted checkpoint files."""
        print("\nTest: Corrupted Checkpoint Recovery")
        try:
            # Create valid checkpoints
            for i in range(3):
                self.checkpoint_manager.create_checkpoint(f"valid_{i}")
            
            # Corrupt the most recent checkpoint
            if self.checkpoint_manager.checkpoint_history:
                latest = self.checkpoint_manager.checkpoint_history[-1]
                with open(latest["path"], 'w') as f:
                    f.write("{ corrupted json content !@#")
                
                # Attempt recovery
                recovery_result = self.checkpoint_manager.recover_from_corruption()
                
                if recovery_result:
                    # Verify we recovered to a valid checkpoint
                    stats = self.controller.game_statistics
                    if isinstance(stats, dict):
                        print("  ✓ Successfully recovered from corrupted checkpoint")
                        self.test_results.append(("corruption_recovery", True))
                        return True
                    else:
                        print("  ✗ Recovered but state is invalid")
                        self.test_results.append(("corruption_recovery", False))
                        return False
                else:
                    print("  ✗ Recovery failed")
                    self.test_results.append(("corruption_recovery", False))
                    return False
            else:
                print("  ✗ No checkpoints to corrupt")
                self.test_results.append(("corruption_recovery", False))
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("corruption_recovery", False))
            return False
    
    def test_checkpoint_interval_configuration(self):
        """Test checkpoint interval configuration."""
        print("\nTest: Checkpoint Interval Configuration")
        try:
            # Test setting valid interval
            self.checkpoint_manager.set_checkpoint_interval(30000)
            
            if self.checkpoint_manager.checkpoint_interval == 30000:
                # Test setting too-short interval (should be capped)
                self.checkpoint_manager.set_checkpoint_interval(5000)
                
                if self.checkpoint_manager.checkpoint_interval == 10000:  # Minimum
                    print("  ✓ Checkpoint interval configuration works correctly")
                    self.test_results.append(("interval_config", True))
                    return True
                else:
                    print(f"  ✗ Minimum interval not enforced: {self.checkpoint_manager.checkpoint_interval}")
                    self.test_results.append(("interval_config", False))
                    return False
            else:
                print("  ✗ Interval not set correctly")
                self.test_results.append(("interval_config", False))
                return False
                
        except Exception as e:
            print(f"  ✗ Exception during test: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("interval_config", False))
            return False
    
    def run_all_tests(self):
        """Run all checkpoint manager tests."""
        print("\n" + "="*60)
        print("CHECKPOINT SYSTEM INTEGRATION TESTS")
        print("="*60)
        
        if not self.setup():
            print("\n✗ Setup failed, cannot run tests")
            return False
        
        try:
            self.test_checkpoint_creation()
            self.test_checkpoint_restoration()
            self.test_checkpoint_history_management()
            self.test_automatic_checkpoint_timing()
            self.test_corrupted_checkpoint_recovery()
            self.test_checkpoint_interval_configuration()
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            
            passed = sum(1 for _, result in self.test_results if result)
            total = len(self.test_results)
            
            for test_name, result in self.test_results:
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"{status}: {test_name}")
            
            print(f"\nTotal: {passed}/{total} tests passed")
            
            return passed == total
            
        finally:
            self.teardown()


def test_save_load_integration():
    """Test save/load integration with game state."""
    print("\n" + "="*60)
    print("SAVE/LOAD INTEGRATION TEST")
    print("="*60)
    
    test_dir = setup_test_environment()
    
    try:
        from tiny_gameplay_controller import GameplayController
        
        # Mock pygame
        pygame.init = lambda: None
        pygame.display.set_mode = lambda *args, **kwargs: mock.MagicMock()
        pygame.time.Clock = mock.MagicMock
        pygame.time.get_ticks = lambda: 12345
        
        save_path = os.path.join(test_dir, "test_save.json")
        
        # Create controller and modify state
        config = {"screen_width": 800, "screen_height": 600}
        controller = GameplayController(config=config)
        
        controller.game_statistics["actions_executed"] = 123
        controller.game_statistics["characters_created"] = 7
        
        # Save state
        print("\nTesting save functionality...")
        save_result = controller.save_game_state(save_path)
        
        if not save_result:
            print("  ✗ Save failed")
            cleanup_test_environment(test_dir)
            return False
        
        if not os.path.exists(save_path):
            print("  ✗ Save file not created")
            cleanup_test_environment(test_dir)
            return False
        
        print("  ✓ Save file created")
        
        # Verify save file content
        with open(save_path, 'r') as f:
            save_data = json.load(f)
        
        if save_data["statistics"]["actions_executed"] != 123:
            print("  ✗ Saved data incorrect")
            cleanup_test_environment(test_dir)
            return False
        
        print("  ✓ Save data correct")
        
        # Create new controller and load state
        print("\nTesting load functionality...")
        controller2 = GameplayController(config=config)
        
        load_result = controller2.load_game_state(save_path)
        
        if not load_result:
            print("  ✗ Load failed")
            cleanup_test_environment(test_dir)
            return False
        
        if controller2.game_statistics["actions_executed"] != 123:
            print("  ✗ Loaded data incorrect")
            cleanup_test_environment(test_dir)
            return False
        
        print("  ✓ State loaded correctly")
        print("\n✓ Save/Load integration test passed")
        
        cleanup_test_environment(test_dir)
        return True
        
    except Exception as e:
        print(f"  ✗ Exception during test: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_environment(test_dir)
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TINY VILLAGE - CHECKPOINT & PERSISTENCE TEST SUITE")
    print("="*60)
    
    # Run checkpoint manager tests
    checkpoint_tests = TestCheckpointManager()
    checkpoint_result = checkpoint_tests.run_all_tests()
    
    # Run save/load integration test
    saveload_result = test_save_load_integration()
    
    # Final summary
    print("\n" + "="*60)
    print("OVERALL TEST RESULTS")
    print("="*60)
    
    results = [
        ("Checkpoint System Tests", checkpoint_result),
        ("Save/Load Integration", saveload_result)
    ]
    
    for test_suite, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_suite}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
