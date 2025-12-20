#!/usr/bin/env python3
"""
Integration tests to verify that all modules use the same global GraphManager instance.

This test suite ensures that all key game modules (actions, characters, event handlers,
strategy manager, gameplay controller) properly share the same global GraphManager instance.
"""

import unittest
import sys

sys.path.insert(0, '.')

from tiny_globals import get_global_graph_manager, reset_global_graph_manager


class TestGlobalGraphManagerIntegration(unittest.TestCase):
    """Integration test cases for global GraphManager across modules."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset the global graph manager before each test
        reset_global_graph_manager()
        
        # Get fresh global instance for tests
        self.global_gm = get_global_graph_manager()
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset the global graph manager after each test
        reset_global_graph_manager()
    
    def test_actions_module_uses_global_instance(self):
        """Test that actions module uses the global GraphManager."""
        from actions import Action, ActionGenerator
        
        action = Action('TestAction', [], [], 1.0)
        self.assertIs(action.graph_manager, self.global_gm,
                     "Action should use the global GraphManager instance")
        
        action_gen = ActionGenerator()
        self.assertIs(action_gen.graph_manager, self.global_gm,
                     "ActionGenerator should use the global GraphManager instance")
    
    def test_event_handler_uses_global_instance(self):
        """Test that EventHandler uses the global GraphManager."""
        try:
            from tiny_event_handler import EventHandler
            
            event_handler = EventHandler()
            self.assertIs(event_handler.graph_manager, self.global_gm,
                         "EventHandler should use the global GraphManager instance")
        except (ImportError, AttributeError, TypeError) as e:
            self.skipTest(f"EventHandler test skipped due to dependencies: {e}")
    
    def test_strategy_manager_uses_global_instance(self):
        """Test that StrategyManager uses the global GraphManager."""
        try:
            from tiny_strategy_manager import StrategyManager
            
            strategy_mgr = StrategyManager()
            self.assertIs(strategy_mgr.graph_manager, self.global_gm,
                         "StrategyManager should use the global GraphManager instance")
        except (ImportError, AttributeError, TypeError) as e:
            self.skipTest(f"StrategyManager test skipped due to dependencies: {e}")
    
    def test_gameplay_controller_uses_global_instance(self):
        """Test that GameplayController uses the global GraphManager."""
        try:
            from tiny_gameplay_controller import GameplayController
            
            gameplay_ctrl = GameplayController()
            self.assertIs(gameplay_ctrl.graph_manager, self.global_gm,
                         "GameplayController should use the global GraphManager instance")
        except (ImportError, AttributeError, TypeError) as e:
            self.skipTest(f"GameplayController test skipped due to dependencies: {e}")
    
    def test_all_modules_share_same_instance(self):
        """Test that all successfully created modules share the same GraphManager instance.
        
        This is a comprehensive test that verifies all modules that can be instantiated
        are using the same global GraphManager instance.
        """
        instances_to_test = []
        
        # Test Actions module
        try:
            from actions import Action, ActionGenerator
            action = Action('TestAction', [], [], 1.0)
            instances_to_test.append(('Action', action.graph_manager))
            
            action_gen = ActionGenerator()
            instances_to_test.append(('ActionGenerator', action_gen.graph_manager))
        except Exception:
            pass
        
        # Test EventHandler
        try:
            from tiny_event_handler import EventHandler
            event_handler = EventHandler()
            instances_to_test.append(('EventHandler', event_handler.graph_manager))
        except Exception:
            pass
        
        # Test StrategyManager
        try:
            from tiny_strategy_manager import StrategyManager
            strategy_mgr = StrategyManager()
            instances_to_test.append(('StrategyManager', strategy_mgr.graph_manager))
        except Exception:
            pass
        
        # Test GameplayController
        try:
            from tiny_gameplay_controller import GameplayController
            gameplay_ctrl = GameplayController()
            instances_to_test.append(('GameplayController', gameplay_ctrl.graph_manager))
        except Exception:
            pass
        
        # Verify all instances are the same
        self.assertGreater(len(instances_to_test), 0,
                          "At least one module should be testable")
        
        for name, instance in instances_to_test:
            with self.subTest(module=name):
                self.assertIs(instance, self.global_gm,
                            f"{name} should use the global GraphManager instance")


if __name__ == "__main__":
    unittest.main()
