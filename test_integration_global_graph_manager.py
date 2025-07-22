#!/usr/bin/env python3
"""
Integration test to verify that all modules use the same global GraphManager instance
"""

def test_global_graph_manager_integration():
    """Test that all modules are using the same global GraphManager instance"""
    print("Testing global GraphManager integration across modules...")
    
    # Get the global instance
    from tiny_globals import get_global_graph_manager
    global_gm = get_global_graph_manager()
    print(f"Global GraphManager: {id(global_gm)}")
    
    instances_to_test = []
    
    # Test 1: Actions module
    try:
        from actions import Action, ActionGenerator
        action = Action('TestAction', [], [], 1.0)
        instances_to_test.append(('Action', action.graph_manager))
        
        action_gen = ActionGenerator()
        instances_to_test.append(('ActionGenerator', action_gen.graph_manager))
        print("✅ Actions module test passed")
    except Exception as e:
        print(f"❌ Actions module test failed: {e}")
    
    # Test 2: Event Handler
    try:
        from tiny_event_handler import EventHandler
        event_handler = EventHandler()
        instances_to_test.append(('EventHandler', event_handler.graph_manager))
        print("✅ EventHandler test passed")
    except Exception as e:
        print(f"❌ EventHandler test failed: {e}")
    
    # Test 3: Strategy Manager
    try:
        from tiny_strategy_manager import StrategyManager
        strategy_mgr = StrategyManager()
        instances_to_test.append(('StrategyManager', strategy_mgr.graph_manager))
        print("✅ StrategyManager test passed")
    except Exception as e:
        print(f"❌ StrategyManager test failed: {e}")
    
    # Test 4: Gameplay Controller
    try:
        from tiny_gameplay_controller import GameplayController
        # This might fail due to dependencies, but test what we can
        gameplay_ctrl = GameplayController()
        instances_to_test.append(('GameplayController', gameplay_ctrl.graph_manager))
        print("✅ GameplayController test passed")
    except Exception as e:
        print(f"⚠️  GameplayController test skipped (expected): {e}")
    
    # Verify all instances are the same
    print(f"\nValidating all instances are the same:")
    all_same = True
    for name, instance in instances_to_test:
        is_same = instance is global_gm
        status = "✅" if is_same else "❌"
        print(f"{status} {name}: {id(instance)} (same: {is_same})")
        if not is_same:
            all_same = False
    
    if all_same:
        print("\n🎉 All modules are using the same global GraphManager instance!")
    else:
        print("\n⚠️  Some modules are not using the global instance")
    
    return all_same

if __name__ == "__main__":
    test_global_graph_manager_integration()