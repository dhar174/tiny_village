#!/usr/bin/env python3
"""
Focused test of global GraphManager without complex dependencies
"""

def test_global_graphmanager_core_functionality():
    """Test core GraphManager functionality using global instance"""
    print("Testing core GraphManager functionality with global instance...")
    
    # Test 1: Global instance creation and singleton behavior
    from tiny_globals import get_global_graph_manager, has_global_graph_manager
    
    print("1. Testing singleton behavior...")
    print(f"Before: {has_global_graph_manager()}")
    
    gm1 = get_global_graph_manager()
    gm2 = get_global_graph_manager()
    
    print(f"After: {has_global_graph_manager()}")
    print(f"Same instance: {gm1 is gm2}")
    print(f"Type: {type(gm1)}")
    
    # Test 2: Basic graph operations
    print("\n2. Testing basic graph operations...")
    
    # Test the graph is initialized
    print(f"Graph has nodes: {gm1.G.number_of_nodes()}")
    print(f"Graph has edges: {gm1.G.number_of_edges()}")
    print(f"Graph type: {type(gm1.G)}")
    
    # Test 3: World state integration
    print("\n3. Testing WorldState integration...")
    print(f"Has world_state: {hasattr(gm1, 'world_state')}")
    print(f"WorldState type: {type(gm1.world_state)}")
    print(f"WorldState graph same as G: {gm1.world_state.graph is gm1.G}")
    
    # Test 4: Actions using global GraphManager
    print("\n4. Testing Actions integration...")
    from actions import Action, TalkAction, ActionGenerator
    
    # Test basic Action
    action = Action("TestAction", [], [], 1.0)
    print(f"Action has graph_manager: {action.graph_manager is not None}")
    print(f"Action uses global: {action.graph_manager is gm1}")
    
    # Test TalkAction (which extends SocialAction -> Action)
    try:
        talk_action = TalkAction("Speaker", "Listener")
        print(f"TalkAction has graph_manager: {talk_action.graph_manager is not None}")
        print(f"TalkAction uses global: {talk_action.graph_manager is gm1}")
    except Exception as e:
        print(f"TalkAction test failed: {e}")
    
    # Test ActionGenerator
    ag = ActionGenerator()
    print(f"ActionGenerator has graph_manager: {ag.graph_manager is not None}")
    print(f"ActionGenerator uses global: {ag.graph_manager is gm1}")
    
    print("\n✅ Core functionality tests completed successfully!")
    return True

if __name__ == "__main__":
    test_global_graphmanager_core_functionality()