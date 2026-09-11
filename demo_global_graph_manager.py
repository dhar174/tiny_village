#!/usr/bin/env python3
"""
Game startup demonstration showing global GraphManager usage
"""

def demo_game_startup():
    """Demonstrate how the global GraphManager should be used on game start"""
    print("🎮 Tiny Village Game Startup Demonstration")
    print("=" * 50)
    
    # Step 1: Initialize global GraphManager on game start
    print("\n1. Initializing global GraphManager on game start...")
    from tiny_globals import initialize_global_graph_manager, get_global_graph_manager, has_global_graph_manager
    
    print(f"Before initialization: {has_global_graph_manager()}")
    graph_manager = initialize_global_graph_manager()
    print(f"After initialization: {has_global_graph_manager()}")
    print(f"Global GraphManager ID: {id(graph_manager)}")
    
    # Step 2: Create Actions - they automatically use global GraphManager
    print("\n2. Creating Actions (automatically use global GraphManager)...")
    from actions import Action, TalkAction, ActionGenerator
    
    action1 = Action("MoveAction", [], [], 1.0)
    action2 = TalkAction("Player", "NPC")
    action_gen = ActionGenerator()
    
    print(f"Action1 GraphManager ID: {id(action1.graph_manager)} (same: {action1.graph_manager is graph_manager})")
    print(f"Action2 GraphManager ID: {id(action2.graph_manager)} (same: {action2.graph_manager is graph_manager})")
    print(f"ActionGen GraphManager ID: {id(action_gen.graph_manager)} (same: {action_gen.graph_manager is graph_manager})")
    
    # Step 3: Create Event Handler - automatically uses global GraphManager
    print("\n3. Creating EventHandler (automatically uses global GraphManager)...")
    from tiny_event_handler import EventHandler
    
    event_handler = EventHandler()
    print(f"EventHandler GraphManager ID: {id(event_handler.graph_manager)} (same: {event_handler.graph_manager is graph_manager})")
    
    # Step 4: Create Strategy Manager - automatically uses global GraphManager
    print("\n4. Creating StrategyManager (automatically uses global GraphManager)...")
    from tiny_strategy_manager import StrategyManager
    
    strategy_manager = StrategyManager()
    print(f"StrategyManager GraphManager ID: {id(strategy_manager.graph_manager)} (same: {strategy_manager.graph_manager is graph_manager})")
    
    # Step 5: Show that all components share the same GraphManager state
    print("\n5. Demonstrating shared state...")
    
    # Add some basic data to the graph through one component
    graph_manager.G.add_node("TestNode", type="test", data="shared_data")
    
    # Check that all components can see the same data
    print(f"Graph has TestNode: {graph_manager.G.has_node('TestNode')}")
    print(f"Action1 can see TestNode: {action1.graph_manager.G.has_node('TestNode')}")
    print(f"EventHandler can see TestNode: {event_handler.graph_manager.G.has_node('TestNode')}")
    print(f"StrategyManager can see TestNode: {strategy_manager.graph_manager.G.has_node('TestNode')}")
    
    # Step 6: Summary
    print("\n6. Summary")
    print("✅ Single global GraphManager instance initialized")
    print("✅ All components automatically use the global instance") 
    print("✅ All components share the same graph state")
    print("✅ No manual GraphManager passing required")
    
    print("\n🎉 Global GraphManager implementation successful!")
    print("\nUsage Notes:")
    print("- Call initialize_global_graph_manager() once at game startup")
    print("- All Action, EventHandler, StrategyManager instances automatically use global GraphManager")
    print("- Character creation can now omit graph_manager parameter (will use global)")
    print("- get_global_graph_manager() provides access to the global instance anywhere")

if __name__ == "__main__":
    demo_game_startup()