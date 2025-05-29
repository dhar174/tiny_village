#!/usr/bin/env python3

"""
Simple test script for GraphManager.calculate_goal_difficulty()
"""

import sys
import traceback

print("Starting simple GraphManager test...")

try:
    # Test basic imports
    print("1. Testing NetworkX import...")
    import networkx as nx

    print("   ✓ NetworkX imported successfully")

    print("2. Testing GraphManager import...")
    from tiny_graph_manager import GraphManager

    print("   ✓ GraphManager imported successfully")

    print("3. Testing Character import...")
    from tiny_characters import Character

    print("   ✓ Character imported successfully")

    print("4. Testing basic GraphManager functionality...")
    # Create a simple test
    gm = GraphManager()
    print("   ✓ GraphManager instance created")

    # Create a simple graph with nodes
    gm.G = nx.Graph()
    gm.G.add_node("test_node", type="test", item_type="food")
    print("   ✓ Simple graph created")

    # Mock the required methods for testing
    class MockGoal:
        def __init__(self, criteria):
            self.criteria = criteria

    # Test a simple scenario
    goal = MockGoal([{"node_attributes": {"type": "test"}}])

    # Define get_filtered_nodes method temporarily
    def mock_get_filtered_nodes(**kwargs):
        if kwargs.get("node_attributes", {}).get("type") == "test":
            return {"test_node": {}}
        return {}

    def mock_calculate_action_viability_cost(node, goal, character):
        return {
            "action_cost": {"test_action": 1},
            "viable": {"test_action": True},
            "goal_cost": {"test_action": 2},
            "conditions_fulfilled_by_action": {"test_action": ["test_condition"]},
            "actions_that_fulfill_condition": {
                "test_condition": [("test_action", node)]
            },
        }

    def mock_calculate_edge_cost(u, v):
        return 1

    # Monkey patch the methods
    gm.get_filtered_nodes = mock_get_filtered_nodes
    gm.calculate_action_viability_cost = mock_calculate_action_viability_cost
    gm.calculate_edge_cost = mock_calculate_edge_cost

    print("5. Testing calculate_goal_difficulty...")
    result = gm.calculate_goal_difficulty(goal, None)
    print(f"   ✓ Result: {type(result)}")

    if isinstance(result, dict) and "difficulty" in result:
        print(f"   ✓ Difficulty calculated: {result['difficulty']}")
        print("   ✓ Test completed successfully!")
    else:
        print(f"   ⚠ Unexpected result format: {result}")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
