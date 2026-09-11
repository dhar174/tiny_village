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

    # Enhanced Mock Goal that better matches real Goal interface
    class MockGoal:
        def __init__(self, criteria, name="TestGoal", priority=0.5, target_effects=None):
            self.criteria = criteria
            self.name = name
            self.priority = priority
            self.score = priority  # alias for compatibility
            self.target_effects = target_effects if target_effects else {}
            self.completed = False
            self.description = "Test goal for difficulty calculation"
            
            # Required attributes for goal difficulty calculation
            self.character = None
            self.target = None
            self.required_items = self._extract_required_items()
            
        def _extract_required_items(self):
            """Extract required items from criteria to match real Goal interface."""
            required_items = []
            for criterion in self.criteria:
                if "node_attributes" in criterion:
                    if "item_type" in criterion["node_attributes"]:
                        required_items.append(criterion["node_attributes"])
            return required_items
            
        def check_completion(self, state=None):
            """Check goal completion - matches real Goal interface."""
            return self.completed

    # Test a simple scenario with enhanced goal
    goal = MockGoal(
        criteria=[{"node_attributes": {"type": "test"}}],
        name="FindTestItem",
        priority=0.8,
        target_effects={"test_completion": 1.0}
    )

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
