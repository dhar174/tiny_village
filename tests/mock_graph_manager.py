#!/usr/bin/env python3
"""
Minimal mock GraphManager for testing GOAPPlanner functionality.

This mock provides the basic interface that GOAPPlanner expects without 
requiring the full complexity of the real GraphManager with NetworkX dependencies.
"""

from actions import Action, State


class MockGraphManager:
    """
    Minimal mock implementation of GraphManager for testing GOAPPlanner.
    
    This mock provides the essential methods that GOAPPlanner calls on the 
    graph_manager, allowing tests to exercise the full code paths without 
    requiring the complexity of the real GraphManager.
    """
    
    def __init__(self, character=None):
        """
        Initialize the mock graph manager.
        
        Args:
            character: Optional character to associate with this graph manager
        """
        self.character = character or MockCharacter()
        self.G = {}  # Simple dict to simulate NetworkX graph
        
    def get_character_state(self, character_name):
        """
        Mock implementation of get_character_state.
        
        Args:
            character_name (str): Name of the character
            
        Returns:
            dict: Basic character state information
        """
        return {
            "location": "Home",
            "relationships": {"friend_count": 2},
            "activities": ["reading", "cooking"],
            "energy": getattr(self.character, 'energy', 50),
            "health": getattr(self.character, 'health', 80),
            "social_wellbeing": 60
        }
    
    def get_environment_conditions(self):
        """
        Mock implementation of get_environment_conditions.
        
        Returns:
            dict: Basic environment information
        """
        return {
            "weather": "sunny",
            "time_of_day": "morning",
            "season": "spring",
            "location_availability": ["park", "cafe", "library"]
        }
    
    def get_possible_actions(self, character_name):
        """
        Mock implementation of get_possible_actions.
        
        Args:
            character_name (str): Name of the character
            
        Returns:
            list: List of available Action objects
        """
        # Return a set of basic actions that any character can perform
        actions = [
            Action(
                name="Rest",
                preconditions=[],
                effects=[{"attribute": "energy", "change_value": 15}],
                cost=0.5
            ),
            Action(
                name="SocialActivity",
                preconditions=[],
                effects=[{"attribute": "social_wellbeing", "change_value": 10}],
                cost=1.0
            ),
            Action(
                name="Study",
                preconditions=[],
                effects=[{"attribute": "knowledge", "change_value": 5}],
                cost=2.0
            ),
            Action(
                name="Exercise",
                preconditions=[],
                effects=[
                    {"attribute": "health", "change_value": 8},
                    {"attribute": "energy", "change_value": -5}
                ],
                cost=1.5
            )
        ]
        
        # Add satisfaction and urgency attributes for utility calculations
        for i, action in enumerate(actions):
            action.satisfaction = 5 + i * 2  # Varying satisfaction levels
            action.urgency = 1.0 + i * 0.5   # Varying urgency levels
            
        return actions
    
    def find_alternative_actions(self, failed_action):
        """
        Mock implementation of find_alternative_actions.
        
        Args:
            failed_action: The action that failed
            
        Returns:
            list: List of alternative actions
        """
        # Create a simple alternative based on the failed action
        alternative_name = f"Alternative_{failed_action.name}"
        
        alternative_action = Action(
            name=alternative_name,
            preconditions=[],
            effects=getattr(failed_action, 'effects', []),
            cost=getattr(failed_action, 'cost', 1.0) + 0.5  # Slightly more expensive
        )
        
        # Copy utility attributes if they exist
        alternative_action.satisfaction = getattr(failed_action, 'satisfaction', 5)
        alternative_action.urgency = getattr(failed_action, 'urgency', 1.0)
        
        return [{
            "action": alternative_action,
            "priority": 5.0,  # Medium priority
            "dependencies": []
        }]
    
    def calculate_goal_difficulty(self, goal, character):
        """
        Mock implementation of calculate_goal_difficulty.
        
        Args:
            goal: The goal to evaluate
            character: The character attempting the goal
            
        Returns:
            dict: Goal difficulty information
        """
        # Simple mock difficulty calculation
        base_difficulty = 2.0
        
        # Adjust based on goal attributes if available
        if hasattr(goal, 'target_effects') and goal.target_effects:
            # More target effects = higher difficulty
            base_difficulty += len(goal.target_effects) * 0.5
        
        return {
            "difficulty": base_difficulty,
            "calc_path_cost": base_difficulty * 1.2,
            "calc_goal_cost": base_difficulty * 0.8,
            "action_viability_cost": 1.0,
            "viable_paths": 2,
            "shortest_path": ["action1", "action2"],
            "lowest_goal_cost_path": ["action1", "action3"],
            "shortest_path_goal_cost": base_difficulty,
            "lowest_goal_cost_path_cost": base_difficulty * 0.9
        }
    
    def plan_actions(self, character, goal, current_state, available_actions):
        """
        Mock implementation of plan_actions for replanning.
        
        Args:
            character: Character (can be None in replanning context)
            goal: Goal to achieve
            current_state: Current state
            available_actions: Available actions
            
        Returns:
            list: Simple plan of actions
        """
        if not available_actions:
            return []
        
        # Simple planning: return first available action that has positive effects
        for action in available_actions:
            if hasattr(action, 'effects') and action.effects:
                return [action]
        
        # Fallback: return first action
        return [available_actions[0]] if available_actions else []


class MockCharacter:
    """
    Simple mock character for testing.
    """
    
    def __init__(self, name="MockChar", energy=50, health=80):
        self.name = name
        self.energy = energy
        self.health = health
        self.social_wellbeing = 60
        
    def get_state(self):
        """Return character state as State object."""
        return State({
            "energy": self.energy,
            "health": self.health,
            "social_wellbeing": self.social_wellbeing,
            "name": self.name
        })


def create_mock_graph_manager(character=None):
    """
    Factory function to create a mock graph manager for testing.
    
    Args:
        character: Optional character to associate with the graph manager
        
    Returns:
        MockGraphManager: A minimal mock graph manager instance
    """
    return MockGraphManager(character)