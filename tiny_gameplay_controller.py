from tiny_strategy_manager import StrategyManager
from tiny_event_handler import EventHandler, Event

""" 
This script integrates with the game loop, applying decisions from the strategy manager to the game state.
5. Gameplay Execution
Where it happens: gameplay_controller.py
What happens: The gameplay controller applies the decided plan to the game state, triggering animations, interactions, and state changes in the game. 

"""


class GameplayController:
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.event_handler = EventHandler()

    def update(self, game_state):
        # Check for new events
        events = self.event_handler.check_events()
        # Update strategy based on events
        decisions = self.strategy_manager.update_strategy(events)
        # Apply decisions to game state
        for decision in decisions:
            self.apply_decision(decision, game_state)

    def apply_decision(self, decision, game_state):
        # Applies each action in the decision to the game state
        for action in decision:
            # Execute actions like visiting the cafe, going jogging, etc.
            print(f"Executing {action['name']} action")
            action.execute(game_state)
