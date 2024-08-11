import pygame
from tiny_strategy_manager import StrategyManager
from tiny_event_handler import EventHandler, Event
from tiny_types import GraphManager
from tiny_map_controller import MapController

""" 
This script integrates with the game loop, applying decisions from the strategy manager to the game state.
5. Gameplay Execution
Where it happens: gameplay_controller.py
What happens: The gameplay controller applies the decided plan to the game state, triggering animations, interactions, and state changes in the game. 

"""


class GameplayController:
    def __init__(self, graph_manager: GraphManager):
        self.strategy_manager = StrategyManager()
        self.graph_manager = graph_manager
        self.event_handler = EventHandler(self.graph_manager)
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.running = True

        # Initialize the Map Controller
        self.map_controller = MapController(
            "path_to_map_image.png",
            map_data={
                "width": 100,
                "height": 100,
                "buildings": [
                    {"name": "Town Hall", "rect": pygame.Rect(100, 150, 50, 50)},
                    # Add more buildings with their positions and sizes
                ],
            },
        )

        # Initialize other game systems (e.g., characters, events, etc.)
        self.initialize_game_systems()

    def initialize_game_systems(self):
        # Initialize other game components like characters, events, etc.
        pass

    def game_loop(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Frame time in seconds
            self.handle_events()
            self.update_game_state(dt)
            self.render()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            else:
                # Pass events to the Map Controller for handling
                self.map_controller.handle_event(event)

    def update_game_state(self, dt):
        # Update all game systems
        self.map_controller.update(dt)
        # Update other systems (e.g., AI, events, etc.)

    def render(self):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Render the map and game world
        self.map_controller.render(self.screen)

        # Render other game elements (UI, HUD, etc.)

        # Flip the display to show the updated frame
        pygame.display.flip()

    def run(self):
        # Main entry point to start the game loop
        self.game_loop()

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


if __name__ == "__main__":
    game_controller = GameplayController()
    game_controller.run()
    pygame.quit()
