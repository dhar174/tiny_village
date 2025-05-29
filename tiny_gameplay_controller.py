import pygame
import random
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
    def __init__(self, graph_manager: GraphManager = None):
        self.strategy_manager = StrategyManager()

        # Initialize graph manager if not provided
        if graph_manager is None:
            from tiny_graph_manager import GraphManager as ActualGraphManager

            self.graph_manager = ActualGraphManager()
        else:
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
        """Initialize all game systems and create sample characters."""
        try:
            # Import required systems
            from tiny_characters import Character
            from tiny_locations import Location
            from actions import ActionSystem
            from tiny_time_manager import GameTimeManager, GameCalendar
            from tiny_items import ItemInventory, ItemObject, FoodItem
            from tiny_animation_system import get_animation_system

            # Initialize core systems
            self.action_system = ActionSystem()
            self.action_system.setup_actions()

            # Initialize time management
            calendar = GameCalendar()
            self.gametime_manager = GameTimeManager(calendar)

            # Initialize animation system
            self.animation_system = get_animation_system()

            # Create sample characters for the village
            self.characters = {}
            sample_characters = self._create_sample_characters()

            # Add characters to the map controller
            for character in sample_characters:
                # Add character as a node in the graph
                self.graph_manager.add_character_node(character)

                # Add character to map controller with random starting position
                import random

                start_x = random.randint(50, 750)
                start_y = random.randint(50, 550)
                character.position = pygame.math.Vector2(start_x, start_y)
                character.color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                )
                character.path = []

                self.map_controller.characters[character.uuid] = character
                self.characters[character.uuid] = character

            # Initialize events system
            self.events = []

            print(
                f"Game systems initialized successfully with {len(self.characters)} characters"
            )

        except Exception as e:
            print(f"Error initializing game systems: {e}")
            # Fallback initialization
            self.characters = {}
            self.events = []

    def _create_sample_characters(self):
        """Create a few sample characters for the village."""
        from tiny_characters import Character
        from tiny_locations import Location
        from tiny_items import ItemInventory, ItemObject, FoodItem

        characters = []

        # Sample character data
        character_data = [
            {
                "name": "Alice Cooper",
                "age": 28,
                "pronouns": "she/her",
                "job": "Baker",
                "recent_event": "Opened new bakery",
                "long_term_goal": "Expand business to neighboring towns",
            },
            {
                "name": "Bob Wilson",
                "age": 35,
                "pronouns": "he/him",
                "job": "Blacksmith",
                "recent_event": "Crafted special sword for mayor",
                "long_term_goal": "Master legendary crafting techniques",
            },
            {
                "name": "Charlie Green",
                "age": 42,
                "pronouns": "they/them",
                "job": "Farmer",
                "recent_event": "Harvested record crop",
                "long_term_goal": "Develop sustainable farming practices",
            },
            {
                "name": "Diana Stone",
                "age": 31,
                "pronouns": "she/her",
                "job": "Teacher",
                "recent_event": "Student won regional competition",
                "long_term_goal": "Build the village's first library",
            },
        ]

        for char_data in character_data:
            try:
                # Create basic inventory
                inventory = ItemInventory(
                    food_items=[
                        FoodItem("Bread", "food", 2, 1, 1, "Fresh bread", calories=200),
                        FoodItem("Apple", "food", 1, 0.5, 2, "Red apple", calories=80),
                    ]
                )

                # Create location for character
                location = Location(
                    f"{char_data['name']}'s Home",
                    random.randint(0, 100),
                    random.randint(0, 100),
                    1,
                    1,
                    self.action_system,
                )

                # Create character
                character = Character(
                    name=char_data["name"],
                    age=char_data["age"],
                    pronouns=char_data["pronouns"],
                    job=char_data["job"],
                    health_status=random.randint(80, 100),
                    hunger_level=random.randint(20, 50),
                    wealth_money=random.randint(50, 200),
                    mental_health=random.randint(70, 90),
                    social_wellbeing=random.randint(60, 90),
                    job_performance=random.randint(70, 95),
                    community=random.randint(50, 80),
                    recent_event=char_data["recent_event"],
                    long_term_goal=char_data["long_term_goal"],
                    inventory=inventory,
                    personality_traits={
                        "extraversion": random.randint(30, 80),
                        "openness": random.randint(40, 90),
                        "conscientiousness": random.randint(50, 90),
                        "agreeableness": random.randint(40, 85),
                        "neuroticism": random.randint(10, 50),
                    },
                    action_system=self.action_system,
                    gametime_manager=self.gametime_manager,
                    location=location,
                    graph_manager=self.graph_manager,
                    energy=random.randint(60, 100),
                )

                characters.append(character)

            except Exception as e:
                print(f"Error creating character {char_data['name']}: {e}")
                continue

        return characters

    def game_loop(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Frame time in seconds
            self.handle_events()
            self.update_game_state(dt)
            self.render()

    def handle_events(self):
        """Handle pygame events and user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause simulation
                    self.paused = not getattr(self, "paused", False)
                    print(f"Game {'paused' if self.paused else 'unpaused'}")
                elif event.key == pygame.K_r:
                    # Reset/regenerate characters
                    self._reset_characters()
            else:
                # Pass events to the Map Controller for handling
                self.map_controller.handle_event(event)

    def _reset_characters(self):
        """Reset and regenerate characters."""
        try:
            # Clear existing characters
            self.map_controller.characters.clear()
            self.characters.clear()

            # Create new characters
            sample_characters = self._create_sample_characters()

            for character in sample_characters:
                # Add character to graph
                self.graph_manager.add_character_node(character)

                # Add character to map with random position
                start_x = random.randint(50, 750)
                start_y = random.randint(50, 550)
                character.position = pygame.math.Vector2(start_x, start_y)
                character.color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                )
                character.path = []

                self.map_controller.characters[character.uuid] = character
                self.characters[character.uuid] = character

            print(f"Reset complete - {len(self.characters)} new characters created")

        except Exception as e:
            print(f"Error resetting characters: {e}")

    def update_game_state(self, dt):
        """Update all game systems with delta time."""
        # Check if game is paused
        if getattr(self, "paused", False):
            return  # Skip all updates when paused

        # Update the map controller (handles character movement and pathfinding)
        self.map_controller.update(dt)

        # Update character AI and decision making
        for character in self.characters.values():
            try:
                # Update character's memory and decision making
                character.recall_recent_memories()
                goals = character.evaluate_goals()

                # Use strategy manager to plan actions
                if goals:
                    current_goal = goals[0][1]  # Get highest priority goal
                    actions = self.strategy_manager.get_daily_actions(character)
                    # Execute planned actions based on current state

            except Exception as e:
                print(f"Error updating character {character.name}: {e}")
                continue

        # Update time manager
        if hasattr(self, "gametime_manager"):
            # Process any scheduled behaviors
            for behavior in self.gametime_manager.get_scheduled_behaviors():
                behavior.check_calendar()

        # Update animation system
        if hasattr(self, "animation_system"):
            self.animation_system.update(dt)

        # Process any pending events
        if hasattr(self, "events"):
            for event in self.events[:]:  # Copy list to safely modify during iteration
                # Process event logic here
                pass

    def render(self):
        """Render all game elements."""
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Render the map and game world
        self.map_controller.render(self.screen)

        # Render UI elements
        self._render_ui()

        # Flip the display to show the updated frame
        pygame.display.flip()

    def _render_ui(self):
        """Render user interface elements."""
        try:
            # Create font for UI text
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)

            # Render character count
            char_count_text = font.render(
                f"Characters: {len(self.characters)}", True, (255, 255, 255)
            )
            self.screen.blit(char_count_text, (10, 10))

            # Render pause status
            if getattr(self, "paused", False):
                pause_text = font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.screen.get_width() - 100, 10))

            # Render time if available
            if hasattr(self, "gametime_manager"):
                try:
                    game_time = (
                        self.gametime_manager.get_calendar().get_game_time_string()
                    )
                    time_text = small_font.render(
                        f"Time: {game_time}", True, (255, 255, 255)
                    )
                    self.screen.blit(time_text, (10, 35))
                except:
                    pass

            # Render selected character info
            if (
                hasattr(self.map_controller, "selected_character")
                and self.map_controller.selected_character
            ):
                char = self.map_controller.selected_character
                char_info = [
                    f"Selected: {char.name}",
                    f"Job: {char.job}",
                    f"Energy: {char.energy}",
                    f"Health: {char.health_status}",
                ]

                for i, info in enumerate(char_info):
                    info_text = small_font.render(info, True, (255, 255, 0))
                    self.screen.blit(info_text, (10, 60 + i * 20))

            # Render instructions
            instructions = [
                "Click characters to select them",
                "Click buildings to interact",
                "SPACE to pause/unpause",
                "R to reset characters",
                "ESC to quit",
            ]

            for i, instruction in enumerate(instructions):
                inst_text = small_font.render(instruction, True, (200, 200, 200))
                self.screen.blit(
                    inst_text, (10, self.screen.get_height() - 80 + i * 15)
                )

        except Exception as e:
            # Fallback to minimal UI
            font = pygame.font.Font(None, 24)
            error_text = font.render("UI Error", True, (255, 0, 0))
            self.screen.blit(error_text, (10, 10))

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
