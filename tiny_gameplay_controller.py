import pygame
import random
import logging
import traceback
from typing import Dict, List, Any, Union, Optional
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

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionResolver:
    """
    Handles conversion between different action formats and execution.

    TODO: Add action caching for performance optimization
    TODO: Add action validation and precondition checking
    TODO: Add action priority and scheduling system
    TODO: Add action dependency resolution
    TODO: Add action effect prediction and simulation
    TODO: Add action history tracking and analytics
    TODO: Add custom action type registration system
    TODO: Add action serialization/deserialization
    TODO: Add action middleware/plugin system
    TODO: Add action cost calculation and resource management
    """

    def __init__(self, action_system=None):
        self.action_system = action_system
        self.fallback_actions = {
            "default_rest": {"name": "Rest", "energy_cost": -10, "satisfaction": 5},
            "default_idle": {"name": "Idle", "energy_cost": 0, "satisfaction": 1},
        }

    def resolve_action(
        self, action_data: Union[Dict, Any], character=None
    ) -> Optional[Any]:
        """Convert action data to executable action object."""
        try:
            # If it's already an Action object with execute method
            if hasattr(action_data, "execute"):
                return action_data

            # If it's a dictionary, try to convert to Action
            if isinstance(action_data, dict):
                return self._dict_to_action(action_data, character)

            # If it's a string (action name), try to resolve
            if isinstance(action_data, str):
                return self._resolve_by_name(action_data, character)

            logger.warning(f"Unknown action format: {type(action_data)}")
            return None

        except Exception as e:
            logger.error(f"Error resolving action {action_data}: {e}")
            return None

    def _dict_to_action(self, action_dict: Dict, character=None):
        """Convert dictionary action to Action object."""
        try:
            from actions import Action

            # Extract action information
            name = action_dict.get("name", "Unknown Action")
            energy_cost = action_dict.get("energy_cost", 0)
            satisfaction = action_dict.get("satisfaction", 0)

            # Create basic effects based on dictionary
            effects = []
            if energy_cost != 0:
                effects.append(
                    {
                        "targets": ["initiator"],
                        "attribute": "energy",
                        "change_value": -abs(
                            energy_cost
                        ),  # Energy cost is always negative
                    }
                )
            if satisfaction != 0:
                effects.append(
                    {
                        "targets": ["initiator"],
                        "attribute": "current_satisfaction",
                        "change_value": satisfaction,
                    }
                )

            # Create simple Action object
            action = Action(
                name=name,
                preconditions={},  # No preconditions for dict actions
                effects=effects,
                cost=max(1, abs(energy_cost)),
                initiator=character,
                default_target_is_initiator=True,
            )

            return action

        except Exception as e:
            logger.error(f"Error converting dict to action: {e}")
            return None

    def _resolve_by_name(self, action_name: str, character=None):
        """Resolve action by name using action system."""
        try:
            if self.action_system:
                # Try to generate action using action system
                actions = self.action_system.generate_actions(character)
                for action in actions:
                    if (
                        hasattr(action, "name")
                        and action.name.lower() == action_name.lower()
                    ):
                        return action

            # Fallback: create simple action
            return self._dict_to_action(
                {"name": action_name, "energy_cost": 5, "satisfaction": 2}, character
            )

        except Exception as e:
            logger.error(f"Error resolving action by name {action_name}: {e}")
            return None

    def get_fallback_action(self, character=None):
        """Get a safe fallback action when others fail."""
        return self._dict_to_action(self.fallback_actions["default_rest"], character)


class GameplayController:
    def __init__(
        self, graph_manager: GraphManager = None, config: Dict[str, Any] = None
    ):
        """Initialize the gameplay controller with improved error handling and configuration."""
        self.config = config or {}
        self.initialization_errors = []
        self.running = True
        self.paused = False

        # Initialize core systems with error handling
        try:
            self.strategy_manager = StrategyManager()
        except Exception as e:
            logger.error(f"Failed to initialize StrategyManager: {e}")
            self.strategy_manager = None
            self.initialization_errors.append("StrategyManager initialization failed")

        # Initialize graph manager if not provided
        if graph_manager is None:
            try:
                from tiny_graph_manager import GraphManager as ActualGraphManager

                self.graph_manager = ActualGraphManager()
            except Exception as e:
                logger.error(f"Failed to initialize GraphManager: {e}")
                self.graph_manager = None
                self.initialization_errors.append("GraphManager initialization failed")
        else:
            self.graph_manager = graph_manager

        # Initialize event handler
        try:
            self.event_handler = (
                EventHandler(self.graph_manager) if self.graph_manager else None
            )
        except Exception as e:
            logger.error(f"Failed to initialize EventHandler: {e}")
            self.event_handler = None
            self.initialization_errors.append("EventHandler initialization failed")

        # Initialize pygame with error handling
        try:
            pygame.init()
            screen_width = self.config.get("screen_width", 800)
            screen_height = self.config.get("screen_height", 600)
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Tiny Village")
            self.clock = pygame.time.Clock()
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            self.screen = None
            self.clock = None
            self.initialization_errors.append("Pygame initialization failed")

        # Initialize the Map Controller with dynamic configuration
        try:
            map_config = self.config.get("map", {})
            self.map_controller = MapController(
                map_config.get("image_path", "assets/default_map.png"),
                map_data={
                    "width": map_config.get("width", 100),
                    "height": map_config.get("height", 100),
                    "buildings": self._get_default_buildings(map_config),
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize MapController: {e}")
            self.map_controller = None
            self.initialization_errors.append("MapController initialization failed")

        # Initialize ActionResolver
        self.action_resolver = None

        # Initialize other game systems
        self.characters = {}
        self.events = []
        self.game_statistics = {
            "actions_executed": 0,
            "actions_failed": 0,
            "characters_created": 0,
            "errors_recovered": 0,
        }

        # Initialize all game systems
        self.initialize_game_systems()

        # Log initialization status
        if self.initialization_errors:
            logger.warning(
                f"Initialization completed with {len(self.initialization_errors)} errors: {self.initialization_errors}"
            )
        else:
            logger.info("GameplayController initialized successfully")

    def _get_default_buildings(self, map_config: Dict) -> List[Dict]:
        """Get default buildings configuration."""
        default_buildings = [
            {"name": "Town Hall", "rect": pygame.Rect(100, 150, 50, 50)},
            {"name": "Market", "rect": pygame.Rect(200, 100, 40, 40)},
            {"name": "Tavern", "rect": pygame.Rect(300, 200, 45, 45)},
            {"name": "Blacksmith", "rect": pygame.Rect(150, 300, 35, 35)},
        ]

        # Allow custom buildings from config
        custom_buildings = map_config.get("buildings", [])
        if custom_buildings:
            # Convert custom building data to pygame.Rect if needed
            for building in custom_buildings:
                if "rect" not in building and "x" in building:
                    building["rect"] = pygame.Rect(
                        building["x"],
                        building["y"],
                        building.get("width", 40),
                        building.get("height", 40),
                    )
            return custom_buildings

        return default_buildings

    def initialize_game_systems(self):
        """Initialize all game systems and create characters with improved error handling."""
        system_init_errors = []

        try:
            # Import required systems with error handling
            required_modules = {
                "Character": "tiny_characters",
                "Location": "tiny_locations",
                "ActionSystem": "actions",
                "GameTimeManager": "tiny_time_manager",
                "GameCalendar": "tiny_time_manager",
                "ItemInventory": "tiny_items",
                "ItemObject": "tiny_items",
                "FoodItem": "tiny_items",
            }

            imported_modules = {}
            for class_name, module_name in required_modules.items():
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    imported_modules[class_name] = getattr(module, class_name)
                except Exception as e:
                    logger.error(
                        f"Failed to import {class_name} from {module_name}: {e}"
                    )
                    system_init_errors.append(f"Import {class_name} failed")

            # Initialize core systems with fallbacks
            if "ActionSystem" in imported_modules:
                try:
                    self.action_system = imported_modules["ActionSystem"]()
                    self.action_system.setup_actions()
                    self.action_resolver = ActionResolver(self.action_system)
                    logger.info("Action system initialized successfully")
                except Exception as e:
                    logger.error(f"ActionSystem initialization failed: {e}")
                    self.action_system = None
                    self.action_resolver = (
                        ActionResolver()
                    )  # Fallback without action system
                    system_init_errors.append("ActionSystem setup failed")
            else:
                self.action_system = None
                self.action_resolver = ActionResolver()

            # Initialize time management with fallback
            if (
                "GameCalendar" in imported_modules
                and "GameTimeManager" in imported_modules
            ):
                try:
                    calendar = imported_modules["GameCalendar"]()
                    self.gametime_manager = imported_modules["GameTimeManager"](
                        calendar
                    )
                    logger.info("Time management system initialized")
                except Exception as e:
                    logger.error(f"Time management initialization failed: {e}")
                    self.gametime_manager = None
                    system_init_errors.append("Time management failed")
            else:
                self.gametime_manager = None

            # Initialize animation system with fallback
            try:
                from tiny_animation_system import get_animation_system

                self.animation_system = get_animation_system()
                logger.info("Animation system initialized")
            except Exception as e:
                logger.warning(f"Animation system initialization failed: {e}")
                self.animation_system = None
                # Not adding to errors as animation is optional

            # Create characters with improved error handling
            character_creation_config = self.config.get("characters", {})
            self.characters = {}

            if self._can_create_characters(imported_modules):
                try:
                    sample_characters = self._create_sample_characters(
                        imported_modules, character_creation_config
                    )
                    self._register_characters(sample_characters)
                    self.game_statistics["characters_created"] = len(self.characters)
                    logger.info(
                        f"Successfully created {len(self.characters)} characters"
                    )
                except Exception as e:
                    logger.error(f"Character creation failed: {e}")
                    system_init_errors.append("Character creation failed")
                    # Create minimal fallback characters
                    self._create_fallback_characters()
            else:
                logger.warning("Cannot create characters due to missing dependencies")
                self._create_fallback_characters()

            # Initialize events system
            self.events = []

            # Log system initialization results
            if system_init_errors:
                logger.warning(
                    f"Game systems initialized with {len(system_init_errors)} errors: {system_init_errors}"
                )
                self.game_statistics["errors_recovered"] = len(system_init_errors)
            else:
                logger.info("All game systems initialized successfully")

        except Exception as e:
            logger.error(f"Critical error initializing game systems: {e}")
            logger.error(traceback.format_exc())
            # Ensure minimal fallback state
            self.characters = {}
            self.events = []
            if not hasattr(self, "action_resolver"):
                self.action_resolver = ActionResolver()

    def _can_create_characters(self, imported_modules: Dict) -> bool:
        """Check if we have the minimum required modules to create characters."""
        required_for_characters = ["Character", "Location", "ItemInventory", "FoodItem"]
        return all(module in imported_modules for module in required_for_characters)

    def _create_fallback_characters(self):
        """Create minimal fallback characters when full character creation fails."""
        try:
            logger.info("Creating fallback characters")
            # Create simple character data without complex dependencies
            fallback_character_data = {
                "name": "Fallback Villager",
                "position": pygame.math.Vector2(400, 300),
                "color": (100, 150, 200),
                "uuid": "fallback_001",
                "energy": 80,
                "health_status": 90,
                "path": [],
            }

            # Create a simple character dict (not a full Character object)
            fallback_char = type("FallbackCharacter", (), fallback_character_data)()

            # Add to systems if they exist
            if self.map_controller:
                self.map_controller.characters[fallback_char.uuid] = fallback_char
            self.characters[fallback_char.uuid] = fallback_char

            logger.info("Fallback character created successfully")
        except Exception as e:
            logger.error(f"Even fallback character creation failed: {e}")
            # Ensure characters dict exists even if empty
            self.characters = {}

    def _create_sample_characters(self, imported_modules: Dict, config: Dict):
        """Create sample characters with dynamic configuration."""
        Character = imported_modules["Character"]
        Location = imported_modules["Location"]
        ItemInventory = imported_modules["ItemInventory"]
        FoodItem = imported_modules["FoodItem"]

        characters = []

        # Get character configuration
        character_count = config.get("count", 4)
        use_custom_data = config.get("use_custom", False)

        if use_custom_data and "character_data" in config:
            character_data_list = config["character_data"]
        else:
            # Default character data
            character_data_list = [
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

        # Limit character creation to requested count
        character_data_list = character_data_list[:character_count]

        for char_data in character_data_list:
            try:
                character = self._create_single_character(char_data, imported_modules)
                if character:
                    characters.append(character)
                    logger.debug(f"Created character: {char_data['name']}")
            except Exception as e:
                logger.error(
                    f"Error creating character {char_data.get('name', 'Unknown')}: {e}"
                )
                continue

        return characters

    def _create_single_character(self, char_data: Dict, imported_modules: Dict):
        """Create a single character with error handling."""
        Character = imported_modules["Character"]
        Location = imported_modules["Location"]
        ItemInventory = imported_modules["ItemInventory"]
        FoodItem = imported_modules["FoodItem"]

        try:
            # Create basic inventory with error handling
            inventory_items = char_data.get(
                "inventory_items",
                [
                    {
                        "name": "Bread",
                        "type": "food",
                        "quantity": 2,
                        "weight": 1,
                        "value": 1,
                        "description": "Fresh bread",
                        "calories": 200,
                    },
                    {
                        "name": "Apple",
                        "type": "food",
                        "quantity": 1,
                        "weight": 0.5,
                        "value": 2,
                        "description": "Red apple",
                        "calories": 80,
                    },
                ],
            )

            food_items = []
            for item_data in inventory_items:
                try:
                    food_item = FoodItem(
                        item_data["name"],
                        item_data["type"],
                        item_data["quantity"],
                        item_data["weight"],
                        item_data["value"],
                        item_data["description"],
                        calories=item_data.get("calories", 100),
                    )
                    food_items.append(food_item)
                except Exception as e:
                    logger.warning(
                        f"Failed to create food item {item_data.get('name', 'Unknown')}: {e}"
                    )

            inventory = ItemInventory(food_items=food_items)

            # Create location for character with error handling
            location_name = f"{char_data['name']}'s Home"
            try:
                location = Location(
                    location_name,
                    random.randint(0, 100),
                    random.randint(0, 100),
                    1,
                    1,
                    self.action_system,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create location for {char_data['name']}: {e}"
                )
                location = None

            # Generate personality traits with defaults
            personality_config = char_data.get("personality", {})
            personality_traits = {
                "extraversion": personality_config.get(
                    "extraversion", random.randint(30, 80)
                ),
                "openness": personality_config.get("openness", random.randint(40, 90)),
                "conscientiousness": personality_config.get(
                    "conscientiousness", random.randint(50, 90)
                ),
                "agreeableness": personality_config.get(
                    "agreeableness", random.randint(40, 85)
                ),
                "neuroticism": personality_config.get(
                    "neuroticism", random.randint(10, 50)
                ),
            }

            # Create character with comprehensive error handling
            character_kwargs = {
                "name": char_data["name"],
                "age": char_data.get("age", random.randint(18, 65)),
                "pronouns": char_data.get("pronouns", "they/them"),
                "job": char_data.get("job", "Villager"),
                "health_status": char_data.get(
                    "health_status", random.randint(80, 100)
                ),
                "hunger_level": char_data.get("hunger_level", random.randint(20, 50)),
                "wealth_money": char_data.get("wealth_money", random.randint(50, 200)),
                "mental_health": char_data.get("mental_health", random.randint(70, 90)),
                "social_wellbeing": char_data.get(
                    "social_wellbeing", random.randint(60, 90)
                ),
                "job_performance": char_data.get(
                    "job_performance", random.randint(70, 95)
                ),
                "community": char_data.get("community", random.randint(50, 80)),
                "recent_event": char_data.get(
                    "recent_event", "Recently joined the village"
                ),
                "long_term_goal": char_data.get(
                    "long_term_goal", "Live a fulfilling life"
                ),
                "inventory": inventory,
                "personality_traits": personality_traits,
                "action_system": self.action_system,
                "gametime_manager": self.gametime_manager,
                "location": location,
                "graph_manager": self.graph_manager,
                "energy": char_data.get("energy", random.randint(60, 100)),
            }

            character = Character(**character_kwargs)
            return character

        except Exception as e:
            logger.error(
                f"Failed to create character {char_data.get('name', 'Unknown')}: {e}"
            )
            logger.error(traceback.format_exc())
            return None

    def _register_characters(self, characters: List):
        """Register characters with game systems."""
        for character in characters:
            try:
                # Add character as a node in the graph
                if self.graph_manager:
                    self.graph_manager.add_character_node(character)

                # Add character to map controller with random starting position
                screen_width = self.config.get("screen_width", 800)
                screen_height = self.config.get("screen_height", 600)

                start_x = random.randint(50, screen_width - 50)
                start_y = random.randint(50, screen_height - 50)
                character.position = pygame.math.Vector2(start_x, start_y)
                character.color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                )
                character.path = []

                # Register with map controller and local registry
                if self.map_controller:
                    self.map_controller.characters[character.uuid] = character
                self.characters[character.uuid] = character

            except Exception as e:
                logger.error(f"Error registering character {character.name}: {e}")
                continue

    def game_loop(self):
        """Main game loop with configurable frame rate and performance monitoring."""
        # TODO: Add performance profiling and optimization
        # TODO: Add frame rate adjustment based on performance
        # TODO: Add game state persistence and checkpointing
        # TODO: Add network synchronization for multiplayer
        # TODO: Add mod system integration
        # TODO: Add automated testing hooks
        # TODO: Add real-time configuration updates

        target_fps = self.config.get("target_fps", 60)

        while self.running:
            dt = self.clock.tick(target_fps) / 1000.0  # Frame time in seconds

            # TODO: Add performance monitoring
            # frame_start_time = time.time()

            self.handle_events()
            self.update_game_state(dt)
            self.render()

            # TODO: Add frame time analysis and optimization suggestions
            # frame_end_time = time.time()
            # self._analyze_frame_performance(frame_end_time - frame_start_time)

    def handle_events(self):
        """Handle pygame events and user input with improved interaction support."""
        # TODO: Implement more sophisticated input handling
        # TODO: Add keyboard shortcuts configuration
        # TODO: Add mouse gesture recognition
        # TODO: Add multi-touch support for mobile
        # TODO: Add gamepad/controller support
        # TODO: Add accessibility features (screen reader, high contrast mode)
        # TODO: Add customizable key bindings
        # TODO: Add input recording and playback for testing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # TODO: Add mouse interaction handling
                # TODO: Add right-click context menus
                pass
            elif event.type == pygame.MOUSEWHEEL:
                # TODO: Add zoom functionality
                # TODO: Add scroll-based UI navigation
                pass
            else:
                # Pass events to the Map Controller for handling
                if self.map_controller:
                    try:
                        self.map_controller.handle_event(event)
                    except Exception as e:
                        logger.warning(f"Error handling event in map controller: {e}")

    def _handle_keydown(self, event):
        """Handle keyboard input with configurable key bindings."""
        # TODO: Load key bindings from configuration
        key_bindings = self.config.get(
            "key_bindings",
            {
                "quit": [pygame.K_ESCAPE],
                "pause": [pygame.K_SPACE],
                "reset": [pygame.K_r],
                "save": [pygame.K_s],
                "load": [pygame.K_l],
                "help": [pygame.K_h, pygame.K_F1],
                "debug": [pygame.K_F3],
                "fullscreen": [pygame.K_F11],
            },
        )

        if event.key in key_bindings.get("quit", [pygame.K_ESCAPE]):
            self.running = False
        elif event.key in key_bindings.get("pause", [pygame.K_SPACE]):
            # Pause/unpause simulation
            self.paused = not getattr(self, "paused", False)
            logger.info(f"Game {'paused' if self.paused else 'unpaused'}")
        elif event.key in key_bindings.get("reset", [pygame.K_r]):
            # Reset/regenerate characters
            self._reset_characters()
        elif event.key in key_bindings.get("save", []):
            # TODO: Implement save game functionality
            logger.info("Save game functionality not yet implemented")
        elif event.key in key_bindings.get("load", []):
            # TODO: Implement load game functionality
            logger.info("Load game functionality not yet implemented")
        elif event.key in key_bindings.get("help", []):
            # TODO: Show help overlay
            logger.info("Help system not yet implemented")
        elif event.key in key_bindings.get("debug", []):
            # TODO: Toggle debug information display
            logger.info("Debug mode toggle not yet implemented")
        elif event.key in key_bindings.get("fullscreen", []):
            # TODO: Toggle fullscreen mode
            logger.info("Fullscreen toggle not yet implemented")

    def _reset_characters(self):
        """Reset and regenerate characters with improved error handling."""
        try:
            logger.info("Starting character reset...")

            # Clear existing characters safely
            if self.map_controller and hasattr(self.map_controller, "characters"):
                self.map_controller.characters.clear()
            self.characters.clear()

            # Get required modules for character creation
            try:
                required_modules = {
                    "Character": __import__(
                        "tiny_characters", fromlist=["Character"]
                    ).Character,
                    "Location": __import__(
                        "tiny_locations", fromlist=["Location"]
                    ).Location,
                    "ItemInventory": __import__(
                        "tiny_items", fromlist=["ItemInventory"]
                    ).ItemInventory,
                    "FoodItem": __import__(
                        "tiny_items", fromlist=["FoodItem"]
                    ).FoodItem,
                }
            except Exception as e:
                logger.error(
                    f"Failed to import required modules for character reset: {e}"
                )
                self._create_fallback_characters()
                return

            # Create new characters
            character_config = self.config.get("characters", {})
            if self._can_create_characters(required_modules):
                try:
                    sample_characters = self._create_sample_characters(
                        required_modules, character_config
                    )
                    self._register_characters(sample_characters)

                    logger.info(
                        f"Reset complete - {len(self.characters)} new characters created"
                    )
                    self.game_statistics["characters_created"] += len(self.characters)

                except Exception as e:
                    logger.error(f"Error during character creation in reset: {e}")
                    self._create_fallback_characters()
            else:
                logger.warning(
                    "Cannot create full characters during reset, using fallback"
                )
                self._create_fallback_characters()

        except Exception as e:
            logger.error(f"Critical error resetting characters: {e}")
            logger.error(traceback.format_exc())
            # Ensure we have at least empty character dict
            self.characters = {}
            if self.map_controller and hasattr(self.map_controller, "characters"):
                self.map_controller.characters = {}

    def update_game_state(self, dt):
        """Update all game systems with delta time and improved error handling."""
        # Check if game is paused
        if getattr(self, "paused", False):
            return  # Skip all updates when paused

        update_errors = []

        # Update the map controller (handles character movement and pathfinding)
        if self.map_controller:
            try:
                self.map_controller.update(dt)
            except Exception as e:
                logger.error(f"Error updating map controller: {e}")
                update_errors.append("Map controller update failed")

        # Update character AI and decision making
        for character_id, character in list(self.characters.items()):
            try:
                success = self._update_character(character, dt)
                if not success:
                    update_errors.append(
                        f"Character {character.name if hasattr(character, 'name') else character_id} update failed"
                    )
            except Exception as e:
                logger.error(f"Error updating character {character_id}: {e}")
                update_errors.append(f"Character {character_id} update failed")
                continue

        # Update time manager
        if hasattr(self, "gametime_manager") and self.gametime_manager:
            try:
                # Process any scheduled behaviors
                behaviors = self.gametime_manager.get_scheduled_behaviors()
                for behavior in behaviors:
                    try:
                        behavior.check_calendar()
                    except Exception as e:
                        logger.warning(f"Error processing scheduled behavior: {e}")
            except Exception as e:
                logger.error(f"Error updating time manager: {e}")
                update_errors.append("Time manager update failed")

        # Update animation system
        if hasattr(self, "animation_system") and self.animation_system:
            try:
                self.animation_system.update(dt)
            except Exception as e:
                logger.warning(f"Error updating animation system: {e}")
                # Animation errors are not critical

        # Process any pending events
        if hasattr(self, "events") and self.events:
            try:
                self._process_pending_events()
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                update_errors.append("Event processing failed")

        # Log update errors if any occurred
        if update_errors:
            logger.warning(
                f"Game state update completed with {len(update_errors)} errors"
            )
            self.game_statistics["errors_recovered"] += len(update_errors)

    def _update_character(self, character, dt) -> bool:
        """Update a single character with comprehensive error handling."""
        try:
            # Check if character has required methods
            if not hasattr(character, "name"):
                logger.warning(f"Character missing name attribute")
                return False

            # Update character's memory and decision making
            memory_success = self._update_character_memory(character)
            goals_success = self._update_character_goals(character)
            actions_success = self._execute_character_actions(character)

            # Character update is successful if at least one component works
            return memory_success or goals_success or actions_success

        except Exception as e:
            logger.error(
                f"Critical error updating character {getattr(character, 'name', 'Unknown')}: {e}"
            )
            return False

    def _update_character_memory(self, character) -> bool:
        """Update character memory with error handling."""
        try:
            if hasattr(character, "recall_recent_memories"):
                character.recall_recent_memories()
                return True
        except Exception as e:
            logger.warning(f"Error updating memory for {character.name}: {e}")
        return False

    def _update_character_goals(self, character) -> bool:
        """Update character goals with error handling."""
        try:
            if hasattr(character, "evaluate_goals"):
                goals = character.evaluate_goals()
                return goals is not None
        except Exception as e:
            logger.warning(f"Error evaluating goals for {character.name}: {e}")
        return False

    def _execute_character_actions(self, character) -> bool:
        """Execute character actions with error handling."""
        try:
            if not self.strategy_manager:
                return False

            # Use strategy manager to plan actions
            actions = self.strategy_manager.get_daily_actions(character)

            if actions:
                # Execute planned actions based on current state
                success = self._apply_character_actions(character, actions)
                if success:
                    self.game_statistics["actions_executed"] += len(actions)
                else:
                    self.game_statistics["actions_failed"] += 1
                return success
            return True  # No actions is not necessarily a failure

        except Exception as e:
            logger.warning(f"Error executing actions for {character.name}: {e}")
            self.game_statistics["actions_failed"] += 1
            return False

    def _apply_character_actions(self, character, actions) -> bool:
        """Apply a list of actions to a character."""
        try:
            for action_data in actions:
                success = self._execute_single_action(character, action_data)
                if not success:
                    logger.warning(f"Action {action_data} failed for {character.name}")
                    # Continue with other actions even if one fails
            return True
        except Exception as e:
            logger.error(f"Error applying actions to {character.name}: {e}")
            return False

    def _execute_single_action(self, character, action_data) -> bool:
        """Execute a single action with proper resolution and state updates."""
        try:
            # Resolve action to executable format
            action = self.action_resolver.resolve_action(action_data, character)

            if not action:
                logger.warning(f"Could not resolve action: {action_data}")
                return False

            # Execute the action
            if hasattr(action, "execute"):
                try:
                    result = action.execute(target=character, initiator=character)
                    if result:
                        # Update character state after successful action
                        self._update_character_state_after_action(character, action)
                        return True
                    else:
                        logger.warning(f"Action {action.name} execution returned False")
                        return False
                except Exception as e:
                    logger.error(f"Error executing action {action.name}: {e}")
                    # Try fallback action
                    return self._execute_fallback_action(character)
            else:
                logger.warning(f"Action {action} has no execute method")
                return False

        except Exception as e:
            logger.error(f"Critical error executing single action: {e}")
            return self._execute_fallback_action(character)

    def _execute_fallback_action(self, character) -> bool:
        """Execute a safe fallback action when normal actions fail."""
        try:
            fallback_action = self.action_resolver.get_fallback_action(character)
            if fallback_action and hasattr(fallback_action, "execute"):
                result = fallback_action.execute(target=character, initiator=character)
                if result:
                    logger.info(f"Fallback action executed for {character.name}")
                    return True

            # If even fallback fails, just update character energy minimally
            if hasattr(character, "energy"):
                character.energy = max(
                    0, character.energy - 1
                )  # Minimal energy cost for existing
            return True

        except Exception as e:
            logger.error(f"Even fallback action failed for {character.name}: {e}")
            return False

    def _update_character_state_after_action(self, character, action):
        """Update character state and related systems after action execution."""
        try:
            # Update graph manager if available
            if self.graph_manager and hasattr(
                self.graph_manager, "update_character_state"
            ):
                try:
                    self.graph_manager.update_character_state(character)
                except Exception as e:
                    logger.warning(
                        f"Error updating graph manager for {character.name}: {e}"
                    )

            # Update any other systems that need to know about state changes
            # TODO: Add memory system updates
            # TODO: Add relationship system updates
            # TODO: Add event generation based on action outcomes
            # TODO: Add achievement/milestone tracking
            # TODO: Add social interaction consequences
            # TODO: Add economic system updates (market prices, resource availability)
            # TODO: Add weather/seasonal effects on character state
            # TODO: Add skill progression updates
            # TODO: Add reputation system updates

        except Exception as e:
            logger.warning(f"Error updating character state after action: {e}")

    def _process_pending_events(self):
        """Process any pending events with error handling."""
        events_to_remove = []

        for i, event in enumerate(self.events):
            try:
                # Process event logic here
                # TODO: Implement proper event processing system
                # TODO: Add event types (social, economic, environmental, etc.)
                # TODO: Add event consequences and ripple effects
                # TODO: Add event prioritization and scheduling
                # TODO: Add cross-character event interactions
                # TODO: Add event persistence and memory
                # TODO: Add event-driven story generation
                # For now, just mark events as processed
                events_to_remove.append(i)
            except Exception as e:
                logger.error(f"Error processing event {event}: {e}")
                events_to_remove.append(i)  # Remove problematic events

        # Remove processed/failed events in reverse order to maintain indices
        for i in reversed(events_to_remove):
            try:
                del self.events[i]
            except IndexError:
                pass  # Event already removed

    def render(self):
        """Render all game elements with configurable quality and effects."""
        # TODO: Add render quality settings (low, medium, high, ultra)
        # TODO: Add dynamic resolution scaling based on performance
        # TODO: Add anti-aliasing options
        # TODO: Add post-processing effects (bloom, shadows, etc.)
        # TODO: Add level-of-detail (LOD) system for distant objects
        # TODO: Add particle effects system
        # TODO: Add lighting and shadow system
        # TODO: Add weather and atmospheric effects
        # TODO: Add screenshot and video recording functionality
        # TODO: Add VR/AR rendering support

        # Get render configuration
        render_config = self.config.get("render", {})
        background_color = render_config.get("background_color", (0, 0, 0))
        enable_vsync = render_config.get("vsync", True)

        # Clear the screen with configurable background
        self.screen.fill(background_color)

        # Render the map and game world
        if self.map_controller:
            try:
                self.map_controller.render(self.screen)
            except Exception as e:
                logger.error(f"Error rendering map: {e}")
                # TODO: Add fallback rendering for when map fails

        # Render UI elements
        self._render_ui()

        # TODO: Add render effect layers (lighting, particles, post-processing)

        # Flip the display to show the updated frame
        if enable_vsync:
            pygame.display.flip()
        else:
            pygame.display.update()

    def _render_ui(self):
        """Render user interface elements with improved layout and information."""
        try:
            # TODO: Implement modular UI system with panels
            # TODO: Add character relationship visualization
            # TODO: Add village statistics dashboard
            # TODO: Add interactive building information panels
            # TODO: Add mini-map or overview mode
            # TODO: Add save/load game functionality UI
            # TODO: Add settings and configuration panels
            # TODO: Add help and tutorial overlays
            # TODO: Add drag-and-drop interaction hints
            # TODO: Add notification system for important events

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
        """Main entry point to start the game loop."""
        try:
            logger.info("Starting Tiny Village gameplay...")
            if self.initialization_errors:
                logger.warning(
                    f"Starting with {len(self.initialization_errors)} initialization errors"
                )
            self.game_loop()
        except Exception as e:
            logger.error(f"Critical error in main game loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Gameplay controller shutting down")

    def update(self, game_state=None):
        """
        Legacy update method for compatibility with external systems.
        TODO: Integrate this with the main update_game_state method.
        """
        try:
            # Check for new events if event handler exists
            events = []
            if self.event_handler:
                try:
                    events = self.event_handler.check_events()
                except Exception as e:
                    logger.warning(f"Error checking events: {e}")

            # Update strategy based on events if strategy manager exists
            decisions = []
            if self.strategy_manager and events:
                try:
                    decisions = self.strategy_manager.update_strategy(events)
                except Exception as e:
                    logger.warning(f"Error updating strategy: {e}")

            # Apply decisions to game state
            for decision in decisions:
                try:
                    self.apply_decision(decision, game_state)
                except Exception as e:
                    logger.error(f"Error applying decision: {e}")

        except Exception as e:
            logger.error(f"Error in legacy update method: {e}")

    def apply_decision(self, decision, game_state=None):
        """
        Improved apply_decision method that properly handles action resolution.

        Args:
            decision: Can be a single action, list of actions, or decision object
            game_state: Optional game state (for legacy compatibility)
        """
        try:
            # Handle different decision formats
            actions_to_execute = []

            if isinstance(decision, list):
                actions_to_execute = decision
            elif isinstance(decision, dict) and "actions" in decision:
                actions_to_execute = decision["actions"]
                target_character = decision.get("character")
            elif hasattr(decision, "actions"):
                actions_to_execute = decision.actions
                target_character = getattr(decision, "character", None)
            else:
                # Single action
                actions_to_execute = [decision]
                target_character = None

            # Find target character if not specified
            if not target_character and actions_to_execute:
                # Try to determine character from first action
                first_action = actions_to_execute[0]
                if isinstance(first_action, dict) and "character_id" in first_action:
                    target_character = self.characters.get(first_action["character_id"])
                elif hasattr(first_action, "initiator"):
                    target_character = first_action.initiator

            # Execute actions
            successful_actions = 0
            for action in actions_to_execute:
                try:
                    success = self._execute_decision_action(
                        action, target_character, game_state
                    )
                    if success:
                        successful_actions += 1
                        logger.debug(f"Successfully executed action: {action}")
                    else:
                        logger.warning(f"Failed to execute action: {action}")
                except Exception as e:
                    logger.error(f"Error executing decision action {action}: {e}")
                    continue

            # Log decision execution results
            total_actions = len(actions_to_execute)
            if successful_actions == total_actions:
                logger.info(
                    f"Decision fully executed: {successful_actions}/{total_actions} actions successful"
                )
            elif successful_actions > 0:
                logger.warning(
                    f"Decision partially executed: {successful_actions}/{total_actions} actions successful"
                )
            else:
                logger.error(
                    f"Decision execution failed: 0/{total_actions} actions successful"
                )

        except Exception as e:
            logger.error(f"Critical error applying decision: {e}")
            logger.error(traceback.format_exc())

    def _execute_decision_action(
        self, action, target_character=None, game_state=None
    ) -> bool:
        """
        Execute a single action from a decision with proper error handling.

        Args:
            action: Action data (dict, object, or string)
            target_character: Character to execute action on
            game_state: Optional game state for legacy compatibility

        Returns:
            bool: True if action executed successfully
        """
        try:
            # Use action resolver to convert action to executable format
            resolved_action = self.action_resolver.resolve_action(
                action, target_character
            )

            if not resolved_action:
                logger.warning(f"Could not resolve action: {action}")
                return False

            # Execute the resolved action
            if hasattr(resolved_action, "execute"):
                try:
                    # Try different execution signatures
                    if target_character:
                        result = resolved_action.execute(
                            target=target_character, initiator=target_character
                        )
                    elif game_state:
                        result = resolved_action.execute(game_state)
                    else:
                        result = resolved_action.execute()

                    if result:
                        # Update statistics
                        self.game_statistics["actions_executed"] += 1

                        # Update character state if needed
                        if target_character:
                            self._update_character_state_after_action(
                                target_character, resolved_action
                            )

                        return True
                    else:
                        logger.warning(
                            f"Action {resolved_action.name if hasattr(resolved_action, 'name') else resolved_action} execution returned False"
                        )
                        self.game_statistics["actions_failed"] += 1
                        return False

                except Exception as e:
                    logger.error(f"Error executing resolved action: {e}")
                    self.game_statistics["actions_failed"] += 1
                    return False
            else:
                logger.error(
                    f"Resolved action has no execute method: {resolved_action}"
                )
                return False

        except Exception as e:
            logger.error(f"Critical error executing decision action: {e}")
            self.game_statistics["actions_failed"] += 1
            return False

    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report of all game systems.

        TODO: Add detailed memory usage monitoring
        TODO: Add performance metrics collection
        TODO: Add predictive failure detection
        TODO: Add automated system recovery recommendations

        Returns:
            Dict containing health status of all systems
        """
        health_report = {
            "timestamp": pygame.time.get_ticks(),
            "initialization_errors": self.initialization_errors,
            "game_statistics": self.game_statistics,
            "systems": {},
            "overall_health": "unknown",
        }

        # Check core systems
        systems_to_check = [
            ("strategy_manager", self.strategy_manager),
            ("graph_manager", self.graph_manager),
            ("event_handler", self.event_handler),
            ("map_controller", self.map_controller),
            ("action_system", getattr(self, "action_system", None)),
            ("gametime_manager", getattr(self, "gametime_manager", None)),
            ("animation_system", getattr(self, "animation_system", None)),
        ]

        healthy_systems = 0
        for system_name, system_obj in systems_to_check:
            if system_obj is not None:
                health_report["systems"][system_name] = "healthy"
                healthy_systems += 1
            else:
                health_report["systems"][system_name] = "failed"

        # Calculate overall health
        total_systems = len(systems_to_check)
        health_percentage = (healthy_systems / total_systems) * 100

        if health_percentage >= 90:
            health_report["overall_health"] = "excellent"
        elif health_percentage >= 70:
            health_report["overall_health"] = "good"
        elif health_percentage >= 50:
            health_report["overall_health"] = "degraded"
        else:
            health_report["overall_health"] = "critical"

        health_report["health_percentage"] = health_percentage
        health_report["character_count"] = len(self.characters)

        return health_report

    def attempt_system_recovery(self) -> bool:
        """
        Attempt to recover failed systems and restore functionality.

        TODO: Add intelligent recovery strategies for different failure types
        TODO: Add system dependency analysis and recovery ordering
        TODO: Add partial recovery support (recover what can be recovered)
        TODO: Add user notification system for recovery attempts

        Returns:
            bool: True if recovery was successful, False otherwise
        """
        try:
            logger.info("Attempting system recovery...")
            recovery_successful = True

            # Attempt to recover strategy manager
            if not self.strategy_manager:
                try:
                    self.strategy_manager = StrategyManager()
                    logger.info("Strategy manager recovery successful")
                except Exception as e:
                    logger.error(f"Strategy manager recovery failed: {e}")
                    recovery_successful = False

            # Attempt to recover graph manager
            if not self.graph_manager:
                try:
                    from tiny_graph_manager import GraphManager as ActualGraphManager

                    self.graph_manager = ActualGraphManager()
                    logger.info("Graph manager recovery successful")
                except Exception as e:
                    logger.error(f"Graph manager recovery failed: {e}")
                    recovery_successful = False

            # Attempt to recover event handler
            if not self.event_handler and self.graph_manager:
                try:
                    self.event_handler = EventHandler(self.graph_manager)
                    logger.info("Event handler recovery successful")
                except Exception as e:
                    logger.error(f"Event handler recovery failed: {e}")
                    recovery_successful = False

            # Attempt to recover action system
            if not getattr(self, "action_system", None):
                try:
                    from actions import ActionSystem

                    self.action_system = ActionSystem()
                    self.action_system.setup_actions()
                    self.action_resolver = ActionResolver(self.action_system)
                    logger.info("Action system recovery successful")
                except Exception as e:
                    logger.error(f"Action system recovery failed: {e}")
                    recovery_successful = False

            # Update recovery statistics
            if recovery_successful:
                self.game_statistics["errors_recovered"] += 1
                logger.info("System recovery completed successfully")
            else:
                logger.warning("System recovery partially failed")

            return recovery_successful

        except Exception as e:
            logger.error(f"Critical error during system recovery: {e}")
            return False

    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current configuration for saving or sharing.

        TODO: Add configuration validation and schema checking
        TODO: Add configuration versioning and migration support
        TODO: Add configuration encryption for sensitive data
        TODO: Add configuration templates and presets

        Returns:
            Dict containing exportable configuration
        """
        try:
            export_config = {
                "version": "1.0",  # TODO: Add proper versioning system
                "timestamp": pygame.time.get_ticks(),
                "config": self.config.copy(),
                "game_statistics": self.game_statistics.copy(),
                "character_count": len(self.characters),
                "system_health": self.get_system_health_report(),
            }

            # Remove sensitive or non-serializable data
            # TODO: Add proper data sanitization

            return export_config

        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return {"error": str(e)}

    def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """
        Import configuration from external source.

        TODO: Add configuration validation and safety checks
        TODO: Add configuration migration for version compatibility
        TODO: Add backup creation before importing new configuration
        TODO: Add selective configuration import (only specific sections)

        Args:
            config_data: Configuration data to import

        Returns:
            bool: True if import was successful
        """
        try:
            # TODO: Add configuration validation
            if "config" in config_data:
                self.config.update(config_data["config"])
                logger.info("Configuration imported successfully")
                return True
            else:
                logger.error("Invalid configuration data format")
                return False

        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

    # TODO: Add save/load game state functionality
    def save_game_state(self, save_path: str) -> bool:
        """
        Save complete game state to file.

        TODO: Implement full game state serialization
        TODO: Add compression and optimization for large saves
        TODO: Add incremental save functionality
        TODO: Add save file integrity checking
        TODO: Add multiple save slot management

        Args:
            save_path: Path to save the game state

        Returns:
            bool: True if save was successful
        """
        logger.warning("Save game functionality not yet implemented")
        return False

    def load_game_state(self, save_path: str) -> bool:
        """
        Load complete game state from file.

        TODO: Implement full game state deserialization
        TODO: Add save file validation and error recovery
        TODO: Add partial loading for corrupted saves
        TODO: Add save file migration for version compatibility
        TODO: Add loading progress feedback

        Args:
            save_path: Path to load the game state from

        Returns:
            bool: True if load was successful
        """
        logger.warning("Load game functionality not yet implemented")
        return False


if __name__ == "__main__":
    game_controller = GameplayController()
    game_controller.run()
    pygame.quit()
