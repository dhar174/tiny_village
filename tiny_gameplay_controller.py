import pygame
import random
import logging
import traceback
import json
import os
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
    Provides robust action resolution with validation and fallback mechanisms.
    """

    def __init__(self, action_system=None):
        self.action_system = action_system
        self.action_cache = {}  # Cache for performance optimization
        self.execution_history = []  # Track action execution for analytics
        self.fallback_actions = {
            "default_rest": {"name": "Rest", "energy_cost": -10, "satisfaction": 5},
            "default_idle": {"name": "Idle", "energy_cost": 0, "satisfaction": 1},
            "emergency_recover": {
                "name": "Emergency Recovery",
                "energy_cost": -20,
                "satisfaction": 10,
            },
        }

    def validate_action_preconditions(self, action, character) -> bool:
        """Validate that action preconditions are met."""
        try:
            if hasattr(action, "preconditions") and action.preconditions:
                for condition, value in action.preconditions.items():
                    if hasattr(character, condition):
                        if getattr(character, condition) < value:
                            return False
            return True
        except Exception as e:
            logger.warning(f"Error validating preconditions: {e}")
            return False

    def predict_action_effects(self, action, character) -> Dict[str, Any]:
        """Predict the effects of an action before execution."""
        predicted_effects = {
            "energy_change": 0,
            "satisfaction_change": 0,
            "health_change": 0,
            "success_probability": 0.8,  # Default success rate
        }

        try:
            if hasattr(action, "effects"):
                for effect in action.effects:
                    if "attribute" in effect and "change_value" in effect:
                        attr = effect["attribute"]
                        change = effect["change_value"]
                        if attr == "energy":
                            predicted_effects["energy_change"] = change
                        elif attr == "current_satisfaction":
                            predicted_effects["satisfaction_change"] = change
                        elif attr == "health_status":
                            predicted_effects["health_change"] = change

            # Adjust success probability based on character state
            if hasattr(character, "energy") and character.energy < 20:
                predicted_effects[
                    "success_probability"
                ] *= 0.7  # Lower success when tired

        except Exception as e:
            logger.warning(f"Error predicting action effects: {e}")

        return predicted_effects

    def resolve_action(
        self, action_data: Union[Dict, Any], character=None
    ) -> Optional[Any]:
        """Convert action data to executable action object with validation and caching."""
        try:
            # Create cache key for performance
            cache_key = self._generate_cache_key(action_data, character)
            if cache_key in self.action_cache:
                logger.debug(f"Using cached action for {cache_key}")
                return self.action_cache[cache_key]

            # If it's already an Action object with execute method
            if hasattr(action_data, "execute"):
                if self.validate_action_preconditions(action_data, character):
                    self.action_cache[cache_key] = action_data
                    return action_data
                else:
                    logger.warning(f"Action preconditions not met: {action_data}")
                    return self.get_fallback_action(character)

            # If it's a dictionary, try to convert to Action
            if isinstance(action_data, dict):
                resolved_action = self._dict_to_action(action_data, character)
                if resolved_action:
                    self.action_cache[cache_key] = resolved_action
                return resolved_action

            # If it's a string (action name), try to resolve
            if isinstance(action_data, str):
                resolved_action = self._resolve_by_name(action_data, character)
                if resolved_action:
                    self.action_cache[cache_key] = resolved_action
                return resolved_action

            logger.warning(f"Unknown action format: {type(action_data)}")
            return self.get_fallback_action(character)

        except Exception as e:
            logger.error(f"Error resolving action {action_data}: {e}")
            return self.get_fallback_action(character)

    def _generate_cache_key(self, action_data, character):
        """Generate a cache key for action data."""
        try:
            char_id = getattr(character, "uuid", "unknown") if character else "none"
            if isinstance(action_data, dict):
                return f"dict_{action_data.get('name', 'unknown')}_{char_id}"
            elif isinstance(action_data, str):
                return f"str_{action_data}_{char_id}"
            elif hasattr(action_data, "name"):
                return f"obj_{action_data.name}_{char_id}"
            else:
                return f"unknown_{type(action_data).__name__}_{char_id}"
        except:
            return f"fallback_{id(action_data)}_{id(character)}"

    def track_action_execution(self, action, character, success: bool):
        """Track action execution for analytics and improvement."""
        try:
            execution_record = {
                "timestamp": pygame.time.get_ticks(),
                "action_name": getattr(action, "name", str(action)),
                "character_id": (
                    getattr(character, "uuid", "unknown") if character else None
                ),
                "character_name": (
                    getattr(character, "name", "unknown") if character else None
                ),
                "success": success,
                "character_energy_before": (
                    getattr(character, "energy", None) if character else None
                ),
            }

            self.execution_history.append(execution_record)

            # Keep history limited to last 1000 actions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

        except Exception as e:
            logger.warning(f"Error tracking action execution: {e}")

    def get_action_analytics(self) -> Dict[str, Any]:
        """Get analytics about action execution performance."""
        try:
            if not self.execution_history:
                return {"total_actions": 0, "success_rate": 0.0}

            total_actions = len(self.execution_history)
            successful_actions = sum(
                1 for record in self.execution_history if record["success"]
            )
            success_rate = (
                successful_actions / total_actions if total_actions > 0 else 0.0
            )

            # Action type analysis
            action_stats = {}
            for record in self.execution_history:
                action_name = record["action_name"]
                if action_name not in action_stats:
                    action_stats[action_name] = {"count": 0, "successes": 0}
                action_stats[action_name]["count"] += 1
                if record["success"]:
                    action_stats[action_name]["successes"] += 1

            return {
                "total_actions": total_actions,
                "success_rate": success_rate,
                "action_breakdown": action_stats,
                "cache_size": len(self.action_cache),
            }

        except Exception as e:
            logger.error(f"Error generating action analytics: {e}")
            return {"error": str(e)}

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


class SystemRecoveryManager:
    """Manages system recovery and fallback strategies."""

    def __init__(self, gameplay_controller):
        self.gameplay_controller = gameplay_controller
        self.recovery_strategies = {}
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.setup_recovery_strategies()

    def setup_recovery_strategies(self):
        """Define recovery strategies for different system failures."""
        self.recovery_strategies = {
            "strategy_manager": self._recover_strategy_manager,
            "graph_manager": self._recover_graph_manager,
            "event_handler": self._recover_event_handler,
            "action_system": self._recover_action_system,
            "map_controller": self._recover_map_controller,
            "character_system": self._recover_character_system,
        }

    def attempt_recovery(self, system_name: str) -> bool:
        """Attempt to recover a failed system with intelligent retry logic."""
        try:
            if system_name not in self.recovery_strategies:
                logger.error(f"No recovery strategy for system: {system_name}")
                return False

            # Check recovery attempt limits
            attempts = self.recovery_attempts.get(system_name, 0)
            if attempts >= self.max_recovery_attempts:
                logger.error(f"Max recovery attempts reached for {system_name}")
                return False

            logger.info(
                f"Attempting recovery for {system_name} (attempt {attempts + 1})"
            )

            # Increment attempt counter
            self.recovery_attempts[system_name] = attempts + 1

            # Execute recovery strategy
            success = self.recovery_strategies[system_name]()

            if success:
                logger.info(f"Successfully recovered {system_name}")
                # Reset attempt counter on success
                self.recovery_attempts[system_name] = 0
                return True
            else:
                logger.warning(f"Recovery attempt failed for {system_name}")
                return False

        except Exception as e:
            logger.error(f"Error during recovery attempt for {system_name}: {e}")
            return False

    def _recover_strategy_manager(self) -> bool:
        """Recover the strategy manager system."""
        try:
            if not self.gameplay_controller.strategy_manager:
                self.gameplay_controller.strategy_manager = StrategyManager()
                return True
            return True
        except Exception as e:
            logger.error(f"Strategy manager recovery failed: {e}")
            return False

    def _recover_graph_manager(self) -> bool:
        """Recover the graph manager system."""
        try:
            if not self.gameplay_controller.graph_manager:
                from tiny_graph_manager import GraphManager as ActualGraphManager

                self.gameplay_controller.graph_manager = ActualGraphManager()
                return True
            return True
        except Exception as e:
            logger.error(f"Graph manager recovery failed: {e}")
            return False

    def _recover_event_handler(self) -> bool:
        """Recover the event handler system."""
        try:
            if (
                not self.gameplay_controller.event_handler
                and self.gameplay_controller.graph_manager
            ):
                self.gameplay_controller.event_handler = EventHandler(
                    self.gameplay_controller.graph_manager
                )
                return True
            return True
        except Exception as e:
            logger.error(f"Event handler recovery failed: {e}")
            return False

    def _recover_action_system(self) -> bool:
        """Recover the action system."""
        try:
            if not getattr(self.gameplay_controller, "action_system", None):
                from actions import ActionSystem

                self.gameplay_controller.action_system = ActionSystem()
                self.gameplay_controller.action_system.setup_actions()
                self.gameplay_controller.action_resolver = ActionResolver(
                    self.gameplay_controller.action_system
                )
                return True
            return True
        except Exception as e:
            logger.error(f"Action system recovery failed: {e}")
            # Fallback to basic action resolver
            self.gameplay_controller.action_resolver = ActionResolver()
            return False

    def _recover_map_controller(self) -> bool:
        """Recover the map controller system."""
        try:
            if not self.gameplay_controller.map_controller:
                map_config = self.gameplay_controller.config.get("map", {})
                self.gameplay_controller.map_controller = MapController(
                    map_config.get("image_path", "assets/default_map.png"),
                    map_data={
                        "width": map_config.get("width", 100),
                        "height": map_config.get("height", 100),
                        "buildings": self.gameplay_controller._get_default_buildings(
                            map_config
                        ),
                    },
                )
                return True
            return True
        except Exception as e:
            logger.error(f"Map controller recovery failed: {e}")
            return False

    def _recover_character_system(self) -> bool:
        """Recover the character system."""
        try:
            if not self.gameplay_controller.characters:
                self.gameplay_controller._create_fallback_characters()
                return len(self.gameplay_controller.characters) > 0
            return True
        except Exception as e:
            logger.error(f"Character system recovery failed: {e}")
            return False

    def get_system_status(self) -> Dict[str, str]:
        """Get current status of all systems."""
        status = {}
        gc = self.gameplay_controller

        status["strategy_manager"] = "healthy" if gc.strategy_manager else "failed"
        status["graph_manager"] = "healthy" if gc.graph_manager else "failed"
        status["event_handler"] = "healthy" if gc.event_handler else "failed"
        status["action_system"] = (
            "healthy" if getattr(gc, "action_system", None) else "failed"
        )
        status["map_controller"] = "healthy" if gc.map_controller else "failed"
        status["characters"] = "healthy" if gc.characters else "failed"

        return status


class GameplayController:
    def __init__(
        self, graph_manager: GraphManager = None, config: Dict[str, Any] = None
    ):
        """Initialize the gameplay controller with improved error handling and configuration."""
        self.config = config or {}
        self.initialization_errors = []
        self.running = True
        self.paused = False

        # Initialize recovery manager
        self.recovery_manager = SystemRecoveryManager(self)

        # Initialize core systems with error handling and recovery
        try:
            self.strategy_manager = StrategyManager()
        except Exception as e:
            logger.error(f"Failed to initialize StrategyManager: {e}")
            self.strategy_manager = None
            self.initialization_errors.append("StrategyManager initialization failed")
            # Attempt immediate recovery
            self.recovery_manager.attempt_recovery("strategy_manager")

        # Initialize graph manager if not provided
        if graph_manager is None:
            try:
                from tiny_graph_manager import GraphManager as ActualGraphManager

                self.graph_manager = ActualGraphManager()
            except Exception as e:
                logger.error(f"Failed to initialize GraphManager: {e}")
                self.graph_manager = None
                self.initialization_errors.append("GraphManager initialization failed")
                self.recovery_manager.attempt_recovery("graph_manager")
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
            self.recovery_manager.attempt_recovery("event_handler")

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

        # Setup user-driven configuration
        self.setup_user_driven_configuration()

        # Initialize feature systems
        self.implement_achievement_system()
        self.implement_weather_system()
        self.implement_social_network_system()
        self.implement_quest_system()

        # Log initialization status
        if self.initialization_errors:
            logger.warning(
                f"Initialization completed with {len(self.initialization_errors)} errors: {self.initialization_errors}"
            )
        else:
            logger.info("GameplayController initialized successfully")

    def _get_default_buildings(self, map_config: Dict) -> List[Dict]:
        """Get buildings configuration with dynamic loading support."""

        # Try to load buildings from external file first
        buildings_file = map_config.get("buildings_file")
        if buildings_file:
            try:
                buildings = self._load_buildings_from_file(buildings_file)
                if buildings:
                    logger.info(
                        f"Loaded {len(buildings)} buildings from {buildings_file}"
                    )
                    return buildings
            except Exception as e:
                logger.warning(f"Failed to load buildings from {buildings_file}: {e}")

        # Fallback to default buildings
        default_buildings = [
            {
                "name": "Town Hall",
                "rect": pygame.Rect(100, 150, 50, 50),
                "type": "civic",
            },
            {
                "name": "Market",
                "rect": pygame.Rect(200, 100, 40, 40),
                "type": "commercial",
            },
            {"name": "Tavern", "rect": pygame.Rect(300, 200, 45, 45), "type": "social"},
            {
                "name": "Blacksmith",
                "rect": pygame.Rect(150, 300, 35, 35),
                "type": "crafting",
            },
            {
                "name": "Farm",
                "rect": pygame.Rect(400, 350, 60, 40),
                "type": "agricultural",
            },
            {
                "name": "School",
                "rect": pygame.Rect(500, 120, 45, 35),
                "type": "educational",
            },
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
                # Add default type if missing
                if "type" not in building:
                    building["type"] = "generic"
            return custom_buildings

        return default_buildings

    def _load_buildings_from_file(self, filepath: str) -> List[Dict]:
        """Load building configuration from external file."""
        try:
            import json
            import os

            if not os.path.exists(filepath):
                logger.warning(f"Buildings file not found: {filepath}")
                return []

            with open(filepath, "r") as f:
                data = json.load(f)

            buildings = []
            for building_data in data.get("buildings", []):
                # Convert to required format
                building = {
                    "name": building_data.get("name", "Unknown Building"),
                    "type": building_data.get("type", "generic"),
                    "rect": pygame.Rect(
                        building_data.get("x", 0),
                        building_data.get("y", 0),
                        building_data.get("width", 40),
                        building_data.get("height", 40),
                    ),
                }

                # Add any additional properties
                for key, value in building_data.items():
                    if key not in ["x", "y", "width", "height"]:
                        building[key] = value

                buildings.append(building)

            return buildings

        except Exception as e:
            logger.error(f"Error loading buildings from file {filepath}: {e}")
            return []

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
        """Create sample characters with dynamic configuration and external file support."""
        Character = imported_modules["Character"]
        Location = imported_modules["Location"]
        ItemInventory = imported_modules["ItemInventory"]
        FoodItem = imported_modules["FoodItem"]

        characters = []

        # Get character configuration
        character_count = config.get("count", 4)
        use_custom_data = config.get("use_custom", False)
        characters_file = config.get("characters_file")

        # Try to load characters from external file first
        character_data_list = None
        if characters_file:
            try:
                character_data_list = self._load_characters_from_file(characters_file)
                if character_data_list:
                    logger.info(
                        f"Loaded {len(character_data_list)} characters from {characters_file}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load characters from {characters_file}: {e}")

        # Fallback to config or default data
        if not character_data_list:
            if use_custom_data and "character_data" in config:
                character_data_list = config["character_data"]
            else:
                # Enhanced default character data with more variety
                character_data_list = [
                    {
                        "name": "Alice Cooper",
                        "age": 28,
                        "pronouns": "she/her",
                        "job": "Baker",
                        "recent_event": "Opened new bakery",
                        "long_term_goal": "Expand business to neighboring towns",
                        "personality": {"extraversion": 75, "conscientiousness": 85},
                        "specialties": ["baking", "customer_service"],
                    },
                    {
                        "name": "Bob Wilson",
                        "age": 35,
                        "pronouns": "he/him",
                        "job": "Blacksmith",
                        "recent_event": "Crafted special sword for mayor",
                        "long_term_goal": "Master legendary crafting techniques",
                        "personality": {"conscientiousness": 90, "openness": 60},
                        "specialties": ["metalworking", "tool_making"],
                    },
                    {
                        "name": "Charlie Green",
                        "age": 42,
                        "pronouns": "they/them",
                        "job": "Farmer",
                        "recent_event": "Harvested record crop",
                        "long_term_goal": "Develop sustainable farming practices",
                        "personality": {"conscientiousness": 80, "agreeableness": 85},
                        "specialties": ["agriculture", "animal_care"],
                    },
                    {
                        "name": "Diana Stone",
                        "age": 31,
                        "pronouns": "she/her",
                        "job": "Teacher",
                        "recent_event": "Student won regional competition",
                        "long_term_goal": "Build the village's first library",
                        "personality": {"openness": 90, "agreeableness": 80},
                        "specialties": ["education", "research"],
                    },
                    {
                        "name": "Erik Thornfield",
                        "age": 27,
                        "pronouns": "he/him",
                        "job": "Guard",
                        "recent_event": "Prevented bandit attack",
                        "long_term_goal": "Establish village militia",
                        "personality": {"conscientiousness": 85, "neuroticism": 30},
                        "specialties": ["combat", "security"],
                    },
                    {
                        "name": "Fiona Rivers",
                        "age": 39,
                        "pronouns": "she/her",
                        "job": "Healer",
                        "recent_event": "Cured mysterious illness",
                        "long_term_goal": "Open medical school",
                        "personality": {"agreeableness": 90, "conscientiousness": 80},
                        "specialties": ["medicine", "herbalism"],
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

    def _load_characters_from_file(self, filepath: str) -> List[Dict]:
        """Load character configuration from external file."""
        try:
            import json
            import os

            if not os.path.exists(filepath):
                logger.warning(f"Characters file not found: {filepath}")
                return []

            with open(filepath, "r") as f:
                data = json.load(f)

            characters_data = data.get("characters", [])

            # Validate and sanitize character data
            validated_characters = []
            for char_data in characters_data:
                if "name" in char_data:  # Minimum requirement
                    # Set defaults for missing fields
                    char_data.setdefault("age", random.randint(18, 65))
                    char_data.setdefault("pronouns", "they/them")
                    char_data.setdefault("job", "Villager")
                    char_data.setdefault("recent_event", "Recently joined the village")
                    char_data.setdefault("long_term_goal", "Live a fulfilling life")
                    char_data.setdefault("personality", {})
                    char_data.setdefault("specialties", [])

                    validated_characters.append(char_data)
                else:
                    logger.warning(f"Skipping invalid character data: {char_data}")

            return validated_characters

        except Exception as e:
            logger.error(f"Error loading characters from file {filepath}: {e}")
            return []

    def setup_user_driven_configuration(self):
        """Setup user-driven configuration options for dynamic gameplay setup."""
        try:
            # Create default configuration files if they don't exist
            self._create_default_config_files()

            # Check for user preference file
            user_prefs_file = self.config.get(
                "user_preferences_file", "user_preferences.json"
            )
            if os.path.exists(user_prefs_file):
                try:
                    with open(user_prefs_file, "r") as f:
                        user_prefs = json.load(f)

                    # Apply user preferences to configuration
                    self.config.update(user_prefs)
                    logger.info(f"Applied user preferences from {user_prefs_file}")

                except Exception as e:
                    logger.warning(f"Error loading user preferences: {e}")

        except Exception as e:
            logger.error(f"Error setting up user-driven configuration: {e}")

    def _create_default_config_files(self):
        """Create default configuration files for user customization."""
        try:
            import json
            import os

            # Create characters template file
            characters_template = {
                "characters": [
                    {
                        "name": "Custom Character 1",
                        "age": 25,
                        "pronouns": "they/them",
                        "job": "Artisan",
                        "recent_event": "Moved to the village",
                        "long_term_goal": "Master their craft",
                        "personality": {
                            "extraversion": 60,
                            "openness": 80,
                            "conscientiousness": 70,
                            "agreeableness": 75,
                            "neuroticism": 40,
                        },
                        "specialties": ["crafting", "creativity"],
                        "inventory_items": [
                            {"name": "Tools", "type": "equipment", "quantity": 1},
                            {"name": "Bread", "type": "food", "quantity": 2},
                        ],
                    }
                ]
            }

            characters_file = "custom_characters.json"
            if not os.path.exists(characters_file):
                with open(characters_file, "w") as f:
                    json.dump(characters_template, f, indent=2)
                logger.info(f"Created character template file: {characters_file}")

            # Create buildings template file
            buildings_template = {
                "buildings": [
                    {
                        "name": "Custom Building",
                        "type": "special",
                        "x": 250,
                        "y": 250,
                        "width": 50,
                        "height": 50,
                        "description": "A custom building for special purposes",
                    }
                ]
            }

            buildings_file = "custom_buildings.json"
            if not os.path.exists(buildings_file):
                with open(buildings_file, "w") as f:
                    json.dump(buildings_template, f, indent=2)
                logger.info(f"Created buildings template file: {buildings_file}")

            # Create user preferences template
            user_prefs_template = {
                "screen_width": 1024,
                "screen_height": 768,
                "target_fps": 60,
                "characters": {
                    "count": 6,
                    "use_custom": True,
                    "characters_file": "custom_characters.json",
                },
                "map": {"buildings_file": "custom_buildings.json"},
                "render": {"background_color": [20, 50, 80], "vsync": True},
            }

            prefs_file = "user_preferences.json"
            if not os.path.exists(prefs_file):
                with open(prefs_file, "w") as f:
                    json.dump(user_prefs_template, f, indent=2)
                logger.info(f"Created user preferences template: {prefs_file}")

        except Exception as e:
            logger.error(f"Error creating default config files: {e}")

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
                        name=item_data["name"],
                        description=item_data.get("description", "A food item"),
                        value=item_data.get("value", 1),
                        perishable=item_data.get("perishable", False),
                        weight=item_data.get("weight", 1),
                        quantity=item_data.get("quantity", 1),
                        action_system=(
                            self.action_system
                            if self.action_system
                            else imported_modules.get("ActionSystem", lambda: None)()
                        ),
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
                "inventory": inventory,  # Pass the ItemInventory object we created
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
        """Handle keyboard input with configurable key bindings and new features."""
        # Load key bindings from configuration
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
                "feature_status": [pygame.K_f],
                "system_recovery": [pygame.K_F5],
                "analytics": [pygame.K_a],
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
        elif event.key in key_bindings.get("save", [pygame.K_s]):
            # Save game functionality
            save_path = "saves/quicksave.json"
            if self.save_game_state(save_path):
                logger.info(f"Game saved to {save_path}")
            else:
                logger.error("Failed to save game")
        elif event.key in key_bindings.get("load", [pygame.K_l]):
            # Load game functionality
            save_path = "saves/quicksave.json"
            if self.load_game_state(save_path):
                logger.info(f"Game loaded from {save_path}")
            else:
                logger.error("Failed to load game")
        elif event.key in key_bindings.get("help", []):
            # Show help overlay
            self._show_help_info()
        elif event.key in key_bindings.get("debug", []):
            # Toggle debug information display
            self._toggle_debug_mode()
        elif event.key in key_bindings.get("fullscreen", []):
            # Toggle fullscreen mode
            self._toggle_fullscreen()
        elif event.key in key_bindings.get("feature_status", [pygame.K_f]):
            # Toggle feature status overlay
            self._show_feature_status = not getattr(self, "_show_feature_status", False)
            logger.info(
                f"Feature status overlay {'shown' if self._show_feature_status else 'hidden'}"
            )
        elif event.key in key_bindings.get("system_recovery", [pygame.K_F5]):
            # Force system recovery attempt
            self._force_system_recovery()
        elif event.key in key_bindings.get("analytics", [pygame.K_a]):
            # Show analytics information
            self._show_analytics_info()

    def _show_help_info(self):
        """Display help information."""
        try:
            logger.info("=== TINY VILLAGE HELP ===")
            logger.info("Controls:")
            logger.info("  SPACE - Pause/Unpause")
            logger.info("  R - Reset characters")
            logger.info("  S - Save game")
            logger.info("  L - Load game")
            logger.info("  F - Show feature status")
            logger.info("  F5 - Force system recovery")
            logger.info("  A - Show analytics")
            logger.info("  ESC - Quit")
            logger.info("Features:")
            logger.info("  - Character AI with goals and actions")
            logger.info("  - Basic quest system")
            logger.info("  - Weather simulation")
            logger.info("  - Social relationship tracking")
            logger.info("  - Achievement system")
            logger.info("  - Automatic error recovery")
        except Exception as e:
            logger.error(f"Error showing help: {e}")

    def _toggle_debug_mode(self):
        """Toggle debug information display."""
        try:
            self._debug_mode = not getattr(self, "_debug_mode", False)
            if self._debug_mode:
                logger.setLevel(logging.DEBUG)
                logger.info("Debug mode enabled")
            else:
                logger.setLevel(logging.INFO)
                logger.info("Debug mode disabled")
        except Exception as e:
            logger.error(f"Error toggling debug mode: {e}")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        try:
            # This is a basic implementation - can be enhanced
            flags = pygame.display.get_surface().get_flags()
            if flags & pygame.FULLSCREEN:
                # Switch to windowed
                screen_width = self.config.get("screen_width", 800)
                screen_height = self.config.get("screen_height", 600)
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                logger.info("Switched to windowed mode")
            else:
                # Switch to fullscreen
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                logger.info("Switched to fullscreen mode")
        except Exception as e:
            logger.error(f"Error toggling fullscreen: {e}")

    def _force_system_recovery(self):
        """Force system recovery attempt for all failed systems."""
        try:
            logger.info("Forcing system recovery for all systems...")
            if hasattr(self, "recovery_manager"):
                system_status = self.recovery_manager.get_system_status()
                failed_systems = [
                    name for name, status in system_status.items() if status == "failed"
                ]

                if failed_systems:
                    logger.info(
                        f"Attempting recovery for failed systems: {failed_systems}"
                    )
                    recovery_results = {}
                    for system_name in failed_systems:
                        success = self.recovery_manager.attempt_recovery(system_name)
                        recovery_results[system_name] = (
                            "SUCCESS" if success else "FAILED"
                        )

                    logger.info(f"Recovery results: {recovery_results}")
                else:
                    logger.info("No failed systems detected")
            else:
                logger.warning("Recovery manager not available")
        except Exception as e:
            logger.error(f"Error during forced system recovery: {e}")

    def _show_analytics_info(self):
        """Display analytics information."""
        try:
            logger.info("=== ANALYTICS REPORT ===")

            # Game statistics
            stats = self.game_statistics
            logger.info(f"Game Statistics:")
            logger.info(f"  Actions executed: {stats['actions_executed']}")
            logger.info(f"  Actions failed: {stats['actions_failed']}")
            logger.info(f"  Characters created: {stats['characters_created']}")
            logger.info(f"  Errors recovered: {stats['errors_recovered']}")

            # Action analytics
            if hasattr(self, "action_resolver"):
                analytics = self.action_resolver.get_action_analytics()
                logger.info(f"Action System:")
                logger.info(f"  Success rate: {analytics['success_rate']:.1%}")
                logger.info(f"  Cache size: {analytics['cache_size']}")

                if "action_breakdown" in analytics:
                    logger.info("  Action breakdown:")
                    for action_name, action_stats in analytics[
                        "action_breakdown"
                    ].items():
                        success_rate = (
                            action_stats["successes"] / action_stats["count"]
                            if action_stats["count"] > 0
                            else 0
                        )
                        logger.info(
                            f"    {action_name}: {action_stats['successes']}/{action_stats['count']} ({success_rate:.1%})"
                        )

            # System health
            if hasattr(self, "recovery_manager"):
                system_status = self.recovery_manager.get_system_status()
                healthy_count = sum(
                    1 for status in system_status.values() if status == "healthy"
                )
                logger.info(
                    f"System Health: {healthy_count}/{len(system_status)} systems healthy"
                )

                for system_name, status in system_status.items():
                    logger.info(f"  {system_name}: {status}")

            # Feature status
            feature_status = self.get_feature_implementation_status()
            implemented_features = sum(
                1
                for status in feature_status.values()
                if status in ["BASIC_IMPLEMENTED", "FULLY_IMPLEMENTED"]
            )
            logger.info(
                f"Features: {implemented_features}/{len(feature_status)} implemented"
            )

        except Exception as e:
            logger.error(f"Error showing analytics: {e}")

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
        """Update all game systems with delta time, improved error handling, and automatic recovery."""
        # Check if game is paused
        if getattr(self, "paused", False):
            return  # Skip all updates when paused

        update_errors = []
        systems_to_recover = []

        # Update the map controller (handles character movement and pathfinding)
        if self.map_controller:
            try:
                self.map_controller.update(dt)
            except Exception as e:
                logger.error(f"Error updating map controller: {e}")
                update_errors.append("Map controller update failed")
                systems_to_recover.append("map_controller")
        elif not self.map_controller:
            systems_to_recover.append("map_controller")

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

        # Check if we have no characters and attempt recovery
        if not self.characters:
            systems_to_recover.append("character_system")

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

        # Update feature systems
        try:
            self._update_feature_systems(dt)
        except Exception as e:
            logger.warning(f"Error updating feature systems: {e}")

        # Attempt automatic recovery for failed systems
        if systems_to_recover:
            logger.info(
                f"Attempting automatic recovery for systems: {systems_to_recover}"
            )
            for system_name in systems_to_recover:
                try:
                    if self.recovery_manager.attempt_recovery(system_name):
                        logger.info(f"Successfully recovered {system_name}")
                        if update_errors and system_name in str(update_errors):
                            self.game_statistics["errors_recovered"] += 1
                except Exception as e:
                    logger.error(f"Recovery failed for {system_name}: {e}")

        # Log update errors if any occurred
        if update_errors:
            logger.warning(
                f"Game state update completed with {len(update_errors)} errors"
            )
            self.game_statistics["errors_recovered"] += len(
                [e for e in update_errors if "recovered" in e]
            )

    def _update_feature_systems(self, dt):
        """Update all implemented feature systems."""
        try:
            # Update weather system
            if hasattr(self, "weather_system"):
                self.implement_weather_system()  # This updates weather state

            # Update social networks (relationship decay/growth)
            if hasattr(self, "social_networks"):
                self._update_social_relationships(dt)

            # Update quest system (time-based quest events)
            if hasattr(self, "quest_system"):
                self._update_quest_timers(dt)

        except Exception as e:
            logger.warning(f"Error updating feature systems: {e}")

    def _update_social_relationships(self, dt):
        """Update social relationships over time."""
        try:
            if not hasattr(self, "social_networks"):
                return

            # Slow relationship decay/growth over time
            for char_id, relationships in self.social_networks["relationships"].items():
                for other_id, strength in relationships.items():
                    # Very slow decay towards neutral (50)
                    if strength > 50:
                        relationships[other_id] = max(50, strength - 0.1 * dt)
                    elif strength < 50:
                        relationships[other_id] = min(50, strength + 0.1 * dt)

        except Exception as e:
            logger.warning(f"Error updating social relationships: {e}")

    def _update_quest_timers(self, dt):
        """Update quest-related timers and generate new quests."""
        try:
            if not hasattr(self, "quest_system"):
                return

            current_time = pygame.time.get_ticks()

            # Generate new quests for characters with no active quests
            for char_id in self.characters.keys():
                if char_id in self.quest_system["active_quests"]:
                    active_quests = self.quest_system["active_quests"][char_id]

                    # Remove expired quests (older than 5 minutes)
                    active_quests[:] = [
                        q
                        for q in active_quests
                        if current_time - q.get("assigned_time", 0) < 300000
                    ]

                    # Assign new quest if character has none
                    if (
                        len(active_quests) == 0 and random.random() < 0.1
                    ):  # 10% chance per update
                        template = random.choice(self.quest_system["quest_templates"])
                        quest = {
                            "id": f"{char_id}_{current_time}",
                            "name": template["name"],
                            "description": template["description"],
                            "type": template["type"],
                            "progress": 0,
                            "target": 100,
                            "assigned_time": current_time,
                        }
                        active_quests.append(quest)
                        logger.debug(
                            f"Assigned new quest to {self.characters[char_id].name}: {quest['name']}"
                        )

        except Exception as e:
            logger.warning(f"Error updating quest timers: {e}")

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
        """Execute a single action with proper resolution, state updates, and comprehensive tracking."""
        try:
            # Resolve action to executable format
            action = self.action_resolver.resolve_action(action_data, character)

            if not action:
                logger.warning(f"Could not resolve action: {action_data}")
                self.action_resolver.track_action_execution(
                    action_data, character, False
                )
                return False

            # Validate action preconditions
            if not self.action_resolver.validate_action_preconditions(
                action, character
            ):
                logger.warning(f"Action preconditions not met for {action}")
                fallback_success = self._execute_fallback_action(character)
                self.action_resolver.track_action_execution(
                    action, character, fallback_success
                )
                return fallback_success

            # Predict action effects for logging
            predicted_effects = self.action_resolver.predict_action_effects(
                action, character
            )
            logger.debug(f"Predicted effects for {action}: {predicted_effects}")

            # Execute the action
            if hasattr(action, "execute"):
                try:
                    result = action.execute(target=character, initiator=character)
                    if result:
                        # Update character state after successful action
                        self._update_character_state_after_action(character, action)

                        # Track successful execution
                        self.action_resolver.track_action_execution(
                            action, character, True
                        )

                        # Update quest progress if applicable
                        self._update_quest_progress(character, action)

                        return True
                    else:
                        logger.warning(f"Action {action.name} execution returned False")
                        self.action_resolver.track_action_execution(
                            action, character, False
                        )
                        return False
                except Exception as e:
                    logger.error(f"Error executing action {action.name}: {e}")
                    # Try fallback action
                    fallback_success = self._execute_fallback_action(character)
                    self.action_resolver.track_action_execution(
                        action, character, fallback_success
                    )
                    return fallback_success
            else:
                logger.warning(f"Action {action} has no execute method")
                self.action_resolver.track_action_execution(action, character, False)
                return False

        except Exception as e:
            logger.error(f"Critical error executing single action: {e}")
            fallback_success = self._execute_fallback_action(character)
            self.action_resolver.track_action_execution(
                action_data, character, fallback_success
            )
            return fallback_success

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
        """Update character state and related systems after action execution with comprehensive tracking."""
        try:
            # Track state before updates for comparison
            initial_state = self._capture_character_state(character)

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

            # Update memory system - record the action as a memory
            if hasattr(character, "add_memory"):
                try:
                    memory_text = (
                        f"Performed action: {getattr(action, 'name', str(action))}"
                    )
                    character.add_memory(memory_text)
                except Exception as e:
                    logger.warning(
                        f"Error adding action memory for {character.name}: {e}"
                    )

            # Update skill progression based on action type
            self._update_character_skills(character, action)

            # Update social relationships if action affects others
            self._update_social_consequences(character, action)

            # Update economic effects (resource consumption, production)
            self._update_economic_state(character, action)

            # Generate secondary events based on action outcomes
            self._generate_action_events(character, action, initial_state)

            # Update achievement tracking
            self._check_achievements(character, action)

            # Update reputation system
            self._update_reputation(character, action)

            # Track state changes for analytics
            final_state = self._capture_character_state(character)
            self._track_state_changes(character, action, initial_state, final_state)

        except Exception as e:
            logger.warning(f"Error updating character state after action: {e}")

    def _capture_character_state(self, character) -> Dict[str, Any]:
        """Capture current character state for comparison."""
        try:
            return {
                "energy": getattr(character, "energy", 0),
                "health_status": getattr(character, "health_status", 0),
                "hunger_level": getattr(character, "hunger_level", 0),
                "satisfaction": getattr(character, "current_satisfaction", 0),
                "social_wellbeing": getattr(character, "social_wellbeing", 0),
                "wealth": getattr(character, "wealth_money", 0),
                "job_performance": getattr(character, "job_performance", 0),
            }
        except Exception as e:
            logger.warning(f"Error capturing character state: {e}")
            return {}

    def _update_character_skills(self, character, action):
        """Update character skills based on performed action."""
        try:
            action_name = getattr(action, "name", str(action)).lower()

            # Map actions to skill improvements
            skill_mappings = {
                "work": "job_performance",
                "craft": "job_performance",
                "cook": "job_performance",
                "socialize": "social_wellbeing",
                "exercise": "health_status",
                "rest": "mental_health",
                "study": "intelligence",
                "trade": "wealth_money",
            }

            for action_type, skill in skill_mappings.items():
                if action_type in action_name and hasattr(character, skill):
                    current_value = getattr(character, skill, 0)
                    # Small skill improvement with diminishing returns
                    improvement = max(1, int(5 * (100 - current_value) / 100))
                    setattr(character, skill, min(100, current_value + improvement))

        except Exception as e:
            logger.warning(f"Error updating character skills: {e}")

    def _update_social_consequences(self, character, action):
        """Update social relationships based on action."""
        try:
            action_name = getattr(action, "name", str(action)).lower()

            # Actions that improve community standing
            positive_actions = ["help", "share", "give", "assist", "cooperate"]
            negative_actions = ["steal", "fight", "ignore", "refuse"]

            if any(word in action_name for word in positive_actions):
                if hasattr(character, "community"):
                    character.community = min(100, character.community + 2)
                if hasattr(character, "social_wellbeing"):
                    character.social_wellbeing = min(
                        100, character.social_wellbeing + 1
                    )

            elif any(word in action_name for word in negative_actions):
                if hasattr(character, "community"):
                    character.community = max(0, character.community - 3)
                if hasattr(character, "social_wellbeing"):
                    character.social_wellbeing = max(0, character.social_wellbeing - 2)

        except Exception as e:
            logger.warning(f"Error updating social consequences: {e}")

    def _update_economic_state(self, character, action):
        """Update economic state based on action costs and benefits."""
        try:
            action_name = getattr(action, "name", str(action)).lower()

            # Economic actions and their effects
            if "work" in action_name or "craft" in action_name:
                if hasattr(character, "wealth_money"):
                    earning = random.randint(5, 15)
                    character.wealth_money += earning

            elif "buy" in action_name or "purchase" in action_name:
                if hasattr(character, "wealth_money"):
                    cost = random.randint(3, 10)
                    character.wealth_money = max(0, character.wealth_money - cost)

            elif "trade" in action_name:
                if hasattr(character, "wealth_money"):
                    # Trading can be profitable or costly
                    change = random.randint(-5, 10)
                    character.wealth_money = max(0, character.wealth_money + change)

        except Exception as e:
            logger.warning(f"Error updating economic state: {e}")

    def _generate_action_events(self, character, action, initial_state):
        """Generate secondary events based on action outcomes."""
        try:
            action_name = getattr(action, "name", str(action))

            # Generate events for significant actions
            significant_actions = ["work", "craft", "major_purchase", "celebration"]

            if any(
                sig_action in action_name.lower() for sig_action in significant_actions
            ):
                event_text = f"{character.name} completed: {action_name}"

                # Add to events list if available
                if hasattr(self, "events"):
                    self.events.append(
                        {
                            "type": "action_completion",
                            "character": character.name,
                            "action": action_name,
                            "timestamp": pygame.time.get_ticks(),
                            "description": event_text,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error generating action events: {e}")

    def _check_achievements(self, character, action):
        """Check and award achievements based on character actions."""
        try:
            # Initialize achievements if not present
            if not hasattr(character, "achievements"):
                character.achievements = set()

            action_name = getattr(action, "name", str(action)).lower()

            # Define achievement conditions
            achievements = {
                "first_work": ("work" in action_name, "Completed first work action"),
                "social_butterfly": (
                    hasattr(character, "social_wellbeing")
                    and character.social_wellbeing > 80,
                    "High social wellbeing",
                ),
                "hard_worker": (
                    hasattr(character, "job_performance")
                    and character.job_performance > 90,
                    "Excellent job performance",
                ),
                "wealthy": (
                    hasattr(character, "wealth_money") and character.wealth_money > 500,
                    "Accumulated significant wealth",
                ),
            }

            for achievement_id, (condition, description) in achievements.items():
                if condition and achievement_id not in character.achievements:
                    character.achievements.add(achievement_id)
                    logger.info(f"{character.name} earned achievement: {description}")

        except Exception as e:
            logger.warning(f"Error checking achievements: {e}")

    def _update_reputation(self, character, action):
        """Update character reputation based on actions."""
        try:
            if not hasattr(character, "reputation"):
                character.reputation = 50  # Neutral starting reputation

            action_name = getattr(action, "name", str(action)).lower()

            # Reputation modifiers based on action type
            if any(word in action_name for word in ["help", "share", "generous"]):
                character.reputation = min(100, character.reputation + 2)
            elif any(word in action_name for word in ["steal", "cheat", "selfish"]):
                character.reputation = max(0, character.reputation - 5)
            elif "work" in action_name:
                character.reputation = min(100, character.reputation + 1)

        except Exception as e:
            logger.warning(f"Error updating reputation: {e}")

    def _track_state_changes(self, character, action, initial_state, final_state):
        """Track state changes for analytics and debugging."""
        try:
            changes = {}
            for key in initial_state:
                if key in final_state:
                    change = final_state[key] - initial_state[key]
                    if change != 0:
                        changes[key] = change

            if changes:
                logger.debug(
                    f"State changes for {character.name} after {getattr(action, 'name', action)}: {changes}"
                )

        except Exception as e:
            logger.warning(f"Error tracking state changes: {e}")

    def _update_quest_progress(self, character, action):
        """Update quest progress based on completed action."""
        try:
            if not hasattr(self, "quest_system") or not hasattr(character, "uuid"):
                return

            char_id = character.uuid
            if char_id not in self.quest_system["active_quests"]:
                return

            action_name = getattr(action, "name", str(action)).lower()

            # Update progress for relevant quests
            for quest in self.quest_system["active_quests"][char_id]:
                quest_type = quest.get("type", "unknown")

                # Simple progress logic based on quest type and action
                progress_added = 0

                if quest_type == "collection" and any(
                    word in action_name for word in ["gather", "collect", "harvest"]
                ):
                    progress_added = 20
                elif quest_type == "social" and any(
                    word in action_name for word in ["help", "talk", "assist"]
                ):
                    progress_added = 25
                elif quest_type == "crafting" and any(
                    word in action_name for word in ["craft", "make", "build"]
                ):
                    progress_added = 30
                elif "work" in action_name:
                    progress_added = 10  # Any work contributes to most quests

                if progress_added > 0:
                    quest["progress"] = min(
                        quest["target"], quest["progress"] + progress_added
                    )
                    logger.debug(
                        f"Quest progress updated: {quest['name']} - {quest['progress']}/{quest['target']}"
                    )

                    # Check if quest is completed
                    if quest["progress"] >= quest["target"]:
                        self._complete_quest(character, quest)

        except Exception as e:
            logger.warning(f"Error updating quest progress: {e}")

    def _complete_quest(self, character, quest):
        """Handle quest completion."""
        try:
            char_id = character.uuid

            # Move quest to completed list
            if char_id not in self.quest_system["completed_quests"]:
                self.quest_system["completed_quests"][char_id] = []

            self.quest_system["completed_quests"][char_id].append(quest)
            self.quest_system["active_quests"][char_id].remove(quest)

            # Reward the character
            if hasattr(character, "wealth_money"):
                reward = random.randint(10, 30)
                character.wealth_money += reward

            if hasattr(character, "current_satisfaction"):
                character.current_satisfaction = min(
                    100, character.current_satisfaction + 15
                )

            logger.info(f"{character.name} completed quest: {quest['name']}")

            # Generate completion event
            if hasattr(self, "events"):
                self.events.append(
                    {
                        "type": "quest_completion",
                        "character": character.name,
                        "quest_name": quest["name"],
                        "timestamp": pygame.time.get_ticks(),
                        "description": f"{character.name} completed the quest '{quest['name']}'",
                    }
                )

        except Exception as e:
            logger.error(f"Error completing quest: {e}")

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
        """Render user interface elements with improved layout, new features, and system status."""
        try:
            # Create font for UI text
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)
            tiny_font = pygame.font.Font(None, 16)

            # Render character count and basic info
            char_count_text = font.render(
                f"Characters: {len(self.characters)}", True, (255, 255, 255)
            )
            self.screen.blit(char_count_text, (10, 10))

            # Render pause status
            if getattr(self, "paused", False):
                pause_text = font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.screen.get_width() - 100, 10))

            # Render time if available
            y_offset = 35
            if hasattr(self, "gametime_manager") and self.gametime_manager:
                try:
                    game_time = (
                        self.gametime_manager.get_calendar().get_game_time_string()
                    )
                    time_text = small_font.render(
                        f"Time: {game_time}", True, (255, 255, 255)
                    )
                    self.screen.blit(time_text, (10, y_offset))
                    y_offset += 20
                except:
                    pass

            # Render weather information
            if hasattr(self, "weather_system"):
                weather_text = small_font.render(
                    f"Weather: {self.weather_system['current_weather']} {self.weather_system['temperature']}°C",
                    True,
                    (200, 220, 255),
                )
                self.screen.blit(weather_text, (10, y_offset))
                y_offset += 20

            # Render game statistics
            stats = self.game_statistics
            stats_text = tiny_font.render(
                f"Actions: {stats['actions_executed']} | Failed: {stats['actions_failed']} | Recovered: {stats['errors_recovered']}",
                True,
                (180, 180, 180),
            )
            self.screen.blit(stats_text, (10, y_offset))
            y_offset += 15

            # Render action analytics if available
            if hasattr(self, "action_resolver"):
                try:
                    analytics = self.action_resolver.get_action_analytics()
                    analytics_text = tiny_font.render(
                        f"Success Rate: {analytics['success_rate']:.1%} | Cache: {analytics['cache_size']}",
                        True,
                        (150, 150, 150),
                    )
                    self.screen.blit(analytics_text, (10, y_offset))
                    y_offset += 15
                except:
                    pass

            # Render system health status
            if hasattr(self, "recovery_manager"):
                try:
                    system_status = self.recovery_manager.get_system_status()
                    healthy_systems = sum(
                        1 for status in system_status.values() if status == "healthy"
                    )
                    total_systems = len(system_status)

                    health_color = (
                        (0, 255, 0)
                        if healthy_systems == total_systems
                        else (
                            (255, 255, 0)
                            if healthy_systems > total_systems // 2
                            else (255, 0, 0)
                        )
                    )
                    health_text = tiny_font.render(
                        f"Systems: {healthy_systems}/{total_systems} healthy",
                        True,
                        health_color,
                    )
                    self.screen.blit(health_text, (10, y_offset))
                    y_offset += 15
                except:
                    pass

            # Render selected character info (enhanced)
            if (
                hasattr(self.map_controller, "selected_character")
                and self.map_controller.selected_character
            ):
                char = self.map_controller.selected_character
                char_info = [
                    f"Selected: {char.name}",
                    f"Job: {getattr(char, 'job', 'Unknown')}",
                    f"Energy: {getattr(char, 'energy', 0)}",
                    f"Health: {getattr(char, 'health_status', 0)}",
                ]

                # Add social and quest info
                if hasattr(char, "uuid") and hasattr(self, "social_networks"):
                    try:
                        relationships = self.social_networks["relationships"].get(
                            char.uuid, {}
                        )
                        avg_relationship = (
                            sum(relationships.values()) / len(relationships)
                            if relationships
                            else 50
                        )
                        char_info.append(f"Social: {avg_relationship:.0f}")
                    except:
                        pass

                if hasattr(char, "uuid") and hasattr(self, "quest_system"):
                    try:
                        active_quests = len(
                            self.quest_system["active_quests"].get(char.uuid, [])
                        )
                        completed_quests = len(
                            self.quest_system["completed_quests"].get(char.uuid, [])
                        )
                        char_info.append(
                            f"Quests: {active_quests} active, {completed_quests} done"
                        )
                    except:
                        pass

                for i, info in enumerate(char_info):
                    info_text = small_font.render(info, True, (255, 255, 0))
                    self.screen.blit(info_text, (10, y_offset + i * 20))

            # Render enhanced instructions
            instructions = [
                "Click characters to select them",
                "Click buildings to interact",
                "SPACE to pause/unpause",
                "R to reset characters",
                "S to save game (basic)",
                "L to load game (basic)",
                "F to show feature status",
                "ESC to quit",
            ]

            instruction_start_y = self.screen.get_height() - len(instructions) * 15 - 10
            for i, instruction in enumerate(instructions):
                inst_text = tiny_font.render(instruction, True, (200, 200, 200))
                self.screen.blit(inst_text, (10, instruction_start_y + i * 15))

            # Show feature implementation status on F key press (stored state)
            if getattr(self, "_show_feature_status", False):
                self._render_feature_status_overlay()

        except Exception as e:
            # Fallback to minimal UI
            try:
                font = pygame.font.Font(None, 24)
                error_text = font.render("UI Error - Fallback Mode", True, (255, 0, 0))
                self.screen.blit(error_text, (10, 10))
                char_text = font.render(
                    f"Characters: {len(self.characters)}", True, (255, 255, 255)
                )
                self.screen.blit(char_text, (10, 35))
            except:
                pass  # Even fallback failed

    def _render_feature_status_overlay(self):
        """Render an overlay showing feature implementation status."""
        try:
            font = pygame.font.Font(None, 18)

            # Create semi-transparent overlay
            overlay = pygame.Surface((400, 300))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))

            # Get feature status
            feature_status = self.get_feature_implementation_status()

            # Render title
            title_text = font.render(
                "Feature Implementation Status", True, (255, 255, 255)
            )
            overlay.blit(title_text, (10, 10))

            y_pos = 35
            for feature, status in feature_status.items():
                # Color code based on status
                if status == "FULLY_IMPLEMENTED":
                    color = (0, 255, 0)
                elif status == "BASIC_IMPLEMENTED":
                    color = (255, 255, 0)
                elif status == "STUB_IMPLEMENTED":
                    color = (255, 165, 0)
                else:  # NOT_STARTED
                    color = (255, 0, 0)

                # Format feature name
                display_name = feature.replace("_", " ").title()
                status_text = font.render(f"{display_name}: {status}", True, color)
                overlay.blit(status_text, (10, y_pos))
                y_pos += 18

                if y_pos > 260:  # Prevent overflow
                    break

            # Blit overlay to main screen
            self.screen.blit(overlay, (self.screen.get_width() - 420, 50))

        except Exception as e:
            logger.warning(f"Error rendering feature status overlay: {e}")

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
        Save complete game state to file with basic implementation.
        """
        try:
            import json
            import os

            # Create save directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_data = {
                "version": "1.0",
                "timestamp": pygame.time.get_ticks(),
                "config": self.config,
                "game_statistics": self.game_statistics,
                "character_count": len(self.characters),
                "paused": getattr(self, "paused", False),
                "characters": [],
                "events": getattr(self, "events", []),
            }

            # Save basic character data
            for char_id, character in self.characters.items():
                char_data = {
                    "uuid": char_id,
                    "name": getattr(character, "name", "Unknown"),
                    "job": getattr(character, "job", "Villager"),
                    "age": getattr(character, "age", 25),
                    "energy": getattr(character, "energy", 50),
                    "health_status": getattr(character, "health_status", 100),
                    "position": {
                        "x": (
                            character.position.x
                            if hasattr(character, "position")
                            else 0
                        ),
                        "y": (
                            character.position.y
                            if hasattr(character, "position")
                            else 0
                        ),
                    },
                }
                save_data["characters"].append(char_data)

            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Game state saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            return False

    def load_game_state(self, save_path: str) -> bool:
        """
        Load complete game state from file with basic implementation.
        """
        try:
            import json
            import os

            if not os.path.exists(save_path):
                logger.error(f"Save file not found: {save_path}")
                return False

            with open(save_path, "r") as f:
                save_data = json.load(f)

            # Validate save file version
            version = save_data.get("version", "unknown")
            if version != "1.0":
                logger.warning(f"Save file version {version} may not be compatible")

            # Restore basic game state
            if "config" in save_data:
                self.config.update(save_data["config"])

            if "game_statistics" in save_data:
                self.game_statistics.update(save_data["game_statistics"])

            if "paused" in save_data:
                self.paused = save_data["paused"]

            if "events" in save_data:
                self.events = save_data["events"]

            # TODO: Restore character states (requires more complex character system)
            # TODO: Restore map state and building conditions
            # TODO: Restore relationship networks
            # TODO: Restore economic state

            logger.info(f"Game state loaded from {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading game state: {e}")
            return False

    # Feature Implementation Stubs with Clear Tracking

    def implement_achievement_system(self) -> bool:
        """
        Basic achievement system implementation.

        STATUS: BASIC_IMPLEMENTED
        TODO: Add achievement notifications
        TODO: Add achievement rewards
        TODO: Add complex achievement conditions
        TODO: Add achievement persistence
        """
        try:
            if not hasattr(self, "global_achievements"):
                self.global_achievements = {
                    "village_milestones": {
                        "first_character_created": False,
                        "five_characters_active": False,
                        "successful_harvest": False,
                        "trade_established": False,
                    },
                    "social_achievements": {
                        "first_friendship": False,
                        "community_event": False,
                        "conflict_resolved": False,
                    },
                    "economic_achievements": {
                        "first_transaction": False,
                        "wealthy_villager": False,
                        "market_established": False,
                    },
                }

            # Check for milestone achievements
            if len(self.characters) >= 1:
                self.global_achievements["village_milestones"][
                    "first_character_created"
                ] = True

            if len(self.characters) >= 5:
                self.global_achievements["village_milestones"][
                    "five_characters_active"
                ] = True

            return True

        except Exception as e:
            logger.error(f"Error implementing achievement system: {e}")
            return False

    def implement_weather_system(self) -> Dict[str, Any]:
        """
        Basic weather system implementation.

        STATUS: STUB_IMPLEMENTED
        TODO: Add seasonal changes
        TODO: Add weather effects on characters
        TODO: Add weather-based events
        TODO: Add visual weather effects
        """
        try:
            if not hasattr(self, "weather_system"):
                self.weather_system = {
                    "current_weather": "clear",
                    "temperature": 20,  # Celsius
                    "season": "spring",
                    "weather_effects_active": False,
                }

            # Simple weather simulation
            weather_options = ["clear", "cloudy", "rainy", "sunny"]
            if random.random() < 0.1:  # 10% chance to change weather
                self.weather_system["current_weather"] = random.choice(weather_options)

            return self.weather_system

        except Exception as e:
            logger.error(f"Error implementing weather system: {e}")
            return {"error": str(e)}

    def implement_social_network_system(self) -> bool:
        """
        Basic social network system implementation.

        STATUS: STUB_IMPLEMENTED
        TODO: Add relationship strength tracking
        TODO: Add relationship events
        TODO: Add social influence on decisions
        TODO: Add group formation dynamics
        """
        try:
            if not hasattr(self, "social_networks"):
                self.social_networks = {
                    "relationships": {},  # character_id -> {other_id: relationship_strength}
                    "groups": [],  # List of character groups
                    "social_events": [],  # List of social events
                }

            # Initialize relationships for existing characters
            char_ids = list(self.characters.keys())
            for char_id in char_ids:
                if char_id not in self.social_networks["relationships"]:
                    self.social_networks["relationships"][char_id] = {}

                # Create basic relationships with other characters
                for other_id in char_ids:
                    if (
                        other_id != char_id
                        and other_id
                        not in self.social_networks["relationships"][char_id]
                    ):
                        # Random initial relationship strength (0-100)
                        self.social_networks["relationships"][char_id][other_id] = (
                            random.randint(30, 70)
                        )

            return True

        except Exception as e:
            logger.error(f"Error implementing social network system: {e}")
            return False

    def implement_quest_system(self) -> bool:
        """
        Basic quest system implementation.

        STATUS: STUB_IMPLEMENTED
        TODO: Add quest generation algorithms
        TODO: Add quest rewards and consequences
        TODO: Add multi-step quests
        TODO: Add quest sharing between characters
        """
        try:
            if not hasattr(self, "quest_system"):
                self.quest_system = {
                    "active_quests": {},  # character_id -> [quest_objects]
                    "completed_quests": {},
                    "available_quests": [],
                    "quest_templates": [
                        {
                            "name": "Gather Resources",
                            "description": "Collect materials for the village",
                            "type": "collection",
                            "difficulty": "easy",
                        },
                        {
                            "name": "Help Neighbor",
                            "description": "Assist another villager with their work",
                            "type": "social",
                            "difficulty": "medium",
                        },
                        {
                            "name": "Craft Special Item",
                            "description": "Create a unique item for the community",
                            "type": "crafting",
                            "difficulty": "hard",
                        },
                    ],
                }

            # Generate some basic quests for characters
            for char_id in self.characters.keys():
                if char_id not in self.quest_system["active_quests"]:
                    self.quest_system["active_quests"][char_id] = []

                # Assign a random quest if character has none
                if len(self.quest_system["active_quests"][char_id]) == 0:
                    template = random.choice(self.quest_system["quest_templates"])
                    quest = {
                        "id": f"{char_id}_{pygame.time.get_ticks()}",
                        "name": template["name"],
                        "description": template["description"],
                        "type": template["type"],
                        "progress": 0,
                        "target": 100,
                        "assigned_time": pygame.time.get_ticks(),
                    }
                    self.quest_system["active_quests"][char_id].append(quest)

            return True

        except Exception as e:
            logger.error(f"Error implementing quest system: {e}")
            return False

    def get_feature_implementation_status(self) -> Dict[str, str]:
        """
        Get the implementation status of all planned features.

        Returns a dictionary with feature names and their implementation status:
        - NOT_STARTED: Feature not yet implemented
        - STUB_IMPLEMENTED: Basic structure in place, core functionality missing
        - BASIC_IMPLEMENTED: Core functionality working, needs enhancement
        - FULLY_IMPLEMENTED: Feature complete and polished
        """
        return {
            "save_load_system": "BASIC_IMPLEMENTED",
            "achievement_system": "BASIC_IMPLEMENTED",
            "weather_system": "STUB_IMPLEMENTED",
            "social_network_system": "STUB_IMPLEMENTED",
            "quest_system": "STUB_IMPLEMENTED",
            "skill_progression": "BASIC_IMPLEMENTED",
            "reputation_system": "BASIC_IMPLEMENTED",
            "economic_simulation": "STUB_IMPLEMENTED",
            "event_driven_storytelling": "NOT_STARTED",
            "mod_system": "NOT_STARTED",
            "multiplayer_support": "NOT_STARTED",
            "advanced_ai_behaviors": "NOT_STARTED",
            "procedural_content_generation": "NOT_STARTED",
            "advanced_graphics_effects": "NOT_STARTED",
            "sound_and_music_system": "NOT_STARTED",
            "accessibility_features": "NOT_STARTED",
            "performance_optimization": "NOT_STARTED",
            "automated_testing": "NOT_STARTED",
            "configuration_ui": "NOT_STARTED",
        }


if __name__ == "__main__":
    game_controller = GameplayController()
    game_controller.run()
    pygame.quit()
